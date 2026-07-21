"""Per-sample DP-SGD client round via Opacus, for experiment blocks E1 (full)
and E2 (partial, backbone frozen).

Ultralytics' high-level `YOLO.train()` owns its optimizer/backward internally
and gives no hook for Opacus to wrap it, so this reimplements just enough of
`DetectionTrainer`'s setup to get a model + optimizer + dataloader, then
drives the step loop by hand. Validated end-to-end (forward/backward/step +
epsilon accounting, for both freeze=None and freeze=backbone) against a
synthetic dataset on CPU in a network-restricted sandbox -- re-verify the
first real run on your GPU box, since Ultralytics' internal APIs can shift
between point releases.

Privacy accounting is PERSISTENT PER CLIENT across communication rounds: the
caller passes the client's saved accountant state in and stores the updated
state back out after each round (see scripts/_dp_sweep_common.py). A fresh
PrivacyEngine is created each round (model/optimizer are rebuilt from the new
global weights), but its accountant is restored from the client's saved state
first, so `get_epsilon` returns the epsilon accumulated over ALL of that
client's local optimizer steps since round 0 -- not just the current round.
Epsilon is therefore monotonically non-decreasing round over round. Because
each image lives on exactly one client, epsilons are NEVER summed across
clients; the sweep reports per-client epsilon and the maximum as a
conservative summary.
"""
from __future__ import annotations

import torch
from opacus import PrivacyEngine

from fedxpalm.federated.trainer_utils import build_trainer_from_checkpoint
from fedxpalm.models.groupnorm import disable_inplace_ops
from fedxpalm.privacy.opacus_patch import patch_opacus_for_dict_datasets

patch_opacus_for_dict_datasets()


def _restore_accountant(privacy_engine: PrivacyEngine, saved_state) -> None:
    """Load a client's saved accountant state into this round's engine so the
    privacy budget keeps accumulating instead of restarting at zero. Tries the
    Opacus state_dict API first, then falls back to restoring the raw history
    list (robust across Opacus point releases)."""
    if not saved_state:
        return
    try:
        privacy_engine.accountant.load_state_dict(saved_state)
        return
    except Exception:
        history = saved_state.get("history") if isinstance(saved_state, dict) else None
        if history is not None:
            privacy_engine.accountant.history = list(history)


def _dump_accountant(privacy_engine: PrivacyEngine) -> dict:
    try:
        return privacy_engine.accountant.state_dict()
    except Exception:
        return {"history": list(getattr(privacy_engine.accountant, "history", []))}


def _cumulative_steps(privacy_engine: PrivacyEngine) -> int:
    """Total optimizer steps the accountant has recorded across all rounds.
    History entries are (noise_multiplier, sample_rate, num_steps)-shaped for
    both the PRV and RDP accountants."""
    history = getattr(privacy_engine.accountant, "history", [])
    total = 0
    for entry in history:
        try:
            total += int(entry[2])
        except (TypeError, IndexError, ValueError):
            pass
    return total


def _per_sample_grad_norms(dp_optimizer) -> "torch.Tensor":
    """Pre-clipping per-sample gradient L2 norm for the current micro-batch.

    Must be called AFTER loss.backward() (so every param's `.grad_sample` is
    populated) and BEFORE dp_optimizer.step() (which calls
    clip_and_accumulate() internally and discards these norms without
    exposing them). Replicates Opacus DPOptimizer.clip_and_accumulate()'s own
    formula verbatim, using only its public `grad_samples` property -- so
    this is guaranteed to match what Opacus actually clips on, without
    monkey-patching or duplicating any clipping DECISION logic, only
    reading the same per-sample norms it computes internally.
    """
    grad_samples = dp_optimizer.grad_samples  # list of (batch, *param_shape) tensors, one per param
    if not grad_samples or len(grad_samples[0]) == 0:
        return torch.zeros(0)
    per_param_norms = [g.reshape(len(g), -1).norm(2, dim=-1) for g in grad_samples]
    return torch.stack(per_param_norms, dim=1).norm(2, dim=1).detach().cpu()


def _strip_frozen_from_optimizer(optimizer, model) -> list[str]:
    """Remove every requires_grad=False parameter from the optimizer's
    param_groups, BEFORE Opacus wraps it, so the DPOptimizer holds exactly the
    trainable set.

    Ultralytics' build_optimizer places ALL model parameters into its three
    groups (decay weights / norm weights / biases) without checking
    requires_grad -- freezing (E2's backbone, or the always-frozen DFL fixed
    conv) only zeroes their grads, it does not remove them from the groups.
    That is inert at runtime (a param with no grad is never updated -- the E2
    smoke confirmed backbone_changed=False even pre-filter), but a clean
    partial-DP implementation should hand Opacus only the trainable
    parameters. Filtering is done IN PLACE on the existing groups: each
    group's hyperparameters (lr, momentum, weight_decay, ...) are untouched
    and Ultralytics' grouping is preserved; groups left empty are dropped.

    Returns the names of the removed parameters.
    """
    name_by_id = {id(p): n for n, p in model.named_parameters()}
    removed = []
    new_groups = []
    for grp in optimizer.param_groups:
        kept = [p for p in grp["params"] if p.requires_grad]
        removed.extend(name_by_id.get(id(p), "<unknown>")
                       for p in grp["params"] if not p.requires_grad)
        if kept:
            grp["params"] = kept
            new_groups.append(grp)
    optimizer.param_groups[:] = new_groups
    return sorted(removed)


def train_client_round_dp(
    global_weights_path: str,
    client_data_yaml: str,
    hyp: dict,
    dp_hyp: dict,
    round_idx: int,
    client_id: str,
    out_dir: str,
    device: str = "0",
    freeze_stages: list[int] | None = None,
    accountant_state: dict | None = None,
    max_steps: int | None = None,
    collect_grad_norms: bool = False,
) -> tuple[dict, dict]:
    """Fine-tunes `global_weights_path` on one client with per-sample DP-SGD.

    `accountant_state` is this client's saved privacy-accountant state from the
    previous round (None on round 0). `max_steps` caps the number of DP
    optimizer steps taken (default None = run the full epochs_per_round as
    normal; unused by any real E1/E2 sweep call site) -- for
    scripts/20_dp_freeze_audit.py's fast diagnostic, which needs only a
    couple of real optimizer steps against the production code path, not a
    full epoch. `collect_grad_norms` (default False = zero added cost, unused
    by any real E1/E2 sweep call site) records pre-clipping per-sample
    gradient L2 norms every step and returns summary statistics in
    info["grad_norm_stats"] -- for scripts/23_diag_b_clipping_only.py's
    clipping-severity diagnostic. Returns (state_dict, info) where state_dict
    has clean (non-Opacus-prefixed) keys ready for fedavg(), and info carries
    the per-round/per-client privacy log plus `accountant_state` (the updated
    state to persist for next round).
    """
    run_name = f"r{round_idx}_client{client_id}_dp"
    overrides = dict(
        data=client_data_yaml,
        model=global_weights_path,
        epochs=hyp["epochs_per_round"],
        batch=hyp["batch_size"],
        imgsz=hyp.get("imgsz", 640),
        optimizer=hyp.get("optimizer", "SGD"),
        lr0=hyp.get("lr0", 0.01),
        momentum=hyp.get("momentum", 0.9),
        weight_decay=hyp.get("weight_decay", 0.0005),
        warmup_epochs=hyp.get("warmup_epochs", 0.0),
        device=device,
        workers=hyp.get("workers", 4),
        amp=False,          # Opacus does not officially support mixed precision
        plots=False,
        val=False,
        verbose=False,
        freeze=freeze_stages,
        exist_ok=True,
        project=out_dir,
        name=run_name,
    )
    trainer = build_trainer_from_checkpoint(global_weights_path, overrides)
    disable_inplace_ops(trainer.model)  # in case a fresh (non-DP-prepared) checkpoint slips in
    trainer._setup_train()

    # Strip every requires_grad=False parameter (E2's frozen backbone AND the
    # always-frozen DFL fixed conv) out of the optimizer BEFORE Opacus wraps
    # it, so the DPOptimizer holds exactly -- and only -- the trainable set.
    removed_frozen = _strip_frozen_from_optimizer(trainer.optimizer, trainer.model)

    # Post-filter audit (set comparison, by param identity):
    #   trainable model params  ==  params in optimizer groups
    # missing_trainable_params and unexpected_frozen_params must both be [].
    name_by_id = {id(p): n for n, p in trainer.model.named_parameters()}
    n_trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in trainer.model.parameters() if not p.requires_grad)
    trainable_ids = {id(p) for p in trainer.model.parameters() if p.requires_grad}
    optimizer_ids = {id(p) for grp in trainer.optimizer.param_groups for p in grp["params"]}
    missing_trainable = sorted(name_by_id[i] for i in trainable_ids - optimizer_ids)
    frozen_in_optimizer = sorted(
        name_by_id.get(i, "<unknown>") for i in optimizer_ids - trainable_ids
    )

    privacy_engine = PrivacyEngine(accountant=dp_hyp.get("accountant", "prv"))
    dp_model, dp_optimizer, dp_loader = privacy_engine.make_private(
        module=trainer.model,
        optimizer=trainer.optimizer,
        data_loader=trainer.train_loader,
        noise_multiplier=dp_hyp["sigma"],
        max_grad_norm=dp_hyp["max_grad_norm"],
        poisson_sampling=True,
    )
    # resume this client's privacy budget from prior rounds BEFORE stepping,
    # so get_epsilon() below is cumulative over the whole run
    _restore_accountant(privacy_engine, accountant_state)
    steps_before = _cumulative_steps(privacy_engine)

    sample_rate = float(getattr(dp_loader, "sample_rate",
                                hyp["batch_size"] / max(1, len(dp_loader.dataset))))

    dp_model.train()
    n_steps = 0
    nan_inf_detected = False
    observed_norms = [] if collect_grad_norms else None
    for _epoch in range(hyp["epochs_per_round"]):
        for batch in dp_loader:
            if len(batch["img"]) == 0:  # Poisson sampling can draw an empty batch
                continue
            if len(batch["cls"]) == 0:
                # A batch with images but zero ground-truth boxes across all of them
                # (small Poisson draw landing entirely on an unlabeled/background image)
                # leaves Ultralytics' box-regression branch (model.23.cv2.*) disconnected
                # from the loss graph -- Opacus then raises "Per sample gradient is not
                # initialized" since those parameters never got a backward pass. Skipping
                # is safe: such a batch carries no positive detection signal anyway.
                continue
            batch = trainer.preprocess_batch(batch)
            dp_optimizer.zero_grad()
            loss, _loss_items = dp_model(batch)
            total_loss = loss.sum()
            if not torch.isfinite(total_loss):
                nan_inf_detected = True
            total_loss.backward()
            if collect_grad_norms:
                # must read BEFORE step(): step() -> pre_step() -> clip_and_accumulate()
                # computes this same quantity internally then discards it
                observed_norms.append(_per_sample_grad_norms(dp_optimizer))
            dp_optimizer.step()
            n_steps += 1
            if max_steps is not None and n_steps >= max_steps:
                break
        if max_steps is not None and n_steps >= max_steps:
            break

    grad_norm_stats = None
    if collect_grad_norms and observed_norms:
        all_norms = torch.cat([n for n in observed_norms if n.numel() > 0])
        if all_norms.numel() > 0:
            c = dp_hyp["max_grad_norm"]
            # "fraction of samples with norm > C" and "clipping fraction" are the
            # SAME quantity under L2 clipping (clip_factor = min(C/norm, 1) < 1
            # iff norm > C) -- both reported since they were asked for
            # separately, but they are mathematically identical by construction,
            # not two independent measurements.
            frac_above_c = float((all_norms > c).float().mean())
            grad_norm_stats = {
                "n_samples_observed": int(all_norms.numel()),
                "median": float(all_norms.median()),
                "p75": float(torch.quantile(all_norms, 0.75)),
                "p90": float(torch.quantile(all_norms, 0.90)),
                "p95": float(torch.quantile(all_norms, 0.95)),
                "max": float(all_norms.max()),
                "fraction_norm_above_C": frac_above_c,
                "clip_fraction": frac_above_c,
                "clip_fraction_note": "identical to fraction_norm_above_C by construction (L2 clipping)",
            }

    # sigma == 0 is the "DP mechanism ablation: clipping-only diagnostic":
    # per-sample clipping still runs, but with no Gaussian noise there is NO
    # finite (epsilon, delta) guarantee -- epsilon must not be reported as a
    # DP number (and the PRV accountant cannot evaluate sigma=0 anyway).
    if dp_hyp["sigma"] > 0:
        epsilon = float(privacy_engine.get_epsilon(delta=dp_hyp["delta"]))
        privacy_guarantee = "dp_sgd"
    else:
        epsilon = None
        privacy_guarantee = "not_applicable_no_noise"
    cumulative_steps = _cumulative_steps(privacy_engine)
    state_dict = {k.replace("_module.", "", 1): v.detach().clone() for k, v in dp_model.state_dict().items()}

    info = {
        "round": round_idx,
        "client_id": client_id,
        "sigma": dp_hyp["sigma"],
        "max_grad_norm": dp_hyp["max_grad_norm"],
        "delta": dp_hyp["delta"],
        "n_samples": len(dp_loader.dataset),
        "sample_rate_q": sample_rate,
        "steps_this_round": n_steps,
        "cumulative_steps": cumulative_steps,
        "epsilon": epsilon,                        # cumulative over all rounds; None when sigma=0
        "privacy_guarantee": privacy_guarantee,    # "dp_sgd" | "not_applicable_no_noise"
        "nan_inf": bool(nan_inf_detected),
        "clip_fraction": grad_norm_stats["clip_fraction"] if grad_norm_stats else None,
        "clip_fraction_note": "see grad_norm_stats" if grad_norm_stats else "not_measured (collect_grad_norms=False)",
        "grad_norm_stats": grad_norm_stats,  # None unless collect_grad_norms=True
        "frozen": freeze_stages is not None,
        "n_trainable_params": int(n_trainable),
        "n_frozen_params": int(n_frozen),
        # optimizer<->model set audit (post-filter): both lists must be empty
        "missing_trainable_params": missing_trainable,
        "unexpected_frozen_params": frozen_in_optimizer,
        "frozen_in_optimizer": frozen_in_optimizer,               # back-compat alias
        "optimizer_has_frozen_params": bool(frozen_in_optimizer),  # back-compat
        # what the pre-Opacus filter stripped out (frozen backbone stages on E2,
        # plus the DFL fixed conv on both variants); count + capped name sample
        "removed_frozen_from_optimizer_count": len(removed_frozen),
        "removed_frozen_from_optimizer_sample": removed_frozen[:12]
            + ([f"...(+{len(removed_frozen) - 12} more)"] if len(removed_frozen) > 12 else []),
        "accountant_state": _dump_accountant(privacy_engine),
    }
    # sanity: the accountant must have grown by exactly this round's steps
    if cumulative_steps != steps_before + n_steps:
        info["accountant_step_mismatch"] = {"before": steps_before, "round": n_steps, "after": cumulative_steps}
    return state_dict, info
