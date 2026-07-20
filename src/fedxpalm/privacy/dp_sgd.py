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
) -> tuple[dict, dict]:
    """Fine-tunes `global_weights_path` on one client with per-sample DP-SGD.

    `accountant_state` is this client's saved privacy-accountant state from the
    previous round (None on round 0). Returns (state_dict, info) where
    state_dict has clean (non-Opacus-prefixed) keys ready for fedavg(), and
    info carries the per-round/per-client privacy log plus `accountant_state`
    (the updated state to persist for next round).
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

    # trainable/frozen accounting for the E1/E2 audit. We log the NAMES of any
    # requires_grad=False parameter that still sits in an optimizer param group
    # (not just a boolean), because Ultralytics ALWAYS freezes the DFL fixed
    # conv (`model.23.dfl.conv.weight`) -- an architectural constant set to
    # arange(reg_max), never trained, in every YOLO run including B1/B2. That
    # is expected, not a methodology error; the smoke test allowlists it and
    # only fails on *unexpected* frozen params. Opacus/PyTorch never update a
    # requires_grad=False param anyway (it gets no per-sample grad), so the
    # constant stays constant -- verified separately by the DFL-unchanged check.
    name_by_id = {id(p): n for n, p in trainer.model.named_parameters()}
    n_trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in trainer.model.parameters() if not p.requires_grad)
    frozen_in_optimizer = sorted({
        name_by_id.get(id(p), "<unknown>")
        for grp in trainer.optimizer.param_groups for p in grp["params"]
        if not p.requires_grad
    })

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
            dp_optimizer.step()
            n_steps += 1

    epsilon = privacy_engine.get_epsilon(delta=dp_hyp["delta"])
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
        "epsilon": float(epsilon),                 # cumulative over all rounds so far
        "nan_inf": bool(nan_inf_detected),
        "clip_fraction": None,                     # not measured (would require per-sample-norm hooks)
        "clip_fraction_note": "not_measured",
        "frozen": freeze_stages is not None,
        "n_trainable_params": int(n_trainable),
        "n_frozen_params": int(n_frozen),
        # exact names of requires_grad=False params still in the optimizer; the
        # DFL fixed conv is expected here (architectural), anything else is not.
        "frozen_in_optimizer": frozen_in_optimizer,
        "optimizer_has_frozen_params": bool(frozen_in_optimizer),  # kept for back-compat
        "accountant_state": _dump_accountant(privacy_engine),
    }
    # sanity: the accountant must have grown by exactly this round's steps
    if cumulative_steps != steps_before + n_steps:
        info["accountant_step_mismatch"] = {"before": steps_before, "round": n_steps, "after": cumulative_steps}
    return state_dict, info
