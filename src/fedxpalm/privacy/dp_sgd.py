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

from fedxpalm.federated.client import effective_seed
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


def _is_bad_alloc(exc: BaseException) -> bool:
    """True for a genuine out-of-memory failure: either a real MemoryError,
    or a RuntimeError whose message is actually a wrapped C++ std::bad_alloc
    (some scipy/numpy FFT backends surface it that way depending on platform/
    version). Anything else returns False so the caller re-raises it
    untouched -- this must never become a broad except-and-swallow."""
    if isinstance(exc, MemoryError):
        return True
    if isinstance(exc, RuntimeError) and "bad_alloc" in str(exc).lower():
        return True
    return False


def _get_epsilon_with_fallback(privacy_engine: PrivacyEngine, delta: float) -> dict:
    """Computes epsilon via the PRIMARY accountant (PRV, unconditionally the
    one actually used for accounting/state persistence -- this function never
    touches privacy_engine.accountant itself, read-only). If PRV's own
    get_epsilon() fails with a genuine out-of-memory error (observed in
    practice: PRVAccountant's FFT-based composition needs a discretization
    mesh that grows very large at small sigma combined with small delta and
    many composed steps -- a numerical/memory limitation of that algorithm,
    not a bug in this pipeline), falls back to reconstructing a FRESH,
    throwaway RDPAccountant from the SAME history (list of
    (noise_multiplier, sample_rate, num_steps) tuples -- identical shape for
    both accountant types) purely to produce a reportable epsilon for THIS
    call. The real PRV accountant object/state is never mutated or replaced,
    so per-round persistence (_dump_accountant/_restore_accountant) and every
    sigma that doesn't hit this failure mode are completely unaffected.

    Returns a dict with epsilon, accountant_used ("prv"|"rdp_fallback"),
    prv_failed, and (when applicable) the fallback reason -- so a fallback
    epsilon can never be silently mistaken for a PRV one downstream.
    """
    try:
        epsilon = float(privacy_engine.get_epsilon(delta=delta))
        return {"epsilon": epsilon, "accountant_used": "prv", "prv_failed": False,
               "epsilon_fallback_reason": None}
    except (MemoryError, RuntimeError) as e:
        if not _is_bad_alloc(e):
            raise  # anything else is a real error -- never swallowed
        from opacus.accountants import RDPAccountant
        rdp_acct = RDPAccountant()
        # same (noise_multiplier, sample_rate, num_steps) history shape as PRV --
        # a plain list copy, not a live reference, so nothing about the real
        # PRV accountant's own state is touched
        rdp_acct.history = list(getattr(privacy_engine.accountant, "history", []))
        epsilon = float(rdp_acct.get_epsilon(delta=delta))
        return {"epsilon": epsilon, "accountant_used": "rdp_fallback", "prv_failed": True,
               "epsilon_fallback_reason": f"{type(e).__name__}: {e}"}


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
    physical_batch_size: int | None = None,
    per_layer_max_grad_norms: dict[str, float] | None = None,
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
    clipping-severity diagnostic. `physical_batch_size` (default None =
    unwrapped, unused by any real E1/E2 sweep call site) wraps the DP
    dataloader in Opacus's own `BatchMemoryManager` so the LOGICAL batch
    (hyp["batch_size"], which drives the Poisson sample_rate/expected batch
    size) can exceed the PHYSICAL batch actually placed on the device --
    for scripts/25_diag_noise_signal_probe.py's logical-batch-size sweep.
    clip_and_accumulate()/add_noise() still fire exactly once per LOGICAL
    batch (Opacus's own skip-step accounting), so this changes nothing about
    per-round semantics other than how many physical forward/backward calls
    it takes to reach one logical step.

    `per_layer_max_grad_norms` (default None = the existing FLAT clipping
    path, byte-identical, used by every E1/E2/pilot run so far) opts into
    Opacus's official per-layer clipping (clipping="per_layer" ->
    DPPerLayerOptimizer): a dict mapping every trainable parameter NAME to
    its own clipping threshold C_i. Passing a name-keyed dict (not a bare
    list) makes correctness independent of parameter ordering -- the ordered
    list Opacus requires is built here from the optimizer's own canonical
    param sequence (opacus.optimizers.utils.params, the exact helper
    DPPerLayerOptimizer's own length assert uses), with hard assertions that
    the dict's names and the optimizer's params match EXACTLY (no missing,
    no extra). Privacy accounting is UNCHANGED: Opacus sets the scalar
    sensitivity bound to ||C_vec||_2 internally and adds noise with
    std = sigma * ||C_vec||_2, so with ||C_vec||_2 equal to the flat C the
    mechanism has the same noise multiplier sigma, the same accountant
    history shape, and the same epsilon as flat clipping -- this function
    additionally asserts ||C_vec||_2 matches dp_hyp["max_grad_norm"] within
    0.1% so a mis-scaled threshold vector cannot silently change the
    sensitivity/noise level relative to the flat baseline it is compared
    against.

    Returns (state_dict, info) where state_dict
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
        # per-(seed, round, client) -- same effective_seed() derivation
        # federated/client.py's non-DP path already uses, consumed by
        # SeededDetectionTrainer to drive shuffle + augmentation. Without
        # this, hyp["seed"] was silently dropped: EVERY prior DP run
        # (E1/E2 sweeps, diagnostics 20/23/25/28/36/37) trained under
        # Ultralytics' internal default seed regardless of any --seed CLI
        # value passed in, since args.seed never reached this overrides
        # dict -- confirmed by inspection, not just suspicion. Harmless for
        # every past single-seed run (nothing claimed otherwise), but would
        # have silently defeated any multi-seed variance estimate.
        seed=effective_seed(hyp.get("seed", 0), round_idx, client_id),
        deterministic=True,
        exist_ok=True,
        project=out_dir,
        name=run_name,
    )
    trainer = build_trainer_from_checkpoint(global_weights_path, overrides)
    disable_inplace_ops(trainer.model)  # in case a fresh (non-DP-prepared) checkpoint slips in
    trainer._setup_train()
    # _setup_train() does NOT reliably leave the model in train() mode: it depends on
    # whatever mode the loaded checkpoint's model object was pickled in. base_groupnorm.pt
    # (created by a one-time offline conversion script) happens to be train()-mode, so this
    # was latent; a checkpoint saved via Ultralytics' own training loop (e.g. a warm-started
    # init produced by scripts/34) is pickled in eval() mode (Ultralytics' own save_model()
    # convention), and Opacus's ModuleValidator hard-rejects eval-mode modules in
    # make_private() ("Model needs to be in training mode") before any DP wrapping happens.
    # Force train() explicitly so this doesn't depend on how any given init checkpoint was saved.
    trainer.model.train()

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

    # HARD-STOP here, before ANY DP mechanism activity for this round: no
    # PrivacyEngine yet, no GradSampleModule/DPOptimizer wrapping, no
    # backward(), no clip_and_accumulate(), no add_noise(), no optimizer
    # step, no accountant step. This is deliberately NOT deferred to the
    # caller via the returned `info` dict -- checking only after this whole
    # function (and thus the whole round's real DP training) has already
    # run would mean a genuine frozen/trainable mismatch (or DFL leaking
    # into the optimizer -- DFL is always requires_grad=False regardless of
    # freeze_stages, so it is covered by this SAME trainable_ids/
    # optimizer_ids comparison, not a separate check) gets detected only
    # after real per-sample gradients, clipping, Gaussian noise, and a real
    # accountant step already happened and consumed privacy budget for this
    # client this round. Raising here means the optimizer/model set was
    # already wrong before Opacus ever touched it -- nothing DP-related has
    # run yet, so nothing needs to be undone.
    if missing_trainable or frozen_in_optimizer:
        raise RuntimeError(
            f"round {round_idx} client {client_id}: optimizer/trainable set mismatch BEFORE "
            f"any DP mechanism ran -- missing_trainable={missing_trainable[:5]} "
            f"frozen_in_optimizer={frozen_in_optimizer[:5]} "
            f"(freeze_stages={freeze_stages})")

    privacy_engine = PrivacyEngine(accountant=dp_hyp.get("accountant", "prv"))
    if per_layer_max_grad_norms is None:
        # FLAT clipping -- the pre-existing path, byte-identical
        dp_model, dp_optimizer, dp_loader = privacy_engine.make_private(
            module=trainer.model,
            optimizer=trainer.optimizer,
            data_loader=trainer.train_loader,
            noise_multiplier=dp_hyp["sigma"],
            max_grad_norm=dp_hyp["max_grad_norm"],
            poisson_sampling=True,
            # CANONICAL fix (supersedes an earlier, INCOMPLETE loss_reduction="sum"
            # attempt -- see git history / thesis log for why that was rejected).
            # Ultralytics' own detection loss returns mean_per_sample_loss *
            # batch_size (v8DetectionLoss.loss()) -- a batch-invariant per-sample
            # average, rescaled back up. Opacus's "mean" mode assumes backward()
            # ran on a genuine PyTorch mean and corrects for it by multiplying
            # backprops by n (the actual micro-batch size) when reconstructing
            # grad_sample. Passing Ultralytics' raw (rescaled-up) loss straight
            # to backward() under "mean" double-counts that rescaling, inflating
            # every per-sample grad_sample by an extra factor of n -- this is
            # what scripts/40_verify_loss_reduction_scaling.py measured (ratio
            # exactly 8 for physical_batch=8) and is a genuine measurement bug in
            # grad_norm_stats/clip_fraction reporting.
            #
            # The EARLIER "fix" for this (loss_reduction="sum", literally telling
            # Opacus not to correct at all) repaired that measurement but broke
            # something more important: DPOptimizer.scale_grad() only divides the
            # clipped+noised summed_grad by expected_batch_size when
            # loss_reduction=="mean" -- under "sum" it does nothing. That division
            # is the standard, correct final step of DP-SGD (Abadi et al.): clip
            # each example, sum, add noise, then average over the batch before
            # applying lr0. Removing it made every real logical step's SGD update
            # ~expected_batch_size (~60-64x) too large -- confirmed by a rejected
            # 5-round pilot (tag: rejected_direct_sum_loss_reduction) that showed
            # unstable, oscillating precision/recall and a best mAP50 of 0.027,
            # far below the legacy (pre-fix) run's 0.1766 over the same 5 rounds.
            #
            # The actual bug was narrower than either prior attempt assumed:
            # Ultralytics' loss is a rescaled MEAN, not a genuine SUM -- so the
            # correct move is to keep loss_reduction="mean" (so scale_grad's
            # essential division still happens) and undo Ultralytics' own
            # rescaling explicitly, immediately below, before backward() ever
            # sees it -- dividing by the ACTUAL microbatch size processed in
            # this specific forward call (not logical_batch, not
            # expected_batch_size, not any config value), so the loss handed to
            # backward() is a genuine, undiluted-by-anything-else PyTorch mean,
            # exactly what loss_reduction="mean" is designed to consume. See the
            # explicit division a few lines below, and
            # scripts/41_verify_canonical_loss_normalization.py for the matched
            # 3-way (legacy / rejected direct-sum / canonical) real-model probe
            # that confirms canonical's true per-sample grad norms match
            # direct-sum's (correct clipping-scale measurement) while its final
            # per-step parameter delta matches legacy's (correct, ~1/expected_batch_size
            # update magnitude) -- getting both right simultaneously, in every
            # clip_fraction regime, not just by relying on clipping saturation.
            loss_reduction="mean",
        )
        clipping_mode = "flat"
        per_layer_info = None
    else:
        # PER-LAYER clipping via Opacus's official DPPerLayerOptimizer.
        # Build the ordered threshold list from the optimizer's canonical
        # param sequence -- the same helper DPPerLayerOptimizer itself uses
        # for its length assert -- and hard-verify the name<->param mapping.
        from opacus.optimizers.utils import params as opacus_params

        opt_params = opacus_params(trainer.optimizer)
        opt_names = [name_by_id[id(p)] for p in opt_params]
        dict_names = set(per_layer_max_grad_norms)
        missing_thresholds = sorted(set(opt_names) - dict_names)
        extra_thresholds = sorted(dict_names - set(opt_names))
        if missing_thresholds or extra_thresholds:
            raise ValueError(
                "per_layer_max_grad_norms does not exactly match the optimizer's trainable "
                f"parameter set: missing={missing_thresholds[:5]} extra={extra_thresholds[:5]} "
                f"(counts: {len(missing_thresholds)} missing / {len(extra_thresholds)} extra)")
        c_vec = [float(per_layer_max_grad_norms[n]) for n in opt_names]
        if any(c <= 0 for c in c_vec):
            raise ValueError("per_layer_max_grad_norms must be strictly positive for every tensor")
        c_l2 = float(torch.tensor(c_vec).norm(2))
        flat_c = float(dp_hyp["max_grad_norm"])
        if abs(c_l2 - flat_c) > 1e-3 * flat_c:
            raise ValueError(
                f"||C_vec||_2 = {c_l2:.6f} must equal dp_hyp['max_grad_norm'] = {flat_c} "
                "(within 0.1%) so total sensitivity -- and hence noise scale and accounting -- "
                "stays identical to the flat baseline this run is compared against")
        dp_model, dp_optimizer, dp_loader = privacy_engine.make_private(
            module=trainer.model,
            optimizer=trainer.optimizer,
            data_loader=trainer.train_loader,
            noise_multiplier=dp_hyp["sigma"],
            max_grad_norm=c_vec,
            clipping="per_layer",
            poisson_sampling=True,
            loss_reduction="mean",  # canonical fix -- see the flat-clipping branch's comment above
        )
        clipping_mode = "per_layer"
        per_layer_info = {
            "n_thresholds": len(c_vec),
            "C_vec_l2_norm": c_l2,
            "C_min": min(c_vec), "C_max": max(c_vec),
            "ordered_param_names": opt_names,
        }
    # Hard runtime check: both branches above must have actually wrapped with
    # loss_reduction="mean" -- if a future edit ever drops this argument (or
    # Opacus's own default ever changes), this fails loudly here rather than
    # silently reintroducing either the rejected direct-sum bug (missing
    # scale_grad division) or the original measurement bug (missing explicit
    # microbatch normalization below).
    assert dp_optimizer.loss_reduction == "mean", (
        f"expected dp_optimizer.loss_reduction == 'mean', got {dp_optimizer.loss_reduction!r}")

    # resume this client's privacy budget from prior rounds BEFORE stepping,
    # so get_epsilon() below is cumulative over the whole run
    _restore_accountant(privacy_engine, accountant_state)
    steps_before = _cumulative_steps(privacy_engine)

    sample_rate = float(getattr(dp_loader, "sample_rate",
                                hyp["batch_size"] / max(1, len(dp_loader.dataset))))

    # NOTE: when physical_batch_size is set, `active_loader` yields PHYSICAL
    # micro-batches, so `n_steps`/`max_steps` below count physical iterations,
    # not logical ones -- clip_and_accumulate()/add_noise() (and thus a real
    # step) still only fire once per LOGICAL group (Opacus's own skip-step
    # queue), so max_steps can cut off mid-logical-group, silently discarding
    # that group's partial accumulation without ever completing a step for
    # it. Harmless for a diagnostic that requests generous headroom and reads
    # its OWN step count (e.g. via a step-completion hook), rather than
    # relying on max_steps to mean "N logical steps" -- not currently an
    # issue for any real E1/E2 call site, since none of them pass
    # physical_batch_size.
    if physical_batch_size is not None:
        from opacus.utils.batch_memory_manager import BatchMemoryManager
        batch_ctx = BatchMemoryManager(data_loader=dp_loader, max_physical_batch_size=physical_batch_size,
                                       optimizer=dp_optimizer)
    else:
        from contextlib import nullcontext
        batch_ctx = nullcontext(dp_loader)  # identical to the pre-existing unwrapped loop when unset

    dp_model.train()
    n_steps = 0
    n_logical_steps = 0  # increments only on a REAL (non-skipped) logical DP step -- see below
    nan_inf_detected = False
    observed_norms = [] if collect_grad_norms else None
    with batch_ctx as active_loader:
        for _epoch in range(hyp["epochs_per_round"]):
            for batch in active_loader:
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
                # Ultralytics' loss is mean_per_sample * batch_size (an already
                # batch-invariant average, rescaled up) -- divide by the ACTUAL
                # microbatch size processed in THIS forward call (not
                # logical_batch, not expected_batch_size, not any config value)
                # to undo that rescaling explicitly, so what backward() sees is
                # a genuine PyTorch mean, exactly what loss_reduction="mean"
                # (set above) is designed to consume. See the long comment at
                # this round's make_private() call for the full derivation.
                actual_microbatch_size = int(batch["img"].shape[0])
                if actual_microbatch_size <= 0:
                    raise RuntimeError(f"round {round_idx} client {client_id}: empty physical "
                                       f"microbatch reached backward -- should have been skipped "
                                       f"by the empty-batch check above")
                total_loss = loss.sum() / actual_microbatch_size
                if not torch.isfinite(total_loss):
                    nan_inf_detected = True
                    raise RuntimeError(f"round {round_idx} client {client_id}: non-finite "
                                       f"normalized DP loss ({float(loss.sum())} / "
                                       f"{actual_microbatch_size})")
                total_loss.backward()
                if collect_grad_norms:
                    # must read BEFORE step(): step() -> pre_step() -> clip_and_accumulate()
                    # computes this same quantity internally then discards it
                    observed_norms.append(_per_sample_grad_norms(dp_optimizer))
                dp_optimizer.step()
                # dp_optimizer.step() returns None in BOTH the skipped and the real-step case
                # here (no closure is passed, so a real step just forwards SGD.step()'s own
                # None return) -- it cannot be used to tell them apart. pre_step() (called
                # internally by step()) sets _is_last_step_skipped explicitly in both branches,
                # so read THAT instead: False means this call just completed a real logical DP
                # step (clip_and_accumulate + add_noise + the underlying optimizer step all
                # fired), True means it only accumulated into a not-yet-complete logical group.
                if not getattr(dp_optimizer, "_is_last_step_skipped", False):
                    n_logical_steps += 1
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
        eps_result = _get_epsilon_with_fallback(privacy_engine, dp_hyp["delta"])
        epsilon = eps_result["epsilon"]
        privacy_guarantee = "dp_sgd"
    else:
        eps_result = {"epsilon": None, "accountant_used": None, "prv_failed": False,
                     "epsilon_fallback_reason": None}
        epsilon = None
        privacy_guarantee = "not_applicable_no_noise"
    cumulative_steps = _cumulative_steps(privacy_engine)
    state_dict = {k.replace("_module.", "", 1): v.detach().clone() for k, v in dp_model.state_dict().items()}

    info = {
        "round": round_idx,
        "client_id": client_id,
        "sigma": dp_hyp["sigma"],
        "max_grad_norm": dp_hyp["max_grad_norm"],
        "clipping": clipping_mode,                 # "flat" | "per_layer"
        "loss_reduction": dp_optimizer.loss_reduction,  # "mean" (canonical fix) -- see make_private() comment above
        "expected_batch_size": dp_optimizer.expected_batch_size,  # Opacus's own attribute, unaffected by loss_reduction
        # canonical-fix identity fields -- distinguish this from both the
        # legacy (pre-fix) and rejected direct-sum generations without
        # needing to inspect the code that produced a given record
        "dp_loss_reduction": dp_optimizer.loss_reduction,          # "mean"
        "upstream_loss_convention": "ultralytics_sum",             # what Ultralytics' v8DetectionLoss actually returns
        "explicit_loss_normalization": "actual_microbatch_mean",   # loss.sum() / actual_microbatch_size, done here
        "per_layer": per_layer_info,               # None unless per-layer clipping active
        "delta": dp_hyp["delta"],
        "n_samples": len(dp_loader.dataset),
        "sample_rate_q": sample_rate,
        "steps_this_round": n_steps,  # PHYSICAL loop iterations (== logical when physical_batch_size unset)
        "logical_steps_this_round": n_logical_steps,  # real (non-skipped) DP steps only -- always logical
        "cumulative_steps": cumulative_steps,
        "epsilon": epsilon,                        # cumulative over all rounds; None when sigma=0
        "privacy_guarantee": privacy_guarantee,    # "dp_sgd" | "not_applicable_no_noise"
        # which accountant actually produced `epsilon` above -- "prv" (normal case) or
        # "rdp_fallback" (PRV hit a genuine out-of-memory failure; see
        # _get_epsilon_with_fallback's docstring). Never conflate the two: a
        # rdp_fallback epsilon is NOT a PRV epsilon and must not be reported as one.
        "accountant_used": eps_result["accountant_used"],
        "prv_failed": eps_result["prv_failed"],
        "epsilon_fallback_reason": eps_result["epsilon_fallback_reason"],
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
    # sanity: the accountant must have grown by exactly this round's LOGICAL
    # steps. BUG FIX: this previously compared against n_steps (physical loop
    # iterations -- always >= logical steps, and strictly greater whenever
    # physical_batch_size splits a logical group into multiple physical
    # chunks), so it spuriously fired on every physical_batch_size run even
    # though the accountant itself was correct. n_logical_steps only counts
    # calls where dp_optimizer's own _is_last_step_skipped flag was False,
    # i.e. a real (non-skipped) logical DP step actually completed. Confirmed
    # this bug never affected epsilon (computed independently by Opacus's
    # accountant, untouched by this counter), training, or any reported
    # metric -- purely a stale self-diagnostic assertion.
    if cumulative_steps != steps_before + n_logical_steps:
        info["accountant_step_mismatch"] = {"before": steps_before, "round": n_logical_steps,
                                            "after": cumulative_steps}
    return state_dict, info
