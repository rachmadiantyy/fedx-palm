#!/usr/bin/env python3
"""VERIFICATION-ONLY PROBE (no training, no checkpoint writes, no accountant
persisted) -- checks whether dp_sgd.py's train_client_round_dp computes
per-sample gradient norms inflated by the physical micro-batch size, because
Opacus's PrivacyEngine.make_private() is called without an explicit
loss_reduction, defaulting to "mean", while Ultralytics' own detection loss
already returns `mean_per_sample_loss * batch_size` (effectively sum-style):

    # ultralytics/utils/loss.py, v8DetectionLoss.loss()
    batch_size = preds["boxes"].shape[0]
    loss, loss_detach = self.get_assigned_targets_and_loss(preds, batch)[1:]
    return loss * batch_size, loss_detach

Opacus's own per-sample-gradient reconstruction hook branches on
loss_reduction like this (opacus/grad_sample/grad_sample_module.py):

    if loss_reduction == "mean":
        backprops = backprops * n      # n = the actual micro-batch size
    elif loss_reduction == "sum":
        backprops = backprops          # no extra scaling

STAGE 1 (below, unchanged from the first version, already run and confirmed:
ratio == physical batch size exactly on the real P2 model) proved the
per-sample grad_sample INFLATION is real. It did NOT prove that this
inflation actually changes anything E1/E2 used, because Opacus's own
clipping (`min(C/norm, 1) * grad`) provably cancels a uniform positive
scalar inflation whenever both the true and inflated norms already exceed
C (clip_fraction=1 either way) -- algebra alone doesn't settle two further
questions, hence STAGE 2:

  A. Does loss_reduction also change WHICH samples get clipped (clip_fraction),
     not just the diagnostic norm number?
  B. Is `clip_and_accumulate()`'s actual `p.summed_grad` output really
     identical between "mean" and "sum" (not just algebra saying it should be)?
  C. `DPOptimizer.scale_grad()` divides by `expected_batch_size *
     accumulated_iterations` ONLY when loss_reduction=="mean" -- a clip-stage
     match does NOT imply the final per-step gradient matches too.
  D. Does one COMPLETE logical optimizer step (real API: .step(), no BMM)
     produce an identical parameter delta between the two modes -- first at
     sigma=0 (deterministic), then at sigma=0.75 with an IDENTICALLY SEEDED
     noise generator on both copies (so noise realizations are directly
     comparable, not just their statistics)?
  E. Does the SAME conclusion hold under the ACTUAL production protocol
     (logical batch=64, physical batch=8, BatchMemoryManager) -- not just a
     simplified single-shot batch?

Every probe below reuses dp_sgd.py's own module-private helpers
(_strip_frozen_from_optimizer) and calls Opacus's REAL, unmodified methods
(clip_and_accumulate(), add_noise(), scale_grad(), .step()) directly --
nothing about the DP-SGD mechanism is reimplemented or approximated.
optimizer.step() from Probes A/B/C is never called (observation only,
zero_grad() between reads); Probes D/E call the real .step() but on
THROWAWAY model copies -- no checkpoint is ever written, no accountant
state is ever persisted, dp_sgd.py itself is not imported for its
mechanism (only for two tiny already-audited helper functions) and is
NOT modified by running this file.

    python scripts/40_verify_loss_reduction_scaling.py --device 0

Output: prints Stage 1's per-sample table plus Stage 2's probes A-E, and
saves:
  results/audit_dp_seedfix/loss_reduction_probe_client{id}.json          (stage 1)
  results/audit_dp_seedfix/loss_reduction_probe_stage2_client{id}.json   (stage 2)
Both filenames are keyed by --client-id (default: client0) so re-running for
a different client (e.g. --client-id 2) never overwrites a previous
client's output; both paths are also checked and refused if already
present, so re-running the SAME client never silently overwrites either.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402
from opacus import PrivacyEngine  # noqa: E402
from opacus.utils.batch_memory_manager import BatchMemoryManager  # noqa: E402

import fedxpalm  # noqa: E402,F401 (GroupNorm-safe fuse() patch)
from fedxpalm.federated.trainer_utils import build_trainer_from_checkpoint  # noqa: E402
from fedxpalm.models.groupnorm import disable_inplace_ops  # noqa: E402
# reusing dp_sgd.py's own (module-private but same-package) helpers verbatim
# -- NOT reimplemented here, so there is no risk of this probe silently
# diverging from what the real production path actually does
from fedxpalm.privacy.dp_sgd import _per_sample_grad_norms, _strip_frozen_from_optimizer  # noqa: E402

P2_FREEZE_STAGES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 20, 21]


# ============================================================
# Shared helpers
# ============================================================

def _build_dp_copy(base_weights: str, overrides: dict, loss_reduction: str,
                    sigma: float, max_grad_norm: float, noise_generator=None,
                    poisson_sampling: bool = True):
    """Fresh model+optimizer loaded from the SAME checkpoint, P2-frozen the
    same way train_client_round_dp does, wrapped by its own PrivacyEngine
    with an explicit loss_reduction. Returns (trainer, dp_model, dp_optimizer,
    dp_loader). Two independent calls to this (one per loss_reduction) start
    from byte-identical weights, since both load the same checkpoint file."""
    trainer = build_trainer_from_checkpoint(base_weights, overrides)
    disable_inplace_ops(trainer.model)
    trainer._setup_train()
    trainer.model.train()
    _strip_frozen_from_optimizer(trainer.optimizer, trainer.model)

    privacy_engine = PrivacyEngine(accountant="rdp")  # throwaway; never persisted
    dp_model, dp_optimizer, dp_loader = privacy_engine.make_private(
        module=trainer.model,
        optimizer=trainer.optimizer,
        data_loader=trainer.train_loader,
        noise_multiplier=sigma,
        max_grad_norm=max_grad_norm,
        poisson_sampling=poisson_sampling,
        loss_reduction=loss_reduction,
        noise_generator=noise_generator,
    )
    return trainer, dp_model, dp_optimizer, dp_loader


def _named_params_clean(dp_model) -> dict:
    return {n.replace("_module.", "", 1): p for n, p in dp_model.named_parameters()}


def _concat_grad_samples(dp_optimizer) -> torch.Tensor:
    """(n_samples, total_trainable_params) -- concatenates every parameter's
    grad_sample into one flat per-sample vector. Read-only (grad_samples is a
    property, not a consuming call -- see opacus source), safe to call before
    the real clip_and_accumulate()."""
    grad_samples = dp_optimizer.grad_samples
    n = len(grad_samples[0])
    return torch.cat([g.reshape(n, -1) for g in grad_samples], dim=1)


def _flatten_params(named_params: dict):
    names = sorted(named_params)
    flat = torch.cat([named_params[n].detach().float().reshape(-1) for n in names])
    return flat, names


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item())


def _rel_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm().clamp(min=1e-12))


def _torch_device_str(device_arg: str) -> str:
    """Ultralytics-style --device args ("0", "0,1", "cpu") -> a real torch
    device string ("cuda:0", "cpu"). torch.Generator(device=...) must match
    the model/tensor device, unlike Ultralytics' own --device convention."""
    if device_arg.lower() == "cpu":
        return "cpu"
    if device_arg.startswith("cuda"):
        return device_arg
    return f"cuda:{device_arg.split(',')[0]}"


# ============================================================
# STAGE 1 -- per-sample grad_sample norm inflation (unchanged)
# ============================================================

def stage1_probe(base_weights, overrides, batch, n_physical, client0) -> dict:
    results = {}
    for loss_reduction in ("mean", "sum"):
        _trainer, dp_model, dp_optimizer, _dp_loader = _build_dp_copy(
            base_weights, overrides, loss_reduction, sigma=0.0, max_grad_norm=1e9)
        dp_optimizer.zero_grad()
        loss, _loss_items = dp_model(batch)
        total_loss = loss.sum()
        if not torch.isfinite(total_loss):
            raise RuntimeError(f"non-finite loss under loss_reduction={loss_reduction!r}")
        total_loss.backward()
        norms = _per_sample_grad_norms(dp_optimizer)
        dp_optimizer.zero_grad()
        results[loss_reduction] = norms.tolist()
        print(f"[stage1] loss_reduction={loss_reduction!r}: "
              f"median={norms.median():.4f}  p95={norms.quantile(0.95):.4f}  max={norms.max():.4f}")

    norms_mean = torch.tensor(results["mean"])
    norms_sum = torch.tensor(results["sum"])
    ratio = norms_mean / norms_sum.clamp(min=1e-12)
    ratio_matches_n = bool(torch.allclose(ratio, torch.full_like(ratio, float(n_physical)), rtol=1e-3))

    print(f"\n[stage1] ratio (mean/sum) per sample: {[round(r, 4) for r in ratio.tolist()]}")
    print(f"[stage1] expected ratio if hypothesis correct: {n_physical}")
    print(f"[stage1] ratio_matches_physical_batch_size (rtol=1e-3): {ratio_matches_n}")

    record = {
        "note": "STAGE 1 -- per-sample grad_sample norm inflation only. Does NOT by itself prove "
                "clipped sums or final SGD updates differ -- see stage 2.",
        "client_id": client0, "n_physical_observed": n_physical,
        "per_sample_norms_loss_reduction_mean": results["mean"],
        "per_sample_norms_loss_reduction_sum": results["sum"],
        "ratio_mean_over_sum": ratio.tolist(),
        "expected_ratio_if_hypothesis_correct": n_physical,
        "ratio_matches_physical_batch_size": ratio_matches_n,
    }
    # filename keyed by client_id so re-running for a different client (e.g.
    # --client-id 2) never overwrites a previous client's output. Existence
    # is checked by the caller (main()) BEFORE any probe runs, so this write
    # itself is never expected to collide.
    out_json = Path(f"results/audit_dp_seedfix/loss_reduction_probe_client{client0}.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(record, f, indent=2)
    print(f"[stage1] Saved {out_json}")
    return record


# ============================================================
# STAGE 2 -- Probes A + B + C (share one backward pass per mode)
# ============================================================

def probes_abc(base_weights, overrides, batch, C) -> dict:
    flat_gs, summed, grad_before_scale, grad_after_scale = {}, {}, {}, {}
    expected_batch_size, accumulated_iterations = {}, {}

    for lr in ("mean", "sum"):
        _trainer, dp_model, dp_opt, _dp_loader = _build_dp_copy(
            base_weights, overrides, lr, sigma=0.0, max_grad_norm=C, poisson_sampling=False)
        name_by_id = {id(p): n.replace("_module.", "", 1) for n, p in dp_model.named_parameters()}

        dp_opt.zero_grad()
        loss, _ = dp_model(batch)
        loss.sum().backward()

        # Probe A's raw material: read grad_sample BEFORE clip_and_accumulate
        # consumes it (property read only, non-destructive -- see opacus source)
        flat_gs[lr] = _concat_grad_samples(dp_opt).detach().clone()
        expected_batch_size[lr] = dp_opt.expected_batch_size
        accumulated_iterations[lr] = dp_opt.accumulated_iterations

        # Probe B: the REAL clip_and_accumulate(), not a manual reimplementation
        dp_opt.clip_and_accumulate()
        summed[lr] = {name_by_id[id(p)]: p.summed_grad.detach().clone() for p in dp_opt.params}

        # Probe C: REAL add_noise() (sigma=0 -> deterministic exact zeros) then
        # REAL scale_grad(), reading p.grad at each stage
        dp_opt.add_noise()
        grad_before_scale[lr] = {name_by_id[id(p)]: p.grad.detach().clone() for p in dp_opt.params}
        dp_opt.scale_grad()
        grad_after_scale[lr] = {name_by_id[id(p)]: p.grad.detach().clone() for p in dp_opt.params}
        dp_opt.zero_grad()  # cleanup -- no .step() ever called

    # ---- Probe A: clipping decision, per sample ----
    norm_mean = flat_gs["mean"].norm(dim=1)
    norm_sum = flat_gs["sum"].norm(dim=1)
    clip_factor_mean = (C / (norm_mean + 1e-6)).clamp(max=1.0)
    clip_factor_sum = (C / (norm_sum + 1e-6)).clamp(max=1.0)
    whether_clipped_mean = norm_mean > C
    whether_clipped_sum = norm_sum > C
    clipped_mean = flat_gs["mean"] * clip_factor_mean.unsqueeze(1)
    clipped_sum = flat_gs["sum"] * clip_factor_sum.unsqueeze(1)
    ratio = norm_mean / norm_sum.clamp(min=1e-12)
    cos_per_sample = F.cosine_similarity(clipped_mean, clipped_sum, dim=1)

    per_sample = [
        {"raw_norm_mean": float(norm_mean[i]), "raw_norm_sum": float(norm_sum[i]),
         "ratio_mean_over_sum": float(ratio[i]),
         "clip_factor_mean": float(clip_factor_mean[i]), "clip_factor_sum": float(clip_factor_sum[i]),
         "whether_clipped_mean": bool(whether_clipped_mean[i]), "whether_clipped_sum": bool(whether_clipped_sum[i]),
         "clipped_grad_norm_mean": float(clipped_mean[i].norm()), "clipped_grad_norm_sum": float(clipped_sum[i].norm()),
         "cosine_similarity_clipped": float(cos_per_sample[i])}
        for i in range(len(norm_mean))
    ]
    probe_a = {
        "per_sample": per_sample,
        "fraction_clipped_mean": float(whether_clipped_mean.float().mean()),
        "fraction_clipped_sum": float(whether_clipped_sum.float().mean()),
        "clip_fraction_matches": bool(torch.equal(whether_clipped_mean, whether_clipped_sum)),
    }
    print(f"\n[probe A] fraction_clipped: mean={probe_a['fraction_clipped_mean']:.4f}  "
          f"sum={probe_a['fraction_clipped_sum']:.4f}  matches={probe_a['clip_fraction_matches']}")

    # ---- Probe B: clip_and_accumulate() output comparison ----
    names_b = sorted(summed["mean"])
    assert names_b == sorted(summed["sum"]), "parameter name sets differ between mean/sum copies"
    per_tensor_max_abs_diff = {n: float((summed["mean"][n].float() - summed["sum"][n].float()).abs().max())
                              for n in names_b}
    gm = torch.cat([summed["mean"][n].float().reshape(-1) for n in names_b])
    gs = torch.cat([summed["sum"][n].float().reshape(-1) for n in names_b])
    probe_b = {
        "global_summed_grad_norm_mean": float(gm.norm()), "global_summed_grad_norm_sum": float(gs.norm()),
        "max_abs_diff": float((gm - gs).abs().max()), "relative_error": _rel_error(gm, gs),
        "cosine_similarity": _cos(gm, gs), "bitwise_equal": bool(torch.equal(gm, gs)),
        "per_tensor_max_abs_diff_top5": dict(sorted(per_tensor_max_abs_diff.items(),
                                                     key=lambda kv: -kv[1])[:5]),
        "pass_tight_tolerance": _rel_error(gm, gs) < 1e-4,
    }
    print(f"[probe B] summed_grad relative_error={probe_b['relative_error']:.2e}  "
          f"cosine={probe_b['cosine_similarity']:.8f}  bitwise_equal={probe_b['bitwise_equal']}  "
          f"PASS={probe_b['pass_tight_tolerance']}")

    # ---- Probe C: scale_grad() effect ----
    def _flat(d):
        return torch.cat([d[n].float().reshape(-1) for n in names_b])
    before_mean, before_sum = _flat(grad_before_scale["mean"]), _flat(grad_before_scale["sum"])
    after_mean, after_sum = _flat(grad_after_scale["mean"]), _flat(grad_after_scale["sum"])
    predicted_ratio = 1.0 / (expected_batch_size["mean"] * accumulated_iterations["mean"])
    probe_c = {
        "expected_batch_size": {"mean": expected_batch_size["mean"], "sum": expected_batch_size["sum"]},
        "accumulated_iterations": {"mean": accumulated_iterations["mean"], "sum": accumulated_iterations["sum"]},
        "global_grad_norm_before_scale": {"mean": float(before_mean.norm()), "sum": float(before_sum.norm())},
        "global_grad_norm_after_scale": {"mean": float(after_mean.norm()), "sum": float(after_sum.norm())},
        "ratio_after_scale_mean_over_sum": float(after_mean.norm() / after_sum.norm().clamp(min=1e-12)),
        "predicted_ratio_from_scale_grad_division": predicted_ratio,
        "cosine_similarity_after_scale": _cos(after_mean, after_sum),
    }
    print(f"[probe C] ratio_after_scale (mean/sum)={probe_c['ratio_after_scale_mean_over_sum']:.6f}  "
          f"predicted={predicted_ratio:.6f}  cosine={probe_c['cosine_similarity_after_scale']:.8f}")

    return {"probe_a_clipping_decision": probe_a, "probe_b_clip_and_accumulate": probe_b,
            "probe_c_scale_grad": probe_c}


# ============================================================
# STAGE 2 -- Probe D: one complete logical step, no BMM
# ============================================================

def probe_d(base_weights, overrides_full_batch, batch, sigma, C, noise_seed, device: str) -> dict:
    gen_device = _torch_device_str(device)

    def _one_step(loss_reduction, sigma_val, gen_seed):
        gen = None
        if gen_seed is not None:
            # must match the model/tensor device -- add_noise() calls torch.normal()
            # directly on CUDA tensors when --device is a GPU, and a CPU-default
            # generator() cannot be used as the source of randomness for that call
            gen = torch.Generator(device=gen_device)
            gen.manual_seed(gen_seed)
        _trainer, dp_model, dp_opt, _dp_loader = _build_dp_copy(
            base_weights, overrides_full_batch, loss_reduction, sigma=sigma_val, max_grad_norm=C,
            noise_generator=gen, poisson_sampling=False)
        pre_flat, names = _flatten_params(_named_params_clean(dp_model))
        dp_opt.zero_grad()
        loss, _ = dp_model(batch)
        loss.sum().backward()
        dp_opt.step()  # REAL public API -- clip_and_accumulate + add_noise + scale_grad + SGD step
        post_flat, _ = _flatten_params(_named_params_clean(dp_model))
        return post_flat - pre_flat, names

    out = {}
    for label, sigma_val, seed in (("sigma0", 0.0, None), (f"sigma{sigma}", sigma, noise_seed)):
        delta_mean, names = _one_step("mean", sigma_val, seed)
        delta_sum, _ = _one_step("sum", sigma_val, seed)
        ratio = float(delta_mean.norm() / delta_sum.norm().clamp(min=1e-12))
        out[label] = {
            "sigma": sigma_val, "noise_seed": seed,
            "global_delta_norm_mean": float(delta_mean.norm()), "global_delta_norm_sum": float(delta_sum.norm()),
            "ratio_delta_mean_over_sum": ratio,
            "cosine_similarity": _cos(delta_mean, delta_sum),
            "max_abs_diff": float((delta_mean - delta_sum).abs().max()),
            "relative_error": _rel_error(delta_mean, delta_sum),
        }
        print(f"[probe D:{label}] ratio_delta(mean/sum)={ratio:.6f}  "
              f"cosine={out[label]['cosine_similarity']:.8f}  "
              f"relative_error={out[label]['relative_error']:.2e}")
    return out


# ============================================================
# STAGE 2 -- Probe E: real production protocol (BatchMemoryManager)
# ============================================================

def probe_e(base_weights, overrides, sigma, C, physical_batch, max_chunks=64) -> dict:
    out = {}
    for lr in ("mean", "sum"):
        trainer, dp_model, dp_opt, dp_loader = _build_dp_copy(
            base_weights, overrides, lr, sigma=sigma, max_grad_norm=C, poisson_sampling=True)
        pre_flat, names = _flatten_params(_named_params_clean(dp_model))
        chunk_sizes = []
        accumulated_iterations_at_real_step = None
        expected_batch_size = dp_opt.expected_batch_size
        n_iter = 0
        with BatchMemoryManager(data_loader=dp_loader, max_physical_batch_size=physical_batch,
                                optimizer=dp_opt) as active_loader:
            for chunk in active_loader:
                if len(chunk["img"]) == 0 or len(chunk["cls"]) == 0:
                    continue
                chunk = trainer.preprocess_batch(chunk)
                chunk_sizes.append(len(chunk["img"]))
                dp_opt.zero_grad()
                loss, _ = dp_model(chunk)
                loss.sum().backward()
                accumulated_iterations_at_real_step = dp_opt.accumulated_iterations
                dp_opt.step()
                n_iter += 1
                if not dp_opt._is_last_step_skipped:
                    break  # exactly one REAL logical step completed
                if n_iter >= max_chunks:
                    raise RuntimeError(f"no real logical step completed after {max_chunks} physical "
                                       f"chunks -- expected_batch_size={expected_batch_size} too large "
                                       f"relative to physical_batch={physical_batch}?")
        post_flat, _ = _flatten_params(_named_params_clean(dp_model))
        out[lr] = {
            "chunk_sizes": chunk_sizes, "expected_batch_size": expected_batch_size,
            "accumulated_iterations_at_real_step": accumulated_iterations_at_real_step,
            "delta": post_flat - pre_flat,
        }
        print(f"[probe E:{lr}] chunk_sizes={chunk_sizes}  expected_batch_size={expected_batch_size}  "
              f"accumulated_iterations_at_real_step={accumulated_iterations_at_real_step}")

    delta_mean, delta_sum = out["mean"]["delta"], out["sum"]["delta"]
    ratio = float(delta_mean.norm() / delta_sum.norm().clamp(min=1e-12))
    result = {
        "chunk_sizes_mean": out["mean"]["chunk_sizes"], "chunk_sizes_sum": out["sum"]["chunk_sizes"],
        "expected_batch_size": out["mean"]["expected_batch_size"],
        "accumulated_iterations_at_real_step_mean": out["mean"]["accumulated_iterations_at_real_step"],
        "accumulated_iterations_at_real_step_sum": out["sum"]["accumulated_iterations_at_real_step"],
        "global_delta_norm_mean": float(delta_mean.norm()), "global_delta_norm_sum": float(delta_sum.norm()),
        "ratio_delta_mean_over_sum": ratio, "cosine_similarity": _cos(delta_mean, delta_sum),
        "max_abs_diff": float((delta_mean - delta_sum).abs().max()), "relative_error": _rel_error(delta_mean, delta_sum),
    }
    print(f"[probe E] ratio_delta(mean/sum)={ratio:.6f}  cosine={result['cosine_similarity']:.8f}  "
          f"relative_error={result['relative_error']:.2e}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8,
                        help="physical batch size for stage 1 / probes A-C (default 8, matching "
                             "the physical_batch used in the real E2 confirmation runs)")
    parser.add_argument("--logical-batch", type=int, default=64,
                        help="logical batch size for probe D's single-shot (no-BMM) step")
    parser.add_argument("--physical-batch", type=int, default=8,
                        help="physical batch size for probe E's BatchMemoryManager step")
    parser.add_argument("--c", type=float, default=1.0, help="clipping norm C for probes A-E")
    parser.add_argument("--sigma-nonzero", type=float, default=0.75,
                        help="the real E2 sigma, used for probe D/E's noisy comparisons")
    parser.add_argument("--noise-seed", type=int, default=1234,
                        help="fixed torch.Generator seed shared by both loss_reduction copies "
                             "for probe D's sigma>0 sub-test, so noise realizations are directly "
                             "comparable rather than merely statistically similar")
    parser.add_argument("--base-weights", default="models/base_groupnorm.pt")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--client-id", default=None,
                        help="which client's real data to probe (default: the first client in "
                             "sorted order, i.e. client0 -- unchanged from before). Pass e.g. "
                             "--client-id 2 to re-verify the same conclusion on a different "
                             "client's sample count/expected_batch_size")
    args = parser.parse_args()

    if not Path(args.base_weights).exists():
        print(f"FAIL: {args.base_weights} not found")
        return 1

    with open("configs/dataset.yaml") as f:
        ds_cfg = yaml.safe_load(f)
    with open("configs/fl_config.yaml") as f:
        fl_cfg = yaml.safe_load(f)

    splits_dir = Path(ds_cfg["output_dir"])
    manifest_path = splits_dir / "federated_partitions" / "manifest.json"
    if not manifest_path.exists():
        print(f"FAIL: {manifest_path} not found")
        return 1
    with open(manifest_path) as f:
        manifest = json.load(f)["4"]
    if args.client_id is not None:
        if args.client_id not in manifest["sizes"]:
            print(f"FAIL: --client-id {args.client_id!r} not found in manifest (available: "
                  f"{sorted(manifest['sizes'])})")
            return 1
        client0 = args.client_id
    else:
        client0 = sorted(manifest["sizes"], key=lambda c: int(c) if str(c).isdigit() else c)[0]
    print(f"Probing client{client0} (n_samples={manifest['sizes'][client0]})")

    # both output paths are keyed by client_id -- checked BEFORE any probe
    # runs so a re-run for a different client never overwrites a previous
    # client's results, and this client's own re-run doesn't silently clobber
    # an existing file either
    stage1_out = Path(f"results/audit_dp_seedfix/loss_reduction_probe_client{client0}.json")
    stage2_out = Path(f"results/audit_dp_seedfix/loss_reduction_probe_stage2_client{client0}.json")
    for p in (stage1_out, stage2_out):
        if p.exists():
            print(f"FAIL: {p} already exists -- refusing to overwrite; move/rename it first if "
                  f"you intend to redo this client's probe")
            return 1

    clients_dir = splits_dir / "federated_partitions" / "k4_clients"
    data_yaml = str(clients_dir / f"client{client0}" / "data.yaml")

    imgsz = args.imgsz or fl_cfg["model"]["imgsz"]

    def make_overrides(batch_size):
        return dict(
            data=data_yaml, model=args.base_weights, epochs=1, batch=batch_size, imgsz=imgsz,
            optimizer=fl_cfg["local_training"].get("optimizer", "SGD"),
            lr0=fl_cfg["local_training"].get("lr0", 0.01),
            momentum=fl_cfg["local_training"].get("momentum", 0.9),
            weight_decay=fl_cfg["local_training"].get("weight_decay", 0.0005),
            warmup_epochs=0.0, device=args.device, workers=0, amp=False, plots=False,
            val=False, verbose=False, freeze=P2_FREEZE_STAGES, exist_ok=True,
            project="runs/_loss_reduction_probe_scratch", name="probe",
        )

    overrides_stage1 = make_overrides(args.batch)

    # ---- pull ONE real physical batch (stage 1 / probes A-C share this) ----
    raw_trainer = build_trainer_from_checkpoint(args.base_weights, overrides_stage1)
    disable_inplace_ops(raw_trainer.model)
    raw_trainer._setup_train()
    batch_small = next(iter(raw_trainer.train_loader))
    batch_small = raw_trainer.preprocess_batch(batch_small)
    n_physical = len(batch_small["img"])
    print(f"Pulled 1 real physical batch from client{client0}: n_physical={n_physical} images "
          f"(target --batch={args.batch})")
    if len(batch_small["cls"]) == 0:
        print("FAIL: drawn batch has zero ground-truth boxes -- rerun (unlucky empty draw)")
        return 1

    print("\n" + "=" * 60 + "\nSTAGE 1 -- per-sample grad_sample norm inflation\n" + "=" * 60)
    stage1 = stage1_probe(args.base_weights, overrides_stage1, batch_small, n_physical, client0)

    print("\n" + "=" * 60 + "\nSTAGE 2 -- Probes A/B/C (clip decision, clip_and_accumulate, scale_grad)\n" + "=" * 60)
    abc = probes_abc(args.base_weights, overrides_stage1, batch_small, args.c)

    print("\n" + "=" * 60 + "\nSTAGE 2 -- Probe D (one complete logical step, no BMM)\n" + "=" * 60)
    overrides_logical = make_overrides(args.logical_batch)
    raw_trainer_d = build_trainer_from_checkpoint(args.base_weights, overrides_logical)
    disable_inplace_ops(raw_trainer_d.model)
    raw_trainer_d._setup_train()
    batch_logical = next(iter(raw_trainer_d.train_loader))
    batch_logical = raw_trainer_d.preprocess_batch(batch_logical)
    if len(batch_logical["cls"]) == 0:
        print("FAIL: logical-batch draw has zero ground-truth boxes -- rerun")
        return 1
    d = probe_d(args.base_weights, overrides_logical, batch_logical, args.sigma_nonzero, args.c,
                args.noise_seed, args.device)

    print("\n" + "=" * 60 + "\nSTAGE 2 -- Probe E (real protocol: BatchMemoryManager)\n" + "=" * 60)
    overrides_e = make_overrides(args.logical_batch)
    e = probe_e(args.base_weights, overrides_e, args.sigma_nonzero, args.c, args.physical_batch)

    # ---- verdict ----
    clip_fraction_matches = abc["probe_a_clipping_decision"]["clip_fraction_matches"]
    summed_grad_pass = abc["probe_b_clip_and_accumulate"]["pass_tight_tolerance"]
    final_delta_sigma0_pass = d["sigma0"]["relative_error"] < 1e-3
    final_delta_sigma_nonzero_pass = d[f"sigma{args.sigma_nonzero}"]["relative_error"] < 1e-3
    probe_e_pass = e["relative_error"] < 1e-3

    if not clip_fraction_matches or not summed_grad_pass:
        verdict = "C"
        verdict_text = ("Clipping decisions juga berbeda: loss_reduction mismatch memengaruhi "
                        "clipping serta final update.")
    elif summed_grad_pass and (final_delta_sigma0_pass and final_delta_sigma_nonzero_pass and probe_e_pass):
        verdict = "A"
        verdict_text = ("Clipped sums dan final SGD updates identik: hasil E1/E2 lama TIDAK "
                        "terpengaruh; hanya diagnostics norm yang salah.")
    elif summed_grad_pass and not (final_delta_sigma0_pass and final_delta_sigma_nonzero_pass and probe_e_pass):
        verdict = "B"
        verdict_text = ("Clipped sums identik tetapi final SGD update berbeda karena scale_grad: "
                        "E1/E2 lama memakai unintended effective optimization scale dan perlu "
                        "diperlakukan sebagai legacy baseline.")
    else:
        verdict = "D"
        verdict_text = "Temuan lain -- lihat angka probe A-E di atas/JSON untuk detail."

    print("\n" + "=" * 60)
    print(f"VERDICT: {verdict} -- {verdict_text}")
    print("=" * 60)

    record = {
        "note": "STAGE 2 -- verification-only probe, no training, no checkpoint written, no "
                "accountant persisted, dp_sgd.py NOT modified.",
        "config": {"c": args.c, "sigma_nonzero": args.sigma_nonzero, "noise_seed": args.noise_seed,
                  "physical_batch_stage1_abc": args.batch, "logical_batch_probe_d": args.logical_batch,
                  "logical_batch_probe_e": args.logical_batch, "physical_batch_probe_e": args.physical_batch,
                  "client_id": client0},
        "probe_a_clipping_decision": abc["probe_a_clipping_decision"],
        "probe_b_clip_and_accumulate": abc["probe_b_clip_and_accumulate"],
        "probe_c_scale_grad": abc["probe_c_scale_grad"],
        "probe_d_one_step_delta": d,
        "probe_e_batch_memory_manager": e,
        "verdict": verdict, "verdict_text": verdict_text,
    }
    out_json = Path(f"results/audit_dp_seedfix/loss_reduction_probe_stage2_client{client0}.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nSaved {out_json}")
    return 0 if verdict == "A" else 1


if __name__ == "__main__":
    raise SystemExit(main())
