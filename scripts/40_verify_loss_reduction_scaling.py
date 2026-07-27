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

If Ultralytics' loss is already effectively summed but Opacus is never told
so (dp_sgd.py's make_private() call has no loss_reduction= argument), every
per-sample gradient captured via grad_sample gets an extra, spurious `* n`
on top of what it should be. This was reproduced EXACTLY (ratio == n, to the
last decimal) on a synthetic toy model in an isolated test; this script
checks whether the SAME exact mismatch reproduces on the REAL P2 model with
REAL client data, which is the only way to know whether it explains the
400-600 median per-sample gradient norms observed in the real E2 runs.

Method: build the REAL P2 model from models/base_groupnorm.pt, pull ONE real
physical batch of client0's images through the SAME production data
pipeline (build_trainer_from_checkpoint, matching train_client_round_dp's
own setup exactly -- same freeze stages, same disable_inplace_ops), then run
TWO independent forward+backward passes on that IDENTICAL batch -- one
model/optimizer copy wrapped with loss_reduction="mean" (current dp_sgd.py
behavior, Opacus's own default), one with loss_reduction="sum" (matches
Ultralytics' actual loss convention) -- and compare the resulting per-sample
gradient norms.

optimizer.step() is NEVER called on either copy (zero_grad() only). No
PrivacyEngine/accountant state is persisted anywhere. No checkpoint is
written. Nothing in dp_sgd.py, or any production DP-SGD mechanism, is
modified or touched by running this.

    python scripts/40_verify_loss_reduction_scaling.py --device 0

Output: prints a per-sample comparison table (mean-mode norm, sum-mode norm,
ratio, expected ratio == physical batch size actually processed) and saves
results/audit_dp_seedfix/loss_reduction_probe.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402
import yaml  # noqa: E402
from opacus import PrivacyEngine  # noqa: E402

import fedxpalm  # noqa: E402,F401 (GroupNorm-safe fuse() patch)
from fedxpalm.federated.trainer_utils import build_trainer_from_checkpoint  # noqa: E402
from fedxpalm.models.groupnorm import disable_inplace_ops  # noqa: E402
# reusing dp_sgd.py's own (module-private but same-package) helpers verbatim
# -- NOT reimplemented here, so there is no risk of this probe silently
# diverging from what the real production path actually does
from fedxpalm.privacy.dp_sgd import _per_sample_grad_norms, _strip_frozen_from_optimizer  # noqa: E402

P2_FREEZE_STAGES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 20, 21]


def _build_dp_copy(base_weights: str, overrides: dict, loss_reduction: str, device: str):
    """Fresh model+optimizer loaded from the SAME checkpoint, P2-frozen the
    same way train_client_round_dp does, wrapped by its own PrivacyEngine
    with an explicit loss_reduction. Returns (dp_model, dp_optimizer) --
    the data_loader Opacus also returns is discarded; the real batch used
    for the comparison is fed in manually so both copies see IDENTICAL data."""
    trainer = build_trainer_from_checkpoint(base_weights, overrides)
    disable_inplace_ops(trainer.model)
    trainer._setup_train()
    trainer.model.train()
    _strip_frozen_from_optimizer(trainer.optimizer, trainer.model)

    privacy_engine = PrivacyEngine(accountant="rdp")  # throwaway; never persisted
    dp_model, dp_optimizer, _dp_loader = privacy_engine.make_private(
        module=trainer.model,
        optimizer=trainer.optimizer,
        data_loader=trainer.train_loader,
        noise_multiplier=0.0,      # sigma=0: isolate the scaling question from noise
        max_grad_norm=1e9,         # effectively no clipping: isolate scaling from clipping too
        poisson_sampling=True,
        loss_reduction=loss_reduction,
    )
    return trainer, dp_model, dp_optimizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8,
                        help="physical batch size for the probe (default 8, matching the "
                             "physical_batch used in the real E2 confirmation runs under "
                             "BatchMemoryManager -- this IS the 'n' the hypothesis is about)")
    parser.add_argument("--base-weights", default="models/base_groupnorm.pt")
    parser.add_argument("--imgsz", type=int, default=None)
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
    client0 = sorted(manifest["sizes"], key=lambda c: int(c) if str(c).isdigit() else c)[0]
    clients_dir = splits_dir / "federated_partitions" / "k4_clients"
    data_yaml = str(clients_dir / f"client{client0}" / "data.yaml")

    imgsz = args.imgsz or fl_cfg["model"]["imgsz"]
    overrides = dict(
        data=data_yaml, model=args.base_weights, epochs=1, batch=args.batch, imgsz=imgsz,
        optimizer=fl_cfg["local_training"].get("optimizer", "SGD"),
        lr0=fl_cfg["local_training"].get("lr0", 0.01),
        momentum=fl_cfg["local_training"].get("momentum", 0.9),
        weight_decay=fl_cfg["local_training"].get("weight_decay", 0.0005),
        warmup_epochs=0.0, device=args.device, workers=0, amp=False, plots=False,
        val=False, verbose=False, freeze=P2_FREEZE_STAGES, exist_ok=True,
        project="runs/_loss_reduction_probe_scratch", name="probe",
    )

    # ---- pull ONE real physical batch, from a plain (non-DP-wrapped) trainer,
    # so both loss_reduction variants below see byte-identical input ----
    raw_trainer = build_trainer_from_checkpoint(args.base_weights, overrides)
    disable_inplace_ops(raw_trainer.model)
    raw_trainer._setup_train()
    batch = next(iter(raw_trainer.train_loader))
    batch = raw_trainer.preprocess_batch(batch)
    n_physical = len(batch["img"])
    print(f"Pulled 1 real physical batch from client{client0}: n_physical={n_physical} images "
          f"(target --batch={args.batch})")
    if len(batch["cls"]) == 0:
        print("FAIL: drawn batch has zero ground-truth boxes -- rerun (Poisson/shuffle can draw "
              "an unlucky empty batch); this is not itself evidence of anything")
        return 1

    results = {}
    for loss_reduction in ("mean", "sum"):
        _trainer, dp_model, dp_optimizer = _build_dp_copy(
            args.base_weights, overrides, loss_reduction, args.device)
        dp_optimizer.zero_grad()
        loss, _loss_items = dp_model(batch)
        total_loss = loss.sum()
        if not torch.isfinite(total_loss):
            print(f"FAIL: non-finite loss under loss_reduction={loss_reduction!r}")
            return 1
        total_loss.backward()
        # read grad_sample BEFORE any step()/clip_and_accumulate() call -- same
        # timing dp_sgd.py's own collect_grad_norms=True instrumentation uses
        norms = _per_sample_grad_norms(dp_optimizer)
        dp_optimizer.zero_grad()  # no step() ever called -- purely observational
        results[loss_reduction] = norms.tolist()
        print(f"loss_reduction={loss_reduction!r}: "
              f"median={norms.median():.4f}  p95={norms.quantile(0.95):.4f}  max={norms.max():.4f}")

    norms_mean = torch.tensor(results["mean"])
    norms_sum = torch.tensor(results["sum"])
    if len(norms_mean) != len(norms_sum) or len(norms_mean) == 0:
        print("FAIL: per-sample norm count mismatch or empty -- cannot compare")
        return 1

    ratio = norms_mean / norms_sum.clamp(min=1e-12)
    ratio_matches_n = bool(torch.allclose(ratio, torch.full_like(ratio, float(n_physical)), rtol=1e-3))

    print(f"\nratio (mean/sum) per sample: {[round(r, 4) for r in ratio.tolist()]}")
    print(f"expected ratio if the hypothesis is exactly right: {n_physical} (the physical batch size)")
    print(f"ratio_matches_physical_batch_size (rtol=1e-3): {ratio_matches_n}")

    record = {
        "note": "VERIFICATION PROBE ONLY -- no training, no checkpoint written, no accountant "
                "persisted. Confirms/refutes whether Opacus's default loss_reduction='mean' "
                "spuriously inflates per-sample gradient norms by the physical batch size, given "
                "Ultralytics' own loss already returns mean*batch_size (effectively sum-style).",
        "client_id": client0, "physical_batch_requested": args.batch, "n_physical_observed": n_physical,
        "freeze_stages": P2_FREEZE_STAGES,
        "per_sample_norms_loss_reduction_mean": results["mean"],
        "per_sample_norms_loss_reduction_sum": results["sum"],
        "ratio_mean_over_sum": ratio.tolist(),
        "expected_ratio_if_hypothesis_correct": n_physical,
        "ratio_matches_physical_batch_size": ratio_matches_n,
        "median_norm_mean_mode": float(norms_mean.median()), "median_norm_sum_mode": float(norms_sum.median()),
    }
    out_json = Path("results/audit_dp_seedfix/loss_reduction_probe.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nSaved {out_json}")
    print("\nVERDICT: " + ("HYPOTHESIS CONFIRMED on the real model -- loss_reduction='mean' "
                           "inflates per-sample gradient norms by exactly the physical batch "
                           "size, matching the synthetic reproduction exactly."
                           if ratio_matches_n else
                           "HYPOTHESIS NOT CONFIRMED as stated -- ratio does not match the "
                           "physical batch size; do not apply the loss_reduction='sum' fix "
                           "based on this alone, investigate further."))
    return 0 if ratio_matches_n else 1


if __name__ == "__main__":
    raise SystemExit(main())
