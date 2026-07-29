#!/usr/bin/env python3
"""Non-private warm-start training on the PUBLIC auxiliary dataset (from
scripts/33), producing a NEW init checkpoint for E2: models/base_groupnorm_warmstart.pt.

SCOPE: this is standard, ordinary transfer-learning fine-tuning -- plain SGD,
no Opacus, no PrivacyEngine, no clipping, no noise, no federation, no client
data, no accountant. It touches ONLY the public auxiliary dataset built by
scripts/33. This is not a new architecture and not a new privacy mechanism:
the output checkpoint has the EXACT SAME shape/architecture as
models/base_groupnorm.pt (verified below) -- it is a drop-in replacement for
--init-weights in scripts/21 and for global_weights_path in the E2 DP path,
nothing else changes.

WHY ONLY THE P2 SUBSET IS TRAINED HERE (freeze=P2_FROZEN_STAGES, the same
list every P2 script already uses): this isolates exactly the hypothesis
being tested -- does warm-starting P2's own trainable stages (16/19/22/23)
on public in-domain data, before DP noise is ever added, raise per-sample
gradient SIGNAL (measured next, via the existing scripts/25 probe) without
touching noise or epsilon at all? Also warm-starting the backbone would
confound the result (unclear whether any later improvement came from
backbone adaptation or from P2 adaptation) -- and P2's own backbone stays
frozen in every DP run anyway, so backbone weights are irrelevant to what
E2 will ever update.

CLASS-COUNT NOTE: the auxiliary dataset's nc (from scripts/33's data.yaml)
may be less than 6 (e.g. a single generic "bunch" class). No checkpoint
surgery is needed: models/base_groupnorm.pt's head already has 6 output
channels, and training on aux labels using indices 0..(nc_aux-1) simply
exercises a SUBSET of those channels -- scripts/33 already hard-fails if
nc_aux > 6 (which would be unsafe). The untouched channels keep their
original values and will be fully retrained on the real 6-class labels once
federated training begins regardless, exactly as happens in every ordinary
transfer-learning fine-tune.

  python scripts/34_build_warmstart_checkpoint.py --device 0 --data data/warmstart_ffb/data.yaml

Outputs:
  models/base_groupnorm_warmstart.pt      (new; base_groupnorm.pt untouched;
                                            refuses to overwrite without --force)
  runs/warmstart_ffb/train/                (Ultralytics run artifacts)
  results/warmstart_checkpoint_audit.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import yaml  # noqa: E402

import fedxpalm  # noqa: E402,F401 (GroupNorm-safe fuse() patch)
from fedxpalm.eval.detection_metrics import evaluate_detector  # noqa: E402
from fedxpalm.federated.trainer_utils import build_trainer_from_checkpoint  # noqa: E402

# same list every P2 script uses (dp_config.yaml's "variants.partial" still holds the
# older P0 [0..10] pattern -- P2 has always been passed explicitly, matching that convention)
P2_FROZEN_STAGES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 20, 21]
EXPECTED_BASE_TOTAL = 2_591_010  # locked P2 accounting: 929,522 trainable + 1,661,488 frozen


def _arch_fingerprint(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    n_bn = sum(1 for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm))
    n_gn = sum(1 for m in model.modules() if isinstance(m, nn.GroupNorm))
    shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    return {"total_params": int(total), "batchnorm_count": n_bn, "groupnorm_count": n_gn, "shapes": shapes}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    parser.add_argument("--base", default="models/base_groupnorm.pt")
    parser.add_argument("--data", required=True, help="path to scripts/33's data.yaml")
    parser.add_argument("--out", default="models/base_groupnorm_warmstart.pt")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--optimizer", default="SGD")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--out-dir", default="runs/warmstart_ffb")
    args = parser.parse_args()

    if not Path(args.base).exists():
        print(f"FAIL: {args.base} not found"); return 1
    if not Path(args.data).exists():
        print(f"FAIL: {args.data} not found -- run scripts/33_prepare_warmstart_dataset.py first")
        return 1
    if Path(args.out).exists() and not args.force:
        print(f"FAIL: {args.out} already exists -- refusing to overwrite (use --force only if intended)")
        return 1

    with open(args.data) as f:
        aux_cfg = yaml.safe_load(f)
    print(f"auxiliary dataset: {args.data}  nc={aux_cfg['nc']}  (must be <= 6, enforced by scripts/33)")
    if aux_cfg["nc"] > 6:
        print(f"FAIL: aux nc={aux_cfg['nc']} > 6 -- scripts/33 should have refused this; do not proceed")
        return 1

    # ---- pre-audit: fingerprint the base checkpoint before anything happens ----
    base_ckpt = torch.load(args.base, map_location="cpu", weights_only=False)
    base_model = base_ckpt["model"] if isinstance(base_ckpt, dict) and "model" in base_ckpt else base_ckpt
    base_fp = _arch_fingerprint(base_model)
    print(f"[pre] base: total_params={base_fp['total_params']} (expected {EXPECTED_BASE_TOTAL})  "
          f"BN={base_fp['batchnorm_count']}  GN={base_fp['groupnorm_count']}")
    failures = []
    if base_fp["total_params"] != EXPECTED_BASE_TOTAL:
        failures.append(f"base total {base_fp['total_params']} != {EXPECTED_BASE_TOTAL}")
    if base_fp["batchnorm_count"] != 0 or base_fp["groupnorm_count"] != 81:
        failures.append(f"base norm layout BN={base_fp['batchnorm_count']}/GN={base_fp['groupnorm_count']} != 0/81")

    overrides = dict(
        data=args.data,
        model=args.base,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        optimizer=args.optimizer,
        lr0=args.lr0,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        freeze=P2_FROZEN_STAGES,     # ONLY P2's stages train; backbone stays frozen (see docstring)
        device=args.device,
        project=args.out_dir,
        name="train",
        exist_ok=True,
        plots=False,
        verbose=True,
        val=True,
    )
    print(f"Non-private warm-start: freeze={P2_FROZEN_STAGES}  epochs={args.epochs}  "
          f"imgsz={args.imgsz}  batch={args.batch}  optimizer={args.optimizer}  lr0={args.lr0}  "
          f"-- NO Opacus, NO clipping, NO noise, NO federation, PUBLIC data only -> {args.out_dir}")
    trainer = build_trainer_from_checkpoint(args.base, overrides)
    trainer.train()

    best_weights = Path(args.out_dir) / "train" / "weights" / "best.pt"
    if not best_weights.exists():
        print(f"FAIL: expected {best_weights}, training may have failed"); return 1

    # ---- post-audit: architecture must be UNCHANGED, only weights differ ----
    post_ckpt = torch.load(best_weights, map_location="cpu", weights_only=False)
    post_model = post_ckpt["model"] if isinstance(post_ckpt, dict) and "model" in post_ckpt else post_ckpt
    post_fp = _arch_fingerprint(post_model)
    shape_mismatches = [k for k in base_fp["shapes"] if base_fp["shapes"].get(k) != post_fp["shapes"].get(k)]
    print(f"[post] warm-started: total_params={post_fp['total_params']}  "
          f"BN={post_fp['batchnorm_count']}  GN={post_fp['groupnorm_count']}  "
          f"shape_mismatches_vs_base={len(shape_mismatches)}")
    if post_fp["total_params"] != base_fp["total_params"]:
        failures.append(f"post total_params {post_fp['total_params']} != base {base_fp['total_params']}")
    if shape_mismatches:
        failures.append(f"shape mismatches vs base: {shape_mismatches[:5]}")
    if post_fp["batchnorm_count"] != 0 or post_fp["groupnorm_count"] != 81:
        failures.append(f"post norm layout BN={post_fp['batchnorm_count']}/GN={post_fp['groupnorm_count']} != 0/81")

    # frozen-region check: stages NOT in P2_FROZEN_STAGES trained; the rest didn't
    sd_base, sd_post = base_model.state_dict(), post_model.state_dict()
    freeze_names = [f"model.{s}." for s in P2_FROZEN_STAGES] + [".dfl"]
    frozen_changed = trainable_changed = False
    for k in sd_post:
        if k not in sd_base or sd_base[k].shape != sd_post[k].shape:
            continue
        differs = not torch.equal(sd_base[k].float(), sd_post[k].float())
        is_frozen = any(x in k for x in freeze_names)
        if is_frozen and differs:
            frozen_changed = True
        if not is_frozen and differs:
            trainable_changed = True
    print(f"frozen_region_changed={frozen_changed} (must be False)  "
          f"trainable_region_changed={trainable_changed} (must be True)")
    if frozen_changed:
        failures.append("frozen region changed during warm-start (freeze mechanism failure)")
    if not trainable_changed:
        failures.append("trainable (P2) region did NOT change -- warm-start had no effect")

    if failures:
        for msg in failures:
            print(f"FAIL: {msg}")
        print("WARM-START CHECKPOINT AUDIT: FAILED -- not saving as models/base_groupnorm_warmstart.pt")
        return 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(post_ckpt, args.out)
    print(f"Saved {args.out} (base checkpoint {args.base} untouched)")

    val_metrics = evaluate_detector(str(best_weights), args.data, split="val", imgsz=args.imgsz, device=args.device)
    print(f"warm-start aux-domain val: mAP50={val_metrics.get('map50')}  "
          f"mAP50-95={val_metrics.get('map50_95')}  (NOT comparable to any B1/B2/E2 result -- "
          f"different dataset/classes, informational only)")

    record = {
        "note": "NON-PRIVATE warm-start on PUBLIC auxiliary data only -- no client/federated data, "
                "no DP mechanism. Architecture identical to base_groupnorm.pt (verified below).",
        "base_checkpoint": args.base, "output_checkpoint": args.out,
        "aux_data_yaml": args.data, "aux_nc": aux_cfg["nc"],
        "freeze_stages": P2_FROZEN_STAGES,
        "hyp": {"epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
                "optimizer": args.optimizer, "lr0": args.lr0, "momentum": args.momentum,
                "weight_decay": args.weight_decay},
        "pre_audit": {k: v for k, v in base_fp.items() if k != "shapes"},
        "post_audit": {k: v for k, v in post_fp.items() if k != "shapes"},
        "shape_mismatches_vs_base": shape_mismatches,
        "frozen_region_changed": frozen_changed, "trainable_region_changed": trainable_changed,
        "aux_domain_val_metrics": val_metrics,
        "failures": failures, "passed": not failures,
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/warmstart_checkpoint_audit.json", "w") as f:
        json.dump(record, f, indent=2)
    print("Saved results/warmstart_checkpoint_audit.json")
    print("\nNext step: run scripts/25_diag_noise_signal_probe.py with "
          f"--base-weights {args.out} at sigma=0.75/batch64 (P2 freeze/subset-label) and compare "
          "signal_norm_before_noise against the existing COCO-init measurement (2.827 at batch8) "
          "-- this is the falsifiable test of whether warm-start actually raised signal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
