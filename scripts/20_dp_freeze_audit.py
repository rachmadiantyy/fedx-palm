#!/usr/bin/env python3
"""Fast E1/E2 freeze + optimizer + weight-change diagnostic -- NOT a 40-round
(or even 2-round) sweep. Runs ONE client through the exact production
`train_client_round_dp()` function for a handful of real DP optimizer steps
(default 2) at a small imgsz/batch, purely to verify the mechanism, then
reports:

  - trainable/frozen parameter counts and which YOLO stages are frozen
  - optimizer membership audit (missing_trainable_params /
    unexpected_frozen_params -- both must be empty)
  - backbone / neck+head / DFL weight change, comparing the state_dict
    BEFORE the diagnostic step(s) to the one train_client_round_dp() returns

This is the SAME function used by scripts/07/08 (the real sweeps) and by
scripts/19_dp_smoke_test.py -- no separate/parallel implementation -- so a
PASS here is direct evidence about the production path, not an analogy to it.
It does NOT touch the held-out test split, does NOT run 40 rounds, and does
NOT write anything into runs/ or results/ used by real experiments (its own
output directory is a dedicated --out-dir, separate from any sweep's).

    python scripts/20_dp_freeze_audit.py --variant partial --device 0
    python scripts/20_dp_freeze_audit.py --variant full --device 0 --steps 4

Expected PASS (E2/partial):
    backbone_changed=False  neck_head_changed=True  DFL_changed=False
    missing_trainable=[]  frozen_in_optimizer=[]  n_frozen>0  n_trainable<total
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402

import fedxpalm  # noqa: E402,F401 (applies the GroupNorm-safe `fuse()` patch)
from fedxpalm.privacy.dp_sgd import train_client_round_dp  # noqa: E402
from fedxpalm.privacy.freeze_audit import diff_state_dicts  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["full", "partial"], required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--client", default=None, help="client id from the K=4 partition (default: first one found)")
    parser.add_argument("--steps", type=int, default=2, help="real DP optimizer steps to execute (>=1)")
    parser.add_argument("--imgsz", type=int, default=320, help="small for speed; this is a mechanism check")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--base-weights", default="models/base_groupnorm.pt")
    parser.add_argument("--out-dir", default="runs/_dp_freeze_audit_scratch")
    args = parser.parse_args()

    if args.steps < 1:
        print("FAIL: --steps must be >= 1"); return 1
    if not Path(args.base_weights).exists():
        print(f"FAIL: {args.base_weights} not found -- run scripts/04_prepare_base_model.py first"); return 1

    with open("configs/dataset.yaml") as f:
        ds_cfg = yaml.safe_load(f)
    with open("configs/fl_config.yaml") as f:
        fl_cfg = yaml.safe_load(f)
    with open("configs/dp_config.yaml") as f:
        dp_cfg = yaml.safe_load(f)

    splits_dir = Path(ds_cfg["output_dir"])
    manifest_path = splits_dir / "federated_partitions" / "manifest.json"
    if not manifest_path.exists():
        print(f"FAIL: {manifest_path} not found -- run scripts/03_partition_clients.py --k 4 first"); return 1
    import json
    with open(manifest_path) as f:
        manifest = json.load(f)["4"]
    client_id = args.client or sorted(manifest["sizes"], key=lambda c: int(c) if str(c).isdigit() else c)[0]
    data_yaml = str(splits_dir / "federated_partitions" / "k4_clients" / f"client{client_id}" / "data.yaml")
    if not Path(data_yaml).exists():
        print(f"FAIL: {data_yaml} not found"); return 1

    freeze_stages = dp_cfg["variants"][args.variant]["freeze_stages"]
    hyp = dict(fl_cfg["local_training"], imgsz=args.imgsz, batch_size=args.batch, workers=0)
    dp_hyp = dict(dp_cfg["dp_sgd"], sigma=args.sigma)
    dp_hyp.pop("noise_multiplier_values", None)

    # snapshot BEFORE state -- load the exact same checkpoint object the
    # production path loads, so the diff is apples-to-apples
    import torch
    ckpt_before = torch.load(args.base_weights, map_location="cpu", weights_only=False)
    sd_before = {k: v.detach().clone() for k, v in ckpt_before["model"].state_dict().items()}

    print(f"variant={args.variant}  client={client_id} (n={manifest['sizes'][client_id]})  "
          f"freeze_stages={freeze_stages}  steps={args.steps}  imgsz={args.imgsz}  batch={args.batch}")

    state_dict, info = train_client_round_dp(
        global_weights_path=args.base_weights, client_data_yaml=data_yaml,
        hyp=hyp, dp_hyp=dp_hyp, round_idx=0, client_id=f"diag_{client_id}",
        out_dir=args.out_dir, device=args.device, freeze_stages=freeze_stages,
        accountant_state=None, max_steps=args.steps,
    )

    backbone_changed, neck_head_changed, dfl_changed = diff_state_dicts(sd_before, state_dict)

    missing = info.get("missing_trainable_params") or []
    unexpected = info.get("unexpected_frozen_params") or []
    n_trainable = info.get("n_trainable_params")
    n_frozen = info.get("n_frozen_params")
    total = (n_trainable or 0) + (n_frozen or 0)

    print(f"\nsteps executed        : {info.get('steps_this_round')}")
    print(f"n_trainable_params     : {n_trainable:,}" if n_trainable is not None else "n_trainable_params     : ?")
    print(f"n_frozen_params        : {n_frozen:,}" if n_frozen is not None else "n_frozen_params        : ?")
    print(f"removed_frozen_from_optimizer_count: {info.get('removed_frozen_from_optimizer_count')}")
    print(f"missing_trainable_params (must be []): {missing}")
    print(f"unexpected_frozen_params (must be []): {unexpected}")
    print(f"nan_inf                : {info.get('nan_inf')}")
    print(f"\nbackbone_changed       : {backbone_changed}")
    print(f"neck_head_changed      : {neck_head_changed}")
    print(f"DFL_changed            : {dfl_changed}")

    failures = []
    if missing:
        failures.append(f"trainable params missing from optimizer: {missing[:5]}")
    if unexpected:
        failures.append(f"unexpected frozen params in optimizer: {unexpected[:5]}")
    if dfl_changed:
        failures.append("DFL fixed conv changed (must stay constant)")
    if info.get("nan_inf"):
        failures.append("NaN/Inf during diagnostic step(s)")
    if args.variant == "partial":
        if backbone_changed:
            failures.append("E2: backbone changed (must be frozen/identical)")
        if not neck_head_changed:
            failures.append("E2: neck+head did NOT change (should train)")
        if not (n_frozen and n_frozen > 0):
            failures.append("E2: n_frozen_params is not > 0")
        if total and n_trainable is not None and n_trainable >= total:
            failures.append("E2: n_trainable is not clearly less than total model parameters")
    elif args.variant == "full":
        if not (backbone_changed or neck_head_changed):
            failures.append("E1: no weights changed at all")

    print()
    for f in failures:
        print(f"FAIL: {f}")
    print("\nDP FREEZE AUDIT: PASSED" if not failures else "\nDP FREEZE AUDIT: FAILED")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
