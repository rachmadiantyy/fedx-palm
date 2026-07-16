#!/usr/bin/env python3
"""Cheap seed-effectiveness smoke test for B2 -- run BEFORE any 40-round
multi-seed experiment.

Runs three 1-round K=4 federated runs (partition seed fixed at 42):
    A : --seed 42    --tag seedsmoke_a
    B : --seed 123   --tag seedsmoke_b
    A2: --seed 42    --tag seedsmoke_a2   (repeat of A)

Then checks, on client0's local checkpoint and the aggregated global model:
  1. recorded train_args.seed differs between A and B (effective seed logged)
  2. A vs B weights are NOT tensor-identical (total |dw| > 0)
  3. A vs A2 weights ARE tensor-identical (total |dw| == 0, reproducible)

The test set is never touched (scripts/06 without --eval-test). Exit code
is non-zero on any failure.

    python scripts/18_seed_smoke_test.py --device 0
    python scripts/18_seed_smoke_test.py --device 0 --skip-runs   # compare only
"""
import argparse
import subprocess
import sys
from pathlib import Path

RUNS = [
    ("a", 42),
    ("b", 123),
    ("a2", 42),
]


def total_abs_diff(path_a: Path, path_b: Path) -> tuple[float, int, int]:
    """(sum |wa - wb|, recorded seed A, recorded seed B) over the full state_dict."""
    import torch

    ck_a = torch.load(path_a, map_location="cpu", weights_only=False)
    ck_b = torch.load(path_b, map_location="cpu", weights_only=False)
    sd_a = ck_a["model"].state_dict()
    sd_b = ck_b["model"].state_dict()
    assert sd_a.keys() == sd_b.keys(), "state_dict keys differ -- not comparable"
    diff = sum(float((sd_a[k].float() - sd_b[k].float()).abs().sum()) for k in sd_a)
    seed_of = lambda ck: (ck.get("train_args") or {}).get("seed") if isinstance(ck.get("train_args"), dict) \
        else getattr(ck.get("train_args"), "seed", None)
    return diff, seed_of(ck_a), seed_of(ck_b)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--partition-seed", type=int, default=42)
    parser.add_argument("--skip-runs", action="store_true", help="only compare existing outputs")
    args = parser.parse_args()

    if not args.skip_runs:
        for tag_suffix, seed in RUNS:
            cmd = [sys.executable, "scripts/06_train_b2_federated.py",
                   "--device", args.device, "--k", str(args.k),
                   "--seed", str(seed), "--partition-seed", str(args.partition_seed),
                   "--rounds", "1", "--warmup-epochs", "0",
                   "--tag", f"seedsmoke_{tag_suffix}"]
            print(f"\n=== run {tag_suffix} (seed {seed}) ===\n  {' '.join(cmd)}")
            r = subprocess.run(cmd)
            if r.returncode != 0:
                print(f"FAIL: training run '{tag_suffix}' exited {r.returncode}")
                return 1

    def run_dir(tag_suffix, seed):
        return Path(f"runs/b2_federated/k{args.k}_seed{seed}_seedsmoke_{tag_suffix}")

    dir_a, dir_b, dir_a2 = (run_dir(t, s) for t, s in RUNS)
    failures = []

    # compare client0 local checkpoints and the aggregated global round-0 model
    for label, rel in (("client0 local", "r0_client0/weights/last.pt"),
                       ("global aggregated", "global_round_0.pt")):
        pa, pb, pa2 = dir_a / rel, dir_b / rel, dir_a2 / rel
        missing = [p for p in (pa, pb, pa2) if not p.exists()]
        if missing:
            failures.append(f"{label}: missing checkpoint(s): {missing}")
            continue

        d_ab, seed_a, seed_b = total_abs_diff(pa, pb)
        d_aa, _, seed_a2 = total_abs_diff(pa, pa2)
        print(f"\n[{label}]")
        print(f"  recorded seed A={seed_a}  B={seed_b}  A2={seed_a2}")
        print(f"  total |dw| seed42 vs seed123 : {d_ab:.6f}  (must be > 0)")
        print(f"  total |dw| seed42 vs seed42  : {d_aa:.6f}  (must be == 0)")

        if label == "client0 local":
            if seed_a is None or seed_b is None:
                failures.append(f"{label}: train_args.seed not recorded in checkpoint")
            else:
                if seed_a == seed_b:
                    failures.append(f"{label}: effective seeds identical across experiment seeds ({seed_a})")
                if seed_a != seed_a2:
                    failures.append(f"{label}: repeat run recorded different seed ({seed_a} vs {seed_a2})")
        if d_ab <= 0.0:
            failures.append(f"{label}: seed 42 vs 123 tensor-identical -- seed still ineffective")
        if d_aa != 0.0:
            failures.append(f"{label}: seed 42 repeat NOT reproducible (|dw|={d_aa:.6f})")

    print()
    for f in failures:
        print(f"FAIL: {f}")
    if failures:
        print("\nSEED SMOKE TEST: FAILED -- do not run multi-seed experiments yet.")
        return 1
    print("SEED SMOKE TEST: PASSED -- different seeds diverge, same seed reproduces.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
