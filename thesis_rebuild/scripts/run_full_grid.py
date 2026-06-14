"""
Orchestrator: run the FULL experimental grid for the rebuild thesis.

PHASE 1 only (1 seed, default 42, full grid):
  1. B1 centralized baseline (1 run, ~25 min on RTX 4080) — SKIPPED if
     thesis_rebuild/runs/b1_centralized/weights/best.pt already exists.
  2. B2 federated baseline x5 K (5 runs)
  3. E1-FL full DP-SGD x5 K x5 sigma (25 runs)
  4. E2-FL partial DP-SGD x5 K x5 sigma (25 runs)
  Total: 56 main runs.

PHASE 2 (run manually AFTER Phase 1, on a small interesting subset):
  Re-run promising (K, sigma) cells with 3 seeds for mean +/- std.
  Use the per-script --K --sigma --seed flags directly; no orchestrator
  needed because Phase 2 is small (~6-9 runs).

Behavior:
- Continue-on-error: if one (K, sigma) crashes, log the failure and
  continue with the next combo. Failures listed at the end. This avoids
  losing many hours of compute to a single transient crash.
- Each child phase writes its own best.pt + rounds.csv under
  thesis_rebuild/runs/ as it goes (incremental, crash-safe).
- Comment out / use --skip-bX / --skip-eX to resume after fixing a
  systemic issue (e.g. OOM at K=16).

Run all:
    python thesis_rebuild/scripts/run_full_grid.py

Run subset:
    python thesis_rebuild/scripts/run_full_grid.py --skip-b1 --skip-b2

Tip for overnight: run inside `tmux` or PowerShell `Start-Process` so a
disconnect doesn't kill the loop.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "thesis_rebuild" / "scripts"
B1_BEST = REPO_ROOT / "thesis_rebuild" / "runs" / "b1_centralized" / "weights" / "best.pt"


def run(label: str, cmd: list[str], failures: list[tuple[str, int]]) -> None:
    """Run a child phase; record (not raise) failures so the grid continues."""
    start = time.time()
    print(f"\n>>> [{label}] {' '.join(cmd)}")
    rc = subprocess.call(cmd)
    elapsed = (time.time() - start) / 60
    print(f"<<< [{label}] exit={rc} after {elapsed:.1f} min")
    if rc != 0:
        print(f"WARN: phase '{label}' returned exit={rc}; continuing with next phase.")
        failures.append((label, rc))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full experimental grid (Phase 1)")
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--weights", type=str, default="yolo11n.pt")
    p.add_argument("--epochs", type=int, default=50, help="B1 epochs")
    p.add_argument("--rounds", type=int, default=5, help="FL rounds")
    p.add_argument("--local_epochs", type=int, default=2)
    p.add_argument("--seed", type=int, default=42, help="Phase 1 seed")
    p.add_argument("--skip-b1", action="store_true")
    p.add_argument("--skip-b2", action="store_true")
    p.add_argument("--skip-e1", action="store_true")
    p.add_argument("--skip-e2", action="store_true")
    p.add_argument("--force-b1", action="store_true",
                   help="Re-run B1 even if best.pt already exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    py = sys.executable
    failures: list[tuple[str, int]] = []
    grid_start = time.time()

    # Phase 1.1: B1 centralized (auto-skip if already trained)
    if not args.skip_b1:
        if B1_BEST.exists() and not args.force_b1:
            print(f"[B1] skipping: {B1_BEST} already exists "
                  f"(use --force-b1 to retrain).")
        else:
            run("B1", [py, str(SCRIPTS / "train_b1_centralized.py"),
                       "--data", f"{args.data_root}/resplit/data.yaml",
                       "--epochs", str(args.epochs),
                       "--weights", args.weights,
                       "--name", "b1_centralized"], failures)

    # Phase 1.2: B2 federated x5 K
    if not args.skip_b2:
        run("B2", [py, str(SCRIPTS / "train_b2_fl.py"),
                   "--data_root", args.data_root,
                   "--weights", args.weights,
                   "--rounds", str(args.rounds),
                   "--local_epochs", str(args.local_epochs),
                   "--seed", str(args.seed),
                   "--all-K"], failures)

    # Phase 1.3: E1-FL full DP-SGD full grid (K x sigma)
    if not args.skip_e1:
        run("E1-FL", [py, str(SCRIPTS / "train_e1_fl_dp_sgd_full.py"),
                      "--data_root", args.data_root,
                      "--weights", args.weights,
                      "--rounds", str(args.rounds),
                      "--local_epochs", str(args.local_epochs),
                      "--seed", str(args.seed),
                      "--full-grid"], failures)

    # Phase 1.4: E2-FL partial DP-SGD full grid (K x sigma)
    if not args.skip_e2:
        run("E2-FL", [py, str(SCRIPTS / "train_e2_fl_dp_sgd_partial.py"),
                      "--data_root", args.data_root,
                      "--weights", args.weights,
                      "--rounds", str(args.rounds),
                      "--local_epochs", str(args.local_epochs),
                      "--seed", str(args.seed),
                      "--full-grid"], failures)

    total_h = (time.time() - grid_start) / 3600
    print("\n" + "=" * 70)
    print(f"PHASE 1 GRID COMPLETE in {total_h:.1f} h")
    print("=" * 70)
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for label, rc in failures:
            print(f"  - {label}: exit={rc}")
        print("Inspect logs and re-run failed phases with --skip-* flags.")
    else:
        print("All phases succeeded.")
    print("\nSummaries under thesis_rebuild/runs/:")
    print("  - b1_centralized/weights/best.pt")
    print(f"  - b2_fl_summary_seed{args.seed}.csv")
    print(f"  - e1_fl_full_grid_seed{args.seed}.csv")
    print(f"  - e2_fl_full_grid_seed{args.seed}.csv")
    print("\nNEXT:")
    print("  1. Inspect e1/e2 CSVs to pick Phase 2 (K, sigma) cells.")
    print("  2. Re-run those with seeds {7, 123} for mean +/- std:")
    print("     python ... --K 4 --sigma 1.0 --seed 7")
    print("     python ... --K 4 --sigma 1.0 --seed 123")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
