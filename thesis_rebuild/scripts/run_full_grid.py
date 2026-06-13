"""
Orchestrator: run the FULL experimental grid for the rebuild thesis.

Sequence (estimated 200+ hours on RTX 4080):
  1. B1 centralized baseline (1 run, ~2h)
  2. B2 federated baseline x4 K (4 runs, ~12h)
  3. E1-FL full DP-SGD x4 K x5 sigma (20 runs, ~100h)
  4. E2-FL partial DP-SGD x4 K x5 sigma (20 runs, ~80h)
  Total: 45 main runs, ~200h

  R1 robustness (3 seeds on selected configs) is run separately AFTER
  the main grid lands, since seed-pinned reruns of a few key cells are
  cheap relative to the grid.

This script is intentionally a thin loop so you can comment out any
phase (-skip-bX / -skip-eX) and resume after failure. Each child run
writes its own best.pt + rounds.csv under thesis_rebuild/runs/.

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


def run(cmd: list[str]) -> None:
    start = time.time()
    print(f"\n>>> {' '.join(cmd)}")
    rc = subprocess.call(cmd)
    elapsed = (time.time() - start) / 60
    print(f"<<< exit={rc} after {elapsed:.1f} min")
    if rc != 0:
        print("ERROR: phase failed, see log above. Stopping orchestrator.")
        sys.exit(rc)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full experimental grid")
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--weights", type=str, default="yolo11n.pt")
    p.add_argument("--epochs", type=int, default=50, help="B1 epochs")
    p.add_argument("--rounds", type=int, default=5, help="FL rounds")
    p.add_argument("--local_epochs", type=int, default=2)
    p.add_argument("--skip-b1", action="store_true")
    p.add_argument("--skip-b2", action="store_true")
    p.add_argument("--skip-e1", action="store_true")
    p.add_argument("--skip-e2", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    py = sys.executable

    # Phase 1: B1 centralized
    if not args.skip_b1:
        run([py, str(SCRIPTS / "train_b1_centralized.py"),
             "--data", f"{args.data_root}/resplit/data.yaml",
             "--epochs", str(args.epochs),
             "--weights", args.weights,
             "--name", "b1_centralized"])

    # Phase 2: B2 federated x4 K
    if not args.skip_b2:
        run([py, str(SCRIPTS / "train_b2_fl.py"),
             "--data_root", args.data_root,
             "--weights", args.weights,
             "--rounds", str(args.rounds),
             "--local_epochs", str(args.local_epochs),
             "--all-K"])

    # Phase 3: E1-FL full DP-SGD full grid
    if not args.skip_e1:
        run([py, str(SCRIPTS / "train_e1_fl_dp_sgd_full.py"),
             "--data_root", args.data_root,
             "--weights", args.weights,
             "--rounds", str(args.rounds),
             "--local_epochs", str(args.local_epochs),
             "--full-grid"])

    # Phase 4: E2-FL partial DP-SGD full grid
    if not args.skip_e2:
        run([py, str(SCRIPTS / "train_e2_fl_dp_sgd_partial.py"),
             "--data_root", args.data_root,
             "--weights", args.weights,
             "--rounds", str(args.rounds),
             "--local_epochs", str(args.local_epochs),
             "--full-grid"])

    print("\n" + "=" * 70)
    print("FULL GRID COMPLETE")
    print("=" * 70)
    print(f"Summaries under thesis_rebuild/runs/:")
    print(f"  - b1_centralized/results.csv")
    print(f"  - b2_fl_summary.csv")
    print(f"  - e1_fl_full_grid.csv")
    print(f"  - e2_fl_full_grid.csv")
    print("\nNEXT: aggregate_results.py (Day 4) -> plots for Bab 4")


if __name__ == "__main__":
    main()
