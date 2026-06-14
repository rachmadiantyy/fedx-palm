"""
Orchestrator: run the FULL experimental grid, ONE SUBPROCESS PER CELL.

Why per-cell subprocesses: running all (K, sigma) cells inside a single
Python process leaked CUDA/Opacus state and deadlocked on the 2nd cell
(observed: K=2 sigma=1.0 hung with GPU idle). Spawning a fresh process per
cell guarantees memory is released between cells, and a per-cell timeout
turns any residual hang into a skip-and-continue instead of a dead grid.

PHASE 1 (1 seed, default 42):
  1. B1 centralized            (skipped if best.pt exists)
  2. B2 federated, per K       (5 cells)
  3. E1-FL full DP-SGD, per (K, sigma)     (25 cells)
  4. E2-FL partial DP-SGD, per (K, sigma)  (25 cells)

Resumable: each cell's wrapper skips itself if its run dir already has a
complete rounds.csv + best.pt (see fl_dp_loop.already_complete). Re-launch
freely after a crash/timeout; finished cells are skipped instantly.

Aggregate AFTER with: python thesis_rebuild/scripts/aggregate_results.py
(it reads run dirs directly, so per-cell summary CSVs are not needed.)

Run all:        python thesis_rebuild/scripts/run_full_grid.py
Resume E only:  python thesis_rebuild/scripts/run_full_grid.py --skip-b1 --skip-b2
"""
import argparse
import itertools
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "thesis_rebuild" / "scripts"
B1_BEST = REPO_ROOT / "thesis_rebuild" / "runs" / "b1_centralized" / "weights" / "best.pt"

K_GRID = [2, 4, 8, 12, 16]
SIGMA_GRID = [0.5, 1.0, 1.5, 2.0, 3.0]


def cell_timeout(K: int) -> int:
    """Generous per-cell wall-clock budget; scales with client count.
    K=2 -> 40 min, K=16 -> ~110 min. A true hang is indefinite, so any of
    these still catches it."""
    return 1800 + K * 300


def run(label: str, cmd: list[str], failures: list, timeout: int) -> None:
    """Run one cell in a fresh subprocess; kill on timeout, never raise."""
    start = time.time()
    print(f"\n>>> [{label}] (timeout {timeout//60}m) {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd)
    try:
        rc = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        print(f"!!! [{label}] TIMEOUT after {timeout//60}m — killed, continuing.")
        failures.append((label, "timeout"))
        return
    elapsed = (time.time() - start) / 60
    print(f"<<< [{label}] exit={rc} after {elapsed:.1f} min")
    if rc != 0:
        print(f"WARN: [{label}] exit={rc}; continuing.")
        failures.append((label, rc))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full experimental grid (per-cell)")
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--weights", type=str, default="yolo11n.pt")
    p.add_argument("--epochs", type=int, default=50, help="B1 epochs")
    p.add_argument("--rounds", type=int, default=5, help="FL rounds")
    p.add_argument("--local_epochs", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-b1", action="store_true")
    p.add_argument("--skip-b2", action="store_true")
    p.add_argument("--skip-e1", action="store_true")
    p.add_argument("--skip-e2", action="store_true")
    p.add_argument("--force-b1", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    py = sys.executable
    failures: list = []
    grid_start = time.time()

    common = ["--data_root", args.data_root, "--weights", args.weights,
              "--rounds", str(args.rounds),
              "--local_epochs", str(args.local_epochs),
              "--seed", str(args.seed)]

    # Phase 1.1: B1 centralized (single process, auto-skip if done)
    if not args.skip_b1:
        if B1_BEST.exists() and not args.force_b1:
            print(f"[B1] skip: {B1_BEST} exists (use --force-b1 to retrain).")
        else:
            run("B1", [py, str(SCRIPTS / "train_b1_centralized.py"),
                       "--data", f"{args.data_root}/resplit/data.yaml",
                       "--epochs", str(args.epochs),
                       "--weights", args.weights,
                       "--name", "b1_centralized"], failures, timeout=4 * 3600)

    # Phase 1.2: B2 federated, one subprocess per K
    if not args.skip_b2:
        for K in K_GRID:
            run(f"B2 K={K}",
                [py, str(SCRIPTS / "train_b2_fl.py"), "--K", str(K)] + common,
                failures, timeout=cell_timeout(K))

    # Phase 1.3: E1-FL full DP-SGD, one subprocess per (K, sigma)
    if not args.skip_e1:
        for K, sigma in itertools.product(K_GRID, SIGMA_GRID):
            run(f"E1 K={K} s={sigma}",
                [py, str(SCRIPTS / "train_e1_fl_dp_sgd_full.py"),
                 "--K", str(K), "--sigma", str(sigma)] + common,
                failures, timeout=cell_timeout(K))

    # Phase 1.4: E2-FL partial DP-SGD, one subprocess per (K, sigma)
    if not args.skip_e2:
        for K, sigma in itertools.product(K_GRID, SIGMA_GRID):
            run(f"E2 K={K} s={sigma}",
                [py, str(SCRIPTS / "train_e2_fl_dp_sgd_partial.py"),
                 "--K", str(K), "--sigma", str(sigma)] + common,
                failures, timeout=cell_timeout(K))

    total_h = (time.time() - grid_start) / 3600
    print("\n" + "=" * 70)
    print(f"PHASE 1 GRID COMPLETE in {total_h:.1f} h")
    print("=" * 70)
    if failures:
        print(f"FAILURES/TIMEOUTS ({len(failures)}):")
        for label, rc in failures:
            print(f"  - {label}: {rc}")
        print("Re-launch the same command to retry only the unfinished cells.")
    else:
        print("All cells succeeded.")
    print("\nNEXT: python thesis_rebuild/scripts/aggregate_results.py")
    print("      python thesis_rebuild/scripts/evaluate_xai.py --weights <best.pt>")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
