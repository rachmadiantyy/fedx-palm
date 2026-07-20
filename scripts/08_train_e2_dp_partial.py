#!/usr/bin/env python3
"""E2 -- DP-SGD federated, partial (backbone frozen, only neck+head get
per-sample noise), swept over K x sigma. Mirrors thesis Table 4.5."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _dp_sweep_common import run_dp_sweep  # noqa: E402

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None, help="override configs/fl_config.yaml model.imgsz")
    parser.add_argument("--batch", type=int, default=None, help="override local_training.batch_size (lower if OOM)")
    parser.add_argument("--workers", type=int, default=0,
                        help="dataloader workers (default 0: dataloading in the main process -- the "
                             "validated-stable setting; workers>0 crashes on Windows because Opacus' "
                             "locally-defined collate fn can't be pickled into spawned workers)")
    args = parser.parse_args()

    run_dp_sweep("partial", device=args.device, k_override=args.k,
                 sigma_override=args.sigma, rounds_override=args.rounds,
                 imgsz_override=args.imgsz, batch_override=args.batch,
                 workers_override=args.workers)
