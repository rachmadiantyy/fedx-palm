#!/usr/bin/env python3
"""Re-evaluate B1's locked best.pt on the held-out TEST split with
Ultralytics' own plots enabled, so Figure 4 (Bab 4.1 / manuscript Figure 4)
can be the real confusion_matrix.png / PR_curve.png / F1_curve.png that
Ultralytics generates from this exact evaluation -- not a hand-drawn
per-class AP50 bar chart.

Why a separate run instead of reusing scripts/05's training-time plots:
scripts/05 passes plots=True to the *trainer*, which only ever evaluates
against the VALIDATION split each epoch. Table 4.1's numbers (mAP50=0.8820
etc.) come from the held-out TEST split, evaluated once after the
checkpoint was locked -- so the plots need to come from that same TEST-split
evaluation to actually match the numbers being reported, not from a
different split.

Read-only: does not train, does not touch the held-out test split's role as
a report-once split beyond the one evaluation Bab 3.8 already calls for.

    python scripts/42_generate_b1_test_plots.py \\
        --weights runs/b1_centralized/train/weights/best.pt \\
        --data data/splits_v2/data.yaml

Saves confusion_matrix.png, confusion_matrix_normalized.png, PR_curve.png,
P_curve.png, R_curve.png, F1_curve.png, val_batch*_labels.jpg,
val_batch*_pred.jpg to runs/b1_test_plots/held_out_test/. Prints the exact
save directory and re-confirms the metrics still match Table 4.1 (they
should be bit-for-bit the same numbers -- this run doesn't retrain
anything, it only asks for plots on top of the same evaluation).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import fedxpalm  # noqa: E402,F401 (applies the GroupNorm-safe `fuse()` patch)
from fedxpalm.eval.detection_metrics import evaluate_detector  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="B1's locked best.pt (Subbab 3.6)")
    parser.add_argument("--data", required=True, help="path to data.yaml (Subbab 3.3)")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out-dir", default="runs/b1_test_plots")
    parser.add_argument("--name", default="held_out_test")
    args = parser.parse_args()

    metrics = evaluate_detector(args.weights, args.data, split="test", imgsz=args.imgsz, device=args.device,
                                plots=True, project=args.out_dir, name=args.name)

    print(f"mAP@0.5={metrics['map50']:.4f}  mAP@0.5:0.95={metrics['map50_95']:.4f}  "
          f"Precision={metrics['precision']:.4f}  Recall={metrics['recall']:.4f}")
    print("(cross-check these four numbers against Table 4.1 / results/b1_centralized.json -- "
          "should match exactly, since this is the same evaluation with plots turned on)")
    print(f"\nPlots saved to: {args.out_dir}/{args.name}/")
    print("Use confusion_matrix.png and/or PR_curve.png as Figure 4 (Bab 4.1) instead of a "
          "hand-drawn per-class AP50 bar chart -- both come straight out of this exact "
          "held-out-test evaluation of B1's model.")
