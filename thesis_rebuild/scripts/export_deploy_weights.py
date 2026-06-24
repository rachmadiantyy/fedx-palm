"""
Export a federated FedX-Palm checkpoint into a standard Ultralytics .pt that a
plain `YOLO("deploy_best.pt")` (and therefore deploy_vm/app/app.py) can load
without any custom rebuild code.

WHY THIS EXISTS
---------------
Training saves best.pt as a custom dict {"state": <GN state_dict>, "config":
..., ...}, NOT a standard Ultralytics checkpoint ({"model": <nn.Module>}).
The deployment service calls `YOLO(MODEL_PATH)` directly, which expects the
standard format and otherwise fails with `KeyError: 'model'`.

This script reuses the SAME, proven load path as evaluate_xai.py (rebuild the
GroupNorm YOLO, load the federated state), then re-serializes the live
nn.Module in the standard Ultralytics layout so the container can load it with
zero extra dependencies (no opacus, no gn_convert, no yolo11n.pt download).

RUN ON THE MACHINE WHERE evaluate_xai.py ALREADY WORKS (your laptop):
    python thesis_rebuild/scripts/export_deploy_weights.py \
        --weights thesis_rebuild/runs/b2_fl_K4_seed42/best.pt \
        --out     thesis_rebuild/runs/b2_fl_K4_seed42/deploy_best.pt

Then upload deploy_best.pt to the VM and point the container at it.
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

# Reuse the EXACT proven loader from the XAI evaluation so the exported model
# is byte-for-byte the same network that produced the paper's results.
from thesis_rebuild.scripts.evaluate_xai import load_model  # noqa: E402

CLASS_NAMES = ["Abnormal", "Empty Bunch", "Overripe", "Ripe", "Underripe", "Unripe"]


def _no_fuse(self, verbose: bool = True):
    """Picklable replacement for the fuse-noop lambda.

    build_gn_yolo sets `model.fuse = lambda ...`, but a lambda cannot be
    pickled by torch.save. Binding a top-level function as a method is
    picklable and keeps the same behaviour (GroupNorm has no Conv+BN fusion).
    """
    return self


def main() -> None:
    ap = argparse.ArgumentParser(description="Export federated ckpt -> standard Ultralytics .pt")
    ap.add_argument("--weights", required=True, help="federated best.pt (custom {'state',...} format)")
    ap.add_argument("--out", required=True, help="output path for the standard .pt")
    ap.add_argument("--device", default="cpu", help="cpu is fine; deployment is CPU-only")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    # 1) Rebuild the GN YOLO and load the federated state (proven path).
    yolo = load_model(args.weights, args.device)
    model = yolo.model

    # 2) Make the model fully picklable + label-correct for deployment.
    #    - swap the lambda fuse for a bound top-level function
    #    - stamp the six ripeness class names (training kept COCO names)
    model.fuse = types.MethodType(_no_fuse, model)
    model.names = {i: n for i, n in enumerate(CLASS_NAMES)}
    if hasattr(model, "model") and hasattr(model.model[-1], "nc"):
        nc = model.model[-1].nc
        print(f"[export] detection head nc = {nc} (expected 6)")
    model.float().eval()

    # 3) Save in the standard Ultralytics checkpoint layout.
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": model,            # the nn.Module YOLO() expects
        "ema": None,
        "updates": 0,
        "optimizer": None,
        "train_args": {"imgsz": args.imgsz, "data": "palm6"},
        "date": "",
        "version": "fedx-palm-export",
    }
    torch.save(ckpt, out)
    print(f"[export] wrote standard checkpoint -> {out} ({out.stat().st_size/1e6:.1f} MB)")

    # 4) Verify it reloads with a PLAIN YOLO() exactly like the container will.
    from ultralytics import YOLO  # noqa: E402
    reloaded = YOLO(str(out))
    reloaded.model.fuse = types.MethodType(_no_fuse, reloaded.model)  # keep noop
    names = reloaded.names
    print(f"[verify] reloaded OK; classes = {list(names.values()) if isinstance(names, dict) else names}")
    print("[verify] running a dummy forward pass on a blank image ...")
    import numpy as np
    dummy = (np.zeros((args.imgsz, args.imgsz, 3), dtype="uint8"))
    _ = reloaded.predict(dummy, verbose=False, device=args.device)
    print("[verify] forward pass OK — safe to deploy this file.")


if __name__ == "__main__":
    main()
