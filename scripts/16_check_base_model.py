#!/usr/bin/env python3
"""Validate models/base_groupnorm.pt before training on it:
  - zero BatchNorm layers remain (all converted)
  - GroupNorm layers present (count reported)
  - detection head nc == configs/dataset.yaml nc (6 classes)

Exit code non-zero on failure:
    python scripts/16_check_base_model.py && python scripts/05_...
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/base_groupnorm.pt")
    parser.add_argument("--config", default="configs/dataset.yaml")
    args = parser.parse_args()

    path = Path(args.weights)
    if not path.exists():
        print(f"FAIL: {path} not found -- run scripts/04_prepare_base_model.py first")
        return 1

    import torch
    import torch.nn as nn

    with open(args.config) as f:
        expected_nc = yaml.safe_load(f)["nc"]

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    n_bn = sum(1 for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm))
    n_gn = sum(1 for m in model.modules() if isinstance(m, nn.GroupNorm))
    n_params = sum(p.numel() for p in model.parameters())

    head = model.model[-1]  # Ultralytics Detect head
    nc = getattr(head, "nc", None)
    names = getattr(model, "names", None)

    print(f"checkpoint      : {path}")
    print(f"parameters      : {n_params:,}")
    print(f"BatchNorm layers: {n_bn}  (must be 0)")
    print(f"GroupNorm layers: {n_gn}  (must be > 0)")
    print(f"head nc         : {nc}  (must be {expected_nc})")
    print(f"names           : {names}")

    ok = True
    if n_bn != 0:
        print("FAIL: BatchNorm layers remain -- GroupNorm conversion incomplete")
        ok = False
    if n_gn == 0:
        print("FAIL: no GroupNorm layers found")
        ok = False
    if nc != expected_nc:
        print(f"FAIL: head nc={nc}, expected {expected_nc}")
        ok = False
    print("Base model VALID." if ok else "Base model INVALID -- regenerate with scripts/04.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
