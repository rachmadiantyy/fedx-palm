#!/usr/bin/env python3
"""Step 4: build the shared starting checkpoint used by every experiment
block (B1/B2/E1/E2) -- pretrained COCO weights with every BatchNorm
converted to GroupNorm, so B1/B2 (no DP) and E1/E2 (DP-SGD) all start from
an architecturally identical model and are fairly comparable."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml  # noqa: E402

import fedxpalm  # noqa: E402 (applies the GroupNorm-safe `fuse()` patch)
from fedxpalm.models.groupnorm import prepare_model_for_dp  # noqa: E402

if __name__ == "__main__":
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    with open("configs/dataset.yaml") as f:
        ds_cfg = yaml.safe_load(f)
    with open("configs/fl_config.yaml") as f:
        fl_cfg = yaml.safe_load(f)
    with open("configs/dp_config.yaml") as f:
        dp_cfg = yaml.safe_load(f)

    arch = fl_cfg["model"]["arch"]
    yolo = YOLO(arch)  # auto-downloads the pretrained COCO checkpoint if missing
    # rebuild the head for our nc classes, keep the pretrained backbone weights
    pretrained_sd = yolo.model.state_dict()
    yolo.model = DetectionModel(yolo.model.yaml, nc=ds_cfg["nc"])
    compatible = {k: v for k, v in pretrained_sd.items()
                  if k in yolo.model.state_dict() and yolo.model.state_dict()[k].shape == v.shape}
    yolo.model.load_state_dict(compatible, strict=False)
    print(f"Loaded {len(compatible)}/{len(yolo.model.state_dict())} pretrained tensors "
          f"(head layers differ due to nc={ds_cfg['nc']} vs COCO's 80 and are left at init).")

    prepare_model_for_dp(yolo, max_groups=dp_cfg["groupnorm"]["max_groups"])

    out_path = Path("models/base_groupnorm.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    yolo.save(str(out_path))
    print(f"Saved shared base checkpoint to {out_path}")
