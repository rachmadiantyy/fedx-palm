"""
Custom YOLOv11 training loop with Opacus DP-SGD.

Why a custom loop:
- Ultralytics' BaseTrainer creates its own optimizer + dataloader
  inside `.train()`, leaving no clean hook to wrap them with Opacus
  PrivacyEngine BEFORE the loop starts.
- It also uses AMP autocast and DDP plumbing that interfere with
  Opacus per-sample gradient hooks.

What we reuse FROM Ultralytics:
- v8DetectionLoss (CIoU + cls + DFL)
- build_yolo_dataset (data pipeline with mosaic / augmentation)
- DetectionValidator (mAP computation)
- YOLO model architecture (with our BN→GN swap)

What is OURS:
- Outer epoch loop
- Per-batch forward/backward/step
- Opacus PrivacyEngine wrapping
- Epsilon tracking + best.pt checkpointing
- Optional backbone freezing for E2 partial DP

Used by:
- train_e1_dp_sgd_full.py   (all params trainable)
- train_e2_dp_sgd_partial.py (backbone frozen, head only)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG, colorstr
from ultralytics.utils.loss import v8DetectionLoss

# allow `from thesis_rebuild.scripts.utils.gn_convert import ...`
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from thesis_rebuild.scripts.utils.gn_convert import (  # noqa: E402
    count_bn_layers,
    replace_bn_with_gn,
)


@dataclass
class DPSGDConfig:
    """Hyperparameters for one DP-SGD training run."""

    # Data + model
    data_yaml: str
    weights: str = "yolo11n.pt"
    imgsz: int = 640
    num_classes: int = 6

    # Training
    epochs: int = 50
    batch_size: int = 16
    lr0: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005

    # DP-SGD
    noise_multiplier: float = 1.0  # σ
    max_grad_norm: float = 1.0     # C (per-sample clip)
    target_delta: float = 1e-5

    # E2 partial DP: freeze backbone
    freeze_backbone: bool = False

    # Bookkeeping
    seed: int = 42
    device: str = "cuda"
    project: str = "thesis_rebuild/runs"
    name: str = "dp_sgd"
    log_every: int = 10

    # GN
    num_groups: int = 8


def build_dp_yolo_model(
    weights: str, num_groups: int, freeze_backbone: bool
) -> tuple[nn.Module, int]:
    """Load YOLO, swap BN→GN, optionally freeze backbone.

    Returns (raw_nn_module_in_train_mode, num_trainable_params).
    """
    yolo = YOLO(weights)
    model = yolo.model

    # 1) BN → GN (Opacus requirement)
    n_bn_before, _ = count_bn_layers(model)
    n_converted = replace_bn_with_gn(model, num_groups=num_groups)
    n_bn_after, n_gn_after = count_bn_layers(model)
    assert n_bn_after == 0, f"BN remaining: {n_bn_after}"
    print(f"  BN→GN: converted {n_converted}, GN total={n_gn_after}")

    # 2) Opacus must see train mode
    model.train()

    # 3) Validate (and auto-fix any remaining incompatibilities)
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        print(f"  ModuleValidator: {len(errors)} issues — running fix()")
        model = ModuleValidator.fix(model)

    # 4) Optional backbone freeze for E2 partial DP
    if freeze_backbone:
        for name, p in model.named_parameters():
            # YOLOv11 backbone is the first 10 layers (model.0 - model.9)
            # Head + neck start at model.10
            stage = name.split(".")[1] if name.startswith("model.") else ""
            if stage.isdigit() and int(stage) < 10:
                p.requires_grad = False

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    pct = 100 * n_trainable / n_total
    print(f"  Trainable params: {n_trainable:,} / {n_total:,} ({pct:.1f}%)")

    return model, n_trainable


def build_loaders(
    cfg: DPSGDConfig,
) -> tuple[DataLoader, dict]:
    """Build train DataLoader and return val dataset info (for later)."""
    data_info = check_det_dataset(cfg.data_yaml)
    args = get_cfg(DEFAULT_CFG)
    args.imgsz = cfg.imgsz
    args.batch = cfg.batch_size
    args.workers = 4
    args.cache = False
    args.rect = False

    train_dataset = build_yolo_dataset(
        args, data_info["train"], cfg.batch_size, data_info,
        mode="train", rect=False, stride=32,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        collate_fn=getattr(train_dataset, "collate_fn", None), drop_last=True,
    )
    return train_loader, data_info


def train_one_run(cfg: DPSGDConfig) -> dict:
    """Run a single DP-SGD training to completion.

    Returns a dict with final mAP, ε, σ, etc — caller appends to a
    CSV for the σ-sweep table.
    """
    torch.manual_seed(cfg.seed)

    out_dir = Path(cfg.project) / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)

    print()
    print(colorstr("bold", "=") * 70)
    print(f"DP-SGD Run: σ={cfg.noise_multiplier}, C={cfg.max_grad_norm}, "
          f"freeze_backbone={cfg.freeze_backbone}")
    print(colorstr("bold", "=") * 70)

    # ----- model -----
    model, n_trainable = build_dp_yolo_model(
        cfg.weights, cfg.num_groups, cfg.freeze_backbone,
    )
    device = torch.device(cfg.device)
    model = model.to(device)

    # YOLO loss needs the model with `.args` attribute. The trainer
    # normally sets this; we mimic it.
    model.args = get_cfg(DEFAULT_CFG)
    model.args.box = 7.5
    model.args.cls = 0.5
    model.args.dfl = 1.5
    criterion = v8DetectionLoss(model)

    # ----- data -----
    train_loader, data_info = build_loaders(cfg)

    # ----- optimizer -----
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(params, lr=cfg.lr0, momentum=cfg.momentum,
                    weight_decay=cfg.weight_decay)

    # ----- Opacus wrap -----
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model, optimizer=optimizer, data_loader=train_loader,
        noise_multiplier=cfg.noise_multiplier,
        max_grad_norm=cfg.max_grad_norm,
    )
    print(f"  Opacus accountant: {privacy_engine.accountant.__class__.__name__}")

    # ----- training loop -----
    best_map = 0.0
    history = []
    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            imgs = batch["img"].to(device, non_blocking=True).float() / 255.0
            batch_gpu = {
                k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
            optimizer.zero_grad(set_to_none=True)
            preds = model(imgs)
            loss_components, _ = criterion(preds, batch_gpu)
            # v8DetectionLoss returns [box, cls, dfl] in ultralytics 8.4.x;
            # .sum() reduces to scalar (idempotent if already scalar).
            loss = loss_components.sum()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach())

            if step % cfg.log_every == 0:
                print(f"  ep{epoch:02d} step{step:04d} loss={loss.item():.4f}")

        eps = privacy_engine.get_epsilon(delta=cfg.target_delta)
        avg_loss = running_loss / max(1, len(train_loader))
        print(f"[epoch {epoch:02d}] avg_loss={avg_loss:.4f}  "
              f"ε={eps:.3f} (δ={cfg.target_delta})")

        # TODO: validation pass with DetectionValidator each epoch
        # For Day 2 we save final checkpoint; Day 3 will add per-epoch val.
        history.append({"epoch": epoch, "loss": avg_loss, "epsilon": eps})

    # Save final checkpoint (Day 3: replace with best-on-val tracking)
    ckpt_path = out_dir / "final.pt"
    # unwrap GradSampleModule for clean save
    raw = model._module if hasattr(model, "_module") else model
    torch.save({"model_state": raw.state_dict(), "config": cfg.__dict__,
                "final_epsilon": eps}, ckpt_path)
    print(f"\nSaved {ckpt_path}")

    return {
        "sigma": cfg.noise_multiplier,
        "C": cfg.max_grad_norm,
        "freeze_backbone": cfg.freeze_backbone,
        "n_trainable": n_trainable,
        "final_loss": avg_loss,
        "final_epsilon": eps,
        "best_mAP50": best_map,  # 0 for Day 2 (no val); Day 3 fills in
        "ckpt": str(ckpt_path),
        "history": history,
    }
