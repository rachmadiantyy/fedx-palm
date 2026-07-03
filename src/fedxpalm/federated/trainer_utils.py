"""Shared helper for building an Ultralytics `DetectionTrainer` from an
already-loaded (GroupNorm-converted) checkpoint, instead of a weights path.

Why this exists: `YOLO(path).train(...)` -- the high-level API -- always
rebuilds the model from `self.model.yaml` and transplants weights
(`Model.train()`'s `self.trainer.model = self.trainer.get_model(weights=
self.model, cfg=self.model.yaml)` line, *before* the trainer's own setup
even runs). `parse_model()` hardcodes fresh `nn.BatchNorm2d` for every Conv
block regardless of what was actually saved, so that rebuild silently
reverts every GroupNorm layer back to BatchNorm and drops the
running_mean/var (only `weight`/`bias` transplant by matching key+shape) --
found when `fedavg()`'s aggregated (still GroupNorm-shaped) state_dict no
longer matched a post-round-1 client's (freshly BatchNorm-shaped) one.

Fix: build the `DetectionTrainer` directly and assign the pre-loaded model
to `trainer.model` *before* calling `trainer.train()` -- `setup_model()`
skips rebuilding when `self.model` is already an `nn.Module`.
"""
from __future__ import annotations

import torch
from ultralytics.models.yolo.detect.train import DetectionTrainer


def build_trainer_from_checkpoint(global_weights_path: str, overrides: dict) -> DetectionTrainer:
    """`overrides['model']` should still be set to `global_weights_path` (kept
    for Ultralytics' own bookkeeping/logging), but the actual module used for
    training is the one pre-loaded here, not rebuilt from its `.yaml`.
    """
    ckpt = torch.load(global_weights_path, map_location="cpu", weights_only=False)
    model = ckpt["model"].float()
    trainer = DetectionTrainer(overrides=overrides)
    trainer.model = model
    return trainer
