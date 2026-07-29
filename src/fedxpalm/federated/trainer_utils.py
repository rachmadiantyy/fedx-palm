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

Second, unrelated fix bundled here because every training entrypoint in
this repo goes through this one function: if `overrides['project']` is a
*relative* path, Ultralytics' `get_save_dir()` re-roots it under its own
default runs directory instead of resolving it against the current working
directory -- `project="runs/b1_centralized"` silently becomes
`<runs_dir>/detect/runs/b1_centralized/<name>` (confirmed both in this
sandbox and on the user's real Windows run: weights ended up nested two
levels deeper than every script in this repo assumes, e.g.
`runs/detect/runs/b1_centralized/train/weights/best.pt` instead of
`runs/b1_centralized/train/weights/best.pt`). Resolving `project` to an
absolute path here, once, avoids that for every caller (client.py,
dp_sgd.py, scripts/05) without needing the same fix repeated everywhere.
"""
from __future__ import annotations

from pathlib import Path

import torch
from ultralytics.models.yolo.detect.train import DetectionTrainer


class SeededDetectionTrainer(DetectionTrainer):
    """DetectionTrainer whose dataloader actually honors ``args.seed``.

    Ultralytics 8.4.x seeds the global RNGs from ``args.seed`` (``init_seeds``
    in ``BaseTrainer.__init__``), but ``ultralytics/data/build.py`` hands the
    DataLoader a ``torch.Generator`` seeded with the CONSTANT
    ``6148914691236517205 + RANK``, and ``get_dataloader`` never forwards the
    experiment seed. The shuffle order comes from that generator, and so do
    the dataloader workers' base seeds (drawn from it at iterator creation),
    which in turn drive every augmentation RNG inside the workers. With the
    same init weights, partition, optimizer, and deterministic mode, two runs
    that differ ONLY in ``args.seed`` therefore produce tensor-identical
    weights -- observed for real on B2 K=4 seed 42 vs seed 123
    (total |dw| = 0.0 across the full state_dict).

    Fix: re-seed the loader's generator from ``args.seed`` right after the
    loader is built and before its first iterator is created, so shuffle
    order AND worker seeds derive from the experiment seed. site-packages
    stays untouched.
    """

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        loader = super().get_dataloader(dataset_path, batch_size, rank, mode)
        generator = getattr(loader, "generator", None)
        if generator is not None:
            # +1 offsets the val loader so it never shares a stream with train
            generator.manual_seed(int(getattr(self.args, "seed", 0)) + (0 if mode == "train" else 1))
        else:
            print("[SeededDetectionTrainer] WARNING: dataloader has no generator "
                  "attribute -- shuffle/worker seeding may not follow args.seed "
                  "on this Ultralytics version; verify with scripts/18_seed_smoke_test.py")
        return loader


def build_trainer_from_checkpoint(global_weights_path: str, overrides: dict) -> DetectionTrainer:
    """`overrides['model']` should still be set to `global_weights_path` (kept
    for Ultralytics' own bookkeeping/logging), but the actual module used for
    training is the one pre-loaded here, not rebuilt from its `.yaml`.
    """
    if "project" in overrides:
        overrides = {**overrides, "project": str(Path(overrides["project"]).resolve())}

    ckpt = torch.load(global_weights_path, map_location="cpu", weights_only=False)
    model = ckpt["model"].float()
    trainer = SeededDetectionTrainer(overrides=overrides)
    trainer.model = model
    return trainer
