"""One client's local fine-tuning for one federated round (no DP -- used for B2).

See trainer_utils.build_trainer_from_checkpoint for why this doesn't just
call the high-level `YOLO(path).train(...)` API.
"""
from __future__ import annotations

import csv
import zlib
from pathlib import Path

from fedxpalm.federated.trainer_utils import build_trainer_from_checkpoint


def effective_seed(base_seed: int, round_idx: int, client_id) -> int:
    """Deterministic, machine-independent local seed, distinct per
    (experiment seed, round, client).

    A single constant seed re-used every round would give every client the
    same shuffle/augmentation stream in every round; deriving from the
    triple keeps runs with the same experiment seed bit-reproducible while
    making rounds/clients (and different experiment seeds) actually differ.
    crc32 is standardized, so the value is stable across OS/Python builds.
    """
    return zlib.crc32(f"{int(base_seed)}-{int(round_idx)}-{client_id}".encode()) % (2**31 - 1)


def read_local_train_log(run_dir: str | Path) -> dict:
    """Last-epoch train losses + learning rate from Ultralytics' results.csv.

    Returns e.g. {"train/box_loss": ..., "train/cls_loss": ..., "train/dfl_loss":
    ..., "lr/pg0": ...} (keys as written by Ultralytics), or {} if the csv is
    missing -- callers log this into history.json per round per client, which
    is what makes client-drift/divergence diagnosable after the fact.
    """
    csv_path = Path(run_dir) / "results.csv"
    if not csv_path.exists():
        return {}
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    last = rows[-1]
    out = {}
    for key, val in last.items():
        key = key.strip()
        if key.startswith(("train/", "lr/")) or key == "epoch":
            try:
                out[key] = float(val)
            except (TypeError, ValueError):
                pass
    return out


def train_client_round(
    global_weights_path: str,
    client_data_yaml: str,
    hyp: dict,
    round_idx: int,
    client_id: str,
    out_dir: str,
    device: str = "0",
) -> str:
    """Fine-tunes `global_weights_path` on one client's local data for one round.

    Returns the path to the resulting weights (`.../weights/last.pt`).
    """
    run_name = f"r{round_idx}_client{client_id}"
    overrides = dict(
        data=client_data_yaml,
        model=global_weights_path,
        epochs=hyp["epochs_per_round"],
        batch=hyp["batch_size"],
        imgsz=hyp.get("imgsz", 640),
        optimizer=hyp.get("optimizer", "SGD"),
        lr0=hyp.get("lr0", 0.01),
        momentum=hyp.get("momentum", 0.9),
        weight_decay=hyp.get("weight_decay", 0.0005),
        patience=hyp.get("patience", 100),
        # Ultralytics defaults warmup_epochs to 3.0 -- with epochs_per_round=2
        # the *entire* local run then happens inside the LR/momentum warmup
        # ramp, every round, so the configured lr0 is never actually reached.
        # Default stays 3.0 (reproduces prior runs); tune via fl_config/CLI.
        warmup_epochs=hyp.get("warmup_epochs", 3.0),
        # per-(seed, round, client) -- recorded in the checkpoint's train_args;
        # consumed by SeededDetectionTrainer to drive shuffle + augmentation
        seed=effective_seed(hyp.get("seed", 0), round_idx, client_id),
        deterministic=True,
        device=device,
        project=out_dir,
        name=run_name,
        exist_ok=True,
        verbose=False,
        val=False,
        plots=False,
        workers=hyp.get("workers", 4),
    )
    trainer = build_trainer_from_checkpoint(global_weights_path, overrides)
    trainer.train()

    weights_path = Path(out_dir) / run_name / "weights" / "last.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"expected client weights at {weights_path}, training may have failed")
    return str(weights_path)
