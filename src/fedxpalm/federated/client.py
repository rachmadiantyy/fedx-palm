"""One client's local fine-tuning for one federated round (no DP -- used for B2).

See trainer_utils.build_trainer_from_checkpoint for why this doesn't just
call the high-level `YOLO(path).train(...)` API.
"""
from __future__ import annotations

from pathlib import Path

from fedxpalm.federated.trainer_utils import build_trainer_from_checkpoint


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
        optimizer=hyp.get("optimizer", "AdamW"),
        lr0=hyp.get("lr0", 0.001),
        momentum=hyp.get("momentum", 0.9),
        weight_decay=hyp.get("weight_decay", 0.0005),
        patience=hyp.get("patience", 100),
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
