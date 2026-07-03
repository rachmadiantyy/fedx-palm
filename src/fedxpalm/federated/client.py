"""One client's local fine-tuning for one federated round (no DP -- used for B2).

Uses Ultralytics' high-level `YOLO.train()` API directly: it is well-tested
and handles augmentation/scheduling/checkpointing correctly on its own, so
there is no reason to hand-roll a loop here the way privacy/dp_sgd.py has
to (Opacus needs low-level control that `YOLO.train()` does not expose).
"""
from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


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
    model = YOLO(global_weights_path)
    run_name = f"r{round_idx}_client{client_id}"
    model.train(
        data=client_data_yaml,
        epochs=hyp["epochs_per_round"],
        batch=hyp["batch_size"],
        imgsz=hyp.get("imgsz", 640),
        optimizer=hyp.get("optimizer", "SGD"),
        lr0=hyp.get("lr0", 0.01),
        momentum=hyp.get("momentum", 0.9),
        weight_decay=hyp.get("weight_decay", 0.0005),
        device=device,
        project=out_dir,
        name=run_name,
        exist_ok=True,
        verbose=False,
        val=False,
        plots=False,
        workers=4,
    )
    weights_path = Path(out_dir) / run_name / "weights" / "last.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"expected client weights at {weights_path}, training may have failed")
    return str(weights_path)
