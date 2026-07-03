"""Per-sample DP-SGD client round via Opacus, for experiment blocks E1 (full)
and E2 (partial, backbone frozen).

Ultralytics' high-level `YOLO.train()` owns its optimizer/backward internally
and gives no hook for Opacus to wrap it, so this reimplements just enough of
`DetectionTrainer`'s setup to get a model + optimizer + dataloader, then
drives the step loop by hand. Validated end-to-end (forward/backward/step +
epsilon accounting, for both freeze=None and freeze=backbone) against a
synthetic dataset on CPU in a network-restricted sandbox -- re-verify the
first real run on your GPU box, since Ultralytics' internal APIs can shift
between point releases.
"""
from __future__ import annotations

from pathlib import Path

import torch
from opacus import PrivacyEngine
from ultralytics.models.yolo.detect.train import DetectionTrainer

from fedxpalm.models.groupnorm import disable_inplace_ops


def train_client_round_dp(
    global_weights_path: str,
    client_data_yaml: str,
    hyp: dict,
    dp_hyp: dict,
    round_idx: int,
    client_id: str,
    out_dir: str,
    device: str = "0",
    freeze_stages: list[int] | None = None,
) -> tuple[dict, dict]:
    """Fine-tunes `global_weights_path` on one client with per-sample DP-SGD.

    Returns (state_dict, info) where state_dict has clean (non-Opacus-prefixed)
    keys ready for fedavg(), and info carries {epsilon, steps, sigma, n_samples}
    for the round's history log.
    """
    ckpt = torch.load(global_weights_path, map_location="cpu", weights_only=False)
    model = ckpt["model"].float()
    disable_inplace_ops(model)  # in case a fresh (non-DP-prepared) checkpoint slips in

    run_name = f"r{round_idx}_client{client_id}_dp"
    overrides = dict(
        data=client_data_yaml,
        model=global_weights_path,
        epochs=hyp["epochs_per_round"],
        batch=hyp["batch_size"],
        imgsz=hyp.get("imgsz", 640),
        device=device,
        workers=hyp.get("workers", 4),
        amp=False,          # Opacus does not officially support mixed precision
        plots=False,
        val=False,
        verbose=False,
        freeze=freeze_stages,
        exist_ok=True,
        project=out_dir,
        name=run_name,
    )
    trainer = DetectionTrainer(overrides=overrides)
    trainer.model = model  # pre-built -> setup_model() reuses it instead of re-loading global_weights_path
    trainer._setup_train()

    privacy_engine = PrivacyEngine(accountant=dp_hyp.get("accountant", "prv"))
    dp_model, dp_optimizer, dp_loader = privacy_engine.make_private(
        module=trainer.model,
        optimizer=trainer.optimizer,
        data_loader=trainer.train_loader,
        noise_multiplier=dp_hyp["sigma"],
        max_grad_norm=dp_hyp["max_grad_norm"],
        poisson_sampling=True,
    )

    dp_model.train()
    n_steps = 0
    for _epoch in range(hyp["epochs_per_round"]):
        for batch in dp_loader:
            if len(batch["img"]) == 0:  # Poisson sampling can draw an empty batch
                continue
            batch = trainer.preprocess_batch(batch)
            dp_optimizer.zero_grad()
            loss, _loss_items = dp_model(batch)
            loss.sum().backward()
            dp_optimizer.step()
            n_steps += 1

    epsilon = privacy_engine.get_epsilon(delta=dp_hyp["delta"])
    state_dict = {k.replace("_module.", "", 1): v.detach().clone() for k, v in dp_model.state_dict().items()}

    info = {
        "epsilon": epsilon,
        "delta": dp_hyp["delta"],
        "sigma": dp_hyp["sigma"],
        "max_grad_norm": dp_hyp["max_grad_norm"],
        "steps": n_steps,
        "n_samples": len(dp_loader.dataset),
        "frozen": freeze_stages is not None,
    }
    return state_dict, info
