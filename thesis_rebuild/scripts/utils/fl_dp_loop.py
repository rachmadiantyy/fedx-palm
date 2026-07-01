"""
Federated DP-SGD training loop for YOLOv11n-GN.

Architecture (mirrors thesis Section 3.1.3 communication round):
  for round in 1..T:
    for client in 1..K:
        local_state = copy(global_state)
        wrap with Opacus PrivacyEngine (sigma, C)
        train local_epochs on client shard
    global_state = FedAvg(client_states, weighted by shard size)
    eval on global val set
    log (round, sigma, K, mAP, eps)

Reuses utils/yolo_dp_loop.py for the per-client DP-SGD training,
adds the federation layer on top.

Used by:
  train_b2_fl.py              (no DP, sigma=0 path uses skip)
  train_e1_fl_dp_sgd_full.py  (full DP-SGD federated)
  train_e2_fl_dp_sgd_partial.py (head-only DP-SGD federated)
"""
from __future__ import annotations

import copy
import csv
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from opacus import PrivacyEngine

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.loss import v8DetectionLoss

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from thesis_rebuild.scripts.utils.gn_convert import (  # noqa: E402
    count_bn_layers,
    disable_inplace_activations,
    replace_bn_with_gn,
)


@dataclass
class FedDPConfig:
    """Hyperparameters for one federated DP-SGD run."""
    # Data + model
    K: int
    clients_root: str            # e.g. data/clients_K4/
    global_val_yaml: str         # e.g. data/global_val.yaml
    weights: str = "yolo11n.pt"
    imgsz: int = 640
    num_classes: int = 6

    # Federation
    rounds: int = 5
    local_epochs: int = 2
    batch_size: int = 16
    lr0: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005

    # DP-SGD (set noise_multiplier=0 to disable DP -> pure FedAvg)
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    target_delta: float = 1e-5
    use_dp: bool = True          # False = baseline B2 (no DP)
    freeze_backbone: bool = False

    # Bookkeeping
    seed: int = 42
    device: str = "cuda"
    project: str = "thesis_rebuild/runs"
    name: str = "fl_run"
    num_groups: int = 8


def build_gn_yolo(weights: str, num_groups: int, freeze_backbone: bool) -> nn.Module:
    yolo = YOLO(weights)
    model = yolo.model
    replace_bn_with_gn(model, num_groups=num_groups)
    # Opacus forbids in-place ops (in-place SiLU breaks per-sample grad hooks).
    disable_inplace_activations(model)
    n_bn, n_gn = count_bn_layers(model)
    assert n_bn == 0
    # GroupNorm has no running_var, but ultralytics' is_fused() miscounts GN
    # as a norm layer and tries Conv+BN fusion during val()/AutoBackend,
    # crashing on the missing attribute. Fusion is a BN-only inference
    # speedup (numerically identical result), so neutralize it to a no-op.
    model.fuse = lambda verbose=True: model
    model.train()
    if freeze_backbone:
        for name, p in model.named_parameters():
            stage = name.split(".")[1] if name.startswith("model.") else ""
            if stage.isdigit() and int(stage) < 10:
                p.requires_grad = False
    return model


def make_client_loader(client_dir: Path, cfg: FedDPConfig) -> DataLoader:
    """Build a YOLO DataLoader for one client's training shard.

    Each client_dir has images/ + labels/ flat (no train/val subdirs).
    """
    data_yaml = client_dir / "data.yaml"
    data_info = check_det_dataset(str(data_yaml))
    args = get_cfg(DEFAULT_CFG)
    args.imgsz = cfg.imgsz
    args.batch = cfg.batch_size
    # Windows + Opacus DPDataLoader (Poisson sampler) deadlocks with
    # num_workers > 0; force single-threaded loading. ~15% slower but
    # eliminates the hang we hit on the K=2 sigma=1.0 cell.
    args.workers = 0
    args.cache = False
    args.rect = False

    dataset = build_yolo_dataset(
        args, data_info["train"], cfg.batch_size, data_info,
        mode="train", rect=False, stride=32,
    )
    return DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        collate_fn=getattr(dataset, "collate_fn", None), drop_last=True,
    )


def train_one_client(
    global_state: dict,
    loader: DataLoader,
    cfg: FedDPConfig,
) -> tuple[dict, float, int]:
    """Train one client locally, optionally with Opacus DP-SGD.

    Returns (new_state_dict, epsilon_spent_this_round, num_samples).
    """
    model = build_gn_yolo(cfg.weights, cfg.num_groups, cfg.freeze_backbone)
    model.load_state_dict(global_state, strict=False)
    model = model.to(cfg.device)

    model.args = get_cfg(DEFAULT_CFG)
    model.args.box = 7.5
    model.args.cls = 0.5
    model.args.dfl = 1.5
    criterion = v8DetectionLoss(model)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(params, lr=cfg.lr0, momentum=cfg.momentum,
                    weight_decay=cfg.weight_decay)

    eps = 0.0
    privacy_engine = None
    if cfg.use_dp:
        privacy_engine = PrivacyEngine()
        model, optimizer, loader = privacy_engine.make_private(
            module=model, optimizer=optimizer, data_loader=loader,
            noise_multiplier=cfg.noise_multiplier,
            max_grad_norm=cfg.max_grad_norm,
        )

    n_samples = 0
    step = 0
    for epoch in range(cfg.local_epochs):
        for batch in loader:
            imgs = batch["img"].to(cfg.device, non_blocking=True).float() / 255.0
            batch_gpu = {k: (v.to(cfg.device) if torch.is_tensor(v) else v)
                         for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss_components, _ = criterion(model(imgs), batch_gpu)
            # v8DetectionLoss returns [box, cls, dfl] in ultralytics 8.4.x;
            # .sum() reduces to scalar (idempotent if already scalar) and
            # keeps the per-sample sum semantics Opacus expects.
            loss = loss_components.sum()
            loss.backward()
            optimizer.step()
            n_samples += imgs.shape[0]
            if step % 20 == 0:
                print(f"    ep{epoch} step{step:04d} loss={float(loss.detach()):.4f}")
            step += 1

    if privacy_engine is not None:
        eps = privacy_engine.get_epsilon(delta=cfg.target_delta)

    raw = model._module if hasattr(model, "_module") else model
    return {k: v.detach().cpu() for k, v in raw.state_dict().items()}, eps, n_samples


def fedavg(client_states: list[tuple[dict, int]]) -> dict:
    """Weighted average of client state_dicts by num_samples."""
    total = sum(n for _, n in client_states)
    avg = OrderedDict()
    keys = client_states[0][0].keys()
    for k in keys:
        if not torch.is_tensor(client_states[0][0][k]):
            avg[k] = client_states[0][0][k]
            continue
        weighted = sum(
            state[k].float() * (n / total) for state, n in client_states
        )
        avg[k] = weighted.to(client_states[0][0][k].dtype)
    return avg


def evaluate_global(state: dict, cfg: FedDPConfig) -> dict:
    """Build a fresh GN YOLO with given state and run val mode."""
    model = build_gn_yolo(cfg.weights, cfg.num_groups, freeze_backbone=False)
    model.load_state_dict(state, strict=False)

    yolo = YOLO(cfg.weights)
    yolo.model = model
    yolo.model.to(cfg.device)
    results = yolo.val(
        data=cfg.global_val_yaml, imgsz=cfg.imgsz,
        batch=cfg.batch_size, device=cfg.device,
        verbose=False, plots=False, save=False,
    )
    return {
        "mAP50": float(results.box.map50) if hasattr(results, "box") else 0.0,
        "mAP5095": float(results.box.map) if hasattr(results, "box") else 0.0,
        "precision": float(results.box.mp) if hasattr(results, "box") else 0.0,
        "recall": float(results.box.mr) if hasattr(results, "box") else 0.0,
    }


def already_complete(cfg: FedDPConfig) -> Optional[dict]:
    """If this run already finished (rounds.csv has cfg.rounds rows + best.pt
    exists), return a result dict so the caller can skip. Otherwise None.

    Used by wrappers to make a long grid resumable: re-running the orchestrator
    after a crash will pick up where it left off instead of redoing finished
    cells.
    """
    out_dir = Path(cfg.project) / cfg.name
    csv_path = out_dir / "rounds.csv"
    ckpt = out_dir / "best.pt"
    if not (csv_path.exists() and ckpt.exists()):
        return None
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    if len(rows) < cfg.rounds:
        return None  # partial — re-run from scratch
    best = max(rows, key=lambda r: float(r["mAP50"]))
    return {
        "K": cfg.K,
        "sigma": cfg.noise_multiplier if cfg.use_dp else 0,
        "freeze_backbone": cfg.freeze_backbone,
        "rounds": cfg.rounds,
        "best_mAP50": float(best["mAP50"]),
        "final_epsilon": float(rows[-1]["epsilon"]),
        "ckpt": str(ckpt),
        "history": rows,
    }


def run_federated(cfg: FedDPConfig) -> dict:
    """Drive the K-client × T-round federated DP-SGD loop."""
    cached = already_complete(cfg)
    if cached is not None:
        print(f"[skip] {cfg.name} already complete "
              f"(best mAP50={cached['best_mAP50']:.4f}, eps={cached['final_epsilon']:.3f})")
        return cached

    torch.manual_seed(cfg.seed)
    out_dir = Path(cfg.project) / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"FL DP-SGD: K={cfg.K}, T={cfg.rounds} rounds, "
          f"sigma={cfg.noise_multiplier if cfg.use_dp else 'OFF'}, "
          f"freeze_backbone={cfg.freeze_backbone}")
    print("=" * 70)

    # Init global model
    global_model = build_gn_yolo(cfg.weights, cfg.num_groups, freeze_backbone=False)
    global_state = {k: v.detach().cpu() for k, v in global_model.state_dict().items()}

    clients_root = Path(cfg.clients_root)
    client_dirs = sorted([p for p in clients_root.iterdir()
                          if p.is_dir() and p.name.startswith("client_")])
    assert len(client_dirs) == cfg.K, \
        f"Expected {cfg.K} client dirs, found {len(client_dirs)} in {clients_root}"

    history = []
    final_eps = 0.0
    best_map = 0.0
    best_state = global_state

    # Open rounds.csv and write incrementally so a mid-run crash still
    # leaves usable partial results on disk (long full-grid runs).
    csv_path = out_dir / "rounds.csv"
    csv_f = csv_path.open("w", newline="")
    csv_w = csv.DictWriter(
        csv_f,
        fieldnames=["round", "epsilon", "mAP50", "mAP5095", "precision", "recall"],
    )
    csv_w.writeheader()
    csv_f.flush()

    try:
        for r in range(1, cfg.rounds + 1):
            print(f"\n-- Round {r}/{cfg.rounds} --")
            client_results = []
            round_eps = []
            for cdir in client_dirs:
                loader = make_client_loader(cdir, cfg)
                new_state, eps, n = train_one_client(
                    global_state, loader, cfg,
                )
                client_results.append((new_state, n))
                round_eps.append(eps)
                print(f"  {cdir.name}: n={n}, eps={eps:.3f}")

            global_state = fedavg(client_results)
            # eps after this round (max across clients; same sigma so they match)
            final_eps = max(round_eps) if round_eps else 0.0

            metrics = evaluate_global(global_state, cfg)
            print(f"  GLOBAL val: mAP50={metrics['mAP50']:.4f} "
                  f"mAP5095={metrics['mAP5095']:.4f}  eps={final_eps:.3f}")
            row = {"round": r, "epsilon": final_eps, **metrics}
            history.append(row)
            csv_w.writerow(row)
            csv_f.flush()

            if metrics["mAP50"] > best_map:
                best_map = metrics["mAP50"]
                best_state = global_state
                # Persist best.pt the moment we beat the prior best — mid-run
                # crashes after this point still leave the best checkpoint.
                torch.save(
                    {"state": best_state, "config": cfg.__dict__,
                     "final_eps": final_eps, "best_mAP50": best_map,
                     "round": r},
                    out_dir / "best.pt",
                )
    finally:
        csv_f.close()

    return {
        "K": cfg.K, "sigma": cfg.noise_multiplier if cfg.use_dp else 0,
        "freeze_backbone": cfg.freeze_backbone,
        "rounds": cfg.rounds, "best_mAP50": best_map,
        "final_epsilon": final_eps, "ckpt": str(out_dir / "best.pt"),
        "history": history,
    }
