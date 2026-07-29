"""Federated Averaging (McMahan et al., 2017): w_{t+1} = sum_k (n_k / N) * w_k."""
from __future__ import annotations

import torch


def fedavg(state_dicts: list[dict], sample_counts: list[int]) -> dict:
    """Weighted-average a list of client state_dicts by their local sample counts.

    All state_dicts must share identical keys/shapes (true here since every
    client fine-tunes the same GroupNorm-converted YOLOv11n architecture).
    """
    if len(state_dicts) != len(sample_counts):
        raise ValueError("state_dicts and sample_counts must have the same length")
    total = sum(sample_counts)
    if total == 0:
        raise ValueError("total sample count across clients is zero")

    keys = state_dicts[0].keys()
    for i, sd in enumerate(state_dicts[1:], start=1):
        if sd.keys() != keys:
            raise ValueError(f"client {i} state_dict keys differ from client 0's")

    avg = {}
    for key in keys:
        stacked = torch.stack(
            [sd[key].float() * (n / total) for sd, n in zip(state_dicts, sample_counts)],
            dim=0,
        )
        summed = stacked.sum(dim=0)
        avg[key] = summed.to(state_dicts[0][key].dtype)
    return avg
