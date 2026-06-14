"""
GroupNorm conversion utility for Opacus compatibility.

YOLOv11n has 81 BatchNorm2d layers, which Opacus' GradSampleModule
cannot handle because per-sample gradients are undefined for BN
(BN statistics depend on the entire batch).

We replace each BN2d in-place with GroupNorm. GroupNorm normalizes
per-sample so per-sample gradients are well-defined.

Used by all rebuild training scripts (B1, B2, E1, E2).
"""
from typing import Tuple

import torch
import torch.nn as nn


def replace_bn_with_gn(module: nn.Module, num_groups: int = 8) -> int:
    """Replace all BatchNorm2d in module with GroupNorm (in-place).

    Args:
        module: Root nn.Module (e.g. YOLO detection model).
        num_groups: Preferred group count. Auto-reduces if it doesn't
            divide num_channels (some YOLO layers have e.g. 16, 32, 64,
            128, 256 channels — 8 divides all of these cleanly).

    Returns:
        Number of BN layers converted.
    """
    converted = 0
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            groups = num_groups
            while num_channels % groups != 0 and groups > 1:
                groups -= 1
            gn = nn.GroupNorm(
                num_groups=groups,
                num_channels=num_channels,
                affine=True,
            )
            with torch.no_grad():
                if child.affine:
                    gn.weight.copy_(child.weight)
                    gn.bias.copy_(child.bias)
            setattr(module, name, gn)
            converted += 1
        else:
            converted += replace_bn_with_gn(child, num_groups)
    return converted


def disable_inplace_activations(module: nn.Module) -> int:
    """Set inplace=False on all activations (in-place ops break Opacus).

    YOLOv11 uses in-place SiLU (and similar) activations. Opacus'
    GradSampleModule installs backward hooks whose output views must not be
    modified in-place, otherwise per-sample gradients become incorrect and
    PyTorch raises a RuntimeError. Walking the module and flipping every
    `inplace` flag to False makes each activation allocate a fresh tensor.

    Returns:
        Number of modules whose inplace flag was disabled.
    """
    count = 0
    for m in module.modules():
        if getattr(m, "inplace", False):
            m.inplace = False
            count += 1
    return count


def count_bn_layers(module: nn.Module) -> Tuple[int, int]:
    """Return (n_bn_remaining, n_gn_present)."""
    n_bn = sum(
        1 for m in module.modules()
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
    )
    n_gn = sum(1 for m in module.modules() if isinstance(m, nn.GroupNorm))
    return n_bn, n_gn
