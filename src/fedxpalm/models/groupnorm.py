"""BatchNorm -> GroupNorm conversion, required before per-sample DP-SGD.

Opacus computes a *per-sample* gradient for every parameter. BatchNorm's
statistics are computed across the batch, so its "per-sample" gradient
would leak information about other samples in the batch -- Opacus'
ModuleValidator rejects it outright. GroupNorm normalizes within a single
sample (across channel groups), so it has no such cross-sample coupling
and is the standard substitute for DP training (Kurakin et al., 2022 use
this exact replacement for CNNs).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _pick_num_groups(num_channels: int, max_groups: int = 32) -> int:
    """Largest divisor of num_channels that is <= max_groups."""
    for g in range(min(max_groups, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1


def convert_batchnorm_to_groupnorm(module: nn.Module, max_groups: int = 32) -> nn.Module:
    """Recursively replaces every BatchNorm2d in `module` with GroupNorm in-place.

    GroupNorm's affine (weight/bias) is initialized from the BatchNorm layer
    it replaces so the converted model starts numerically close to the
    pretrained checkpoint; running_mean/running_var are dropped since
    GroupNorm has no running statistics.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            num_groups = _pick_num_groups(num_channels, max_groups)
            gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=child.eps, affine=True)
            with torch.no_grad():
                if child.affine:
                    gn.weight.copy_(child.weight)
                    gn.bias.copy_(child.bias)
            setattr(module, name, gn)
        else:
            convert_batchnorm_to_groupnorm(child, max_groups)
    return module


def count_batchnorm_layers(module: nn.Module) -> int:
    return sum(1 for m in module.modules() if isinstance(m, nn.BatchNorm2d))


def disable_inplace_ops(module: nn.Module) -> int:
    """Sets `.inplace = False` on every activation in `module` (SiLU, ReLU, ...).

    Opacus' GradSampleModule tracks activations via forward hooks to later
    pair them with per-sample output gradients. YOLOv11's Conv block runs
    its activation in-place (`self.act(self.bn(self.conv(x)))` with
    `SiLU(inplace=True)`), which overwrites the tracked activation tensor
    before the backward hook can read it, and raises
    "Output ... is a view and is being modified inplace" during backward.
    """
    disabled = 0
    for m in module.modules():
        if hasattr(m, "inplace") and isinstance(getattr(m, "inplace"), bool):
            m.inplace = False
            disabled += 1
    return disabled


_fuse_patched = False


def patch_fuse_for_groupnorm() -> None:
    """Makes Ultralytics' `BaseModel.fuse()` skip GroupNorm layers instead of crashing.

    `model.val()` / `.predict()` load weights through `AutoBackend`, which
    unconditionally calls `model.fuse()` to algebraically merge each Conv's
    BatchNorm into its preceding conv for faster inference. GroupNorm has no
    such closed-form fusion (and doesn't need one -- it's already cheap), so
    the stock `fuse()` crashes with `AttributeError: 'GroupNorm' object has
    no attribute 'running_var'` the moment it hits a converted layer. This
    patches the class method once, process-wide, to leave GroupNorm'd Conv
    blocks unfused and fuse everything else as normal. Applied automatically
    on `import fedxpalm`.
    """
    global _fuse_patched
    if _fuse_patched:
        return

    from ultralytics.nn.modules import Conv, Conv2, ConvTranspose, DWConv, RepConv, RepVGGDW
    from ultralytics.nn.modules.head import Detect
    from ultralytics.nn.tasks import BaseModel
    from ultralytics.utils.torch_utils import fuse_conv_and_bn, fuse_deconv_and_bn

    def _groupnorm_safe_fuse(self, verbose: bool = True):
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, "bn"):
                    if not isinstance(m.bn, nn.BatchNorm2d):
                        continue  # GroupNorm (or anything else non-fusable): leave as-is
                    if isinstance(m, Conv2):
                        m.fuse_convs()
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)
                    delattr(m, "bn")
                    m.forward = m.forward_fuse
                if isinstance(m, ConvTranspose) and hasattr(m, "bn") and isinstance(m.bn, nn.BatchNorm2d):
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, "bn")
                    m.forward = m.forward_fuse
                if isinstance(m, RepConv):
                    m.fuse_convs()
                    m.forward = m.forward_fuse
                if isinstance(m, RepVGGDW):
                    m.fuse()
                    m.forward = m.forward_fuse
                if isinstance(m, Detect) and getattr(m, "end2end", False):
                    m.fuse()
            self.info(verbose=verbose)
        return self

    BaseModel.fuse = _groupnorm_safe_fuse
    _fuse_patched = True


def prepare_model_for_dp(yolo_model, max_groups: int = 32):
    """Convert BN->GN, disable in-place activations, and run Opacus' ModuleValidator.

    `yolo_model` is an `ultralytics.YOLO` instance; the underlying nn.Module
    lives at `yolo_model.model`. Required before wrapping the model with
    `opacus.PrivacyEngine.make_private()` for per-sample DP-SGD.
    """
    from opacus.validators import ModuleValidator

    n_before = count_batchnorm_layers(yolo_model.model)
    convert_batchnorm_to_groupnorm(yolo_model.model, max_groups=max_groups)
    n_after = count_batchnorm_layers(yolo_model.model)
    print(f"Converted {n_before - n_after} BatchNorm2d layers to GroupNorm "
          f"({n_after} BatchNorm2d layers remain).")

    n_inplace = disable_inplace_ops(yolo_model.model)
    print(f"Disabled in-place=True on {n_inplace} activation module(s).")

    errors = ModuleValidator.validate(yolo_model.model, strict=False)
    if errors:
        print(f"Opacus ModuleValidator found {len(errors)} remaining issue(s); auto-fixing.")
        yolo_model.model = ModuleValidator.fix(yolo_model.model)
        remaining = ModuleValidator.validate(yolo_model.model, strict=False)
        if remaining:
            raise RuntimeError(f"Model still fails Opacus validation after fix: {remaining}")
    return yolo_model
