"""Conv-LoRA injection for the P2 trainable stages (Phase D of the E2 DP
redesign) -- parameter-efficient private adaptation.

Design (Gate 1, capacity control first -- no DP here):
  - Every non-depthwise nn.Conv2d in stages 16/19/22 and in the Detect head's
    cv2 branches + cv3 pointwise convs is wrapped in ConvLoRA: the pretrained
    conv (.base) stays frozen; a parallel low-rank branch lora_B(lora_A(x))
    is trainable, with lora_B ZERO-INITIALIZED so the wrapped module computes
    EXACTLY the base function at initialization (a zero-weight conv outputs
    exactly 0.0, so forward equivalence is bitwise, not approximate).
  - The final nc-class projection convs (cv3.{0,1,2}.2, out_channels=nc<=rank
    for this 6-class model) are NOT wrapped: LoRA is degenerate when
    rank >= out_channels, and they are nc-specific and tiny -- they train
    FULLY instead.
  - Depthwise convs (groups == in_channels; LoRA degenerate) stay frozen.
  - GroupNorm affine parameters inside the four trainable stages stay
    TRAINABLE (an SSF-style recalibration path, ~5k params).
  - DFL stays frozen (Ultralytics' own always-frozen rule, unchanged).

Freezing rides the EXISTING Ultralytics mechanism, no trainer changes:
BaseTrainer._setup_train builds freeze_layer_names = [f"model.{x}." for x in
freeze_list] + [".dfl"] and string entries pass through f"model.{x}." exactly
like ints do (verified against installed ultralytics 8.4.51 trainer.py,
lines 311-324; params NOT matching are force-(re)enabled trainable by that
same loop, which is precisely what the .lora_/cv3.x.2/GN params need).
lora_freeze_spec() therefore returns a MIXED list: the P2 frozen stage ints
(0..15,17,18,20,21) plus one module-prefix string per frozen submodule inside
the trainable stages (every ConvLoRA's .base, every depthwise conv).

Opacus compatibility: ConvLoRA is a plain container; its children are stock
nn.Conv2d modules, so GradSampleModule hooks them natively; frozen
(requires_grad=False) base convs are already tolerated by the existing
dp_sgd path (same mechanism as every P0/P1/P2 run).

FedAvg compatibility: state_dict keys change once at surgery time (base conv
weights move to '<path>.base.weight'), consistently for every client because
all clients initialize from the SAME surgered checkpoint. Frozen keys are
identical across clients, so FedAvg-averaging them is a no-op; LoRA keys
aggregate normally. No fedavg/server change needed.
"""
from __future__ import annotations

import math

import torch  # noqa: F401  (kept for callers doing torch.load-based audits)
import torch.nn as nn

P2_FROZEN_STAGES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 20, 21]
P2_TRAINABLE_STAGES = [16, 19, 22, 23]
_LORA_MARKERS = (".lora_A", ".lora_B", ".base")


class ConvLoRA(nn.Module):
    """Frozen base conv + trainable zero-initialized low-rank parallel branch.

    forward(x) = base(x) + lora_B(lora_A(x)); lora_B starts at zero, so the
    initial function equals the base conv EXACTLY (bitwise)."""

    def __init__(self, base: nn.Conv2d, rank: int):
        super().__init__()
        if base.groups != 1:
            raise ValueError("ConvLoRA does not support grouped/depthwise convs")
        self.base = base
        self.rank = rank
        self.lora_A = nn.Conv2d(
            base.in_channels, rank, kernel_size=base.kernel_size, stride=base.stride,
            padding=base.padding, dilation=base.dilation, bias=False)
        self.lora_B = nn.Conv2d(rank, base.out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x))


def _stage_of_module(name: str) -> int | None:
    parts = name.split(".")
    if len(parts) >= 2 and parts[0] == "model" and parts[1].isdigit():
        return int(parts[1])
    return None


def _inside_lora(name: str) -> bool:
    return any(name.endswith(m) or (m + ".") in (name + ".") for m in _LORA_MARKERS)


def classify_conv(path: str, conv: nn.Conv2d, rank: int) -> str:
    """'lora' | 'full_train' | 'frozen_dw' | 'frozen_dfl' for one candidate
    Conv2d in a trainable stage."""
    if "dfl" in path:
        return "frozen_dfl"
    if conv.groups != 1:
        return "frozen_dw"
    if conv.out_channels <= rank:
        return "full_train"  # e.g. the nc-class cv3.x.2 projections (nc<=rank)
    return "lora"


def _candidate_convs(model: nn.Module):
    """(full_name, conv) for every bare Conv2d in a P2-trainable stage that is
    not already inside a ConvLoRA wrapper."""
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Conv2d):
            continue
        stage = _stage_of_module(name)
        if stage not in P2_TRAINABLE_STAGES or _inside_lora(name):
            continue
        yield name, mod


def inject_conv_lora(model: nn.Module, rank: int = 8) -> dict:
    """Wraps target convs in-place; returns an audit report dict. Refuses to
    double-inject."""
    if any(isinstance(m, ConvLoRA) for m in model.modules()):
        raise RuntimeError("model already contains ConvLoRA modules -- refusing to double-inject")
    report = {"rank": rank, "wrapped": [], "full_train": [], "frozen_dw": [], "frozen_dfl": []}
    # collect first (mutating while iterating named_modules is undefined)
    targets = [(name, conv, classify_conv(name, conv, rank)) for name, conv in _candidate_convs(model)]
    module_by_name = dict(model.named_modules())
    for name, conv, kind in targets:
        if kind != "lora":
            report[kind].append(name)
            continue
        parent_name, _, attr = name.rpartition(".")
        setattr(module_by_name[parent_name], attr, ConvLoRA(conv, rank))
        report["wrapped"].append(name)
    report["n_wrapped"] = len(report["wrapped"])
    report["n_lora_params"] = int(sum(p.numel() for n, p in model.named_parameters() if ".lora_" in n))
    return report


def _strip_model_prefix(name: str) -> str:
    return name[len("model."):] if name.startswith("model.") else name


def lora_freeze_spec(model: nn.Module) -> list:
    """Mixed Ultralytics freeze list (stage ints + module-prefix strings)
    freezing everything except: .lora_ params, full-train tiny convs
    (out_channels<=rank, i.e. cv3.x.2), and GroupNorm affine in the trainable
    stages. Entry x is matched by Ultralytics as substring f'model.{x}.' over
    parameter names; DFL needs no entry (always frozen by Ultralytics)."""
    rank = _infer_rank(model)
    spec: list = list(P2_FROZEN_STAGES)
    for name, mod in model.named_modules():
        if isinstance(mod, ConvLoRA):
            spec.append(_strip_model_prefix(f"{name}.base"))
    for name, conv in _candidate_convs(model):  # bare convs left after injection
        if classify_conv(name, conv, rank) == "frozen_dw":
            spec.append(_strip_model_prefix(name))
    return spec


def _infer_rank(model: nn.Module) -> int:
    for m in model.modules():
        if isinstance(m, ConvLoRA):
            return m.rank
    return 8


def audit_lora_params(model: nn.Module, freeze_spec: list) -> dict:
    """Applies Ultralytics' EXACT freeze-name matching (substring of
    f'model.{x}.' plus the always-frozen '.dfl') to classify every parameter:
    the ground truth for what _setup_train will make trainable."""
    freeze_names = [f"model.{x}." for x in freeze_spec] + [".dfl"]
    n_train = n_frozen = 0
    kinds = {"lora": 0, "full_train_conv": 0, "gn_affine": 0, "other": 0}
    unexpected: list[str] = []
    for name, p in model.named_parameters():
        if any(x in name for x in freeze_names):
            n_frozen += p.numel()
            continue
        n_train += p.numel()
        if ".lora_" in name:
            kinds["lora"] += p.numel()
        elif ".bn." in name:
            kinds["gn_affine"] += p.numel()
        elif ".cv3." in name and (".2.weight" in name or ".2.bias" in name):
            kinds["full_train_conv"] += p.numel()
        else:
            kinds["other"] += p.numel()
            unexpected.append(name)
    return {"n_trainable": int(n_train), "n_frozen": int(n_frozen),
            "trainable_breakdown": {k: int(v) for k, v in kinds.items()},
            "unexpected_trainable": unexpected}
