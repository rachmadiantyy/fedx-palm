"""P3 candidate: "predictor-only" trainable subset for E2 -- the terminal
box-regression and class-prediction convolutions in YOLO11n's Detect head,
discovered PROGRAMMATICALLY from the actual model structure (never
hardcoded), everything else frozen (backbone, neck, earlier Detect convs,
GroupNorm, DFL).

This is NOT LoRA/an adapter -- no new modules are added, no parallel
branches, no zero-init reparameterization. It is a plain subset-selection
of EXISTING parameters, using the same mechanism P1/P2 already use
(Ultralytics' freeze_layer_names substring matching) with a mixed
int+string freeze list, same pattern as models/lora.py's
lora_freeze_spec -- but selecting a different (much smaller) subset.

Rationale: reduces trainable/private dimensionality d far below P2
(929,522) to isolate whether the terminal-only subset (a) has adequate
No-DP learning capacity and (b) if so, whether the resulting ~8x smaller
expected Gaussian noise norm (noise ~ sigma*C*sqrt(d)) translates into
better DP utility than P2 at matched (sigma, C, batch, epsilon).
"""
from __future__ import annotations

import torch.nn as nn


def discover_predictor_terminals(model) -> tuple[list[str], list[str], int]:
    """Walks the Detect head (always model.model[-1] in Ultralytics YOLO)
    and finds, for every branch in cv2 and cv3, the LAST submodule of each
    per-scale sequential -- verified to be a bare nn.Conv2d (no BN/activation,
    i.e. the raw box/class output projection), not assumed from a fixed
    index. Generalizes over however many detection scales/branches the
    model actually has (does not hardcode 3, ".2", or channel counts).

    Returns (terminal_paths, non_terminal_paths, detect_stage_index), where
    each path is like "23.cv2.0.2" (stage.branch.scale.child_index) --
    matches Ultralytics' own freeze substring convention f"model.{x}.".
    """
    detect = model.model[-1]
    stage_idx = len(model.model) - 1
    if not (hasattr(detect, "cv2") and hasattr(detect, "cv3")):
        raise RuntimeError(
            f"model.model[-1] (stage {stage_idx}, type {type(detect).__name__}) has no cv2/cv3 -- "
            f"this does not look like an Ultralytics Detect head; refusing to guess a structure")
    terminal_paths: list[str] = []
    non_terminal_paths: list[str] = []
    for branch_name in ("cv2", "cv3"):
        branch = getattr(detect, branch_name)
        for i, seq in enumerate(branch):
            children = list(seq.children())
            if not children:
                raise RuntimeError(f"{branch_name}[{i}] has no children -- unexpected Detect structure")
            last = children[-1]
            if not isinstance(last, nn.Conv2d):
                raise RuntimeError(
                    f"expected the terminal submodule of {branch_name}[{i}] to be a bare nn.Conv2d "
                    f"(raw prediction output), got {type(last).__name__} instead -- Detect's structure "
                    f"does not match what this subset design assumes; refusing to proceed blindly")
            terminal_paths.append(f"{stage_idx}.{branch_name}.{i}.{len(children) - 1}")
            for j in range(len(children) - 1):
                non_terminal_paths.append(f"{stage_idx}.{branch_name}.{i}.{j}")
    return terminal_paths, non_terminal_paths, stage_idx


def predictor_only_freeze_spec(model) -> list:
    """Mixed Ultralytics freeze list: every stage BEFORE the Detect head as
    plain ints (whole-stage freeze, identical mechanism to every P0/P1/P2
    run) + module-prefix strings for every Detect submodule EXCEPT the
    discovered terminal convs (same mixed-list mechanism as
    fedxpalm.models.lora.lora_freeze_spec, verified against installed
    ultralytics 8.4.51 trainer.py's freeze_layer_names substring matching).
    DFL needs no explicit entry (Ultralytics always freezes '.dfl')."""
    terminal_paths, non_terminal_paths, stage_idx = discover_predictor_terminals(model)
    return list(range(stage_idx)) + non_terminal_paths


def audit_predictor_only_params(model, freeze_spec: list) -> dict:
    """Applies Ultralytics' EXACT freeze-name matching (substring of
    f'model.{x}.' plus the always-frozen '.dfl') to classify every
    parameter -- the ground truth for what _setup_train will make
    trainable. Mirrors fedxpalm.models.lora.audit_lora_params."""
    freeze_names = [f"model.{x}." for x in freeze_spec] + [".dfl"]
    n_train = n_frozen = 0
    trainable_names: list[tuple[str, int]] = []
    unexpected: list[str] = []
    terminal_paths, _, _ = discover_predictor_terminals(model)
    terminal_prefixes = tuple(f"model.{p}." for p in terminal_paths)
    for name, p in model.named_parameters():
        if any(x in name for x in freeze_names):
            n_frozen += p.numel()
            continue
        n_train += p.numel()
        trainable_names.append((name, int(p.numel())))
        if not name.startswith(terminal_prefixes):
            unexpected.append(name)
    return {
        "n_trainable": int(n_train), "n_frozen": int(n_frozen),
        "trainable_param_names": trainable_names,
        "n_trainable_tensors": len(trainable_names),
        "unexpected_trainable": unexpected,
    }
