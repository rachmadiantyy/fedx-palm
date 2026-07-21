"""Shared freeze/normalization audit helpers for E1/E2 DP-SGD runs.

Used by:
  - scripts/19_dp_smoke_test.py (2-round mechanism check)
  - scripts/_dp_sweep_common.py (persists a representative audit into the
    real 40-round sweep's results/*.json -- previously computed only in the
    smoke test and never saved for the actual sweep runs)
  - scripts/20_dp_freeze_audit.py (fast, no-full-round diagnostic)

Extracted from 19_dp_smoke_test.py verbatim (no behavior change) so all
three call sites audit freeze state identically instead of drifting.
"""
from __future__ import annotations

BACKBONE_MAX_STAGE = 10  # model.0..model.10 = backbone; 11..22 neck; 23 head

# Parameters that are frozen BY ARCHITECTURE, not a methodology error:
# Ultralytics always freezes the DFL fixed conv (weight = arange(reg_max),
# never trained) in every YOLO run, DP or not. A frozen param whose name
# matches one of these is expected; anything else that is frozen-but-in-the-
# optimizer is a real problem (a param that should train but won't).
EXPECTED_FROZEN_PATTERNS = ("dfl.conv",)


def classify_frozen(names):
    expected = [n for n in names if any(p in n for p in EXPECTED_FROZEN_PATTERNS)]
    unexpected = [n for n in names if not any(p in n for p in EXPECTED_FROZEN_PATTERNS)]
    return expected, unexpected


def stage_of(param_key: str) -> int | None:
    # keys look like "model.5.cv1.conv.weight"
    parts = param_key.split(".")
    if len(parts) >= 2 and parts[0] == "model" and parts[1].isdigit():
        return int(parts[1])
    return None


def load_state(path):
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    return model.state_dict(), model


def diff_state_dicts(sd0, sd1):
    """Compare two state_dicts (already in memory -- no disk I/O) by YOLO
    stage. Returns (backbone_changed, neck_head_changed, dfl_changed).

    Matches the original (pre-refactor) two-pass semantics exactly: DFL is
    stage 23 (head), so a DFL change also counts toward neck_head_changed in
    addition to dfl_changed -- kept as-is (not "cleaned up") so this refactor
    cannot silently shift the pass/fail boundary of already-validated E1/E2
    smoke-test assertions.
    """
    import torch

    backbone_changed = neck_head_changed = False
    for k in sd1:
        if k not in sd0 or sd0[k].shape != sd1[k].shape:
            continue
        stage = stage_of(k)
        if stage is None:
            continue
        differs = not torch.equal(sd0[k].float(), sd1[k].float())
        if stage <= BACKBONE_MAX_STAGE:
            backbone_changed = backbone_changed or differs
        else:
            neck_head_changed = neck_head_changed or differs

    dfl_changed = False
    for k in sd1:
        if "dfl.conv" in k and k in sd0 and sd0[k].shape == sd1[k].shape:
            if not torch.equal(sd0[k].float(), sd1[k].float()):
                dfl_changed = True
    return backbone_changed, neck_head_changed, dfl_changed


def audit_freeze(init_weights, final_weights):
    """Diff two checkpoints (by path) by YOLO stage. Returns
    (backbone_changed, neck_head_changed, n_batchnorm, n_groupnorm, dfl_changed)."""
    import torch.nn as nn

    sd0, _ = load_state(init_weights)
    sd1, model1 = load_state(final_weights)
    backbone_changed, neck_head_changed, dfl_changed = diff_state_dicts(sd0, sd1)
    n_bn = sum(1 for m in model1.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm))
    n_gn = sum(1 for m in model1.modules() if isinstance(m, nn.GroupNorm))
    return backbone_changed, neck_head_changed, n_bn, n_gn, dfl_changed
