"""Detects which YOLO11 scale variant a loaded checkpoint corresponds to,
from its MEASURED total parameter count -- never trusted from a filename
alone (a checkpoint could be renamed, copied, or the wrong file passed by
mistake). Counts are for nc=6 after BatchNorm->GroupNorm conversion (which
does not change parameter count) and were obtained by actually constructing
each architecture and counting, not assumed from published specs.
"""
from __future__ import annotations

KNOWN_VARIANT_TOTAL_PARAMS = {
    "yolo11n": 2_591_010,
    "yolo11s": 9_430_114,
}


def detect_model_variant(total_params: int) -> str:
    for variant, expected in KNOWN_VARIANT_TOTAL_PARAMS.items():
        if total_params == expected:
            return variant
    return f"unknown_variant(total_params={total_params})"


def resolve_model_variant(requested: str, total_params: int) -> dict:
    """Resolves which model variant a run is actually using, treating an
    EXPLICIT caller-supplied identifier (a --model-variant/--arch CLI
    argument) as authoritative, and the measured total parameter count
    (detect_model_variant) as a fallback sanity-check only -- never the
    sole source of truth, since parameter count can change once the
    detection head is resized for a different nc.

    Returns {"requested": requested, "detected": detected, "mismatch": bool}
    so the caller can hard-stop or warn loudly when they disagree.
    """
    detected = detect_model_variant(total_params)
    return {"requested": requested, "detected": detected, "mismatch": detected != requested}
