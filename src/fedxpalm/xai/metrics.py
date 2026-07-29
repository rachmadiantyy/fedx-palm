"""Grad-CAM++ faithfulness metrics: Average Drop (occlusion-based) and Focus Retention Rate.

Average Drop asks: "if you hide the region the model's own explanation says
matters, does its confidence actually fall?" -- it should, if the explanation
is faithful (Chattopadhay et al., 2018, adapted here to *occlude* rather than
*preserve* the highlighted region, since we want to test whether that region
was informative, not whether the rest of the image is redundant).

Focus Retention Rate asks: "how much of the explanation's total activation
mass sits inside the object's own ground-truth box, vs. leaking into
background/context?" -- a model reasoning about fruit color/texture should
concentrate there, not on soil or foliage.
"""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def class_confidence(model: torch.nn.Module, image_tensor: torch.Tensor, class_id: int, anchor_idx: int) -> float:
    """Sigmoid confidence for `class_id` at a specific anchor, no grad needed."""
    was_training = model.training
    model.eval()
    _y, preds = model(image_tensor)
    conf = torch.sigmoid(preds["scores"][0, class_id, anchor_idx]).item()
    if was_training:
        model.train()
    return conf


def occlude_top_region(image_tensor: torch.Tensor, cam: np.ndarray, top_fraction: float = 0.2) -> torch.Tensor:
    """Replaces the top `top_fraction` highest-activation pixels of `cam` with
    the image's own per-channel mean (a neutral fill), returning a new tensor
    of the same shape as `image_tensor` ([1, 3, H, W]).
    """
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1]")
    flat = cam.flatten()
    k = max(1, int(round(top_fraction * flat.size)))
    threshold = np.partition(flat, -k)[-k]
    mask = (cam >= threshold).astype(np.float32)  # [H, W], 1 = occlude
    mask_t = torch.from_numpy(mask).to(image_tensor.device, image_tensor.dtype)[None, None]  # [1,1,H,W]

    fill = image_tensor.mean(dim=(2, 3), keepdim=True)  # per-channel mean, neutral "blank" fill
    return image_tensor * (1 - mask_t) + fill * mask_t


def average_drop(y_c: float, o_c: float) -> float:
    """AD = max(0, Y_c - O_c) / Y_c. 0 means occlusion didn't hurt confidence
    at all (unfaithful explanation); closer to 1 means the highlighted region
    really was where the model's evidence lived.
    """
    if y_c <= 0:
        return 0.0
    return max(0.0, y_c - o_c) / y_c


def focus_retention_rate(cam: np.ndarray, gt_box_xyxy: tuple[float, float, float, float]) -> float:
    """Fraction of the CAM's total activation mass that falls inside the
    ground-truth box (pixel coords, same frame as `cam`), vs. leaking outside.
    """
    h, w = cam.shape
    x1, y1, x2, y2 = gt_box_xyxy
    x1, x2 = sorted((max(0, int(round(x1))), min(w, int(round(x2)))))
    y1, y2 = sorted((max(0, int(round(y1))), min(h, int(round(y2)))))
    total = cam.sum()
    if total <= 0 or x2 <= x1 or y2 <= y1:
        return 0.0
    inside = cam[y1:y2, x1:x2].sum()
    return float(inside / total)
