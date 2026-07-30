"""Grad-CAM++ for YOLOv11 detection (Chattopadhay et al., 2018, adapted to a
one-stage detector head instead of a single-logit classifier).

Hooks the last shared neck block (default: layer 22, the C3k2 that feeds the
Detect head's deepest scale) and explains a single detection by backpropagating
its raw (pre-sigmoid) class score.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class YOLOGradCAMPlusPlus:
    def __init__(self, detection_model: torch.nn.Module, target_layer_idx: int = 22):
        """`detection_model` is the raw nn.Module (`ultralytics.YOLO(...).model`),
        not the high-level YOLO wrapper.
        """
        self.model = detection_model
        self.target_layer = detection_model.model[target_layer_idx]
        self._activations = None
        self._gradients = None
        self._fwd_handle = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self._activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0]

    def remove_hooks(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def _pick_anchor(self, scores: torch.Tensor, class_id: int, anchor_idx: int | None):
        """scores: [nc, num_anchors] raw logits for one image. Returns an anchor index."""
        if anchor_idx is not None:
            return anchor_idx
        return int(torch.argmax(scores[class_id]).item())

    def generate(
        self,
        image_tensor: torch.Tensor,
        class_id: int,
        anchor_idx: int | None = None,
        output_size: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, float, int, np.ndarray]:
        """Runs one forward+backward pass and returns (cam, raw_score, anchor_idx,
        decoded_box_xywh).

        `image_tensor`: [1, 3, H, W], already preprocessed (0-1 float), NOT under
        torch.no_grad(). `cam` is a float32 array in [0, 1] resized to
        `output_size` (defaults to the input image's H, W). `decoded_box_xywh` is
        the model's own decoded box (center_x, center_y, w, h, in the SAME pixel
        scale as `image_tensor`'s H/W) for the anchor actually explained -- read
        from `_y` (Ultralytics' `Detect._inference()` output: `cat((decoded_boxes,
        scores.sigmoid()), dim=1)`, confirmed via `Detect._get_decode_boxes`/
        `decode_bboxes(..., xywh=True)`), NOT re-derived here. This anchor may or
        may not correspond to a correct detection of the queried class_id/GT box --
        callers must check that separately (e.g. via IoU against the GT box), never
        assume it from the mere existence of this returned box.
        """
        if image_tensor.dim() != 4 or image_tensor.shape[0] != 1:
            raise ValueError("image_tensor must be a single-image batch [1, 3, H, W]")

        self.model.zero_grad(set_to_none=True)
        was_training = self.model.training
        self.model.eval()

        y, preds = self.model(image_tensor)
        raw_scores = preds["scores"][0]  # [nc, num_anchors], pre-sigmoid

        anchor_idx = self._pick_anchor(raw_scores, class_id, anchor_idx)
        score = raw_scores[class_id, anchor_idx]
        decoded_box_xywh = y[0, :4, anchor_idx].detach().cpu().numpy()  # [cx, cy, w, h], pixel scale
        score.backward(retain_graph=False)

        if was_training:
            self.model.train()

        A = self._activations[0]  # [C, h, w]
        G = self._gradients[0]    # [C, h, w]

        g2 = G.pow(2)
        g3 = G.pow(3)
        denom = 2 * g2 + (A * g3).sum(dim=(1, 2), keepdim=True)
        denom = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha = g2 / denom

        weights = (alpha * F.relu(G)).sum(dim=(1, 2))  # [C]
        cam = F.relu((weights[:, None, None] * A).sum(dim=0))  # [h, w]

        cam = cam.detach().cpu().numpy()
        cam_max = cam.max()
        cam = cam / cam_max if cam_max > 0 else cam

        h, w = output_size if output_size is not None else image_tensor.shape[-2:]
        cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)

        return cam, float(score.detach().item()), anchor_idx, decoded_box_xywh


def overlay_heatmap(image_bgr_uint8: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blends a [0,1] cam onto a uint8 BGR image for visualization (Fig. 4.5/4.7 style)."""
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(heatmap, alpha, image_bgr_uint8, 1 - alpha, 0)
