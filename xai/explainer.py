"""
Explainable AI (XAI) Module for FedX-PALM.

Implements Grad-CAM++ (NOT standard Grad-CAM) as specified in thesis Section 2.6.1.
Provides model interpretability with quantitative validation metrics:
- Grad-CAM++ (Gradient-weighted Class Activation Mapping Plus Plus)
- Average Drop metric (Section 2.6.2)
- Focus Retention Rate (FRR) metric (Section 2.6.3)
- Privacy-aware explanations (compatible with DP)

Reference formulas from thesis:
- Grad-CAM++ weights: α_k^c = (1/Z) * Σ_i Σ_j (∂y^c / ∂A^k_ij)  [Eq. 2.6]
- Average Drop: AD = (1/N) * Σ max(0, Y_i^c - O_i^c) / Y_i^c * 100%  [Eq. 2.7]
- FRR: Σ_{p∈ROI} I(p) / Σ_{p∈Image} I(p)  [Eq. 2.8]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import cv2
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Explanation:
    """Container for model explanation results."""
    method: str
    image: Optional[np.ndarray] = None
    heatmap: Optional[np.ndarray] = None
    overlay: Optional[np.ndarray] = None
    scores: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None


class GradCAMPlusPlus:
    """
    Grad-CAM++ (Gradient-weighted Class Activation Mapping Plus Plus).

    As defined in thesis Section 2.6.1:
    - Uses higher-order gradients for improved localization
    - Produces pixel-level importance weights α_k^c
    - Generates heatmap L^c = ReLU(Σ_k α_k^c * A^k)

    Grad-CAM++ improves upon standard Grad-CAM by using a weighted combination
    of positive partial derivatives of the score with respect to feature maps,
    providing better localization for multiple instances of the same class.
    """

    def __init__(self, model, target_layer: Optional[str] = None):
        """
        Args:
            model: YOLOv11 model (ultralytics YOLO instance or PyTorch model)
            target_layer: Name of the target layer for Grad-CAM++.
                         If None, uses the last convolutional layer.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._hooks = []
        self._setup_hooks()

    def _setup_hooks(self):
        """Register forward and backward hooks on the target layer."""
        pytorch_model = self._get_pytorch_model()

        if self.target_layer:
            target = self._find_layer(pytorch_model, self.target_layer)
        else:
            target = self._find_last_conv_layer(pytorch_model)

        if target is None:
            logger.warning("Could not find target layer for Grad-CAM++")
            return

        # Forward hook to capture activations
        def forward_hook(module, input, output):
            self.activations = output.detach()

        # Backward hook to capture gradients
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._hooks.append(target.register_forward_hook(forward_hook))
        self._hooks.append(target.register_full_backward_hook(backward_hook))

    def _get_pytorch_model(self):
        """Extract PyTorch model from YOLO wrapper."""
        if hasattr(self.model, 'model'):
            if hasattr(self.model.model, 'model'):
                return self.model.model.model
            return self.model.model
        return self.model

    def _find_layer(self, model, layer_name: str):
        """Find a layer by name in the model."""
        for name, module in model.named_modules():
            if name == layer_name:
                return module
        return None

    def _find_last_conv_layer(self, model):
        """Find the last convolutional layer in the model."""
        last_conv = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        return last_conv

    def generate(
        self,
        image: np.ndarray,
        target_class: Optional[int] = None,
        target_box_idx: int = 0
    ) -> Explanation:
        """
        Generate Grad-CAM++ explanation for an image.

        Implements the Grad-CAM++ algorithm (thesis Eq. 2.6):
        α_k^c = (1/Z) * Σ_i Σ_j (∂y^c / ∂A^k_ij)

        With the Grad-CAM++ improvement using second and third order gradients
        for better weighting of positive contributions.

        Args:
            image: Input image (HWC, BGR format from OpenCV)
            target_class: Target class index (0-5). If None, uses top prediction.
            target_box_idx: Index of target detection box

        Returns:
            Explanation object with heatmap and overlay
        """
        pytorch_model = self._get_pytorch_model()
        pytorch_model.eval()

        # Preprocess image
        input_tensor = self._preprocess_image(image)
        input_tensor.requires_grad_(True)

        # Forward pass
        pytorch_model.zero_grad()
        output = pytorch_model(input_tensor)

        # Get target score for backpropagation
        target_score = self._get_target_score(output, target_class, target_box_idx)

        if target_score is None:
            logger.warning("Could not compute target score for Grad-CAM++")
            return Explanation(method="grad-cam++", image=image)

        # Backward pass
        target_score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            logger.warning("Gradients or activations not captured")
            return Explanation(method="grad-cam++", image=image)

        # === Grad-CAM++ specific computation ===
        # Standard Grad-CAM uses: weights = global_avg_pool(gradients)
        # Grad-CAM++ uses higher-order derivatives for better weighting

        gradients = self.gradients  # [B, C, H, W]
        activations = self.activations  # [B, C, H, W]

        # Compute Grad-CAM++ weights using positive partial derivatives
        # α_k^c = Σ_i Σ_j (α_ij^kc * relu(∂y^c/∂A^k_ij))
        # where α_ij^kc accounts for second and third order gradients

        # Second derivative (approximation)
        grad_2 = gradients ** 2
        # Third derivative (approximation)
        grad_3 = gradients ** 3

        # Compute spatial importance weights (Grad-CAM++ formula)
        # Denominator: 2 * grad^2 + sum(A^k * grad^3)
        sum_activations = torch.sum(activations, dim=(2, 3), keepdim=True)
        denominator = 2.0 * grad_2 + sum_activations * grad_3 + 1e-8

        # Alpha weights (per-pixel importance)
        alpha = grad_2 / denominator
        alpha = torch.where(
            gradients != 0,
            alpha,
            torch.zeros_like(alpha)
        )

        # Weighted combination with ReLU of gradients
        weights = torch.sum(alpha * F.relu(gradients), dim=(2, 3), keepdim=True)

        # Generate CAM: L^c = ReLU(Σ_k w_k * A^k)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        # Resize to input image size
        heatmap = cv2.resize(cam, (image.shape[1], image.shape[0]))

        # Create colored heatmap
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
        )

        # Create overlay
        overlay = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)

        return Explanation(
            method="grad-cam++",
            image=image,
            heatmap=heatmap,
            overlay=overlay,
            scores={"max_activation": float(cam.max())},
            metadata={
                "target_class": target_class,
                "target_box_idx": target_box_idx,
                "cam_shape": cam.shape,
                "algorithm": "Grad-CAM++ (higher-order gradients)"
            }
        )

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input (640x640 as per thesis Table 3.5)."""
        img = cv2.resize(image, (640, 640))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        tensor = torch.from_numpy(img).unsqueeze(0)

        if torch.cuda.is_available():
            tensor = tensor.cuda()

        return tensor

    def _get_target_score(
        self, output, target_class: Optional[int], target_box_idx: int
    ) -> Optional[torch.Tensor]:
        """Extract target score from model output for backpropagation."""
        try:
            if isinstance(output, (list, tuple)):
                pred = output[0] if len(output) > 0 else output
            else:
                pred = output

            if pred.dim() >= 3:
                if target_class is not None:
                    class_scores = pred[0, :, 5 + target_class] if pred.shape[-1] > 5 else pred[0, :, target_class]
                    return class_scores.max()
                else:
                    if pred.shape[-1] > 4:
                        conf_scores = pred[0, :, 4]
                        return conf_scores.max()
                    return pred[0].sum()
            else:
                return pred.sum()

        except Exception as e:
            logger.warning(f"Error extracting target score: {e}")
            if isinstance(output, torch.Tensor):
                return output.sum()
            return None

    def cleanup(self):
        """Remove hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


class AverageDrop:
    """
    Average Drop metric for quantitative XAI validation.

    As defined in thesis Section 2.6.2 (Eq. 2.7):
    AD = (1/N) * Σ_{i=1}^{N} max(0, Y_i^c - O_i^c) / Y_i^c * 100%

    Where:
    - N: number of test samples
    - Y_i^c: confidence score on original image for class c
    - O_i^c: confidence score after masking important regions

    Higher Average Drop → XAI explanation is more accurate/reliable
    (removing highlighted regions causes bigger confidence drop)
    """

    def __init__(self, model):
        """
        Args:
            model: YOLOv11 model wrapper for inference
        """
        self.model = model

    def compute(
        self,
        images: List[np.ndarray],
        heatmaps: List[np.ndarray],
        target_class: Optional[int] = None,
        mask_threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute Average Drop metric.

        Steps:
        1. Get confidence score Y_i^c on original image
        2. Mask the important regions (where heatmap > threshold)
        3. Get confidence score O_i^c on masked image
        4. Compute AD = (1/N) * Σ max(0, Y-O)/Y * 100%

        Args:
            images: List of input images
            heatmaps: List of corresponding Grad-CAM++ heatmaps
            target_class: Target class for confidence computation
            mask_threshold: Threshold to determine "important" regions

        Returns:
            Dict with average_drop (%), increase_in_entropy, prediction_flip_rate
        """
        drops = []
        entropy_increases = []
        flips = 0
        total = 0

        for image, heatmap in zip(images, heatmaps):
            if heatmap is None:
                continue

            # Get original confidence Y_i^c
            y_original = self._get_confidence(image, target_class)
            if y_original <= 0:
                continue

            # Create mask: important regions where heatmap > threshold
            mask = (heatmap >= mask_threshold).astype(np.float32)

            # Mask the important regions (set to mean pixel value)
            masked_image = image.copy()
            mean_pixel = image.mean(axis=(0, 1)).astype(np.uint8)
            for c in range(3):
                masked_image[:, :, c] = np.where(
                    mask > 0,
                    mean_pixel[c],
                    image[:, :, c]
                )

            # Get masked confidence O_i^c
            o_masked = self._get_confidence(masked_image, target_class)

            # Compute drop: max(0, Y - O) / Y
            drop = max(0.0, y_original - o_masked) / y_original
            drops.append(drop)

            # Check for prediction flip
            orig_pred = self._get_prediction(image)
            masked_pred = self._get_prediction(masked_image)
            if orig_pred != masked_pred:
                flips += 1
            total += 1

            # Entropy increase approximation
            if o_masked > 0:
                entropy_increase = -np.log2(o_masked + 1e-8) - (-np.log2(y_original + 1e-8))
                entropy_increases.append(max(0, entropy_increase))

        # Compute final metrics
        average_drop = np.mean(drops) * 100 if drops else 0.0
        avg_entropy_increase = np.mean(entropy_increases) if entropy_increases else 0.0
        flip_rate = (flips / max(total, 1)) * 100

        return {
            "average_drop_pct": float(average_drop),
            "increase_in_entropy_bits": float(avg_entropy_increase),
            "prediction_flip_rate_pct": float(flip_rate),
            "num_samples": len(drops),
            "reliability": self._assess_reliability(average_drop)
        }

    def _get_confidence(self, image: np.ndarray, target_class: Optional[int]) -> float:
        """Get model confidence score for an image."""
        try:
            results = self.model.predict(source=image, verbose=False)
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    if target_class is not None:
                        cls_mask = result.boxes.cls == target_class
                        if cls_mask.any():
                            return float(result.boxes.conf[cls_mask].max())
                    return float(result.boxes.conf.max())
        except Exception as e:
            logger.debug(f"Confidence computation error: {e}")
        return 0.0

    def _get_prediction(self, image: np.ndarray) -> int:
        """Get top predicted class for an image."""
        try:
            results = self.model.predict(source=image, verbose=False)
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    top_idx = result.boxes.conf.argmax()
                    return int(result.boxes.cls[top_idx])
        except Exception:
            pass
        return -1

    @staticmethod
    def _assess_reliability(average_drop: float) -> str:
        """Assess XAI reliability based on Average Drop value (thesis Table 4.11)."""
        if average_drop >= 30:
            return "Excellent"
        elif average_drop >= 25:
            return "Good"
        elif average_drop >= 20:
            return "Acceptable"
        else:
            return "Questionable"


class FocusRetentionRate:
    """
    Focus Retention Rate (FRR) metric.

    As defined in thesis Section 2.6.3 (Eq. 2.8):
    FRR = Σ_{p∈ROI} I(p) / Σ_{p∈Image} I(p)

    Where:
    - I(p): heatmap intensity at pixel p
    - ROI: ground truth bounding box region of the palm fruit

    FRR close to 1.0 → model focuses on object (good)
    FRR close to 0.0 → model focuses on background (bad)

    Thesis results (real, Bab 4 — Tabel 4.11):
    - Baseline (ε=∞): FRR = 0.962
    - DP (ε=8.0 / 4.0 / 1.0): NaN — model collapse, no valid prediction
    """

    def compute(
        self,
        heatmap: np.ndarray,
        bounding_boxes: List[List[int]],
        image_shape: Tuple[int, int] = None
    ) -> float:
        """
        Compute Focus Retention Rate for a single image.

        Args:
            heatmap: Grad-CAM++ heatmap (H x W, values 0-1)
            bounding_boxes: List of [x1, y1, x2, y2] bounding boxes (ROI)
            image_shape: (height, width) of the original image

        Returns:
            FRR value between 0 and 1
        """
        if heatmap is None or len(bounding_boxes) == 0:
            return 0.0

        h, w = heatmap.shape[:2]

        # Create ROI mask from bounding boxes
        roi_mask = np.zeros((h, w), dtype=bool)
        for box in bounding_boxes:
            x1, y1, x2, y2 = [int(coord) for coord in box]
            # Clamp coordinates
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            roi_mask[y1:y2, x1:x2] = True

        # Compute FRR = Σ_{p∈ROI} I(p) / Σ_{p∈Image} I(p)
        total_intensity = heatmap.sum()
        if total_intensity <= 0:
            return 0.0

        roi_intensity = heatmap[roi_mask].sum()
        frr = float(roi_intensity / total_intensity)

        return frr

    def compute_batch(
        self,
        heatmaps: List[np.ndarray],
        bounding_boxes_list: List[List[List[int]]]
    ) -> Dict[str, float]:
        """
        Compute FRR for a batch of images.

        Args:
            heatmaps: List of heatmaps
            bounding_boxes_list: List of bounding box lists per image

        Returns:
            Dict with mean FRR and per-sample FRR values
        """
        frr_values = []
        for heatmap, boxes in zip(heatmaps, bounding_boxes_list):
            if heatmap is not None and boxes:
                frr = self.compute(heatmap, boxes)
                frr_values.append(frr)

        if not frr_values:
            return {"mean_frr": 0.0, "frr_values": [], "num_samples": 0}

        return {
            "mean_frr": float(np.mean(frr_values)),
            "std_frr": float(np.std(frr_values)),
            "min_frr": float(np.min(frr_values)),
            "max_frr": float(np.max(frr_values)),
            "frr_values": frr_values,
            "num_samples": len(frr_values)
        }


class PrivacyAwareExplainer:
    """
    Privacy-Aware Explainer that provides Grad-CAM++ explanations
    compatible with Differential Privacy constraints.

    Ensures that explanations don't leak private information
    by adding Laplacian noise to explanation outputs.
    """

    def __init__(
        self,
        model,
        epsilon: float = 1.0,
        explanation_sensitivity: float = 1.0
    ):
        """
        Args:
            model: YOLOv11 model wrapper
            epsilon: Privacy budget for explanations
            explanation_sensitivity: Sensitivity of explanation outputs
        """
        self.model = model
        self.epsilon = epsilon
        self.sensitivity = explanation_sensitivity
        self.grad_cam_pp = GradCAMPlusPlus(model)

    def explain_with_privacy(
        self,
        image: np.ndarray,
        target_class: Optional[int] = None
    ) -> Explanation:
        """
        Generate privacy-preserving Grad-CAM++ explanation.

        Adds calibrated Laplacian noise to the heatmap to prevent
        inference attacks on the training data.

        Args:
            image: Input image
            target_class: Target class for explanation (0-5)

        Returns:
            Privacy-preserving explanation
        """
        # Generate Grad-CAM++ explanation
        explanation = self.grad_cam_pp.generate(image, target_class)

        # Add noise to heatmap for privacy
        if explanation.heatmap is not None:
            noisy_heatmap = self._add_privacy_noise(explanation.heatmap)
            explanation.heatmap = noisy_heatmap

            # Regenerate overlay with noisy heatmap
            if explanation.image is not None:
                heatmap_uint8 = (np.clip(noisy_heatmap, 0, 1) * 255).astype(np.uint8)
                heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                explanation.overlay = cv2.addWeighted(
                    explanation.image, 0.6, heatmap_colored, 0.4, 0
                )

        # Update metadata
        if explanation.metadata is None:
            explanation.metadata = {}
        explanation.metadata["privacy_epsilon"] = self.epsilon
        explanation.metadata["privacy_noise_added"] = True
        explanation.metadata["method"] = "Grad-CAM++ (privacy-aware)"

        return explanation

    def _add_privacy_noise(self, heatmap: np.ndarray) -> np.ndarray:
        """Add Laplacian noise to heatmap for differential privacy."""
        noise_scale = self.sensitivity / self.epsilon
        noise = np.random.laplace(0, noise_scale, heatmap.shape)
        noisy_heatmap = heatmap + noise

        # Clip to valid range
        noisy_heatmap = np.clip(noisy_heatmap, 0, 1)
        return noisy_heatmap.astype(np.float32)

    def cleanup(self):
        """Clean up resources."""
        self.grad_cam_pp.cleanup()


class XAIReportGenerator:
    """
    Generate comprehensive XAI reports for federated learning models.

    Combines Grad-CAM++, Average Drop, and Focus Retention Rate
    to provide a complete picture of model interpretability as
    defined in thesis Sections 3.7.3 and 4.5-4.6.
    """

    def __init__(self, model, privacy_epsilon: float = 1.0):
        """
        Args:
            model: YOLOv11 model wrapper
            privacy_epsilon: Privacy budget for explanations
        """
        self.model = model
        self.explainer = PrivacyAwareExplainer(model, epsilon=privacy_epsilon)
        self.average_drop = AverageDrop(model)
        self.frr = FocusRetentionRate()

    def generate_report(
        self,
        images: List[np.ndarray],
        bounding_boxes_list: Optional[List[List[List[int]]]] = None,
        target_class: Optional[int] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive XAI report with Grad-CAM++, Average Drop, and FRR.

        Args:
            images: List of input images for explanation
            bounding_boxes_list: Ground truth bounding boxes per image (for FRR)
            target_class: Target class for explanations (0-5)
            output_dir: Directory to save report artifacts

        Returns:
            Report dictionary with all XAI metrics
        """
        report = {
            "timestamp": str(np.datetime64('now')),
            "method": "Grad-CAM++",
            "num_images": len(images),
            "target_class": target_class,
            "model_info": {},
            "grad_cam_pp": {},
            "average_drop": {},
            "focus_retention_rate": {},
        }

        # Model info
        if hasattr(self.model, 'get_model_size'):
            report["model_info"] = self.model.get_model_size()

        # Generate Grad-CAM++ heatmaps for all images
        heatmaps = []
        explanations = []
        for i, image in enumerate(images):
            try:
                exp = self.explainer.explain_with_privacy(image, target_class)
                explanations.append(exp)
                heatmaps.append(exp.heatmap)

                # Save overlays
                if output_dir and exp.overlay is not None:
                    out_path = Path(output_dir)
                    out_path.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(
                        str(out_path / f"grad_cam_pp_overlay_{i}.jpg"),
                        exp.overlay
                    )
            except Exception as e:
                logger.warning(f"Grad-CAM++ generation failed for image {i}: {e}")
                heatmaps.append(None)

        report["grad_cam_pp"]["num_generated"] = sum(1 for h in heatmaps if h is not None)

        # Compute Average Drop (thesis Eq. 2.7)
        try:
            valid_pairs = [
                (img, hm) for img, hm in zip(images, heatmaps) if hm is not None
            ]
            if valid_pairs:
                valid_images, valid_heatmaps = zip(*valid_pairs)
                ad_result = self.average_drop.compute(
                    list(valid_images), list(valid_heatmaps), target_class
                )
                report["average_drop"] = ad_result
                logger.info(
                    f"Average Drop: {ad_result['average_drop_pct']:.1f}% "
                    f"({ad_result['reliability']})"
                )
        except Exception as e:
            report["average_drop"] = {"error": str(e)}
            logger.warning(f"Average Drop computation failed: {e}")

        # Compute Focus Retention Rate (thesis Eq. 2.8)
        if bounding_boxes_list is not None:
            try:
                frr_result = self.frr.compute_batch(heatmaps, bounding_boxes_list)
                report["focus_retention_rate"] = frr_result
                logger.info(f"Focus Retention Rate: {frr_result['mean_frr']:.3f}")
            except Exception as e:
                report["focus_retention_rate"] = {"error": str(e)}
                logger.warning(f"FRR computation failed: {e}")
        else:
            report["focus_retention_rate"] = {
                "message": "No bounding boxes provided for FRR computation"
            }

        # Save report as JSON
        if output_dir:
            import json
            report_path = Path(output_dir) / "xai_report.json"
            serializable_report = json.loads(json.dumps(report, default=str))
            with open(report_path, "w") as f:
                json.dump(serializable_report, f, indent=2)
            logger.info(f"XAI report saved to: {report_path}")

        return report

    def generate_single_explanation(
        self,
        image: np.ndarray,
        target_class: Optional[int] = None,
        output_dir: Optional[str] = None
    ) -> Explanation:
        """Generate Grad-CAM++ explanation for a single image."""
        explanation = self.explainer.explain_with_privacy(image, target_class)

        if output_dir and explanation.overlay is not None:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path / "grad_cam_pp_overlay.jpg"), explanation.overlay)

        return explanation

    def cleanup(self):
        """Clean up resources."""
        self.explainer.cleanup()
