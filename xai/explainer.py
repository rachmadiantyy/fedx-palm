"""
Explainable AI (XAI) Module for FedX-PALM.

Provides model interpretability for YOLOv11 predictions using:
- Grad-CAM (Gradient-weighted Class Activation Mapping)
- SHAP (SHapley Additive exPlanations)
- Feature importance analysis
- Attention visualization
- Privacy-aware explanations (compatible with DP)
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


class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) for YOLOv11.
    
    Generates visual explanations highlighting image regions
    that are most important for the model's predictions.
    """

    def __init__(self, model, target_layer: Optional[str] = None):
        """
        Args:
            model: YOLOv11 model (ultralytics YOLO instance or PyTorch model)
            target_layer: Name of the target layer for Grad-CAM.
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
            logger.warning("Could not find target layer for Grad-CAM")
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
        Generate Grad-CAM explanation for an image.
        
        Args:
            image: Input image (HWC, BGR format from OpenCV)
            target_class: Target class index. If None, uses top prediction.
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
            logger.warning("Could not compute target score for Grad-CAM")
            return Explanation(method="grad-cam", image=image)

        # Backward pass
        target_score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            logger.warning("Gradients or activations not captured")
            return Explanation(method="grad-cam", image=image)

        # Compute Grad-CAM
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        # Resize to input size
        heatmap = cv2.resize(cam, (image.shape[1], image.shape[0]))

        # Create colored heatmap
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
        )

        # Create overlay
        overlay = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)

        return Explanation(
            method="grad-cam",
            image=image,
            heatmap=heatmap,
            overlay=overlay,
            scores={"max_activation": float(cam.max())},
            metadata={
                "target_class": target_class,
                "target_box_idx": target_box_idx,
                "cam_shape": cam.shape
            }
        )

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
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
                # YOLOv11 typically outputs multiple heads
                # Use the first detection head
                pred = output[0] if len(output) > 0 else output
            else:
                pred = output

            if pred.dim() >= 3:
                # Shape: [batch, num_predictions, num_attributes]
                if target_class is not None:
                    # Get class scores (typically after box coordinates)
                    class_scores = pred[0, :, 5 + target_class] if pred.shape[-1] > 5 else pred[0, :, target_class]
                    return class_scores.max()
                else:
                    # Use objectness/confidence score
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


class SHAPExplainer:
    """
    SHAP-based explanations for YOLOv11 predictions.
    
    Uses a simplified kernel SHAP approach suitable for
    object detection models.
    """

    def __init__(
        self,
        model,
        num_samples: int = 100,
        segment_size: int = 16
    ):
        """
        Args:
            model: YOLOv11 model wrapper
            num_samples: Number of perturbation samples
            segment_size: Size of image segments for perturbation
        """
        self.model = model
        self.num_samples = num_samples
        self.segment_size = segment_size

    def generate(
        self,
        image: np.ndarray,
        target_class: Optional[int] = None
    ) -> Explanation:
        """
        Generate SHAP explanation for an image.
        
        Uses image segmentation and occlusion-based perturbation
        to approximate Shapley values.
        
        Args:
            image: Input image (HWC format)
            target_class: Target class index
            
        Returns:
            Explanation with SHAP values visualization
        """
        h, w = image.shape[:2]
        seg_h = h // self.segment_size
        seg_w = w // self.segment_size
        num_segments = seg_h * seg_w

        # Generate segment masks
        segments = self._create_segments(image)

        # Get baseline prediction
        baseline_score = self._get_prediction_score(image, target_class)

        # Compute SHAP values through perturbation
        shap_values = np.zeros(num_segments)
        coalition_matrix = np.random.binomial(1, 0.5, size=(self.num_samples, num_segments))

        for i in range(self.num_samples):
            # Create perturbed image
            mask = coalition_matrix[i]
            perturbed = self._apply_mask(image, segments, mask)

            # Get prediction for perturbed image
            score = self._get_prediction_score(perturbed, target_class)

            # Accumulate contributions
            for j in range(num_segments):
                if mask[j] == 1:
                    shap_values[j] += score - baseline_score * (1 - mask.mean())

        # Normalize
        shap_values /= max(self.num_samples, 1)

        # Create SHAP heatmap
        heatmap = self._shap_to_heatmap(shap_values, segments, image.shape[:2])

        # Normalize heatmap
        if np.abs(heatmap).max() > 0:
            heatmap = heatmap / np.abs(heatmap).max()

        # Create visualization
        heatmap_viz = self._colorize_shap(heatmap)
        overlay = cv2.addWeighted(image, 0.6, heatmap_viz, 0.4, 0)

        return Explanation(
            method="shap",
            image=image,
            heatmap=heatmap,
            overlay=overlay,
            scores={
                "baseline_score": float(baseline_score),
                "mean_shap": float(np.mean(shap_values)),
                "max_shap": float(np.max(shap_values)),
                "min_shap": float(np.min(shap_values))
            },
            metadata={
                "num_samples": self.num_samples,
                "num_segments": num_segments,
                "segment_size": self.segment_size
            }
        )

    def _create_segments(self, image: np.ndarray) -> np.ndarray:
        """Create grid-based segments for the image."""
        h, w = image.shape[:2]
        segments = np.zeros((h, w), dtype=np.int32)

        seg_h = h // self.segment_size
        seg_w = w // self.segment_size

        for i in range(self.segment_size):
            for j in range(self.segment_size):
                y_start = i * seg_h
                y_end = (i + 1) * seg_h if i < self.segment_size - 1 else h
                x_start = j * seg_w
                x_end = (j + 1) * seg_w if j < self.segment_size - 1 else w
                segments[y_start:y_end, x_start:x_end] = i * self.segment_size + j

        return segments

    def _apply_mask(
        self, image: np.ndarray, segments: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Apply segment mask to image (occlude segments where mask=0)."""
        perturbed = image.copy()
        for seg_idx in range(len(mask)):
            if mask[seg_idx] == 0:
                perturbed[segments == seg_idx] = 128  # Gray occlusion
        return perturbed

    def _get_prediction_score(self, image: np.ndarray, target_class: Optional[int]) -> float:
        """Get model prediction score for an image."""
        try:
            results = self.model.predict(source=image, verbose=False)
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    if target_class is not None:
                        # Filter by class
                        cls_mask = result.boxes.cls == target_class
                        if cls_mask.any():
                            return float(result.boxes.conf[cls_mask].max())
                    return float(result.boxes.conf.max())
        except Exception as e:
            logger.debug(f"Prediction error in SHAP: {e}")

        return 0.0

    def _shap_to_heatmap(
        self, shap_values: np.ndarray, segments: np.ndarray, shape: Tuple
    ) -> np.ndarray:
        """Convert SHAP values to a pixel-level heatmap."""
        heatmap = np.zeros(shape, dtype=np.float32)
        for seg_idx, shap_val in enumerate(shap_values):
            heatmap[segments == seg_idx] = shap_val
        return heatmap

    def _colorize_shap(self, heatmap: np.ndarray) -> np.ndarray:
        """Colorize SHAP heatmap (red=positive, blue=negative)."""
        h, w = heatmap.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)

        # Positive values -> Red
        pos_mask = heatmap > 0
        colored[pos_mask, 2] = (heatmap[pos_mask] * 255).astype(np.uint8)

        # Negative values -> Blue
        neg_mask = heatmap < 0
        colored[neg_mask, 0] = (np.abs(heatmap[neg_mask]) * 255).astype(np.uint8)

        return colored


class FeatureImportanceAnalyzer:
    """
    Analyze feature importance across federated learning rounds.
    
    Tracks which features/layers contribute most to model predictions
    and how they change during federated training.
    """

    def __init__(self, model):
        """
        Args:
            model: YOLOv11FederatedWrapper instance
        """
        self.model = model
        self.importance_history: List[Dict] = []

    def compute_layer_importance(self) -> Dict[str, float]:
        """
        Compute importance score for each layer based on gradient magnitude.
        
        Returns:
            Dictionary mapping layer names to importance scores
        """
        pytorch_model = self.model.model.model if hasattr(self.model.model, 'model') else self.model.model
        importance = {}

        for name, param in pytorch_model.named_parameters():
            if param.grad is not None:
                importance[name] = float(torch.norm(param.grad).item())
            elif param.requires_grad:
                importance[name] = float(torch.norm(param.data).item())

        # Normalize
        total = sum(importance.values()) + 1e-8
        importance = {k: v / total for k, v in importance.items()}

        return importance

    def compute_update_importance(
        self, model_update: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Compute importance based on model update magnitudes.
        
        Useful for understanding which parameters changed most during
        federated learning.
        
        Args:
            model_update: Dictionary of parameter updates
            
        Returns:
            Layer importance scores
        """
        importance = {}
        for name, update in model_update.items():
            importance[name] = float(torch.norm(update).item())

        # Normalize
        total = sum(importance.values()) + 1e-8
        importance = {k: v / total for k, v in importance.items()}

        return importance

    def track_round(self, round_num: int, importance: Dict[str, float]):
        """Track feature importance for a training round."""
        self.importance_history.append({
            "round": round_num,
            "importance": importance,
            "top_5": dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5])
        })

    def get_importance_trends(self) -> Dict[str, List[float]]:
        """Get importance trends across rounds for each layer."""
        if not self.importance_history:
            return {}

        # Collect all layer names
        all_layers = set()
        for record in self.importance_history:
            all_layers.update(record["importance"].keys())

        # Build trends
        trends = {}
        for layer in all_layers:
            trends[layer] = [
                record["importance"].get(layer, 0.0)
                for record in self.importance_history
            ]

        return trends


class PrivacyAwareExplainer:
    """
    Privacy-Aware Explainer that provides explanations
    compatible with Differential Privacy constraints.
    
    Ensures that explanations don't leak private information
    by adding noise to explanation outputs.
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
        self.grad_cam = GradCAM(model)
        self.shap_explainer = SHAPExplainer(model)

    def explain_with_privacy(
        self,
        image: np.ndarray,
        method: str = "grad-cam",
        target_class: Optional[int] = None
    ) -> Explanation:
        """
        Generate privacy-preserving explanation.
        
        Adds calibrated noise to the explanation heatmap to prevent
        inference attacks on the training data.
        
        Args:
            image: Input image
            method: Explanation method ('grad-cam' or 'shap')
            target_class: Target class for explanation
            
        Returns:
            Privacy-preserving explanation
        """
        # Generate base explanation
        if method == "grad-cam":
            explanation = self.grad_cam.generate(image, target_class)
        elif method == "shap":
            explanation = self.shap_explainer.generate(image, target_class)
        else:
            raise ValueError(f"Unknown method: {method}")

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
        self.grad_cam.cleanup()


class XAIReportGenerator:
    """
    Generate comprehensive XAI reports for federated learning models.
    
    Combines multiple explanation methods to provide a complete
    picture of model behavior.
    """

    def __init__(self, model, privacy_epsilon: float = 1.0):
        """
        Args:
            model: YOLOv11 model wrapper
            privacy_epsilon: Privacy budget for explanations
        """
        self.model = model
        self.explainer = PrivacyAwareExplainer(model, epsilon=privacy_epsilon)
        self.feature_analyzer = FeatureImportanceAnalyzer(model)

    def generate_report(
        self,
        image: np.ndarray,
        target_class: Optional[int] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive XAI report.
        
        Args:
            image: Input image for explanation
            target_class: Target class for explanations
            output_dir: Directory to save report artifacts
            
        Returns:
            Report dictionary with explanations and analysis
        """
        report = {
            "timestamp": str(np.datetime64('now')),
            "model_info": {},
            "explanations": {},
            "feature_importance": {},
        }

        # Model info
        if hasattr(self.model, 'get_model_size'):
            report["model_info"] = self.model.get_model_size()

        # Grad-CAM explanation
        try:
            grad_cam_exp = self.explainer.explain_with_privacy(
                image, method="grad-cam", target_class=target_class
            )
            report["explanations"]["grad_cam"] = {
                "scores": grad_cam_exp.scores,
                "metadata": grad_cam_exp.metadata
            }

            if output_dir and grad_cam_exp.overlay is not None:
                out_path = Path(output_dir)
                out_path.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_path / "grad_cam_overlay.jpg"), grad_cam_exp.overlay)

        except Exception as e:
            report["explanations"]["grad_cam"] = {"error": str(e)}
            logger.warning(f"Grad-CAM generation failed: {e}")

        # SHAP explanation
        try:
            shap_exp = self.explainer.explain_with_privacy(
                image, method="shap", target_class=target_class
            )
            report["explanations"]["shap"] = {
                "scores": shap_exp.scores,
                "metadata": shap_exp.metadata
            }

            if output_dir and shap_exp.overlay is not None:
                cv2.imwrite(str(Path(output_dir) / "shap_overlay.jpg"), shap_exp.overlay)

        except Exception as e:
            report["explanations"]["shap"] = {"error": str(e)}
            logger.warning(f"SHAP generation failed: {e}")

        # Feature importance
        try:
            importance = self.feature_analyzer.compute_layer_importance()
            top_features = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])
            report["feature_importance"] = {
                "top_10_layers": top_features,
                "total_layers_analyzed": len(importance)
            }
        except Exception as e:
            report["feature_importance"] = {"error": str(e)}

        # Save report as JSON
        if output_dir:
            import json
            report_path = Path(output_dir) / "xai_report.json"
            # Convert non-serializable values
            serializable_report = json.loads(json.dumps(report, default=str))
            with open(report_path, "w") as f:
                json.dump(serializable_report, f, indent=2)

        return report

    def cleanup(self):
        """Clean up resources."""
        self.explainer.cleanup()
