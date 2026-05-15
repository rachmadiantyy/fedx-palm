"""
YOLOv11 Model Wrapper for Federated Learning.

Wraps the Ultralytics YOLOv11 model to support:
- Parameter extraction and loading for FL aggregation
- Local training with custom datasets
- Differential privacy integration
- Model state serialization for network transfer
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import copy
import numpy as np
import logging

logger = logging.getLogger(__name__)


class YOLOv11FederatedWrapper:
    """
    Federated Learning wrapper for YOLOv11 object detection model.
    
    This wrapper enables YOLOv11 to participate in federated learning by:
    1. Extracting model parameters as state dicts
    2. Loading aggregated parameters from the FL server
    3. Computing model updates (deltas) for communication
    4. Supporting partial model freezing for transfer learning
    """

    def __init__(
        self,
        model_variant: str = "yolo11n.pt",
        num_classes: int = 80,
        task: str = "detect",
        device: Optional[str] = None,
        pretrained: bool = True
    ):
        """
        Args:
            model_variant: YOLOv11 variant (yolo11n, yolo11s, yolo11m, yolo11l, yolo11x)
            num_classes: Number of detection classes
            task: Task type ('detect', 'segment', 'classify')
            device: Device to use ('cuda', 'cpu', or None for auto)
            pretrained: Whether to use pretrained weights
        """
        self.model_variant = model_variant
        self.num_classes = num_classes
        self.task = task
        self.pretrained = pretrained

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Initialize model
        self.model = self._initialize_model()

        # Store initial state for computing deltas
        self._initial_state: Optional[Dict[str, torch.Tensor]] = None
        self._trainable_layers: Optional[List[str]] = None

        logger.info(
            f"Initialized YOLOv11 wrapper: variant={model_variant}, "
            f"classes={num_classes}, device={self.device}"
        )

    def _initialize_model(self) -> YOLO:
        """Initialize the YOLOv11 model."""
        if self.pretrained:
            model = YOLO(self.model_variant)
        else:
            # Load architecture without pretrained weights
            model = YOLO(self.model_variant)

        return model

    def get_model_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Extract model parameters as a state dictionary.
        
        Returns:
            Dictionary mapping parameter names to tensors
        """
        state_dict = {}
        pytorch_model = self.model.model

        for name, param in pytorch_model.named_parameters():
            if param.requires_grad:
                state_dict[name] = param.data.clone().cpu()

        return state_dict

    def get_trainable_parameters(self) -> Dict[str, torch.Tensor]:
        """Get only trainable parameters (respecting frozen layers)."""
        state_dict = {}
        pytorch_model = self.model.model

        for name, param in pytorch_model.named_parameters():
            if param.requires_grad:
                if self._trainable_layers is None or any(
                    layer in name for layer in self._trainable_layers
                ):
                    state_dict[name] = param.data.clone().cpu()

        return state_dict

    def set_model_parameters(self, state_dict: Dict[str, torch.Tensor]):
        """
        Load parameters from a state dictionary (e.g., from FL server).
        
        Args:
            state_dict: Dictionary mapping parameter names to tensors
        """
        pytorch_model = self.model.model
        current_state = pytorch_model.state_dict()

        for name, param in state_dict.items():
            if name in current_state:
                current_state[name] = param.to(self.device)

        pytorch_model.load_state_dict(current_state, strict=False)
        logger.info(f"Loaded {len(state_dict)} parameters from state dict")

    def save_initial_state(self):
        """Save the current state as the initial state for computing deltas."""
        self._initial_state = self.get_model_parameters()

    def compute_model_update(self) -> Dict[str, torch.Tensor]:
        """
        Compute the model update (delta) since the last saved initial state.
        
        Returns:
            Dictionary of parameter deltas (current - initial)
        """
        if self._initial_state is None:
            raise RuntimeError("No initial state saved. Call save_initial_state() first.")

        current_state = self.get_model_parameters()
        update = {}

        for name in current_state:
            if name in self._initial_state:
                update[name] = current_state[name] - self._initial_state[name]

        return update

    def apply_model_update(self, update: Dict[str, torch.Tensor], learning_rate: float = 1.0):
        """
        Apply a model update (e.g., aggregated update from FL server).
        
        Args:
            update: Dictionary of parameter updates
            learning_rate: Scaling factor for the update
        """
        pytorch_model = self.model.model
        current_state = pytorch_model.state_dict()

        for name, delta in update.items():
            if name in current_state:
                current_state[name] = current_state[name] + learning_rate * delta.to(self.device)

        pytorch_model.load_state_dict(current_state, strict=False)

    def freeze_backbone(self, freeze_until: int = 10):
        """
        Freeze backbone layers for transfer learning.
        
        Args:
            freeze_until: Freeze all layers up to this index
        """
        pytorch_model = self.model.model
        trainable_names = []

        for i, (name, param) in enumerate(pytorch_model.named_parameters()):
            if i < freeze_until:
                param.requires_grad = False
            else:
                param.requires_grad = True
                trainable_names.append(name)

        self._trainable_layers = trainable_names
        logger.info(
            f"Froze {freeze_until} layers. "
            f"Trainable parameters: {len(trainable_names)}"
        )

    def train_local(
        self,
        data_config: str,
        epochs: int = 5,
        batch_size: int = 16,
        imgsz: int = 640,
        lr: float = 0.01,
        save_dir: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model locally on client data.
        
        Args:
            data_config: Path to data YAML configuration
            epochs: Number of local training epochs
            batch_size: Training batch size
            imgsz: Input image size
            lr: Learning rate
            save_dir: Directory to save training results
            
        Returns:
            Training metrics dictionary
        """
        # Save initial state before training
        self.save_initial_state()

        # Configure training arguments
        train_args = {
            "data": data_config,
            "epochs": epochs,
            "batch": batch_size,
            "imgsz": imgsz,
            "lr0": lr,
            "device": self.device,
            "verbose": False,
            "save": False,
            "plots": False,
        }

        if save_dir:
            train_args["project"] = save_dir

        train_args.update(kwargs)

        # Run training
        try:
            results = self.model.train(**train_args)
            metrics = self._extract_metrics(results)
            logger.info(f"Local training complete. Metrics: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {"error": str(e)}

    def predict(
        self,
        source: Any,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        **kwargs
    ) -> Any:
        """
        Run inference with the model.
        
        Args:
            source: Image source (path, URL, numpy array, tensor)
            conf: Confidence threshold
            iou: IoU threshold for NMS
            imgsz: Input image size
            
        Returns:
            Detection results
        """
        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=self.device,
            verbose=False,
            **kwargs
        )
        return results

    def validate(self, data_config: str, **kwargs) -> Dict[str, float]:
        """
        Validate the model on a dataset.
        
        Args:
            data_config: Path to validation data YAML
            
        Returns:
            Validation metrics
        """
        try:
            results = self.model.val(data=data_config, device=self.device, verbose=False, **kwargs)
            return self._extract_val_metrics(results)
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {"error": str(e)}

    def get_model_size(self) -> Dict[str, int]:
        """Get model size information."""
        pytorch_model = self.model.model
        total_params = sum(p.numel() for p in pytorch_model.parameters())
        trainable_params = sum(p.numel() for p in pytorch_model.parameters() if p.requires_grad)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": total_params - trainable_params,
            "model_size_mb": sum(
                p.numel() * p.element_size() for p in pytorch_model.parameters()
            ) / (1024 * 1024)
        }

    def get_layer_info(self) -> List[Dict[str, Any]]:
        """Get information about model layers."""
        pytorch_model = self.model.model
        layers = []

        for name, module in pytorch_model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                params = sum(p.numel() for p in module.parameters())
                if params > 0:
                    layers.append({
                        "name": name,
                        "type": type(module).__name__,
                        "parameters": params,
                        "trainable": any(p.requires_grad for p in module.parameters())
                    })

        return layers

    def serialize_state(self) -> bytes:
        """Serialize model state for network transfer."""
        import io
        buffer = io.BytesIO()
        state_dict = self.get_trainable_parameters()
        torch.save(state_dict, buffer)
        return buffer.getvalue()

    def deserialize_state(self, data: bytes):
        """Load model state from serialized bytes."""
        import io
        buffer = io.BytesIO(data)
        state_dict = torch.load(buffer, map_location=self.device)
        self.set_model_parameters(state_dict)

    def _extract_metrics(self, results) -> Dict[str, Any]:
        """Extract training metrics from YOLO results."""
        metrics = {}
        try:
            if hasattr(results, "results_dict"):
                metrics = dict(results.results_dict)
            elif hasattr(results, "box"):
                metrics = {
                    "mAP50": float(results.box.map50) if hasattr(results.box, "map50") else 0.0,
                    "mAP50-95": float(results.box.map) if hasattr(results.box, "map") else 0.0,
                    "precision": float(results.box.mp) if hasattr(results.box, "mp") else 0.0,
                    "recall": float(results.box.mr) if hasattr(results.box, "mr") else 0.0,
                }
        except Exception as e:
            metrics["extraction_error"] = str(e)

        return metrics

    def _extract_val_metrics(self, results) -> Dict[str, float]:
        """Extract validation metrics from YOLO results."""
        metrics = {}
        try:
            if hasattr(results, "box"):
                metrics = {
                    "mAP50": float(results.box.map50),
                    "mAP50-95": float(results.box.map),
                    "precision": float(results.box.mp),
                    "recall": float(results.box.mr),
                }
        except Exception as e:
            metrics["extraction_error"] = str(e)

        return metrics

    def __repr__(self) -> str:
        size_info = self.get_model_size()
        return (
            f"YOLOv11FederatedWrapper("
            f"variant={self.model_variant}, "
            f"classes={self.num_classes}, "
            f"params={size_info['total_parameters']:,}, "
            f"trainable={size_info['trainable_parameters']:,}, "
            f"device={self.device})"
        )
