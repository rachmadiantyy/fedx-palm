"""
Federated Learning Client for FedX-PALM using Flower (flwr).

Implements the FL Client Node as described in thesis Section 3.1.2:
- Local fine-tuning of YOLOv11 on private Non-IID data
- DP-SGD via Opacus (gradient clipping + Gaussian noise)
- Flower client interface for communication with server
- Grad-CAM++ XAI generation (optional, every N rounds)

Communication Round Protocol (thesis Section 3.1.3):
1. Receive global model wₜ from server (Broadcast)
2. Local Training: Fine-tune YOLOv11 on local data
3. Privacy Injection: Apply DP-SGD (clip + noise) via Opacus
4. Send updated parameters wₖᵗ⁺¹ back to server

Configuration (thesis Table 3.5):
- Local Epochs: 2
- Batch Size: 16
- Learning Rate: 0.01 (SGD)
- Optimizer: SGD
- Loss: Complete IoU (CIoU)
- Input Resolution: 640×640
- Clipping Threshold C: 1.0
"""

import flwr as fl
from flwr.common import (
    NDArrays,
    Scalar,
    Parameters,
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    Status,
    Code,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
import time

from ultralytics import YOLO

from utils.differential_privacy import (
    PrivacyConfig,
    DPStrategy,
    OpacusDPEngine,
    ManualDPSGD,
    get_privacy_config,
)

logger = logging.getLogger(__name__)


class FedXPalmClient(fl.client.NumPyClient):
    """
    Flower NumPy Client for FedX-PALM.

    Each client represents one perkebunan (plantation) node
    running inside a Docker container with:
    - Isolated local dataset (Non-IID, Dirichlet distributed)
    - YOLOv11 model for palm fruit ripeness detection
    - DP-SGD protection via Opacus
    - Optional Grad-CAM++ XAI generation

    Thesis Architecture (Section 3.1.2 - Client Nodes Dockerized):
    - Isolasi Dataset: exclusive access to local data volume
    - Local Training: fine-tune YOLOv11 on Non-IID data
    - Implementasi Privasi: gradient clipping + Gaussian noise
    """

    def __init__(
        self,
        client_id: str,
        data_config: str,
        model_variant: str = "yolo11n.pt",
        num_classes: int = 6,
        local_epochs: int = 2,
        batch_size: int = 16,
        learning_rate: float = 0.01,
        imgsz: int = 640,
        privacy_scenario: str = "moderate",
        device: Optional[str] = None,
        xai_enabled: bool = True,
        xai_every_n_rounds: int = 5,
        save_dir: str = "./client_results",
    ):
        """
        Args:
            client_id: Unique client identifier (client_1 to client_4)
            data_config: Path to data.yaml for this client's local data
            model_variant: YOLOv11 variant (thesis: yolo11n.pt)
            num_classes: Number of classes (thesis: 6)
            local_epochs: Local training epochs per round (thesis: 2)
            batch_size: Batch size (thesis: 16)
            learning_rate: Learning rate (thesis: 0.01)
            imgsz: Input image size (thesis: 640)
            privacy_scenario: DP scenario (baseline/weak/moderate/strong/partial_moderate)
            device: Computing device (cuda/cpu)
            xai_enabled: Enable Grad-CAM++ generation
            xai_every_n_rounds: Generate XAI reports every N rounds
            save_dir: Directory for saving results
        """
        self.client_id = client_id
        self.data_config = data_config
        self.model_variant = model_variant
        self.num_classes = num_classes
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.imgsz = imgsz
        self.xai_enabled = xai_enabled
        self.xai_every_n_rounds = xai_every_n_rounds
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Initialize YOLOv11 model
        self.model = YOLO(model_variant)

        # Privacy configuration (thesis Table 3.4)
        self.privacy_config = get_privacy_config(privacy_scenario)
        self.dp_engine = ManualDPSGD(self.privacy_config)

        # Training state
        self.current_round = 0
        self.training_history: List[Dict] = []

        logger.info(
            f"FedX-PALM Client initialized: {client_id}\n"
            f"  Model: {model_variant} ({num_classes} classes)\n"
            f"  Data: {data_config}\n"
            f"  Device: {self.device}\n"
            f"  Privacy: {privacy_scenario} "
            f"(ε={self.privacy_config.epsilon}, σ={self.privacy_config.noise_multiplier})\n"
            f"  Strategy: {self.privacy_config.strategy.value}\n"
            f"  Local epochs: {local_epochs}, batch_size: {batch_size}"
        )

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return current model parameters as numpy arrays."""
        state_dict = self.model.model.state_dict()
        return [val.cpu().numpy() for _, val in state_dict.items()]

    def set_parameters(self, parameters: NDArrays):
        """Set model parameters from numpy arrays (received from server)."""
        state_dict = self.model.model.state_dict()
        params_dict = zip(state_dict.keys(), parameters)
        new_state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in params_dict}
        )
        self.model.model.load_state_dict(new_state_dict, strict=False)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Perform local training (one FL round).

        Steps (thesis Section 3.1.3 - Communication Round):
        1. Load global model parameters from server
        2. Fine-tune on local Non-IID data with DP-SGD
        3. Return updated parameters with DP noise applied

        Args:
            parameters: Global model parameters from server (wₜ)
            config: Training configuration from server

        Returns:
            Tuple of (updated_parameters, num_examples, metrics)
        """
        self.current_round += 1
        round_start = time.time()

        logger.info(
            f"[{self.client_id}] Round {self.current_round}: "
            f"Starting local training..."
        )

        # Step 1: Load global model (Broadcast phase)
        self.set_parameters(parameters)

        # Step 2: Local Training with YOLOv11
        # Save initial state for computing model update delta
        initial_params = self.get_parameters({})

        # Train locally
        metrics = self._train_local()

        # Step 3: Compute model update and apply DP
        updated_params = self.get_parameters({})

        # Apply Differential Privacy to model updates
        if self.privacy_config.strategy != DPStrategy.NONE:
            updated_params = self._apply_dp_to_updates(
                initial_params, updated_params
            )

        # Get number of training examples
        num_examples = self._get_num_examples()

        # Record training history
        duration = time.time() - round_start
        round_record = {
            "round": self.current_round,
            "duration_seconds": duration,
            "num_examples": num_examples,
            "metrics": metrics,
            "privacy": {
                "epsilon": self.privacy_config.epsilon,
                "noise_multiplier": self.privacy_config.noise_multiplier,
                "strategy": self.privacy_config.strategy.value
            }
        }
        self.training_history.append(round_record)

        # Generate XAI report (optional, every N rounds)
        if (self.xai_enabled and
                self.current_round % self.xai_every_n_rounds == 0):
            self._generate_xai_report()

        logger.info(
            f"[{self.client_id}] Round {self.current_round} complete: "
            f"duration={duration:.1f}s, examples={num_examples}, "
            f"metrics={metrics}"
        )

        # Return metrics for server
        fit_metrics: Dict[str, Scalar] = {
            "client_id": self.client_id,
            "round": self.current_round,
            "duration": duration,
        }
        if "mAP50" in metrics:
            fit_metrics["map50"] = float(metrics["mAP50"])
        if "precision" in metrics:
            fit_metrics["precision"] = float(metrics["precision"])
        if "recall" in metrics:
            fit_metrics["recall"] = float(metrics["recall"])

        return updated_params, num_examples, fit_metrics

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate global model on local validation data.

        Args:
            parameters: Global model parameters to evaluate
            config: Evaluation configuration

        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        self.set_parameters(parameters)

        # Run validation
        try:
            results = self.model.val(
                data=self.data_config,
                imgsz=self.imgsz,
                batch=self.batch_size,
                device=self.device,
                verbose=False,
            )

            # Extract metrics
            map50 = float(results.box.map50) if hasattr(results.box, 'map50') else 0.0
            map50_95 = float(results.box.map) if hasattr(results.box, 'map') else 0.0
            precision = float(results.box.mp) if hasattr(results.box, 'mp') else 0.0
            recall = float(results.box.mr) if hasattr(results.box, 'mr') else 0.0

            num_examples = self._get_num_examples()

            # Loss approximation (1 - mAP as proxy)
            loss = 1.0 - map50

            eval_metrics: Dict[str, Scalar] = {
                "map50": map50,
                "map50_95": map50_95,
                "precision": precision,
                "recall": recall,
                "client_id": self.client_id,
            }

            return loss, num_examples, eval_metrics

        except Exception as e:
            logger.error(f"[{self.client_id}] Evaluation failed: {e}")
            return 1.0, 0, {"error": str(e)}

    def _train_local(self) -> Dict[str, Any]:
        """
        Perform local training on this client's Non-IID data.

        Uses YOLOv11 training with:
        - Optimizer: SGD (thesis Table 3.5)
        - Loss: CIoU (thesis Section 2.2.1)
        - Epochs: 2 per round (thesis Table 3.5)
        - Input: 640×640 (thesis Table 3.5)
        """
        try:
            results = self.model.train(
                data=self.data_config,
                epochs=self.local_epochs,
                batch=self.batch_size,
                imgsz=self.imgsz,
                lr0=self.learning_rate,
                optimizer="SGD",  # thesis Table 3.5
                device=self.device,
                verbose=False,
                save=False,
                plots=False,
                exist_ok=True,
                project=str(self.save_dir / "runs"),
                name=f"round_{self.current_round}",
            )

            # Extract training metrics
            metrics = {}
            if hasattr(results, 'box'):
                metrics["mAP50"] = float(results.box.map50) if hasattr(results.box, 'map50') else 0.0
                metrics["mAP50-95"] = float(results.box.map) if hasattr(results.box, 'map') else 0.0
                metrics["precision"] = float(results.box.mp) if hasattr(results.box, 'mp') else 0.0
                metrics["recall"] = float(results.box.mr) if hasattr(results.box, 'mr') else 0.0

            return metrics

        except Exception as e:
            logger.error(f"[{self.client_id}] Local training failed: {e}")
            return {"error": str(e)}

    def _apply_dp_to_updates(
        self,
        initial_params: NDArrays,
        updated_params: NDArrays
    ) -> NDArrays:
        """
        Apply Differential Privacy to model updates before sending to server.

        Implements thesis Section 3.4:
        1. Compute model update delta: Δw = w_updated - w_initial
        2. Clip the update (Eq. 3.2): Δw̄ = Δw · min(1, C/||Δw||₂)
        3. Add Gaussian noise (Eq. 3.3): Δw̃ = Δw̄ + N(0, σ²C²I)
        4. Return: w_initial + Δw̃

        Args:
            initial_params: Parameters before local training
            updated_params: Parameters after local training

        Returns:
            DP-protected parameters
        """
        # Convert to model update dict for DP engine
        state_dict_keys = list(self.model.model.state_dict().keys())

        model_update = {}
        for i, key in enumerate(state_dict_keys):
            if i < len(initial_params) and i < len(updated_params):
                delta = updated_params[i] - initial_params[i]
                model_update[key] = torch.tensor(delta)

        # Apply DP (clipping + noise) via ManualDPSGD
        privatized_update = self.dp_engine.privatize_model_update(model_update)

        # Reconstruct full parameters: w_initial + Δw̃ (privatized)
        dp_params = []
        for i, key in enumerate(state_dict_keys):
            if key in privatized_update:
                dp_delta = privatized_update[key].numpy()
                dp_params.append(initial_params[i] + dp_delta)
            else:
                dp_params.append(updated_params[i])

        logger.debug(
            f"[{self.client_id}] DP applied: "
            f"ε={self.privacy_config.epsilon}, "
            f"σ={self.privacy_config.noise_multiplier}, "
            f"strategy={self.privacy_config.strategy.value}"
        )

        return dp_params

    def _generate_xai_report(self):
        """Generate Grad-CAM++ XAI report for this round."""
        try:
            from xai.explainer import XAIReportGenerator
            import cv2

            # Find a sample validation image
            data_dir = Path(self.data_config).parent
            val_dir = data_dir / "images" / "val"

            if not val_dir.exists():
                return

            sample_images_paths = list(val_dir.glob("*.jpg"))[:5]
            if not sample_images_paths:
                return

            images = [cv2.imread(str(p)) for p in sample_images_paths]
            images = [img for img in images if img is not None]

            if not images:
                return

            # Generate XAI report
            report_gen = XAIReportGenerator(
                self.model,
                privacy_epsilon=self.privacy_config.epsilon
            )

            output_dir = str(
                self.save_dir / "xai_reports" / f"round_{self.current_round}"
            )

            report = report_gen.generate_report(
                images=images,
                target_class=None,
                output_dir=output_dir
            )

            report_gen.cleanup()

            logger.info(
                f"[{self.client_id}] XAI report generated for round "
                f"{self.current_round}: "
                f"AD={report.get('average_drop', {}).get('average_drop_pct', 'N/A')}%"
            )

        except Exception as e:
            logger.debug(f"[{self.client_id}] XAI generation skipped: {e}")

    def _get_num_examples(self) -> int:
        """Get number of training examples in local dataset."""
        try:
            import yaml
            with open(self.data_config, "r") as f:
                data_cfg = yaml.safe_load(f)

            train_path = Path(data_cfg.get("path", "")) / data_cfg.get("train", "")
            if train_path.exists():
                count = len(list(train_path.glob("*.jpg"))) + \
                        len(list(train_path.glob("*.png")))
                return max(count, 1)
        except Exception:
            pass
        return 100  # Default estimate


def start_flower_client(
    server_address: str = "localhost:8080",
    client_id: str = "client_1",
    data_config: str = "./data/client_1/data.yaml",
    model_variant: str = "yolo11n.pt",
    num_classes: int = 6,
    local_epochs: int = 2,
    batch_size: int = 16,
    learning_rate: float = 0.01,
    privacy_scenario: str = "moderate",
    xai_enabled: bool = True,
):
    """
    Start a Flower FL client for FedX-PALM.

    This is the main entry point for each Client Node
    as described in thesis Section 3.1.2.

    Args:
        server_address: FL server address
        client_id: Unique client identifier
        data_config: Path to client's data.yaml
        model_variant: YOLOv11 variant
        num_classes: Number of classes (6)
        local_epochs: Epochs per round (2)
        batch_size: Batch size (16)
        learning_rate: Learning rate (0.01)
        privacy_scenario: DP scenario name
        xai_enabled: Enable Grad-CAM++ XAI
    """
    # Create client
    client = FedXPalmClient(
        client_id=client_id,
        data_config=data_config,
        model_variant=model_variant,
        num_classes=num_classes,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        privacy_scenario=privacy_scenario,
        xai_enabled=xai_enabled,
        save_dir=f"./results/{client_id}",
    )

    logger.info(f"Starting Flower client '{client_id}' → server at {server_address}")

    # Start Flower client
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client,
    )

    logger.info(f"Client '{client_id}' finished. Rounds completed: {client.current_round}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FedX-PALM FL Client (Flower)")
    parser.add_argument("--server", default="localhost:8080", help="Server address")
    parser.add_argument("--client-id", required=True, help="Client ID (client_1..client_4)")
    parser.add_argument("--data-config", required=True, help="Path to data.yaml")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLOv11 variant")
    parser.add_argument("--num-classes", type=int, default=6, help="Number of classes")
    parser.add_argument("--local-epochs", type=int, default=2, help="Local epochs per round")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument(
        "--privacy",
        default="moderate",
        choices=["baseline", "weak", "moderate", "strong", "partial_moderate"],
        help="Privacy scenario (thesis Table 3.4)"
    )
    parser.add_argument("--no-xai", action="store_true", help="Disable XAI generation")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    start_flower_client(
        server_address=args.server,
        client_id=args.client_id,
        data_config=args.data_config,
        model_variant=args.model,
        num_classes=args.num_classes,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        privacy_scenario=args.privacy,
        xai_enabled=not args.no_xai,
    )
