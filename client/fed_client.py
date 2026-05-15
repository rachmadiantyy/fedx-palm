"""
Federated Learning Client for FedX-PALM.

Implements the FL client with:
- Local YOLOv11 training on private data
- Local Differential Privacy (gradient clipping + noise)
- Communication with FL server via REST API
- XAI-based model explanation generation
- FedProx proximal term support
"""

import torch
import numpy as np
import requests
import io
import base64
import logging
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

from models.yolov11_wrapper import YOLOv11FederatedWrapper
from utils.differential_privacy import DifferentialPrivacyEngine, GradientClipper

logger = logging.getLogger(__name__)


@dataclass
class ClientConfig:
    """Configuration for the FL client."""
    client_id: str
    server_url: str = "http://localhost:8080"
    data_config: str = "data.yaml"
    model_variant: str = "yolo11n.pt"
    num_classes: int = 80
    local_epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 0.01
    imgsz: int = 640
    device: Optional[str] = None
    # Differential Privacy (local DP)
    dp_enabled: bool = True
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    dp_max_grad_norm: float = 1.0
    dp_noise_multiplier: float = 1.0
    # FedProx
    fedprox_mu: float = 0.0  # 0 means FedAvg, >0 means FedProx
    # Training settings
    patience: int = 10
    save_dir: str = "./client_results"
    max_retries: int = 3
    retry_delay: float = 5.0


class FederatedClient:
    """
    Federated Learning Client.
    
    Manages local model training, privacy-preserving updates,
    and communication with the FL server.
    """

    def __init__(self, config: ClientConfig):
        """
        Args:
            config: Client configuration
        """
        self.config = config
        self.client_id = config.client_id
        self.server_url = config.server_url.rstrip("/")

        # Initialize local model
        self.model = YOLOv11FederatedWrapper(
            model_variant=config.model_variant,
            num_classes=config.num_classes,
            device=config.device
        )

        # Differential Privacy (local)
        self.dp_enabled = config.dp_enabled
        self.dp_engine: Optional[DifferentialPrivacyEngine] = None
        self.clipper: Optional[GradientClipper] = None

        if config.dp_enabled:
            self.clipper = GradientClipper(max_norm=config.dp_max_grad_norm)
            self.dp_engine = DifferentialPrivacyEngine(
                epsilon=config.dp_epsilon,
                delta=config.dp_delta,
                max_grad_norm=config.dp_max_grad_norm,
                noise_multiplier=config.dp_noise_multiplier,
                num_clients=1,  # Local DP perspective
                num_rounds=100
            )

        # Global model reference (for FedProx)
        self._global_model_params: Optional[Dict[str, torch.Tensor]] = None

        # Training history
        self.training_history: List[Dict] = []
        self.current_round: int = 0

        # Setup save directory
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Client {self.client_id} initialized: model={config.model_variant}, DP={config.dp_enabled}")

    def register_with_server(self) -> bool:
        """
        Register this client with the FL server.
        
        Returns:
            True if registration successful
        """
        try:
            response = self._send_request("POST", "/register", {
                "client_id": self.client_id,
                "data_size": self._get_data_size()
            })

            if response.get("status") == "registered":
                # Load global model parameters
                if "global_model_params_b64" in response:
                    self._load_params_from_b64(response["global_model_params_b64"])

                self.current_round = response.get("current_round", 0)
                logger.info(f"Client {self.client_id} registered successfully")
                return True
            else:
                logger.error(f"Registration failed: {response}")
                return False

        except Exception as e:
            logger.error(f"Registration error: {e}")
            return False

    def fetch_global_model(self) -> bool:
        """
        Fetch the latest global model from the server.
        
        Returns:
            True if fetch successful
        """
        try:
            response = self._send_request("GET", "/global_model")

            if "model_params_b64" in response:
                self._load_params_from_b64(response["model_params_b64"])
                self.current_round = response.get("round", self.current_round)

                # Store global params for FedProx
                self._global_model_params = self.model.get_model_parameters()

                logger.info(f"Global model fetched for round {self.current_round}")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to fetch global model: {e}")
            return False

    def train_local(self) -> Dict[str, Any]:
        """
        Perform local training on private data.
        
        Returns:
            Training metrics and model update
        """
        logger.info(
            f"Client {self.client_id}: Starting local training "
            f"(epochs={self.config.local_epochs}, round={self.current_round})"
        )

        # Save initial state to compute update later
        self.model.save_initial_state()

        # Train locally
        metrics = self.model.train_local(
            data_config=self.config.data_config,
            epochs=self.config.local_epochs,
            batch_size=self.config.batch_size,
            imgsz=self.config.imgsz,
            lr=self.config.learning_rate,
            save_dir=str(self.save_dir / f"round_{self.current_round}")
        )

        # Compute model update (delta)
        model_update = self.model.compute_model_update()

        # Apply FedProx proximal term if configured
        if self.config.fedprox_mu > 0 and self._global_model_params is not None:
            model_update = self._apply_fedprox_proximal(model_update)

        # Apply local differential privacy
        if self.dp_enabled and self.dp_engine:
            model_update = self._apply_local_dp(model_update)

        # Record training history
        training_record = {
            "round": self.current_round,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "update_norm": self._compute_update_norm(model_update),
            "dp_applied": self.dp_enabled
        }
        self.training_history.append(training_record)

        logger.info(f"Local training complete. Metrics: {metrics}")

        return {
            "model_update": model_update,
            "metrics": metrics,
            "data_size": self._get_data_size()
        }

    def submit_update(self, training_result: Dict[str, Any]) -> bool:
        """
        Submit model update to the FL server.
        
        Args:
            training_result: Result from train_local()
            
        Returns:
            True if submission successful
        """
        try:
            # Serialize model update
            model_update = training_result["model_update"]
            buffer = io.BytesIO()
            torch.save(model_update, buffer)
            update_b64 = base64.b64encode(buffer.getvalue()).decode()

            response = self._send_request("POST", "/submit_update", {
                "client_id": self.client_id,
                "model_update_b64": update_b64,
                "metrics": training_result.get("metrics", {}),
                "data_size": training_result.get("data_size", 0)
            })

            if response.get("status") == "received":
                logger.info(f"Update submitted for round {self.current_round}")

                # Check if aggregation was triggered
                if "aggregation" in response:
                    agg = response["aggregation"]
                    logger.info(f"Aggregation completed: round {agg.get('round')}")

                return True
            else:
                logger.error(f"Update submission failed: {response}")
                return False

        except Exception as e:
            logger.error(f"Failed to submit update: {e}")
            return False

    def run_federated_round(self) -> Dict[str, Any]:
        """
        Execute a complete federated learning round.
        
        Steps:
        1. Fetch global model
        2. Train locally
        3. Submit update
        
        Returns:
            Round result dictionary
        """
        round_start = time.time()

        # Step 1: Fetch global model
        if not self.fetch_global_model():
            return {"status": "error", "message": "Failed to fetch global model"}

        # Step 2: Local training
        training_result = self.train_local()

        if "error" in training_result.get("metrics", {}):
            return {"status": "error", "message": training_result["metrics"]["error"]}

        # Step 3: Submit update
        if not self.submit_update(training_result):
            return {"status": "error", "message": "Failed to submit update"}

        duration = time.time() - round_start

        return {
            "status": "success",
            "round": self.current_round,
            "metrics": training_result["metrics"],
            "duration_seconds": duration,
            "dp_applied": self.dp_enabled
        }

    def run_continuous(self, max_rounds: Optional[int] = None):
        """
        Run the client continuously, participating in FL rounds.
        
        Args:
            max_rounds: Maximum number of rounds to participate in (None = unlimited)
        """
        # Register with server
        if not self.register_with_server():
            logger.error("Failed to register with server. Exiting.")
            return

        rounds_completed = 0

        while True:
            if max_rounds and rounds_completed >= max_rounds:
                logger.info(f"Completed {max_rounds} rounds. Stopping.")
                break

            # Check if training is complete
            try:
                status = self._send_request("GET", "/status")
                if status.get("current_round", 0) >= status.get("total_rounds", float("inf")):
                    logger.info("Server training complete. Stopping.")
                    break
            except Exception:
                pass

            # Run a round
            result = self.run_federated_round()

            if result["status"] == "success":
                rounds_completed += 1
                logger.info(
                    f"Round {rounds_completed} complete: "
                    f"duration={result['duration_seconds']:.1f}s"
                )
            else:
                logger.warning(f"Round failed: {result.get('message')}")
                time.sleep(self.config.retry_delay)

            # Brief pause between rounds
            time.sleep(1.0)

        logger.info(f"Client {self.client_id} finished: {rounds_completed} rounds completed")

    def _apply_local_dp(self, model_update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply local differential privacy to model update."""
        return self.dp_engine.privatize_model_update(model_update)

    def _apply_fedprox_proximal(
        self, model_update: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply FedProx proximal term to the model update.
        
        The proximal term penalizes deviation from the global model.
        """
        if self._global_model_params is None:
            return model_update

        mu = self.config.fedprox_mu
        current_params = self.model.get_model_parameters()
        proximal_update = {}

        for name in model_update:
            if name in self._global_model_params and name in current_params:
                # Proximal term: mu * (w - w_global)
                proximal_term = mu * (current_params[name] - self._global_model_params[name])
                proximal_update[name] = model_update[name] - proximal_term
            else:
                proximal_update[name] = model_update[name]

        return proximal_update

    def _load_params_from_b64(self, params_b64: str):
        """Load model parameters from base64-encoded state dict."""
        buffer = io.BytesIO(base64.b64decode(params_b64))
        state_dict = torch.load(buffer, map_location="cpu")
        self.model.set_model_parameters(state_dict)

    def _get_data_size(self) -> int:
        """Get the size of local training data."""
        # Try to read from data config
        try:
            import yaml
            with open(self.config.data_config, "r") as f:
                data_cfg = yaml.safe_load(f)
            # Estimate from train path
            train_path = Path(data_cfg.get("train", ""))
            if train_path.exists():
                return len(list(train_path.glob("*.jpg"))) + len(list(train_path.glob("*.png")))
        except Exception:
            pass
        return 100  # Default estimate

    def _compute_update_norm(self, update: Dict[str, torch.Tensor]) -> float:
        """Compute L2 norm of model update."""
        if not update:
            return 0.0
        flat = torch.cat([v.flatten() for v in update.values()])
        return torch.norm(flat, p=2).item()

    def _send_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """Send HTTP request to the FL server with retry logic."""
        url = f"{self.server_url}{endpoint}"

        for attempt in range(self.config.max_retries):
            try:
                if method == "GET":
                    response = requests.get(url, timeout=60)
                elif method == "POST":
                    response = requests.post(url, json=data, timeout=120)
                else:
                    raise ValueError(f"Unsupported method: {method}")

                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self.config.max_retries}): {e}"
                )
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)

        raise ConnectionError(f"Failed to connect to server after {self.config.max_retries} attempts")

    def get_client_status(self) -> Dict[str, Any]:
        """Get comprehensive client status."""
        status = {
            "client_id": self.client_id,
            "current_round": self.current_round,
            "rounds_completed": len(self.training_history),
            "model_info": self.model.get_model_size(),
            "dp_enabled": self.dp_enabled,
            "config": {
                "server_url": self.server_url,
                "local_epochs": self.config.local_epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "fedprox_mu": self.config.fedprox_mu
            }
        }

        if self.dp_enabled and self.dp_engine:
            status["privacy"] = self.dp_engine.get_privacy_report()

        if self.training_history:
            status["last_training"] = self.training_history[-1]

        return status


def run_client(
    client_id: str,
    server_url: str = "http://localhost:8080",
    data_config: str = "data.yaml",
    model_variant: str = "yolo11n.pt",
    num_classes: int = 80,
    local_epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 0.01,
    dp_enabled: bool = True,
    dp_epsilon: float = 1.0,
    dp_max_grad_norm: float = 1.0,
    max_rounds: Optional[int] = None,
    **kwargs
):
    """
    Run a federated learning client.
    
    Args:
        client_id: Unique client identifier
        server_url: FL server URL
        data_config: Path to data YAML configuration
        model_variant: YOLOv11 variant
        num_classes: Number of classes
        local_epochs: Local training epochs per round
        batch_size: Training batch size
        learning_rate: Learning rate
        dp_enabled: Enable local differential privacy
        dp_epsilon: Privacy budget
        dp_max_grad_norm: Max gradient norm for DP
        max_rounds: Maximum rounds to participate in
    """
    config = ClientConfig(
        client_id=client_id,
        server_url=server_url,
        data_config=data_config,
        model_variant=model_variant,
        num_classes=num_classes,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        dp_enabled=dp_enabled,
        dp_epsilon=dp_epsilon,
        dp_max_grad_norm=dp_max_grad_norm,
        **kwargs
    )

    client = FederatedClient(config)
    client.run_continuous(max_rounds=max_rounds)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FedX-PALM Federated Learning Client")
    parser.add_argument("--client-id", required=True, help="Unique client ID")
    parser.add_argument("--server-url", default="http://localhost:8080", help="Server URL")
    parser.add_argument("--data-config", default="data.yaml", help="Data config YAML")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLOv11 variant")
    parser.add_argument("--num-classes", type=int, default=80, help="Number of classes")
    parser.add_argument("--local-epochs", type=int, default=5, help="Local epochs per round")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--dp-enabled", action="store_true", default=True)
    parser.add_argument("--dp-epsilon", type=float, default=1.0)
    parser.add_argument("--dp-max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-rounds", type=int, default=None)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    run_client(
        client_id=args.client_id,
        server_url=args.server_url,
        data_config=args.data_config,
        model_variant=args.model,
        num_classes=args.num_classes,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        dp_enabled=args.dp_enabled,
        dp_epsilon=args.dp_epsilon,
        dp_max_grad_norm=args.dp_max_grad_norm,
        max_rounds=args.max_rounds,
    )
