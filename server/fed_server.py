"""
Federated Learning Server for FedX-PALM using Flower (flwr).

Implements the FL Server Aggregator as described in thesis Section 3.1.1:
- Initializes global YOLOv11 model with pretrained weights (w₀)
- Aggregates client updates using FedAvg (Eq. 3.4)
- Distributes updated global model back to clients
- Applies Central DP noise after aggregation (optional)

Communication Round Protocol (thesis Section 3.1.3):
1. Broadcast: Server sends global model wₜ to all client nodes
2. Local Training: Each client fine-tunes on local Non-IID data
3. Privacy Injection: Clients apply DP-SGD (clip + noise) on gradients
4. Aggregation: Server computes wₜ₊₁ = Σ(nₖ/N)·wₖ via FedAvg

Configuration (thesis Table 3.5):
- Communication Rounds: 100
- Clients: 4 (K=4)
- Aggregation: FedAvg
- Model: YOLOv11 Nano (640×640)
"""

import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import OrderedDict
from pathlib import Path
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class FedXPalmStrategy(FedAvg):
    """
    Custom FedAvg strategy for FedX-PALM.

    Extends Flower's FedAvg with:
    - YOLOv11 model initialization
    - Privacy budget tracking
    - Round-level metrics logging
    - Model checkpointing
    - Central DP (optional server-side noise)

    FedAvg formula (thesis Eq. 3.4):
    wₜ₊₁ = Σₖ₌₁ᴷ (nₖ/N) · wₖᵗ⁺¹
    """

    def __init__(
        self,
        model_variant: str = "yolo11n.pt",
        num_classes: int = 6,
        min_fit_clients: int = 4,
        min_available_clients: int = 4,
        num_rounds: int = 100,
        checkpoint_dir: str = "./checkpoints",
        save_every_n_rounds: int = 10,
        central_dp_enabled: bool = False,
        central_dp_noise_multiplier: float = 0.0,
        central_dp_clip_norm: float = 1.0,
        **kwargs
    ):
        """
        Args:
            model_variant: YOLOv11 variant (thesis: yolo11n.pt = Nano)
            num_classes: Number of detection classes (thesis: 6)
            min_fit_clients: Minimum clients for training (thesis: K=4)
            min_available_clients: Minimum available clients
            num_rounds: Total communication rounds (thesis: 100)
            checkpoint_dir: Directory for model checkpoints
            save_every_n_rounds: Checkpoint frequency
            central_dp_enabled: Whether to apply server-side DP
            central_dp_noise_multiplier: Server-side noise σ
            central_dp_clip_norm: Server-side clipping threshold
        """
        super().__init__(
            fraction_fit=1.0,  # All clients participate every round
            fraction_evaluate=1.0,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            min_evaluate_clients=min_fit_clients,
            **kwargs
        )

        self.model_variant = model_variant
        self.num_classes = num_classes
        self.num_rounds = num_rounds
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every_n_rounds = save_every_n_rounds

        # Central DP (optional, applied after aggregation)
        self.central_dp_enabled = central_dp_enabled
        self.central_dp_noise_multiplier = central_dp_noise_multiplier
        self.central_dp_clip_norm = central_dp_clip_norm

        # Training state
        self.current_round = 0
        self.round_metrics: List[Dict] = []
        self.best_map = 0.0

        # Initialize global model
        self.initial_parameters = self._initialize_global_model()

        logger.info(
            f"FedX-Palm Strategy initialized: "
            f"model={model_variant}, classes={num_classes}, "
            f"rounds={num_rounds}, min_clients={min_fit_clients}, "
            f"central_dp={central_dp_enabled}"
        )

    def _initialize_global_model(self) -> Parameters:
        """
        Initialize global YOLOv11 model (w₀).

        As per thesis Section 3.1.1:
        Server initializes pretrained YOLOv11 weights before distribution.
        """
        try:
            from ultralytics import YOLO

            model = YOLO(self.model_variant)
            pytorch_model = model.model

            # Extract parameters as numpy arrays
            ndarrays = [
                val.cpu().numpy()
                for _, val in pytorch_model.state_dict().items()
            ]

            logger.info(
                f"Global model initialized: {self.model_variant}, "
                f"{sum(arr.size for arr in ndarrays):,} parameters"
            )

            return ndarrays_to_parameters(ndarrays)

        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            # Return empty parameters as fallback
            return ndarrays_to_parameters([])

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        """Provide initial global model parameters."""
        return self.initial_parameters

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate client model updates using FedAvg.

        Implements thesis Eq. 3.4:
        wₜ₊₁ = Σₖ₌₁ᴷ (nₖ/N) · wₖᵗ⁺¹

        Also applies Central DP (optional server-side noise) after aggregation.
        """
        self.current_round = server_round

        if not results:
            logger.warning(f"Round {server_round}: No results to aggregate")
            return None, {}

        # Log round info
        logger.info(
            f"Round {server_round}/{self.num_rounds}: "
            f"Aggregating {len(results)} client updates, "
            f"{len(failures)} failures"
        )

        # Standard FedAvg aggregation
        aggregated_parameters, metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is None:
            return None, metrics

        # Apply Central DP (optional server-side noise after aggregation)
        if self.central_dp_enabled and self.central_dp_noise_multiplier > 0:
            aggregated_parameters = self._apply_central_dp(aggregated_parameters)

        # Collect client metrics
        round_metrics = self._collect_round_metrics(server_round, results)
        self.round_metrics.append(round_metrics)

        # Checkpoint
        if server_round % self.save_every_n_rounds == 0:
            self._save_checkpoint(aggregated_parameters, server_round)

        # Add metrics to response
        metrics["round"] = server_round
        metrics["num_clients"] = len(results)
        if round_metrics.get("avg_map50"):
            metrics["avg_map50"] = round_metrics["avg_map50"]

        return aggregated_parameters, metrics

    def _apply_central_dp(self, parameters: Parameters) -> Parameters:
        """
        Apply Central DP: add Gaussian noise to aggregated parameters.

        This provides an additional layer of privacy on top of local DP-SGD.
        """
        ndarrays = parameters_to_ndarrays(parameters)

        noisy_arrays = []
        for arr in ndarrays:
            noise_std = self.central_dp_noise_multiplier * self.central_dp_clip_norm
            noise = np.random.normal(0, noise_std, size=arr.shape).astype(arr.dtype)
            noisy_arrays.append(arr + noise)

        logger.debug(
            f"Central DP applied: σ={self.central_dp_noise_multiplier}, "
            f"C={self.central_dp_clip_norm}"
        )

        return ndarrays_to_parameters(noisy_arrays)

    def _collect_round_metrics(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]]
    ) -> Dict:
        """Collect and aggregate metrics from client results."""
        metrics = {
            "round": server_round,
            "timestamp": datetime.now().isoformat(),
            "num_clients": len(results),
            "total_samples": 0,
            "client_metrics": []
        }

        map50_values = []
        for client_proxy, fit_res in results:
            client_metrics = fit_res.metrics if fit_res.metrics else {}
            metrics["total_samples"] += fit_res.num_examples
            metrics["client_metrics"].append({
                "client_id": client_proxy.cid,
                "num_examples": fit_res.num_examples,
                "metrics": client_metrics
            })

            if "map50" in client_metrics:
                map50_values.append(client_metrics["map50"])

        if map50_values:
            metrics["avg_map50"] = float(np.mean(map50_values))
            metrics["min_map50"] = float(np.min(map50_values))
            metrics["max_map50"] = float(np.max(map50_values))

        return metrics

    def _save_checkpoint(self, parameters: Parameters, round_num: int):
        """Save model checkpoint."""
        try:
            ndarrays = parameters_to_ndarrays(parameters)
            checkpoint = {
                "round": round_num,
                "parameters": ndarrays,
                "metrics": self.round_metrics[-1] if self.round_metrics else {},
                "best_map": self.best_map
            }

            path = self.checkpoint_dir / f"global_model_round_{round_num}.pt"
            torch.save(checkpoint, path)
            logger.info(f"Checkpoint saved: {path}")

        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    def get_training_history(self) -> List[Dict]:
        """Get complete training history."""
        return self.round_metrics


def start_flower_server(
    server_address: str = "0.0.0.0:8080",
    num_rounds: int = 100,
    model_variant: str = "yolo11n.pt",
    num_classes: int = 6,
    min_clients: int = 4,
    checkpoint_dir: str = "./checkpoints",
    central_dp_enabled: bool = False,
    central_dp_noise_multiplier: float = 0.0,
):
    """
    Start the Flower FL server for FedX-PALM.

    This is the main entry point for the FL Server Aggregator
    as described in thesis Section 3.1.1.

    Args:
        server_address: Server address (host:port)
        num_rounds: Total communication rounds (thesis: 100)
        model_variant: YOLOv11 variant (thesis: yolo11n.pt)
        num_classes: Detection classes (thesis: 6)
        min_clients: Minimum clients per round (thesis: K=4)
        checkpoint_dir: Checkpoint directory
        central_dp_enabled: Enable server-side DP
        central_dp_noise_multiplier: Server-side noise level
    """
    # Create FedX-PALM strategy
    strategy = FedXPalmStrategy(
        model_variant=model_variant,
        num_classes=num_classes,
        min_fit_clients=min_clients,
        min_available_clients=min_clients,
        num_rounds=num_rounds,
        checkpoint_dir=checkpoint_dir,
        central_dp_enabled=central_dp_enabled,
        central_dp_noise_multiplier=central_dp_noise_multiplier,
    )

    # Configure Flower server
    config = fl.server.ServerConfig(num_rounds=num_rounds)

    logger.info(f"Starting Flower FL Server on {server_address}")
    logger.info(f"  Model: {model_variant} ({num_classes} classes)")
    logger.info(f"  Rounds: {num_rounds}")
    logger.info(f"  Min clients: {min_clients}")
    logger.info(f"  Central DP: {central_dp_enabled}")

    # Start server
    fl.server.start_server(
        server_address=server_address,
        config=config,
        strategy=strategy,
    )

    # Save final training history
    history_path = Path(checkpoint_dir) / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(strategy.get_training_history(), f, indent=2, default=str)
    logger.info(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FedX-PALM FL Server (Flower)")
    parser.add_argument("--address", default="0.0.0.0:8080", help="Server address")
    parser.add_argument("--num-rounds", type=int, default=100, help="Communication rounds")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLOv11 variant")
    parser.add_argument("--num-classes", type=int, default=6, help="Number of classes")
    parser.add_argument("--min-clients", type=int, default=4, help="Minimum clients")
    parser.add_argument("--checkpoint-dir", default="./checkpoints", help="Checkpoint dir")
    parser.add_argument("--central-dp", action="store_true", help="Enable central DP")
    parser.add_argument("--central-dp-sigma", type=float, default=0.0, help="Central DP noise")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    start_flower_server(
        server_address=args.address,
        num_rounds=args.num_rounds,
        model_variant=args.model,
        num_classes=args.num_classes,
        min_clients=args.min_clients,
        checkpoint_dir=args.checkpoint_dir,
        central_dp_enabled=args.central_dp,
        central_dp_noise_multiplier=args.central_dp_sigma,
    )
