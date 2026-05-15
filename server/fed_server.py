"""
Federated Learning Server for FedX-PALM.

Implements the FL server with:
- FedAvg and FedProx aggregation strategies
- Differential Privacy integration (central DP)
- Client management and round orchestration
- Model versioning and checkpointing
- gRPC-based communication
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json
import os
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from utils.differential_privacy import DifferentialPrivacyEngine, MomentsAccountant
from models.yolov11_wrapper import YOLOv11FederatedWrapper

logger = logging.getLogger(__name__)


@dataclass
class ClientInfo:
    """Information about a registered FL client."""
    client_id: str
    data_size: int = 0
    last_seen: Optional[datetime] = None
    rounds_participated: int = 0
    is_active: bool = True
    metrics_history: List[Dict] = field(default_factory=list)


@dataclass
class RoundResult:
    """Result of a single federated learning round."""
    round_number: int
    participating_clients: List[str]
    aggregated_metrics: Dict[str, float]
    privacy_cost: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)
    duration_seconds: float = 0.0


class AggregationStrategy:
    """Base class for aggregation strategies."""

    def aggregate(
        self,
        updates: List[Dict[str, torch.Tensor]],
        weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class FedAvgAggregator(AggregationStrategy):
    """
    Federated Averaging (FedAvg) aggregation.
    
    Computes weighted average of client model updates.
    """

    def aggregate(
        self,
        updates: List[Dict[str, torch.Tensor]],
        weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates using weighted averaging.
        
        Args:
            updates: List of model updates from clients
            weights: Weights for each client (typically proportional to data size)
            
        Returns:
            Aggregated model update
        """
        if not updates:
            return {}

        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # Compute weighted average
        aggregated = {}
        for name in updates[0].keys():
            aggregated[name] = torch.zeros_like(updates[0][name])
            for update, weight in zip(updates, normalized_weights):
                if name in update:
                    aggregated[name] += weight * update[name]

        return aggregated


class FedProxAggregator(AggregationStrategy):
    """
    FedProx aggregation with proximal term.
    
    Extends FedAvg with a proximal term to handle heterogeneous data.
    """

    def __init__(self, mu: float = 0.01):
        """
        Args:
            mu: Proximal term coefficient
        """
        self.mu = mu

    def aggregate(
        self,
        updates: List[Dict[str, torch.Tensor]],
        weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """Aggregate with FedProx (same aggregation as FedAvg, proximal term is on client side)."""
        if not updates:
            return {}

        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        aggregated = {}
        for name in updates[0].keys():
            aggregated[name] = torch.zeros_like(updates[0][name])
            for update, weight in zip(updates, normalized_weights):
                if name in update:
                    aggregated[name] += weight * update[name]

        return aggregated


class FederatedServer:
    """
    Main Federated Learning Server.
    
    Orchestrates the federated learning process including:
    - Client registration and management
    - Round execution with configurable aggregation
    - Differential privacy enforcement
    - Model checkpointing and versioning
    """

    def __init__(
        self,
        model_variant: str = "yolo11n.pt",
        num_classes: int = 80,
        num_rounds: int = 100,
        min_clients: int = 2,
        fraction_fit: float = 1.0,
        aggregation_strategy: str = "fedavg",
        # Differential Privacy settings
        dp_enabled: bool = True,
        dp_epsilon: float = 1.0,
        dp_delta: float = 1e-5,
        dp_max_grad_norm: float = 1.0,
        dp_noise_multiplier: float = 1.0,
        # Server settings
        checkpoint_dir: str = "./checkpoints",
        device: Optional[str] = None,
        fedprox_mu: float = 0.01,
    ):
        """
        Args:
            model_variant: YOLOv11 model variant
            num_classes: Number of detection classes
            num_rounds: Total number of FL rounds
            min_clients: Minimum clients needed to start a round
            fraction_fit: Fraction of clients to select per round
            aggregation_strategy: 'fedavg' or 'fedprox'
            dp_enabled: Whether to use differential privacy
            dp_epsilon: Total privacy budget
            dp_delta: Privacy failure probability
            dp_max_grad_norm: Maximum gradient norm for clipping
            dp_noise_multiplier: Noise multiplier for DP
            checkpoint_dir: Directory for saving checkpoints
            device: Computing device
            fedprox_mu: FedProx proximal coefficient
        """
        self.num_rounds = num_rounds
        self.min_clients = min_clients
        self.fraction_fit = fraction_fit
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize global model
        self.global_model = YOLOv11FederatedWrapper(
            model_variant=model_variant,
            num_classes=num_classes,
            device=device
        )

        # Aggregation strategy
        if aggregation_strategy == "fedprox":
            self.aggregator = FedProxAggregator(mu=fedprox_mu)
        else:
            self.aggregator = FedAvgAggregator()

        self.aggregation_strategy = aggregation_strategy

        # Differential Privacy
        self.dp_enabled = dp_enabled
        self.dp_engine: Optional[DifferentialPrivacyEngine] = None
        self.moments_accountant: Optional[MomentsAccountant] = None

        if dp_enabled:
            self.dp_engine = DifferentialPrivacyEngine(
                epsilon=dp_epsilon,
                delta=dp_delta,
                max_grad_norm=dp_max_grad_norm,
                noise_multiplier=dp_noise_multiplier,
                num_clients=min_clients,
                num_rounds=num_rounds
            )
            self.moments_accountant = MomentsAccountant(
                noise_multiplier=dp_noise_multiplier,
                sample_rate=fraction_fit
            )

        # Client management
        self.clients: Dict[str, ClientInfo] = {}
        self.current_round: int = 0

        # History
        self.round_history: List[RoundResult] = []
        self.global_metrics: List[Dict] = []

        logger.info(
            f"Federated Server initialized: "
            f"model={model_variant}, rounds={num_rounds}, "
            f"aggregation={aggregation_strategy}, DP={dp_enabled}"
        )

    def register_client(self, client_id: str, data_size: int = 0) -> Dict[str, Any]:
        """
        Register a new client with the server.
        
        Args:
            client_id: Unique client identifier
            data_size: Number of training samples on the client
            
        Returns:
            Registration response with global model parameters
        """
        self.clients[client_id] = ClientInfo(
            client_id=client_id,
            data_size=data_size,
            last_seen=datetime.now()
        )

        logger.info(f"Client registered: {client_id} (data_size={data_size})")

        return {
            "status": "registered",
            "client_id": client_id,
            "global_model_params": self.global_model.get_model_parameters(),
            "current_round": self.current_round,
            "config": {
                "aggregation_strategy": self.aggregation_strategy,
                "dp_enabled": self.dp_enabled,
                "num_rounds": self.num_rounds
            }
        }

    def select_clients(self) -> List[str]:
        """Select clients for the current round."""
        active_clients = [
            cid for cid, info in self.clients.items() if info.is_active
        ]

        if len(active_clients) < self.min_clients:
            logger.warning(
                f"Not enough active clients: {len(active_clients)} < {self.min_clients}"
            )
            return []

        # Select fraction of clients
        num_selected = max(self.min_clients, int(len(active_clients) * self.fraction_fit))
        selected = np.random.choice(
            active_clients, size=min(num_selected, len(active_clients)), replace=False
        ).tolist()

        logger.info(f"Round {self.current_round}: Selected {len(selected)} clients: {selected}")
        return selected

    def receive_client_update(
        self,
        client_id: str,
        model_update: Dict[str, torch.Tensor],
        metrics: Dict[str, float],
        data_size: int
    ) -> Dict[str, str]:
        """
        Receive a model update from a client.
        
        Args:
            client_id: Client identifier
            model_update: Model parameter updates
            metrics: Client training metrics
            data_size: Number of samples used for training
            
        Returns:
            Acknowledgment response
        """
        if client_id not in self.clients:
            return {"status": "error", "message": "Client not registered"}

        # Update client info
        self.clients[client_id].last_seen = datetime.now()
        self.clients[client_id].data_size = data_size
        self.clients[client_id].metrics_history.append(metrics)

        return {"status": "received", "round": self.current_round}

    def aggregate_round(
        self,
        client_updates: List[Tuple[str, Dict[str, torch.Tensor], int]]
    ) -> Dict[str, Any]:
        """
        Aggregate client updates for one round.
        
        Args:
            client_updates: List of (client_id, model_update, data_size) tuples
            
        Returns:
            Round result with metrics and privacy cost
        """
        start_time = datetime.now()

        if not client_updates:
            return {"status": "error", "message": "No client updates to aggregate"}

        # Extract updates and weights
        updates = [update for _, update, _ in client_updates]
        weights = [float(data_size) for _, _, data_size in client_updates]
        client_ids = [cid for cid, _, _ in client_updates]

        # Perform aggregation
        aggregated_update = self.aggregator.aggregate(updates, weights)

        # Apply differential privacy (central DP)
        privacy_cost = {"epsilon": 0.0, "delta": 0.0}
        if self.dp_enabled and self.dp_engine:
            aggregated_update = self.dp_engine.privatize_aggregated_update(aggregated_update)
            self.moments_accountant.step()

            privacy_cost = {
                "epsilon": self.dp_engine.budget.spent_epsilon,
                "delta": self.dp_engine.budget.spent_delta,
                "remaining_epsilon": self.dp_engine.budget.remaining_epsilon
            }

        # Apply aggregated update to global model
        self.global_model.apply_model_update(aggregated_update)

        # Update round tracking
        self.current_round += 1
        duration = (datetime.now() - start_time).total_seconds()

        # Record round result
        round_result = RoundResult(
            round_number=self.current_round,
            participating_clients=client_ids,
            aggregated_metrics=self._compute_aggregate_metrics(client_updates),
            privacy_cost=privacy_cost,
            duration_seconds=duration
        )
        self.round_history.append(round_result)

        # Update client participation counts
        for client_id in client_ids:
            if client_id in self.clients:
                self.clients[client_id].rounds_participated += 1

        logger.info(
            f"Round {self.current_round} complete: "
            f"{len(client_ids)} clients, "
            f"privacy_cost={privacy_cost}"
        )

        return {
            "status": "success",
            "round": self.current_round,
            "participating_clients": len(client_ids),
            "privacy_cost": privacy_cost,
            "duration_seconds": duration,
            "global_model_params": self.global_model.get_model_parameters()
        }

    def _compute_aggregate_metrics(
        self, client_updates: List[Tuple[str, Dict[str, torch.Tensor], int]]
    ) -> Dict[str, float]:
        """Compute aggregate metrics from client updates."""
        metrics = {
            "num_clients": len(client_updates),
            "total_samples": sum(ds for _, _, ds in client_updates),
            "avg_update_norm": np.mean([
                torch.norm(
                    torch.cat([v.flatten() for v in update.values()])
                ).item()
                for _, update, _ in client_updates
            ])
        }
        return metrics

    def get_global_model_params(self) -> Dict[str, torch.Tensor]:
        """Get current global model parameters."""
        return self.global_model.get_model_parameters()

    def save_checkpoint(self, filename: Optional[str] = None):
        """Save server state checkpoint."""
        if filename is None:
            filename = f"checkpoint_round_{self.current_round}.pt"

        checkpoint = {
            "round": self.current_round,
            "global_model_state": self.global_model.get_model_parameters(),
            "round_history": [
                {
                    "round": r.round_number,
                    "clients": r.participating_clients,
                    "metrics": r.aggregated_metrics,
                    "privacy_cost": r.privacy_cost,
                    "duration": r.duration_seconds
                }
                for r in self.round_history
            ],
            "dp_report": self.dp_engine.get_privacy_report() if self.dp_engine else None,
        }

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, filepath: str):
        """Load server state from checkpoint."""
        checkpoint = torch.load(filepath, map_location="cpu")
        self.current_round = checkpoint["round"]
        self.global_model.set_model_parameters(checkpoint["global_model_state"])
        logger.info(f"Checkpoint loaded from: {filepath}, round={self.current_round}")

    def get_server_status(self) -> Dict[str, Any]:
        """Get comprehensive server status."""
        status = {
            "current_round": self.current_round,
            "total_rounds": self.num_rounds,
            "registered_clients": len(self.clients),
            "active_clients": sum(1 for c in self.clients.values() if c.is_active),
            "aggregation_strategy": self.aggregation_strategy,
            "model_info": self.global_model.get_model_size(),
        }

        if self.dp_enabled and self.dp_engine:
            status["privacy"] = self.dp_engine.get_privacy_report()

        if self.round_history:
            last_round = self.round_history[-1]
            status["last_round"] = {
                "number": last_round.round_number,
                "clients": len(last_round.participating_clients),
                "duration": last_round.duration_seconds,
                "privacy_cost": last_round.privacy_cost
            }

        return status

    def is_training_complete(self) -> bool:
        """Check if training is complete."""
        if self.current_round >= self.num_rounds:
            return True
        if self.dp_enabled and self.dp_engine and self.dp_engine.budget.is_exhausted:
            logger.warning("Training stopped: Privacy budget exhausted")
            return True
        return False
