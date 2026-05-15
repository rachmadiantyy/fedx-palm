"""
Differential Privacy Module for Federated Learning.

Implements privacy-preserving mechanisms including:
- Gaussian noise addition
- Gradient clipping
- Privacy budget (epsilon) tracking
- Moments accountant for tight privacy analysis
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import math


@dataclass
class PrivacyBudget:
    """Tracks the privacy budget (epsilon, delta) across training rounds."""
    epsilon: float = 1.0
    delta: float = 1e-5
    spent_epsilon: float = 0.0
    spent_delta: float = 0.0
    round_history: List[Dict] = field(default_factory=list)

    @property
    def remaining_epsilon(self) -> float:
        return max(0.0, self.epsilon - self.spent_epsilon)

    @property
    def is_exhausted(self) -> bool:
        return self.spent_epsilon >= self.epsilon

    def consume(self, eps: float, delta: float = 0.0):
        """Consume privacy budget for one round."""
        self.spent_epsilon += eps
        self.spent_delta += delta
        self.round_history.append({
            "epsilon_spent": eps,
            "delta_spent": delta,
            "cumulative_epsilon": self.spent_epsilon,
            "cumulative_delta": self.spent_delta
        })

    def get_summary(self) -> Dict:
        return {
            "total_epsilon": self.epsilon,
            "total_delta": self.delta,
            "spent_epsilon": self.spent_epsilon,
            "spent_delta": self.spent_delta,
            "remaining_epsilon": self.remaining_epsilon,
            "rounds_tracked": len(self.round_history),
            "is_exhausted": self.is_exhausted
        }


class GaussianMechanism:
    """
    Gaussian Mechanism for Differential Privacy.
    
    Adds calibrated Gaussian noise to achieve (epsilon, delta)-differential privacy.
    """

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, sensitivity: float = 1.0):
        """
        Args:
            epsilon: Privacy parameter (lower = more private)
            delta: Failure probability
            sensitivity: L2 sensitivity of the function
        """
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.sigma = self._compute_sigma()

    def _compute_sigma(self) -> float:
        """Compute noise scale (sigma) for Gaussian mechanism."""
        return self.sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon

    def add_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to a tensor."""
        noise = torch.normal(
            mean=0.0,
            std=self.sigma,
            size=tensor.shape,
            device=tensor.device,
            dtype=tensor.dtype
        )
        return tensor + noise

    def add_noise_to_gradients(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """Add noise to a list of gradient tensors."""
        return [self.add_noise(grad) for grad in gradients]


class GradientClipper:
    """
    Gradient Clipping for Differential Privacy.
    
    Clips per-sample gradients to bound sensitivity before noise addition.
    """

    def __init__(self, max_norm: float = 1.0, norm_type: int = 2):
        """
        Args:
            max_norm: Maximum L2 norm for gradient clipping
            norm_type: Type of norm (default: L2)
        """
        self.max_norm = max_norm
        self.norm_type = norm_type

    def clip_gradients(self, gradients: List[torch.Tensor]) -> Tuple[List[torch.Tensor], float]:
        """
        Clip gradients to have bounded norm.
        
        Returns:
            Tuple of (clipped gradients, original norm)
        """
        # Compute total norm
        total_norm = torch.norm(
            torch.stack([torch.norm(g, p=self.norm_type) for g in gradients]),
            p=self.norm_type
        ).item()

        # Compute clipping factor
        clip_factor = min(1.0, self.max_norm / (total_norm + 1e-8))

        # Apply clipping
        clipped = [g * clip_factor for g in gradients]
        return clipped, total_norm

    def clip_model_gradients(self, model: torch.nn.Module) -> float:
        """Clip gradients directly on model parameters."""
        parameters = [p for p in model.parameters() if p.grad is not None]
        if not parameters:
            return 0.0

        total_norm = torch.nn.utils.clip_grad_norm_(
            parameters, self.max_norm, norm_type=self.norm_type
        )
        return total_norm.item()


class DifferentialPrivacyEngine:
    """
    Main Differential Privacy Engine for Federated Learning.
    
    Combines gradient clipping and noise addition with privacy budget tracking.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 1.0,
        num_clients: int = 10,
        num_rounds: int = 100
    ):
        """
        Args:
            epsilon: Total privacy budget
            delta: Failure probability
            max_grad_norm: Maximum gradient norm for clipping
            noise_multiplier: Multiplier for noise scale
            num_clients: Number of FL clients
            num_rounds: Total number of federated rounds
        """
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.num_clients = num_clients
        self.num_rounds = num_rounds

        # Per-round epsilon using simple composition
        self.per_round_epsilon = epsilon / math.sqrt(num_rounds)

        # Initialize components
        self.clipper = GradientClipper(max_norm=max_grad_norm)
        self.mechanism = GaussianMechanism(
            epsilon=self.per_round_epsilon,
            delta=delta / num_rounds,
            sensitivity=max_grad_norm / num_clients
        )
        self.budget = PrivacyBudget(epsilon=epsilon, delta=delta)

        # Statistics
        self.clip_stats: List[Dict] = []

    def privatize_model_update(
        self, model_update: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply differential privacy to a model update (from a client).
        
        Steps:
        1. Clip gradients to bound sensitivity
        2. Add calibrated Gaussian noise
        3. Track privacy budget consumption
        
        Args:
            model_update: Dictionary of parameter name -> update tensor
            
        Returns:
            Privatized model update
        """
        if self.budget.is_exhausted:
            raise RuntimeError(
                "Privacy budget exhausted! Cannot process more updates. "
                f"Spent: {self.budget.spent_epsilon:.4f}/{self.epsilon}"
            )

        privatized_update = {}
        norms_before = []
        norms_after = []

        for name, update in model_update.items():
            # Clip the update
            norm_before = torch.norm(update, p=2).item()
            norms_before.append(norm_before)

            clip_factor = min(1.0, self.max_grad_norm / (norm_before + 1e-8))
            clipped_update = update * clip_factor

            norm_after = torch.norm(clipped_update, p=2).item()
            norms_after.append(norm_after)

            # Add noise
            privatized_update[name] = self.mechanism.add_noise(clipped_update)

        # Track statistics
        self.clip_stats.append({
            "avg_norm_before": np.mean(norms_before),
            "avg_norm_after": np.mean(norms_after),
            "clipping_ratio": np.mean([
                1.0 if nb > self.max_grad_norm else 0.0
                for nb in norms_before
            ])
        })

        # Consume privacy budget
        self.budget.consume(self.per_round_epsilon, self.delta / self.num_rounds)

        return privatized_update

    def privatize_aggregated_update(
        self, aggregated_update: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply DP noise to the aggregated model update on the server side.
        
        This is used in the central DP model where noise is added after aggregation.
        """
        privatized = {}
        for name, param in aggregated_update.items():
            noise_scale = self.noise_multiplier * self.max_grad_norm / self.num_clients
            noise = torch.normal(
                mean=0.0,
                std=noise_scale,
                size=param.shape,
                device=param.device,
                dtype=param.dtype
            )
            privatized[name] = param + noise

        self.budget.consume(self.per_round_epsilon, self.delta / self.num_rounds)
        return privatized

    def get_privacy_report(self) -> Dict:
        """Generate a comprehensive privacy report."""
        report = {
            "privacy_budget": self.budget.get_summary(),
            "mechanism": {
                "type": "Gaussian",
                "sigma": self.mechanism.sigma,
                "noise_multiplier": self.noise_multiplier,
                "max_grad_norm": self.max_grad_norm
            },
            "configuration": {
                "num_clients": self.num_clients,
                "num_rounds": self.num_rounds,
                "per_round_epsilon": self.per_round_epsilon
            }
        }

        if self.clip_stats:
            report["clipping_statistics"] = {
                "avg_clipping_ratio": np.mean([s["clipping_ratio"] for s in self.clip_stats]),
                "avg_norm_before_clip": np.mean([s["avg_norm_before"] for s in self.clip_stats]),
                "avg_norm_after_clip": np.mean([s["avg_norm_after"] for s in self.clip_stats]),
                "total_rounds_processed": len(self.clip_stats)
            }

        return report


class MomentsAccountant:
    """
    Moments Accountant for tight privacy analysis.
    
    Provides tighter privacy bounds than simple composition
    using Rényi Differential Privacy (RDP).
    """

    def __init__(self, noise_multiplier: float, sample_rate: float, orders: Optional[List[float]] = None):
        """
        Args:
            noise_multiplier: Ratio of noise std to sensitivity
            sample_rate: Probability of each client being selected per round
            orders: RDP orders (alpha values) for analysis
        """
        self.noise_multiplier = noise_multiplier
        self.sample_rate = sample_rate
        self.orders = orders or [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        self.rdp_history: List[np.ndarray] = []

    def _compute_rdp_single_step(self) -> np.ndarray:
        """Compute RDP guarantee for a single step of subsampled Gaussian mechanism."""
        rdp = np.zeros(len(self.orders))
        for i, alpha in enumerate(self.orders):
            if alpha <= 1:
                continue
            # RDP of Gaussian mechanism
            rdp_gauss = alpha / (2 * self.noise_multiplier ** 2)
            # Subsampling amplification (simplified)
            if self.sample_rate < 1.0:
                rdp[i] = min(
                    rdp_gauss,
                    math.log(1 + self.sample_rate * (math.exp(rdp_gauss * (alpha - 1)) - 1)) / (alpha - 1)
                )
            else:
                rdp[i] = rdp_gauss
        return rdp

    def step(self):
        """Record one training step."""
        rdp = self._compute_rdp_single_step()
        self.rdp_history.append(rdp)

    def get_epsilon(self, delta: float) -> float:
        """
        Compute epsilon given delta using the accumulated RDP guarantees.
        
        Uses the optimal conversion from RDP to (eps, delta)-DP.
        """
        if not self.rdp_history:
            return 0.0

        # Sum RDP across all steps
        total_rdp = np.sum(self.rdp_history, axis=0)

        # Convert RDP to (eps, delta)-DP
        eps_candidates = []
        for i, alpha in enumerate(self.orders):
            if alpha <= 1:
                continue
            eps = total_rdp[i] + math.log(1 / delta) / (alpha - 1)
            eps_candidates.append(eps)

        return min(eps_candidates) if eps_candidates else float('inf')

    def get_privacy_spent(self, delta: float) -> Dict:
        """Get current privacy expenditure."""
        epsilon = self.get_epsilon(delta)
        return {
            "epsilon": epsilon,
            "delta": delta,
            "num_steps": len(self.rdp_history),
            "optimal_order": self.orders[np.argmin([
                self._get_eps_for_order(i, delta)
                for i in range(len(self.orders))
            ])] if self.rdp_history else None
        }

    def _get_eps_for_order(self, order_idx: int, delta: float) -> float:
        """Helper to get epsilon for a specific order."""
        if not self.rdp_history:
            return float('inf')
        total_rdp = np.sum([h[order_idx] for h in self.rdp_history])
        alpha = self.orders[order_idx]
        if alpha <= 1:
            return float('inf')
        return total_rdp + math.log(1 / delta) / (alpha - 1)
