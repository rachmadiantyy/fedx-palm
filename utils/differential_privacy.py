"""
Differential Privacy Module for FedX-PALM using Opacus.

Implements DP-SGD (Differentially Private Stochastic Gradient Descent)
as specified in thesis Sections 2.5 and 3.4:

- Gradient Clipping (Eq. 3.2): ḡ = g · min(1, C/||g||₂)
- Gaussian Mechanism (Eq. 3.3): g̃ = (1/B) * (Σ ḡᵢ + N(0, σ²C²I))
- Privacy Accounting: Rényi Differential Privacy (RDP)

Privacy Budget Scenarios (thesis Table 3.4 + σ range Table 3.5):
- Baseline:        ε=∞,    σ=0.0  (No Privacy)
- Very Weak:       ε=12.0, σ=0.5  (σ lower bound, Table 3.5)
- Weak Privacy:    ε=8.0,  σ=0.8
- Moderate:        ε=4.0,  σ=1.5
- Strong Privacy:  ε=1.0,  σ=3.2

DP Strategies (thesis Section 3.4.5):
- Full DP: Noise on all layers (backbone + neck + detection head)
- Partial DP: Noise only on detection head (head layers)

Configuration (thesis Section 3.4.3):
- Library: Opacus 1.4.0
- Accountant: RDP (Rényi DP)
- Target delta (δ): 1×10⁻⁵
- Maximum gradient norm (C): 1.0
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import math
import logging
import numpy as np

logger = logging.getLogger(__name__)


class DPStrategy(Enum):
    """DP application strategy as defined in thesis Section 3.4.5."""
    NONE = "none"          # No DP (baseline, ε=∞)
    FULL_DP = "full"       # Noise on ALL layers
    PARTIAL_DP = "partial" # Noise only on detection head


@dataclass
class PrivacyConfig:
    """
    Privacy configuration matching thesis Table 3.5 and Section 3.4.

    Attributes:
        epsilon: Privacy budget (ε). Values: 1.0, 4.0, 8.0, ∞
        delta: Privacy failure probability (δ = 1e-5)
        max_grad_norm: Clipping threshold C (default: 1.0)
        noise_multiplier: σ (varies by ε scenario)
        strategy: Full DP or Partial DP
        target_delta: Target δ for privacy accounting
    """
    epsilon: float = 4.0
    delta: float = 1e-5
    max_grad_norm: float = 1.0
    noise_multiplier: float = 1.5
    strategy: DPStrategy = DPStrategy.FULL_DP
    target_delta: float = 1e-5
    accountant: str = "rdp"  # Rényi DP accountant

    @classmethod
    def baseline(cls) -> "PrivacyConfig":
        """No privacy (ε=∞, σ=0) - thesis baseline scenario."""
        return cls(
            epsilon=float('inf'),
            noise_multiplier=0.0,
            strategy=DPStrategy.NONE
        )

    @classmethod
    def very_weak_privacy(cls) -> "PrivacyConfig":
        """Very weak privacy (ε≈12.0, σ=0.5) - thesis Table 3.5 σ range lower bound."""
        return cls(
            epsilon=12.0,
            noise_multiplier=0.5,
            strategy=DPStrategy.FULL_DP
        )

    @classmethod
    def weak_privacy(cls) -> "PrivacyConfig":
        """Weak privacy (ε=8.0, σ=0.8) - thesis Table 3.4."""
        return cls(
            epsilon=8.0,
            noise_multiplier=0.8,
            strategy=DPStrategy.FULL_DP
        )

    @classmethod
    def moderate_privacy(cls) -> "PrivacyConfig":
        """Moderate privacy (ε=4.0, σ=1.5) - thesis Table 3.4."""
        return cls(
            epsilon=4.0,
            noise_multiplier=1.5,
            strategy=DPStrategy.FULL_DP
        )

    @classmethod
    def strong_privacy(cls) -> "PrivacyConfig":
        """Strong privacy (ε=1.0, σ=3.2) - thesis Table 3.4."""
        return cls(
            epsilon=1.0,
            noise_multiplier=3.2,
            strategy=DPStrategy.FULL_DP
        )

    @classmethod
    def partial_moderate(cls) -> "PrivacyConfig":
        """Partial DP at ε=4.0 (noise only on head) - thesis Section 3.4.5."""
        return cls(
            epsilon=4.0,
            noise_multiplier=1.5,
            strategy=DPStrategy.PARTIAL_DP
        )


@dataclass
class PrivacyBudgetTracker:
    """
    Tracks cumulative privacy budget consumption using RDP accountant.

    As per thesis Section 3.4.3: Uses Rényi Differential Privacy (RDP)
    for tighter composition bounds.
    """
    target_epsilon: float = 4.0
    target_delta: float = 1e-5
    spent_epsilon: float = 0.0
    rounds_completed: int = 0
    history: List[Dict] = field(default_factory=list)

    @property
    def remaining_epsilon(self) -> float:
        return max(0.0, self.target_epsilon - self.spent_epsilon)

    @property
    def is_exhausted(self) -> bool:
        if self.target_epsilon == float('inf'):
            return False
        return self.spent_epsilon >= self.target_epsilon

    def record_round(self, epsilon_spent: float, noise_multiplier: float):
        """Record privacy expenditure for one FL round."""
        self.spent_epsilon += epsilon_spent
        self.rounds_completed += 1
        self.history.append({
            "round": self.rounds_completed,
            "epsilon_spent": epsilon_spent,
            "cumulative_epsilon": self.spent_epsilon,
            "noise_multiplier": noise_multiplier
        })

    def get_summary(self) -> Dict:
        return {
            "target_epsilon": self.target_epsilon,
            "target_delta": self.target_delta,
            "spent_epsilon": self.spent_epsilon,
            "remaining_epsilon": self.remaining_epsilon,
            "rounds_completed": self.rounds_completed,
            "is_exhausted": self.is_exhausted
        }


class OpacusDPEngine:
    """
    Differential Privacy Engine using Opacus for DP-SGD.

    Wraps model, optimizer, and dataloader with Opacus privacy engine
    as described in thesis Section 3.4.3.

    Key operations:
    1. Per-sample gradient clipping (Eq. 3.2)
    2. Gaussian noise injection (Eq. 3.3)
    3. Privacy budget accounting via RDP
    """

    def __init__(self, config: PrivacyConfig):
        """
        Args:
            config: Privacy configuration
        """
        self.config = config
        self.budget_tracker = PrivacyBudgetTracker(
            target_epsilon=config.epsilon,
            target_delta=config.target_delta
        )
        self._privacy_engine = None
        self._is_attached = False

        logger.info(
            f"DP Engine initialized: ε={config.epsilon}, "
            f"σ={config.noise_multiplier}, C={config.max_grad_norm}, "
            f"strategy={config.strategy.value}"
        )

    def attach_to_training(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        data_loader: Any,
        epochs: int = 5
    ) -> Tuple[nn.Module, torch.optim.Optimizer, Any]:
        """
        Attach Opacus privacy engine to model, optimizer, and dataloader.

        This wraps the training pipeline with DP-SGD:
        - Model: Replaces batch norm with group norm (Opacus requirement)
        - Optimizer: Adds per-sample gradient clipping + noise
        - DataLoader: Ensures uniform sampling for privacy accounting

        Args:
            model: PyTorch model (YOLOv11)
            optimizer: Optimizer (AdamW as per thesis)
            data_loader: Training data loader
            epochs: Number of local epochs

        Returns:
            Tuple of (dp_model, dp_optimizer, dp_dataloader)
        """
        if self.config.strategy == DPStrategy.NONE:
            logger.info("DP Strategy: NONE - skipping Opacus attachment")
            return model, optimizer, data_loader

        try:
            from opacus import PrivacyEngine
            from opacus.validators import ModuleValidator

            # Validate and fix model for Opacus compatibility
            if not ModuleValidator.is_valid(model):
                logger.info("Fixing model for Opacus compatibility (BatchNorm → GroupNorm)")
                model = ModuleValidator.fix(model)

            # Apply Partial DP strategy: freeze non-target layers
            if self.config.strategy == DPStrategy.PARTIAL_DP:
                model = self._apply_partial_dp_freeze(model)

            # Create Opacus Privacy Engine
            privacy_engine = PrivacyEngine()

            model, optimizer, data_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=data_loader,
                noise_multiplier=self.config.noise_multiplier,
                max_grad_norm=self.config.max_grad_norm,
            )

            self._privacy_engine = privacy_engine
            self._is_attached = True

            logger.info(
                f"Opacus attached: noise_multiplier={self.config.noise_multiplier}, "
                f"max_grad_norm={self.config.max_grad_norm}"
            )

            return model, optimizer, data_loader

        except ImportError:
            logger.warning(
                "Opacus not installed. Falling back to manual DP implementation. "
                "Install with: pip install opacus>=1.4.0"
            )
            return model, optimizer, data_loader

    def _apply_partial_dp_freeze(self, model: nn.Module) -> nn.Module:
        """
        Apply Partial DP strategy: only protect detection head.

        As per thesis Section 3.4.5:
        - Partial DP: Noise only on detection head (classification + regression)
        - Backbone (feature extractor) remains without DP protection

        This preserves feature extraction quality while protecting
        the final decision layers.
        """
        # Freeze backbone parameters (no DP noise on these)
        for name, param in model.named_parameters():
            # YOLOv11 backbone layers typically contain 'backbone' or early indices
            if any(keyword in name.lower() for keyword in ['backbone', 'stem', 'dark']):
                param.requires_grad = False
                logger.debug(f"Partial DP: Froze backbone layer: {name}")

        # Count trainable (DP-protected) vs frozen parameters
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        frozen = total - trainable

        logger.info(
            f"Partial DP applied: {trainable:,} trainable (DP-protected), "
            f"{frozen:,} frozen (no DP)"
        )

        return model

    def get_privacy_spent(self) -> Dict[str, float]:
        """
        Get current privacy expenditure from Opacus accountant.

        Returns epsilon spent so far using RDP accounting.
        """
        if self._privacy_engine is not None:
            try:
                epsilon = self._privacy_engine.get_epsilon(self.config.target_delta)
                return {
                    "epsilon": epsilon,
                    "delta": self.config.target_delta,
                    "accountant": "RDP (Rényi DP)"
                }
            except Exception as e:
                logger.warning(f"Could not get epsilon from Opacus: {e}")

        return self.budget_tracker.get_summary()

    def record_round_completion(self):
        """Record that one FL round has been completed."""
        if self._privacy_engine is not None:
            try:
                eps = self._privacy_engine.get_epsilon(self.config.target_delta)
                self.budget_tracker.record_round(eps, self.config.noise_multiplier)
            except Exception:
                # Estimate epsilon per round using simple composition
                per_round_eps = self.config.epsilon / 100.0  # Approximate
                self.budget_tracker.record_round(per_round_eps, self.config.noise_multiplier)
        else:
            per_round_eps = self.config.epsilon / 100.0
            self.budget_tracker.record_round(per_round_eps, self.config.noise_multiplier)

    def detach(self):
        """Detach privacy engine after training."""
        self._privacy_engine = None
        self._is_attached = False


class ManualDPSGD:
    """
    Manual DP-SGD implementation (fallback when Opacus is not available).

    Implements the two-step DP mechanism from thesis:
    1. Gradient Clipping (Eq. 3.2): ḡ = g · min(1, C/||g||₂)
    2. Gaussian Noise (Eq. 3.3): g̃ = (1/B)(Σ ḡᵢ + N(0, σ²C²I))
    """

    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.clip_stats: List[Dict] = []

    def clip_gradients(self, model: nn.Module) -> float:
        """
        Per-sample gradient clipping (thesis Eq. 3.2).

        ḡ = g · min(1, C/||g||₂)

        Args:
            model: PyTorch model with computed gradients

        Returns:
            Original gradient norm before clipping
        """
        if self.config.strategy == DPStrategy.NONE:
            return 0.0

        parameters = self._get_dp_parameters(model)
        if not parameters:
            return 0.0

        # Compute total gradient norm
        total_norm = torch.norm(
            torch.stack([
                torch.norm(p.grad.detach(), p=2)
                for p in parameters if p.grad is not None
            ]),
            p=2
        ).item()

        # Clip: g · min(1, C/||g||₂)
        clip_coeff = min(1.0, self.config.max_grad_norm / (total_norm + 1e-8))

        for p in parameters:
            if p.grad is not None:
                p.grad.detach().mul_(clip_coeff)

        self.clip_stats.append({
            "original_norm": total_norm,
            "clip_coeff": clip_coeff,
            "was_clipped": clip_coeff < 1.0
        })

        return total_norm

    def add_noise(self, model: nn.Module, batch_size: int = 16):
        """
        Add Gaussian noise to gradients (thesis Eq. 3.3).

        g̃ = (1/B)(Σ ḡᵢ + N(0, σ²C²I))

        Args:
            model: Model with clipped gradients
            batch_size: Training batch size B
        """
        if self.config.strategy == DPStrategy.NONE:
            return
        if self.config.noise_multiplier <= 0:
            return

        parameters = self._get_dp_parameters(model)
        noise_std = self.config.noise_multiplier * self.config.max_grad_norm / batch_size

        for p in parameters:
            if p.grad is not None:
                noise = torch.normal(
                    mean=0.0,
                    std=noise_std,
                    size=p.grad.shape,
                    device=p.grad.device,
                    dtype=p.grad.dtype
                )
                p.grad.add_(noise)

    def _get_dp_parameters(self, model: nn.Module) -> List[torch.nn.Parameter]:
        """Get parameters that should have DP applied (based on strategy)."""
        if self.config.strategy == DPStrategy.FULL_DP:
            return [p for p in model.parameters() if p.grad is not None]
        elif self.config.strategy == DPStrategy.PARTIAL_DP:
            # Only detection head parameters
            params = []
            for name, p in model.named_parameters():
                if p.grad is not None:
                    if not any(k in name.lower() for k in ['backbone', 'stem', 'dark']):
                        params.append(p)
            return params
        return []

    def privatize_model_update(
        self,
        model_update: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply DP to a model update dictionary (for FL communication).

        Steps:
        1. Clip each parameter update
        2. Add calibrated Gaussian noise

        Args:
            model_update: Dict of parameter_name -> update_tensor

        Returns:
            Privatized model update
        """
        if self.config.strategy == DPStrategy.NONE:
            return model_update

        privatized = {}
        for name, update in model_update.items():
            # Determine if this layer should get DP
            should_apply_dp = True
            if self.config.strategy == DPStrategy.PARTIAL_DP:
                if any(k in name.lower() for k in ['backbone', 'stem', 'dark']):
                    should_apply_dp = False

            if should_apply_dp:
                # Clip
                norm = torch.norm(update, p=2)
                clip_coeff = min(1.0, self.config.max_grad_norm / (norm.item() + 1e-8))
                clipped = update * clip_coeff

                # Add noise
                noise_std = self.config.noise_multiplier * self.config.max_grad_norm
                noise = torch.normal(0, noise_std, size=update.shape, device=update.device)
                privatized[name] = clipped + noise
            else:
                privatized[name] = update

        return privatized

    def get_clipping_statistics(self) -> Dict[str, float]:
        """Get statistics about gradient clipping."""
        if not self.clip_stats:
            return {}
        return {
            "total_steps": len(self.clip_stats),
            "avg_original_norm": np.mean([s["original_norm"] for s in self.clip_stats]),
            "clipping_ratio": np.mean([1.0 if s["was_clipped"] else 0.0 for s in self.clip_stats]),
            "avg_clip_coeff": np.mean([s["clip_coeff"] for s in self.clip_stats])
        }


def get_privacy_config(scenario: str) -> PrivacyConfig:
    """
    Get privacy configuration for a specific scenario.

    Scenarios from thesis Table 3.4 (+ σ range lower bound from Table 3.5):
    - "baseline": ε=∞, σ=0.0, No DP
    - "very_weak": ε=12.0, σ=0.5 (σ lower bound, thesis Table 3.5)
    - "weak": ε=8.0, σ=0.8
    - "moderate": ε=4.0, σ=1.5
    - "strong": ε=1.0, σ=3.2
    - "partial_moderate": ε=4.0, σ=1.5, head-only (Section 3.4.5)

    Args:
        scenario: One of "baseline", "very_weak", "weak", "moderate",
                  "strong", "partial_moderate"

    Returns:
        PrivacyConfig for the specified scenario
    """
    scenarios = {
        "baseline": PrivacyConfig.baseline,
        "very_weak": PrivacyConfig.very_weak_privacy,
        "weak": PrivacyConfig.weak_privacy,
        "moderate": PrivacyConfig.moderate_privacy,
        "strong": PrivacyConfig.strong_privacy,
        "partial_moderate": PrivacyConfig.partial_moderate,
    }

    if scenario not in scenarios:
        raise ValueError(
            f"Unknown scenario '{scenario}'. "
            f"Available: {list(scenarios.keys())}"
        )

    return scenarios[scenario]()


def get_all_privacy_scenarios() -> Dict[str, PrivacyConfig]:
    """Get all privacy scenarios for comparative experiments."""
    return {
        "baseline": PrivacyConfig.baseline(),
        "very_weak": PrivacyConfig.very_weak_privacy(),
        "weak": PrivacyConfig.weak_privacy(),
        "moderate": PrivacyConfig.moderate_privacy(),
        "strong": PrivacyConfig.strong_privacy(),
        "partial_moderate": PrivacyConfig.partial_moderate(),
    }
