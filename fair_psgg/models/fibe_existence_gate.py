from __future__ import annotations

import torch
from torch import nn


class FIBEGeometryExistenceGate(nn.Module):
    """Geometry-only correction of the NONE-vs-positive relation margin.

    The module consumes the existing 21D FIBE geometry vector and predicts one
    scalar margin correction per relation pair. Positive predicate logits are
    never modified; only the NONE logit (index 0) is shifted. Therefore the
    relative ordering among positive predicates is preserved exactly.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        bottleneck_dim: int = 128,
        delta_max: float = 4.0,
    ):
        super().__init__()

        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        if hidden_dim <= 0 or bottleneck_dim <= 0:
            raise ValueError("hidden_dim and bottleneck_dim must be positive")
        if delta_max <= 0.0:
            raise ValueError(f"delta_max must be positive, got {delta_max}")

        self.feature_dim = int(feature_dim)
        self.delta_max = float(delta_max)

        self.encoder = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.LayerNorm(bottleneck_dim),
        )
        self.delta_head = nn.Linear(bottleneck_dim, 1)

        # Strictly neutral initialization: every valid and invalid pair produces
        # an exact zero correction before the first optimizer step.
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def compute_margin_delta(
        self,
        fibe_features: torch.Tensor,
        fibe_valid: torch.Tensor,
        *,
        output_dtype: torch.dtype | None = None,
        output_device: torch.device | None = None,
    ) -> torch.Tensor:
        """Return one bounded existence-margin correction per relation pair."""

        if fibe_features.ndim != 2:
            raise ValueError(
                f"fibe_features must be [R,D], got {tuple(fibe_features.shape)}"
            )
        if int(fibe_features.shape[1]) != self.feature_dim:
            raise ValueError(
                f"Expected FIBE feature_dim={self.feature_dim}, "
                f"got {int(fibe_features.shape[1])}"
            )
        if fibe_valid.ndim != 1 or fibe_valid.shape[0] != fibe_features.shape[0]:
            raise ValueError(
                f"fibe_valid must be [R], got {tuple(fibe_valid.shape)}"
            )

        device = output_device if output_device is not None else fibe_features.device
        dtype = output_dtype if output_dtype is not None else fibe_features.dtype

        features = fibe_features.to(device=device, dtype=dtype)
        valid = fibe_valid.to(device=device, dtype=torch.bool)

        hidden = self.encoder(features)
        raw_delta = self.delta_head(hidden)
        margin_delta = self.delta_max * torch.tanh(raw_delta)

        # Invalid pairs are strict identity cases.
        margin_delta = torch.where(
            valid[:, None],
            margin_delta,
            torch.zeros_like(margin_delta),
        )
        return margin_delta

    @staticmethod
    def apply_margin_delta(
        base_logits: torch.Tensor,
        margin_delta: torch.Tensor,
    ) -> torch.Tensor:
        """Apply delta by shifting NONE only; positive logits stay bitwise equal."""

        if base_logits.ndim != 2:
            raise ValueError(
                f"base_logits must be [R,C], got {tuple(base_logits.shape)}"
            )
        if base_logits.shape[1] < 2:
            raise ValueError(
                "Relation logits must contain NONE plus at least one positive class"
            )
        if margin_delta.ndim != 2 or margin_delta.shape != (base_logits.shape[0], 1):
            raise ValueError(
                "margin_delta must be [R,1], got "
                f"{tuple(margin_delta.shape)} for logits {tuple(base_logits.shape)}"
            )

        output = base_logits.clone()
        output[:, :1] = base_logits[:, :1] - margin_delta
        return output

    def forward(
        self,
        base_logits: torch.Tensor,
        fibe_features: torch.Tensor,
        fibe_valid: torch.Tensor,
    ) -> torch.Tensor:
        if fibe_features.shape[0] != base_logits.shape[0]:
            raise ValueError(
                "FIBE relation count mismatch: "
                f"logits={base_logits.shape[0]}, "
                f"features={fibe_features.shape[0]}"
            )

        margin_delta = self.compute_margin_delta(
            fibe_features=fibe_features,
            fibe_valid=fibe_valid,
            output_dtype=base_logits.dtype,
            output_device=base_logits.device,
        )
        return self.apply_margin_delta(base_logits, margin_delta)
