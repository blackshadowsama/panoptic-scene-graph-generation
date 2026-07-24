from __future__ import annotations

import math

import torch
from torch import nn


class FIBEScalarBranch(nn.Module):
    """Pure-geometry scalar residual branch for DSFormer relation tokens."""

    def __init__(
        self,
        feature_dim: int,
        embed_dim: int,
        hidden_dim: int = 64,
        bottleneck_dim: int = 128,
        alpha_max: float = 0.2,
        alpha_init: float = 0.05,
        gate_bias_init: float = -3.0,
    ):
        super().__init__()

        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if hidden_dim <= 0 or bottleneck_dim <= 0:
            raise ValueError("hidden_dim and bottleneck_dim must be positive")
        if not 0.0 < alpha_init < alpha_max:
            raise ValueError(
                f"Require 0 < alpha_init < alpha_max, got "
                f"{alpha_init} and {alpha_max}"
            )

        self.feature_dim = int(feature_dim)
        self.embed_dim = int(embed_dim)
        self.alpha_max = float(alpha_max)

        self.encoder = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.LayerNorm(bottleneck_dim),
        )
        self.projection = nn.Sequential(
            nn.Linear(bottleneck_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )
        self.gate = nn.Linear(bottleneck_dim, 1)

        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, float(gate_bias_init))

        alpha_fraction = float(alpha_init) / self.alpha_max
        raw_alpha_init = math.log(alpha_fraction / (1.0 - alpha_fraction))
        self.raw_alpha = nn.Parameter(
            torch.tensor(raw_alpha_init, dtype=torch.float32)
        )

    def effective_alpha(self) -> torch.Tensor:
        return self.alpha_max * torch.sigmoid(self.raw_alpha)

    def forward(
        self,
        relation_token: torch.Tensor,
        fibe_features: torch.Tensor,
        fibe_valid: torch.Tensor,
    ) -> torch.Tensor:
        if relation_token.ndim != 2:
            raise ValueError(
                f"relation_token must be [R,E], got {tuple(relation_token.shape)}"
            )
        if fibe_features.ndim != 2:
            raise ValueError(
                f"fibe_features must be [R,D], got {tuple(fibe_features.shape)}"
            )
        if fibe_features.shape[0] != relation_token.shape[0]:
            raise ValueError(
                "FIBE relation count mismatch: "
                f"tokens={relation_token.shape[0]}, "
                f"features={fibe_features.shape[0]}"
            )
        if int(fibe_features.shape[1]) != self.feature_dim:
            raise ValueError(
                f"Expected FIBE feature_dim={self.feature_dim}, "
                f"got {int(fibe_features.shape[1])}"
            )
        if fibe_valid.ndim != 1 or fibe_valid.shape[0] != relation_token.shape[0]:
            raise ValueError(
                f"fibe_valid must be [R], got {tuple(fibe_valid.shape)}"
            )

        fibe_features = fibe_features.to(
            device=relation_token.device,
            dtype=relation_token.dtype,
        )
        fibe_valid = fibe_valid.to(
            device=relation_token.device,
            dtype=torch.bool,
        )

        hidden = self.encoder(fibe_features)
        residual = self.projection(hidden)
        gate = torch.sigmoid(self.gate(hidden))
        delta = (
            self.effective_alpha().to(dtype=relation_token.dtype)
            * gate
            * residual
        )

        # Guarantees exact identity for invalid pairs.
        delta = torch.where(
            fibe_valid[:, None],
            delta,
            torch.zeros_like(delta),
        )
        return relation_token + delta
