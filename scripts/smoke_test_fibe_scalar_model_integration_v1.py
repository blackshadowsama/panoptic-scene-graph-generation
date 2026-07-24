#!/usr/bin/env python3
from __future__ import annotations

import tempfile
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from fair_psgg.config import Config
from fair_psgg.data.fibe_cache import FIBEFeatureCache
from fair_psgg.models.fibe_scalar import FIBEScalarBranch


def make_entry(num_objects: int, feature_dim: int, offset: float):
    features = torch.arange(
        num_objects * num_objects * feature_dim,
        dtype=torch.float32,
    ).reshape(num_objects, num_objects, feature_dim)
    features = features + offset
    valid_pairs = torch.ones((num_objects, num_objects), dtype=torch.bool)
    return {
        "features": features,
        "valid_pairs": valid_pairs,
        "num_objects": num_objects,
    }


def main() -> None:
    torch.manual_seed(3407)
    feature_dim = 21

    with tempfile.TemporaryDirectory(prefix="fibe_scalar_smoke_") as temp_dir:
        cache_path = Path(temp_dir) / "features.pt"
        entry_101 = make_entry(3, feature_dim, 1000.0)
        entry_202 = make_entry(2, feature_dim, 2000.0)
        entry_101["valid_pairs"][2, 0] = False

        torch.save(
            {
                "features_by_image": {
                    "101": entry_101,
                    "202": entry_202,
                }
            },
            cache_path,
        )

        batch = {
            "image_id": torch.tensor([101, 202], dtype=torch.long),
            "num_boxes": torch.tensor([3, 2], dtype=torch.long),
            "num_relations": torch.tensor([2, 1], dtype=torch.long),
            "sampled_relations": torch.tensor(
                [
                    [0, 1, 0],
                    [2, 0, 0],
                    [1, 0, 0],
                ],
                dtype=torch.long,
            ),
        }

        cache = FIBEFeatureCache(cache_path, feature_dim=feature_dim)
        selected, valid = cache.select_for_batch(batch)

        assert selected.shape == (3, feature_dim)
        assert valid.tolist() == [True, False, True]
        assert torch.equal(
            selected[0],
            entry_101["features"][0, 1],
        )
        assert torch.equal(
            selected[1],
            torch.zeros(feature_dim, dtype=torch.float32),
        )
        assert torch.equal(
            selected[2],
            entry_202["features"][1, 0],
        )

    branch = FIBEScalarBranch(
        feature_dim=feature_dim,
        embed_dim=384,
        hidden_dim=64,
        bottleneck_dim=128,
        alpha_max=0.2,
        alpha_init=0.05,
        gate_bias_init=-3.0,
    )

    relation_token = torch.randn(3, 384, requires_grad=True)
    fibe_features = torch.randn(3, feature_dim)
    fibe_valid = torch.tensor([True, False, True], dtype=torch.bool)
    output = branch(
        relation_token=relation_token,
        fibe_features=fibe_features,
        fibe_valid=fibe_valid,
    )

    assert output.shape == relation_token.shape
    assert torch.equal(output[1], relation_token[1])
    assert torch.isclose(
        branch.effective_alpha().detach(),
        torch.tensor(0.05),
        atol=1e-7,
        rtol=0.0,
    )
    assert torch.allclose(
        branch.gate.weight.detach(),
        torch.zeros_like(branch.gate.weight),
    )
    assert torch.allclose(
        branch.gate.bias.detach(),
        torch.full_like(branch.gate.bias, -3.0),
    )

    output.sum().backward()
    assert relation_token.grad is not None
    assert branch.raw_alpha.grad is not None
    assert branch.gate.weight.grad is not None

    legacy_config = Config.model_validate({}, strict=True)
    assert legacy_config.fibe.enabled is False
    assert legacy_config.fibe.feature_dim == 21

    d1_config = Config.model_validate(
        {
            "fibe": {
                "enabled": True,
                "feature_dim": 21,
                "train_cache": "train_features.pt",
                "validation_cache": "validation_features.pt",
                "test_cache": "test_features.pt",
                "hidden_dim": 64,
                "bottleneck_dim": 128,
                "alpha_max": 0.2,
                "alpha_init": 0.05,
                "gate_bias_init": -3.0,
            }
        },
        strict=True,
    )
    assert d1_config.fibe.enabled is True

    print("PASS: FIBE cache selection")
    print("PASS: invalid-pair exact identity")
    print("PASS: alpha/gate initialization")
    print("PASS: gradient flow")
    print("PASS: legacy and D1 config validation")


if __name__ == "__main__":
    main()
