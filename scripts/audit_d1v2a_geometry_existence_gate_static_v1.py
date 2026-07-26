#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fair_psgg.config import Config
from fair_psgg.models.daniformer import DaniFormer
from fair_psgg.models.fibe_existence_gate import FIBEGeometryExistenceGate


SEED = 3407


class TinyExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(self.conv(image), (16, 16))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit D1-v2a geometry-only existence-gate initialization, "
            "parameterization, gradients, shape, identity, and config isolation."
        )
    )
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--gate-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_model(enabled: bool, mode: str) -> DaniFormer:
    return DaniFormer(
        num_node_outputs=5,
        num_rel_outputs=9,
        extractor=TinyExtractor(),
        transformer_depth=1,
        embed_dim=384,
        patch_size=8,
        feature_shape=(16, 16, 16),
        use_semantics=False,
        use_masks=False,
        bg_ratio_strategy="sum",
        encode_coords=False,
        fibe_enabled=enabled,
        fibe_mode=mode,
        fibe_feature_dim=21,
        fibe_hidden_dim=64,
        fibe_bottleneck_dim=128,
        fibe_existence_delta_max=4.0,
    )


def clone_data(data: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in data.items()
    }


def tensor_norm(parameter: torch.Tensor | None) -> float | None:
    if parameter is None:
        return None
    return float(parameter.detach().float().norm().cpu())


def main() -> None:
    args = parse_args()
    base_config_path = Path(args.base_config).resolve()
    gate_config_path = Path(args.gate_config).resolve()
    output_path = Path(args.output).resolve()

    for path in (base_config_path, gate_config_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    base_raw = json.loads(base_config_path.read_text(encoding="utf-8"))
    gate_raw = json.loads(gate_config_path.read_text(encoding="utf-8"))
    base_cfg = Config.from_file(base_config_path)
    gate_cfg = Config.from_file(gate_config_path)

    base_non_fibe = {k: v for k, v in base_raw.items() if k != "fibe"}
    gate_non_fibe = {k: v for k, v in gate_raw.items() if k != "fibe"}

    config_checks = {
        "non_fibe_config_exactly_equal": base_non_fibe == gate_non_fibe,
        "base_mode_defaults_to_token_residual": (
            base_cfg.fibe.mode == "token_residual"
        ),
        "gate_enabled": gate_cfg.fibe.enabled is True,
        "gate_mode": gate_cfg.fibe.mode == "geometry_existence_gate",
        "feature_dim_21": gate_cfg.fibe.feature_dim == 21,
        "hidden_dim_64": gate_cfg.fibe.hidden_dim == 64,
        "bottleneck_dim_128": gate_cfg.fibe.bottleneck_dim == 128,
        "delta_max_4": gate_cfg.fibe.existence_delta_max == 4.0,
        "test_cache_locked": gate_cfg.fibe.test_cache is None,
        "no_category_fields": not any(
            "category" in key.lower() or "family" in key.lower()
            for key in gate_raw["fibe"]
        ),
        "no_localmap_fields": not any(
            "local" in key.lower() or "map" in key.lower()
            for key in gate_raw["fibe"]
        ),
        "no_margin_loss_fields": not any(
            "margin" in key.lower() and key != "existence_delta_max"
            for key in gate_raw["fibe"]
        ),
    }

    seed_all(args.seed)
    base_model = make_model(False, "token_residual")
    seed_all(args.seed)
    gate_model = make_model(True, "geometry_existence_gate")

    base_state = base_model.state_dict()
    gate_state = gate_model.state_dict()
    common_keys = sorted(set(base_state) & set(gate_state))
    extra_keys = sorted(set(gate_state) - set(base_state))
    missing_keys = sorted(set(base_state) - set(gate_state))

    common_mismatches = [
        key for key in common_keys
        if not torch.equal(base_state[key], gate_state[key])
    ]

    expected_prefix = "fibe_existence_gate."
    gate = gate_model.fibe_existence_gate
    if gate is None:
        raise AssertionError("D1-v2a model did not construct the existence gate")

    gate_parameter_count = sum(p.numel() for p in gate.parameters())
    total_base_parameters = sum(p.numel() for p in base_model.parameters())
    total_gate_parameters = sum(p.numel() for p in gate_model.parameters())

    init_checks = {
        "no_base_keys_missing": not missing_keys,
        "all_extra_keys_are_existence_gate": all(
            key.startswith(expected_prefix) for key in extra_keys
        ),
        "all_common_tensors_exactly_equal": not common_mismatches,
        "token_residual_branch_absent": gate_model.fibe_branch is None,
        "existence_gate_present": gate is not None,
        "delta_head_weight_exact_zero": (
            torch.count_nonzero(gate.delta_head.weight).item() == 0
        ),
        "delta_head_bias_exact_zero": (
            torch.count_nonzero(gate.delta_head.bias).item() == 0
        ),
        "gate_parameter_count_10241": gate_parameter_count == 10241,
        "model_parameter_delta_matches_gate": (
            total_gate_parameters - total_base_parameters == gate_parameter_count
        ),
        "all_gate_parameters_finite": all(
            bool(torch.isfinite(parameter).all())
            for parameter in gate.parameters()
        ),
    }

    data = {
        "img": torch.randn(2, 3, 64, 64),
        "bboxes": torch.tensor([
            [2.0, 2.0, 12.0, 12.0],
            [16.0, 3.0, 14.0, 18.0],
            [5.0, 28.0, 20.0, 16.0],
            [4.0, 5.0, 15.0, 20.0],
            [24.0, 18.0, 18.0, 15.0],
        ]),
        "num_boxes": torch.tensor([3, 2]),
        "pair_ids": torch.tensor([[0, 1], [2, 0], [4, 3]]),
        "box_categories": torch.tensor([0, 1, 2, 3, 4]),
    }
    fibe_features = torch.randn(3, 21)
    mixed_valid = torch.tensor([True, False, True])

    base_model.eval()
    gate_model.eval()
    gate_input = clone_data(data)
    gate_input["fibe_features"] = fibe_features.clone()
    gate_input["fibe_valid"] = mixed_valid.clone()

    with torch.inference_mode():
        base_s, base_o, base_rel = base_model(clone_data(data))
        gate_s, gate_o, gate_rel = gate_model(clone_data(gate_input))
        chunk_s, chunk_o, chunk_rel = gate_model(
            clone_data(gate_input),
            max_relations=1,
        )
        initial_delta = gate.compute_margin_delta(
            fibe_features,
            mixed_valid,
            output_dtype=base_rel.dtype,
            output_device=base_rel.device,
        )

    forward_checks = {
        "subject_logits_exact_identity": torch.equal(base_s, gate_s),
        "object_logits_exact_identity": torch.equal(base_o, gate_o),
        "relation_logits_exact_identity": torch.equal(base_rel, gate_rel),
        "relation_shape_R9": tuple(gate_rel.shape) == (3, 9),
        "margin_delta_shape_R1": tuple(initial_delta.shape) == (3, 1),
        "initial_margin_delta_exact_zero": (
            torch.count_nonzero(initial_delta).item() == 0
        ),
        "invalid_pair_margin_delta_exact_zero": (
            torch.count_nonzero(initial_delta[~mixed_valid]).item() == 0
        ),
        "positive_logits_bitwise_unchanged": torch.equal(
            base_rel[:, 1:], gate_rel[:, 1:]
        ),
        "positive_order_unchanged": torch.equal(
            torch.argsort(base_rel[:, 1:], dim=1),
            torch.argsort(gate_rel[:, 1:], dim=1),
        ),
        "chunk_subject_allclose": torch.allclose(
            chunk_s, gate_s, atol=1e-6, rtol=1e-6
        ),
        "chunk_object_allclose": torch.allclose(
            chunk_o, gate_o, atol=1e-6, rtol=1e-6
        ),
        "chunk_relation_allclose": torch.allclose(
            chunk_rel, gate_rel, atol=1e-6, rtol=1e-6
        ),
    }

    # Standalone two-step gradient audit. With a zero-initialized final head,
    # the encoder is expected to receive zero gradient on step 1 and non-zero
    # gradient after the head has moved on step 2.
    seed_all(args.seed)
    grad_gate = FIBEGeometryExistenceGate(
        feature_dim=21,
        hidden_dim=64,
        bottleneck_dim=128,
        delta_max=4.0,
    )
    optimizer = torch.optim.SGD(grad_gate.parameters(), lr=0.1)
    fixed_logits = torch.randn(8, 9)
    fixed_features = torch.randn(8, 21)
    fixed_valid = torch.tensor([True, True, True, True, True, True, False, False])
    labels = torch.tensor([0, 1, 0, 2, 3, 0, 4, 0])

    optimizer.zero_grad(set_to_none=True)
    step1_output = grad_gate(fixed_logits, fixed_features, fixed_valid)
    step1_loss = F.cross_entropy(step1_output, labels)
    step1_loss.backward()

    step1_head_grad = tensor_norm(grad_gate.delta_head.weight.grad)
    step1_encoder_grad = tensor_norm(grad_gate.encoder[0].weight.grad)
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    step2_output = grad_gate(fixed_logits, fixed_features, fixed_valid)
    step2_loss = F.cross_entropy(step2_output, labels)
    step2_loss.backward()

    step2_head_grad = tensor_norm(grad_gate.delta_head.weight.grad)
    step2_encoder_grad = tensor_norm(grad_gate.encoder[0].weight.grad)
    step2_delta = grad_gate.compute_margin_delta(
        fixed_features,
        fixed_valid,
        output_dtype=fixed_logits.dtype,
        output_device=fixed_logits.device,
    )

    gradient_checks = {
        "step1_head_gradient_nonzero": (
            step1_head_grad is not None and step1_head_grad > 0.0
        ),
        "step1_encoder_gradient_exact_zero_expected": (
            step1_encoder_grad is not None and step1_encoder_grad == 0.0
        ),
        "step2_head_gradient_nonzero": (
            step2_head_grad is not None and step2_head_grad > 0.0
        ),
        "step2_encoder_gradient_nonzero": (
            step2_encoder_grad is not None and step2_encoder_grad > 0.0
        ),
        "step2_delta_finite": bool(torch.isfinite(step2_delta).all()),
        "step2_delta_nonzero_for_some_valid_pairs": (
            torch.count_nonzero(step2_delta[fixed_valid]).item() > 0
        ),
        "step2_invalid_delta_exact_zero": (
            torch.count_nonzero(step2_delta[~fixed_valid]).item() == 0
        ),
        "delta_bound_respected": (
            float(step2_delta.abs().max()) <= grad_gate.delta_max
        ),
        "positive_logits_still_bitwise_unchanged_after_training_step": torch.equal(
            fixed_logits[:, 1:], step2_output[:, 1:]
        ),
        "positive_order_still_unchanged_after_training_step": torch.equal(
            torch.argsort(fixed_logits[:, 1:], dim=1),
            torch.argsort(step2_output[:, 1:], dim=1),
        ),
    }

    all_checks = {
        **config_checks,
        **init_checks,
        **forward_checks,
        **gradient_checks,
    }
    status = "PASS" if all(all_checks.values()) else "FAIL"

    report = {
        "protocol": "FloodPSG D1-v2a geometry existence gate static audit V1",
        "status": status,
        "seed": args.seed,
        "final_test_allowed": False,
        "configs": {
            "base": str(base_config_path),
            "gate": str(gate_config_path),
            "base_sha256": sha256(base_config_path),
            "gate_sha256": sha256(gate_config_path),
        },
        "architecture": {
            "input_feature_dim": 21,
            "output": "single relation-existence margin delta",
            "none_class_index": 0,
            "positive_logits_modified": False,
            "generic_relation_token_residual": False,
            "pair_family_or_category_input": False,
            "local_map": False,
            "hard_negative_margin_loss": False,
            "delta_max": gate.delta_max,
        },
        "parameters": {
            "base_model": total_base_parameters,
            "gate_model": total_gate_parameters,
            "existence_gate": gate_parameter_count,
            "common_state_tensors": len(common_keys),
            "extra_gate_state_tensors": len(extra_keys),
            "common_state_mismatches": common_mismatches[:20],
            "unexpected_missing_base_keys": missing_keys[:20],
        },
        "activations": {
            "initial_delta_min": float(initial_delta.min()),
            "initial_delta_max": float(initial_delta.max()),
            "initial_delta_mean": float(initial_delta.mean()),
            "step2_delta_min": float(step2_delta.min()),
            "step2_delta_max": float(step2_delta.max()),
            "step2_delta_mean": float(step2_delta.mean()),
        },
        "gradients": {
            "step1_loss": float(step1_loss.detach()),
            "step2_loss": float(step2_loss.detach()),
            "step1_delta_head_weight_grad_norm": step1_head_grad,
            "step1_encoder_first_linear_grad_norm": step1_encoder_grad,
            "step2_delta_head_weight_grad_norm": step2_head_grad,
            "step2_encoder_first_linear_grad_norm": step2_encoder_grad,
        },
        "checks": all_checks,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    print()
    print(f"D1-v2a GEOMETRY EXISTENCE GATE STATIC AUDIT V1: {status}")

    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
