#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fair_psgg.config import Config
from fair_psgg.data.split_batch import split_batch
from fair_psgg.trainer import Trainer, prepare_batch


PROTOCOL = "FloodPSG D1-v2a real DataLoader activation audit V1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_norm(value: torch.Tensor | None) -> float | None:
    if value is None:
        return None
    return float(value.detach().float().norm().cpu().item())


def finite_tensor(value: Any) -> bool:
    return isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all().item())


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Audit D1-v2a on a real FloodPSG sampled training sub-batch. "
            "The script performs two optimizer steps in an ephemeral Trainer, "
            "writes one JSON report, and never reads the final test split."
        )
    )
    p.add_argument("--config", required=True)
    p.add_argument("--annotation", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--epoch", type=int, default=0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--max-raw-batches", type=int, default=64)
    p.add_argument("--expected-gate-parameters", type=int, default=10241)
    return p


def main() -> None:
    args = parser().parse_args()

    config_path = Path(args.config).expanduser().resolve()
    annotation_path = Path(args.annotation).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    for path in (config_path, annotation_path, data_root):
        if not path.exists():
            raise FileNotFoundError(path)

    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing audit output: {output_path}"
        )

    required_env = {
        "FLOODPSG_NEG_MODE": "flood_hn",
        "FLOODPSG_HARDNEG_FRACTION": "0.50",
        "FLOODPSG_ZERO_NEG_PER_IMAGE": "2",
        "FLOODPSG_KEEP_ZERO_REL": "1",
        "FLOODPSG_DETERMINISTIC_SAMPLING": "1",
        "FLOODPSG_TRAIN_SEED": str(args.seed),
        "FLOODPSG_SAMPLING_SEED": str(args.seed),
        "FLOODPSG_CURRENT_EPOCH": str(args.epoch),
    }
    env_checks = {
        key: os.environ.get(key) == expected
        for key, expected in required_env.items()
    }

    hardneg_index = os.environ.get("FLOODPSG_HARDNEG_INDEX")
    hardneg_index_path = (
        Path(hardneg_index).expanduser().resolve()
        if hardneg_index
        else None
    )
    env_checks["FLOODPSG_HARDNEG_INDEX_present"] = hardneg_index_path is not None
    env_checks["FLOODPSG_HARDNEG_INDEX_exists"] = (
        hardneg_index_path is not None and hardneg_index_path.is_file()
    )

    if not all(env_checks.values()):
        raise RuntimeError(
            "Formal FloodHN environment is incomplete or mismatched: "
            + json.dumps(env_checks, ensure_ascii=False, sort_keys=True)
        )

    seed_all(args.seed)

    config = Config.from_file(config_path)

    config_checks = {
        "fibe_enabled": bool(config.fibe.enabled),
        "mode_geometry_existence_gate": (
            str(config.fibe.mode) == "geometry_existence_gate"
        ),
        "feature_dim_21": int(config.fibe.feature_dim) == 21,
        "test_cache_none": config.fibe.test_cache is None,
        "batch_size_8": int(config.batch_size) == 8,
        "rels_per_batch_128": int(config.rels_per_batch) == 128,
        "lr_3_73e_05": abs(float(config.lr) - 3.73e-05) < 1e-15,
        "weight_decay_0_04": abs(float(config.weight_decay) - 0.04) < 1e-15,
        "neg_ratio_1": abs(float(config.neg_ratio) - 1.0) < 1e-15,
    }
    if not all(config_checks.values()):
        raise RuntimeError(
            "D1-v2a config contract mismatch: "
            + json.dumps(config_checks, ensure_ascii=False, sort_keys=True)
        )

    trainer = Trainer(
        anno_path=annotation_path,
        img_dir=data_root,
        seg_dir=data_root,
        config=config,
        out_dir=None,
        num_workers=args.workers,
        dump_data=False,
        start_state_dict=None,
        hide_batch_progress=True,
    )

    model_checks = {
        "train_cache_present": trainer.fibe_train_cache is not None,
        "validation_cache_present": trainer.fibe_val_cache is not None,
        "token_residual_branch_absent": trainer.model.fibe_branch is None,
        "existence_gate_present": trainer.model.fibe_existence_gate is not None,
        "checkpoint_metric_rel_mean_recall_50": (
            trainer.critical_metric == "rel_mean_recall/50"
        ),
    }
    if not all(model_checks.values()):
        raise RuntimeError(
            "Trainer/model contract mismatch: "
            + json.dumps(model_checks, ensure_ascii=False, sort_keys=True)
        )

    gate = trainer.model.fibe_existence_gate
    gate_parameter_count = sum(p.numel() for p in gate.parameters())

    if gate_parameter_count != args.expected_gate_parameters:
        raise RuntimeError(
            f"Expected {args.expected_gate_parameters} gate parameters, "
            f"got {gate_parameter_count}"
        )

    # The deterministic sampler reads this value when the DataLoader iterator
    # is constructed.
    os.environ["FLOODPSG_CURRENT_EPOCH"] = str(args.epoch)

    selected_batch = None
    selected_input = None
    selected_targets = None
    selected_raw_index = None
    selected_sub_index = None
    fallback = None

    for raw_batch_index, raw_batch in enumerate(trainer.train_loader):
        if raw_batch_index >= args.max_raw_batches:
            break

        for sub_batch_index, sub_batch in enumerate(
            split_batch(raw_batch, max_relations=trainer.rels_per_batch)
        ):
            model_input, sbj_target, obj_target, rel_target = prepare_batch(
                sub_batch,
                trainer.device,
                fibe_cache=trainer.fibe_train_cache,
            )

            valid = model_input["fibe_valid"]
            has_valid = bool(valid.any().item())
            has_invalid = bool((~valid).any().item())
            has_none = bool((rel_target[:, 0] > 0.5).any().item())
            has_positive = bool((rel_target[:, 1:].sum(dim=1) > 0).any().item())

            if has_valid and has_none and has_positive:
                candidate = (
                    sub_batch,
                    model_input,
                    (sbj_target, obj_target, rel_target),
                    raw_batch_index,
                    sub_batch_index,
                )
                if fallback is None:
                    fallback = candidate
                if has_invalid:
                    selected_batch, selected_input, selected_targets, selected_raw_index, selected_sub_index = candidate
                    break

        if selected_batch is not None:
            break

    if selected_batch is None and fallback is not None:
        selected_batch, selected_input, selected_targets, selected_raw_index, selected_sub_index = fallback

    if selected_batch is None:
        raise RuntimeError(
            "No real sub-batch with valid FIBE features and both NONE/positive "
            f"targets was found in the first {args.max_raw_batches} raw batches."
        )

    sbj_target, obj_target, rel_target = selected_targets
    fibe_features = selected_input["fibe_features"]
    fibe_valid = selected_input["fibe_valid"]
    num_relations = int(selected_batch["num_relations"].sum().item())

    cache_checks = {
        "feature_shape_R21": tuple(fibe_features.shape) == (num_relations, 21),
        "valid_shape_R": tuple(fibe_valid.shape) == (num_relations,),
        "features_finite": bool(torch.isfinite(fibe_features).all().item()),
        "at_least_one_valid_pair": bool(fibe_valid.any().item()),
        "invalid_features_exact_zero": (
            True
            if not bool((~fibe_valid).any().item())
            else torch.count_nonzero(fibe_features[~fibe_valid]).item() == 0
        ),
        "contains_none_target": bool((rel_target[:, 0] > 0.5).any().item()),
        "contains_positive_target": bool(
            (rel_target[:, 1:].sum(dim=1) > 0).any().item()
        ),
    }
    if not all(cache_checks.values()):
        raise RuntimeError(
            "Real cache/batch alignment failed: "
            + json.dumps(cache_checks, ensure_ascii=False, sort_keys=True)
        )

    captures: list[dict[str, torch.Tensor]] = []

    def gate_hook(module, hook_args, hook_kwargs, output):
        base_logits = hook_kwargs.get("base_logits")
        features = hook_kwargs.get("fibe_features")
        valid = hook_kwargs.get("fibe_valid")
        if base_logits is None or features is None or valid is None:
            raise RuntimeError("Existence-gate hook did not receive keyword inputs")

        delta = module.compute_margin_delta(
            fibe_features=features,
            fibe_valid=valid,
            output_dtype=base_logits.dtype,
            output_device=base_logits.device,
        )
        captures.append(
            {
                "base_logits": base_logits.detach().clone(),
                "final_logits": output.detach().clone(),
                "margin_delta": delta.detach().clone(),
                "fibe_valid": valid.detach().clone(),
            }
        )

    hook_handle = gate.register_forward_hook(gate_hook, with_kwargs=True)

    trainer.model.train()

    # Step 1: strictly neutral forward, then backward.
    trainer.optimizer.zero_grad(set_to_none=True)
    captures.clear()
    step1 = trainer._common_forward(
        selected_batch,
        fibe_cache=trainer.fibe_train_cache,
    )

    if len(captures) != 1:
        raise RuntimeError(f"Expected one gate call on step 1, got {len(captures)}")

    cap1 = captures[0]
    step1["loss"].backward()

    step1_head_grad = tensor_norm(gate.delta_head.weight.grad)
    step1_head_bias_grad = tensor_norm(gate.delta_head.bias.grad)
    step1_encoder_grad = tensor_norm(gate.encoder[0].weight.grad)

    step1_checks = {
        "loss_finite": finite_tensor(step1["loss"]),
        "node_loss_finite": finite_tensor(step1["node_loss"]),
        "rel_loss_finite": finite_tensor(step1["rel_loss"]),
        "subject_output_finite": finite_tensor(step1["sbj_out"]),
        "object_output_finite": finite_tensor(step1["obj_out"]),
        "relation_output_finite": finite_tensor(step1["rel_out"]),
        "relation_shape_R9": tuple(step1["rel_out"].shape) == (num_relations, 9),
        "hook_output_matches_forward": torch.equal(
            cap1["final_logits"], step1["rel_out"].detach()
        ),
        "initial_margin_delta_exact_zero": (
            torch.count_nonzero(cap1["margin_delta"]).item() == 0
        ),
        "initial_relation_logits_exact_identity": torch.equal(
            cap1["base_logits"], cap1["final_logits"]
        ),
        "initial_positive_logits_bitwise_unchanged": torch.equal(
            cap1["base_logits"][:, 1:], cap1["final_logits"][:, 1:]
        ),
        "initial_positive_order_unchanged": torch.equal(
            torch.argsort(cap1["base_logits"][:, 1:], dim=1),
            torch.argsort(cap1["final_logits"][:, 1:], dim=1),
        ),
        "initial_invalid_delta_exact_zero": (
            True
            if not bool((~cap1["fibe_valid"]).any().item())
            else torch.count_nonzero(
                cap1["margin_delta"][~cap1["fibe_valid"]]
            ).item() == 0
        ),
        "step1_head_weight_gradient_nonzero": (
            step1_head_grad is not None and step1_head_grad > 0.0
        ),
        "step1_head_bias_gradient_nonzero": (
            step1_head_bias_grad is not None and step1_head_bias_grad > 0.0
        ),
        "step1_encoder_gradient_exact_zero_expected": (
            step1_encoder_grad is not None and step1_encoder_grad == 0.0
        ),
    }
    if not all(step1_checks.values()):
        raise RuntimeError(
            "Real-batch step-1 audit failed: "
            + json.dumps(step1_checks, ensure_ascii=False, sort_keys=True)
        )

    trainer.optimizer.step()

    # Step 2: same real sub-batch after one actual AdamW step.
    trainer.optimizer.zero_grad(set_to_none=True)
    captures.clear()
    step2 = trainer._common_forward(
        selected_batch,
        fibe_cache=trainer.fibe_train_cache,
    )

    if len(captures) != 1:
        raise RuntimeError(f"Expected one gate call on step 2, got {len(captures)}")

    cap2 = captures[0]
    step2["loss"].backward()

    step2_head_grad = tensor_norm(gate.delta_head.weight.grad)
    step2_head_bias_grad = tensor_norm(gate.delta_head.bias.grad)
    step2_encoder_grad = tensor_norm(gate.encoder[0].weight.grad)

    valid_delta = cap2["margin_delta"][cap2["fibe_valid"]]
    max_abs_delta = (
        float(valid_delta.detach().abs().max().cpu().item())
        if valid_delta.numel() > 0
        else 0.0
    )

    step2_checks = {
        "loss_finite": finite_tensor(step2["loss"]),
        "node_loss_finite": finite_tensor(step2["node_loss"]),
        "rel_loss_finite": finite_tensor(step2["rel_loss"]),
        "subject_output_finite": finite_tensor(step2["sbj_out"]),
        "object_output_finite": finite_tensor(step2["obj_out"]),
        "relation_output_finite": finite_tensor(step2["rel_out"]),
        "hook_output_matches_forward": torch.equal(
            cap2["final_logits"], step2["rel_out"].detach()
        ),
        "margin_delta_finite": bool(
            torch.isfinite(cap2["margin_delta"]).all().item()
        ),
        "some_valid_margin_delta_nonzero": (
            valid_delta.numel() > 0
            and torch.count_nonzero(valid_delta).item() > 0
        ),
        "delta_bound_respected": max_abs_delta <= float(gate.delta_max),
        "invalid_delta_exact_zero": (
            True
            if not bool((~cap2["fibe_valid"]).any().item())
            else torch.count_nonzero(
                cap2["margin_delta"][~cap2["fibe_valid"]]
            ).item() == 0
        ),
        "positive_logits_bitwise_unchanged": torch.equal(
            cap2["base_logits"][:, 1:], cap2["final_logits"][:, 1:]
        ),
        "positive_order_unchanged": torch.equal(
            torch.argsort(cap2["base_logits"][:, 1:], dim=1),
            torch.argsort(cap2["final_logits"][:, 1:], dim=1),
        ),
        "step2_head_weight_gradient_nonzero": (
            step2_head_grad is not None and step2_head_grad > 0.0
        ),
        "step2_head_bias_gradient_nonzero": (
            step2_head_bias_grad is not None and step2_head_bias_grad > 0.0
        ),
        "step2_encoder_gradient_nonzero": (
            step2_encoder_grad is not None and step2_encoder_grad > 0.0
        ),
    }
    if not all(step2_checks.values()):
        raise RuntimeError(
            "Real-batch step-2 audit failed: "
            + json.dumps(step2_checks, ensure_ascii=False, sort_keys=True)
        )

    hook_handle.remove()

    all_checks = {
        **{f"env/{k}": v for k, v in env_checks.items()},
        **{f"config/{k}": v for k, v in config_checks.items()},
        **{f"model/{k}": v for k, v in model_checks.items()},
        **{f"cache/{k}": v for k, v in cache_checks.items()},
        **{f"step1/{k}": v for k, v in step1_checks.items()},
        **{f"step2/{k}": v for k, v in step2_checks.items()},
    }

    report = {
        "protocol": PROTOCOL,
        "status": "PASS" if all(all_checks.values()) else "FAIL",
        "scope": "real deterministic train DataLoader sub-batch; final test locked",
        "seed": args.seed,
        "epoch": args.epoch,
        "final_test_allowed": False,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(trainer.device),
            "required": required_env,
            "hard_negative_index": (
                str(hardneg_index_path) if hardneg_index_path is not None else None
            ),
            "hard_negative_index_sha256": (
                sha256(hardneg_index_path)
                if hardneg_index_path is not None and hardneg_index_path.is_file()
                else None
            ),
        },
        "inputs": {
            "config": str(config_path),
            "config_sha256": sha256(config_path),
            "annotation": str(annotation_path),
            "annotation_sha256": sha256(annotation_path),
            "data_root": str(data_root),
            "train_cache": str(trainer.fibe_train_cache.path),
            "train_cache_sha256": sha256(trainer.fibe_train_cache.path),
            "validation_cache": str(trainer.fibe_val_cache.path),
            "validation_cache_sha256": sha256(trainer.fibe_val_cache.path),
        },
        "trainer": {
            "critical_metric": trainer.critical_metric,
            "batch_size": trainer.imgs_per_batch,
            "rels_per_batch": trainer.rels_per_batch,
            "optimizer": type(trainer.optimizer).__name__,
            "optimizer_lrs": [
                float(group["lr"]) for group in trainer.optimizer.param_groups
            ],
            "optimizer_weight_decays": [
                float(group.get("weight_decay", 0.0))
                for group in trainer.optimizer.param_groups
            ],
            "gate_parameter_count": gate_parameter_count,
        },
        "selected_batch": {
            "raw_batch_index": selected_raw_index,
            "sub_batch_index": selected_sub_index,
            "image_ids": [
                int(v) for v in selected_batch["image_id"].detach().cpu().tolist()
            ],
            "num_images": int(len(selected_batch["image_id"])),
            "num_boxes": int(selected_batch["num_boxes"].sum().item()),
            "num_relations": num_relations,
            "valid_fibe_pairs": int(fibe_valid.sum().item()),
            "invalid_fibe_pairs": int((~fibe_valid).sum().item()),
            "none_target_rows": int((rel_target[:, 0] > 0.5).sum().item()),
            "positive_target_rows": int(
                (rel_target[:, 1:].sum(dim=1) > 0).sum().item()
            ),
        },
        "step1": {
            "loss": float(step1["loss"].detach().cpu().item()),
            "node_loss": float(step1["node_loss"].detach().cpu().item()),
            "rel_loss": float(step1["rel_loss"].detach().cpu().item()),
            "delta_min": float(cap1["margin_delta"].min().cpu().item()),
            "delta_max": float(cap1["margin_delta"].max().cpu().item()),
            "delta_mean": float(cap1["margin_delta"].mean().cpu().item()),
            "head_weight_grad_norm": step1_head_grad,
            "head_bias_grad_norm": step1_head_bias_grad,
            "encoder_first_linear_grad_norm": step1_encoder_grad,
        },
        "step2": {
            "loss": float(step2["loss"].detach().cpu().item()),
            "node_loss": float(step2["node_loss"].detach().cpu().item()),
            "rel_loss": float(step2["rel_loss"].detach().cpu().item()),
            "delta_min": float(cap2["margin_delta"].min().cpu().item()),
            "delta_max": float(cap2["margin_delta"].max().cpu().item()),
            "delta_mean": float(cap2["margin_delta"].mean().cpu().item()),
            "delta_max_abs_valid": max_abs_delta,
            "head_weight_grad_norm": step2_head_grad,
            "head_bias_grad_norm": step2_head_bias_grad,
            "encoder_first_linear_grad_norm": step2_encoder_grad,
        },
        "checks": all_checks,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))

    if report["status"] != "PASS":
        raise SystemExit(1)

    print()
    print("D1-v2a REAL DATALOADER ACTIVATION AUDIT V1: PASS")


if __name__ == "__main__":
    main()
