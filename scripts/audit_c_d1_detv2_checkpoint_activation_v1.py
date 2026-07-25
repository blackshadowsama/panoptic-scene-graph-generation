#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def torch_load(path: Path) -> Any:
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def extract_state_dict(
    payload: Any,
) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        direct = {
            str(key): value
            for key, value in payload.items()
            if isinstance(value, torch.Tensor)
        }

        if direct:
            return direct

        for key in (
            "state_dict",
            "model_state_dict",
            "model",
            "network",
            "net",
        ):
            candidate = payload.get(key)

            if isinstance(candidate, dict):
                try:
                    return extract_state_dict(
                        candidate
                    )
                except RuntimeError:
                    pass

    raise RuntimeError(
        "Could not locate a tensor state_dict."
    )


def parse_best_metric(
    log_path: Path,
) -> float:
    text = log_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    matches = re.findall(
        r"Best value for rel_mean_recall/50:"
        r"\s*([-+0-9.eE]+)",
        text,
    )

    if not matches:
        raise RuntimeError(
            f"Best validation metric not found: "
            f"{log_path}"
        )

    return float(matches[-1])


def tensor_is_finite(
    tensor: torch.Tensor,
) -> bool:
    if tensor.is_floating_point() or tensor.is_complex():
        return bool(
            torch.isfinite(tensor).all().item()
        )

    return True


def tensor_nonzero_count(
    tensor: torch.Tensor,
) -> int:
    return int(
        torch.count_nonzero(tensor).item()
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--c-output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--d1-output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--c-log",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--d1-log",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    c_best = (
        args.c_output.resolve()
        / "best_state.pth"
    )

    d1_best = (
        args.d1_output.resolve()
        / "best_state.pth"
    )

    c_config = (
        args.c_output.resolve()
        / "config.json"
    )

    d1_config = (
        args.d1_output.resolve()
        / "config.json"
    )

    for path in (
        c_best,
        d1_best,
        c_config,
        d1_config,
        args.c_log.resolve(),
        args.d1_log.resolve(),
    ):
        if not path.is_file():
            raise RuntimeError(
                f"Missing required file: {path}"
            )

    c_payload = torch_load(c_best)
    d1_payload = torch_load(d1_best)

    c_state = extract_state_dict(
        c_payload
    )

    d1_state = extract_state_dict(
        d1_payload
    )

    c_keys = set(c_state)
    d1_keys = set(d1_state)

    c_fibe_keys = sorted(
        key
        for key in c_keys
        if "fibe" in key.lower()
    )

    d1_fibe_keys = sorted(
        key
        for key in d1_keys
        if "fibe" in key.lower()
    )

    d1_fibe_stats = []

    fibe_all_finite = True
    fibe_total_nonzero = 0
    fibe_total_elements = 0

    for key in d1_fibe_keys:
        tensor = d1_state[key]

        finite = tensor_is_finite(
            tensor
        )

        nonzero = tensor_nonzero_count(
            tensor
        )

        elements = int(
            tensor.numel()
        )

        fibe_all_finite = (
            fibe_all_finite
            and finite
        )

        fibe_total_nonzero += nonzero
        fibe_total_elements += elements

        d1_fibe_stats.append({
            "key": key,
            "shape": list(
                tensor.shape
            ),
            "dtype": str(
                tensor.dtype
            ),
            "finite": finite,
            "nonzero": nonzero,
            "elements": elements,
        })

    common_keys = sorted(
        c_keys & d1_keys
    )

    identical_common = 0
    differing_common = 0
    shape_or_dtype_mismatch = 0

    first_differing_common_keys = []

    for key in common_keys:
        c_tensor = c_state[key]
        d1_tensor = d1_state[key]

        if (
            c_tensor.shape
            != d1_tensor.shape
            or c_tensor.dtype
            != d1_tensor.dtype
        ):
            shape_or_dtype_mismatch += 1
            differing_common += 1

            if len(
                first_differing_common_keys
            ) < 30:
                first_differing_common_keys.append(
                    key
                )

            continue

        if torch.equal(
            c_tensor,
            d1_tensor,
        ):
            identical_common += 1
        else:
            differing_common += 1

            if len(
                first_differing_common_keys
            ) < 30:
                first_differing_common_keys.append(
                    key
                )

    c_metric = parse_best_metric(
        args.c_log.resolve()
    )

    d1_metric = parse_best_metric(
        args.d1_log.resolve()
    )

    metric_delta = (
        d1_metric - c_metric
    )

    minimum_allowed = (
        c_metric - 0.02
    )

    checks = {
        "checkpoint_sha_differs": (
            sha256_file(c_best)
            != sha256_file(d1_best)
        ),
        "c_has_no_fibe_tensors": (
            len(c_fibe_keys) == 0
        ),
        "d1_has_fibe_tensors": (
            len(d1_fibe_keys) > 0
        ),
        "d1_fibe_all_finite": (
            fibe_all_finite
        ),
        "d1_fibe_has_nonzero_parameters": (
            fibe_total_nonzero > 0
        ),
        "common_weights_are_not_all_identical": (
            differing_common > 0
        ),
        "validation_mr50_drop_within_002": (
            d1_metric
            >= minimum_allowed
        ),
        "metrics_are_finite": (
            math.isfinite(c_metric)
            and math.isfinite(d1_metric)
        ),
    }

    status = (
        "PASS"
        if all(checks.values())
        else "FAIL"
    )

    report = {
        "protocol": (
            "FloodPSG C-Det/D1-Det "
            "checkpoint activation audit V1"
        ),
        "c_output": str(
            args.c_output.resolve()
        ),
        "d1_output": str(
            args.d1_output.resolve()
        ),
        "checkpoint_sha256": {
            "c_best_state": sha256_file(
                c_best
            ),
            "d1_best_state": sha256_file(
                d1_best
            ),
        },
        "validation": {
            "c_mr50": c_metric,
            "d1_mr50": d1_metric,
            "delta_d1_minus_c": (
                metric_delta
            ),
            "minimum_allowed_d1": (
                minimum_allowed
            ),
        },
        "tensor_inventory": {
            "c_tensor_count": len(
                c_state
            ),
            "d1_tensor_count": len(
                d1_state
            ),
            "common_tensor_count": len(
                common_keys
            ),
            "identical_common_tensor_count": (
                identical_common
            ),
            "differing_common_tensor_count": (
                differing_common
            ),
            "shape_or_dtype_mismatch_count": (
                shape_or_dtype_mismatch
            ),
            "c_only_tensor_count": len(
                c_keys - d1_keys
            ),
            "d1_only_tensor_count": len(
                d1_keys - c_keys
            ),
            "first_differing_common_keys": (
                first_differing_common_keys
            ),
        },
        "fibe": {
            "c_fibe_tensor_count": len(
                c_fibe_keys
            ),
            "d1_fibe_tensor_count": len(
                d1_fibe_keys
            ),
            "d1_fibe_total_elements": (
                fibe_total_elements
            ),
            "d1_fibe_total_nonzero": (
                fibe_total_nonzero
            ),
            "d1_fibe_all_finite": (
                fibe_all_finite
            ),
            "d1_fibe_tensors": (
                d1_fibe_stats
            ),
        },
        "checks": checks,
        "status": status,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()

    if status != "PASS":
        print(
            "C/D1 DETV2 CHECKPOINT "
            "ACTIVATION AUDIT: FAIL"
        )
        raise SystemExit(1)

    print(
        "C/D1 DETV2 CHECKPOINT "
        "ACTIVATION AUDIT: PASS"
    )


if __name__ == "__main__":
    main()
