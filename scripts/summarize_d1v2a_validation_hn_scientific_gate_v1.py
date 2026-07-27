#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROTOCOL = "FloodPSG D1-v2a validation Water-HN scientific gate V1"
MODES = (
    "normal",
    "gate_bypass",
    "raw_feature_zero",
    "family_shuffle",
)
EXPECTED_IMAGES = 75
EXPECTED_PAIRS = 322
THRESHOLDS = (0.5, 0.7, 0.9)
C_EXPECTED_FPR = {
    "0.5": 0.984472049689441,
    "0.7": 0.9285714285714286,
    "0.9": 0.5124223602484472,
}
VALIDATION_FLOOR = 0.8455303120613098
POINT_GATE_IMPROVEMENT_05 = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize four D1-v2a validation Water-HN modes, compare the "
            "formal candidate with the locked C control, run a paired "
            "global_image_key cluster bootstrap, and evaluate the predeclared "
            "point-estimate scientific gate. Final test remains locked."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--formal-c-pickle", type=Path, required=True)
    parser.add_argument("--completion-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family-output", type=Path, required=True)
    parser.add_argument("--pair-output", type=Path, required=True)
    parser.add_argument("--bootstrap-output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def atomic_json_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")
    temporary.replace(path)


def atomic_csv_dump(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8-sig",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    temporary.replace(path)


def finite_float(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"Nonfinite {label}: {value!r}")
    return result


def score_to_logit(score: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=np.float64)
    eps = np.finfo(np.float32).eps
    clipped = np.clip(score, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def result_pair_map(path: Path) -> dict[tuple[str, int, int], np.ndarray]:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, list):
        raise TypeError(f"Expected list result pickle: {path}")
    mapping: dict[tuple[str, int, int], np.ndarray] = {}
    for item in data:
        global_key = str(item["global_image_key"])
        pairs = np.asarray(item["pairs"], dtype=np.int64)
        scores = np.asarray(item["rel_scores"], dtype=np.float64)
        if scores.shape != (len(pairs), 9):
            raise RuntimeError(f"Malformed score matrix in {path}: {scores.shape}")
        for pair, row in zip(pairs.tolist(), scores):
            key = (global_key, int(pair[0]), int(pair[1]))
            if key in mapping:
                raise RuntimeError(f"Duplicate result pair {key} in {path}")
            mapping[key] = row.copy()
    return mapping


def strict_metrics(summary: dict[str, Any], label: str) -> dict[str, float]:
    if summary.get("status") != "PASS":
        raise RuntimeError(f"Strict evaluator status is not PASS: {label}")
    coverage = summary.get("coverage_gate", {})
    if coverage.get("status") != "PASS":
        raise RuntimeError(f"Strict coverage gate failed: {label}")
    raw = summary["pair_level_metrics_on_scored_pairs"]
    return {
        "fpr_0.5": finite_float(
            raw["formal_max_positive_fpr_at_0.5"],
            f"{label}/fpr_0.5",
        ),
        "fpr_0.7": finite_float(
            raw["formal_max_positive_fpr_at_0.7"],
            f"{label}/fpr_0.7",
        ),
        "fpr_0.9": finite_float(
            raw["formal_max_positive_fpr_at_0.9"],
            f"{label}/fpr_0.9",
        ),
        "mean_score_NONE": finite_float(
            raw["mean_score_NONE"],
            f"{label}/mean_score_NONE",
        ),
        "mean_max_positive_score": finite_float(
            raw["mean_max_positive_score"],
            f"{label}/mean_max_positive_score",
        ),
        "mean_foreground_proxy": finite_float(
            raw["mean_foreground_proxy"],
            f"{label}/mean_foreground_proxy",
        ),
    }


def metrics_from_score_matrix(scores: np.ndarray) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64)
    max_positive = scores[:, 1:].max(axis=1)
    none = scores[:, 0]
    return {
        "fpr_0.5": float(np.mean(max_positive > 0.5)),
        "fpr_0.7": float(np.mean(max_positive > 0.7)),
        "fpr_0.9": float(np.mean(max_positive > 0.9)),
        "mean_score_NONE": float(np.mean(none)),
        "mean_max_positive_score": float(np.mean(max_positive)),
        "mean_foreground_proxy": float(np.mean(1.0 - none)),
    }


def ci(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Bootstrap array must be nonempty and one-dimensional")
    if not np.isfinite(values).all():
        raise FloatingPointError("Nonfinite bootstrap values")
    return {
        "valid_iterations": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "ci_2_5": float(np.quantile(values, 0.025)),
        "ci_97_5": float(np.quantile(values, 0.975)),
    }


def paired_image_bootstrap(
    *,
    frame: pd.DataFrame,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    if iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")

    images = sorted(frame["global_image_key"].astype(str).unique().tolist())
    if len(images) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_IMAGES} bootstrap clusters, got {len(images)}"
        )
    rows_by_image = {
        image: frame.index[
            frame["global_image_key"].astype(str) == image
        ].to_numpy(dtype=np.int64)
        for image in images
    }

    rng = np.random.default_rng(seed)
    accum: dict[str, list[float]] = {}
    for threshold in THRESHOLDS:
        token = f"{threshold:.1f}"
        accum[f"c_fpr_{token}"] = []
        accum[f"d1v2a_fpr_{token}"] = []
        accum[f"c_minus_d1v2a_{token}"] = []

    for _ in range(iterations):
        sampled = rng.choice(images, size=len(images), replace=True)
        indices = np.concatenate([rows_by_image[image] for image in sampled])
        subset = frame.loc[indices]

        for threshold in THRESHOLDS:
            token = f"{threshold:.1f}"
            c_fpr = float(
                np.mean(subset["c_max_positive_score"].to_numpy() > threshold)
            )
            d_fpr = float(
                np.mean(
                    subset["d1v2a_max_positive_score"].to_numpy() > threshold
                )
            )
            accum[f"c_fpr_{token}"].append(c_fpr)
            accum[f"d1v2a_fpr_{token}"].append(d_fpr)
            accum[f"c_minus_d1v2a_{token}"].append(c_fpr - d_fpr)

    return {
        "protocol":
            "FloodPSG D1-v2a paired image-cluster bootstrap V1",
        "status": "PASS",
        "cluster": "global_image_key",
        "unique_validation_images": len(images),
        "requested_iterations": iterations,
        "seed": seed,
        "final_test_allowed": False,
        "metrics": {
            key: ci(np.asarray(values, dtype=np.float64))
            for key, values in accum.items()
        },
    }


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    c_pickle = args.formal_c_pickle.expanduser().resolve()
    completion_audit_path = args.completion_audit.expanduser().resolve()
    output = args.output.expanduser().resolve()
    family_output = args.family_output.expanduser().resolve()
    pair_output = args.pair_output.expanduser().resolve()
    bootstrap_output = args.bootstrap_output.expanduser().resolve()

    if not root.is_dir():
        raise NotADirectoryError(root)
    for path in (c_pickle, completion_audit_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (output, family_output, pair_output, bootstrap_output):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite output: {path}")

    completion = load_json(completion_audit_path)
    if completion.get("status") != "PASS":
        raise RuntimeError("Formal completion audit is not PASS")
    if completion.get("final_test_allowed") is not False:
        raise RuntimeError("Completion audit did not lock final test")
    validation_mr50 = finite_float(
        completion["selected_best_value"],
        "completion/selected_best_value",
    )

    mode_reports: dict[str, dict[str, Any]] = {}
    strict_summaries: dict[str, dict[str, Any]] = {}
    pair_frames: dict[str, pd.DataFrame] = {}
    paths_by_mode: dict[str, dict[str, Path]] = {}

    for mode in MODES:
        paths = {
            "pickle":
                root / f"D1V2A_{mode}_validation_hn_gtpairs_v1.pkl",
            "report":
                root / f"D1V2A_{mode}_validation_hn_gtpairs_v1.json",
            "pair_audit":
                root / f"D1V2A_{mode}_pair_audit_v1.csv",
            "strict_summary":
                root
                / f"D1V2A_{mode}_strict_eval_v1"
                / "summary.json",
        }
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        paths_by_mode[mode] = paths
        mode_reports[mode] = load_json(paths["report"])
        strict_summaries[mode] = load_json(paths["strict_summary"])
        pair_frames[mode] = pd.read_csv(
            paths["pair_audit"],
            encoding="utf-8-sig",
            low_memory=False,
        )

    pair_key_columns = [
        "global_image_key",
        "subject_index",
        "object_index",
    ]
    canonical_keys: list[tuple[str, int, int]] | None = None
    for mode, frame in pair_frames.items():
        keys = [
            (
                str(row.global_image_key),
                int(row.subject_index),
                int(row.object_index),
            )
            for row in frame.itertuples(index=False)
        ]
        if len(keys) != EXPECTED_PAIRS or len(set(keys)) != EXPECTED_PAIRS:
            raise RuntimeError(f"Pair coverage failure: mode={mode}")
        if canonical_keys is None:
            canonical_keys = keys
        elif keys != canonical_keys:
            raise RuntimeError(f"Pair order mismatch: mode={mode}")

    mode_metrics = {
        mode: strict_metrics(strict_summaries[mode], mode)
        for mode in MODES
    }

    c_map = result_pair_map(c_pickle)
    if len(c_map) != EXPECTED_PAIRS:
        raise RuntimeError(f"Formal C contains {len(c_map)} pairs")

    normal_frame = pair_frames["normal"].copy()
    normal_frame = normal_frame.set_index(pair_key_columns, drop=False)
    if set(normal_frame.index.tolist()) != set(c_map):
        raise RuntimeError("Formal C pair set differs from D1-v2a frozen pair set")

    pair_rows: list[dict[str, Any]] = []
    c_scores_rows: list[np.ndarray] = []
    d_scores_rows: list[np.ndarray] = []
    for key in canonical_keys or []:
        normal = normal_frame.loc[key]
        c_score = c_map[key]
        d_score = np.asarray(
            [
                float(normal[column])
                for column in normal_frame.columns
                if column.startswith("mode_final_score__")
            ],
            dtype=np.float64,
        )
        if d_score.shape != (9,):
            raise RuntimeError(f"Expected nine D1-v2a scores for pair {key}")
        c_scores_rows.append(c_score)
        d_scores_rows.append(d_score)

        c_logit = score_to_logit(c_score)
        d_logit = score_to_logit(d_score)
        c_max = float(np.max(c_score[1:]))
        d_max = float(np.max(d_score[1:]))
        c_margin = float(np.max(c_logit[1:]) - c_logit[0])
        d_margin = float(np.max(d_logit[1:]) - d_logit[0])
        pair_rows.append(
            {
                "global_image_key": key[0],
                "subject_index": key[1],
                "object_index": key[2],
                "image_id": int(normal["image_id"]),
                "pair_family_v4": str(normal["pair_family_v4"]),
                "scene_context_v4": normal.get("scene_context_v4", ""),
                "c_none_score": float(c_score[0]),
                "d1v2a_none_score": float(d_score[0]),
                "d1v2a_minus_c_none_score": float(d_score[0] - c_score[0]),
                "c_max_positive_score": c_max,
                "d1v2a_max_positive_score": d_max,
                "d1v2a_minus_c_max_positive_score": float(d_max - c_max),
                "c_existence_margin_reconstructed_logit": c_margin,
                "d1v2a_existence_margin_reconstructed_logit": d_margin,
                "d1v2a_minus_c_existence_margin_reconstructed_logit":
                    float(d_margin - c_margin),
                "d1v2a_internal_base_existence_margin": float(
                    normal["base_existence_margin"]
                ),
                "d1v2a_internal_final_existence_margin": float(
                    normal["mode_final_existence_margin"]
                ),
                "d1v2a_internal_margin_delta": float(
                    normal["margin_delta"]
                ),
                "d1v2a_margin_delta_abs": float(
                    normal["margin_delta_abs"]
                ),
                "d1v2a_margin_delta_saturated_abs_ge_3_9": bool(
                    normal["margin_delta_saturated_abs_ge_3_9"]
                ),
            }
        )

    pair_frame = pd.DataFrame(pair_rows)
    c_scores = np.stack(c_scores_rows, axis=0)
    d_scores = np.stack(d_scores_rows, axis=0)
    c_metrics = metrics_from_score_matrix(c_scores)
    normal_direct_metrics = metrics_from_score_matrix(d_scores)

    for key, value in C_EXPECTED_FPR.items():
        observed = c_metrics[f"fpr_{key}"]
        if not math.isclose(observed, value, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(
                f"Formal C FPR mismatch at {key}: {observed} != {value}"
            )

    for key in ("fpr_0.5", "fpr_0.7", "fpr_0.9"):
        if not math.isclose(
            normal_direct_metrics[key],
            mode_metrics["normal"][key],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"Normal strict/direct metric mismatch: {key}")

    bootstrap = paired_image_bootstrap(
        frame=pair_frame,
        iterations=args.bootstrap_iterations,
        seed=args.seed,
    )

    comparisons: dict[str, Any] = {}
    for mode in MODES[1:]:
        comparisons[mode] = {
            threshold: {
                "normal_fpr": mode_metrics["normal"][f"fpr_{threshold}"],
                "counterfactual_fpr": mode_metrics[mode][f"fpr_{threshold}"],
                "normal_minus_counterfactual": (
                    mode_metrics["normal"][f"fpr_{threshold}"]
                    - mode_metrics[mode][f"fpr_{threshold}"]
                ),
                "normal_minus_counterfactual_percentage_points": 100.0 * (
                    mode_metrics["normal"][f"fpr_{threshold}"]
                    - mode_metrics[mode][f"fpr_{threshold}"]
                ),
            }
            for threshold in ("0.5", "0.7", "0.9")
        }
        comparisons[mode]["continuous"] = {
            metric: {
                "normal": mode_metrics["normal"][metric],
                "counterfactual": mode_metrics[mode][metric],
                "counterfactual_minus_normal": (
                    mode_metrics[mode][metric]
                    - mode_metrics["normal"][metric]
                ),
            }
            for metric in (
                "mean_score_NONE",
                "mean_max_positive_score",
                "mean_foreground_proxy",
            )
        }

    family_rows: list[dict[str, Any]] = []
    for family, group in pair_frame.groupby("pair_family_v4", dropna=False):
        indices = group.index.to_numpy(dtype=np.int64)
        c_group_metrics = metrics_from_score_matrix(c_scores[indices])
        family_rows.append(
            {
                "model_mode": "C",
                "pair_family_v4": str(family),
                "images": int(group["global_image_key"].nunique()),
                "pairs": int(len(group)),
                **c_group_metrics,
                "mean_existence_margin": float(
                    group["c_existence_margin_reconstructed_logit"].mean()
                ),
                "mean_margin_delta": np.nan,
                "median_margin_delta": np.nan,
                "p95_abs_margin_delta": np.nan,
                "saturation_rate_abs_ge_3_9": np.nan,
            }
        )

    for mode, frame in pair_frames.items():
        score_columns = [
            column
            for column in frame.columns
            if column.startswith("mode_final_score__")
        ]
        if len(score_columns) != 9:
            raise RuntimeError(f"Mode {mode} has {len(score_columns)} score columns")
        for family, group in frame.groupby("pair_family_v4", dropna=False):
            scores = group[score_columns].to_numpy(dtype=np.float64)
            metrics = metrics_from_score_matrix(scores)
            family_rows.append(
                {
                    "model_mode": mode,
                    "pair_family_v4": str(family),
                    "images": int(group["global_image_key"].nunique()),
                    "pairs": int(len(group)),
                    **metrics,
                    "mean_existence_margin": float(
                        group["mode_final_existence_margin"].mean()
                    ),
                    "mean_margin_delta": float(group["margin_delta"].mean()),
                    "median_margin_delta": float(group["margin_delta"].median()),
                    "p95_abs_margin_delta": float(
                        group["margin_delta_abs"].quantile(0.95)
                    ),
                    "saturation_rate_abs_ge_3_9": float(
                        group["margin_delta_saturated_abs_ge_3_9"].mean()
                    ),
                }
            )

    family_frame = pd.DataFrame(family_rows).sort_values(
        ["pair_family_v4", "model_mode"]
    )

    normal_fpr = mode_metrics["normal"]
    gate_checks = {
        "validation_mR50_floor_pass":
            validation_mr50 >= VALIDATION_FLOOR,
        "FPR_0_5_improves_at_least_5pp": (
            normal_fpr["fpr_0.5"]
            <= c_metrics["fpr_0.5"] - POINT_GATE_IMPROVEMENT_05
        ),
        "FPR_0_7_directional_improvement": (
            normal_fpr["fpr_0.7"] < c_metrics["fpr_0.7"]
        ),
        "FPR_0_9_directional_improvement": (
            normal_fpr["fpr_0.9"] < c_metrics["fpr_0.9"]
        ),
    }
    point_estimate_gate = "PASS" if all(gate_checks.values()) else "FAIL"

    input_hashes = {
        mode: str(report["input_semantic_sha256"])
        for mode, report in mode_reports.items()
    }
    feature_hashes = {
        mode: str(report["mode_fibe_feature_semantic_sha256"])
        for mode, report in mode_reports.items()
    }

    technical_checks = {
        "all_mode_reports_pass": all(
            report.get("status") == "PASS"
            for report in mode_reports.values()
        ),
        "all_strict_summaries_pass": all(
            summary.get("status") == "PASS"
            for summary in strict_summaries.values()
        ),
        "all_pair_audits_complete": all(
            len(frame) == EXPECTED_PAIRS
            for frame in pair_frames.values()
        ),
        "all_input_semantic_hashes_equal": len(set(input_hashes.values())) == 1,
        "normal_and_bypass_feature_hash_equal": (
            feature_hashes["normal"] == feature_hashes["gate_bypass"]
        ),
        "raw_zero_feature_hash_differs": (
            feature_hashes["raw_feature_zero"] != feature_hashes["normal"]
        ),
        "shuffle_feature_hash_differs": (
            feature_hashes["family_shuffle"] != feature_hashes["normal"]
        ),
        "all_invalid_controls_exact": all(
            report["checks"]["invalid_control_delta_exact_zero"]
            and report["checks"]["invalid_control_output_exact_identity"]
            for report in mode_reports.values()
        ),
        "all_positive_logits_unchanged": all(
            report["checks"]["positive_logits_bitwise_unchanged"]
            for report in mode_reports.values()
        ),
        "formal_C_pair_count_322": len(c_map) == EXPECTED_PAIRS,
        "bootstrap_pass": bootstrap["status"] == "PASS",
        "bootstrap_iterations_2000": (
            args.bootstrap_iterations == 2000
        ),
        "final_test_locked": True,
    }
    technical_status = (
        "PASS" if all(technical_checks.values()) else "FAIL"
    )

    normal_activation = mode_reports["normal"]["activation_summary"]
    delta_summary = {
        "margin_delta": normal_activation["margin_delta"],
        "margin_delta_abs": normal_activation["margin_delta_abs"],
        "positive_pairs": normal_activation["margin_delta_positive_pairs"],
        "positive_rate": normal_activation["margin_delta_positive_rate"],
        "negative_pairs": normal_activation["margin_delta_negative_pairs"],
        "negative_rate": normal_activation["margin_delta_negative_rate"],
        "saturated_abs_ge_3_9_pairs":
            normal_activation[
                "margin_delta_saturated_abs_ge_3_9_pairs"
            ],
        "saturated_abs_ge_3_9_rate":
            normal_activation[
                "margin_delta_saturated_abs_ge_3_9_rate"
            ],
    }

    report: dict[str, Any] = {
        "protocol": PROTOCOL,
        "status": technical_status,
        "technical_status": technical_status,
        "scientific_point_estimate_gate": point_estimate_gate,
        "scope":
            "formal D1-v2a validation Water-HN scientific gate; final test locked",
        "images": EXPECTED_IMAGES,
        "pairs": EXPECTED_PAIRS,
        "seed": int(args.seed),
        "bootstrap_iterations": int(args.bootstrap_iterations),
        "final_test_allowed": False,
        "validation_main_task": {
            "metric": "rel_mean_recall/50",
            "D1v2a": validation_mr50,
            "floor": VALIDATION_FLOOR,
            "pass": gate_checks["validation_mR50_floor_pass"],
        },
        "formal_C": c_metrics,
        "D1v2a_modes": mode_metrics,
        "D1v2a_normal_minus_C": {
            "fpr_0.5": normal_fpr["fpr_0.5"] - c_metrics["fpr_0.5"],
            "fpr_0.7": normal_fpr["fpr_0.7"] - c_metrics["fpr_0.7"],
            "fpr_0.9": normal_fpr["fpr_0.9"] - c_metrics["fpr_0.9"],
            "mean_score_NONE": (
                normal_fpr["mean_score_NONE"] - c_metrics["mean_score_NONE"]
            ),
            "mean_max_positive_score": (
                normal_fpr["mean_max_positive_score"]
                - c_metrics["mean_max_positive_score"]
            ),
        },
        "C_minus_D1v2a_improvement": {
            "fpr_0.5": c_metrics["fpr_0.5"] - normal_fpr["fpr_0.5"],
            "fpr_0.7": c_metrics["fpr_0.7"] - normal_fpr["fpr_0.7"],
            "fpr_0.9": c_metrics["fpr_0.9"] - normal_fpr["fpr_0.9"],
        },
        "predeclared_gate_checks": gate_checks,
        "counterfactual_comparisons_against_normal": comparisons,
        "normal_gate_delta_summary": delta_summary,
        "bootstrap": bootstrap,
        "input_semantic_sha256_by_mode": input_hashes,
        "fibe_feature_semantic_sha256_by_mode": feature_hashes,
        "technical_checks": technical_checks,
        "interpretation_guardrails": {
            "gate_bypass": (
                "Exact inference-time removal of the trained scalar correction; "
                "it retains the co-adapted backbone/classifier and need not equal C."
            ),
            "raw_feature_zero": (
                "Raw 21D inputs are zeroed while trained encoder/head biases and "
                "LayerNorm remain active; this is not exact gate removal."
            ),
            "family_shuffle": (
                "The 21D feature multiset is preserved within each pair family "
                "but pair-specific correspondence is broken."
            ),
            "bootstrap": (
                "The paired image-cluster confidence intervals quantify uncertainty. "
                "The predeclared formal pass/fail rule remains the locked point-estimate "
                "gate unless a separate CI-based rule was declared before scoring."
            ),
        },
        "sha256": {
            "formal_C_pickle": sha256_file(c_pickle),
            "completion_audit": sha256_file(completion_audit_path),
            **{
                f"{mode}_pickle":
                    sha256_file(paths_by_mode[mode]["pickle"])
                for mode in MODES
            },
            **{
                f"{mode}_report":
                    sha256_file(paths_by_mode[mode]["report"])
                for mode in MODES
            },
            **{
                f"{mode}_pair_audit":
                    sha256_file(paths_by_mode[mode]["pair_audit"])
                for mode in MODES
            },
            **{
                f"{mode}_strict_summary":
                    sha256_file(paths_by_mode[mode]["strict_summary"])
                for mode in MODES
            },
        },
    }

    if technical_status != "PASS":
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        print()
        print("D1-v2a VALIDATION HN SCIENTIFIC GATE V1: TECHNICAL FAIL")
        raise SystemExit(1)

    atomic_csv_dump(family_frame, family_output)
    atomic_csv_dump(pair_frame, pair_output)
    atomic_json_dump(bootstrap, bootstrap_output)
    report["sha256"]["family_csv"] = sha256_file(family_output)
    report["sha256"]["pair_csv"] = sha256_file(pair_output)
    report["sha256"]["bootstrap_json"] = sha256_file(bootstrap_output)
    atomic_json_dump(report, output)

    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    print()
    print("D1-v2a VALIDATION HN SCIENTIFIC GATE V1: TECHNICAL PASS")
    print(f"SCIENTIFIC POINT-ESTIMATE GATE: {point_estimate_gate}")


if __name__ == "__main__":
    main()
