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

PROTOCOL = "FloodPSG D1 FIBE validation Water-HN counterfactual audit v1"
MODES = (
    "normal",
    "residual_bypass",
    "raw_feature_zero",
    "family_shuffle",
)
EXPECTED_IMAGES = 75
EXPECTED_PAIRS = 322


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize four locked D1 FIBE validation Water-HN counterfactual "
            "runs. This is a diagnostic audit only; final test remains locked."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family-output", type=Path, required=True)
    parser.add_argument("--pair-delta-output", type=Path, required=True)
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


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    formal_root = args.formal_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    family_output = args.family_output.expanduser().resolve()
    pair_delta_output = args.pair_delta_output.expanduser().resolve()

    for path in (root, formal_root):
        if not path.is_dir():
            raise NotADirectoryError(path)
    for path in (output, family_output, pair_delta_output):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite diagnostic output: {path}")

    mode_reports: dict[str, dict[str, Any]] = {}
    strict_summaries: dict[str, dict[str, Any]] = {}
    pair_frames: dict[str, pd.DataFrame] = {}
    paths_by_mode: dict[str, dict[str, Path]] = {}

    for mode in MODES:
        paths = {
            "pickle": root / f"D1_{mode}_validation_hn_gtpairs_v1.pkl",
            "report": root / f"D1_{mode}_validation_hn_gtpairs_v1.json",
            "pair_audit": root / f"D1_{mode}_pair_audit_v1.csv",
            "strict_summary": root / f"D1_{mode}_strict_eval_v1" / "summary.json",
        }
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        paths_by_mode[mode] = paths
        mode_reports[mode] = load_json(paths["report"])
        strict_summaries[mode] = load_json(paths["strict_summary"])
        pair_frames[mode] = pd.read_csv(
            paths["pair_audit"], encoding="utf-8-sig", low_memory=False
        )

    formal_d1_pickle = formal_root / "D1_DetV2_validation_hn_gtpairs_v1.pkl"
    formal_c_pickle = formal_root / "C_DetV2_validation_hn_gtpairs_v1.pkl"
    formal_gate = formal_root / "C_D1_VALIDATION_HN_POINT_ESTIMATE_GATE_V1.json"
    for path in (formal_d1_pickle, formal_c_pickle, formal_gate):
        if not path.is_file():
            raise FileNotFoundError(path)

    input_hashes = {
        mode: str(report["input_semantic_sha256"])
        for mode, report in mode_reports.items()
    }
    feature_hashes = {
        mode: str(report["mode_fibe_feature_semantic_sha256"])
        for mode, report in mode_reports.items()
    }

    pair_key_columns = ["global_image_key", "subject_index", "object_index"]
    canonical_pair_keys: list[tuple[str, int, int]] | None = None
    for mode, frame in pair_frames.items():
        keys = [
            (str(row.global_image_key), int(row.subject_index), int(row.object_index))
            for row in frame.itertuples(index=False)
        ]
        if len(keys) != EXPECTED_PAIRS or len(set(keys)) != EXPECTED_PAIRS:
            raise RuntimeError(f"Pair audit coverage failure for mode={mode}")
        if canonical_pair_keys is None:
            canonical_pair_keys = keys
        elif keys != canonical_pair_keys:
            raise RuntimeError(f"Pair audit order mismatch for mode={mode}")

    threshold_keys = {
        "0.5": "formal_max_positive_fpr_at_0.5",
        "0.7": "formal_max_positive_fpr_at_0.7",
        "0.9": "formal_max_positive_fpr_at_0.9",
    }
    mode_metrics: dict[str, dict[str, float]] = {}
    for mode, summary in strict_summaries.items():
        if summary.get("status") != "PASS":
            raise RuntimeError(f"Strict summary failed for mode={mode}")
        coverage = summary.get("coverage_gate", {})
        if coverage.get("status") != "PASS":
            raise RuntimeError(f"Coverage gate failed for mode={mode}")
        metrics = summary["pair_level_metrics_on_scored_pairs"]
        mode_metrics[mode] = {
            threshold: finite_float(metrics[key], f"{mode}/{threshold}")
            for threshold, key in threshold_keys.items()
        }
        mode_metrics[mode].update(
            {
                "mean_score_NONE": finite_float(
                    metrics["mean_score_NONE"], f"{mode}/mean_score_NONE"
                ),
                "mean_max_positive_score": finite_float(
                    metrics["mean_max_positive_score"],
                    f"{mode}/mean_max_positive_score",
                ),
                "mean_foreground_proxy": finite_float(
                    metrics["mean_foreground_proxy"],
                    f"{mode}/mean_foreground_proxy",
                ),
            }
        )

    normal_fpr = mode_metrics["normal"]
    comparisons: dict[str, Any] = {}
    for mode in MODES[1:]:
        comparisons[mode] = {
            threshold: {
                "normal_fpr": normal_fpr[threshold],
                "counterfactual_fpr": mode_metrics[mode][threshold],
                "improvement_normal_minus_counterfactual": (
                    normal_fpr[threshold] - mode_metrics[mode][threshold]
                ),
                "improvement_percentage_points": 100.0
                * (normal_fpr[threshold] - mode_metrics[mode][threshold]),
            }
            for threshold in ("0.5", "0.7", "0.9")
        }

    family_rows: list[dict[str, Any]] = []
    for mode, frame in pair_frames.items():
        for family, group in frame.groupby("pair_family_v4", dropna=False):
            family_rows.append(
                {
                    "mode": mode,
                    "pair_family_v4": str(family),
                    "pairs": int(len(group)),
                    "gate_mean": float(group["gate"].mean()),
                    "delta_norm_mean": float(group["delta_norm"].mean()),
                    "original_delta_existence_margin_mean": float(
                        group["original_delta_existence_margin"].mean()
                    ),
                    "original_delta_existence_margin_positive_rate": float(
                        (group["original_delta_existence_margin"] > 0).mean()
                    ),
                    "mode_final_existence_margin_mean": float(
                        group["mode_final_existence_margin"].mean()
                    ),
                    "mode_false_positive_rate_at_0_5_from_scores": float(
                        (
                            group.filter(like="mode_final_score__").iloc[:, 1:].max(axis=1)
                            > 0.5
                        ).mean()
                    ),
                }
            )
    family_frame = pd.DataFrame(family_rows).sort_values(
        ["pair_family_v4", "mode"]
    )

    c_scores = result_pair_map(formal_c_pickle)
    d1_scores = result_pair_map(formal_d1_pickle)
    if set(c_scores) != set(d1_scores):
        raise RuntimeError("Formal C/D1 result pair sets differ")
    if len(c_scores) != EXPECTED_PAIRS:
        raise RuntimeError(f"Formal result pair count is {len(c_scores)}")

    normal_frame = pair_frames["normal"].set_index(pair_key_columns, drop=False)
    pair_delta_rows: list[dict[str, Any]] = []
    for key in sorted(c_scores):
        c_row = c_scores[key]
        d1_row = d1_scores[key]
        c_logits = score_to_logit(c_row)
        d1_logits = score_to_logit(d1_row)
        c_none = float(c_logits[0])
        d1_none = float(d1_logits[0])
        c_max_positive = float(np.max(c_logits[1:]))
        d1_max_positive = float(np.max(d1_logits[1:]))
        c_margin = c_max_positive - c_none
        d1_margin = d1_max_positive - d1_none
        normal_meta = normal_frame.loc[key]
        pair_delta_rows.append(
            {
                "global_image_key": key[0],
                "subject_index": key[1],
                "object_index": key[2],
                "image_id": int(normal_meta["image_id"]),
                "pair_family_v4": normal_meta["pair_family_v4"],
                "scene_context_v4": normal_meta.get("scene_context_v4", ""),
                "c_none_score": float(c_row[0]),
                "d1_none_score": float(d1_row[0]),
                "d1_minus_c_none_score": float(d1_row[0] - c_row[0]),
                "c_max_positive_score": float(np.max(c_row[1:])),
                "d1_max_positive_score": float(np.max(d1_row[1:])),
                "d1_minus_c_max_positive_score": float(
                    np.max(d1_row[1:]) - np.max(c_row[1:])
                ),
                "c_existence_margin_reconstructed_logit": c_margin,
                "d1_existence_margin_reconstructed_logit": d1_margin,
                "d1_minus_c_existence_margin_reconstructed_logit": (
                    d1_margin - c_margin
                ),
                "d1_internal_original_delta_existence_margin": float(
                    normal_meta["original_delta_existence_margin"]
                ),
                "d1_gate": float(normal_meta["gate"]),
                "d1_delta_norm": float(normal_meta["delta_norm"]),
            }
        )
    pair_delta_frame = pd.DataFrame(pair_delta_rows)

    checks = {
        "all_mode_scorer_reports_pass": all(
            report.get("status") == "PASS" for report in mode_reports.values()
        ),
        "all_strict_summaries_pass": all(
            summary.get("status") == "PASS"
            for summary in strict_summaries.values()
        ),
        "all_input_semantic_sha256_equal": len(set(input_hashes.values())) == 1,
        "normal_and_bypass_fibe_feature_sha_equal": (
            feature_hashes["normal"] == feature_hashes["residual_bypass"]
        ),
        "raw_zero_fibe_feature_sha_differs": (
            feature_hashes["raw_feature_zero"] != feature_hashes["normal"]
        ),
        "shuffle_fibe_feature_sha_differs": (
            feature_hashes["family_shuffle"] != feature_hashes["normal"]
        ),
        "normal_pickle_exactly_reproduces_formal_d1": (
            sha256_file(paths_by_mode["normal"]["pickle"])
            == sha256_file(formal_d1_pickle)
        ),
        "all_pair_audits_complete": all(
            len(frame) == EXPECTED_PAIRS for frame in pair_frames.values()
        ),
        "final_test_locked": True,
    }
    status = "PASS" if all(checks.values()) else "FAIL"

    audit: dict[str, Any] = {
        "protocol": PROTOCOL,
        "status": status,
        "scope": "validation Water-HN diagnostic counterfactuals only",
        "images": EXPECTED_IMAGES,
        "pairs": EXPECTED_PAIRS,
        "final_test_allowed": False,
        "modes": mode_metrics,
        "comparisons_against_normal": comparisons,
        "normal_activation_summary": mode_reports["normal"][
            "activation_summary"
        ],
        "input_semantic_sha256_by_mode": input_hashes,
        "fibe_feature_semantic_sha256_by_mode": feature_hashes,
        "checks": checks,
        "interpretation_guardrails": {
            "residual_bypass": (
                "A lower FPR than normal supports a direct inference-time "
                "accelerator effect, but it need not reproduce C because D1 was "
                "trained from scratch and its backbone co-adapted with FIBE."
            ),
            "raw_feature_zero": (
                "This retains encoder/projection/gate biases and therefore is not "
                "an exact branch removal."
            ),
            "family_shuffle": (
                "This preserves the feature multiset within each pair family but "
                "breaks pair-specific feature correspondence."
            ),
            "scientific_gate": (
                "Counterfactual findings diagnose D1 failure and do not reverse "
                "the already failed formal D1 Water-HN point-estimate gate."
            ),
        },
        "sha256": {
            "formal_c_pickle": sha256_file(formal_c_pickle),
            "formal_d1_pickle": sha256_file(formal_d1_pickle),
            "formal_gate": sha256_file(formal_gate),
            **{
                f"{mode}_pickle": sha256_file(paths_by_mode[mode]["pickle"])
                for mode in MODES
            },
            **{
                f"{mode}_report": sha256_file(paths_by_mode[mode]["report"])
                for mode in MODES
            },
            **{
                f"{mode}_pair_audit": sha256_file(
                    paths_by_mode[mode]["pair_audit"]
                )
                for mode in MODES
            },
            **{
                f"{mode}_strict_summary": sha256_file(
                    paths_by_mode[mode]["strict_summary"]
                )
                for mode in MODES
            },
        },
    }

    if status != "PASS":
        print(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False))
        print()
        print("D1 FIBE COUNTERFACTUAL AUDIT V1: FAIL")
        raise SystemExit(1)

    atomic_csv_dump(family_frame, family_output)
    atomic_csv_dump(pair_delta_frame, pair_delta_output)
    audit["sha256"]["family_summary_csv"] = sha256_file(family_output)
    audit["sha256"]["c_d1_pair_delta_csv"] = sha256_file(pair_delta_output)
    atomic_json_dump(audit, output)

    print(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False))
    print()
    print("D1 FIBE COUNTERFACTUAL AUDIT V1: PASS")


if __name__ == "__main__":
    main()
