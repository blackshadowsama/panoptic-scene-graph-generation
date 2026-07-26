#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FloodPSG FIBE-21D matched linear-probe diagnostic.

Fits three leakage-safe probes on the frozen group-strict split:
  G  = 21D geometry only
  F  = pair-family only
  GF = 21D geometry + pair-family

Training rows come only from the authoritative train split. Validation negatives
are exactly the frozen 322 Water-HN pairs. The script never touches final test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


WATER_FAMILIES: tuple[str, ...] = (
    "person-water",
    "vehicle-water",
    "building-water",
    "road-water",
    "underpass_bridge-water",
    "sidewalk-water",
    "manhole-water",
)
FEATURE_COLUMNS: tuple[str, ...] = tuple(f"f{i:02d}" for i in range(21))
PAIR_KEY_COLUMNS: tuple[str, ...] = (
    "global_image_key",
    "subject_index_v4",
    "object_index_v4",
)


@dataclass(frozen=True)
class ExtractAudit:
    requested_rows: int
    extracted_rows: int
    duplicate_pair_rows_removed: int
    missing_annotation_images: int
    cache_image_missing: int
    malformed_indices: int
    index_out_of_range: int
    invalid_cache_pairs: int
    nonfinite_feature_rows: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit leakage-safe FIBE-21D geometry/family probes using train-only "
            "fitting and the frozen validation Water-HN subset. --help performs "
            "no project loading and creates no outputs."
        )
    )
    parser.add_argument("--train-cache", required=True, type=Path)
    parser.add_argument("--validation-cache", required=True, type=Path)
    parser.add_argument("--annotation", required=True, type=Path)
    parser.add_argument("--hardneg-source", required=True, type=Path)
    parser.add_argument("--frozen-validation-hn", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--expected-train-images", type=int, default=1500)
    parser.add_argument("--expected-validation-images", type=int, default=173)
    parser.add_argument("--expected-validation-hn-pairs", type=int, default=322)
    parser.add_argument(
        "--allow-incomplete-positive-features",
        action="store_true",
        help=(
            "Allow positive rows with invalid/missing cache features to be excluded. "
            "Frozen validation HN rows are always required to be complete."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_frame_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    ordered = frame.loc[:, list(columns)].copy()
    ordered = ordered.sort_values(list(PAIR_KEY_COLUMNS), kind="mergesort").reset_index(drop=True)
    digest = hashlib.sha256()
    for row in ordered.itertuples(index=False, name=None):
        for value in row:
            if isinstance(value, (float, np.floating)):
                digest.update(np.float64(value).tobytes())
            elif isinstance(value, (int, np.integer, bool, np.bool_)):
                digest.update(str(int(value)).encode("utf-8"))
            else:
                digest.update(str(value).encode("utf-8"))
            digest.update(b"\x1f")
        digest.update(b"\n")
    return digest.hexdigest()


def torch_load_cpu(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_cache(path: Path) -> tuple[Mapping[Any, Any], Mapping[str, Any]]:
    payload = torch_load_cpu(path)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Cache must be a mapping: {path}")
    mapping = payload.get("features_by_image", payload)
    if not isinstance(mapping, Mapping):
        raise TypeError(f"features_by_image must be a mapping: {path}")
    metadata = {
        str(key): value
        for key, value in payload.items()
        if key != "features_by_image"
    }
    return mapping, metadata


def get_cache_entry(mapping: Mapping[Any, Any], image_id: int) -> Mapping[str, Any] | None:
    for key in (str(int(image_id)), int(image_id)):
        if key in mapping:
            entry = mapping[key]
            if not isinstance(entry, Mapping):
                raise TypeError(f"Cache entry for image {image_id} is not a mapping")
            return entry
    return None


def parse_bool_series(series: pd.Series) -> pd.Series:
    true_values = {"1", "true", "t", "yes", "y"}
    false_values = {"0", "false", "f", "no", "n", "", "nan", "none", "<na>"}

    def convert(value: Any) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)):
            return bool(int(value))
        if isinstance(value, (float, np.floating)):
            return False if math.isnan(float(value)) else bool(int(value))
        text = str(value).strip().lower()
        if text in true_values:
            return True
        if text in false_values:
            return False
        raise ValueError(f"Cannot parse boolean value: {value!r}")

    return series.map(convert).astype(bool)


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{label} missing columns: {missing}")


def normalize_source(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    required = [
        *PAIR_KEY_COLUMNS,
        "audit_split_v2",
        "audit_pair_status",
        "pair_family_v4",
        "scene_context_v4",
        "canonical_pair_legal_v4",
        "canonical_pair_positive_v4",
        "canonical_label_consistent_v4",
        "is_high_risk_v4",
        "subject_resolution_v4",
        "object_resolution_v4",
    ]
    require_columns(frame, required, label)
    result = frame.copy()
    for column in (
        "canonical_pair_legal_v4",
        "canonical_pair_positive_v4",
        "canonical_label_consistent_v4",
        "is_high_risk_v4",
    ):
        result[column] = parse_bool_series(result[column])
    for column in (
        "audit_split_v2",
        "audit_pair_status",
        "pair_family_v4",
        "scene_context_v4",
        "subject_resolution_v4",
        "object_resolution_v4",
        "global_image_key",
    ):
        result[column] = result[column].fillna("").astype(str).str.strip()
    return result


def eligible_base(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["pair_family_v4"].isin(WATER_FAMILIES)
        & frame["canonical_pair_legal_v4"]
        & frame["canonical_label_consistent_v4"]
        & frame["is_high_risk_v4"]
        & frame["subject_resolution_v4"].str.lower().eq("resolved")
        & frame["object_resolution_v4"].str.lower().eq("resolved")
    )


def dedupe_pairs(frame: pd.DataFrame, label: str) -> tuple[pd.DataFrame, int]:
    work = frame.copy()
    for column in ("subject_index_v4", "object_index_v4"):
        numeric = pd.to_numeric(work[column], errors="coerce")
        if numeric.isna().any():
            examples = work.loc[numeric.isna(), list(PAIR_KEY_COLUMNS)].head(10).to_dict("records")
            raise ValueError(f"{label}: malformed pair indices: {examples}")
        if not np.allclose(numeric.to_numpy(), np.round(numeric.to_numpy())):
            raise ValueError(f"{label}: non-integer pair indices detected")
        work[column] = np.round(numeric).astype(np.int64)

    duplicate_mask = work.duplicated(list(PAIR_KEY_COLUMNS), keep=False)
    if duplicate_mask.any():
        duplicated = work.loc[duplicate_mask]
        conflict_columns = [
            column
            for column in ("label", "pair_family_v4", "scene_context_v4")
            if column in duplicated.columns
        ]
        grouped = duplicated.groupby(list(PAIR_KEY_COLUMNS), dropna=False)
        conflicts: list[dict[str, Any]] = []
        for key, group in grouped:
            if any(group[column].nunique(dropna=False) > 1 for column in conflict_columns):
                conflicts.append({"pair": list(key), "rows": group[conflict_columns].to_dict("records")})
                if len(conflicts) >= 10:
                    break
        if conflicts:
            raise ValueError(f"{label}: conflicting duplicate pair rows: {conflicts}")
    before = len(work)
    work = work.drop_duplicates(list(PAIR_KEY_COLUMNS), keep="first").reset_index(drop=True)
    return work, before - len(work)


def build_annotation_maps(annotation_path: Path) -> tuple[dict[str, int], set[int], set[int], dict[str, Any]]:
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    data = annotation.get("data")
    test_ids_raw = annotation.get("test_image_ids")
    if not isinstance(data, list) or not isinstance(test_ids_raw, list):
        raise TypeError("Annotation must contain list fields data and test_image_ids")

    key_to_id: dict[str, int] = {}
    all_ids: set[int] = set()
    for item in data:
        image_id = int(item["image_id"])
        image_key = str(item["global_image_key"]).strip()
        if image_key in key_to_id and key_to_id[image_key] != image_id:
            raise ValueError(f"Duplicate global_image_key with conflicting image_id: {image_key}")
        if image_id in all_ids:
            raise ValueError(f"Duplicate image_id: {image_id}")
        key_to_id[image_key] = image_id
        all_ids.add(image_id)

    validation_ids = {int(value) for value in test_ids_raw}
    if not validation_ids.issubset(all_ids):
        raise ValueError("Some validation image IDs are absent from annotation data")
    train_ids = all_ids - validation_ids
    metadata = {
        "data_items": len(data),
        "train_images": len(train_ids),
        "validation_images": len(validation_ids),
        "total_relations": int(sum(len(item.get("relations", [])) for item in data)),
        "zero_relation_images": int(sum(len(item.get("relations", [])) == 0 for item in data)),
    }
    return key_to_id, train_ids, validation_ids, metadata


def extract_features(
    rows: pd.DataFrame,
    cache: Mapping[Any, Any],
    key_to_id: Mapping[str, int],
    split_name: str,
    label_name: str,
) -> tuple[pd.DataFrame, ExtractAudit]:
    rows, duplicate_removed = dedupe_pairs(rows, label_name)
    records: list[dict[str, Any]] = []
    counters = {
        "missing_annotation_images": 0,
        "cache_image_missing": 0,
        "malformed_indices": 0,
        "index_out_of_range": 0,
        "invalid_cache_pairs": 0,
        "nonfinite_feature_rows": 0,
    }

    for row in rows.itertuples(index=False):
        payload = row._asdict()
        image_key = str(payload["global_image_key"]).strip()
        image_id = key_to_id.get(image_key)
        if image_id is None:
            counters["missing_annotation_images"] += 1
            continue
        entry = get_cache_entry(cache, image_id)
        if entry is None:
            counters["cache_image_missing"] += 1
            continue
        try:
            subject_index = int(payload["subject_index_v4"])
            object_index = int(payload["object_index_v4"])
        except (TypeError, ValueError):
            counters["malformed_indices"] += 1
            continue
        features = torch.as_tensor(entry.get("features"), dtype=torch.float32)
        valid_pairs = torch.as_tensor(entry.get("valid_pairs"), dtype=torch.bool)
        if features.ndim != 3 or tuple(valid_pairs.shape) != tuple(features.shape[:2]):
            raise ValueError(f"Malformed cache entry for image {image_id}")
        if features.shape[2] != 21:
            raise ValueError(f"Image {image_id}: expected 21 features, got {features.shape[2]}")
        if (
            subject_index < 0
            or object_index < 0
            or subject_index >= features.shape[0]
            or object_index >= features.shape[1]
        ):
            counters["index_out_of_range"] += 1
            continue
        if not bool(valid_pairs[subject_index, object_index].item()):
            counters["invalid_cache_pairs"] += 1
            continue
        vector = features[subject_index, object_index].detach().cpu().numpy().astype(np.float64)
        if not np.isfinite(vector).all():
            counters["nonfinite_feature_rows"] += 1
            continue
        record: dict[str, Any] = {
            "split": split_name,
            "label": int(payload["label"]),
            "global_image_key": image_key,
            "image_id": int(image_id),
            "subject_index_v4": subject_index,
            "object_index_v4": object_index,
            "pair_family_v4": str(payload["pair_family_v4"]),
            "scene_context_v4": str(payload.get("scene_context_v4", "")),
            "source_kind": str(payload.get("source_kind", "")),
        }
        for index, value in enumerate(vector):
            record[FEATURE_COLUMNS[index]] = float(value)
        records.append(record)

    output = pd.DataFrame.from_records(records)
    if output.empty:
        output = pd.DataFrame(columns=[
            "split", "label", "global_image_key", "image_id",
            "subject_index_v4", "object_index_v4", "pair_family_v4",
            "scene_context_v4", "source_kind", *FEATURE_COLUMNS,
        ])
    audit = ExtractAudit(
        requested_rows=len(rows),
        extracted_rows=len(output),
        duplicate_pair_rows_removed=duplicate_removed,
        **counters,
    )
    return output, audit


def make_probe_pipeline(name: str, seed: int) -> Pipeline:
    classifier = LogisticRegression(
        class_weight="balanced",
        random_state=seed,
        max_iter=5000,
        solver="liblinear",
    )
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = OneHotEncoder(handle_unknown="ignore")

    if name == "G":
        return Pipeline([
            ("numeric", numeric),
            ("classifier", classifier),
        ])
    if name == "F":
        preprocessor = ColumnTransformer([
            ("family", categorical, ["pair_family_v4"]),
        ], remainder="drop")
    elif name == "GF":
        preprocessor = ColumnTransformer([
            ("geometry", numeric, list(FEATURE_COLUMNS)),
            ("family", categorical, ["pair_family_v4"]),
        ], remainder="drop")
    else:
        raise KeyError(name)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def probe_input(frame: pd.DataFrame, name: str) -> Any:
    if name == "G":
        return frame.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    return frame.loc[:, [*FEATURE_COLUMNS, "pair_family_v4"]]


def binary_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= 0.5).astype(np.int64)
    unique = np.unique(y_true)
    auroc = float(roc_auc_score(y_true, probabilities)) if len(unique) == 2 else None
    auprc = float(average_precision_score(y_true, probabilities)) if len(unique) == 2 else None
    balanced = float(balanced_accuracy_score(y_true, predictions)) if len(unique) == 2 else None
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    sensitivity = float(tp / (tp + fn)) if tp + fn else None
    specificity = float(tn / (tn + fp)) if tn + fp else None
    brier = float(brier_score_loss(y_true, probabilities)) if len(y_true) else None
    return {
        "rows": int(len(y_true)),
        "positives": int((y_true == 1).sum()),
        "negatives": int((y_true == 0).sum()),
        "auroc": auroc,
        "auprc": auprc,
        "balanced_accuracy_at_0_5": balanced,
        "sensitivity_at_0_5": sensitivity,
        "specificity_at_0_5": specificity,
        "brier_score": brier,
        "confusion_at_0_5": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "mean_probability_positive": (
            float(probabilities[y_true == 1].mean()) if (y_true == 1).any() else None
        ),
        "mean_probability_negative": (
            float(probabilities[y_true == 0].mean()) if (y_true == 0).any() else None
        ),
    }


def cluster_bootstrap(
    predictions: pd.DataFrame,
    probability_column: str,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    groups = {
        key: group.index.to_numpy(dtype=np.int64)
        for key, group in predictions.groupby("global_image_key", sort=True)
    }
    image_keys = np.array(sorted(groups), dtype=object)
    if image_keys.size == 0:
        raise ValueError("No validation images for bootstrap")
    rng = np.random.default_rng(seed)
    metric_names = (
        "auroc",
        "auprc",
        "balanced_accuracy_at_0_5",
        "sensitivity_at_0_5",
        "specificity_at_0_5",
        "brier_score",
    )
    values: dict[str, list[float]] = {name: [] for name in metric_names}
    skipped_single_class = 0
    for _ in range(iterations):
        sampled = rng.choice(image_keys, size=len(image_keys), replace=True)
        indices = np.concatenate([groups[key] for key in sampled])
        sample = predictions.loc[indices]
        metrics = binary_metrics(
            sample["label"].to_numpy(),
            sample[probability_column].to_numpy(),
        )
        if metrics["auroc"] is None:
            skipped_single_class += 1
            continue
        for name in metric_names:
            value = metrics[name]
            if value is not None and math.isfinite(float(value)):
                values[name].append(float(value))
    summary: dict[str, Any] = {}
    for name, metric_values in values.items():
        array = np.asarray(metric_values, dtype=np.float64)
        summary[name] = {
            "valid_iterations": int(array.size),
            "mean": float(array.mean()) if array.size else None,
            "median": float(np.median(array)) if array.size else None,
            "ci_2_5": float(np.quantile(array, 0.025)) if array.size else None,
            "ci_97_5": float(np.quantile(array, 0.975)) if array.size else None,
        }
    return {
        "cluster": "global_image_key",
        "unique_validation_images": int(len(image_keys)),
        "requested_iterations": int(iterations),
        "skipped_single_class_iterations": int(skipped_single_class),
        "seed": int(seed),
        "metrics": summary,
    }


def extract_coefficients(pipeline: Pipeline, probe_name: str) -> pd.DataFrame:
    classifier: LogisticRegression = pipeline.named_steps["classifier"]
    coefficients = classifier.coef_.reshape(-1)
    if probe_name == "G":
        names = list(FEATURE_COLUMNS)
    else:
        preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
        try:
            names = [str(value) for value in preprocessor.get_feature_names_out()]
        except Exception:
            names = [f"feature_{index}" for index in range(len(coefficients))]
    if len(names) != len(coefficients):
        names = [f"feature_{index}" for index in range(len(coefficients))]
    return pd.DataFrame({
        "probe": probe_name,
        "feature": names,
        "coefficient": coefficients.astype(float),
        "abs_coefficient": np.abs(coefficients.astype(float)),
    }).sort_values(["probe", "abs_coefficient"], ascending=[True, False])


def safe_json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Mapping):
        return {str(key): safe_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json_value(item) for item in value]
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(safe_json_value(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.bootstrap_iterations <= 0:
        raise ValueError("bootstrap-iterations must be positive")
    input_paths = (
        args.train_cache,
        args.validation_cache,
        args.annotation,
        args.hardneg_source,
        args.frozen_validation_hn,
    )
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    key_to_id, train_ids, validation_ids, annotation_metadata = build_annotation_maps(args.annotation)
    if len(train_ids) != args.expected_train_images:
        raise ValueError(f"Expected {args.expected_train_images} train images, got {len(train_ids)}")
    if len(validation_ids) != args.expected_validation_images:
        raise ValueError(
            f"Expected {args.expected_validation_images} validation images, got {len(validation_ids)}"
        )

    train_cache, train_cache_metadata = load_cache(args.train_cache)
    validation_cache, validation_cache_metadata = load_cache(args.validation_cache)
    if len(train_cache) != args.expected_train_images:
        raise ValueError(f"Train cache entries mismatch: {len(train_cache)}")
    if len(validation_cache) != args.expected_validation_images:
        raise ValueError(f"Validation cache entries mismatch: {len(validation_cache)}")

    source_raw = pd.read_csv(args.hardneg_source, encoding="utf-8-sig", low_memory=False)
    frozen_raw = pd.read_csv(args.frozen_validation_hn, encoding="utf-8-sig", low_memory=False)
    source = normalize_source(source_raw, "hard-negative source")
    frozen = normalize_source(frozen_raw, "frozen validation HN")

    source_mask = eligible_base(source)
    positive_mask = (
        source_mask
        & source["audit_pair_status"].str.lower().eq("positive")
        & source["canonical_pair_positive_v4"]
    )
    negative_mask = (
        source_mask
        & source["audit_pair_status"].str.lower().eq("verified_negative")
        & ~source["canonical_pair_positive_v4"]
    )

    train_positive_rows = source.loc[
        positive_mask & source["audit_split_v2"].str.lower().eq("train")
    ].copy()
    train_negative_rows = source.loc[
        negative_mask & source["audit_split_v2"].str.lower().eq("train")
    ].copy()
    validation_positive_rows = source.loc[
        positive_mask & source["audit_split_v2"].str.lower().eq("validation")
    ].copy()

    frozen_mask = (
        eligible_base(frozen)
        & frozen["audit_split_v2"].str.lower().eq("validation")
        & frozen["audit_pair_status"].str.lower().eq("verified_negative")
        & ~frozen["canonical_pair_positive_v4"]
    )
    validation_negative_rows = frozen.loc[frozen_mask].copy()
    if len(validation_negative_rows) != args.expected_validation_hn_pairs:
        raise ValueError(
            f"Expected {args.expected_validation_hn_pairs} frozen validation negatives, "
            f"got {len(validation_negative_rows)}"
        )

    for frame, label, kind in (
        (train_positive_rows, 1, "positive"),
        (train_negative_rows, 0, "verified_water_hn"),
        (validation_positive_rows, 1, "positive"),
        (validation_negative_rows, 0, "frozen_validation_water_hn"),
    ):
        frame["label"] = int(label)
        frame["source_kind"] = kind

    train_positive, audit_train_positive = extract_features(
        train_positive_rows, train_cache, key_to_id, "train", "train positives"
    )
    train_negative, audit_train_negative = extract_features(
        train_negative_rows, train_cache, key_to_id, "train", "train negatives"
    )
    validation_positive, audit_validation_positive = extract_features(
        validation_positive_rows,
        validation_cache,
        key_to_id,
        "validation",
        "validation positives",
    )
    validation_negative, audit_validation_negative = extract_features(
        validation_negative_rows,
        validation_cache,
        key_to_id,
        "validation",
        "frozen validation negatives",
    )

    if len(validation_negative) != args.expected_validation_hn_pairs:
        raise ValueError(
            "Frozen validation HN feature coverage is incomplete: "
            f"{len(validation_negative)}/{args.expected_validation_hn_pairs}"
        )
    positive_incomplete = any(
        audit.requested_rows != audit.extracted_rows
        for audit in (audit_train_positive, audit_validation_positive)
    )
    if positive_incomplete and not args.allow_incomplete_positive_features:
        raise ValueError(
            "Positive feature extraction was incomplete. Re-run with "
            "--allow-incomplete-positive-features only after auditing exclusions."
        )
    if audit_train_negative.requested_rows != audit_train_negative.extracted_rows:
        raise ValueError("Train Water-HN feature extraction was incomplete")

    train = pd.concat([train_positive, train_negative], ignore_index=True)
    validation = pd.concat([validation_positive, validation_negative], ignore_index=True)
    if train["label"].nunique() != 2 or validation["label"].nunique() != 2:
        raise ValueError("Both train and validation probe datasets must contain two classes")

    train_images_observed = set(train["image_id"].astype(int))
    validation_images_observed = set(validation["image_id"].astype(int))
    if not train_images_observed.issubset(train_ids):
        raise ValueError("Train rows contain non-train annotation images")
    if not validation_images_observed.issubset(validation_ids):
        raise ValueError("Validation rows contain non-validation annotation images")
    image_overlap = train_images_observed & validation_images_observed
    if image_overlap:
        raise ValueError(f"Train/validation image leakage: {sorted(image_overlap)[:20]}")

    pair_columns = list(PAIR_KEY_COLUMNS)
    train_pair_keys = set(map(tuple, train[pair_columns].itertuples(index=False, name=None)))
    validation_pair_keys = set(map(tuple, validation[pair_columns].itertuples(index=False, name=None)))
    pair_overlap = train_pair_keys & validation_pair_keys
    if pair_overlap:
        raise ValueError(f"Train/validation pair leakage: {list(pair_overlap)[:20]}")

    args.output_dir.mkdir(parents=True, exist_ok=False)

    predictions = validation.loc[:, [
        "split", "label", "global_image_key", "image_id", "subject_index_v4",
        "object_index_v4", "pair_family_v4", "scene_context_v4", "source_kind",
    ]].copy()
    train_predictions = train.loc[:, [
        "split", "label", "global_image_key", "image_id", "subject_index_v4",
        "object_index_v4", "pair_family_v4", "scene_context_v4", "source_kind",
    ]].copy()

    probe_reports: dict[str, Any] = {}
    bootstrap_reports: dict[str, Any] = {}
    coefficient_frames: list[pd.DataFrame] = []
    per_family_rows: list[dict[str, Any]] = []

    for probe_index, probe_name in enumerate(("G", "F", "GF")):
        pipeline = make_probe_pipeline(probe_name, args.seed)
        pipeline.fit(probe_input(train, probe_name), train["label"].to_numpy(dtype=np.int64))
        train_probability = pipeline.predict_proba(probe_input(train, probe_name))[:, 1]
        validation_probability = pipeline.predict_proba(probe_input(validation, probe_name))[:, 1]
        probability_column = f"probability_{probe_name}"
        train_predictions[probability_column] = train_probability
        predictions[probability_column] = validation_probability
        train_metrics = binary_metrics(train["label"].to_numpy(), train_probability)
        validation_metrics = binary_metrics(validation["label"].to_numpy(), validation_probability)
        probe_reports[probe_name] = {
            "definition": {
                "G": "21D geometry only",
                "F": "pair-family only control",
                "GF": "21D geometry plus pair-family",
            }[probe_name],
            "train": train_metrics,
            "validation": validation_metrics,
            "fit": {
                "class_weight": "balanced",
                "max_iter": 5000,
                "solver": "liblinear",
                "random_state": int(args.seed),
                "n_iter": [int(value) for value in pipeline.named_steps["classifier"].n_iter_],
                "intercept": [float(value) for value in pipeline.named_steps["classifier"].intercept_],
            },
        }
        bootstrap_reports[probe_name] = cluster_bootstrap(
            predictions,
            probability_column,
            args.bootstrap_iterations,
            args.seed + 1000 * probe_index,
        )
        coefficient_frames.append(extract_coefficients(pipeline, probe_name))

        for family in WATER_FAMILIES:
            subset = predictions.loc[predictions["pair_family_v4"].eq(family)]
            if subset.empty:
                metrics = {
                    "rows": 0, "positives": 0, "negatives": 0, "auroc": None,
                    "auprc": None, "balanced_accuracy_at_0_5": None,
                    "sensitivity_at_0_5": None, "specificity_at_0_5": None,
                    "brier_score": None, "confusion_at_0_5": None,
                    "mean_probability_positive": None, "mean_probability_negative": None,
                }
            else:
                metrics = binary_metrics(
                    subset["label"].to_numpy(), subset[probability_column].to_numpy()
                )
            row = {
                "probe": probe_name,
                "pair_family_v4": family,
                "images": int(subset["global_image_key"].nunique()) if not subset.empty else 0,
                **{key: value for key, value in metrics.items() if key != "confusion_at_0_5"},
            }
            if metrics.get("confusion_at_0_5"):
                row.update({f"confusion_{key}": value for key, value in metrics["confusion_at_0_5"].items()})
            per_family_rows.append(row)

    dataset_columns_for_hash = [
        *PAIR_KEY_COLUMNS,
        "label", "pair_family_v4", "scene_context_v4", *FEATURE_COLUMNS,
    ]
    dataset_audit = {
        "protocol": "FloodPSG FIBE 21D matched linear-probe dataset audit V1",
        "status": "PASS",
        "final_test_allowed": False,
        "water_families": list(WATER_FAMILIES),
        "annotation": annotation_metadata,
        "cache_entries": {
            "train": len(train_cache),
            "validation": len(validation_cache),
        },
        "source_rows": {
            "all": len(source),
            "eligible_train_positive": len(train_positive_rows),
            "eligible_train_verified_negative": len(train_negative_rows),
            "eligible_validation_positive": len(validation_positive_rows),
            "frozen_validation_verified_negative": len(validation_negative_rows),
        },
        "extraction": {
            "train_positive": audit_train_positive.__dict__,
            "train_negative": audit_train_negative.__dict__,
            "validation_positive": audit_validation_positive.__dict__,
            "validation_negative": audit_validation_negative.__dict__,
        },
        "final_rows": {
            "train": len(train),
            "train_positive": int(train["label"].sum()),
            "train_negative": int((train["label"] == 0).sum()),
            "train_images": int(train["global_image_key"].nunique()),
            "validation": len(validation),
            "validation_positive": int(validation["label"].sum()),
            "validation_negative": int((validation["label"] == 0).sum()),
            "validation_images": int(validation["global_image_key"].nunique()),
        },
        "family_counts": {
            "train": train.groupby(["pair_family_v4", "label"]).size().unstack(fill_value=0).to_dict("index"),
            "validation": validation.groupby(["pair_family_v4", "label"]).size().unstack(fill_value=0).to_dict("index"),
        },
        "leakage_checks": {
            "train_validation_image_overlap": len(image_overlap),
            "train_validation_pair_overlap": len(pair_overlap),
        },
        "semantic_sha256": {
            "train_probe_rows": stable_frame_sha256(train, dataset_columns_for_hash),
            "validation_probe_rows": stable_frame_sha256(validation, dataset_columns_for_hash),
        },
        "cache_metadata": {
            "train": train_cache_metadata,
            "validation": validation_cache_metadata,
        },
    }

    main_report = {
        "protocol": "FloodPSG FIBE 21D matched linear-probe V1",
        "status": "PASS",
        "scope": "group-strict train fit; authoritative validation evaluation; final test locked",
        "seed": int(args.seed),
        "bootstrap_iterations": int(args.bootstrap_iterations),
        "final_test_allowed": False,
        "probe_definitions": {
            "G": "21D geometry only",
            "F": "pair-family only control",
            "GF": "21D geometry plus pair-family",
        },
        "probes": probe_reports,
        "decision_support": {
            "geometry_auroc": probe_reports["G"]["validation"]["auroc"],
            "family_control_auroc": probe_reports["F"]["validation"]["auroc"],
            "geometry_plus_family_auroc": probe_reports["GF"]["validation"]["auroc"],
            "gf_minus_f_auroc": (
                probe_reports["GF"]["validation"]["auroc"]
                - probe_reports["F"]["validation"]["auroc"]
            ),
            "gf_minus_g_auroc": (
                probe_reports["GF"]["validation"]["auroc"]
                - probe_reports["G"]["validation"]["auroc"]
            ),
            "guardrail": (
                "AUROC thresholds guide route selection but do not by themselves prove "
                "feature collapse or identify a unique causal failure mechanism."
            ),
        },
        "input_sha256": {
            "train_cache": sha256_file(args.train_cache),
            "validation_cache": sha256_file(args.validation_cache),
            "annotation": sha256_file(args.annotation),
            "hardneg_source": sha256_file(args.hardneg_source),
            "frozen_validation_hn": sha256_file(args.frozen_validation_hn),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "sklearn": __import__("sklearn").__version__,
            "pid": os.getpid(),
        },
    }

    dataset_audit_path = args.output_dir / "FIBE_21D_LINEAR_PROBE_DATASET_AUDIT_V1.json"
    report_path = args.output_dir / "FIBE_21D_LINEAR_PROBE_V1.json"
    bootstrap_path = args.output_dir / "FIBE_21D_IMAGE_CLUSTER_BOOTSTRAP_V1.json"
    predictions_path = args.output_dir / "FIBE_21D_LINEAR_PROBE_PREDICTIONS_V1.csv"
    train_predictions_path = args.output_dir / "FIBE_21D_LINEAR_PROBE_TRAIN_PREDICTIONS_V1.csv"
    family_path = args.output_dir / "FIBE_21D_LINEAR_PROBE_PER_FAMILY_V1.csv"
    coefficients_path = args.output_dir / "FIBE_21D_LINEAR_PROBE_COEFFICIENTS_V1.csv"

    write_json(dataset_audit_path, dataset_audit)
    write_json(report_path, main_report)
    write_json(bootstrap_path, {
        "protocol": "FloodPSG FIBE 21D image-cluster bootstrap V1",
        "status": "PASS",
        "cluster": "global_image_key",
        "final_test_allowed": False,
        "probes": bootstrap_reports,
    })
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    train_predictions.to_csv(train_predictions_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(per_family_rows).to_csv(family_path, index=False, encoding="utf-8-sig")
    pd.concat(coefficient_frames, ignore_index=True).to_csv(
        coefficients_path, index=False, encoding="utf-8-sig"
    )

    # The report does not hash itself; the runner lock records its final SHA256.
    output_hashes = {
        path.name: sha256_file(path)
        for path in (
            dataset_audit_path,
            bootstrap_path,
            predictions_path,
            train_predictions_path,
            family_path,
            coefficients_path,
        )
    }
    main_report["output_sha256_excluding_self"] = output_hashes
    write_json(report_path, main_report)

    print(json.dumps(main_report, ensure_ascii=False, indent=2, allow_nan=False))
    print("\nFIBE 21D MATCHED LINEAR PROBE V1: PASS")


if __name__ == "__main__":
    main()
