from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (
    PROJECT_ROOT / "data" / "floodpsg"
).resolve()

DEFAULT_INPUT = (
    DATA_ROOT
    / "features"
    / "fibe_scalar_v1"
    / "fibe_boundary_audit_eligible_v1.csv"
)

DEFAULT_ANNOTATION = (
    DATA_ROOT
    / "annotations"
    / "floodpsg_canonical_trainval_coarse8_groupstrict_v1.json"
)

DEFAULT_SUMMARY = (
    DATA_ROOT
    / "stats"
    / "dsformer_fibe_d1_v1"
    / "FIBE_CONTACT_RADII_CALIBRATION_V1.json"
)

DEFAULT_GRID = (
    DATA_ROOT
    / "stats"
    / "dsformer_fibe_d1_v1"
    / "FIBE_CONTACT_RADII_THRESHOLD_GRID_V1.csv"
)

DEFAULT_BOOTSTRAP = (
    DATA_ROOT
    / "stats"
    / "dsformer_fibe_d1_v1"
    / "FIBE_CONTACT_RADII_BOOTSTRAP_V1.csv"
)

DEFAULT_FAMILY_METRICS = (
    DATA_ROOT
    / "stats"
    / "dsformer_fibe_d1_v1"
    / "FIBE_CONTACT_RADII_FAMILY_METRICS_V1.csv"
)

DEFAULT_DISTRIBUTIONS = (
    DATA_ROOT
    / "stats"
    / "dsformer_fibe_d1_v1"
    / "FIBE_CONTACT_DISTANCE_DISTRIBUTIONS_V1.csv"
)

DEFAULT_SEED = 3407
DEFAULT_BOOTSTRAP_REPEATS = 2000

STRONG_CONTACT_LABELS = {
    "overlap",
    "touching",
}

POSSIBLE_CONTACT_LABELS = {
    "overlap",
    "touching",
    "near_gap",
}

ALLOWED_GEOMETRY_LABELS = {
    "overlap",
    "touching",
    "near_gap",
    "far_gap",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate global normalized FIBE contact radii "
            "from the training-only manually reviewed boundary "
            "audit sample."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=DEFAULT_ANNOTATION,
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
    )
    parser.add_argument(
        "--grid-output",
        type=Path,
        default=DEFAULT_GRID,
    )
    parser.add_argument(
        "--bootstrap-output",
        type=Path,
        default=DEFAULT_BOOTSTRAP,
    )
    parser.add_argument(
        "--family-output",
        type=Path,
        default=DEFAULT_FAMILY_METRICS,
    )
    parser.add_argument(
        "--distribution-output",
        type=Path,
        default=DEFAULT_DISTRIBUTIONS,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )
    parser.add_argument(
        "--bootstrap-repeats",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPEATS,
    )

    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def clean_text(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def percentile_interval(
    values: Iterable[float],
    *,
    lower: float = 2.5,
    upper: float = 97.5,
) -> dict[str, float]:
    array = np.asarray(
        list(values),
        dtype=np.float64,
    )

    if len(array) == 0:
        raise ValueError(
            "Cannot calculate interval from no values"
        )

    return {
        "median": float(
            np.percentile(array, 50.0)
        ),
        "lower_95": float(
            np.percentile(array, lower)
        ),
        "upper_95": float(
            np.percentile(array, upper)
        ),
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float | int]:
    y_true = np.asarray(
        y_true,
        dtype=bool,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=bool,
    )

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred have different shapes"
        )

    tp = int(
        np.logical_and(
            y_true,
            y_pred,
        ).sum()
    )

    tn = int(
        np.logical_and(
            ~y_true,
            ~y_pred,
        ).sum()
    )

    fp = int(
        np.logical_and(
            ~y_true,
            y_pred,
        ).sum()
    )

    fn = int(
        np.logical_and(
            y_true,
            ~y_pred,
        ).sum()
    )

    sensitivity = (
        tp / (tp + fn)
        if tp + fn > 0
        else float("nan")
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp > 0
        else float("nan")
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else float("nan")
    )

    negative_predictive_value = (
        tn / (tn + fn)
        if tn + fn > 0
        else float("nan")
    )

    accuracy = (
        (tp + tn) / len(y_true)
        if len(y_true) > 0
        else float("nan")
    )

    balanced_accuracy = (
        0.5
        * (
            sensitivity
            + specificity
        )
        if (
            math.isfinite(sensitivity)
            and math.isfinite(specificity)
        )
        else float("nan")
    )

    f1 = (
        2.0
        * precision
        * sensitivity
        / (
            precision
            + sensitivity
        )
        if (
            math.isfinite(precision)
            and math.isfinite(sensitivity)
            and precision + sensitivity > 0
        )
        else float("nan")
    )

    return {
        "rows": int(len(y_true)),
        "positive_rows": int(
            y_true.sum()
        ),
        "negative_rows": int(
            (~y_true).sum()
        ),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": float(
            sensitivity
        ),
        "specificity": float(
            specificity
        ),
        "precision": float(
            precision
        ),
        "negative_predictive_value": float(
            negative_predictive_value
        ),
        "accuracy": float(
            accuracy
        ),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "f1": float(f1),
    }


def threshold_candidates(
    distances: np.ndarray,
) -> np.ndarray:
    values = np.unique(
        np.asarray(
            distances,
            dtype=np.float64,
        )
    )

    if len(values) == 0:
        raise ValueError(
            "No distance values"
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "Distance values contain NaN or Inf"
        )

    if (values < 0).any():
        raise ValueError(
            "Distance values contain negative values"
        )

    candidates: list[float] = []

    candidates.append(
        max(
            0.0,
            float(values[0])
            - np.finfo(np.float64).eps,
        )
    )

    for left, right in zip(
        values[:-1],
        values[1:],
    ):
        candidates.append(
            float(
                left
                + 0.5 * (right - left)
            )
        )

    candidates.append(
        float(
            values[-1]
            + max(
                np.finfo(np.float64).eps,
                abs(values[-1]) * 1e-9,
            )
        )
    )

    return np.unique(
        np.asarray(
            candidates,
            dtype=np.float64,
        )
    )


def make_binary_target(
    labels: pd.Series,
    positive_labels: set[str],
) -> np.ndarray:
    return (
        labels.astype(str)
        .isin(positive_labels)
        .to_numpy(dtype=bool)
    )


def build_threshold_grid(
    *,
    distances: np.ndarray,
    y_true: np.ndarray,
    task_name: str,
    minimum_threshold_exclusive: float | None = None,
) -> pd.DataFrame:
    candidates = threshold_candidates(
        distances
    )

    if minimum_threshold_exclusive is not None:
        candidates = candidates[
            candidates
            > minimum_threshold_exclusive
            + 1e-15
        ]

    if len(candidates) == 0:
        raise RuntimeError(
            f"No threshold candidates for {task_name}"
        )

    records: list[dict[str, Any]] = []

    for threshold in candidates:
        prediction = (
            distances <= threshold
        )

        metrics = classification_metrics(
            y_true,
            prediction,
        )

        records.append(
            {
                "task": task_name,
                "threshold": float(
                    threshold
                ),
                **metrics,
            }
        )

    return pd.DataFrame(records)


def select_best_threshold(
    grid: pd.DataFrame,
) -> pd.Series:
    valid = grid[
        np.isfinite(
            grid["balanced_accuracy"]
            .to_numpy(dtype=float)
        )
    ].copy()

    if valid.empty:
        raise RuntimeError(
            "No valid threshold candidates"
        )

    # Primary criterion:
    # maximum balanced accuracy.
    #
    # Tie breaking:
    # 1. higher specificity;
    # 2. smaller radius.
    valid.sort_values(
        by=[
            "balanced_accuracy",
            "specificity",
            "threshold",
        ],
        ascending=[
            False,
            False,
            True,
        ],
        kind="stable",
        inplace=True,
    )

    return valid.iloc[0]


def calibrate_thresholds(
    frame: pd.DataFrame,
) -> tuple[
    pd.Series,
    pd.Series,
    pd.DataFrame,
]:
    distances = frame[
        "minimum_boundary_distance_norm"
    ].to_numpy(dtype=np.float64)

    labels = frame[
        "manual_geometry_relation"
    ]

    y_strong = make_binary_target(
        labels,
        STRONG_CONTACT_LABELS,
    )

    y_possible = make_binary_target(
        labels,
        POSSIBLE_CONTACT_LABELS,
    )

    if (
        y_strong.all()
        or (~y_strong).all()
    ):
        raise RuntimeError(
            "Strong-contact task has only one class"
        )

    if (
        y_possible.all()
        or (~y_possible).all()
    ):
        raise RuntimeError(
            "Possible-contact task has only one class"
        )

    r1_grid = build_threshold_grid(
        distances=distances,
        y_true=y_strong,
        task_name="r1_strong_contact",
    )

    r1_best = select_best_threshold(
        r1_grid
    )

    r1 = float(
        r1_best["threshold"]
    )

    r2_grid = build_threshold_grid(
        distances=distances,
        y_true=y_possible,
        task_name="r2_possible_contact",
        minimum_threshold_exclusive=r1,
    )

    r2_best = select_best_threshold(
        r2_grid
    )

    r2 = float(
        r2_best["threshold"]
    )

    if not (0.0 <= r1 < r2):
        raise RuntimeError(
            f"Invalid calibrated radii: "
            f"r1={r1}, r2={r2}"
        )

    full_grid = pd.concat(
        [
            r1_grid,
            r2_grid,
        ],
        ignore_index=True,
    )

    full_grid["selected"] = False

    full_grid.loc[
        (
            full_grid["task"]
            == "r1_strong_contact"
        )
        & np.isclose(
            full_grid["threshold"],
            r1,
            rtol=0.0,
            atol=1e-15,
        ),
        "selected",
    ] = True

    full_grid.loc[
        (
            full_grid["task"]
            == "r2_possible_contact"
        )
        & np.isclose(
            full_grid["threshold"],
            r2,
            rtol=0.0,
            atol=1e-15,
        ),
        "selected",
    ] = True

    return (
        r1_best,
        r2_best,
        full_grid,
    )


def metrics_at_threshold(
    frame: pd.DataFrame,
    *,
    positive_labels: set[str],
    threshold: float,
) -> dict[str, float | int]:
    distances = frame[
        "minimum_boundary_distance_norm"
    ].to_numpy(dtype=np.float64)

    y_true = make_binary_target(
        frame["manual_geometry_relation"],
        positive_labels,
    )

    y_pred = (
        distances <= threshold
    )

    return classification_metrics(
        y_true,
        y_pred,
    )


def cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    repeats: int,
    seed: int,
    fixed_r1: float,
    fixed_r2: float,
) -> pd.DataFrame:
    image_ids = np.asarray(
        sorted(
            frame["image_id"]
            .astype(int)
            .unique()
        ),
        dtype=np.int64,
    )

    groups = {
        int(image_id): frame[
            frame["image_id"]
            .astype(int)
            == int(image_id)
        ].copy()
        for image_id in image_ids
    }

    rng = np.random.default_rng(
        seed
    )

    records: list[
        dict[str, Any]
    ] = []

    successful = 0
    attempts = 0
    maximum_attempts = max(
        repeats * 10,
        repeats + 100,
    )

    while (
        successful < repeats
        and attempts < maximum_attempts
    ):
        attempts += 1

        sampled_ids = rng.choice(
            image_ids,
            size=len(image_ids),
            replace=True,
        )

        sampled_parts = [
            groups[int(image_id)]
            for image_id in sampled_ids
        ]

        bootstrap_frame = pd.concat(
            sampled_parts,
            ignore_index=True,
        )

        try:
            (
                r1_best,
                r2_best,
                _,
            ) = calibrate_thresholds(
                bootstrap_frame
            )
        except RuntimeError:
            continue

        r1 = float(
            r1_best["threshold"]
        )
        r2 = float(
            r2_best["threshold"]
        )

        fixed_r1_metrics = (
            metrics_at_threshold(
                bootstrap_frame,
                positive_labels=(
                    STRONG_CONTACT_LABELS
                ),
                threshold=fixed_r1,
            )
        )

        fixed_r2_metrics = (
            metrics_at_threshold(
                bootstrap_frame,
                positive_labels=(
                    POSSIBLE_CONTACT_LABELS
                ),
                threshold=fixed_r2,
            )
        )

        records.append(
            {
                "bootstrap_index": (
                    successful + 1
                ),
                "sampled_cluster_count": int(
                    len(sampled_ids)
                ),
                "unique_sampled_images": int(
                    len(set(sampled_ids.tolist()))
                ),
                "sampled_rows": int(
                    len(bootstrap_frame)
                ),
                "calibrated_r1": r1,
                "calibrated_r2": r2,
                "calibrated_r1_balanced_accuracy": (
                    float(
                        r1_best[
                            "balanced_accuracy"
                        ]
                    )
                ),
                "calibrated_r2_balanced_accuracy": (
                    float(
                        r2_best[
                            "balanced_accuracy"
                        ]
                    )
                ),
                "fixed_r1_balanced_accuracy": (
                    fixed_r1_metrics[
                        "balanced_accuracy"
                    ]
                ),
                "fixed_r1_sensitivity": (
                    fixed_r1_metrics[
                        "sensitivity"
                    ]
                ),
                "fixed_r1_specificity": (
                    fixed_r1_metrics[
                        "specificity"
                    ]
                ),
                "fixed_r2_balanced_accuracy": (
                    fixed_r2_metrics[
                        "balanced_accuracy"
                    ]
                ),
                "fixed_r2_sensitivity": (
                    fixed_r2_metrics[
                        "sensitivity"
                    ]
                ),
                "fixed_r2_specificity": (
                    fixed_r2_metrics[
                        "specificity"
                    ]
                ),
            }
        )

        successful += 1

    if successful < repeats:
        raise RuntimeError(
            "Insufficient successful bootstrap "
            f"replicates: {successful}/{repeats}; "
            f"attempts={attempts}"
        )

    return pd.DataFrame(records)


def grouped_metrics(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    r1: float,
    r2: float,
) -> pd.DataFrame:
    records: list[
        dict[str, Any]
    ] = []

    grouped = frame.groupby(
        group_columns,
        observed=True,
        sort=True,
    )

    for group_key, group in grouped:
        if not isinstance(
            group_key,
            tuple,
        ):
            group_key = (
                group_key,
            )

        base = {
            column: value
            for column, value in zip(
                group_columns,
                group_key,
            )
        }

        r1_metrics = metrics_at_threshold(
            group,
            positive_labels=(
                STRONG_CONTACT_LABELS
            ),
            threshold=r1,
        )

        r2_metrics = metrics_at_threshold(
            group,
            positive_labels=(
                POSSIBLE_CONTACT_LABELS
            ),
            threshold=r2,
        )

        records.append(
            {
                **base,
                "rows": int(len(group)),
                "images": int(
                    group["image_id"]
                    .nunique()
                ),
                "r1_positive_rows": int(
                    r1_metrics[
                        "positive_rows"
                    ]
                ),
                "r1_negative_rows": int(
                    r1_metrics[
                        "negative_rows"
                    ]
                ),
                "r1_balanced_accuracy": (
                    r1_metrics[
                        "balanced_accuracy"
                    ]
                ),
                "r1_sensitivity": (
                    r1_metrics[
                        "sensitivity"
                    ]
                ),
                "r1_specificity": (
                    r1_metrics[
                        "specificity"
                    ]
                ),
                "r2_positive_rows": int(
                    r2_metrics[
                        "positive_rows"
                    ]
                ),
                "r2_negative_rows": int(
                    r2_metrics[
                        "negative_rows"
                    ]
                ),
                "r2_balanced_accuracy": (
                    r2_metrics[
                        "balanced_accuracy"
                    ]
                ),
                "r2_sensitivity": (
                    r2_metrics[
                        "sensitivity"
                    ]
                ),
                "r2_specificity": (
                    r2_metrics[
                        "specificity"
                    ]
                ),
            }
        )

    return pd.DataFrame(records)


def distance_distributions(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    records: list[
        dict[str, Any]
    ] = []

    for geometry, group in frame.groupby(
        "manual_geometry_relation",
        observed=True,
        sort=True,
    ):
        norm = group[
            "minimum_boundary_distance_norm"
        ].to_numpy(dtype=np.float64)

        pixels = group[
            "minimum_boundary_distance_px"
        ].to_numpy(dtype=np.float64)

        records.append(
            {
                "manual_geometry_relation": (
                    str(geometry)
                ),
                "rows": int(len(group)),
                "images": int(
                    group["image_id"]
                    .nunique()
                ),
                "distance_norm_min": float(
                    np.min(norm)
                ),
                "distance_norm_q10": float(
                    np.percentile(norm, 10)
                ),
                "distance_norm_q25": float(
                    np.percentile(norm, 25)
                ),
                "distance_norm_median": float(
                    np.percentile(norm, 50)
                ),
                "distance_norm_q75": float(
                    np.percentile(norm, 75)
                ),
                "distance_norm_q90": float(
                    np.percentile(norm, 90)
                ),
                "distance_norm_max": float(
                    np.max(norm)
                ),
                "distance_px_min": float(
                    np.min(pixels)
                ),
                "distance_px_q25": float(
                    np.percentile(pixels, 25)
                ),
                "distance_px_median": float(
                    np.percentile(pixels, 50)
                ),
                "distance_px_q75": float(
                    np.percentile(pixels, 75)
                ),
                "distance_px_max": float(
                    np.max(pixels)
                ),
            }
        )

    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()

    input_path = (
        args.input.expanduser().resolve()
    )
    annotation_path = (
        args.annotation.expanduser().resolve()
    )
    summary_path = (
        args.summary.expanduser().resolve()
    )
    grid_path = (
        args.grid_output
        .expanduser()
        .resolve()
    )
    bootstrap_path = (
        args.bootstrap_output
        .expanduser()
        .resolve()
    )
    family_path = (
        args.family_output
        .expanduser()
        .resolve()
    )
    distribution_path = (
        args.distribution_output
        .expanduser()
        .resolve()
    )

    if not input_path.is_file():
        raise FileNotFoundError(
            input_path
        )

    if not annotation_path.is_file():
        raise FileNotFoundError(
            annotation_path
        )

    frame = pd.read_csv(
        input_path,
        encoding="utf-8-sig",
        keep_default_na=False,
        low_memory=False,
    )

    required_columns = {
        "audit_pair_id",
        "image_id",
        "source_type",
        "pair_family",
        "manual_review_status",
        "manual_mask_pair_quality",
        "manual_geometry_relation",
        "manual_semantic_contact",
        "minimum_boundary_distance_norm",
        "minimum_boundary_distance_px",
        "image_width",
        "image_height",
    }

    missing_columns = (
        required_columns
        - set(frame.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Eligible audit missing columns: "
            f"{sorted(missing_columns)}"
        )

    if len(frame) != 182:
        raise RuntimeError(
            f"Expected 182 eligible rows, "
            f"found {len(frame)}"
        )

    if frame[
        "audit_pair_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate audit_pair_id values"
        )

    if not (
        frame["manual_review_status"]
        == "complete"
    ).all():
        raise RuntimeError(
            "Eligible table contains "
            "non-complete rows"
        )

    if not (
        frame["manual_mask_pair_quality"]
        == "valid"
    ).all():
        raise RuntimeError(
            "Eligible table contains "
            "non-valid masks"
        )

    geometry_labels = set(
        frame[
            "manual_geometry_relation"
        ].astype(str)
    )

    unexpected_geometry = (
        geometry_labels
        - ALLOWED_GEOMETRY_LABELS
    )

    if unexpected_geometry:
        raise RuntimeError(
            "Unexpected manual geometry labels: "
            f"{sorted(unexpected_geometry)}"
        )

    numeric_columns = [
        "minimum_boundary_distance_norm",
        "minimum_boundary_distance_px",
        "image_width",
        "image_height",
    ]

    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="raise",
        )

    if not np.isfinite(
        frame[
            "minimum_boundary_distance_norm"
        ].to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            "Normalized distances contain "
            "NaN or Inf"
        )

    annotation = json.loads(
        annotation_path.read_text(
            encoding="utf-8"
        )
    )

    validation_ids = {
        int(value)
        for value in annotation.get(
            "test_image_ids",
            []
        )
    }

    leakage = int(
        frame["image_id"]
        .astype(int)
        .isin(validation_ids)
        .sum()
    )

    if leakage:
        raise RuntimeError(
            "Validation leakage detected: "
            f"{leakage}"
        )

    (
        r1_best,
        r2_best,
        threshold_grid,
    ) = calibrate_thresholds(frame)

    r1 = float(
        r1_best["threshold"]
    )
    r2 = float(
        r2_best["threshold"]
    )

    bootstrap = cluster_bootstrap(
        frame,
        repeats=args.bootstrap_repeats,
        seed=args.seed,
        fixed_r1=r1,
        fixed_r2=r2,
    )

    family_metrics = grouped_metrics(
        frame,
        group_columns=[
            "pair_family",
        ],
        r1=r1,
        r2=r2,
    )

    source_family_metrics = (
        grouped_metrics(
            frame,
            group_columns=[
                "source_type",
                "pair_family",
            ],
            r1=r1,
            r2=r2,
        )
    )

    source_family_metrics.insert(
        0,
        "metric_scope",
        "source_family",
    )

    family_metrics.insert(
        0,
        "metric_scope",
        "family",
    )

    combined_group_metrics = pd.concat(
        [
            family_metrics,
            source_family_metrics,
        ],
        ignore_index=True,
        sort=False,
    )

    distributions = (
        distance_distributions(frame)
    )

    diagonals = np.hypot(
        frame["image_width"]
        .to_numpy(dtype=float),
        frame["image_height"]
        .to_numpy(dtype=float),
    )

    diagonal_quantiles = {
        "q25": float(
            np.percentile(
                diagonals,
                25,
            )
        ),
        "median": float(
            np.percentile(
                diagonals,
                50,
            )
        ),
        "q75": float(
            np.percentile(
                diagonals,
                75,
            )
        ),
    }

    pixel_equivalents = {
        name: {
            "image_diagonal_px": diagonal,
            "r1_px": float(
                r1 * diagonal
            ),
            "r2_px": float(
                r2 * diagonal
            ),
        }
        for name, diagonal
        in diagonal_quantiles.items()
    }

    r1_interval = percentile_interval(
        bootstrap["calibrated_r1"]
    )

    r2_interval = percentile_interval(
        bootstrap["calibrated_r2"]
    )

    r1_ba_interval = percentile_interval(
        bootstrap[
            "fixed_r1_balanced_accuracy"
        ]
    )

    r2_ba_interval = percentile_interval(
        bootstrap[
            "fixed_r2_balanced_accuracy"
        ]
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    grid_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    threshold_grid.to_csv(
        grid_path,
        index=False,
        encoding="utf-8-sig",
    )

    bootstrap.to_csv(
        bootstrap_path,
        index=False,
        encoding="utf-8-sig",
    )

    combined_group_metrics.to_csv(
        family_path,
        index=False,
        encoding="utf-8-sig",
    )

    distributions.to_csv(
        distribution_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "version": (
            "FIBE_CONTACT_RADII_CALIBRATION_V1"
        ),
        "input_path": str(
            input_path
        ),
        "input_sha256": (
            file_sha256(input_path)
        ),
        "annotation_path": str(
            annotation_path
        ),
        "rows": int(len(frame)),
        "images": int(
            frame["image_id"]
            .nunique()
        ),
        "validation_leakage": int(
            leakage
        ),
        "selection_rule": {
            "distance": (
                "minimum_boundary_distance_norm"
            ),
            "normalization": (
                "full-image diagonal"
            ),
            "prediction_rule": (
                "distance <= radius"
            ),
            "optimization": (
                "maximum balanced accuracy"
            ),
            "tie_breaking": [
                "higher specificity",
                "smaller radius",
            ],
            "r1_positive_manual_labels": (
                sorted(
                    STRONG_CONTACT_LABELS
                )
            ),
            "r2_positive_manual_labels": (
                sorted(
                    POSSIBLE_CONTACT_LABELS
                )
            ),
            "constraint": (
                "0 <= r1 < r2"
            ),
        },
        "r1": {
            "normalized_radius": r1,
            "point_metrics": {
                key: (
                    float(value)
                    if isinstance(
                        value,
                        (
                            np.floating,
                            float,
                        ),
                    )
                    else int(value)
                    if isinstance(
                        value,
                        (
                            np.integer,
                            int,
                        ),
                    )
                    else value
                )
                for key, value
                in r1_best.to_dict().items()
            },
            "cluster_bootstrap_interval": (
                r1_interval
            ),
            "fixed_threshold_balanced_accuracy_interval": (
                r1_ba_interval
            ),
        },
        "r2": {
            "normalized_radius": r2,
            "point_metrics": {
                key: (
                    float(value)
                    if isinstance(
                        value,
                        (
                            np.floating,
                            float,
                        ),
                    )
                    else int(value)
                    if isinstance(
                        value,
                        (
                            np.integer,
                            int,
                        ),
                    )
                    else value
                )
                for key, value
                in r2_best.to_dict().items()
            },
            "cluster_bootstrap_interval": (
                r2_interval
            ),
            "fixed_threshold_balanced_accuracy_interval": (
                r2_ba_interval
            ),
        },
        "image_diagonal_quantiles_px": (
            diagonal_quantiles
        ),
        "pixel_equivalents": (
            pixel_equivalents
        ),
        "bootstrap": {
            "seed": int(args.seed),
            "requested_repeats": int(
                args.bootstrap_repeats
            ),
            "successful_repeats": int(
                len(bootstrap)
            ),
            "cluster_unit": "image_id",
        },
        "outputs": {
            "threshold_grid": str(
                grid_path
            ),
            "bootstrap": str(
                bootstrap_path
            ),
            "family_metrics": str(
                family_path
            ),
            "distance_distributions": str(
                distribution_path
            ),
        },
    }

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 100)
    print(
        "FIBE CONTACT RADII "
        "CALIBRATION V1"
    )
    print("=" * 100)
    print("eligible rows:", len(frame))
    print(
        "unique images:",
        frame["image_id"].nunique(),
    )
    print(
        "validation leakage:",
        leakage,
    )

    print("\nMANUAL GEOMETRY COUNTS")
    print(
        frame[
            "manual_geometry_relation"
        ]
        .value_counts()
        .to_string()
    )

    print("\nDISTANCE DISTRIBUTIONS")
    print(
        distributions.to_string(
            index=False
        )
    )

    print("\nCALIBRATED RADII")
    print(
        "r1 normalized:",
        r1,
    )
    print(
        "r1 balanced accuracy:",
        r1_best[
            "balanced_accuracy"
        ],
    )
    print(
        "r1 sensitivity:",
        r1_best["sensitivity"],
    )
    print(
        "r1 specificity:",
        r1_best["specificity"],
    )
    print(
        "r1 bootstrap 95%:",
        (
            r1_interval[
                "lower_95"
            ],
            r1_interval[
                "upper_95"
            ],
        ),
    )

    print()
    print(
        "r2 normalized:",
        r2,
    )
    print(
        "r2 balanced accuracy:",
        r2_best[
            "balanced_accuracy"
        ],
    )
    print(
        "r2 sensitivity:",
        r2_best["sensitivity"],
    )
    print(
        "r2 specificity:",
        r2_best["specificity"],
    )
    print(
        "r2 bootstrap 95%:",
        (
            r2_interval[
                "lower_95"
            ],
            r2_interval[
                "upper_95"
            ],
        ),
    )

    print("\nPIXEL EQUIVALENTS")
    for name, values in (
        pixel_equivalents.items()
    ):
        print(
            f"{name}: "
            f"diagonal="
            f"{values['image_diagonal_px']:.3f}, "
            f"r1_px="
            f"{values['r1_px']:.3f}, "
            f"r2_px="
            f"{values['r2_px']:.3f}"
        )

    print("\nFAMILY METRICS")
    print(
        family_metrics.to_string(
            index=False
        )
    )

    print("\nOUTPUTS")
    print("summary:", summary_path)
    print(
        "threshold grid:",
        grid_path,
    )
    print(
        "bootstrap:",
        bootstrap_path,
    )
    print(
        "family metrics:",
        family_path,
    )
    print(
        "distance distributions:",
        distribution_path,
    )

    print(
        "\nFIBE CONTACT RADII "
        "CALIBRATION: PASS"
    )


if __name__ == "__main__":
    main()
