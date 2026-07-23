from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (PROJECT_ROOT / "data" / "floodpsg").resolve()

DEFAULT_ANNOTATION = (
    DATA_ROOT
    / "annotations"
    / "floodpsg_canonical_trainval_coarse8_groupstrict_v1.json"
)

DEFAULT_MANIFEST = (
    DATA_ROOT
    / "features"
    / "fibe_scalar_v1"
    / "fibe_canonical_mask_manifest_v1.csv"
)

DEFAULT_TRAIN_HN = (
    DATA_ROOT
    / "stats"
    / "frozen_hardneg_eval_v1"
    / "01_train_hardneg_all.csv"
)

DEFAULT_OUTPUT_CSV = (
    DATA_ROOT
    / "features"
    / "fibe_scalar_v1"
    / "fibe_boundary_audit_pair_pool_train_v1.csv"
)

DEFAULT_SUMMARY = (
    DATA_ROOT
    / "stats"
    / "dsformer_fibe_d1_v1"
    / "FIBE_PAIR_POOL_AUDIT_V1.json"
)


HAZARD_CATEGORIES = {
    "water",
    "river",
    "mud_debris",
    "barricade",
}

TARGET_CATEGORIES = {
    "person",
    "vehicle",
    "road",
    "sidewalk",
    "ground",
    "building",
    "underpass_bridge",
    "drain",
    "manhole",
}

ROAD_SURFACE_CATEGORIES = {
    "road",
    "sidewalk",
    "ground",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and audit the training-only FIBE pair pool "
            "from positive FloodPSG relations and frozen hard "
            "negative pairs."
        )
    )

    parser.add_argument(
        "--annotation",
        type=Path,
        default=DEFAULT_ANNOTATION,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--train-hardneg",
        type=Path,
        default=DEFAULT_TRAIN_HN,
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
    )

    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def scale_bin_fixed(area_ratio: float) -> str:
    if area_ratio < 0.0025:
        return "tiny"
    if area_ratio < 0.01:
        return "small"
    if area_ratio < 0.05:
        return "medium"
    return "large"


def bbox_touches_border(
    bbox: list[Any],
    width: int,
    height: int,
    *,
    tolerance: int = 1,
) -> bool:
    if len(bbox) != 4:
        return False

    x, y, w, h = [
        float(value)
        for value in bbox
    ]

    x2 = x + w
    y2 = y + h

    return bool(
        x <= tolerance
        or y <= tolerance
        or x2 >= width - tolerance
        or y2 >= height - tolerance
    )


def pair_family(
    target_category: str,
    hazard_category: str,
) -> str:
    if hazard_category in {"water", "river"}:
        if target_category == "person":
            return "human-water"

        if target_category == "vehicle":
            return "vehicle-water"

        if target_category in ROAD_SURFACE_CATEGORIES:
            return "road-surface-water"

        if target_category == "building":
            return "building-water"

        if target_category == "underpass_bridge":
            return "bridge-water"

        if target_category in {"drain", "manhole"}:
            return "drainage-water"

    if (
        hazard_category == "mud_debris"
        and target_category in ROAD_SURFACE_CATEGORIES
    ):
        return "road-debris"

    if (
        hazard_category == "barricade"
        and target_category in ROAD_SURFACE_CATEGORIES
    ):
        return "road-barricade"

    return "other"


def orient_pair(
    segments: list[dict[str, Any]],
    first_index: int,
    second_index: int,
) -> dict[str, Any]:
    if first_index == second_index:
        return {
            "valid": False,
            "reason": "self_pair",
        }

    if not (
        0 <= first_index < len(segments)
        and 0 <= second_index < len(segments)
    ):
        return {
            "valid": False,
            "reason": "index_out_of_range",
        }

    first_category = normalize_text(
        segments[first_index].get(
            "category_name"
        )
    )
    second_category = normalize_text(
        segments[second_index].get(
            "category_name"
        )
    )

    first_is_hazard = (
        first_category in HAZARD_CATEGORIES
    )
    second_is_hazard = (
        second_category in HAZARD_CATEGORIES
    )

    if first_is_hazard == second_is_hazard:
        return {
            "valid": False,
            "reason": (
                "both_hazard"
                if first_is_hazard
                else "no_hazard"
            ),
            "first_category": first_category,
            "second_category": second_category,
        }

    if first_is_hazard:
        hazard_index = first_index
        target_index = second_index
    else:
        hazard_index = second_index
        target_index = first_index

    target_category = normalize_text(
        segments[target_index].get(
            "category_name"
        )
    )
    hazard_category = normalize_text(
        segments[hazard_index].get(
            "category_name"
        )
    )

    if target_category not in TARGET_CATEGORIES:
        return {
            "valid": False,
            "reason": "unsupported_target_category",
            "target_category": target_category,
            "hazard_category": hazard_category,
        }

    family = pair_family(
        target_category,
        hazard_category,
    )

    if family == "other":
        return {
            "valid": False,
            "reason": "unsupported_pair_family",
            "target_category": target_category,
            "hazard_category": hazard_category,
        }

    return {
        "valid": True,
        "reason": "valid",
        "target_index": int(target_index),
        "hazard_index": int(hazard_index),
        "target_category": target_category,
        "hazard_category": hazard_category,
        "pair_family": family,
        "direction_swapped": int(
            target_index != first_index
        ),
    }


def load_annotation(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(
        data.get("data"),
        list,
    ):
        raise RuntimeError(
            "Annotation has no valid data list"
        )

    return data


def load_manifest(
    path: Path,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)

    frame = pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required = {
        "image_id",
        "global_image_key",
        "canonical_object_index",
        "foreground_pixels",
        "local_mask_path",
        "category_name",
    }

    missing = required - set(frame.columns)

    if missing:
        raise RuntimeError(
            f"Manifest missing columns: "
            f"{sorted(missing)}"
        )

    frame["image_id"] = (
        frame["image_id"].astype(int)
    )
    frame["canonical_object_index"] = (
        frame[
            "canonical_object_index"
        ].astype(int)
    )

    duplicate_count = int(
        frame.duplicated(
            subset=[
                "image_id",
                "canonical_object_index",
            ],
            keep=False,
        ).sum()
    )

    if duplicate_count:
        raise RuntimeError(
            "Manifest contains duplicate "
            "image/object indexes: "
            f"{duplicate_count}"
        )

    return frame


def make_manifest_index(
    frame: pd.DataFrame,
) -> dict[tuple[int, int], dict[str, Any]]:
    result: dict[
        tuple[int, int],
        dict[str, Any],
    ] = {}

    for row in frame.to_dict(
        orient="records"
    ):
        key = (
            int(row["image_id"]),
            int(
                row[
                    "canonical_object_index"
                ]
            ),
        )

        result[key] = row

    return result


def make_pair_record(
    *,
    source_type: str,
    item: dict[str, Any],
    original_subject_index: int,
    original_object_index: int,
    orientation: dict[str, Any],
    manifest_index: dict[
        tuple[int, int],
        dict[str, Any],
    ],
    predicate_id: int | None,
    predicate_name: str,
    source_pair_key: str,
) -> dict[str, Any]:
    image_id = int(item["image_id"])
    width = int(item["width"])
    height = int(item["height"])
    image_area = int(width * height)

    target_index = int(
        orientation["target_index"]
    )
    hazard_index = int(
        orientation["hazard_index"]
    )

    segments = item["segments_info"]
    target_segment = segments[target_index]
    hazard_segment = segments[hazard_index]

    target_manifest = manifest_index.get(
        (image_id, target_index)
    )
    hazard_manifest = manifest_index.get(
        (image_id, hazard_index)
    )

    if target_manifest is None:
        raise RuntimeError(
            f"Missing target manifest row: "
            f"{image_id}/{target_index}"
        )

    if hazard_manifest is None:
        raise RuntimeError(
            f"Missing hazard manifest row: "
            f"{image_id}/{hazard_index}"
        )

    target_mask_path = (
        DATA_ROOT
        / str(
            target_manifest[
                "local_mask_path"
            ]
        )
    ).resolve()

    hazard_mask_path = (
        DATA_ROOT
        / str(
            hazard_manifest[
                "local_mask_path"
            ]
        )
    ).resolve()

    if not target_mask_path.is_file():
        raise FileNotFoundError(
            target_mask_path
        )

    if not hazard_mask_path.is_file():
        raise FileNotFoundError(
            hazard_mask_path
        )

    target_pixels = int(
        target_manifest[
            "foreground_pixels"
        ]
    )
    hazard_pixels = int(
        hazard_manifest[
            "foreground_pixels"
        ]
    )

    target_area_ratio = (
        target_pixels / image_area
        if image_area > 0
        else 0.0
    )

    hazard_area_ratio = (
        hazard_pixels / image_area
        if image_area > 0
        else 0.0
    )

    target_truncated = bbox_touches_border(
        target_segment.get(
            "bbox",
            [],
        ),
        width,
        height,
    )
    hazard_truncated = bbox_touches_border(
        hazard_segment.get(
            "bbox",
            [],
        ),
        width,
        height,
    )

    return {
        "source_type": source_type,
        "image_id": image_id,
        "global_image_key": normalize_text(
            item.get("global_image_key")
        ),
        "image_path": normalize_text(
            item.get("file_name")
        ),
        "image_width": width,
        "image_height": height,
        "original_subject_index": int(
            original_subject_index
        ),
        "original_object_index": int(
            original_object_index
        ),
        "target_index": target_index,
        "hazard_index": hazard_index,
        "target_category": (
            orientation[
                "target_category"
            ]
        ),
        "hazard_category": (
            orientation[
                "hazard_category"
            ]
        ),
        "pair_family": (
            orientation["pair_family"]
        ),
        "direction_swapped": int(
            orientation[
                "direction_swapped"
            ]
        ),
        "predicate_id": (
            ""
            if predicate_id is None
            else int(predicate_id)
        ),
        "predicate_name": predicate_name,
        "source_pair_key": (
            source_pair_key
        ),
        "canonical_pair_key": (
            f"{item['global_image_key']}"
            f"::{target_index}"
            f"::{hazard_index}"
        ),
        "target_object_key": normalize_text(
            target_segment.get(
                "canonical_object_key"
            )
        ),
        "hazard_object_key": normalize_text(
            hazard_segment.get(
                "canonical_object_key"
            )
        ),
        "target_mask_path": (
            target_manifest[
                "local_mask_path"
            ]
        ),
        "hazard_mask_path": (
            hazard_manifest[
                "local_mask_path"
            ]
        ),
        "target_foreground_pixels": (
            target_pixels
        ),
        "hazard_foreground_pixels": (
            hazard_pixels
        ),
        "target_area_ratio": float(
            target_area_ratio
        ),
        "hazard_area_ratio": float(
            hazard_area_ratio
        ),
        "target_scale_bin_fixed": (
            scale_bin_fixed(
                target_area_ratio
            )
        ),
        "hazard_scale_bin_fixed": (
            scale_bin_fixed(
                hazard_area_ratio
            )
        ),
        "target_truncated": int(
            target_truncated
        ),
        "hazard_truncated": int(
            hazard_truncated
        ),
        "pair_truncated": int(
            target_truncated
            or hazard_truncated
        ),
    }


def main() -> None:
    args = parse_args()

    annotation_path = (
        args.annotation.expanduser().resolve()
    )
    manifest_path = (
        args.manifest.expanduser().resolve()
    )
    hardneg_path = (
        args.train_hardneg.expanduser().resolve()
    )
    output_csv = (
        args.output_csv.expanduser().resolve()
    )
    summary_path = (
        args.summary.expanduser().resolve()
    )

    annotation = load_annotation(
        annotation_path
    )

    predicate_classes = list(
        annotation["predicate_classes"]
    )

    validation_ids = {
        int(value)
        for value in annotation.get(
            "test_image_ids",
            [],
        )
    }

    train_items = [
        item
        for item in annotation["data"]
        if int(item["image_id"])
        not in validation_ids
    ]

    train_by_global_key: dict[
        str,
        dict[str, Any],
    ] = {}

    for item in train_items:
        global_key = normalize_text(
            item.get("global_image_key")
        )

        if not global_key:
            raise RuntimeError(
                "Training item missing "
                "global_image_key"
            )

        if global_key in train_by_global_key:
            raise RuntimeError(
                "Duplicate training "
                f"global_image_key: {global_key}"
            )

        train_by_global_key[
            global_key
        ] = item

    manifest = load_manifest(
        manifest_path
    )

    manifest_train = manifest[
        manifest["image_id"].isin(
            {
                int(item["image_id"])
                for item in train_items
            }
        )
    ].copy()

    manifest_index = make_manifest_index(
        manifest_train
    )

    raw_records: list[
        dict[str, Any]
    ] = []

    invalid_positive_reasons: Counter[
        str
    ] = Counter()

    invalid_hn_reasons: Counter[
        str
    ] = Counter()

    positive_relation_rows = 0

    for item in train_items:
        segments = item.get(
            "segments_info",
            [],
        )

        for relation_number, relation in enumerate(
            item.get("relations", [])
        ):
            positive_relation_rows += 1

            if len(relation) != 3:
                invalid_positive_reasons[
                    "invalid_relation_shape"
                ] += 1
                continue

            subject_index = int(
                relation[0]
            )
            object_index = int(
                relation[1]
            )
            predicate_id = int(
                relation[2]
            )

            if not (
                0
                <= predicate_id
                < len(predicate_classes)
            ):
                invalid_positive_reasons[
                    "predicate_out_of_range"
                ] += 1
                continue

            orientation = orient_pair(
                segments,
                subject_index,
                object_index,
            )

            if not orientation["valid"]:
                invalid_positive_reasons[
                    orientation["reason"]
                ] += 1
                continue

            predicate_name = str(
                predicate_classes[
                    predicate_id
                ]
            )

            record = make_pair_record(
                source_type="positive",
                item=item,
                original_subject_index=(
                    subject_index
                ),
                original_object_index=(
                    object_index
                ),
                orientation=orientation,
                manifest_index=(
                    manifest_index
                ),
                predicate_id=predicate_id,
                predicate_name=(
                    predicate_name
                ),
                source_pair_key=(
                    f"positive::"
                    f"{item['image_id']}::"
                    f"{relation_number}"
                ),
            )

            raw_records.append(record)

    if not hardneg_path.is_file():
        raise FileNotFoundError(
            hardneg_path
        )

    hardneg = pd.read_csv(
        hardneg_path,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required_hn_columns = {
        "global_image_key",
        "subject_index_v4",
        "object_index_v4",
    }

    missing_hn_columns = (
        required_hn_columns
        - set(hardneg.columns)
    )

    if missing_hn_columns:
        raise RuntimeError(
            "Hard-negative table missing "
            f"columns: "
            f"{sorted(missing_hn_columns)}"
        )

    hardneg_rows = 0
    hardneg_outside_train = 0

    for row_number, row in hardneg.iterrows():
        hardneg_rows += 1

        global_key = normalize_text(
            row.get("global_image_key")
        )

        item = train_by_global_key.get(
            global_key
        )

        if item is None:
            hardneg_outside_train += 1
            continue

        try:
            subject_index = int(
                row["subject_index_v4"]
            )
            object_index = int(
                row["object_index_v4"]
            )
        except (
            TypeError,
            ValueError,
        ):
            invalid_hn_reasons[
                "invalid_pair_index"
            ] += 1
            continue

        orientation = orient_pair(
            item.get("segments_info", []),
            subject_index,
            object_index,
        )

        if not orientation["valid"]:
            invalid_hn_reasons[
                orientation["reason"]
            ] += 1
            continue

        source_pair_key = normalize_text(
            row.get(
                "canonical_pair_key"
            )
        )

        if not source_pair_key:
            source_pair_key = (
                f"hard_negative::"
                f"{global_key}::"
                f"{subject_index}::"
                f"{object_index}::"
                f"{row_number}"
            )

        record = make_pair_record(
            source_type="hard_negative",
            item=item,
            original_subject_index=(
                subject_index
            ),
            original_object_index=(
                object_index
            ),
            orientation=orientation,
            manifest_index=(
                manifest_index
            ),
            predicate_id=None,
            predicate_name="NONE",
            source_pair_key=(
                source_pair_key
            ),
        )

        raw_records.append(record)

    frame = pd.DataFrame(
        raw_records
    )

    if frame.empty:
        raise RuntimeError(
            "No valid FIBE pairs were built"
        )

    frame.sort_values(
        by=[
            "source_type",
            "image_id",
            "target_index",
            "hazard_index",
            "predicate_name",
        ],
        inplace=True,
        kind="stable",
    )

    duplicate_subset = [
        "source_type",
        "canonical_pair_key",
    ]

    duplicate_before = int(
        frame.duplicated(
            subset=duplicate_subset,
            keep=False,
        ).sum()
    )

    if duplicate_before:
        positive_duplicates = (
            frame[
                frame["source_type"]
                == "positive"
            ]
            .groupby(
                "canonical_pair_key",
                sort=False,
            )
            .agg(
                predicate_id=(
                    "predicate_id",
                    lambda values: "|".join(
                        sorted(
                            {
                                str(value)
                                for value in values
                                if normalize_text(
                                    value
                                )
                            }
                        )
                    ),
                ),
                predicate_name=(
                    "predicate_name",
                    lambda values: "|".join(
                        sorted(
                            set(
                                str(value)
                                for value in values
                            )
                        )
                    ),
                ),
            )
        )

        positive_frame = frame[
            frame["source_type"]
            == "positive"
        ].drop_duplicates(
            subset=[
                "canonical_pair_key"
            ],
            keep="first",
        )

        for index, row in (
            positive_frame.iterrows()
        ):
            key = row[
                "canonical_pair_key"
            ]

            if key in (
                positive_duplicates.index
            ):
                positive_frame.at[
                    index,
                    "predicate_id",
                ] = positive_duplicates.at[
                    key,
                    "predicate_id",
                ]
                positive_frame.at[
                    index,
                    "predicate_name",
                ] = positive_duplicates.at[
                    key,
                    "predicate_name",
                ]

        hard_negative_frame = frame[
            frame["source_type"]
            == "hard_negative"
        ].drop_duplicates(
            subset=[
                "canonical_pair_key"
            ],
            keep="first",
        )

        frame = pd.concat(
            [
                positive_frame,
                hard_negative_frame,
            ],
            ignore_index=True,
        )

    frame.sort_values(
        by=[
            "source_type",
            "pair_family",
            "image_id",
            "target_index",
            "hazard_index",
        ],
        inplace=True,
        kind="stable",
    )

    frame.reset_index(
        drop=True,
        inplace=True,
    )

    ratios = frame[
        "target_area_ratio"
    ].to_numpy(dtype=float)

    q33 = float(
        np.quantile(ratios, 1 / 3)
    )
    q67 = float(
        np.quantile(ratios, 2 / 3)
    )

    def scale_tertile(
        value: float,
    ) -> str:
        if value <= q33:
            return "small"
        if value <= q67:
            return "medium"
        return "large"

    frame[
        "target_scale_tertile"
    ] = frame[
        "target_area_ratio"
    ].map(scale_tertile)

    frame.insert(
        0,
        "pool_pair_id",
        [
            f"FIBE-POOL-{index:06d}"
            for index in range(
                1,
                len(frame) + 1,
            )
        ],
    )

    duplicate_after = int(
        frame.duplicated(
            subset=duplicate_subset,
            keep=False,
        ).sum()
    )

    if duplicate_after:
        raise RuntimeError(
            "Pair-pool deduplication failed: "
            f"{duplicate_after}"
        )

    validation_leakage = int(
        frame["image_id"]
        .isin(validation_ids)
        .sum()
    )

    if validation_leakage:
        raise RuntimeError(
            "Validation leakage detected: "
            f"{validation_leakage}"
        )

    missing_mask_paths = 0

    for column in (
        "target_mask_path",
        "hazard_mask_path",
    ):
        for value in frame[column]:
            path = (
                DATA_ROOT
                / str(value)
            ).resolve()

            if not path.is_file():
                missing_mask_paths += 1

    if missing_mask_paths:
        raise RuntimeError(
            "Pair pool references missing "
            f"masks: {missing_mask_paths}"
        )

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
    )

    source_counts = {
        str(key): int(value)
        for key, value in (
            frame["source_type"]
            .value_counts()
            .sort_index()
            .items()
        )
    }

    family_counts = (
        frame.groupby(
            [
                "source_type",
                "pair_family",
            ],
            observed=True,
        )
        .size()
        .reset_index(name="count")
    )

    family_count_records = (
        family_counts.to_dict(
            orient="records"
        )
    )

    scale_counts = (
        frame.groupby(
            [
                "source_type",
                "pair_family",
                "target_scale_tertile",
            ],
            observed=True,
        )
        .size()
        .reset_index(name="count")
        .to_dict(orient="records")
    )

    positive_predicate_counts = (
        frame[
            frame["source_type"]
            == "positive"
        ]["predicate_name"]
        .value_counts()
        .sort_index()
    )

    summary = {
        "version": (
            "FIBE_PAIR_POOL_AUDIT_V1"
        ),
        "annotation_path": str(
            annotation_path
        ),
        "manifest_path": str(
            manifest_path
        ),
        "hardneg_path": str(
            hardneg_path
        ),
        "output_csv": str(
            output_csv
        ),
        "train_images": len(
            train_items
        ),
        "validation_ids_excluded": len(
            validation_ids
        ),
        "manifest_train_objects": len(
            manifest_train
        ),
        "positive_relation_rows": (
            positive_relation_rows
        ),
        "hard_negative_input_rows": (
            hardneg_rows
        ),
        "hard_negative_outside_train": (
            hardneg_outside_train
        ),
        "valid_pair_pool_rows": len(
            frame
        ),
        "source_counts": source_counts,
        "duplicate_rows_before_dedup": (
            duplicate_before
        ),
        "duplicate_rows_after_dedup": (
            duplicate_after
        ),
        "validation_leakage": (
            validation_leakage
        ),
        "missing_mask_paths": (
            missing_mask_paths
        ),
        "target_area_ratio_q33": q33,
        "target_area_ratio_q67": q67,
        "invalid_positive_reasons": dict(
            sorted(
                invalid_positive_reasons.items()
            )
        ),
        "invalid_hard_negative_reasons": dict(
            sorted(
                invalid_hn_reasons.items()
            )
        ),
        "family_counts": (
            family_count_records
        ),
        "scale_counts": scale_counts,
        "positive_predicate_counts": {
            str(key): int(value)
            for key, value in (
                positive_predicate_counts.items()
            )
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
    print("FIBE PAIR POOL AUDIT V1")
    print("=" * 100)
    print(
        "train images:",
        len(train_items),
    )
    print(
        "validation IDs excluded:",
        len(validation_ids),
    )
    print(
        "manifest train objects:",
        len(manifest_train),
    )
    print(
        "positive relation rows:",
        positive_relation_rows,
    )
    print(
        "hard-negative input rows:",
        hardneg_rows,
    )
    print(
        "hard-negative outside train:",
        hardneg_outside_train,
    )
    print(
        "valid pair pool rows:",
        len(frame),
    )
    print(
        "duplicates before dedup:",
        duplicate_before,
    )
    print(
        "duplicates after dedup:",
        duplicate_after,
    )
    print(
        "validation leakage:",
        validation_leakage,
    )
    print(
        "missing mask paths:",
        missing_mask_paths,
    )
    print(
        "target area q33:",
        q33,
    )
    print(
        "target area q67:",
        q67,
    )

    print("\nSOURCE COUNTS")
    print(
        frame["source_type"]
        .value_counts()
        .to_string()
    )

    print("\nPAIR FAMILY COUNTS")
    print(
        family_counts.to_string(
            index=False
        )
    )

    print("\nPAIR FAMILY × SCALE")
    print(
        frame.groupby(
            [
                "source_type",
                "pair_family",
                "target_scale_tertile",
            ],
            observed=True,
        )
        .size()
        .unstack(
            fill_value=0
        )
        .to_string()
    )

    print("\nPOSITIVE PREDICATE COUNTS")
    print(
        positive_predicate_counts
        .to_string()
    )

    print("\nINVALID POSITIVE REASONS")
    print(
        dict(
            sorted(
                invalid_positive_reasons.items()
            )
        )
    )

    print("\nINVALID HARD-NEGATIVE REASONS")
    print(
        dict(
            sorted(
                invalid_hn_reasons.items()
            )
        )
    )

    print("\nOUTPUTS")
    print("pair pool:", output_csv)
    print("summary:", summary_path)

    print("\nFIBE PAIR POOL AUDIT: PASS")


if __name__ == "__main__":
    main()
