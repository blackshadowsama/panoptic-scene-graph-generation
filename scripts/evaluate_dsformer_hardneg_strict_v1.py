#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path.home() / (
    "projects/panoptic-scene-graph-generation"
)

DATA_ROOT = ROOT / "data/floodpsg"

RESULTS = ROOT / (
    "outputs/"
    "flood_groupstrict_coarse8_40epoch_v1_"
    "final_test_results.pkl"
)

ANNOTATION = DATA_ROOT / (
    "annotations/"
    "floodpsg_canonical_full_coarse8_"
    "groupstrict_v1.json"
)

HARDNEG = DATA_ROOT / (
    "stats/frozen_hardneg_eval_v1/"
    "06_finaltest_hardneg_water.csv"
)

OUT = DATA_ROOT / (
    "stats/dsformer_hardneg_eval_v1/"
    "existing_outputs_v1"
)


IOU_THRESHOLD = 0.5
SCORE_THRESHOLDS = (0.5, 0.7, 0.9)


def image_id(item: dict[str, Any]) -> str:
    for key in ("image_id", "img_id", "id"):
        if key in item:
            return str(item[key])

    raise KeyError(
        f"Image item has no ID field: {sorted(item.keys())}"
    )


def rgb2id(color: np.ndarray) -> np.ndarray:
    if color.dtype == np.uint8:
        color = color.astype(np.int32)

    if color.ndim == 2:
        return color.astype(np.int64)

    if color.ndim != 3 or color.shape[2] < 3:
        raise ValueError(
            f"Unexpected panoptic PNG shape: {color.shape}"
        )

    return (
        color[..., 0].astype(np.int64)
        + 256 * color[..., 1].astype(np.int64)
        + 256 * 256 * color[..., 2].astype(np.int64)
    )


def load_gt_index_mask(
    item: dict[str, Any],
) -> np.ndarray:
    relative = str(
        item["pan_seg_file_name"]
    ).replace("\\", "/")

    path = DATA_ROOT / relative

    if not path.is_file():
        raise FileNotFoundError(path)

    with Image.open(path) as image:
        panoptic_id = rgb2id(np.asarray(image))

    segments = item["segments_info"]

    output = np.full(
        panoptic_id.shape,
        fill_value=-1,
        dtype=np.int32,
    )

    # 与正式evaluate_v1_1.py一致：
    # segments_info顺序就是GT节点索引顺序。
    for index, segment in enumerate(segments):
        output[
            panoptic_id == int(segment["id"])
        ] = index

    expected_nodes = len(segments)

    if expected_nodes:
        actual_nodes = int(output.max()) + 1

        if actual_nodes != expected_nodes:
            raise RuntimeError(
                f"GT node-count mismatch for "
                f"image {image_id(item)}: "
                f"{actual_nodes} vs {expected_nodes}"
            )

    return output


def gt_category_labels(
    item: dict[str, Any],
) -> np.ndarray:
    # 正式评价器使用annotations中的category_id，
    # 而不是自行推测类别。
    annotations = item.get("annotations")

    if not isinstance(annotations, list):
        raise KeyError(
            f"Image {image_id(item)} has no annotations list."
        )

    if len(annotations) != len(item["segments_info"]):
        raise RuntimeError(
            f"annotations/segments_info mismatch for "
            f"image {image_id(item)}: "
            f"{len(annotations)} vs "
            f"{len(item['segments_info'])}"
        )

    return np.asarray(
        [
            int(annotation["category_id"])
            for annotation in annotations
        ],
        dtype=np.int64,
    )


def match_masks_fast(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
) -> tuple[dict[int, int], np.ndarray]:
    """
    快速计算正式evaluate_v1_1.py相同的匹配结果。

    返回：
        pred_index -> gt_index
        IoU矩阵，shape=(num_pred, num_gt)
    """
    gt_mask = np.asarray(gt_mask)
    pred_mask = np.asarray(pred_mask)
    gt_labels = np.asarray(gt_labels)
    pred_labels = np.asarray(pred_labels)

    if gt_mask.shape != pred_mask.shape:
        raise ValueError(
            f"GT/pred mask shape mismatch: "
            f"{gt_mask.shape} vs {pred_mask.shape}"
        )

    num_gt = len(gt_labels)
    num_pred = len(pred_labels)

    if num_gt == 0 or num_pred == 0:
        return {}, np.zeros(
            (num_pred, num_gt),
            dtype=np.float64,
        )

    if int(gt_mask.max()) + 1 != num_gt:
        raise RuntimeError(
            "GT mask index count does not match GT labels."
        )

    if int(pred_mask.max()) + 1 != num_pred:
        raise RuntimeError(
            "Prediction mask index count does not match "
            "prediction labels."
        )

    gt_valid = gt_mask >= 0
    pred_valid = pred_mask >= 0

    gt_areas = np.bincount(
        gt_mask[gt_valid].astype(np.int64),
        minlength=num_gt,
    )

    pred_areas = np.bincount(
        pred_mask[pred_valid].astype(np.int64),
        minlength=num_pred,
    )

    both = gt_valid & pred_valid

    pair_codes = (
        pred_mask[both].astype(np.int64) * num_gt
        + gt_mask[both].astype(np.int64)
    )

    intersections = np.bincount(
        pair_codes,
        minlength=num_pred * num_gt,
    ).reshape(num_pred, num_gt)

    unions = (
        pred_areas[:, None]
        + gt_areas[None, :]
        - intersections
    )

    ious = np.divide(
        intersections,
        unions,
        out=np.zeros_like(
            intersections,
            dtype=np.float64,
        ),
        where=unions > 0,
    )

    # 正式评价器默认不使用--ignore-lbl，
    # 因此预测类别与GT类别必须一致。
    label_mismatch = (
        pred_labels[:, None]
        != gt_labels[None, :]
    )

    ious[label_mismatch] = 0.0

    # 完全复现evaluate_v1_1.py：
    # 每个预测节点先指向IoU最大的GT节点；
    # 一个GT节点若对应多个预测节点，只保留IoU最高者。
    gt_assign: dict[int, list[int]] = defaultdict(list)

    for pred_index, row in enumerate(ious):
        if (row > IOU_THRESHOLD).any():
            gt_index = int(row.argmax())
            gt_assign[gt_index].append(pred_index)

    matching: dict[int, int] = {}

    for gt_index, pred_indices in gt_assign.items():
        pred_array = np.asarray(
            pred_indices,
            dtype=np.int64,
        )

        best_position = int(
            ious[pred_array, gt_index].argmax()
        )

        best_pred = int(
            pred_array[best_position]
        )

        matching[best_pred] = gt_index

    return matching, ious


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(result):
        return None

    return result


def grouped_summary(
    frame: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    rows = []

    for group_value, group in frame.groupby(
        column,
        dropna=False,
    ):
        scored = group[
            group["evaluation_status"]
            == "scored"
        ]

        row = {
            column: group_value,
            "total_hard_negative_pairs": len(group),
            "scored_pairs": len(scored),
            "score_coverage": (
                len(scored) / len(group)
                if len(group)
                else np.nan
            ),
            "missing_image_output_pairs": int(
                (
                    group["evaluation_status"]
                    == "missing_image_output"
                ).sum()
            ),
            "endpoint_unmatched_pairs": int(
                (
                    group["evaluation_status"]
                    == "endpoint_unmatched"
                ).sum()
            ),
            "pair_missing_pairs": int(
                (
                    group["evaluation_status"]
                    == "pair_missing_after_matching"
                ).sum()
            ),
        }

        if len(scored):
            row.update({
                "formal_any_predicate_fpr_at_0_5":
                    float(
                        scored[
                            "formal_any_positive_gt_0_5"
                        ].mean()
                    ),
                "none_failure_rate_at_0_5":
                    float(
                        scored[
                            "none_failure_at_0_5"
                        ].mean()
                    ),
                "argmax9_relation_rate":
                    float(
                        scored[
                            "argmax9_is_relation"
                        ].mean()
                    ),
                "foreground_proxy_rate_at_0_5":
                    float(
                        scored[
                            "foreground_proxy_gt_0_5"
                        ].mean()
                    ),
                "mean_score_none":
                    float(scored["score_NONE"].mean()),
                "mean_foreground_proxy":
                    float(
                        scored[
                            "foreground_proxy"
                        ].mean()
                    ),
                "mean_max_positive_score":
                    float(
                        scored[
                            "max_positive_score"
                        ].mean()
                    ),
            })
        else:
            row.update({
                "formal_any_predicate_fpr_at_0_5":
                    np.nan,
                "none_failure_rate_at_0_5":
                    np.nan,
                "argmax9_relation_rate":
                    np.nan,
                "foreground_proxy_rate_at_0_5":
                    np.nan,
                "mean_score_none": np.nan,
                "mean_foreground_proxy": np.nan,
                "mean_max_positive_score": np.nan,
            })

        rows.append(row)

    return pd.DataFrame(rows)



EXPECTED_WATER_COUNTS = {
    "train": {
        "pairs": 2198,
        "images": 527,
    },
    "validation": {
        "pairs": 322,
        "images": 75,
    },
    "final_test": {
        "pairs": 363,
        "images": 72,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one frozen FloodPSG water hard-negative split "
            "from a DSFormer result pickle. --help performs no data "
            "loading and creates no output."
        ),
    )

    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="DSFormer prediction pickle.",
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        required=True,
        help="Canonical Coarse8 annotation JSON.",
    )
    parser.add_argument(
        "--hardneg",
        type=Path,
        required=True,
        help="Frozen water hard-negative CSV for the selected split.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output directory; existing paths are refused.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="FloodPSG data root used to resolve panoptic PNG paths.",
    )
    parser.add_argument(
        "--split",
        choices=(
            "train",
            "validation",
            "final_test",
        ),
        required=True,
        help="Frozen split represented by --hardneg.",
    )
    parser.add_argument(
        "--expected-pairs",
        type=int,
        default=None,
        help=(
            "Expected frozen water HN pair count. "
            "Defaults to the protocol count for the selected split."
        ),
    )
    parser.add_argument(
        "--expected-images",
        type=int,
        default=None,
        help=(
            "Expected frozen water HN image count. "
            "Defaults to the protocol count for the selected split."
        ),
    )
    parser.add_argument(
        "--strict-complete",
        action="store_true",
        help=(
            "Require complete image and pair coverage, zero endpoint "
            "mismatches, zero missing pairs, zero duplicates, and "
            "finite predicate scores. Failure exits nonzero before "
            "any output directory is created."
        ),
    )
    parser.add_argument(
        "--allow-final-test",
        action="store_true",
        help=(
            "Explicitly permit final_test evaluation. "
            "Do not use this during D1 development/model selection."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    global ROOT
    global DATA_ROOT
    global RESULTS
    global ANNOTATION
    global HARDNEG
    global OUT

    ROOT = Path.cwd().resolve()
    DATA_ROOT = (
        args.data_root.expanduser().resolve()
    )
    RESULTS = (
        args.results.expanduser().resolve()
    )
    ANNOTATION = (
        args.annotation.expanduser().resolve()
    )
    HARDNEG = (
        args.hardneg.expanduser().resolve()
    )
    OUT = (
        args.output.expanduser().resolve()
    )

    if (
        args.split == "final_test"
        and not args.allow_final_test
    ):
        raise SystemExit(
            "ERROR: final_test is locked. "
            "Pass --allow-final-test only after all "
            "validation gates are frozen."
        )

    if OUT.exists():
        raise FileExistsError(
            "Refusing to overwrite output path: "
            f"{OUT}"
        )

    for path in (
        RESULTS,
        ANNOTATION,
        HARDNEG,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    if not DATA_ROOT.is_dir():
        raise NotADirectoryError(
            DATA_ROOT
        )

    expected_defaults = (
        EXPECTED_WATER_COUNTS[
            args.split
        ]
    )

    expected_pairs = (
        int(args.expected_pairs)
        if args.expected_pairs is not None
        else int(
            expected_defaults["pairs"]
        )
    )

    expected_images = (
        int(args.expected_images)
        if args.expected_images is not None
        else int(
            expected_defaults["images"]
        )
    )

    print("Loading annotation...")

    with ANNOTATION.open(
        "r",
        encoding="utf-8",
    ) as file:
        annotation = json.load(file)

    predicate_names = annotation[
        "predicate_classes"
    ]

    if len(predicate_names) != 8:
        raise RuntimeError(
            f"Expected 8 predicates, got "
            f"{len(predicate_names)}"
        )

    items = annotation["data"]

    by_id = {
        image_id(item): item
        for item in items
    }

    by_global_key = {
        str(item["global_image_key"]): item
        for item in items
        if item.get("global_image_key")
    }

    print("Loading DSFormer result pickle...")

    with RESULTS.open("rb") as file:
        outputs = pickle.load(file)

    output_by_id = {
        str(output["img_id"]): output
        for output in outputs
    }

    if len(output_by_id) != len(outputs):
        raise RuntimeError(
            "Duplicate prediction image IDs."
        )

    hardneg = pd.read_csv(
        HARDNEG,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required_hardneg_columns = {
        "global_image_key",
        "audit_split_v2",
        "subject_index_v4",
        "object_index_v4",
        "pair_family_v4",
    }

    missing_hardneg_columns = sorted(
        required_hardneg_columns
        - set(hardneg.columns)
    )

    if missing_hardneg_columns:
        raise KeyError(
            "Missing hard-negative columns: "
            f"{missing_hardneg_columns}"
        )

    hardneg_split_values = sorted(
        str(value)
        for value in hardneg[
            "audit_split_v2"
        ].dropna().unique()
    )

    if hardneg_split_values != [
        args.split
    ]:
        raise RuntimeError(
            "Hard-negative split mismatch: "
            f"expected={[args.split]}, "
            f"actual={hardneg_split_values}"
        )

    records: list[dict[str, Any]] = []
    image_matching_rows = []

    grouped = hardneg.groupby(
        "global_image_key",
        sort=True,
        dropna=False,
    )

    for group_index, (
        global_key,
        group,
    ) in enumerate(grouped, start=1):
        global_key = str(global_key)

        if global_key not in by_global_key:
            raise KeyError(
                f"Unknown global_image_key: {global_key}"
            )

        gt_item = by_global_key[global_key]
        iid = image_id(gt_item)

        print(
            f"[{group_index:03d}/{grouped.ngroups:03d}] "
            f"image_id={iid}, pairs={len(group)}"
        )

        if iid not in output_by_id:
            for _, source_row in group.iterrows():
                record = source_row.to_dict()
                record.update({
                    "canonical_image_id": iid,
                    "evaluation_status":
                        "missing_image_output",
                    "subject_gt_matched": False,
                    "object_gt_matched": False,
                })
                records.append(record)

            image_matching_rows.append({
                "canonical_image_id": iid,
                "global_image_key": global_key,
                "hard_negative_pairs": len(group),
                "has_prediction_output": False,
                "gt_nodes": len(
                    gt_item["segments_info"]
                ),
                "pred_nodes": 0,
                "matched_nodes": 0,
            })

            continue

        output = output_by_id[iid]

        gt_mask = load_gt_index_mask(gt_item)
        pred_mask = np.asarray(output["mask"])

        gt_labels = gt_category_labels(gt_item)
        pred_labels = np.asarray(
            output["box_label"],
            dtype=np.int64,
        )

        matching, ious = match_masks_fast(
            gt_mask,
            pred_mask,
            gt_labels,
            pred_labels,
        )

        # 正式匹配的方向是pred -> GT。
        gt_to_pred = {
            gt_index: pred_index
            for pred_index, gt_index
            in matching.items()
        }

        pairs = np.asarray(
            output["pairs"],
            dtype=np.int64,
        )

        scores = np.asarray(
            output["rel_scores"],
            dtype=np.float64,
        )

        rel_rank = np.asarray(
            output.get(
                "rel_rank",
                np.full(
                    len(pairs),
                    np.nan,
                    dtype=np.float64,
                ),
            ),
            dtype=np.float64,
        )

        if scores.shape != (
            len(pairs),
            len(predicate_names) + 1,
        ):
            raise RuntimeError(
                f"Unexpected score shape for image {iid}: "
                f"{scores.shape}"
            )

        pair_lookup = {}

        for pair_row_index, pair in enumerate(pairs):
            key = (
                int(pair[0]),
                int(pair[1]),
            )

            if key in pair_lookup:
                raise RuntimeError(
                    f"Duplicate predicted pair in image "
                    f"{iid}: {key}"
                )

            pair_lookup[key] = pair_row_index

        image_matching_rows.append({
            "canonical_image_id": iid,
            "global_image_key": global_key,
            "hard_negative_pairs": len(group),
            "has_prediction_output": True,
            "gt_nodes": len(gt_labels),
            "pred_nodes": len(pred_labels),
            "matched_nodes": len(matching),
            "gt_node_match_rate": (
                len(matching) / len(gt_labels)
                if len(gt_labels)
                else np.nan
            ),
        })

        for _, source_row in group.iterrows():
            record = source_row.to_dict()

            subject_gt = int(
                source_row["subject_index_v4"]
            )

            object_gt = int(
                source_row["object_index_v4"]
            )

            subject_pred = gt_to_pred.get(
                subject_gt
            )

            object_pred = gt_to_pred.get(
                object_gt
            )

            subject_matched = (
                subject_pred is not None
            )

            object_matched = (
                object_pred is not None
            )

            record.update({
                "canonical_image_id": iid,
                "subject_gt_index": subject_gt,
                "object_gt_index": object_gt,
                "subject_gt_matched":
                    subject_matched,
                "object_gt_matched":
                    object_matched,
                "subject_pred_index":
                    subject_pred,
                "object_pred_index":
                    object_pred,
            })

            if subject_matched:
                record["subject_match_iou"] = float(
                    ious[
                        int(subject_pred),
                        subject_gt,
                    ]
                )

            if object_matched:
                record["object_match_iou"] = float(
                    ious[
                        int(object_pred),
                        object_gt,
                    ]
                )

            if not (
                subject_matched
                and object_matched
            ):
                record[
                    "evaluation_status"
                ] = "endpoint_unmatched"

                records.append(record)
                continue

            predicted_pair = (
                int(subject_pred),
                int(object_pred),
            )

            pair_row_index = pair_lookup.get(
                predicted_pair
            )

            if pair_row_index is None:
                record[
                    "evaluation_status"
                ] = (
                    "pair_missing_after_matching"
                )

                records.append(record)
                continue

            score_row = scores[pair_row_index]

            score_none = float(score_row[0])
            positive_scores = score_row[1:]

            max_positive_index = int(
                positive_scores.argmax()
            )

            max_positive_score = float(
                positive_scores[
                    max_positive_index
                ]
            )

            foreground_proxy = (
                1.0 - score_none
            )

            positive_over_half = [
                predicate_names[index]
                for index, value
                in enumerate(positive_scores)
                if value > 0.5
            ]

            argmax9 = int(score_row.argmax())

            record.update({
                "evaluation_status": "scored",
                "pred_pair_row_index":
                    pair_row_index,
                "score_sum": float(
                    score_row.sum()
                ),
                "score_NONE": score_none,
                "foreground_proxy":
                    foreground_proxy,
                "max_positive_score":
                    max_positive_score,
                "top_positive_index":
                    max_positive_index,
                "top_positive_predicate":
                    predicate_names[
                        max_positive_index
                    ],
                "positive_predicates_gt_0_5":
                    "|".join(
                        positive_over_half
                    ),
                "num_positive_predicates_gt_0_5":
                    len(positive_over_half),
                # 与正式评价器逐列>0.5一致
                "formal_any_positive_gt_0_5":
                    bool(
                        len(positive_over_half)
                        > 0
                    ),
                # NONE列没有超过0.5
                "none_failure_at_0_5":
                    bool(
                        score_none <= 0.5
                    ),
                "argmax9_index": argmax9,
                "argmax9_is_relation":
                    bool(argmax9 != 0),
                "argmax9_label": (
                    "NONE"
                    if argmax9 == 0
                    else predicate_names[
                        argmax9 - 1
                    ]
                ),
                "rel_rank": float(
                    rel_rank[pair_row_index]
                ),
                "rel_rank_minus_foreground":
                    float(
                        rel_rank[pair_row_index]
                        - foreground_proxy
                    ),
            })

            for threshold in SCORE_THRESHOLDS:
                suffix = str(threshold).replace(
                    ".",
                    "_",
                )

                record[
                    f"formal_max_positive_gt_{suffix}"
                ] = bool(
                    max_positive_score
                    > threshold
                )

                record[
                    f"foreground_proxy_gt_{suffix}"
                ] = bool(
                    foreground_proxy
                    > threshold
                )

            for predicate_index, predicate in enumerate(
                predicate_names,
                start=1,
            ):
                record[
                    f"score_{predicate}"
                ] = float(
                    score_row[
                        predicate_index
                    ]
                )

            records.append(record)

        del gt_mask
        del pred_mask
        del ious
        gc.collect()

    result = pd.DataFrame(records)

    matching_df = pd.DataFrame(
        image_matching_rows
    )

    scored_for_gate = result[
        result["evaluation_status"]
        == "scored"
    ].copy()

    gate_status_counts = Counter(
        result["evaluation_status"]
    )

    frozen_pair_duplicates = int(
        hardneg.duplicated(
            subset=[
                "global_image_key",
                "subject_index_v4",
                "object_index_v4",
            ],
            keep=False,
        ).sum()
    )

    scored_pair_duplicates = int(
        scored_for_gate.duplicated(
            subset=[
                "global_image_key",
                "subject_index_v4",
                "object_index_v4",
            ],
            keep=False,
        ).sum()
    )

    score_columns = [
        "score_NONE",
        *[
            f"score_{predicate}"
            for predicate in predicate_names
        ],
    ]

    missing_score_columns = sorted(
        set(score_columns)
        - set(scored_for_gate.columns)
    )

    nonfinite_score_values = 0

    if not missing_score_columns:
        score_matrix = (
            scored_for_gate[
                score_columns
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        nonfinite_score_values = int(
            (
                ~np.isfinite(
                    score_matrix
                )
            ).sum()
        )

    frozen_pairs = int(
        len(hardneg)
    )

    frozen_images = int(
        hardneg[
            "global_image_key"
        ].nunique()
    )

    prediction_images = int(
        matching_df[
            "has_prediction_output"
        ].sum()
    )

    missing_prediction_images = int(
        (
            ~matching_df[
                "has_prediction_output"
            ]
        ).sum()
    )

    scored_pairs_for_gate = int(
        len(scored_for_gate)
    )

    unscored_pairs_for_gate = int(
        len(result)
        - scored_pairs_for_gate
    )

    endpoint_unmatched_pairs = int(
        gate_status_counts.get(
            "endpoint_unmatched",
            0,
        )
    )

    pair_missing_pairs = int(
        gate_status_counts.get(
            "pair_missing_after_matching",
            0,
        )
    )

    missing_image_output_pairs = int(
        gate_status_counts.get(
            "missing_image_output",
            0,
        )
    )

    coverage_checks = {
        "split_exact": (
            hardneg_split_values
            == [args.split]
        ),
        "frozen_pair_count_matches": (
            frozen_pairs
            == expected_pairs
        ),
        "frozen_image_count_matches": (
            frozen_images
            == expected_images
        ),
        "prediction_images_complete": (
            prediction_images
            == expected_images
        ),
        "missing_prediction_images_zero": (
            missing_prediction_images
            == 0
        ),
        "missing_image_output_pairs_zero": (
            missing_image_output_pairs
            == 0
        ),
        "scored_pairs_complete": (
            scored_pairs_for_gate
            == expected_pairs
        ),
        "unscored_pairs_zero": (
            unscored_pairs_for_gate
            == 0
        ),
        "endpoint_unmatched_pairs_zero": (
            endpoint_unmatched_pairs
            == 0
        ),
        "pair_missing_pairs_zero": (
            pair_missing_pairs
            == 0
        ),
        "frozen_pair_duplicates_zero": (
            frozen_pair_duplicates
            == 0
        ),
        "scored_pair_duplicates_zero": (
            scored_pair_duplicates
            == 0
        ),
        "score_columns_complete": (
            len(missing_score_columns)
            == 0
        ),
        "nonfinite_score_values_zero": (
            nonfinite_score_values
            == 0
        ),
    }

    strict_pass = all(
        coverage_checks.values()
    )

    formal_status = (
        "PASS"
        if (
            args.strict_complete
            and strict_pass
        )
        else (
            "FAIL"
            if args.strict_complete
            else "DIAGNOSTIC_ONLY"
        )
    )

    coverage_gate = {
        "strict_complete_requested": bool(
            args.strict_complete
        ),
        "split": args.split,
        "expected_images": expected_images,
        "expected_pairs": expected_pairs,
        "frozen_images": frozen_images,
        "frozen_pairs": frozen_pairs,
        "prediction_output_images": (
            prediction_images
        ),
        "missing_prediction_output_images": (
            missing_prediction_images
        ),
        "scored_pairs": (
            scored_pairs_for_gate
        ),
        "unscored_pairs": (
            unscored_pairs_for_gate
        ),
        "missing_image_output_pairs": (
            missing_image_output_pairs
        ),
        "endpoint_unmatched_pairs": (
            endpoint_unmatched_pairs
        ),
        "pair_missing_after_matching": (
            pair_missing_pairs
        ),
        "frozen_pair_duplicate_rows": (
            frozen_pair_duplicates
        ),
        "scored_pair_duplicate_rows": (
            scored_pair_duplicates
        ),
        "missing_score_columns": (
            missing_score_columns
        ),
        "nonfinite_score_values": (
            nonfinite_score_values
        ),
        "checks": coverage_checks,
        "status": formal_status,
    }

    print()
    print(
        "STRICT COVERAGE GATE"
        if args.strict_complete
        else "DIAGNOSTIC COVERAGE REPORT"
    )
    print(
        json.dumps(
            coverage_gate,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )

    if (
        args.strict_complete
        and not strict_pass
    ):
        print()
        print(
            "DSFORMER HARD-NEGATIVE "
            "STRICT-COMPLETE GATE: FAIL"
        )
        raise SystemExit(1)

    OUT.mkdir(
        parents=True,
        exist_ok=False,
    )

    result.to_csv(
        OUT / "01_pair_level_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    matching_df.to_csv(
        OUT / "00_image_node_matching.csv",
        index=False,
        encoding="utf-8-sig",
    )

    scored = result[
        result["evaluation_status"]
        == "scored"
    ].copy()

    unscored = result[
        result["evaluation_status"]
        != "scored"
    ].copy()

    unscored.to_csv(
        OUT / "06_unscored_pairs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    family_summary = grouped_summary(
        result,
        "pair_family_v4",
    )

    family_summary.to_csv(
        OUT / "02_pair_family_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    scene_column = (
        "scene_context_v4"
        if "scene_context_v4" in result.columns
        else "scene_type"
    )

    scene_summary = grouped_summary(
        result,
        scene_column,
    )

    scene_summary.to_csv(
        OUT / "03_scene_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    image_rows = []

    for (
        iid,
        global_key,
    ), group in result.groupby(
        [
            "canonical_image_id",
            "global_image_key",
        ],
        dropna=False,
    ):
        image_scored = group[
            group["evaluation_status"]
            == "scored"
        ]

        image_rows.append({
            "canonical_image_id": iid,
            "global_image_key": global_key,
            "total_hard_negative_pairs":
                len(group),
            "scored_pairs":
                len(image_scored),
            "score_coverage": (
                len(image_scored) / len(group)
                if len(group)
                else np.nan
            ),
            "formal_fp_pairs_at_0_5": (
                int(
                    image_scored[
                        "formal_any_positive_gt_0_5"
                    ].sum()
                )
                if len(image_scored)
                else 0
            ),
            "formal_pair_fpr_at_0_5": (
                float(
                    image_scored[
                        "formal_any_positive_gt_0_5"
                    ].mean()
                )
                if len(image_scored)
                else np.nan
            ),
            "image_has_formal_fp_at_0_5": (
                bool(
                    image_scored[
                        "formal_any_positive_gt_0_5"
                    ].any()
                )
                if len(image_scored)
                else pd.NA
            ),
            "none_failure_pairs_at_0_5": (
                int(
                    image_scored[
                        "none_failure_at_0_5"
                    ].sum()
                )
                if len(image_scored)
                else 0
            ),
            "mean_foreground_proxy": (
                float(
                    image_scored[
                        "foreground_proxy"
                    ].mean()
                )
                if len(image_scored)
                else np.nan
            ),
            "max_positive_score": (
                float(
                    image_scored[
                        "max_positive_score"
                    ].max()
                )
                if len(image_scored)
                else np.nan
            ),
        })

    image_summary = pd.DataFrame(
        image_rows
    )

    image_summary.to_csv(
        OUT / "04_image_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    formal_false_positives = scored[
        scored[
            "formal_any_positive_gt_0_5"
        ]
    ].copy()

    formal_false_positives = (
        formal_false_positives.sort_values(
            [
                "max_positive_score",
                "foreground_proxy",
            ],
            ascending=False,
        )
    )

    formal_false_positives.to_csv(
        OUT
        / "05_formal_false_positive_pairs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    relation_distribution = (
        formal_false_positives[
            "top_positive_predicate"
        ]
        .value_counts(dropna=False)
        .rename_axis("predicted_predicate")
        .reset_index(name="false_positive_pairs")
    )

    relation_distribution.to_csv(
        OUT
        / "07_false_positive_relation_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    scored_images = image_summary[
        image_summary["scored_pairs"] > 0
    ]

    status_counts = Counter(
        result["evaluation_status"]
    )

    max_rank_delta = (
        float(
            scored[
                "rel_rank_minus_foreground"
            ].abs().max()
        )
        if len(scored)
        else None
    )

    summary = {
        "protocol": (
            "DSFormer Coarse8 frozen hard-negative "
            "evaluation v1"
        ),
        "scope": (
            "Frozen split evaluation. Formal use requires "
            "--strict-complete and complete dedicated "
            "prediction coverage."
        ),
        "total_frozen_water_hard_negative_pairs":
            int(len(result)),
        "total_frozen_water_hard_negative_images":
            int(
                result[
                    "global_image_key"
                ].nunique()
            ),
        "prediction_output_images_in_subset":
            int(
                (
                    matching_df[
                        "has_prediction_output"
                    ]
                ).sum()
            ),
        "missing_prediction_output_images":
            int(
                (
                    ~matching_df[
                        "has_prediction_output"
                    ]
                ).sum()
            ),
        "status_counts": {
            str(key): int(value)
            for key, value
            in status_counts.items()
        },
        "scored_pair_coverage_of_full_subset": (
            safe_float(
                len(scored) / len(result)
            )
            if len(result)
            else None
        ),
        "scored_pairs": int(len(scored)),
        "unscored_pairs": int(len(unscored)),
        "score_semantics": {
            "primary_formal_false_positive": (
                "At least one positive-predicate "
                "score is strictly greater than 0.5."
            ),
            "none_failure": (
                "NONE score is less than or equal "
                "to 0.5."
            ),
            "foreground_proxy": (
                "1 - score_NONE; used as a "
                "confidence/ranking proxy, not "
                "as a normalized nine-class "
                "probability."
            ),
        },
        "pair_level_metrics_on_scored_pairs": {},
        "image_level_metrics_on_scored_images": {},
        "maximum_abs_rel_rank_minus_foreground_proxy":
            safe_float(max_rank_delta),
    }

    summary.update({
        "split": args.split,
        "strict_complete": bool(
            args.strict_complete
        ),
        "final_test_explicitly_allowed": bool(
            args.allow_final_test
        ),
        "results": str(RESULTS),
        "annotation": str(ANNOTATION),
        "hardneg": str(HARDNEG),
        "output": str(OUT),
        "coverage_gate": coverage_gate,
        "status": formal_status,
    })

    if len(scored):
        primary_fpr = float(
            scored[
                "formal_any_positive_gt_0_5"
            ].mean()
        )

        pair_metrics = {
            "formal_any_predicate_fpr_at_0_5":
                primary_fpr,
            "formal_specificity_at_0_5":
                1.0 - primary_fpr,
            "none_failure_rate_at_0_5":
                float(
                    scored[
                        "none_failure_at_0_5"
                    ].mean()
                ),
            "argmax9_relation_rate":
                float(
                    scored[
                        "argmax9_is_relation"
                    ].mean()
                ),
            "mean_score_NONE":
                float(
                    scored["score_NONE"].mean()
                ),
            "mean_foreground_proxy":
                float(
                    scored[
                        "foreground_proxy"
                    ].mean()
                ),
            "mean_max_positive_score":
                float(
                    scored[
                        "max_positive_score"
                    ].mean()
                ),
        }

        for threshold in SCORE_THRESHOLDS:
            suffix = str(threshold).replace(
                ".",
                "_",
            )

            pair_metrics[
                f"formal_max_positive_fpr_at_{threshold}"
            ] = float(
                scored[
                    f"formal_max_positive_gt_{suffix}"
                ].mean()
            )

            pair_metrics[
                f"foreground_proxy_rate_at_{threshold}"
            ] = float(
                scored[
                    f"foreground_proxy_gt_{suffix}"
                ].mean()
            )

        summary[
            "pair_level_metrics_on_scored_pairs"
        ] = pair_metrics

    if len(scored_images):
        valid_image_flags = (
            scored_images[
                "image_has_formal_fp_at_0_5"
            ].astype(bool)
        )

        summary[
            "image_level_metrics_on_scored_images"
        ] = {
            "scored_images":
                int(len(scored_images)),
            "images_with_at_least_one_formal_fp":
                int(valid_image_flags.sum()),
            "image_level_any_fp_rate_at_0_5":
                float(valid_image_flags.mean()),
            "macro_image_pair_fpr_at_0_5":
                float(
                    scored_images[
                        "formal_pair_fpr_at_0_5"
                    ].mean()
                ),
        }

    with (
        OUT / "summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        file.write("\n")

    print("\n" + "=" * 80)
    print("DSFORMER HARD-NEGATIVE RESULT")
    print("=" * 80)
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\nPair-family summary:")
    print(
        family_summary.to_string(
            index=False
        )
    )

    print("\nScene summary:")
    print(
        scene_summary.to_string(
            index=False
        )
    )

    print("\nFalse-positive relation distribution:")
    if len(relation_distribution):
        print(
            relation_distribution.to_string(
                index=False
            )
        )
    else:
        print("No formal false positives at threshold 0.5.")

    print("\nSaved to:")
    print(OUT.resolve())

    if args.strict_complete:
        print(
            "\nDSFORMER HARD-NEGATIVE "
            "STRICT-COMPLETE EVALUATION V1: PASS"
        )
    else:
        print(
            "\nDSFORMER HARD-NEGATIVE "
            "EVALUATION V1: DIAGNOSTIC_ONLY"
        )

    del outputs
    gc.collect()


if __name__ == "__main__":
    main()