#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

B_OUTPUT = ROOT / (
    "data/floodpsg/stats/"
    "dsformer_floodhn_sampling_v1/"
    "validation_inference_seed3407_v1/"
    "B_uniform_all_validation_all173.pkl"
)

C_OUTPUT = ROOT / (
    "data/floodpsg/stats/"
    "dsformer_floodhn_sampling_v1/"
    "validation_inference_seed3407_v1/"
    "C_floodhn_target50_validation_all173.pkl"
)

HN_ALL = ROOT / (
    "data/floodpsg/stats/"
    "frozen_hardneg_eval_v1/"
    "03_validation_hardneg_all.csv"
)

HN_WATER = ROOT / (
    "data/floodpsg/stats/"
    "frozen_hardneg_eval_v1/"
    "04_validation_hardneg_water.csv"
)

ANNOTATION = ROOT / (
    "data/floodpsg/annotations/"
    "floodpsg_canonical_trainval_coarse8_groupstrict_v1.json"
)

OUTPUT_DIR = ROOT / (
    "data/floodpsg/stats/"
    "dsformer_floodhn_sampling_v1/"
    "validation_hn_bc_seed3407_v1"
)


def normalize_id(value) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))

    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise ValueError(f"Invalid image ID: {value}")
        if float(value).is_integer():
            return str(int(value))

    text = str(value).strip()

    try:
        return str(int(text))
    except ValueError:
        return text



# FLOODPSG_HN_GLOBAL_IMAGE_KEY_RESOLUTION_V2
def normalize_global_key(value) -> str:
    if pd.isna(value):
        return ""

    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ""

        if float(value).is_integer():
            return str(int(value))

    return str(value).strip()


def build_global_key_to_image_id(
    annotation: dict,
) -> dict[str, str]:
    mapping: dict[str, str] = {}

    for item in annotation["data"]:
        global_key = normalize_global_key(
            item.get("global_image_key")
        )

        if not global_key:
            continue

        image_id = normalize_id(
            item["image_id"]
        )

        previous = mapping.get(global_key)

        if (
            previous is not None
            and previous != image_id
        ):
            raise RuntimeError(
                "global_image_key maps to multiple "
                f"image IDs: key={global_key!r}, "
                f"first={previous}, second={image_id}"
            )

        mapping[global_key] = image_id

    if not mapping:
        raise RuntimeError(
            "No global_image_key mappings were "
            "found in the annotation."
        )

    return mapping


def resolve_hn_image_id(
    row: pd.Series,
    *,
    global_key_to_image_id: dict[str, str],
) -> str:
    # FLOODPSG_HN_GLOBAL_KEY_FIRST_V3
    #
    # global_image_key is authoritative because the image_id
    # column in historical hard-negative CSV files may contain
    # stale pre-canonical IDs.
    global_key = normalize_global_key(
        row.get(
            "global_image_key",
            np.nan,
        )
    )

    if global_key:
        mapped_image_id = (
            global_key_to_image_id.get(
                global_key
            )
        )

        if mapped_image_id is None:
            raise RuntimeError(
                "Hard-negative global_image_key cannot "
                "be resolved in the canonical annotation: "
                f"{global_key!r}"
            )

        return mapped_image_id

    # Fallback only for rows that genuinely lack global_image_key.
    raw_image_id = row.get(
        "image_id",
        np.nan,
    )

    if not pd.isna(raw_image_id):
        return normalize_id(raw_image_id)

    raise RuntimeError(
        "Hard-negative row has neither a valid "
        "global_image_key nor image_id."
    )


def load_prediction_file(path: Path) -> dict:
    with path.open("rb") as file:
        outputs = pickle.load(file)

    result = {}

    for item in outputs:
        image_id = normalize_id(item["img_id"])

        if image_id in result:
            raise RuntimeError(
                f"Duplicate prediction image ID: {image_id}"
            )

        pairs = np.asarray(
            item["pairs"],
            dtype=np.int64,
        )

        scores = np.asarray(
            item["rel_scores"],
            dtype=np.float64,
        )

        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise RuntimeError(
                f"Image {image_id}: invalid pair shape "
                f"{pairs.shape}"
            )

        if scores.ndim != 2:
            raise RuntimeError(
                f"Image {image_id}: invalid score shape "
                f"{scores.shape}"
            )

        if len(pairs) != len(scores):
            raise RuntimeError(
                f"Image {image_id}: pair/score length mismatch"
            )

        pair_map = {}

        for index, pair in enumerate(pairs.tolist()):
            key = (
                int(pair[0]),
                int(pair[1]),
            )

            if key in pair_map:
                raise RuntimeError(
                    f"Image {image_id}: duplicate pair {key}"
                )

            pair_map[key] = index

        result[image_id] = {
            "pairs": pairs,
            "scores": scores,
            "pair_map": pair_map,
        }

    return result


def enrich(
    frame: pd.DataFrame,
    *,
    prefix: str,
    predictions: dict,
    predicate_names: list[str],
    global_key_to_image_id: dict[str, str],
) -> pd.DataFrame:
    none_scores = []
    max_positive_scores = []
    argmax_ids = []
    argmax_names = []
    fp05 = []
    fp07 = []
    fp09 = []
    none_fail05 = []
    resolved_image_ids = []

    missing = []

    for row_index, row in frame.iterrows():
        image_id = resolve_hn_image_id(
            row,
            global_key_to_image_id=(
                global_key_to_image_id
            ),
        )

        resolved_image_ids.append(image_id)

        subject = int(row["subject_index_v4"])
        obj = int(row["object_index_v4"])

        prediction = predictions.get(image_id)

        if prediction is None:
            missing.append(
                {
                    "row": row_index,
                    "image_id": image_id,
                    "subject": subject,
                    "object": obj,
                    "reason": "missing_image",
                }
            )
            continue

        pair_index = prediction["pair_map"].get(
            (subject, obj)
        )

        if pair_index is None:
            missing.append(
                {
                    "row": row_index,
                    "image_id": image_id,
                    "subject": subject,
                    "object": obj,
                    "reason": "missing_pair",
                }
            )
            continue

        scores = prediction["scores"][pair_index]

        if len(scores) != 1 + len(predicate_names):
            raise RuntimeError(
                f"Image {image_id}, pair {(subject, obj)}: "
                f"score columns={len(scores)}, expected "
                f"{1 + len(predicate_names)}"
            )

        none_score = float(scores[0])
        positive_scores = scores[1:]
        max_positive = float(
            positive_scores.max()
        )

        argmax_id = int(scores.argmax())

        if argmax_id == 0:
            argmax_name = "NONE"
        else:
            argmax_name = predicate_names[
                argmax_id - 1
            ]

        none_scores.append(none_score)
        max_positive_scores.append(max_positive)
        argmax_ids.append(argmax_id)
        argmax_names.append(argmax_name)
        fp05.append(max_positive > 0.5)
        fp07.append(max_positive > 0.7)
        fp09.append(max_positive > 0.9)
        none_fail05.append(none_score <= 0.5)

    if missing:
        missing_path = OUTPUT_DIR / (
            f"{prefix}_missing_pairs.csv"
        )

        pd.DataFrame(missing).to_csv(
            missing_path,
            index=False,
            encoding="utf-8-sig",
        )

        raise RuntimeError(
            f"{prefix}: {len(missing)} hard-negative "
            f"pairs were not scored. See {missing_path}"
        )

    result = frame.copy()

    result["resolved_image_id"] = (
        resolved_image_ids
    )

    result[f"{prefix}_score_none"] = none_scores
    result[f"{prefix}_max_positive_score"] = (
        max_positive_scores
    )
    result[f"{prefix}_argmax_id"] = argmax_ids
    result[f"{prefix}_argmax_predicate"] = (
        argmax_names
    )
    result[f"{prefix}_formal_fp_05"] = fp05
    result[f"{prefix}_formal_fp_07"] = fp07
    result[f"{prefix}_formal_fp_09"] = fp09
    result[f"{prefix}_none_failure_05"] = (
        none_fail05
    )

    return result


def summarize_model(
    frame: pd.DataFrame,
    prefix: str,
) -> dict:
    return {
        "pairs": int(len(frame)),
        "formal_fpr_at_0_5": float(
            frame[f"{prefix}_formal_fp_05"].mean()
        ),
        "formal_fpr_at_0_7": float(
            frame[f"{prefix}_formal_fp_07"].mean()
        ),
        "formal_fpr_at_0_9": float(
            frame[f"{prefix}_formal_fp_09"].mean()
        ),
        "none_failure_rate_at_0_5": float(
            frame[
                f"{prefix}_none_failure_05"
            ].mean()
        ),
        "argmax_relation_rate": float(
            (
                frame[f"{prefix}_argmax_id"]
                != 0
            ).mean()
        ),
        "mean_score_none": float(
            frame[f"{prefix}_score_none"].mean()
        ),
        "mean_max_positive_score": float(
            frame[
                f"{prefix}_max_positive_score"
            ].mean()
        ),
        "high_confidence_fp_count_at_0_9": int(
            frame[f"{prefix}_formal_fp_09"].sum()
        ),
    }


def exact_mcnemar_p(
    b_only_fp: int,
    c_only_fp: int,
) -> float:
    discordant = b_only_fp + c_only_fp

    if discordant == 0:
        return 1.0

    smaller = min(
        b_only_fp,
        c_only_fp,
    )

    probability = sum(
        math.comb(discordant, value)
        for value in range(smaller + 1)
    ) / (2 ** discordant)

    return float(min(1.0, 2.0 * probability))


def paired_summary(
    frame: pd.DataFrame,
) -> dict:
    b = frame["B_formal_fp_05"].to_numpy(
        dtype=bool
    )
    c = frame["C_formal_fp_05"].to_numpy(
        dtype=bool
    )

    b_only = int(
        np.logical_and(
            b,
            np.logical_not(c),
        ).sum()
    )

    c_only = int(
        np.logical_and(
            c,
            np.logical_not(b),
        ).sum()
    )

    unique_images = np.asarray(
        sorted(
            frame["resolved_image_id"]
            .map(normalize_id)
            .unique()
        )
    )

    frame_with_id = frame.copy()
    frame_with_id["_normalized_image_id"] = (
        frame_with_id["resolved_image_id"]
        .map(normalize_id)
    )

    rng = np.random.default_rng(3407)
    bootstrap_delta = []

    for _ in range(10000):
        sampled_images = rng.choice(
            unique_images,
            size=len(unique_images),
            replace=True,
        )

        b_count = 0
        c_count = 0
        pair_count = 0

        for image_id in sampled_images:
            subset = frame_with_id[
                frame_with_id[
                    "_normalized_image_id"
                ] == image_id
            ]

            pair_count += len(subset)
            b_count += int(
                subset["B_formal_fp_05"].sum()
            )
            c_count += int(
                subset["C_formal_fp_05"].sum()
            )

        bootstrap_delta.append(
            (
                c_count / pair_count
                - b_count / pair_count
            )
        )

    lower, upper = np.quantile(
        bootstrap_delta,
        [0.025, 0.975],
    )

    return {
        "delta_fpr_C_minus_B": float(
            c.mean() - b.mean()
        ),
        "B_false_positive_C_true_negative":
            b_only,
        "C_false_positive_B_true_negative":
            c_only,
        "mcnemar_exact_p": exact_mcnemar_p(
            b_only,
            c_only,
        ),
        "cluster_bootstrap_95ci_C_minus_B": [
            float(lower),
            float(upper),
        ],
    }


def grouped_summary(
    frame: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    rows = []

    for value, subset in frame.groupby(
        column,
        dropna=False,
    ):
        row = {
            column: value,
            "pairs": len(subset),
        }

        for prefix in ("B", "C"):
            metrics = summarize_model(
                subset,
                prefix,
            )

            for name, metric_value in metrics.items():
                if name == "pairs":
                    continue

                row[
                    f"{prefix}_{name}"
                ] = metric_value

        row["C_minus_B_fpr_at_0_5"] = (
            row["C_formal_fpr_at_0_5"]
            - row["B_formal_fpr_at_0_5"]
        )

        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_subset(
    csv_path: Path,
    subset_name: str,
    *,
    b_predictions: dict,
    c_predictions: dict,
    predicate_names: list[str],
    global_key_to_image_id: dict[str, str],
) -> dict:
    frame = pd.read_csv(
        csv_path,
        encoding="utf-8-sig",
    )

    if not (
        frame["audit_split_v2"]
        .astype(str)
        .eq("validation")
        .all()
    ):
        raise RuntimeError(
            f"{csv_path} contains non-validation rows."
        )

    frame = enrich(
        frame,
        prefix="B",
        predictions=b_predictions,
        predicate_names=predicate_names,
        global_key_to_image_id=(
            global_key_to_image_id
        ),
    )

    frame = enrich(
        frame,
        prefix="C",
        predictions=c_predictions,
        predicate_names=predicate_names,
        global_key_to_image_id=(
            global_key_to_image_id
        ),
    )

    frame.to_csv(
        OUTPUT_DIR / f"{subset_name}_pair_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )

    family_summary = grouped_summary(
        frame,
        "pair_family_v4",
    )

    family_summary.to_csv(
        OUTPUT_DIR / f"{subset_name}_family_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    scene_summary = grouped_summary(
        frame,
        "scene_context_v4",
    )

    scene_summary.to_csv(
        OUTPUT_DIR / f"{subset_name}_scene_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return {
        "rows": int(len(frame)),
        "B": summarize_model(frame, "B"),
        "C": summarize_model(frame, "C"),
        "paired_comparison": paired_summary(frame),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    annotation = json.loads(
        ANNOTATION.read_text(encoding="utf-8")
    )

    predicate_names = annotation[
        "predicate_classes"
    ]

    global_key_to_image_id = (
        build_global_key_to_image_id(
            annotation
        )
    )

    print(
        "Annotation global-key mappings:",
        len(global_key_to_image_id),
    )

    if len(predicate_names) != 8:
        raise RuntimeError(
            f"Expected 8 predicates, "
            f"found {len(predicate_names)}"
        )

    print("Loading B predictions...")
    b_predictions = load_prediction_file(
        B_OUTPUT
    )

    print("Loading C predictions...")
    c_predictions = load_prediction_file(
        C_OUTPUT
    )

    # FLOODPSG_PAIR_CAPABLE_VALIDATION_V1
    validation_ids = {
        normalize_id(value)
        for value in annotation["test_image_ids"]
    }

    validation_items = {
        normalize_id(item["image_id"]): item
        for item in annotation["data"]
        if normalize_id(item["image_id"])
        in validation_ids
    }

    if set(validation_items) != validation_ids:
        missing_items = sorted(
            validation_ids - set(validation_items)
        )
        raise RuntimeError(
            "Validation annotation items are missing: "
            f"{missing_items}"
        )

    pair_capable_ids = {
        image_id
        for image_id, item
        in validation_items.items()
        if len(item.get("annotations", [])) >= 2
    }

    pairless_ids = (
        validation_ids - pair_capable_ids
    )

    for model_name, predictions in (
        ("B", b_predictions),
        ("C", c_predictions),
    ):
        output_ids = set(predictions)

        missing = sorted(
            pair_capable_ids - output_ids
        )
        extra = sorted(
            output_ids - pair_capable_ids
        )

        if missing or extra:
            raise RuntimeError(
                f"{model_name} pair-capable coverage "
                f"failed: output_images={len(output_ids)}, "
                f"expected={len(pair_capable_ids)}, "
                f"missing={missing}, extra={extra}"
            )

    print(
        "Validation images:",
        len(validation_ids),
    )
    print(
        "Pair-capable validation images:",
        len(pair_capable_ids),
    )
    print(
        "Pairless validation images:",
        len(pairless_ids),
    )
    print(
        "Pairless validation IDs:",
        sorted(pairless_ids),
    )

    summary = {
        "protocol": (
            "FloodPSG validation hard-negative "
            "B/C comparison v1"
        ),
        "B_model": str(B_OUTPUT.relative_to(ROOT)),
        "C_model": str(C_OUTPUT.relative_to(ROOT)),
        "threshold_semantics": (
            "False positive when at least one positive "
            "predicate sigmoid score is strictly above "
            "the specified threshold."
        ),
        "validation_hardneg_all": evaluate_subset(
            HN_ALL,
            "validation_hardneg_all",
            b_predictions=b_predictions,
            c_predictions=c_predictions,
            predicate_names=predicate_names,
            global_key_to_image_id=(
                global_key_to_image_id
            ),
        ),
        "validation_hardneg_water": evaluate_subset(
            HN_WATER,
            "validation_hardneg_water",
            b_predictions=b_predictions,
            c_predictions=c_predictions,
            predicate_names=predicate_names,
            global_key_to_image_id=(
                global_key_to_image_id
            ),
        ),
    }

    summary_path = OUTPUT_DIR / "summary.json"

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("VALIDATION HARD-NEGATIVE B/C RESULT")
    print("=" * 80)
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )
    print("\nSaved:", summary_path)
    print(
        "DSFORMER VALIDATION HN B/C "
        "EVALUATION V1: PASS"
    )


if __name__ == "__main__":
    main()
