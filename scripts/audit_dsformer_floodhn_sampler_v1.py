#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

# FLOODPSG_PROJECT_ROOT_V1
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from fair_psgg.data.data import (
    make_multilabel_target,
)
from fair_psgg.data.flood_hn_sampling import (
    load_flood_neg_sampling_config,
    sample_flood_negatives,
    sampling_budget,
)
from fair_psgg.data.load_entries import (
    load_psg_entries,
)


ROOT = Path.home() / "projects/panoptic-scene-graph-generation"

ANNOTATION = ROOT / (
    "data/floodpsg/annotations/"
    "floodpsg_canonical_trainval_coarse8_groupstrict_v1.json"
)

INDEX = ROOT / (
    "data/floodpsg/stats/"
    "dsformer_floodhn_sampling_v1/"
    "train_hardneg_index_v1.json"
)

OUTPUT = ROOT / (
    "data/floodpsg/stats/"
    "dsformer_floodhn_sampling_v1/"
    "sampler_audit_v1.json"
)


def set_environment(
    mode: str,
    fraction: float,
) -> None:
    os.environ["FLOODPSG_KEEP_ZERO_REL"] = "1"
    os.environ["FLOODPSG_NEG_MODE"] = mode
    os.environ[
        "FLOODPSG_ZERO_NEG_PER_IMAGE"
    ] = "2"
    os.environ[
        "FLOODPSG_HARDNEG_FRACTION"
    ] = str(fraction)
    os.environ[
        "FLOODPSG_HARDNEG_INDEX"
    ] = str(INDEX)


def run_mode(
    mode: str,
    fraction: float,
) -> dict:
    set_environment(mode, fraction)

    entries, _, predicates = load_psg_entries(
        anno_path=ANNOTATION,
        split="train",
    )

    config = load_flood_neg_sampling_config(
        is_train=True
    )

    budget = sampling_budget(
        entries=entries,
        neg_ratio=1.0,
        config=config,
    )

    torch.manual_seed(3407)

    sampled_positive = 0
    sampled_negative = 0
    sampled_hard = 0

    hard_mapping = (
        config.hardneg_by_image_id
    )

    for entry in entries:
        relations = make_multilabel_target(
            entry.get("relations", []),
            num_classes=len(predicates),
        )

        boxes = torch.zeros(
            (
                len(entry["annotations"]),
                4,
            ),
            dtype=torch.float32,
        )

        sampled = sample_flood_negatives(
            boxes=boxes,
            rel_targets=relations,
            neg_ratio=1.0,
            image_id=entry["image_id"],
            config=config,
        )

        num_positive = len(relations)
        negatives = sampled[num_positive:, :2]

        sampled_positive += num_positive
        sampled_negative += len(negatives)

        hard_set = set(
            hard_mapping.get(
                str(entry["image_id"]),
                (),
            )
        )

        sampled_hard += sum(
            (int(pair[0]), int(pair[1]))
            in hard_set
            for pair in negatives.tolist()
        )

    return {
        "mode": mode,
        "hardneg_fraction": fraction,
        "loaded_training_images": len(entries),
        "analytic_budget": budget,
        "sampled_positive_pairs": sampled_positive,
        "sampled_negative_pairs": sampled_negative,
        "sampled_hard_negative_pairs":
            sampled_hard,
    }


def main() -> None:
    if not ANNOTATION.is_file():
        raise FileNotFoundError(ANNOTATION)

    if not INDEX.is_file():
        raise FileNotFoundError(INDEX)

    uniform = run_mode(
        "uniform_all",
        0.0,
    )

    flood = run_mode(
        "flood_hn",
        0.25,
    )

    errors = []

    for result in (uniform, flood):
        if result[
            "loaded_training_images"
        ] != 1500:
            errors.append(
                f"{result['mode']} did not load "
                f"1500 training images"
            )

        if result[
            "sampled_positive_pairs"
        ] != 3232:
            errors.append(
                f"{result['mode']} positive pairs "
                f"!= 3232"
            )

        if result[
            "analytic_budget"
        ]["zero_relation_images"] != 764:
            errors.append(
                f"{result['mode']} zero-relation "
                f"images != 764"
            )

    if (
        uniform["sampled_negative_pairs"]
        != flood["sampled_negative_pairs"]
    ):
        errors.append(
            "Uniform-All and FloodHN have different "
            "negative-pair budgets."
        )

    if flood[
        "sampled_hard_negative_pairs"
    ] <= 0:
        errors.append(
            "FloodHN sampled no hard negatives."
        )

    if uniform[
        "sampled_hard_negative_pairs"
    ] != 0:
        # Uniform mode deliberately has no hard-negative
        # prioritization. Coincidental overlaps are not
        # counted because no hard index is loaded.
        errors.append(
            "Uniform mode unexpectedly reports "
            "prioritized hard negatives."
        )

    report = {
        "protocol": (
            "FloodPSG DSFormer Uniform-All/FloodHN "
            "sampler audit v1"
        ),
        "annotation": str(
            ANNOTATION.relative_to(ROOT)
        ),
        "hard_negative_index": str(
            INDEX.relative_to(ROOT)
        ),
        "uniform_all": uniform,
        "flood_hn_r25": flood,
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")

    print("=" * 80)
    print("DSFORMER FLOODHN SAMPLER AUDIT")
    print("=" * 80)
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )
    print("\nSaved:", OUTPUT)

    if errors:
        raise SystemExit(
            "\nDSFORMER FLOODHN SAMPLER "
            "AUDIT V1: FAIL"
        )

    print(
        "\nDSFORMER FLOODHN SAMPLER "
        "AUDIT V1: PASS"
    )


if __name__ == "__main__":
    main()
