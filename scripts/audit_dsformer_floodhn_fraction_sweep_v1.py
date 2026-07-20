#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from fair_psgg.data.data import make_multilabel_target
from fair_psgg.data.flood_hn_sampling import (
    load_flood_neg_sampling_config,
    sample_flood_negatives,
)
from fair_psgg.data.load_entries import load_psg_entries


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
    "fraction_sweep_v1.json"
)

FRACTIONS = [
    0.10,
    0.25,
    0.50,
    0.75,
    1.00,
]

SEEDS = [
    3407,
    3408,
    3409,
    3410,
    3411,
]


def configure(fraction: float) -> None:
    os.environ["FLOODPSG_KEEP_ZERO_REL"] = "1"
    os.environ["FLOODPSG_NEG_MODE"] = "flood_hn"
    os.environ["FLOODPSG_ZERO_NEG_PER_IMAGE"] = "2"
    os.environ["FLOODPSG_HARDNEG_FRACTION"] = str(fraction)
    os.environ["FLOODPSG_HARDNEG_INDEX"] = str(INDEX)


def run_once(
    entries: list[dict],
    predicates: list[str],
    *,
    fraction: float,
    seed: int,
) -> dict:
    configure(fraction)

    config = load_flood_neg_sampling_config(
        is_train=True,
    )

    torch.manual_seed(seed)

    positive_total = 0
    negative_total = 0
    hard_total = 0

    unique_hard_pairs: set[
        tuple[str, int, int]
    ] = set()

    hard_from_zero_relation_images = 0

    for entry in entries:
        targets = make_multilabel_target(
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
            rel_targets=targets,
            neg_ratio=1.0,
            image_id=entry["image_id"],
            config=config,
        )

        num_positive = len(targets)

        negative_pairs = sampled[
            num_positive:,
            :2,
        ]

        hard_set = set(
            config.hardneg_by_image_id.get(
                str(entry["image_id"]),
                (),
            )
        )

        image_hard_count = 0

        for subject, obj in negative_pairs.tolist():
            pair = (
                int(subject),
                int(obj),
            )

            if pair in hard_set:
                hard_total += 1
                image_hard_count += 1

                unique_hard_pairs.add(
                    (
                        str(entry["image_id"]),
                        pair[0],
                        pair[1],
                    )
                )

        if num_positive == 0:
            hard_from_zero_relation_images += (
                image_hard_count
            )

        positive_total += num_positive
        negative_total += len(negative_pairs)

    return {
        "target_fraction": fraction,
        "seed": seed,
        "positive_pairs": positive_total,
        "negative_pairs": negative_total,
        "hard_negative_instances": hard_total,
        "realized_hard_fraction": (
            hard_total / negative_total
            if negative_total
            else 0.0
        ),
        "unique_hard_negative_pairs":
            len(unique_hard_pairs),
        "unique_hard_negative_coverage": (
            len(unique_hard_pairs) / 2278
        ),
        "hard_negative_instances_from_zero_relation_images":
            hard_from_zero_relation_images,
    }


def main() -> None:
    for path in (
        ANNOTATION,
        INDEX,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    configure(0.25)

    entries, _, predicates = load_psg_entries(
        ANNOTATION,
        "train",
    )

    if len(entries) != 1500:
        raise RuntimeError(
            f"Expected 1500 training images, "
            f"found {len(entries)}"
        )

    runs = []

    for fraction in FRACTIONS:
        for seed in SEEDS:
            result = run_once(
                entries,
                predicates,
                fraction=fraction,
                seed=seed,
            )

            runs.append(result)

            print(
                f"target={fraction:.2f} "
                f"seed={seed} "
                f"hard={result['hard_negative_instances']} "
                f"realized="
                f"{result['realized_hard_fraction']:.4f} "
                f"unique="
                f"{result['unique_hard_negative_pairs']}"
            )

    summary = {}

    for fraction in FRACTIONS:
        selected = [
            row
            for row in runs
            if row["target_fraction"] == fraction
        ]

        realized = [
            row["realized_hard_fraction"]
            for row in selected
        ]

        hard_counts = [
            row["hard_negative_instances"]
            for row in selected
        ]

        unique_counts = [
            row["unique_hard_negative_pairs"]
            for row in selected
        ]

        summary[str(fraction)] = {
            "runs": len(selected),
            "mean_hard_negative_instances":
                statistics.mean(hard_counts),
            "min_hard_negative_instances":
                min(hard_counts),
            "max_hard_negative_instances":
                max(hard_counts),
            "mean_realized_hard_fraction":
                statistics.mean(realized),
            "min_realized_hard_fraction":
                min(realized),
            "max_realized_hard_fraction":
                max(realized),
            "mean_unique_hard_negative_pairs":
                statistics.mean(unique_counts),
            "mean_unique_hard_negative_coverage":
                statistics.mean(unique_counts)
                / 2278,
        }

    payload = {
        "protocol": (
            "FloodPSG DSFormer hard-negative "
            "fraction sweep v1"
        ),
        "training_images": len(entries),
        "positive_pairs": 3232,
        "negative_budget": 4642,
        "hard_negative_pool": 2278,
        "fractions": FRACTIONS,
        "seeds": SEEDS,
        "runs": runs,
        "summary": summary,
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
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")

    print("\n" + "=" * 80)
    print("FRACTION SWEEP SUMMARY")
    print("=" * 80)

    for fraction, values in summary.items():
        print(
            f"target={float(fraction):.2f}  "
            f"realized_mean="
            f"{values['mean_realized_hard_fraction']:.4f}  "
            f"hard_mean="
            f"{values['mean_hard_negative_instances']:.1f}  "
            f"unique_mean="
            f"{values['mean_unique_hard_negative_pairs']:.1f}"
        )

    print("\nSaved:", OUTPUT)
    print(
        "DSFORMER FLOODHN FRACTION "
        "SWEEP V1: PASS"
    )


if __name__ == "__main__":
    main()
