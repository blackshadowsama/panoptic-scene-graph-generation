#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path.home() / "projects/panoptic-scene-graph-generation"

ANNOTATION = ROOT / (
    "data/floodpsg/annotations/"
    "floodpsg_canonical_trainval_coarse8_groupstrict_v1.json"
)

HARDNEG = ROOT / (
    "data/floodpsg/stats/frozen_hardneg_eval_v1/"
    "01_train_hardneg_all.csv"
)

OUTPUT = ROOT / (
    "data/floodpsg/stats/"
    "dsformer_floodhn_sampling_v1/"
    "train_hardneg_index_v1.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def image_id(item: dict[str, Any]) -> str:
    value = item.get(
        "image_id",
        item.get("img_id", item.get("id")),
    )

    if value is None:
        raise KeyError("Missing image_id/img_id/id")

    return str(value)


def main() -> None:
    for path in (ANNOTATION, HARDNEG):
        if not path.is_file():
            raise FileNotFoundError(path)

    with ANNOTATION.open(
        "r",
        encoding="utf-8",
    ) as file:
        document = json.load(file)

    validation_ids = {
        str(value)
        for value in document["test_image_ids"]
    }

    train_items = [
        item
        for item in document["data"]
        if image_id(item) not in validation_ids
    ]

    if len(train_items) != 1500:
        raise RuntimeError(
            f"Expected 1500 train images, "
            f"found {len(train_items)}"
        )

    by_global_key = {}

    for item in train_items:
        key = str(item.get("global_image_key", ""))

        if not key:
            raise RuntimeError(
                f"global_image_key missing in image "
                f"{image_id(item)}"
            )

        if key in by_global_key:
            raise RuntimeError(
                f"Duplicate global_image_key: {key}"
            )

        by_global_key[key] = item

    frame = pd.read_csv(
        HARDNEG,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required = {
        "global_image_key",
        "subject_index_v4",
        "object_index_v4",
    }

    missing = required - set(frame.columns)

    if missing:
        raise KeyError(
            f"Missing hard-negative columns: "
            f"{sorted(missing)}"
        )

    pairs_by_image: dict[
        str,
        set[tuple[int, int]],
    ] = defaultdict(set)

    family_counts: Counter[str] = Counter()
    scene_counts: Counter[str] = Counter()

    zero_relation_pairs = 0

    for _, row in frame.iterrows():
        global_key = str(row["global_image_key"])

        item = by_global_key.get(global_key)

        if item is None:
            raise RuntimeError(
                f"Hard negative outside training split: "
                f"{global_key}"
            )

        iid = image_id(item)

        subject = int(row["subject_index_v4"])
        obj = int(row["object_index_v4"])

        num_nodes = len(item["segments_info"])

        if not (
            0 <= subject < num_nodes
            and 0 <= obj < num_nodes
        ):
            raise RuntimeError(
                f"Invalid endpoint: image={iid}, "
                f"pair=({subject}, {obj}), "
                f"nodes={num_nodes}"
            )

        if subject == obj:
            raise RuntimeError(
                f"Self-pair: image={iid}, "
                f"pair=({subject}, {obj})"
            )

        positives = {
            (int(rel[0]), int(rel[1]))
            for rel in item.get("relations", [])
        }

        if (subject, obj) in positives:
            raise RuntimeError(
                f"Hard negative collides with positive: "
                f"image={iid}, pair=({subject}, {obj})"
            )

        pairs_by_image[iid].add(
            (subject, obj)
        )

        family_counts[
            str(row.get("pair_family_v4", "unknown"))
        ] += 1

        scene_counts[
            str(row.get("scene_context_v4", "unknown"))
        ] += 1

        if not item.get("relations"):
            zero_relation_pairs += 1

    total_pairs = sum(
        len(pairs)
        for pairs in pairs_by_image.values()
    )

    if total_pairs != 2278:
        raise RuntimeError(
            f"Expected 2278 unique train hard negatives, "
            f"found {total_pairs}"
        )

    payload = {
        "protocol": (
            "FloodPSG DSFormer train-only "
            "hard-negative index v1"
        ),
        "annotation": str(
            ANNOTATION.relative_to(ROOT)
        ),
        "hard_negative_source": str(
            HARDNEG.relative_to(ROOT)
        ),
        "annotation_sha256": sha256(ANNOTATION),
        "hard_negative_source_sha256":
            sha256(HARDNEG),
        "train_images": len(train_items),
        "hard_negative_images":
            len(pairs_by_image),
        "hard_negative_pairs": total_pairs,
        "hard_negative_pairs_in_zero_relation_images":
            zero_relation_pairs,
        "family_counts": dict(
            sorted(family_counts.items())
        ),
        "scene_counts": dict(
            sorted(scene_counts.items())
        ),
        "pairs_by_image_id": {
            iid: [
                [subject, obj]
                for subject, obj in sorted(pairs)
            ]
            for iid, pairs in sorted(
                pairs_by_image.items(),
                key=lambda item: int(item[0]),
            )
        },
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

    print("=" * 80)
    print("DSFORMER TRAIN HARD-NEGATIVE INDEX")
    print("=" * 80)
    print("train images:", len(train_items))
    print(
        "hard-negative images:",
        len(pairs_by_image),
    )
    print(
        "hard-negative pairs:",
        total_pairs,
    )
    print(
        "hard negatives in zero-relation images:",
        zero_relation_pairs,
    )
    print("family counts:")
    for name, count in sorted(
        family_counts.items()
    ):
        print(f"  {name}: {count}")

    print("\nSaved:", OUTPUT)
    print(
        "\nDSFORMER TRAIN HARD-NEGATIVE "
        "INDEX V1: PASS"
    )


if __name__ == "__main__":
    main()
