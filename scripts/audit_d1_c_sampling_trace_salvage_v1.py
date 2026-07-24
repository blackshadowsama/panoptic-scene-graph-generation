#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_samples(root: Path) -> list[dict[str, Any]]:
    files = sorted(root.glob("*.jsonl"))

    if not files:
        raise RuntimeError(
            f"No JSONL files found under: {root}"
        )

    samples: list[dict[str, Any]] = []

    for path in files:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line_number, raw in enumerate(
                file,
                start=1,
            ):
                raw = raw.strip()

                if not raw:
                    continue

                try:
                    record = json.loads(raw)
                except json.JSONDecodeError as error:
                    raise RuntimeError(
                        f"Invalid JSON: {path}:{line_number}"
                    ) from error

                if (
                    record.get("kind")
                    != "sample_flood_negatives"
                ):
                    continue

                record["_source_file"] = path.name
                record["_source_line"] = line_number

                samples.append(record)

    if not samples:
        raise RuntimeError(
            f"No sample_flood_negatives records: {root}"
        )

    return samples


def recover_epochs(
    root: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    samples = load_samples(root)

    cohorts: dict[
        int,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for record in samples:
        worker_seed = int(record["worker_seed"])
        worker_id = int(record["worker_id"])

        base_seed = worker_seed - worker_id

        cohorts[base_seed].append(record)

    ordered_cohorts = sorted(
        cohorts.items(),
        key=lambda item: min(
            int(record["time_ns"])
            for record in item[1]
        ),
    )

    epoch_summaries: list[dict[str, Any]] = []
    recovered: list[dict[str, Any]] = []

    for epoch, (
        base_seed,
        records,
    ) in enumerate(ordered_cohorts):
        records = sorted(
            records,
            key=lambda record: int(
                record["time_ns"]
            ),
        )

        image_ids = [
            str(record["image_id"])
            for record in records
        ]

        unique_image_ids = set(image_ids)

        worker_ids = sorted({
            int(record["worker_id"])
            for record in records
        })

        source_files = sorted({
            str(record["_source_file"])
            for record in records
        })

        duplicate_images = (
            len(image_ids)
            - len(unique_image_ids)
        )

        epoch_summaries.append({
            "epoch": epoch,
            "base_seed": base_seed,
            "records": len(records),
            "unique_images": len(
                unique_image_ids
            ),
            "duplicate_images": (
                duplicate_images
            ),
            "worker_ids": worker_ids,
            "source_files": source_files,
            "first_time_ns": min(
                int(record["time_ns"])
                for record in records
            ),
            "last_time_ns": max(
                int(record["time_ns"])
                for record in records
            ),
        })

        for record in records:
            copied = dict(record)
            copied["_recovered_epoch"] = epoch
            copied["_worker_base_seed"] = (
                base_seed
            )
            recovered.append(copied)

    return recovered, epoch_summaries


def pair_views(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    exact = []
    positives = []
    negatives = []
    hard_negatives = []

    for row in rows:
        subject = int(row["subject_index"])
        obj = int(row["object_index"])

        labels = tuple(
            int(value)
            for value in row["labels"]
        )

        is_positive = bool(
            row["is_positive"]
        )

        is_hard_negative = bool(
            row["is_hard_negative"]
        )

        exact.append((
            subject,
            obj,
            labels,
            is_positive,
            is_hard_negative,
        ))

        if is_positive:
            positives.append((
                subject,
                obj,
                labels,
            ))
        else:
            negatives.append((
                subject,
                obj,
            ))

            if is_hard_negative:
                hard_negatives.append((
                    subject,
                    obj,
                ))

    return {
        "exact": exact,
        "positives": sorted(positives),
        "negatives": sorted(negatives),
        "hard_negatives": sorted(
            hard_negatives
        ),
    }


def make_sample_map(
    records: list[dict[str, Any]],
) -> tuple[
    dict[tuple[int, str], dict[str, Any]],
    list[tuple[int, str]],
]:
    mapping: dict[
        tuple[int, str],
        dict[str, Any],
    ] = {}

    duplicates: list[
        tuple[int, str]
    ] = []

    for record in records:
        key = (
            int(record["_recovered_epoch"]),
            str(record["image_id"]),
        )

        if key in mapping:
            duplicates.append(key)
            continue

        mapping[key] = record

    return mapping, duplicates


def safe_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--c-trace",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--d1-trace",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    c_records, c_epochs = recover_epochs(
        args.c_trace.resolve()
    )

    d1_records, d1_epochs = recover_epochs(
        args.d1_trace.resolve()
    )

    c_map, c_duplicates = make_sample_map(
        c_records
    )

    d1_map, d1_duplicates = make_sample_map(
        d1_records
    )

    c_keys = set(c_map)
    d1_keys = set(d1_map)

    missing_from_c = sorted(
        d1_keys - c_keys
    )

    missing_from_d1 = sorted(
        c_keys - d1_keys
    )

    shared_keys = sorted(
        c_keys & d1_keys
    )

    pair_order_mismatches = 0
    positive_pair_mismatches = 0
    negative_pair_mismatches = 0
    hard_negative_mismatches = 0
    rng_before_mismatches = 0
    worker_seed_mismatch_epochs = 0

    mismatch_rows: list[
        dict[str, Any]
    ] = []

    min_epochs = min(
        len(c_epochs),
        len(d1_epochs),
    )

    for epoch in range(min_epochs):
        if (
            int(c_epochs[epoch]["base_seed"])
            != int(d1_epochs[epoch]["base_seed"])
        ):
            worker_seed_mismatch_epochs += 1

    for key in shared_keys:
        epoch, image_id = key

        c_record = c_map[key]
        d1_record = d1_map[key]

        c_view = pair_views(
            c_record["rows"]
        )

        d1_view = pair_views(
            d1_record["rows"]
        )

        checks = {
            "pair_order": (
                c_view["exact"]
                == d1_view["exact"]
            ),
            "positive_pairs": (
                c_view["positives"]
                == d1_view["positives"]
            ),
            "negative_pairs": (
                c_view["negatives"]
                == d1_view["negatives"]
            ),
            "hard_negative_pairs": (
                c_view["hard_negatives"]
                == d1_view["hard_negatives"]
            ),
        }

        if (
            c_record.get("rng_before_sha256")
            != d1_record.get(
                "rng_before_sha256"
            )
        ):
            rng_before_mismatches += 1

        if not checks["pair_order"]:
            pair_order_mismatches += 1

        if not checks["positive_pairs"]:
            positive_pair_mismatches += 1

        if not checks["negative_pairs"]:
            negative_pair_mismatches += 1

        if not checks["hard_negative_pairs"]:
            hard_negative_mismatches += 1

        for mismatch_type, passed in checks.items():
            if passed:
                continue

            if len(mismatch_rows) >= 500:
                continue

            view_key = {
                "pair_order": "exact",
                "positive_pairs": "positives",
                "negative_pairs": "negatives",
                "hard_negative_pairs": (
                    "hard_negatives"
                ),
            }[mismatch_type]

            mismatch_rows.append({
                "epoch": epoch,
                "image_id": image_id,
                "mismatch_type": (
                    mismatch_type
                ),
                "c_value": safe_json(
                    c_view[view_key]
                ),
                "d1_value": safe_json(
                    d1_view[view_key]
                ),
            })

    c_config_signatures = sorted({
        (
            str(record["sampling_mode"]),
            int(
                record[
                    "zero_negatives_per_image"
                ]
            ),
            float(
                record["hardneg_fraction"]
            ),
            str(
                record["hardneg_index_path"]
            ),
        )
        for record in c_records
    })

    d1_config_signatures = sorted({
        (
            str(record["sampling_mode"]),
            int(
                record[
                    "zero_negatives_per_image"
                ]
            ),
            float(
                record["hardneg_fraction"]
            ),
            str(
                record["hardneg_index_path"]
            ),
        )
        for record in d1_records
    })

    pair_sampling_equal = all([
        len(c_epochs) == 3,
        len(d1_epochs) == 3,
        not c_duplicates,
        not d1_duplicates,
        not missing_from_c,
        not missing_from_d1,
        pair_order_mismatches == 0,
        positive_pair_mismatches == 0,
        negative_pair_mismatches == 0,
        hard_negative_mismatches == 0,
        c_config_signatures
        == d1_config_signatures,
    ])

    if pair_sampling_equal:
        status = (
            "PAIR_SAMPLING_MATCH_"
            "IMAGE_ORDER_UNVERIFIED"
        )
    else:
        status = "PAIR_SAMPLING_MISMATCH"

    summary = {
        "protocol": (
            "D1/C existing sampling trace "
            "worker-cohort recovery v1"
        ),
        "warning": (
            "The original trace did not record "
            "main-process batch order. This is "
            "diagnostic, not the final formal "
            "sampling-equivalence audit."
        ),
        "c_epochs": c_epochs,
        "d1_epochs": d1_epochs,
        "c_total_sample_records": (
            len(c_records)
        ),
        "d1_total_sample_records": (
            len(d1_records)
        ),
        "c_unique_epoch_image_records": (
            len(c_map)
        ),
        "d1_unique_epoch_image_records": (
            len(d1_map)
        ),
        "shared_epoch_image_records": (
            len(shared_keys)
        ),
        "missing_from_c": len(
            missing_from_c
        ),
        "missing_from_d1": len(
            missing_from_d1
        ),
        "c_duplicate_epoch_image_records": (
            len(c_duplicates)
        ),
        "d1_duplicate_epoch_image_records": (
            len(d1_duplicates)
        ),
        "worker_seed_mismatch_epochs": (
            worker_seed_mismatch_epochs
        ),
        "rng_before_mismatches": (
            rng_before_mismatches
        ),
        "pair_order_mismatches": (
            pair_order_mismatches
        ),
        "positive_pair_mismatches": (
            positive_pair_mismatches
        ),
        "negative_pair_mismatches": (
            negative_pair_mismatches
        ),
        "hard_negative_pair_mismatches": (
            hard_negative_mismatches
        ),
        "c_sampling_config_signatures": (
            c_config_signatures
        ),
        "d1_sampling_config_signatures": (
            d1_config_signatures
        ),
        "image_order_verified": False,
        "pair_sampling_equal": (
            pair_sampling_equal
        ),
        "status": status,
    }

    args.output_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output_json.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    with args.output_csv.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "image_id",
                "mismatch_type",
                "c_value",
                "d1_value",
            ],
        )

        writer.writeheader()
        writer.writerows(mismatch_rows)

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print("JSON:", args.output_json)
    print("CSV:", args.output_csv)
    print()
    print("Recovered trace status:", status)


if __name__ == "__main__":
    main()
