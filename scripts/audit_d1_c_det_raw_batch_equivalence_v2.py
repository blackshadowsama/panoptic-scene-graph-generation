#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from fair_psgg.data.load_entries import (
    load_psg_entries,
)


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_epoch_file(
    path: Path,
) -> dict[str, Any]:
    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, raw in enumerate(
            handle,
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

            records.append(record)

    starts = [
        record
        for record in records
        if record.get("kind")
        == "trace_start"
    ]

    ends = [
        record
        for record in records
        if record.get("kind")
        == "trace_end"
    ]

    batches = [
        record
        for record in records
        if record.get("kind")
        == "raw_batch"
    ]

    if len(starts) != 1:
        raise RuntimeError(
            f"{path}: expected one trace_start"
        )

    if len(ends) != 1:
        raise RuntimeError(
            f"{path}: expected one trace_end"
        )

    batches = sorted(
        batches,
        key=lambda record: int(
            record["batch_index"]
        ),
    )

    expected_indices = list(
        range(len(batches))
    )

    actual_indices = [
        int(record["batch_index"])
        for record in batches
    ]

    if actual_indices != expected_indices:
        raise RuntimeError(
            f"{path}: non-contiguous batch indices"
        )

    return {
        "path": str(path),
        "file_sha256": sha256_file(path),
        "start": starts[0],
        "end": ends[0],
        "batches": batches,
    }


def load_run(
    root: Path,
    expected_epochs: int,
) -> dict[int, dict[str, Any]]:
    files = sorted(
        root.glob("*.jsonl")
    )

    if len(files) != expected_epochs:
        raise RuntimeError(
            f"{root}: expected {expected_epochs} "
            f"files, got {len(files)}"
        )

    output = {}

    for path in files:
        epoch_data = load_epoch_file(
            path
        )

        epoch = int(
            epoch_data["start"]["epoch"]
        )

        if epoch in output:
            raise RuntimeError(
                f"Duplicate epoch {epoch}: {root}"
            )

        output[epoch] = epoch_data

    expected = set(
        range(expected_epochs)
    )

    if set(output) != expected:
        raise RuntimeError(
            f"{root}: epoch IDs are "
            f"{sorted(output)}, expected "
            f"{sorted(expected)}"
        )

    return output


def flatten_images(
    epoch_data: dict[str, Any],
) -> list[dict[str, Any]]:
    output = []

    for batch in epoch_data["batches"]:
        batch_index = int(
            batch["batch_index"]
        )

        for image in batch["images"]:
            copied = dict(image)

            copied["_batch_index"] = (
                batch_index
            )

            copied["_global_position"] = (
                len(output)
            )

            output.append(copied)

    return output


def image_pair_views(
    image: dict[str, Any],
) -> dict[str, Any]:
    rows = image["rows"]

    exact_rows = [
        (
            int(row["subject_index"]),
            int(row["object_index"]),
            int(row["none_label"]),
            tuple(
                int(value)
                for value in (
                    row["positive_labels"]
                )
            ),
            bool(row["is_positive"]),
            bool(row["is_hard_negative"]),
        )
        for row in rows
    ]

    positive_pairs = [
        tuple(
            int(value)
            for value in pair
        )
        for pair in (
            image["positive_pair_ids"]
        )
    ]

    negative_pairs = [
        tuple(
            int(value)
            for value in pair
        )
        for pair in (
            image[
                "selected_negative_pair_ids"
            ]
        )
    ]

    hard_flags = [
        bool(value)
        for value in (
            image[
                "selected_negative_is_hard"
            ]
        )
    ]

    return {
        "exact_rows": exact_rows,
        "positive_pairs": positive_pairs,
        "negative_pairs": negative_pairs,
        "hard_flags": hard_flags,
        "zero_treatment": (
            bool(
                image[
                    "zero_relation_input"
                ]
            ),
            int(
                image[
                    "requested_negative_budget"
                ]
            ),
            int(
                image[
                    "expected_actual_negative_budget"
                ]
            ),
            int(
                image[
                    "actual_negative_budget"
                ]
            ),
            bool(
                image[
                    "capacity_limited"
                ]
            ),
            bool(
                image[
                    "enters_split_batch"
                ]
            ),
        ),
    }


def append_mismatch(
    rows: list[dict[str, Any]],
    *,
    epoch: int,
    position: int,
    mismatch_type: str,
    c_value: Any,
    d1_value: Any,
) -> None:
    if len(rows) >= 1000:
        return

    rows.append({
        "epoch": epoch,
        "global_position": position,
        "mismatch_type": mismatch_type,
        "c_value": json.dumps(
            c_value,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ),
        "d1_value": json.dumps(
            d1_value,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ),
    })


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
        "--annotation",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--hardneg-index",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--c-config",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--d1-config",
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

    parser.add_argument(
        "--expected-epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--expected-batches",
        type=int,
        default=187,
    )

    parser.add_argument(
        "--expected-raw-images",
        type=int,
        default=1496,
    )

    parser.add_argument(
        "--expected-train-images",
        type=int,
        default=1500,
    )

    args = parser.parse_args()

    os.environ[
        "FLOODPSG_KEEP_ZERO_REL"
    ] = "1"

    train_entries, _, _ = load_psg_entries(
        str(args.annotation.resolve()),
        "train",
    )

    all_train_image_ids = {
        int(entry["image_id"])
        for entry in train_entries
    }

    if (
        len(all_train_image_ids)
        != args.expected_train_images
    ):
        raise RuntimeError(
            "Unexpected full training image count: "
            f"{len(all_train_image_ids)}"
        )

    c_run = load_run(
        args.c_trace.resolve(),
        args.expected_epochs,
    )

    d1_run = load_run(
        args.d1_trace.resolve(),
        args.expected_epochs,
    )

    counters = {
        "raw_batch_count_mismatches": 0,
        "raw_image_count_mismatches": 0,
        "image_order_mismatches": 0,
        "dataset_index_order_mismatches": 0,
        "pair_order_mismatches": 0,
        "positive_pair_mismatches": 0,
        "negative_pair_mismatches": 0,
        "hard_negative_flag_mismatches": 0,
        "positive_budget_mismatches": 0,
        "negative_budget_mismatches": 0,
        "zero_relation_treatment_mismatches": 0,
        "omitted_image_set_mismatches": 0,
        "trace_config_mismatches": 0,
        "internal_budget_failures": 0,
        "duplicate_image_failures": 0,
        "model_eligible_count_mismatches": 0,
    }

    mismatch_rows = []
    epoch_reports = []

    for epoch in range(
        args.expected_epochs
    ):
        c_epoch = c_run[epoch]
        d1_epoch = d1_run[epoch]

        c_start = c_epoch["start"]
        d1_start = d1_epoch["start"]

        config_keys = [
            "dataset_length",
            "loader_length",
            "batch_size",
            "drop_last",
            "num_workers",
            "neg_ratio",
            "sampling_mode",
            "zero_negatives_per_image",
            "hardneg_fraction",
            "hardneg_index_path",
            "deterministic_loader",
            "loader_seed",
            "deterministic_sampling",
            "sampling_seed",
            "keep_zero_rel",
        ]

        for key in config_keys:
            if c_start.get(key) != d1_start.get(
                key
            ):
                counters[
                    "trace_config_mismatches"
                ] += 1

                append_mismatch(
                    mismatch_rows,
                    epoch=epoch,
                    position=-1,
                    mismatch_type=(
                        f"trace_config:{key}"
                    ),
                    c_value=c_start.get(key),
                    d1_value=d1_start.get(key),
                )

        c_end = c_epoch["end"]
        d1_end = d1_epoch["end"]

        for end in (c_end, d1_end):
            if int(
                end["raw_batch_count"]
            ) != args.expected_batches:
                counters[
                    "raw_batch_count_mismatches"
                ] += 1

            if int(
                end["raw_image_count"]
            ) != args.expected_raw_images:
                counters[
                    "raw_image_count_mismatches"
                ] += 1

            if int(
                end["unique_image_ids"]
            ) != args.expected_raw_images:
                counters[
                    "duplicate_image_failures"
                ] += 1

            if int(
                end[
                    "model_eligible_image_count"
                ]
            ) != args.expected_raw_images:
                counters[
                    "model_eligible_count_mismatches"
                ] += 1

        c_images = flatten_images(
            c_epoch
        )

        d1_images = flatten_images(
            d1_epoch
        )

        if (
            len(c_images)
            != len(d1_images)
        ):
            counters[
                "raw_image_count_mismatches"
            ] += 1

        shared_length = min(
            len(c_images),
            len(d1_images),
        )

        for position in range(
            shared_length
        ):
            c_image = c_images[position]
            d1_image = d1_images[position]

            if (
                int(c_image["image_id"])
                != int(d1_image["image_id"])
            ):
                counters[
                    "image_order_mismatches"
                ] += 1

                append_mismatch(
                    mismatch_rows,
                    epoch=epoch,
                    position=position,
                    mismatch_type="image_order",
                    c_value=c_image["image_id"],
                    d1_value=d1_image["image_id"],
                )

            if (
                int(
                    c_image["dataset_index"]
                )
                != int(
                    d1_image["dataset_index"]
                )
            ):
                counters[
                    "dataset_index_order_mismatches"
                ] += 1

            c_view = image_pair_views(
                c_image
            )

            d1_view = image_pair_views(
                d1_image
            )

            checks = {
                "pair_order_mismatches": (
                    c_view["exact_rows"]
                    == d1_view["exact_rows"]
                ),
                "positive_pair_mismatches": (
                    c_view["positive_pairs"]
                    == d1_view[
                        "positive_pairs"
                    ]
                ),
                "negative_pair_mismatches": (
                    c_view["negative_pairs"]
                    == d1_view[
                        "negative_pairs"
                    ]
                ),
                "hard_negative_flag_mismatches": (
                    c_view["hard_flags"]
                    == d1_view["hard_flags"]
                ),
                "zero_relation_treatment_mismatches": (
                    c_view["zero_treatment"]
                    == d1_view[
                        "zero_treatment"
                    ]
                ),
                "positive_budget_mismatches": (
                    int(
                        c_image[
                            "positive_budget"
                        ]
                    )
                    == int(
                        d1_image[
                            "positive_budget"
                        ]
                    )
                ),
                "negative_budget_mismatches": (
                    int(
                        c_image[
                            "actual_negative_budget"
                        ]
                    )
                    == int(
                        d1_image[
                            "actual_negative_budget"
                        ]
                    )
                ),
            }

            for counter_name, passed in (
                checks.items()
            ):
                if passed:
                    continue

                counters[counter_name] += 1

                append_mismatch(
                    mismatch_rows,
                    epoch=epoch,
                    position=position,
                    mismatch_type=counter_name,
                    c_value=c_view,
                    d1_value=d1_view,
                )

            for image in (
                c_image,
                d1_image,
            ):
                if int(
                    image[
                        "actual_negative_budget"
                    ]
                ) != int(
                    image[
                        "expected_actual_negative_budget"
                    ]
                ):
                    counters[
                        "internal_budget_failures"
                    ] += 1

        c_seen = {
            int(image["image_id"])
            for image in c_images
        }

        d1_seen = {
            int(image["image_id"])
            for image in d1_images
        }

        c_omitted = sorted(
            all_train_image_ids - c_seen
        )

        d1_omitted = sorted(
            all_train_image_ids - d1_seen
        )

        if c_omitted != d1_omitted:
            counters[
                "omitted_image_set_mismatches"
            ] += 1

        epoch_reports.append({
            "epoch": epoch,
            "c_raw_batch_count": int(
                c_end["raw_batch_count"]
            ),
            "d1_raw_batch_count": int(
                d1_end["raw_batch_count"]
            ),
            "c_raw_image_count": int(
                c_end["raw_image_count"]
            ),
            "d1_raw_image_count": int(
                d1_end["raw_image_count"]
            ),
            "c_model_eligible_image_count": int(
                c_end[
                    "model_eligible_image_count"
                ]
            ),
            "d1_model_eligible_image_count": int(
                d1_end[
                    "model_eligible_image_count"
                ]
            ),
            "c_zero_relation_images": int(
                c_end["zero_relation_images"]
            ),
            "d1_zero_relation_images": int(
                d1_end[
                    "zero_relation_images"
                ]
            ),
            "c_positive_rows": int(
                c_end["positive_row_count"]
            ),
            "d1_positive_rows": int(
                d1_end["positive_row_count"]
            ),
            "c_negative_rows": int(
                c_end["negative_row_count"]
            ),
            "d1_negative_rows": int(
                d1_end["negative_row_count"]
            ),
            "c_hard_negative_rows": int(
                c_end[
                    "hard_negative_row_count"
                ]
            ),
            "d1_hard_negative_rows": int(
                d1_end[
                    "hard_negative_row_count"
                ]
            ),
            "c_omitted_image_ids": c_omitted,
            "d1_omitted_image_ids": d1_omitted,
            "c_canonical_trace_sha256": (
                c_end[
                    "canonical_raw_batch_sha256"
                ]
            ),
            "d1_canonical_trace_sha256": (
                d1_end[
                    "canonical_raw_batch_sha256"
                ]
            ),
            "canonical_trace_hash_equal": (
                c_end[
                    "canonical_raw_batch_sha256"
                ]
                == d1_end[
                    "canonical_raw_batch_sha256"
                ]
            ),
            "c_file_sha256": (
                c_epoch["file_sha256"]
            ),
            "d1_file_sha256": (
                d1_epoch["file_sha256"]
            ),
        })

    status = (
        "PASS"
        if all(
            value == 0
            for value in counters.values()
        )
        else "FAIL"
    )

    report = {
        "protocol": (
            "FloodPSG C-Det versus D1-Det "
            "raw-batch sampling equivalence"
        ),
        "audit_schema": (
            "D1_C_SAMPLING_EQUIVALENCE_"
            "AUDIT_V1_RAW_BATCH_TRACE_V2"
        ),
        "epochs_compared": (
            args.expected_epochs
        ),
        "expected_train_images": (
            args.expected_train_images
        ),
        "expected_raw_batches_per_epoch": (
            args.expected_batches
        ),
        "expected_raw_images_per_epoch": (
            args.expected_raw_images
        ),
        "seed": 3407,
        "workers": 4,
        "keep_zero_rel": True,
        "zero_negatives_per_image": 2,
        "hard_negative_fraction": 0.5,
        "input_sha256": {
            "annotation": sha256_file(
                args.annotation.resolve()
            ),
            "hardneg_index": sha256_file(
                args.hardneg_index.resolve()
            ),
            "c_config": sha256_file(
                args.c_config.resolve()
            ),
            "d1_config": sha256_file(
                args.d1_config.resolve()
            ),
        },
        "epoch_reports": epoch_reports,
        **counters,
        "mismatch_rows_written": len(
            mismatch_rows
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
            report,
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
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "global_position",
                "mismatch_type",
                "c_value",
                "d1_value",
            ],
        )

        writer.writeheader()
        writer.writerows(
            mismatch_rows
        )

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print(
        "JSON:",
        args.output_json,
    )
    print(
        "CSV:",
        args.output_csv,
    )
    print()

    if status != "PASS":
        print(
            "D1/C SAMPLING EQUIVALENCE "
            "AUDIT: FAIL"
        )
        raise SystemExit(1)

    print(
        "D1/C SAMPLING EQUIVALENCE "
        "AUDIT: PASS"
    )


if __name__ == "__main__":
    main()
