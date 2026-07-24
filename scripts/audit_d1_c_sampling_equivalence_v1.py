#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import bisect
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_events(root: Path) -> list[dict[str, Any]]:
    files = sorted(root.glob("*.jsonl"))

    if not files:
        raise RuntimeError(
            f"No JSONL trace files found: {root}"
        )

    events: list[dict[str, Any]] = []

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

                record["_source_file"] = path.name
                record["_source_line"] = line_number

                events.append(record)

    return events


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

        is_positive = bool(row["is_positive"])
        is_hard = bool(
            row["is_hard_negative"]
        )

        exact.append((
            subject,
            obj,
            labels,
            is_positive,
            is_hard,
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

            if is_hard:
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


def parse_run(
    root: Path,
    *,
    expected_epochs: int,
    expected_images: int,
) -> dict[str, Any]:
    events = load_events(root)

    starts = sorted(
        (
            event
            for event in events
            if event.get("kind")
            == "batch_sampler_start"
        ),
        key=lambda event: (
            int(event["epoch"]),
            int(event["time_ns"]),
        ),
    )

    ends = sorted(
        (
            event
            for event in events
            if event.get("kind")
            == "batch_sampler_end"
        ),
        key=lambda event: (
            int(event["epoch"]),
            int(event["time_ns"]),
        ),
    )

    batches = [
        event
        for event in events
        if event.get("kind")
        == "batch_indices"
    ]

    samples = [
        event
        for event in events
        if event.get("kind")
        == "sample_flood_negatives"
    ]

    if len(starts) != expected_epochs:
        raise RuntimeError(
            f"{root}: expected {expected_epochs} "
            f"batch_sampler_start records, got {len(starts)}"
        )

    if len(ends) != expected_epochs:
        raise RuntimeError(
            f"{root}: expected {expected_epochs} "
            f"batch_sampler_end records, got {len(ends)}"
        )

    start_times = [
        int(event["time_ns"])
        for event in sorted(
            starts,
            key=lambda event: int(
                event["time_ns"]
            ),
        )
    ]

    time_to_epoch = {
        int(event["time_ns"]): int(
            event["epoch"]
        )
        for event in starts
    }

    epoch_orders: dict[int, list[int]] = {
        epoch: []
        for epoch in range(expected_epochs)
    }

    epoch_batches: dict[
        int,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for event in batches:
        epoch = int(event["epoch"])

        if epoch not in epoch_orders:
            continue

        epoch_batches[epoch].append(event)

    for epoch in range(expected_epochs):
        ordered_batches = sorted(
            epoch_batches[epoch],
            key=lambda event: int(
                event["batch_index"]
            ),
        )

        for event in ordered_batches:
            epoch_orders[epoch].extend(
                int(index)
                for index in event["indices"]
            )

        if len(epoch_orders[epoch]) != expected_images:
            raise RuntimeError(
                f"{root}: epoch {epoch} has "
                f"{len(epoch_orders[epoch])} indices; "
                f"expected {expected_images}"
            )

        if len(set(epoch_orders[epoch])) != expected_images:
            raise RuntimeError(
                f"{root}: epoch {epoch} contains "
                "duplicate or missing image indices"
            )

    sample_map: dict[
        tuple[int, str],
        dict[str, Any],
    ] = {}

    duplicate_keys: list[
        tuple[int, str]
    ] = []

    unassigned_samples = 0

    for sample in sorted(
        samples,
        key=lambda event: int(
            event["time_ns"]
        ),
    ):
        timestamp = int(sample["time_ns"])

        start_position = (
            bisect.bisect_right(
                start_times,
                timestamp,
            )
            - 1
        )

        if start_position < 0:
            unassigned_samples += 1
            continue

        start_time = start_times[start_position]
        epoch = time_to_epoch[start_time]

        if epoch >= expected_epochs:
            unassigned_samples += 1
            continue

        image_id = str(sample["image_id"])
        key = (epoch, image_id)

        if key in sample_map:
            duplicate_keys.append(key)
            continue

        sample = dict(sample)
        sample["_assigned_epoch"] = epoch

        sample_map[key] = sample

    expected_records = (
        expected_epochs
        * expected_images
    )

    config_signatures = sorted({
        (
            str(sample["sampling_mode"]),
            int(
                sample[
                    "zero_negatives_per_image"
                ]
            ),
            float(
                sample["hardneg_fraction"]
            ),
            str(
                sample["hardneg_index_path"]
            ),
        )
        for sample in sample_map.values()
    })

    return {
        "root": str(root),
        "event_count": len(events),
        "start_count": len(starts),
        "end_count": len(ends),
        "batch_count": len(batches),
        "sample_event_count": len(samples),
        "expected_sample_records": expected_records,
        "sample_map": sample_map,
        "duplicate_keys": duplicate_keys,
        "unassigned_samples": unassigned_samples,
        "epoch_orders": epoch_orders,
        "config_signatures": config_signatures,
    }


def json_text(value: Any) -> str:
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

    parser.add_argument(
        "--expected-epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--expected-images",
        type=int,
        default=1500,
    )

    args = parser.parse_args()

    c_run = parse_run(
        args.c_trace.resolve(),
        expected_epochs=args.expected_epochs,
        expected_images=args.expected_images,
    )

    d1_run = parse_run(
        args.d1_trace.resolve(),
        expected_epochs=args.expected_epochs,
        expected_images=args.expected_images,
    )

    image_order_mismatch_epochs = []

    for epoch in range(
        args.expected_epochs
    ):
        if (
            c_run["epoch_orders"][epoch]
            != d1_run["epoch_orders"][epoch]
        ):
            image_order_mismatch_epochs.append(
                epoch
            )

    c_map = c_run["sample_map"]
    d1_map = d1_run["sample_map"]

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
    hard_negative_flag_mismatches = 0

    mismatch_rows: list[
        dict[str, Any]
    ] = []

    for epoch, image_id in shared_keys:
        c_views = pair_views(
            c_map[(epoch, image_id)]["rows"]
        )

        d1_views = pair_views(
            d1_map[(epoch, image_id)]["rows"]
        )

        checks = {
            "pair_order": (
                c_views["exact"]
                == d1_views["exact"]
            ),
            "positive_pairs": (
                c_views["positives"]
                == d1_views["positives"]
            ),
            "negative_pairs": (
                c_views["negatives"]
                == d1_views["negatives"]
            ),
            "hard_negative_flags": (
                c_views["hard_negatives"]
                == d1_views["hard_negatives"]
            ),
        }

        if not checks["pair_order"]:
            pair_order_mismatches += 1

        if not checks["positive_pairs"]:
            positive_pair_mismatches += 1

        if not checks["negative_pairs"]:
            negative_pair_mismatches += 1

        if not checks["hard_negative_flags"]:
            hard_negative_flag_mismatches += 1

        for mismatch_type, passed in checks.items():
            if passed:
                continue

            if len(mismatch_rows) >= 500:
                continue

            mismatch_rows.append({
                "epoch": epoch,
                "image_id": image_id,
                "mismatch_type": mismatch_type,
                "c_value": json_text(
                    c_views[
                        {
                            "pair_order": "exact",
                            "positive_pairs": "positives",
                            "negative_pairs": "negatives",
                            "hard_negative_flags": "hard_negatives",
                        }[mismatch_type]
                    ]
                ),
                "d1_value": json_text(
                    d1_views[
                        {
                            "pair_order": "exact",
                            "positive_pairs": "positives",
                            "negative_pairs": "negatives",
                            "hard_negative_flags": "hard_negatives",
                        }[mismatch_type]
                    ]
                ),
            })

    duplicate_record_count = (
        len(c_run["duplicate_keys"])
        + len(d1_run["duplicate_keys"])
    )

    missing_record_count = (
        len(missing_from_c)
        + len(missing_from_d1)
        + int(c_run["unassigned_samples"])
        + int(d1_run["unassigned_samples"])
    )

    expected_config = {
        "mode": "flood_hn",
        "zero_negatives_per_image": 2,
        "hardneg_fraction": 0.5,
    }

    c_config_valid = (
        len(c_run["config_signatures"])
        == 1
        and c_run["config_signatures"][0][0]
        == "flood_hn"
        and c_run["config_signatures"][0][1]
        == 2
        and abs(
            c_run["config_signatures"][0][2]
            - 0.5
        )
        < 1e-12
    )

    d1_config_valid = (
        len(d1_run["config_signatures"])
        == 1
        and d1_run["config_signatures"][0][0]
        == "flood_hn"
        and d1_run["config_signatures"][0][1]
        == 2
        and abs(
            d1_run["config_signatures"][0][2]
            - 0.5
        )
        < 1e-12
    )

    same_index_path = (
        c_config_valid
        and d1_config_valid
        and c_run["config_signatures"][0][3]
        == d1_run["config_signatures"][0][3]
    )

    status = "PASS"

    failure_conditions = [
        len(image_order_mismatch_epochs) != 0,
        missing_record_count != 0,
        duplicate_record_count != 0,
        pair_order_mismatches != 0,
        positive_pair_mismatches != 0,
        negative_pair_mismatches != 0,
        hard_negative_flag_mismatches != 0,
        not c_config_valid,
        not d1_config_valid,
        not same_index_path,
        len(c_map)
        != args.expected_epochs
        * args.expected_images,
        len(d1_map)
        != args.expected_epochs
        * args.expected_images,
    ]

    if any(failure_conditions):
        status = "FAIL"

    summary = {
        "protocol": (
            "FloodPSG D1/C first-three-epoch "
            "sampling equivalence audit v1"
        ),
        "expected_sampling_config": expected_config,
        "epochs_compared": args.expected_epochs,
        "images_per_epoch": args.expected_images,
        "expected_image_epoch_records": (
            args.expected_epochs
            * args.expected_images
        ),
        "c_trace": {
            "path": c_run["root"],
            "event_count": c_run["event_count"],
            "sample_event_count": (
                c_run["sample_event_count"]
            ),
            "assigned_sample_records": len(
                c_map
            ),
            "config_signatures": (
                c_run["config_signatures"]
            ),
        },
        "d1_trace": {
            "path": d1_run["root"],
            "event_count": d1_run["event_count"],
            "sample_event_count": (
                d1_run["sample_event_count"]
            ),
            "assigned_sample_records": len(
                d1_map
            ),
            "config_signatures": (
                d1_run["config_signatures"]
            ),
        },
        "image_order_mismatch_epochs": len(
            image_order_mismatch_epochs
        ),
        "image_order_mismatch_epoch_ids": (
            image_order_mismatch_epochs
        ),
        "missing_record_count": (
            missing_record_count
        ),
        "duplicate_record_count": (
            duplicate_record_count
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
        "hard_negative_flag_mismatches": (
            hard_negative_flag_mismatches
        ),
        "c_sampling_config_valid": (
            c_config_valid
        ),
        "d1_sampling_config_valid": (
            d1_config_valid
        ),
        "same_hard_negative_index_path": (
            same_index_path
        ),
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

    if status != "PASS":
        print(
            "D1/C SAMPLING EQUIVALENCE AUDIT: FAIL"
        )
        raise SystemExit(1)

    print(
        "D1/C SAMPLING EQUIVALENCE AUDIT: PASS"
    )


if __name__ == "__main__":
    main()
