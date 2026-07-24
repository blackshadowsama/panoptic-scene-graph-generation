#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Passive raw-batch tracing for FloodPSG C-Det/D1-Det audits.

The tracer runs in the main training process after DataLoader
collation and before split_batch_iter(). It never changes a batch,
a tensor, an RNG state, or the order of iteration.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator

import torch


TRACE_SCHEMA = "FloodPSG-RawBatch-Trace-V2"


def _json_line(
    payload: dict[str, Any],
) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _write_record(
    handle,
    payload: dict[str, Any],
) -> bytes:
    encoded = _json_line(payload)

    handle.write(
        encoded.decode("utf-8")
    )
    handle.flush()

    return encoded


def _tensor_int_list(
    value: torch.Tensor,
) -> list[int]:
    return [
        int(item)
        for item in (
            value.detach()
            .cpu()
            .reshape(-1)
            .tolist()
        )
    ]


def _sampling_config(
    loader,
) -> tuple[
    Any,
    float,
    dict[str, set[tuple[int, int]]],
]:
    dataset = loader.dataset

    config = getattr(
        dataset,
        "flood_neg_sampling",
        None,
    )

    if config is None:
        raise RuntimeError(
            "Raw-batch tracing requires "
            "dataset.flood_neg_sampling."
        )

    neg_ratio = getattr(
        dataset,
        "neg_ratio",
        None,
    )

    if not isinstance(
        neg_ratio,
        (int, float),
    ):
        raise RuntimeError(
            "Raw-batch tracing requires a numeric "
            "training neg_ratio."
        )

    hard_mapping: dict[
        str,
        set[tuple[int, int]],
    ] = {}

    for image_id, pairs in (
        config.hardneg_by_image_id.items()
    ):
        hard_mapping[str(image_id)] = {
            (
                int(pair[0]),
                int(pair[1]),
            )
            for pair in pairs
        }

    return (
        config,
        float(neg_ratio),
        hard_mapping,
    )


def trace_raw_train_batches(
    loader,
    *,
    epoch: int,
) -> Iterator[dict]:
    """Yield raw batches unchanged while recording their contents."""

    trace_dir_raw = os.environ.get(
        "FLOODPSG_RAW_BATCH_TRACE_DIR",
        "",
    ).strip()

    if not trace_dir_raw:
        yield from loader
        return

    run_id = os.environ.get(
        "FLOODPSG_TRACE_RUN_ID",
        "",
    ).strip()

    if not run_id:
        raise RuntimeError(
            "FLOODPSG_TRACE_RUN_ID is required "
            "when raw-batch tracing is enabled."
        )

    trace_dir = Path(
        trace_dir_raw
    ).expanduser().resolve()

    trace_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    trace_path = (
        trace_dir
        / f"{run_id}.epoch{int(epoch):03d}.jsonl"
    )

    if trace_path.exists():
        raise RuntimeError(
            f"Refusing to overwrite raw trace: "
            f"{trace_path}"
        )

    config, neg_ratio, hard_mapping = (
        _sampling_config(loader)
    )

    dataset_length = len(
        loader.dataset
    )

    expected_raw_batches = len(
        loader
    )

    batch_hasher = hashlib.sha256()

    raw_batch_count = 0
    raw_image_count = 0
    model_eligible_image_count = 0

    positive_relation_images = 0
    zero_relation_images = 0
    capacity_limited_images = 0
    zero_sampled_relation_images = 0

    positive_row_count = 0
    negative_row_count = 0
    hard_negative_row_count = 0

    seen_dataset_indices: list[int] = []
    seen_image_ids: list[int] = []

    with trace_path.open(
        "x",
        encoding="utf-8",
        buffering=1,
    ) as handle:
        _write_record(
            handle,
            {
                "kind": "trace_start",
                "schema": TRACE_SCHEMA,
                "run_id": run_id,
                "epoch": int(epoch),
                "code_commit": os.environ.get(
                    "FLOODPSG_TRACE_CODE_COMMIT",
                    "",
                ),
                "dataset_length": int(
                    dataset_length
                ),
                "loader_length": int(
                    expected_raw_batches
                ),
                "batch_size": int(
                    loader.batch_size
                ),
                "drop_last": bool(
                    loader.drop_last
                ),
                "num_workers": int(
                    loader.num_workers
                ),
                "neg_ratio": neg_ratio,
                "sampling_mode": str(
                    config.mode
                ),
                "zero_negatives_per_image": int(
                    config.zero_negatives_per_image
                ),
                "hardneg_fraction": float(
                    config.hardneg_fraction
                ),
                "hardneg_index_path": str(
                    Path(
                        config.hardneg_index_path
                    )
                    .expanduser()
                    .resolve()
                ),
                "deterministic_loader": (
                    os.environ.get(
                        "FLOODPSG_DETERMINISTIC_LOADER",
                        "0",
                    )
                ),
                "loader_seed": (
                    os.environ.get(
                        "FLOODPSG_LOADER_SEED",
                        "",
                    )
                ),
                "deterministic_sampling": (
                    os.environ.get(
                        "FLOODPSG_DETERMINISTIC_SAMPLING",
                        "0",
                    )
                ),
                "sampling_seed": (
                    os.environ.get(
                        "FLOODPSG_SAMPLING_SEED",
                        "",
                    )
                ),
                "keep_zero_rel": (
                    os.environ.get(
                        "FLOODPSG_KEEP_ZERO_REL",
                        "0",
                    )
                ),
            },
        )

        for batch_index, batch in enumerate(
            loader
        ):
            image_ids = _tensor_int_list(
                batch["image_id"]
            )

            dataset_indices = _tensor_int_list(
                batch["idx"]
            )

            num_boxes = _tensor_int_list(
                batch["num_boxes"]
            )

            num_relations = _tensor_int_list(
                batch["num_relations"]
            )

            batch_size = len(image_ids)

            if not (
                len(dataset_indices)
                == len(num_boxes)
                == len(num_relations)
                == batch_size
            ):
                raise RuntimeError(
                    "Raw batch metadata length mismatch."
                )

            sampled = (
                batch["sampled_relations"]
                .detach()
                .cpu()
                .to(dtype=torch.long)
            )

            images: list[
                dict[str, Any]
            ] = []

            relation_offset = 0

            for image_position in range(
                batch_size
            ):
                image_id = int(
                    image_ids[image_position]
                )

                dataset_index = int(
                    dataset_indices[
                        image_position
                    ]
                )

                image_num_boxes = int(
                    num_boxes[image_position]
                )

                image_num_relations = int(
                    num_relations[
                        image_position
                    ]
                )

                relation_end = (
                    relation_offset
                    + image_num_relations
                )

                image_rows = sampled[
                    relation_offset:relation_end
                ]

                relation_offset = relation_end

                hard_set = hard_mapping.get(
                    str(image_id),
                    set(),
                )

                rows: list[
                    dict[str, Any]
                ] = []

                positive_pair_ids: list[
                    list[int]
                ] = []

                negative_pair_ids: list[
                    list[int]
                ] = []

                negative_is_hard: list[
                    bool
                ] = []

                for row_position, row in enumerate(
                    image_rows.tolist()
                ):
                    if len(row) < 3:
                        raise RuntimeError(
                            "Invalid sampled relation row."
                        )

                    subject = int(row[0])
                    obj = int(row[1])
                    none_label = int(row[2])

                    positive_labels = [
                        int(value)
                        for value in row[3:]
                    ]

                    is_positive = any(
                        value != 0
                        for value in positive_labels
                    )

                    is_negative = not is_positive

                    if is_positive and none_label != 0:
                        raise RuntimeError(
                            "Positive row has nonzero "
                            "NONE label."
                        )

                    if is_negative and none_label != 1:
                        raise RuntimeError(
                            "Negative row does not have "
                            "NONE label equal to one."
                        )

                    is_hard_negative = (
                        is_negative
                        and (
                            subject,
                            obj,
                        )
                        in hard_set
                    )

                    rows.append({
                        "row_position": int(
                            row_position
                        ),
                        "subject_index": subject,
                        "object_index": obj,
                        "none_label": none_label,
                        "positive_labels": (
                            positive_labels
                        ),
                        "is_positive": bool(
                            is_positive
                        ),
                        "is_negative": bool(
                            is_negative
                        ),
                        "is_hard_negative": bool(
                            is_hard_negative
                        ),
                    })

                    if is_positive:
                        positive_pair_ids.append([
                            subject,
                            obj,
                        ])
                    else:
                        negative_pair_ids.append([
                            subject,
                            obj,
                        ])

                        negative_is_hard.append(
                            bool(
                                is_hard_negative
                            )
                        )

                positive_budget = len(
                    positive_pair_ids
                )

                negative_budget = len(
                    negative_pair_ids
                )

                positive_pair_set = {
                    (
                        int(pair[0]),
                        int(pair[1]),
                    )
                    for pair in (
                        positive_pair_ids
                    )
                }

                negative_capacity = max(
                    image_num_boxes
                    * max(
                        image_num_boxes - 1,
                        0,
                    )
                    - len(
                        positive_pair_set
                    ),
                    0,
                )

                zero_relation_input = (
                    positive_budget == 0
                )

                if zero_relation_input:
                    requested_negative_budget = (
                        int(
                            config
                            .zero_negatives_per_image
                        )
                    )
                else:
                    requested_negative_budget = (
                        round(
                            neg_ratio
                            * positive_budget
                        )
                    )

                expected_actual_negative_budget = (
                    min(
                        requested_negative_budget,
                        negative_capacity,
                    )
                )

                capacity_limited = (
                    expected_actual_negative_budget
                    < requested_negative_budget
                )

                if (
                    negative_budget
                    != expected_actual_negative_budget
                ):
                    raise RuntimeError(
                        "Negative-budget mismatch for "
                        f"image_id={image_id}: "
                        f"actual={negative_budget}, "
                        f"expected="
                        f"{expected_actual_negative_budget}"
                    )

                enters_split_batch = (
                    image_num_relations > 0
                )

                if zero_relation_input:
                    zero_relation_images += 1
                else:
                    positive_relation_images += 1

                if capacity_limited:
                    capacity_limited_images += 1

                if not enters_split_batch:
                    zero_sampled_relation_images += 1
                else:
                    model_eligible_image_count += 1

                positive_row_count += (
                    positive_budget
                )

                negative_row_count += (
                    negative_budget
                )

                hard_negative_row_count += sum(
                    1
                    for flag in negative_is_hard
                    if flag
                )

                images.append({
                    "image_position": int(
                        image_position
                    ),
                    "dataset_index": (
                        dataset_index
                    ),
                    "image_id": image_id,
                    "num_boxes": (
                        image_num_boxes
                    ),
                    "num_relations": (
                        image_num_relations
                    ),
                    "positive_pair_ids": (
                        positive_pair_ids
                    ),
                    "selected_negative_pair_ids": (
                        negative_pair_ids
                    ),
                    "selected_negative_is_hard": (
                        negative_is_hard
                    ),
                    "positive_budget": (
                        positive_budget
                    ),
                    "requested_negative_budget": (
                        requested_negative_budget
                    ),
                    "expected_actual_negative_budget": (
                        expected_actual_negative_budget
                    ),
                    "actual_negative_budget": (
                        negative_budget
                    ),
                    "negative_capacity": (
                        negative_capacity
                    ),
                    "zero_relation_input": bool(
                        zero_relation_input
                    ),
                    "capacity_limited": bool(
                        capacity_limited
                    ),
                    "enters_split_batch": bool(
                        enters_split_batch
                    ),
                    "rows": rows,
                })

                seen_dataset_indices.append(
                    dataset_index
                )

                seen_image_ids.append(
                    image_id
                )

            if relation_offset != len(sampled):
                raise RuntimeError(
                    "Raw batch relation slicing did "
                    "not consume all relation rows."
                )

            record = {
                "kind": "raw_batch",
                "schema": TRACE_SCHEMA,
                "run_id": run_id,
                "epoch": int(epoch),
                "batch_index": int(
                    batch_index
                ),
                "batch_size": int(
                    batch_size
                ),
                "dataset_indices": (
                    dataset_indices
                ),
                "image_ids": image_ids,
                "num_boxes": num_boxes,
                "num_relations": (
                    num_relations
                ),
                "images": images,
            }

            encoded = _write_record(
                handle,
                record,
            )

            batch_hasher.update(
                encoded
            )

            raw_batch_count += 1
            raw_image_count += (
                batch_size
            )

            yield batch

        summary = {
            "kind": "trace_end",
            "schema": TRACE_SCHEMA,
            "run_id": run_id,
            "epoch": int(epoch),
            "raw_batch_count": int(
                raw_batch_count
            ),
            "raw_image_count": int(
                raw_image_count
            ),
            "unique_dataset_indices": int(
                len(set(
                    seen_dataset_indices
                ))
            ),
            "unique_image_ids": int(
                len(set(
                    seen_image_ids
                ))
            ),
            "model_eligible_image_count": int(
                model_eligible_image_count
            ),
            "positive_relation_images": int(
                positive_relation_images
            ),
            "zero_relation_images": int(
                zero_relation_images
            ),
            "capacity_limited_images": int(
                capacity_limited_images
            ),
            "zero_sampled_relation_images": int(
                zero_sampled_relation_images
            ),
            "positive_row_count": int(
                positive_row_count
            ),
            "negative_row_count": int(
                negative_row_count
            ),
            "hard_negative_row_count": int(
                hard_negative_row_count
            ),
            "canonical_raw_batch_sha256": (
                batch_hasher.hexdigest()
            ),
        }

        _write_record(
            handle,
            summary,
        )
