#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Passive FloodPSG sampling trace for D1/C equivalence audit.

Activated only when FLOODPSG_SAMPLING_TRACE_DIR is defined.
The hook records data; it does not reseed or replace sampling logic.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import BatchSampler
from torch.utils.data import get_worker_info


TRACE_DIR_RAW = os.environ.get(
    "FLOODPSG_SAMPLING_TRACE_DIR",
    "",
).strip()

RUN_ID = os.environ.get(
    "FLOODPSG_TRACE_RUN_ID",
    "unknown",
).strip()

EXPECTED_TRAIN_IMAGES = int(
    os.environ.get(
        "FLOODPSG_TRACE_EXPECTED_TRAIN_IMAGES",
        "1500",
    )
)

TRACE_CODE_COMMIT = os.environ.get(
    "FLOODPSG_TRACE_CODE_COMMIT",
    "",
).strip()


def _json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()

        return value.detach().cpu().tolist()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]

    if isinstance(value, list):
        return [_json_safe(item) for item in value]

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    return value


def _write_event(record: dict[str, Any]) -> None:
    if not TRACE_DIR_RAW:
        return

    trace_dir = Path(TRACE_DIR_RAW).expanduser().resolve()
    trace_dir.mkdir(parents=True, exist_ok=True)

    record = dict(record)

    record.setdefault(
        "time_ns",
        time.time_ns(),
    )
    record.setdefault(
        "run_id",
        RUN_ID,
    )
    record.setdefault(
        "pid",
        os.getpid(),
    )
    record.setdefault(
        "trace_code_commit",
        TRACE_CODE_COMMIT,
    )

    output = (
        trace_dir
        / f"{RUN_ID}.pid{os.getpid()}.jsonl"
    )

    with output.open(
        "a",
        encoding="utf-8",
        buffering=1,
    ) as file:
        file.write(
            json.dumps(
                _json_safe(record),
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )


def _rng_sha256() -> str:
    state = torch.get_rng_state()

    return hashlib.sha256(
        state.detach().cpu().numpy().tobytes()
    ).hexdigest()


if TRACE_DIR_RAW:
    _write_event({
        "kind": "sitecustomize_init",
        "torch_initial_seed": int(
            torch.initial_seed()
        ),
    })


# ------------------------------------------------------------------
# Trace main-process batch order without materializing the whole epoch.
# ------------------------------------------------------------------

if (
    TRACE_DIR_RAW
    and not getattr(
        BatchSampler,
        "_floodpsg_trace_v1_patched",
        False,
    )
):
    _original_batch_sampler_iter = BatchSampler.__iter__
    _train_epoch_counter = 0

    def _traced_batch_sampler_iter(self):
        global _train_epoch_counter

        sampler = getattr(
            self,
            "sampler",
            None,
        )

        data_source = getattr(
            sampler,
            "data_source",
            None,
        )

        try:
            data_source_length = len(data_source)
        except Exception:
            data_source_length = -1

        is_train_sampler = (
            data_source_length
            == EXPECTED_TRAIN_IMAGES
        )

        if not is_train_sampler:
            yield from _original_batch_sampler_iter(self)
            return

        epoch = int(_train_epoch_counter)
        _train_epoch_counter += 1

        _write_event({
            "kind": "batch_sampler_start",
            "epoch": epoch,
            "data_source_length": data_source_length,
            "batch_size": getattr(
                self,
                "batch_size",
                None,
            ),
            "drop_last": getattr(
                self,
                "drop_last",
                None,
            ),
        })

        batch_count = 0
        image_count = 0

        for batch_index, batch in enumerate(
            _original_batch_sampler_iter(self)
        ):
            indices = [
                int(index)
                for index in batch
            ]

            _write_event({
                "kind": "batch_indices",
                "epoch": epoch,
                "batch_index": int(batch_index),
                "indices": indices,
            })

            batch_count += 1
            image_count += len(indices)

            yield batch

        _write_event({
            "kind": "batch_sampler_end",
            "epoch": epoch,
            "batch_count": batch_count,
            "image_count": image_count,
        })

    BatchSampler.__iter__ = (
        _traced_batch_sampler_iter
    )

    BatchSampler._floodpsg_trace_v1_patched = True


# ------------------------------------------------------------------
# Trace actual output of sample_flood_negatives().
# ------------------------------------------------------------------

if TRACE_DIR_RAW:
    import fair_psgg.data.flood_hn_sampling as flood_module
    import fair_psgg.data.data as data_module

    if not getattr(
        flood_module,
        "_floodpsg_sampling_trace_v1_patched",
        False,
    ):
        _original_sample_flood_negatives = (
            flood_module.sample_flood_negatives
        )

        def _traced_sample_flood_negatives(
            *,
            boxes,
            rel_targets,
            neg_ratio,
            image_id,
            config,
        ):
            worker = get_worker_info()

            worker_id = (
                int(worker.id)
                if worker is not None
                else -1
            )

            worker_seed = (
                int(worker.seed)
                if worker is not None
                else int(torch.initial_seed())
            )

            rng_before = _rng_sha256()

            output = _original_sample_flood_negatives(
                boxes=boxes,
                rel_targets=rel_targets,
                neg_ratio=neg_ratio,
                image_id=image_id,
                config=config,
            )

            rng_after = _rng_sha256()

            hard_set = {
                (int(pair[0]), int(pair[1]))
                for pair in config.hardneg_by_image_id.get(
                    str(image_id),
                    (),
                )
            }

            rows = []

            output_cpu = (
                output.detach()
                .cpu()
                .to(dtype=torch.long)
            )

            for row_position, row in enumerate(
                output_cpu.tolist()
            ):
                subject = int(row[0])
                obj = int(row[1])

                labels = [
                    int(value)
                    for value in row[2:]
                ]

                is_positive = any(
                    value != 0
                    for value in labels
                )

                is_hard_negative = (
                    not is_positive
                    and (subject, obj) in hard_set
                )

                rows.append({
                    "row_position": int(row_position),
                    "subject_index": subject,
                    "object_index": obj,
                    "labels": labels,
                    "is_positive": bool(is_positive),
                    "is_negative": bool(
                        not is_positive
                    ),
                    "is_hard_negative": bool(
                        is_hard_negative
                    ),
                })

            try:
                num_boxes = len(boxes)
            except Exception:
                num_boxes = int(boxes)

            index_path = str(
                Path(
                    config.hardneg_index_path
                )
                .expanduser()
                .resolve()
            )

            _write_event({
                "kind": "sample_flood_negatives",
                "image_id": str(image_id),
                "worker_id": worker_id,
                "worker_seed": worker_seed,
                "torch_initial_seed": int(
                    torch.initial_seed()
                ),
                "rng_before_sha256": rng_before,
                "rng_after_sha256": rng_after,
                "num_boxes": int(num_boxes),
                "num_positive_input": int(
                    rel_targets.shape[0]
                ),
                "num_sampled_rows": int(
                    output.shape[0]
                ),
                "neg_ratio": float(neg_ratio),
                "sampling_mode": str(config.mode),
                "zero_negatives_per_image": int(
                    config.zero_negatives_per_image
                ),
                "hardneg_fraction": float(
                    config.hardneg_fraction
                ),
                "hardneg_index_path": index_path,
                "rows": rows,
            })

            return output

        flood_module.sample_flood_negatives = (
            _traced_sample_flood_negatives
        )

        # data.py imported the function directly, so both bindings
        # must point to the passive wrapper.
        data_module.sample_flood_negatives = (
            _traced_sample_flood_negatives
        )

        flood_module._floodpsg_sampling_trace_v1_patched = True

        _write_event({
            "kind": "sampling_function_patched",
            "status": "PASS",
        })
