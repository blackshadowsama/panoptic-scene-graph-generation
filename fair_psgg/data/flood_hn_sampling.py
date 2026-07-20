#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""FloodPSG training-only negative sampler.

This module does not change annotation relations. It only changes which
NONE pairs are sampled during DSFormer training.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

VALID_MODES = {
    "original",
    "uniform_all",
    "flood_hn",
}


@dataclass(frozen=True)
class FloodNegSamplingConfig:
    mode: str
    zero_negatives_per_image: int
    hardneg_fraction: float
    hardneg_index_path: str
    hardneg_by_image_id: dict[str, tuple[tuple[int, int], ...]]

    @property
    def enabled(self) -> bool:
        return self.mode != "original"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default)).strip()

    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(
            f"{name} must be an integer, got {raw!r}"
        ) from error

    return value


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default)).strip()

    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(
            f"{name} must be a float, got {raw!r}"
        ) from error

    return value


def _resolve_path(raw: str) -> Path:
    path = Path(raw).expanduser()

    if not path.is_absolute():
        path = Path.cwd() / path

    return path.resolve()


def _load_hardneg_index(
    raw_path: str,
) -> dict[str, tuple[tuple[int, int], ...]]:
    path = _resolve_path(raw_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Hard-negative index not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    raw_mapping = payload.get("pairs_by_image_id")

    if not isinstance(raw_mapping, dict):
        raise RuntimeError(
            "Hard-negative index has no pairs_by_image_id mapping."
        )

    mapping: dict[str, tuple[tuple[int, int], ...]] = {}

    for image_id, pairs in raw_mapping.items():
        parsed: list[tuple[int, int]] = []

        for pair in pairs:
            if not isinstance(pair, list) or len(pair) != 2:
                raise RuntimeError(
                    f"Invalid hard-negative pair for image "
                    f"{image_id}: {pair!r}"
                )

            subject = int(pair[0])
            obj = int(pair[1])

            if subject == obj:
                raise RuntimeError(
                    f"Self hard-negative pair in image "
                    f"{image_id}: {pair!r}"
                )

            parsed.append((subject, obj))

        mapping[str(image_id)] = tuple(sorted(set(parsed)))

    return mapping


def load_flood_neg_sampling_config(
    *,
    is_train: bool,
) -> FloodNegSamplingConfig:
    # Validation and inference always retain the original logic.
    if not is_train:
        return FloodNegSamplingConfig(
            mode="original",
            zero_negatives_per_image=0,
            hardneg_fraction=0.0,
            hardneg_index_path="",
            hardneg_by_image_id={},
        )

    mode = os.environ.get(
        "FLOODPSG_NEG_MODE",
        "original",
    ).strip()

    if mode not in VALID_MODES:
        raise ValueError(
            f"FLOODPSG_NEG_MODE={mode!r}; "
            f"expected one of {sorted(VALID_MODES)}"
        )

    zero_quota = _env_int(
        "FLOODPSG_ZERO_NEG_PER_IMAGE",
        0,
    )

    hard_fraction = _env_float(
        "FLOODPSG_HARDNEG_FRACTION",
        0.0,
    )

    if zero_quota < 0:
        raise ValueError(
            "FLOODPSG_ZERO_NEG_PER_IMAGE must be >= 0."
        )

    if not 0.0 <= hard_fraction <= 1.0:
        raise ValueError(
            "FLOODPSG_HARDNEG_FRACTION must be in [0, 1]."
        )

    raw_index_path = os.environ.get(
        "FLOODPSG_HARDNEG_INDEX",
        "",
    ).strip()

    hard_mapping: dict[
        str,
        tuple[tuple[int, int], ...],
    ] = {}

    if mode == "flood_hn":
        if not raw_index_path:
            raise RuntimeError(
                "FLOODPSG_HARDNEG_INDEX is required for "
                "FLOODPSG_NEG_MODE=flood_hn."
            )

        hard_mapping = _load_hardneg_index(
            raw_index_path
        )

    return FloodNegSamplingConfig(
        mode=mode,
        zero_negatives_per_image=zero_quota,
        hardneg_fraction=hard_fraction,
        hardneg_index_path=raw_index_path,
        hardneg_by_image_id=hard_mapping,
    )


def _take_random(
    pool: torch.Tensor,
    count: int,
) -> torch.Tensor:
    if count <= 0 or len(pool) == 0:
        return pool[:0]

    count = min(count, len(pool))

    indices = torch.randperm(
        len(pool),
        device=pool.device,
    )[:count]

    return pool[indices]


def _stochastic_quota(
    total: int,
    fraction: float,
) -> int:
    if total <= 0 or fraction <= 0:
        return 0

    # Binomial sampling gives the requested proportion in expectation
    # and avoids round(0.5) always becoming zero for small images.
    return int(
        (
            torch.rand(total)
            < fraction
        ).sum().item()
    )


def _pair_codes(
    pairs: torch.Tensor,
    num_boxes: int,
) -> torch.Tensor:
    return pairs[:, 0] * num_boxes + pairs[:, 1]


# FLOODPSG_PAIR_SHAPE_NORMALIZATION_V1

def _normalize_pair_tensor(
    pairs: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    """Normalize an object-pair tensor to shape [N, 2].

    The original Fair-PSGG helper may squeeze a singleton
    pair from [1, 2] to [2]. Empty results may also appear
    as a one-dimensional empty tensor. This function keeps
    the new FloodPSG sampler shape-safe without changing
    the historical DSFormer sampler.
    """
    if not isinstance(pairs, torch.Tensor):
        raise TypeError(
            f"{name} must be a torch.Tensor, "
            f"got {type(pairs)!r}"
        )

    if pairs.ndim == 2:
        if pairs.shape[1] != 2:
            raise RuntimeError(
                f"{name} must have shape [N, 2], "
                f"got {tuple(pairs.shape)}"
            )

        return pairs

    if pairs.ndim == 1:
        if pairs.numel() == 0:
            return pairs.reshape(0, 2)

        if pairs.numel() == 2:
            return pairs.reshape(1, 2)

    raise RuntimeError(
        f"{name} cannot be normalized to [N, 2]: "
        f"shape={tuple(pairs.shape)}, "
        f"numel={pairs.numel()}"
    )



# FLOODPSG_ORDERED_NEGATIVE_PAIR_POOL_V2
def _get_all_negative_pairs(
    *,
    boxes,
    rel_targets: torch.Tensor,
) -> torch.Tensor:
    """Return every legal negative ordered pair as [N, 2].

    Legal pair:
    1. subject != object;
    2. both endpoints are valid object indices;
    3. the endpoint pair is not a positive relation pair.

    The historical Fair-PSGG private helper returns one-dimensional
    candidate IDs rather than endpoint pairs. The FloodPSG sampler
    therefore constructs endpoint pairs explicitly.
    """
    if isinstance(boxes, int):
        num_boxes = int(boxes)
    else:
        num_boxes = len(boxes)

    device = rel_targets.device

    if num_boxes < 0:
        raise ValueError(
            f"num_boxes must be non-negative, got {num_boxes}"
        )

    if num_boxes < 2:
        return torch.empty(
            (0, 2),
            dtype=torch.long,
            device=device,
        )

    node_ids = torch.arange(
        num_boxes,
        dtype=torch.long,
        device=device,
    )

    subjects = node_ids.repeat_interleave(
        num_boxes
    )

    objects = node_ids.repeat(
        num_boxes
    )

    non_self = subjects != objects

    negative_pairs = torch.stack(
        (
            subjects[non_self],
            objects[non_self],
        ),
        dim=1,
    )

    if rel_targets.numel() == 0:
        return negative_pairs

    if (
        rel_targets.ndim != 2
        or rel_targets.shape[1] < 2
    ):
        raise RuntimeError(
            "rel_targets must have shape [R, 2 + K], "
            f"got {tuple(rel_targets.shape)}"
        )

    positive_pairs = rel_targets[
        :, :2
    ].to(
        device=device,
        dtype=torch.long,
    )

    if len(positive_pairs) == 0:
        return negative_pairs

    invalid_endpoint = (
        (positive_pairs < 0)
        | (positive_pairs >= num_boxes)
    ).any()

    if invalid_endpoint:
        raise RuntimeError(
            "Positive relation contains an endpoint "
            f"outside [0, {num_boxes - 1}]."
        )

    positive_codes = (
        positive_pairs[:, 0] * num_boxes
        + positive_pairs[:, 1]
    )

    candidate_codes = (
        negative_pairs[:, 0] * num_boxes
        + negative_pairs[:, 1]
    )

    keep = ~torch.isin(
        candidate_codes,
        positive_codes,
    )

    negative_pairs = negative_pairs[keep]

    if negative_pairs.ndim != 2:
        raise RuntimeError(
            "Negative-pair construction returned "
            f"invalid shape {tuple(negative_pairs.shape)}"
        )

    if negative_pairs.shape[1] != 2:
        raise RuntimeError(
            "Negative-pair construction must return "
            f"[N, 2], got {tuple(negative_pairs.shape)}"
        )

    return negative_pairs


def sample_flood_negatives(
    *,
    boxes: torch.Tensor,
    rel_targets: torch.Tensor,
    neg_ratio: float,
    image_id: Any,
    config: FloodNegSamplingConfig,
) -> torch.Tensor:
    if config.mode not in {
        "uniform_all",
        "flood_hn",
    }:
        raise ValueError(
            f"Unsupported FloodPSG sampler mode: "
            f"{config.mode}"
        )

    negative_pool = _get_all_negative_pairs(
        boxes=boxes,
        rel_targets=rel_targets,
    )

    num_positive = int(rel_targets.size(0))

    if num_positive > 0:
        requested_negative = round(
            float(neg_ratio) * num_positive
        )
    else:
        requested_negative = (
            config.zero_negatives_per_image
        )

    num_negative = min(
        requested_negative,
        len(negative_pool),
    )

    if num_negative <= 0:
        return rel_targets

    if config.mode == "uniform_all":
        selected_negative = _take_random(
            negative_pool,
            num_negative,
        )

    else:
        iid = str(image_id)

        hard_pairs = config.hardneg_by_image_id.get(
            iid,
            (),
        )

        num_boxes = len(boxes)

        if hard_pairs:
            hard_tensor = torch.tensor(
                hard_pairs,
                dtype=negative_pool.dtype,
                device=negative_pool.device,
            ).reshape(-1, 2)

            negative_codes = _pair_codes(
                negative_pool,
                num_boxes,
            )

            hard_codes = _pair_codes(
                hard_tensor,
                num_boxes,
            )

            hard_mask = torch.isin(
                negative_codes,
                hard_codes,
            )
        else:
            hard_mask = torch.zeros(
                len(negative_pool),
                dtype=torch.bool,
                device=negative_pool.device,
            )

        hard_pool = negative_pool[hard_mask]
        easy_pool = negative_pool[~hard_mask]

        hard_quota = _stochastic_quota(
            num_negative,
            config.hardneg_fraction,
        )

        hard_quota = min(
            hard_quota,
            len(hard_pool),
            num_negative,
        )

        selected_hard = _take_random(
            hard_pool,
            hard_quota,
        )

        easy_quota = min(
            num_negative - len(selected_hard),
            len(easy_pool),
        )

        selected_easy = _take_random(
            easy_pool,
            easy_quota,
        )

        remaining = (
            num_negative
            - len(selected_hard)
            - len(selected_easy)
        )

        if remaining > 0:
            if len(selected_hard):
                selected_codes = _pair_codes(
                    selected_hard,
                    num_boxes,
                )

                remaining_hard_mask = ~torch.isin(
                    _pair_codes(
                        hard_pool,
                        num_boxes,
                    ),
                    selected_codes,
                )

                remaining_hard_pool = hard_pool[
                    remaining_hard_mask
                ]
            else:
                remaining_hard_pool = hard_pool

            selected_fill = _take_random(
                remaining_hard_pool,
                remaining,
            )
        else:
            selected_fill = hard_pool[:0]

        selected_negative = torch.cat(
            (
                selected_hard,
                selected_easy,
                selected_fill,
            ),
            dim=0,
        )

        if len(selected_negative) > 1:
            selected_negative = selected_negative[
                torch.randperm(
                    len(selected_negative),
                    device=selected_negative.device,
                )
            ]

    selected_negative = _normalize_pair_tensor(
        selected_negative,
        name="selected_negative",
    )

    num_outputs = int(rel_targets.shape[1] - 2)

    negative_targets = torch.cat(
        (
            selected_negative,
            torch.zeros(
                (
                    len(selected_negative),
                    num_outputs,
                ),
                dtype=rel_targets.dtype,
                device=rel_targets.device,
            ),
        ),
        dim=1,
    )

    return torch.cat(
        (
            rel_targets,
            negative_targets,
        ),
        dim=0,
    )


def sampling_budget(
    *,
    entries: list[dict],
    neg_ratio: float,
    config: FloodNegSamplingConfig,
) -> dict[str, int | float]:
    positive_pairs = 0
    negative_pairs = 0
    zero_relation_images = 0
    capacity_limited_images = 0

    for entry in entries:
        positive_set = {
            (int(rel[0]), int(rel[1]))
            for rel in entry.get("relations", [])
        }

        num_positive = len(positive_set)
        num_nodes = len(entry.get("annotations", []))

        all_ordered = num_nodes * max(
            num_nodes - 1,
            0,
        )

        negative_capacity = max(
            all_ordered - num_positive,
            0,
        )

        if num_positive > 0:
            requested = round(
                float(neg_ratio) * num_positive
            )
        else:
            zero_relation_images += 1
            requested = (
                config.zero_negatives_per_image
            )

        actual = min(
            requested,
            negative_capacity,
        )

        if actual < requested:
            capacity_limited_images += 1

        positive_pairs += num_positive
        negative_pairs += actual

    if positive_pairs <= 0:
        raise RuntimeError(
            "No positive relation pairs in training entries."
        )

    return {
        "positive_pairs": positive_pairs,
        "negative_pairs": negative_pairs,
        "zero_relation_images": zero_relation_images,
        "capacity_limited_images": capacity_limited_images,
        "effective_none_ratio":
            negative_pairs / positive_pairs,
    }


def effective_none_ratio(
    *,
    entries: list[dict],
    neg_ratio: float,
    config: FloodNegSamplingConfig,
) -> float:
    if not config.enabled:
        return float(neg_ratio)

    return float(
        sampling_budget(
            entries=entries,
            neg_ratio=neg_ratio,
            config=config,
        )["effective_none_ratio"]
    )
