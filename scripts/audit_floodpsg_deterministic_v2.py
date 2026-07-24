#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import os

import torch
from torch.utils.data import TensorDataset

from fair_psgg.data.common import (
    get_generic_loader,
)

from fair_psgg.data.data import (
    _sample_flood_negatives_reproducibly,
)

from fair_psgg.data.flood_hn_sampling import (
    FloodNegSamplingConfig,
)


def tensor_sha256(
    value: torch.Tensor,
) -> str:
    return hashlib.sha256(
        value.detach()
        .cpu()
        .contiguous()
        .numpy()
        .tobytes()
    ).hexdigest()


os.environ[
    "FLOODPSG_TRAIN_SEED"
] = "3407"

os.environ[
    "FLOODPSG_DETERMINISTIC_LOADER"
] = "1"

os.environ[
    "FLOODPSG_LOADER_SEED"
] = "3407"

os.environ[
    "FLOODPSG_DETERMINISTIC_SAMPLING"
] = "1"

os.environ[
    "FLOODPSG_SAMPLING_SEED"
] = "3407"

os.environ[
    "FLOODPSG_CURRENT_EPOCH"
] = "0"


# ------------------------------------------------------------
# Audit 1: loader order is independent of global RNG use.
# ------------------------------------------------------------

dataset = TensorDataset(
    torch.arange(1500)
)


def make_loader_order(
    global_rng_burn: int,
) -> list[int]:
    loader = get_generic_loader(
        dataset=dataset,
        batch_size=8,
        num_workers=0,
        is_train=True,
    )

    # Simulate a different quantity of model-parameter
    # initialization after loader construction.
    torch.rand(global_rng_burn)

    return [
        int(index)
        for index in loader.sampler
    ]


torch.manual_seed(11)
order_c = make_loader_order(100)

torch.manual_seed(991)
order_d1 = make_loader_order(100_000)

assert len(order_c) == 1500
assert len(order_d1) == 1500
assert order_c == order_d1

print(
    "loader order identical after different "
    "global RNG consumption: PASS"
)

print(
    "loader order first 20:",
    order_c[:20],
)


# ------------------------------------------------------------
# Audit 2: positive-image FloodHN pairs are stable.
# ------------------------------------------------------------

num_classes = 8

positive_targets = torch.zeros(
    (2, 2 + num_classes),
    dtype=torch.long,
)

positive_targets[0, 0] = 0
positive_targets[0, 1] = 1
positive_targets[0, 2] = 1

positive_targets[1, 0] = 2
positive_targets[1, 1] = 3
positive_targets[1, 3] = 1

boxes = torch.zeros(
    (6, 4),
    dtype=torch.float32,
)

config = FloodNegSamplingConfig(
    mode="flood_hn",
    zero_negatives_per_image=2,
    hardneg_fraction=0.5,
    hardneg_index_path="unit-test",
    hardneg_by_image_id={
        "42": (
            (0, 2),
            (1, 3),
            (3, 0),
            (4, 5),
            (5, 4),
        ),
        "77": (
            (0, 2),
            (1, 4),
            (4, 0),
        ),
    },
)


def sample_positive(
    global_rng_seed: int,
    burn: int,
) -> torch.Tensor:
    torch.manual_seed(
        global_rng_seed
    )

    torch.rand(burn)

    state_before = (
        torch.get_rng_state().clone()
    )

    output = (
        _sample_flood_negatives_reproducibly(
            boxes=boxes,
            rel_targets=positive_targets,
            neg_ratio=3.0,
            image_id=42,
            config=config,
        )
    )

    state_after = (
        torch.get_rng_state().clone()
    )

    assert torch.equal(
        state_before,
        state_after,
    )

    return output


positive_c = sample_positive(
    123,
    7,
)

positive_d1 = sample_positive(
    98765,
    100_000,
)

assert torch.equal(
    positive_c,
    positive_d1,
)

print(
    "positive-image sampled pairs identical: PASS"
)

print(
    "positive-image outer RNG preserved: PASS"
)

print(
    "positive-image sample SHA256:",
    tensor_sha256(positive_c),
)


# ------------------------------------------------------------
# Audit 3: zero-relation images are deterministic.
# ------------------------------------------------------------

zero_targets = torch.zeros(
    (0, 2 + num_classes),
    dtype=torch.long,
)


def sample_zero(
    global_rng_seed: int,
    burn: int,
) -> torch.Tensor:
    torch.manual_seed(
        global_rng_seed
    )

    torch.rand(burn)

    state_before = (
        torch.get_rng_state().clone()
    )

    output = (
        _sample_flood_negatives_reproducibly(
            boxes=boxes,
            rel_targets=zero_targets,
            neg_ratio=1.0,
            image_id=77,
            config=config,
        )
    )

    state_after = (
        torch.get_rng_state().clone()
    )

    assert torch.equal(
        state_before,
        state_after,
    )

    return output


zero_c = sample_zero(
    456,
    5,
)

zero_d1 = sample_zero(
    654321,
    50_000,
)

assert torch.equal(
    zero_c,
    zero_d1,
)

assert zero_c.shape[0] == 2

print(
    "zero-relation sampled pairs identical: PASS"
)

print(
    "zero-relation quota equals 2: PASS"
)

print(
    "zero-relation outer RNG preserved: PASS"
)

print(
    "zero-relation sample SHA256:",
    tensor_sha256(zero_c),
)


# ------------------------------------------------------------
# Audit 4: epoch is part of the pair-sampling seed.
# ------------------------------------------------------------

os.environ[
    "FLOODPSG_CURRENT_EPOCH"
] = "1"

epoch_one = (
    _sample_flood_negatives_reproducibly(
        boxes=boxes,
        rel_targets=positive_targets,
        neg_ratio=3.0,
        image_id=42,
        config=config,
    )
)

print(
    "epoch 0 sample SHA256:",
    tensor_sha256(positive_c),
)

print(
    "epoch 1 sample SHA256:",
    tensor_sha256(epoch_one),
)

print()
print(
    "FLOODPSG DETERMINISTIC V2 UNIT AUDIT: PASS"
)
