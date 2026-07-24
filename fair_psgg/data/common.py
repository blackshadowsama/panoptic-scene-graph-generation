# functions used by different datasets/dataloaders
from collections import defaultdict
import os

import torch
from torch.utils.data import Dataset, DataLoader


def custom_collate(samples):
    collated = defaultdict(list)
    for sample in samples:
        for k, v in sample.items():
            # only tensors can be stacked/concatenated
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            collated[k].append(v)

    output = {}
    for key in collated:
        if key in ("img", "idx", "image_id"):
            output[key] = torch.stack(collated[key])
        else:
            output[key] = torch.cat(collated[key])
            if key == "bboxes":
                output["num_boxes"] = torch.tensor([len(x) for x in collated[key]])

            if key == "sampled_relations":
                output["num_relations"] = torch.tensor([len(x) for x in collated[key]])

    assert output["num_boxes"].shape == output["num_relations"].shape

    return output


def get_generic_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    is_train: bool,
):
    # FLOODPSG_DETERMINISTIC_LOADER_V2
    #
    # The generator is separate from PyTorch's global RNG.
    # Model initialization, including optional FIBE parameters,
    # therefore cannot alter training-image shuffle order or
    # DataLoader worker base seeds.
    generator = None

    deterministic_loader = (
        is_train
        and os.environ.get(
            "FLOODPSG_DETERMINISTIC_LOADER",
            "0",
        ) == "1"
    )

    if deterministic_loader:
        loader_seed = int(
            os.environ.get(
                "FLOODPSG_LOADER_SEED",
                os.environ.get(
                    "FLOODPSG_TRAIN_SEED",
                    "3407",
                ),
            )
        )

        generator = torch.Generator(
            device="cpu"
        )

        generator.manual_seed(
            loader_seed
        )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=custom_collate,
        shuffle=is_train,
        drop_last=is_train,
        num_workers=num_workers,
        generator=generator,
    )
