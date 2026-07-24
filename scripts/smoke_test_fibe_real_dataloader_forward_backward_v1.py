
import random
from pathlib import Path

import numpy as np
import torch

from fair_psgg.config import Config
from fair_psgg.data.split_batch import split_batch
from fair_psgg.trainer import Trainer, prepare_batch


SEED = 3407

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

config_path = Path(
    "configs/floodpsg/"
    "masks-loc-sem-flood-groupstrict-coarse8-fibe-d1-v1.json"
)
annotation_path = Path(
    "data/floodpsg/annotations/"
    "floodpsg_canonical_trainval_coarse8_groupstrict_v1.json"
)
image_dir = Path("data/floodpsg")
segmentation_dir = Path("data/floodpsg")

for path in (
    config_path,
    annotation_path,
    image_dir,
    segmentation_dir,
):
    if not path.exists():
        raise FileNotFoundError(path)

config = Config.from_file(config_path)

assert config.fibe.enabled is True
assert config.fibe.feature_dim == 21
assert config.batch_size == 8
assert config.rels_per_batch == 128
assert config.lr == 3.73e-05
assert config.weight_decay == 0.04
assert config.neg_ratio == 1.0

trainer = Trainer(
    anno_path=annotation_path,
    img_dir=image_dir,
    seg_dir=segmentation_dir,
    config=config,
    out_dir=None,
    num_workers=0,
    dump_data=False,
    start_state_dict=None,
    hide_batch_progress=True,
)

assert trainer.fibe_train_cache is not None
assert trainer.fibe_val_cache is not None
assert trainer.model.fibe_branch is not None

selected_batch = None
selected_input = None

# 找到至少含一个有效 FIBE 对象对的真实训练子批次。
for raw_batch_index, raw_batch in enumerate(trainer.train_loader):
    for sub_batch_index, sub_batch in enumerate(
        split_batch(
            raw_batch,
            max_relations=trainer.rels_per_batch,
        )
    ):
        model_input, _, _, _ = prepare_batch(
            sub_batch,
            trainer.device,
            fibe_cache=trainer.fibe_train_cache,
        )

        valid = model_input["fibe_valid"]

        if bool(valid.any().item()):
            selected_batch = sub_batch
            selected_input = model_input
            print(
                "selected raw/sub batch:",
                raw_batch_index,
                sub_batch_index,
            )
            break

    if selected_batch is not None:
        break

if selected_batch is None or selected_input is None:
    raise RuntimeError(
        "No real training sub-batch with a valid FIBE pair was found."
    )

num_relations = int(
    selected_batch["num_relations"].sum().item()
)
fibe_features = selected_input["fibe_features"]
fibe_valid = selected_input["fibe_valid"]

assert tuple(fibe_features.shape) == (num_relations, 21)
assert tuple(fibe_valid.shape) == (num_relations,)
assert torch.isfinite(fibe_features).all()

if bool((~fibe_valid).any().item()):
    invalid_values = fibe_features[~fibe_valid]
    assert torch.count_nonzero(invalid_values).item() == 0

branch = trainer.model.fibe_branch
branch_parameter_count = sum(
    parameter.numel()
    for parameter in branch.parameters()
)

assert branch_parameter_count == 60546, (
    f"Unexpected FIBE parameter count: "
    f"{branch_parameter_count}"
)

print("device:", trainer.device)
print("image_ids:", selected_batch["image_id"].tolist())
print("num_images:", len(selected_batch["image_id"]))
print("num_boxes:", int(selected_batch["num_boxes"].sum()))
print("num_relations:", num_relations)
print("valid_fibe_pairs:", int(fibe_valid.sum()))
print("invalid_fibe_pairs:", int((~fibe_valid).sum()))
print("fibe_parameter_count:", branch_parameter_count)
print("initial_alpha:", float(branch.effective_alpha().detach().cpu()))
print(
    "initial_gate_bias:",
    float(branch.gate.bias.detach().cpu().item()),
)

trainer.model.train()
trainer.optimizer.zero_grad(set_to_none=True)

forward = trainer._common_forward(
    selected_batch,
    fibe_cache=trainer.fibe_train_cache,
)

for key in (
    "loss",
    "node_loss",
    "rel_loss",
    "sbj_out",
    "obj_out",
    "rel_out",
):
    value = forward[key]

    if isinstance(value, torch.Tensor):
        assert torch.isfinite(value).all(), key

loss = forward["loss"]
loss.backward()

gradient_parameters = {
    "raw_alpha": branch.raw_alpha,
    "encoder_linear_1": branch.encoder[0].weight,
    "encoder_linear_2": branch.encoder[3].weight,
    "projection": branch.projection[0].weight,
    "gate": branch.gate.weight,
}

for name, parameter in gradient_parameters.items():
    gradient = parameter.grad

    if gradient is None:
        raise AssertionError(
            f"Missing real-batch gradient: {name}"
        )

    if not torch.isfinite(gradient).all():
        raise AssertionError(
            f"Non-finite real-batch gradient: {name}"
        )

    print(
        f"gradient_norm/{name}:",
        float(gradient.norm().detach().cpu()),
    )

print("loss:", float(forward["loss"].detach().cpu()))
print(
    "node_loss:",
    float(forward["node_loss"]),
)
print(
    "rel_loss:",
    float(forward["rel_loss"]),
)
print("subject_output_shape:", tuple(forward["sbj_out"].shape))
print("object_output_shape:", tuple(forward["obj_out"].shape))
print("relation_output_shape:", tuple(forward["rel_out"].shape))

print()
print("PASS: real DataLoader cache lookup")
print("PASS: real sampled-pair alignment")
print("PASS: real DaniFormer forward")
print("PASS: finite real training losses")
print("PASS: real FIBE backward gradients")
print()
print("FIBE REAL DATALOADER FORWARD/BACKWARD SMOKE: PASS")