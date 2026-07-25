#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROTOCOL = "FloodPSG frozen validation Water-HN GT-pair scorer v1"
EXPECTED_SPLIT = "validation"
DEFAULT_EXPECTED_IMAGES = 75
DEFAULT_EXPECTED_PAIRS = 322
REQUIRED_HN_COLUMNS = {
    "global_image_key",
    "audit_split_v2",
    "subject_index_v4",
    "object_index_v4",
    "pair_family_v4",
    "audit_pair_status",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load one formal DSFormer checkpoint and force-score every frozen "
            "validation Water-HN GT node pair. The result pickle uses the "
            "standard DSFormer inference structure, with canonical GT masks "
            "and labels so the strict evaluator has identity node matching. "
            "--help performs no data loading and creates no output."
        )
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Formal model directory containing config.json and best_state.pth.",
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        required=True,
        help="Canonical train/validation Coarse8 annotation JSON.",
    )
    parser.add_argument(
        "--hardneg",
        type=Path,
        required=True,
        help="Frozen validation Water-HN CSV.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="FloodPSG root used to resolve images and panoptic masks.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New result pickle path; existing paths are refused.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="New JSON audit-report path; existing paths are refused.",
    )
    parser.add_argument(
        "--split",
        choices=("validation",),
        default="validation",
        help="Development gate is locked to validation.",
    )
    parser.add_argument(
        "--expected-images",
        type=int,
        default=DEFAULT_EXPECTED_IMAGES,
    )
    parser.add_argument(
        "--expected-pairs",
        type=int,
        default=DEFAULT_EXPECTED_PAIRS,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--max-relations",
        type=int,
        default=512,
        help="Maximum relation rows processed per model chunk.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--expect-fibe",
        choices=("enabled", "disabled", "either"),
        default="either",
        help="Fail if the model configuration has an unexpected FIBE state.",
    )
    parser.add_argument(
        "--require-all-fibe-valid",
        action="store_true",
        help=(
            "For an enabled FIBE model, require every frozen pair to be marked "
            "valid in the validation feature cache."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_global_key(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_image_id(value: Any) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or not float(value).is_integer():
            raise ValueError(f"Invalid image_id: {value!r}")
        return int(value)
    text = str(value).strip()
    return int(text)


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    cpu = tensor.detach().cpu().contiguous()
    array = cpu.numpy()
    return np.ascontiguousarray(array).tobytes()


def update_tensor_hash(
    digest: "hashlib._Hash",
    name: str,
    tensor: torch.Tensor,
) -> None:
    cpu = tensor.detach().cpu().contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(cpu.dtype).encode("utf-8"))
    digest.update(b"\0")
    digest.update(json.dumps(list(cpu.shape)).encode("utf-8"))
    digest.update(b"\0")
    digest.update(tensor_bytes(cpu))
    digest.update(b"\0")


def atomic_pickle_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_json_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def torch_load_cpu(path: Path) -> Any:
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location="cpu")


def raw_gt_index_mask(
    dataset: Any,
    entry: dict[str, Any],
) -> np.ndarray:
    masks = dataset._load_seg(
        entry["pan_seg_file_name"],
        entry["segments_info"],
    )
    if masks.ndim != 3:
        raise RuntimeError(
            f"Image {entry['image_id']}: expected raw masks [N,H,W], "
            f"got {tuple(masks.shape)}"
        )
    mask_ids = masks.long().argmax(0)
    mask_ids[masks.sum(0) == 0] = -1
    return mask_ids.numpy().astype(np.int32, copy=False)


def make_frozen_hn_dataset_class(base_class: type) -> type:
    class FrozenHNPairDataset(base_class):
        def __init__(
            self,
            *,
            pairs_by_image_id: dict[int, torch.Tensor],
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self.pairs_by_image_id = pairs_by_image_id

        def __getitem__(self, idx: int) -> dict[str, Any]:
            sample = super().__getitem__(idx)
            image_id = int(sample["image_id"])
            if image_id not in self.pairs_by_image_id:
                raise KeyError(
                    f"No frozen HN pairs for image_id={image_id}"
                )
            pairs = (
                self.pairs_by_image_id[image_id]
                .clone()
                .to(torch.long)
            )
            if pairs.ndim != 2 or pairs.shape[1] != 2:
                raise RuntimeError(
                    f"Image {image_id}: invalid frozen pair tensor "
                    f"{tuple(pairs.shape)}"
                )
            target = torch.zeros(
                (len(pairs), 2 + len(self.rel_names)),
                dtype=torch.long,
            )
            target[:, :2] = pairs
            target[:, 2] = 1
            sample["sampled_relations"] = target
            return sample

    return FrozenHNPairDataset

def choose_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    return (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )


def main() -> None:
    args = parse_args()

    # Project imports are intentionally delayed until after argparse.
    # Therefore --help exits before importing model/data code and performs
    # no project data loading or output creation.
    from fair_psgg import from_config
    from fair_psgg.config import Config
    from fair_psgg.data.common import get_generic_loader
    from fair_psgg.data.data import SGDataset
    from fair_psgg.data.fibe_cache import build_fibe_cache
    from fair_psgg.trainer import prepare_batch

    FrozenHNPairDataset = make_frozen_hn_dataset_class(
        SGDataset
    )

    model_dir = args.model_dir.expanduser().resolve()
    annotation_path = args.annotation.expanduser().resolve()
    hardneg_path = args.hardneg.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    config_path = model_dir / "config.json"
    checkpoint_path = model_dir / "best_state.pth"

    for path in (
        model_dir,
        annotation_path,
        hardneg_path,
        data_root,
        config_path,
        checkpoint_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    if not model_dir.is_dir():
        raise NotADirectoryError(model_dir)
    if not data_root.is_dir():
        raise NotADirectoryError(data_root)
    for path in (
        annotation_path,
        hardneg_path,
        config_path,
        checkpoint_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    if output_path == report_path:
        raise ValueError("--output and --report must be different paths")
    for path in (output_path, report_path):
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing formal output: {path}"
            )

    if args.expected_images <= 0 or args.expected_pairs <= 0:
        raise ValueError("Expected counts must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.workers < 0:
        raise ValueError("--workers must be nonnegative")
    if args.max_relations <= 0:
        raise ValueError("--max-relations must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False

    hardneg = pd.read_csv(
        hardneg_path,
        encoding="utf-8-sig",
        low_memory=False,
    )
    missing_hn_columns = sorted(REQUIRED_HN_COLUMNS - set(hardneg.columns))
    if missing_hn_columns:
        raise KeyError(f"Frozen HN CSV missing columns: {missing_hn_columns}")

    hardneg = hardneg.copy()
    hardneg["global_image_key"] = hardneg["global_image_key"].map(
        normalize_global_key
    )
    if (hardneg["global_image_key"] == "").any():
        raise RuntimeError("Frozen HN CSV contains empty global_image_key values")

    split_values = sorted(
        hardneg["audit_split_v2"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )
    if split_values != [EXPECTED_SPLIT]:
        raise RuntimeError(
            f"Frozen HN split contamination: expected [{EXPECTED_SPLIT!r}], "
            f"got {split_values}"
        )

    hardneg["subject_index_v4"] = pd.to_numeric(
        hardneg["subject_index_v4"],
        errors="raise",
    ).astype(int)
    hardneg["object_index_v4"] = pd.to_numeric(
        hardneg["object_index_v4"],
        errors="raise",
    ).astype(int)

    pair_key_columns = [
        "global_image_key",
        "subject_index_v4",
        "object_index_v4",
    ]
    duplicate_pair_rows = int(
        hardneg.duplicated(
            subset=pair_key_columns,
            keep=False,
        ).sum()
    )
    if duplicate_pair_rows:
        raise RuntimeError(
            f"Frozen HN CSV contains {duplicate_pair_rows} duplicate pair rows"
        )

    frozen_images = int(hardneg["global_image_key"].nunique())
    frozen_pairs = int(len(hardneg))
    if frozen_images != args.expected_images:
        raise RuntimeError(
            f"Frozen image count mismatch: {frozen_images} != "
            f"{args.expected_images}"
        )
    if frozen_pairs != args.expected_pairs:
        raise RuntimeError(
            f"Frozen pair count mismatch: {frozen_pairs} != "
            f"{args.expected_pairs}"
        )

    config = Config.from_file(config_path)
    fibe_enabled = bool(config.fibe.enabled)
    if args.expect_fibe == "enabled" and not fibe_enabled:
        raise RuntimeError("Expected FIBE enabled, but model config disables it")
    if args.expect_fibe == "disabled" and fibe_enabled:
        raise RuntimeError("Expected FIBE disabled, but model config enables it")
    if args.require_all_fibe_valid and not fibe_enabled:
        raise RuntimeError(
            "--require-all-fibe-valid requires a FIBE-enabled model"
        )

    # load_psg_entries intentionally omits some validation images (for
    # example, entries with no positive relations).  The frozen HN gate must
    # instead use the complete authoritative validation membership encoded by
    # annotation.test_image_ids.  We still call the project loader once to
    # obtain its canonical node/predicate name ordering.
    loaded = from_config.get_data_entries(
        config,
        anno_path=annotation_path,
        split="val",
        img_dir=data_root,
    )
    if not isinstance(loaded, tuple) or len(loaded) != 3:
        raise RuntimeError(
            "from_config.get_data_entries did not return "
            "(entries, node_names, rel_names)"
        )
    loader_validation_entries, node_names, rel_names = loaded

    annotation_root = json.loads(
        annotation_path.read_text(encoding="utf-8")
    )
    annotation_entries = annotation_root.get("data")
    validation_ids_raw = annotation_root.get("test_image_ids")
    if not isinstance(annotation_entries, list):
        raise RuntimeError("Canonical annotation has no data list")
    if not isinstance(validation_ids_raw, list):
        raise RuntimeError(
            "Canonical annotation has no test_image_ids list"
        )

    validation_ids = {
        normalize_image_id(value)
        for value in validation_ids_raw
    }
    if len(validation_ids) != len(validation_ids_raw):
        raise RuntimeError(
            "Canonical annotation contains duplicate test_image_ids"
        )

    annotation_by_image_id: dict[int, dict[str, Any]] = {}
    for entry in annotation_entries:
        image_id = normalize_image_id(entry["image_id"])
        if image_id in annotation_by_image_id:
            raise RuntimeError(
                f"Duplicate canonical annotation image_id: {image_id}"
            )
        annotation_by_image_id[image_id] = entry

    missing_validation_ids = sorted(
        validation_ids - set(annotation_by_image_id)
    )
    if missing_validation_ids:
        raise RuntimeError(
            "Canonical test_image_ids absent from annotation data: "
            f"{missing_validation_ids[:20]}"
        )

    validation_entries = [
        annotation_by_image_id[image_id]
        for image_id in sorted(validation_ids)
    ]

    validation_by_global_key: dict[str, dict[str, Any]] = {}
    for entry in validation_entries:
        global_key = normalize_global_key(entry.get("global_image_key"))
        if not global_key:
            continue
        if global_key in validation_by_global_key:
            raise RuntimeError(
                f"Duplicate validation global_image_key: {global_key!r}"
            )
        validation_by_global_key[global_key] = entry

    requested_keys = hardneg["global_image_key"].drop_duplicates().tolist()
    missing_validation_keys = sorted(
        set(requested_keys) - set(validation_by_global_key)
    )
    if missing_validation_keys:
        raise RuntimeError(
            "Frozen HN images absent from canonical validation entries: "
            f"{missing_validation_keys[:20]}"
        )

    selected_entries = [
        validation_by_global_key[key]
        for key in requested_keys
    ]
    selected_entries.sort(key=lambda item: normalize_image_id(item["image_id"]))

    image_id_to_global_key: dict[int, str] = {}
    global_key_to_image_id: dict[str, int] = {}
    pairs_by_image_id: dict[int, torch.Tensor] = {}
    pair_order_by_image_id: dict[int, list[tuple[int, int]]] = {}

    for entry in selected_entries:
        image_id = normalize_image_id(entry["image_id"])
        global_key = normalize_global_key(entry["global_image_key"])
        if image_id in image_id_to_global_key:
            raise RuntimeError(f"Duplicate selected image_id: {image_id}")
        image_id_to_global_key[image_id] = global_key
        global_key_to_image_id[global_key] = image_id

        group = hardneg[
            hardneg["global_image_key"] == global_key
        ]
        pairs_list = [
            (int(row.subject_index_v4), int(row.object_index_v4))
            for row in group.itertuples(index=False)
        ]
        num_objects = len(entry["annotations"])
        for subject_index, object_index in pairs_list:
            if subject_index == object_index:
                raise RuntimeError(
                    f"Image {image_id}: diagonal frozen pair "
                    f"({subject_index}, {object_index})"
                )
            if (
                subject_index < 0
                or object_index < 0
                or subject_index >= num_objects
                or object_index >= num_objects
            ):
                raise IndexError(
                    f"Image {image_id}: frozen pair "
                    f"({subject_index}, {object_index}) outside "
                    f"[0, {num_objects - 1}]"
                )
        pair_order_by_image_id[image_id] = pairs_list
        pairs_by_image_id[image_id] = torch.tensor(
            pairs_list,
            dtype=torch.long,
        )

    dataset = FrozenHNPairDataset(
        pairs_by_image_id=pairs_by_image_id,
        is_train=False,
        entries=selected_entries,
        node_names=node_names,
        rel_names=rel_names,
        img_dir=data_root,
        seg_dir=data_root,
        augmentations=from_config.get_augmentations(
            config,
            split="test",
        ),
        neg_ratio=None,
    )
    loader = get_generic_loader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        is_train=False,
    )

    device = choose_device(args.device)
    model = from_config.get_model(
        config,
        num_node_outputs=len(node_names),
        num_rel_outputs=len(dataset.rel_names),
    )
    checkpoint = torch_load_cpu(checkpoint_path)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise KeyError(
            f"Checkpoint has no top-level 'model' state_dict: {checkpoint_path}"
        )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()

    fibe_cache = build_fibe_cache(
        config.fibe,
        split="validation",
    )

    # Reset all user-space RNGs after model construction. This keeps the
    # validation input stream identical between C and D1 even though D1 has
    # additional FIBE parameters during object construction.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_by_image_id: dict[int, dict[str, Any]] = {}
    input_digest = hashlib.sha256()
    total_scored_pairs = 0
    total_fibe_valid = 0
    total_fibe_invalid = 0
    score_columns = len(dataset.rel_names)

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            update_tensor_hash(
                input_digest,
                f"batch{batch_index}/image_id",
                batch["image_id"],
            )
            for key in (
                "img",
                "bboxes",
                "num_boxes",
                "box_categories",
                "segmentation",
                "sampled_relations",
                "num_relations",
            ):
                update_tensor_hash(
                    input_digest,
                    f"batch{batch_index}/{key}",
                    batch[key],
                )

            model_input, _, _, _ = prepare_batch(
                batch,
                device,
                fibe_cache=fibe_cache,
            )
            if fibe_enabled:
                valid = model_input["fibe_valid"].detach().cpu().to(torch.bool)
                total_fibe_valid += int(valid.sum().item())
                total_fibe_invalid += int((~valid).sum().item())

            _, _, relation_logits = model(
                model_input,
                max_relations=args.max_relations,
            )
            relation_scores = relation_logits.sigmoid().detach().cpu()
            if relation_scores.ndim != 2:
                raise RuntimeError(
                    f"Unexpected relation score shape: "
                    f"{tuple(relation_scores.shape)}"
                )
            if int(relation_scores.shape[1]) != score_columns:
                raise RuntimeError(
                    f"Expected {score_columns} score columns, got "
                    f"{int(relation_scores.shape[1])}"
                )
            if not torch.isfinite(relation_scores).all():
                raise FloatingPointError("Relation scores contain NaN/Inf")

            relation_positions = torch.repeat_interleave(
                torch.arange(
                    len(batch["image_id"]),
                    dtype=torch.long,
                ),
                batch["num_relations"].detach().cpu().to(torch.long),
            )
            batch_pairs = (
                batch["sampled_relations"][:, :2]
                .detach()
                .cpu()
                .to(torch.long)
            )

            for image_position, (
                image_id_tensor,
                dataset_index_tensor,
            ) in enumerate(zip(batch["image_id"], batch["idx"])):
                image_id = int(image_id_tensor.item())
                dataset_index = int(dataset_index_tensor.item())
                if image_id in output_by_image_id:
                    raise RuntimeError(
                        f"Duplicate scored output for image_id={image_id}"
                    )
                entry = dataset.entries[dataset_index]
                global_key = normalize_global_key(entry["global_image_key"])
                row_mask = relation_positions == image_position
                local_pairs = batch_pairs[row_mask]
                local_scores = relation_scores[row_mask]

                expected_pairs_list = pair_order_by_image_id[image_id]
                actual_pairs_list = [
                    (int(pair[0]), int(pair[1]))
                    for pair in local_pairs.tolist()
                ]
                if actual_pairs_list != expected_pairs_list:
                    raise RuntimeError(
                        f"Image {image_id}: scored pair order/content differs "
                        "from frozen HN CSV"
                    )

                raw_labels = np.asarray(
                    [
                        int(annotation["category_id"])
                        for annotation in entry["annotations"]
                    ],
                    dtype=np.int64,
                )
                raw_bboxes = np.asarray(
                    [
                        annotation["bbox"]
                        for annotation in entry["annotations"]
                    ],
                    dtype=np.float32,
                )
                raw_mask = raw_gt_index_mask(dataset, entry)

                output_by_image_id[image_id] = {
                    "img_id": str(image_id),
                    "global_image_key": global_key,
                    "bboxes": raw_bboxes,
                    "mask": raw_mask,
                    "box_label": raw_labels,
                    "pairs": local_pairs.numpy().astype(
                        np.int64,
                        copy=False,
                    ),
                    "rel_scores": local_scores.numpy().astype(
                        np.float32,
                        copy=False,
                    ),
                    "rel_rank": (
                        1.0 - local_scores[:, 0]
                    ).numpy().astype(
                        np.float32,
                        copy=False,
                    ),
                }
                total_scored_pairs += int(len(local_pairs))

    output_data = [
        output_by_image_id[image_id]
        for image_id in sorted(output_by_image_id)
    ]

    output_images = len(output_data)
    output_pairs = sum(len(item["pairs"]) for item in output_data)
    output_pair_keys: list[tuple[str, int, int]] = []
    nonfinite_scores = 0
    malformed_score_rows = 0

    for item in output_data:
        scores = np.asarray(item["rel_scores"])
        pairs = np.asarray(item["pairs"])
        if scores.shape != (len(pairs), score_columns):
            malformed_score_rows += len(pairs)
        nonfinite_scores += int((~np.isfinite(scores)).sum())
        global_key = str(item["global_image_key"])
        for subject_index, object_index in pairs.tolist():
            output_pair_keys.append(
                (
                    global_key,
                    int(subject_index),
                    int(object_index),
                )
            )

    expected_pair_keys = [
        (
            str(row.global_image_key),
            int(row.subject_index_v4),
            int(row.object_index_v4),
        )
        for row in hardneg.itertuples(index=False)
    ]
    duplicate_output_pairs = (
        len(output_pair_keys) - len(set(output_pair_keys))
    )
    missing_output_pairs = sorted(
        set(expected_pair_keys) - set(output_pair_keys)
    )
    unexpected_output_pairs = sorted(
        set(output_pair_keys) - set(expected_pair_keys)
    )

    checks = {
        "split_is_validation": args.split == EXPECTED_SPLIT,
        "frozen_images_match": frozen_images == args.expected_images,
        "frozen_pairs_match": frozen_pairs == args.expected_pairs,
        "frozen_pair_duplicates_zero": duplicate_pair_rows == 0,
        "validation_entries_complete": (
            len(selected_entries) == args.expected_images
        ),
        "output_images_complete": output_images == args.expected_images,
        "output_pairs_complete": output_pairs == args.expected_pairs,
        "scored_pair_counter_complete": (
            total_scored_pairs == args.expected_pairs
        ),
        "output_pair_duplicates_zero": duplicate_output_pairs == 0,
        "missing_output_pairs_zero": len(missing_output_pairs) == 0,
        "unexpected_output_pairs_zero": len(unexpected_output_pairs) == 0,
        "score_column_count_is_nine": score_columns == 9,
        "malformed_score_rows_zero": malformed_score_rows == 0,
        "nonfinite_scores_zero": nonfinite_scores == 0,
        "fibe_expectation_matches": (
            args.expect_fibe == "either"
            or (
                args.expect_fibe == "enabled"
                and fibe_enabled
            )
            or (
                args.expect_fibe == "disabled"
                and not fibe_enabled
            )
        ),
        "all_fibe_pairs_valid_when_required": (
            not args.require_all_fibe_valid
            or (
                fibe_enabled
                and total_fibe_valid == args.expected_pairs
                and total_fibe_invalid == 0
            )
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"

    report: dict[str, Any] = {
        "protocol": PROTOCOL,
        "status": status,
        "split": args.split,
        "final_test_allowed": False,
        "model_dir": str(model_dir),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "annotation": str(annotation_path),
        "hardneg": str(hardneg_path),
        "data_root": str(data_root),
        "output": str(output_path),
        "report": str(report_path),
        "device": str(device),
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "workers": int(args.workers),
        "max_relations": int(args.max_relations),
        "fibe": {
            "enabled": fibe_enabled,
            "expected": args.expect_fibe,
            "require_all_valid": bool(
                args.require_all_fibe_valid
            ),
            "cache_path": (
                str(fibe_cache.path)
                if fibe_cache is not None
                else None
            ),
            "selected_valid_pairs": int(total_fibe_valid),
            "selected_invalid_pairs": int(total_fibe_invalid),
        },
        "entry_selection": {
            "authoritative_source": "annotation.test_image_ids",
            "loader_validation_entries": len(
                loader_validation_entries
            ),
            "authoritative_validation_entries": len(
                validation_entries
            ),
            "loader_omitted_authoritative_entries": (
                len(validation_entries)
                - len(loader_validation_entries)
            ),
        },
        "counts": {
            "frozen_images": frozen_images,
            "frozen_pairs": frozen_pairs,
            "selected_validation_entries": len(selected_entries),
            "output_images": output_images,
            "output_pairs": output_pairs,
            "score_columns": score_columns,
            "duplicate_frozen_pair_rows": duplicate_pair_rows,
            "duplicate_output_pairs": duplicate_output_pairs,
            "missing_output_pairs": len(missing_output_pairs),
            "unexpected_output_pairs": len(unexpected_output_pairs),
            "malformed_score_rows": malformed_score_rows,
            "nonfinite_score_values": nonfinite_scores,
        },
        "input_semantic_sha256": input_digest.hexdigest(),
        "sha256": {
            "config": sha256_file(config_path),
            "checkpoint": sha256_file(checkpoint_path),
            "annotation": sha256_file(annotation_path),
            "hardneg": sha256_file(hardneg_path),
        },
        "checks": checks,
        "first_missing_output_pairs": missing_output_pairs[:20],
        "first_unexpected_output_pairs": unexpected_output_pairs[:20],
    }

    if status != "PASS":
        print(
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        print()
        print("FROZEN VALIDATION HN GT-PAIR SCORER V1: FAIL")
        raise SystemExit(1)

    atomic_pickle_dump(output_data, output_path)
    report["sha256"]["output_pickle"] = sha256_file(output_path)
    atomic_json_dump(report, report_path)

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    print()
    print("FROZEN VALIDATION HN GT-PAIR SCORER V1: PASS")


if __name__ == "__main__":
    main()
