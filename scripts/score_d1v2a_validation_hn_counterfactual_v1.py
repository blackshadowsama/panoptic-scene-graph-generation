#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROTOCOL = "FloodPSG D1-v2a validation Water-HN counterfactual scorer V1"
EXPECTED_SPLIT = "validation"
DEFAULT_EXPECTED_IMAGES = 75
DEFAULT_EXPECTED_PAIRS = 322
MODES = (
    "normal",
    "gate_bypass",
    "raw_feature_zero",
    "family_shuffle",
)
REQUIRED_HN_COLUMNS = {
    "global_image_key",
    "audit_split_v2",
    "subject_index_v4",
    "object_index_v4",
    "pair_family_v4",
    "audit_pair_status",
}
PAIR_KEY_COLUMNS = [
    "global_image_key",
    "subject_index_v4",
    "object_index_v4",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score the frozen validation Water-HN pairs with the formal D1-v2a "
            "Geometry-only Relation-Existence Gate checkpoint. Conditions are "
            "normal, exact gate bypass, exact raw-21D zeroing, and deterministic "
            "pair-family-preserving feature shuffle. The script never loads the "
            "final-test split."
        )
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--hardneg", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--pair-audit", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--split", choices=("validation",), default="validation")
    parser.add_argument("--expected-images", type=int, default=DEFAULT_EXPECTED_IMAGES)
    parser.add_argument("--expected-pairs", type=int, default=DEFAULT_EXPECTED_PAIRS)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-relations", type=int, default=512)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--require-all-fibe-valid", action="store_true")
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
    return int(str(value).strip())


def normalize_family(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def safe_column_name(value: Any) -> str:
    text = re.sub(r"[^0-9A-Za-z_]+", "_", str(value).strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unnamed"


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    array = tensor.detach().cpu().contiguous().numpy()
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
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
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
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_csv_dump(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def torch_load_cpu(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def choose_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def raw_gt_index_mask(dataset: Any, entry: dict[str, Any]) -> np.ndarray:
    masks = dataset._load_seg(entry["pan_seg_file_name"], entry["segments_info"])
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
                raise KeyError(f"No frozen HN pairs for image_id={image_id}")
            pairs = self.pairs_by_image_id[image_id].clone().to(torch.long)
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


class GateActivationRecorder:
    """Capture the trained existence gate and optionally return exact base logits."""

    def __init__(self, gate: torch.nn.Module, mode: str):
        if mode not in MODES:
            raise ValueError(mode)
        self.gate = gate
        self.mode = mode
        self._chunks: dict[str, list[torch.Tensor]] = defaultdict(list)
        self._handles = [
            gate.encoder.register_forward_hook(self._encoder_hook),
            gate.delta_head.register_forward_hook(self._delta_head_hook),
            gate.register_forward_hook(self._gate_hook, with_kwargs=True),
        ]

    @staticmethod
    def _detach_cpu(value: torch.Tensor) -> torch.Tensor:
        return value.detach().cpu().contiguous()

    def _encoder_hook(self, module: Any, inputs: Any, output: torch.Tensor) -> None:
        del module, inputs
        self._chunks["hidden"].append(self._detach_cpu(output))

    def _delta_head_hook(
        self,
        module: Any,
        inputs: Any,
        output: torch.Tensor,
    ) -> None:
        del module, inputs
        self._chunks["raw_delta"].append(self._detach_cpu(output))

    def _gate_hook(
        self,
        module: Any,
        inputs: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: torch.Tensor,
    ) -> torch.Tensor | None:
        if kwargs:
            base_logits = kwargs["base_logits"]
            fibe_features = kwargs["fibe_features"]
            fibe_valid = kwargs["fibe_valid"]
        else:
            if len(inputs) != 3:
                raise RuntimeError(
                    f"Unexpected existence-gate hook inputs: {len(inputs)}"
                )
            base_logits, fibe_features, fibe_valid = inputs

        gate_final_logits = output
        margin_delta = base_logits[:, :1] - gate_final_logits[:, :1]

        self._chunks["base_logits"].append(self._detach_cpu(base_logits))
        self._chunks["features"].append(self._detach_cpu(fibe_features))
        self._chunks["valid"].append(
            self._detach_cpu(fibe_valid.to(dtype=torch.bool))
        )
        self._chunks["gate_final_logits"].append(
            self._detach_cpu(gate_final_logits)
        )
        self._chunks["margin_delta"].append(self._detach_cpu(margin_delta))

        if self.mode == "gate_bypass":
            return base_logits
        return None

    def consume(self) -> dict[str, torch.Tensor]:
        required = {
            "hidden",
            "raw_delta",
            "base_logits",
            "features",
            "valid",
            "gate_final_logits",
            "margin_delta",
        }
        missing = sorted(required - set(self._chunks))
        if missing:
            raise RuntimeError(f"Missing existence-gate hook tensors: {missing}")

        result = {
            key: torch.cat(self._chunks[key], dim=0)
            for key in sorted(required)
        }
        self._chunks.clear()
        return result

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def descriptive_stats(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(array) == 0:
        return {"count": 0}
    if not np.isfinite(array).all():
        raise FloatingPointError("Nonfinite values in descriptive statistics")
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=0)),
        "min": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def pair_key(
    global_key: str,
    subject_index: int,
    object_index: int,
) -> tuple[str, int, int]:
    return (str(global_key), int(subject_index), int(object_index))


def build_family_shuffle_plan(
    hardneg: pd.DataFrame,
    feature_by_pair: dict[tuple[str, int, int], torch.Tensor],
    seed: int,
) -> tuple[
    dict[tuple[str, int, int], tuple[str, int, int]],
    dict[str, Any],
]:
    groups: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for row in hardneg.itertuples(index=False):
        key = pair_key(
            row.global_image_key,
            row.subject_index_v4,
            row.object_index_v4,
        )
        groups[normalize_family(row.pair_family_v4)].append(key)

    donor_by_target: dict[
        tuple[str, int, int],
        tuple[str, int, int],
    ] = {}
    family_counts: dict[str, int] = {}
    rng = random.Random(int(seed))

    for family in sorted(groups):
        keys = sorted(groups[family])
        if len(keys) < 2:
            raise RuntimeError(
                f"Cannot derange pair family {family!r} with {len(keys)} pair"
            )
        rng.shuffle(keys)
        donors = keys[1:] + keys[:1]
        for target, donor in zip(keys, donors):
            if target == donor:
                raise RuntimeError(f"Shuffle fixed point in family {family!r}")
            donor_by_target[target] = donor
        family_counts[family] = len(keys)

    if set(donor_by_target) != set(feature_by_pair):
        raise RuntimeError("Shuffle plan pair set differs from frozen feature set")

    source_counts: dict[tuple[str, int, int], int] = defaultdict(int)
    for donor in donor_by_target.values():
        source_counts[donor] += 1
    if any(count != 1 for count in source_counts.values()):
        raise RuntimeError("Family shuffle is not a bijection")

    cross_image = sum(
        int(target[0] != donor[0])
        for target, donor in donor_by_target.items()
    )
    summary = {
        "strategy":
            "deterministic_global_cyclic_derangement_within_pair_family",
        "seed": int(seed),
        "families": family_counts,
        "pairs": int(len(donor_by_target)),
        "fixed_points": 0,
        "same_image_donors": int(len(donor_by_target) - cross_image),
        "cross_image_donors": int(cross_image),
        "bijection": True,
    }
    return donor_by_target, summary


def main() -> None:
    args = parse_args()

    # Delayed imports keep --help side-effect free.
    from fair_psgg import from_config
    from fair_psgg.config import Config
    from fair_psgg.data.common import get_generic_loader
    from fair_psgg.data.data import SGDataset
    from fair_psgg.data.fibe_cache import build_fibe_cache
    from fair_psgg.trainer import prepare_batch

    FrozenHNPairDataset = make_frozen_hn_dataset_class(SGDataset)

    model_dir = args.model_dir.expanduser().resolve()
    annotation_path = args.annotation.expanduser().resolve()
    hardneg_path = args.hardneg.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    pair_audit_path = args.pair_audit.expanduser().resolve()
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
    if not model_dir.is_dir() or not data_root.is_dir():
        raise NotADirectoryError(
            model_dir if not model_dir.is_dir() else data_root
        )
    for path in (annotation_path, hardneg_path, config_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    output_paths = (output_path, report_path, pair_audit_path)
    if len(set(output_paths)) != len(output_paths):
        raise ValueError("--output, --report and --pair-audit must differ")
    for path in output_paths:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite output: {path}")

    if args.expected_images <= 0 or args.expected_pairs <= 0:
        raise ValueError("Expected counts must be positive")
    if args.batch_size <= 0 or args.workers < 0 or args.max_relations <= 0:
        raise ValueError("Invalid loader/max-relations arguments")
    if args.split != EXPECTED_SPLIT:
        raise RuntimeError("Scoring is locked to validation")

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
    hardneg["pair_family_v4"] = hardneg["pair_family_v4"].map(
        normalize_family
    )
    hardneg["subject_index_v4"] = hardneg["subject_index_v4"].astype(int)
    hardneg["object_index_v4"] = hardneg["object_index_v4"].astype(int)

    split_values = set(
        hardneg["audit_split_v2"].fillna("").astype(str).str.strip()
    )
    if split_values != {EXPECTED_SPLIT}:
        raise RuntimeError(f"Frozen HN split mismatch: {sorted(split_values)}")
    status_values = set(
        hardneg["audit_pair_status"].fillna("").astype(str).str.strip()
    )
    if status_values != {"verified_negative"}:
        raise RuntimeError(f"Frozen HN status mismatch: {sorted(status_values)}")

    duplicate_pair_rows = int(
        hardneg.duplicated(subset=PAIR_KEY_COLUMNS, keep=False).sum()
    )
    if duplicate_pair_rows:
        raise RuntimeError(
            f"Frozen HN CSV contains {duplicate_pair_rows} duplicate pair rows"
        )
    frozen_images = int(hardneg["global_image_key"].nunique())
    frozen_pairs = int(len(hardneg))
    if frozen_images != args.expected_images or frozen_pairs != args.expected_pairs:
        raise RuntimeError(
            f"Frozen count mismatch: images={frozen_images}, pairs={frozen_pairs}"
        )

    config = Config.from_file(config_path)
    if not bool(config.fibe.enabled):
        raise RuntimeError("Scorer requires an enabled FIBE config")
    if str(config.fibe.mode) != "geometry_existence_gate":
        raise RuntimeError(f"Unexpected FIBE mode: {config.fibe.mode}")
    if config.fibe.test_cache is not None:
        raise RuntimeError("Final-test cache must remain absent")

    loaded = from_config.get_data_entries(
        config,
        anno_path=annotation_path,
        split="val",
        img_dir=data_root,
    )
    if not isinstance(loaded, tuple) or len(loaded) != 3:
        raise RuntimeError(
            "get_data_entries did not return entries/node_names/rel_names"
        )
    loader_validation_entries, node_names, rel_names = loaded

    annotation_root = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotation_entries = annotation_root.get("data")
    validation_ids_raw = annotation_root.get("test_image_ids")
    if not isinstance(annotation_entries, list):
        raise RuntimeError("Canonical annotation has no data list")
    if not isinstance(validation_ids_raw, list):
        raise RuntimeError("Canonical annotation has no test_image_ids list")

    validation_ids = {normalize_image_id(value) for value in validation_ids_raw}
    if len(validation_ids) != len(validation_ids_raw):
        raise RuntimeError("Duplicate canonical validation image ids")

    annotation_by_image_id: dict[int, dict[str, Any]] = {}
    for entry in annotation_entries:
        image_id = normalize_image_id(entry["image_id"])
        if image_id in annotation_by_image_id:
            raise RuntimeError(f"Duplicate annotation image_id: {image_id}")
        annotation_by_image_id[image_id] = entry

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
            raise RuntimeError(f"Duplicate validation global key: {global_key}")
        validation_by_global_key[global_key] = entry

    requested_keys = hardneg["global_image_key"].drop_duplicates().tolist()
    missing_validation_keys = sorted(
        set(requested_keys) - set(validation_by_global_key)
    )
    if missing_validation_keys:
        raise RuntimeError(
            "Frozen HN images absent from authoritative validation: "
            f"{missing_validation_keys[:20]}"
        )

    selected_entries = [validation_by_global_key[key] for key in requested_keys]
    selected_entries.sort(key=lambda item: normalize_image_id(item["image_id"]))

    image_id_to_global_key: dict[int, str] = {}
    global_key_to_image_id: dict[str, int] = {}
    pairs_by_image_id: dict[int, torch.Tensor] = {}
    pair_order_by_image_id: dict[int, list[tuple[int, int]]] = {}
    hardneg_row_by_pair: dict[tuple[str, int, int], dict[str, Any]] = {}

    for row in hardneg.to_dict(orient="records"):
        key = pair_key(
            row["global_image_key"],
            row["subject_index_v4"],
            row["object_index_v4"],
        )
        hardneg_row_by_pair[key] = row

    for entry in selected_entries:
        image_id = normalize_image_id(entry["image_id"])
        global_key = normalize_global_key(entry["global_image_key"])
        image_id_to_global_key[image_id] = global_key
        global_key_to_image_id[global_key] = image_id
        group = hardneg[hardneg["global_image_key"] == global_key]
        pairs_list = [
            (int(row.subject_index_v4), int(row.object_index_v4))
            for row in group.itertuples(index=False)
        ]
        num_objects = len(entry["annotations"])
        for subject_index, object_index in pairs_list:
            if subject_index == object_index:
                raise RuntimeError(
                    f"Image {image_id}: diagonal pair "
                    f"({subject_index}, {object_index})"
                )
            if not (
                0 <= subject_index < num_objects
                and 0 <= object_index < num_objects
            ):
                raise IndexError(
                    f"Image {image_id}: pair ({subject_index}, {object_index}) "
                    f"outside [0, {num_objects - 1}]"
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
        augmentations=from_config.get_augmentations(config, split="test"),
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
        raise KeyError(f"Checkpoint has no model state: {checkpoint_path}")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()

    if model.fibe_branch is not None:
        raise RuntimeError("D1-v2a must not contain token-residual FIBE branch")
    if model.fibe_existence_gate is None:
        raise RuntimeError("Loaded model has no geometry existence gate")

    gate = model.fibe_existence_gate
    fibe_cache = build_fibe_cache(config.fibe, split="validation")
    if fibe_cache is None:
        raise RuntimeError("Validation FIBE cache did not build")

    feature_by_pair: dict[tuple[str, int, int], torch.Tensor] = {}
    valid_by_pair: dict[tuple[str, int, int], bool] = {}
    for row in hardneg.itertuples(index=False):
        global_key = str(row.global_image_key)
        image_id = global_key_to_image_id[global_key]
        subject_index = int(row.subject_index_v4)
        object_index = int(row.object_index_v4)
        entry = fibe_cache._get_entry(image_id)
        features = torch.as_tensor(entry["features"], dtype=torch.float32)
        valid_pairs = torch.as_tensor(entry["valid_pairs"], dtype=torch.bool)
        key = pair_key(global_key, subject_index, object_index)
        feature = features[subject_index, object_index].clone()
        valid = bool(valid_pairs[subject_index, object_index].item())
        if not torch.isfinite(feature).all():
            raise FloatingPointError(f"Nonfinite cached feature for {key}")
        if not valid:
            feature.zero_()
        feature_by_pair[key] = feature
        valid_by_pair[key] = valid

    valid_pair_count = sum(valid_by_pair.values())
    invalid_pair_count = len(valid_by_pair) - valid_pair_count
    if args.require_all_fibe_valid and invalid_pair_count:
        raise RuntimeError(
            f"Expected all FIBE pairs valid, found {invalid_pair_count} invalid"
        )

    donor_by_target: dict[
        tuple[str, int, int],
        tuple[str, int, int],
    ] | None = None
    shuffle_summary: dict[str, Any] | None = None
    if args.mode == "family_shuffle":
        donor_by_target, shuffle_summary = build_family_shuffle_plan(
            hardneg=hardneg,
            feature_by_pair=feature_by_pair,
            seed=args.seed,
        )

    recorder = GateActivationRecorder(gate, mode=args.mode)

    output_by_image_id: dict[int, dict[str, Any]] = {}
    pair_audit_rows: list[dict[str, Any]] = []
    input_digest = hashlib.sha256()
    mode_feature_digest = hashlib.sha256()
    total_scored_pairs = 0
    total_fibe_valid = 0
    total_fibe_invalid = 0
    invalid_control_base_chunks: list[torch.Tensor] = []
    invalid_control_feature_chunks: list[torch.Tensor] = []
    score_columns = len(dataset.rel_names)
    rel_column_names = [safe_column_name(name) for name in dataset.rel_names]

    try:
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
                batch_pairs = (
                    batch["sampled_relations"][:, :2]
                    .detach()
                    .cpu()
                    .to(torch.long)
                )
                relation_positions = torch.repeat_interleave(
                    torch.arange(len(batch["image_id"]), dtype=torch.long),
                    batch["num_relations"].detach().cpu().to(torch.long),
                )

                target_keys: list[tuple[str, int, int]] = []
                for row_index, pair in enumerate(batch_pairs.tolist()):
                    image_position = int(relation_positions[row_index].item())
                    image_id = int(batch["image_id"][image_position].item())
                    global_key = image_id_to_global_key[image_id]
                    target_keys.append(
                        pair_key(global_key, int(pair[0]), int(pair[1]))
                    )

                donor_keys: list[tuple[str, int, int] | None] = [
                    None for _ in target_keys
                ]
                if args.mode == "raw_feature_zero":
                    model_input["fibe_features"] = torch.zeros_like(
                        model_input["fibe_features"]
                    )
                elif args.mode == "family_shuffle":
                    if donor_by_target is None:
                        raise RuntimeError("Missing shuffle plan")
                    donor_keys = [donor_by_target[key] for key in target_keys]
                    model_input["fibe_features"] = torch.stack(
                        [feature_by_pair[key] for key in donor_keys],
                        dim=0,
                    ).to(device=device, non_blocking=True)
                    model_input["fibe_valid"] = torch.tensor(
                        [valid_by_pair[key] for key in donor_keys],
                        dtype=torch.bool,
                        device=device,
                    )

                update_tensor_hash(
                    mode_feature_digest,
                    f"batch{batch_index}/fibe_features",
                    model_input["fibe_features"],
                )
                update_tensor_hash(
                    mode_feature_digest,
                    f"batch{batch_index}/fibe_valid",
                    model_input["fibe_valid"],
                )

                valid = model_input["fibe_valid"].detach().cpu().to(torch.bool)
                total_fibe_valid += int(valid.sum().item())
                total_fibe_invalid += int((~valid).sum().item())

                _, _, relation_logits = model(
                    model_input,
                    max_relations=args.max_relations,
                )
                relation_scores = relation_logits.sigmoid().detach().cpu()
                if relation_scores.shape != (
                    len(batch_pairs),
                    score_columns,
                ):
                    raise RuntimeError(
                        f"Unexpected relation score shape: "
                        f"{tuple(relation_scores.shape)}"
                    )
                if not torch.isfinite(relation_scores).all():
                    raise FloatingPointError("Relation scores contain NaN/Inf")

                activation = recorder.consume()
                relation_count = len(batch_pairs)
                for key, tensor in activation.items():
                    if int(tensor.shape[0]) != relation_count:
                        raise RuntimeError(
                            f"Activation row mismatch for {key}: "
                            f"{tensor.shape[0]} != {relation_count}"
                        )

                base_logits = activation["base_logits"]
                gate_final_logits = activation["gate_final_logits"]
                mode_logits = relation_logits.detach().cpu()
                margin_delta = activation["margin_delta"]
                features_used = activation["features"]
                valid_used = activation["valid"].to(torch.bool)
                hidden = activation["hidden"]
                raw_delta = activation["raw_delta"]

                if args.mode == "gate_bypass":
                    if not torch.equal(mode_logits, base_logits):
                        raise RuntimeError("Gate bypass did not return exact base logits")
                else:
                    if not torch.equal(mode_logits, gate_final_logits):
                        raise RuntimeError(
                            f"Mode {args.mode} changed gate output unexpectedly"
                        )

                if not torch.equal(
                    base_logits[:, 1:],
                    gate_final_logits[:, 1:],
                ):
                    raise RuntimeError("Existence gate changed positive logits")
                if not torch.equal(
                    torch.argsort(base_logits[:, 1:], dim=1),
                    torch.argsort(gate_final_logits[:, 1:], dim=1),
                ):
                    raise RuntimeError("Existence gate changed positive ordering")

                # Save the same real inputs for a post-hook invalid-control audit.
                # Running compute_margin_delta while recorder hooks are active would
                # contaminate the next recorder.consume() call.
                invalid_control_base_chunks.append(base_logits.clone())
                invalid_control_feature_chunks.append(
                    model_input["fibe_features"].detach().cpu().contiguous()
                )

                base_none = base_logits[:, 0]
                base_max_positive, base_argmax_positive = (
                    base_logits[:, 1:].max(dim=1)
                )
                base_argmax_positive = base_argmax_positive + 1
                gate_none = gate_final_logits[:, 0]
                gate_max_positive, gate_argmax_positive = (
                    gate_final_logits[:, 1:].max(dim=1)
                )
                gate_argmax_positive = gate_argmax_positive + 1
                mode_none = mode_logits[:, 0]
                mode_max_positive, mode_argmax_positive = (
                    mode_logits[:, 1:].max(dim=1)
                )
                mode_argmax_positive = mode_argmax_positive + 1

                base_margin = base_max_positive - base_none
                gate_margin = gate_max_positive - gate_none
                mode_margin = mode_max_positive - mode_none

                feature_norm = torch.linalg.vector_norm(features_used, dim=1)
                hidden_norm = torch.linalg.vector_norm(hidden, dim=1)
                delta_abs = margin_delta.reshape(-1).abs()

                for row_index, key in enumerate(target_keys):
                    global_key, subject_index, object_index = key
                    image_id = global_key_to_image_id[global_key]
                    entry = annotation_by_image_id[image_id]
                    annotations = entry["annotations"]
                    subject_category_id = int(
                        annotations[subject_index]["category_id"]
                    )
                    object_category_id = int(
                        annotations[object_index]["category_id"]
                    )
                    metadata = hardneg_row_by_pair[key]
                    donor_key = donor_keys[row_index]

                    row: dict[str, Any] = {
                        "mode": args.mode,
                        "image_id": image_id,
                        "global_image_key": global_key,
                        "subject_index": subject_index,
                        "object_index": object_index,
                        "subject_category_id": subject_category_id,
                        "object_category_id": object_category_id,
                        "subject_category_name": (
                            node_names[subject_category_id]
                            if 0 <= subject_category_id < len(node_names)
                            else str(subject_category_id)
                        ),
                        "object_category_name": (
                            node_names[object_category_id]
                            if 0 <= object_category_id < len(node_names)
                            else str(object_category_id)
                        ),
                        "pair_family_v4": metadata.get("pair_family_v4", ""),
                        "scene_context_v4": metadata.get("scene_context_v4", ""),
                        "fibe_valid": bool(valid_used[row_index].item()),
                        "donor_global_image_key": (
                            donor_key[0] if donor_key is not None else ""
                        ),
                        "donor_subject_index": (
                            donor_key[1] if donor_key is not None else -1
                        ),
                        "donor_object_index": (
                            donor_key[2] if donor_key is not None else -1
                        ),
                        "donor_same_image": (
                            bool(donor_key[0] == global_key)
                            if donor_key is not None
                            else False
                        ),
                        "feature_norm": float(feature_norm[row_index].item()),
                        "hidden_norm": float(hidden_norm[row_index].item()),
                        "raw_delta": float(raw_delta[row_index, 0].item()),
                        "margin_delta": float(margin_delta[row_index, 0].item()),
                        "margin_delta_abs": float(delta_abs[row_index].item()),
                        "margin_delta_saturated_abs_ge_3_9": bool(
                            delta_abs[row_index].item() >= 3.9
                        ),
                        "base_none_logit": float(base_none[row_index].item()),
                        "base_max_positive_logit": float(
                            base_max_positive[row_index].item()
                        ),
                        "base_argmax_positive_index": int(
                            base_argmax_positive[row_index].item()
                        ),
                        "base_argmax_positive_name": dataset.rel_names[
                            int(base_argmax_positive[row_index].item())
                        ],
                        "base_existence_margin": float(
                            base_margin[row_index].item()
                        ),
                        "gate_final_none_logit": float(
                            gate_none[row_index].item()
                        ),
                        "gate_final_max_positive_logit": float(
                            gate_max_positive[row_index].item()
                        ),
                        "gate_final_argmax_positive_index": int(
                            gate_argmax_positive[row_index].item()
                        ),
                        "gate_final_argmax_positive_name": dataset.rel_names[
                            int(gate_argmax_positive[row_index].item())
                        ],
                        "gate_final_existence_margin": float(
                            gate_margin[row_index].item()
                        ),
                        "mode_final_none_logit": float(
                            mode_none[row_index].item()
                        ),
                        "mode_final_max_positive_logit": float(
                            mode_max_positive[row_index].item()
                        ),
                        "mode_final_argmax_positive_index": int(
                            mode_argmax_positive[row_index].item()
                        ),
                        "mode_final_argmax_positive_name": dataset.rel_names[
                            int(mode_argmax_positive[row_index].item())
                        ],
                        "mode_final_existence_margin": float(
                            mode_margin[row_index].item()
                        ),
                        "gate_margin_change": float(
                            (gate_margin - base_margin)[row_index].item()
                        ),
                        "mode_margin_change": float(
                            (mode_margin - base_margin)[row_index].item()
                        ),
                    }
                    for relation_index, column_name in enumerate(rel_column_names):
                        row[f"base_logit__{column_name}"] = float(
                            base_logits[row_index, relation_index].item()
                        )
                        row[f"gate_final_logit__{column_name}"] = float(
                            gate_final_logits[row_index, relation_index].item()
                        )
                        row[f"mode_final_logit__{column_name}"] = float(
                            mode_logits[row_index, relation_index].item()
                        )
                        row[f"mode_final_score__{column_name}"] = float(
                            relation_scores[row_index, relation_index].item()
                        )
                    pair_audit_rows.append(row)

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
                            f"Image {image_id}: pair order/content mismatch"
                        )
                    raw_labels = np.asarray(
                        [
                            int(annotation["category_id"])
                            for annotation in entry["annotations"]
                        ],
                        dtype=np.int64,
                    )
                    raw_bboxes = np.asarray(
                        [annotation["bbox"] for annotation in entry["annotations"]],
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
                        ).numpy().astype(np.float32, copy=False),
                    }
                    total_scored_pairs += int(len(local_pairs))
    finally:
        recorder.close()

    invalid_control_base = torch.cat(
        invalid_control_base_chunks,
        dim=0,
    )
    invalid_control_features = torch.cat(
        invalid_control_feature_chunks,
        dim=0,
    ).to(device=device)
    invalid_control_valid = torch.zeros(
        invalid_control_features.shape[0],
        dtype=torch.bool,
        device=device,
    )
    with torch.inference_mode():
        invalid_control_delta = gate.compute_margin_delta(
            fibe_features=invalid_control_features,
            fibe_valid=invalid_control_valid,
            output_dtype=invalid_control_base.dtype,
            output_device=device,
        )
        invalid_control_output = gate.apply_margin_delta(
            invalid_control_base.to(device=device),
            invalid_control_delta,
        ).detach().cpu()

    invalid_control_rows = int(invalid_control_base.shape[0])
    invalid_control_delta_nonzero = int(
        torch.count_nonzero(invalid_control_delta).item()
    )
    invalid_control_output_mismatches = int(
        not torch.equal(invalid_control_output, invalid_control_base)
    )

    output_data = [
        output_by_image_id[image_id]
        for image_id in sorted(output_by_image_id)
    ]
    pair_audit = pd.DataFrame(pair_audit_rows)

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
                pair_key(global_key, subject_index, object_index)
            )

    expected_pair_keys = [
        pair_key(
            row.global_image_key,
            row.subject_index_v4,
            row.object_index_v4,
        )
        for row in hardneg.itertuples(index=False)
    ]
    duplicate_output_pairs = len(output_pair_keys) - len(set(output_pair_keys))
    missing_output_pairs = sorted(
        set(expected_pair_keys) - set(output_pair_keys)
    )
    unexpected_output_pairs = sorted(
        set(output_pair_keys) - set(expected_pair_keys)
    )
    pair_audit_duplicate_rows = int(
        pair_audit.duplicated(
            subset=["global_image_key", "subject_index", "object_index"],
            keep=False,
        ).sum()
    )

    activation_summary = {
        column: descriptive_stats(pair_audit[column].to_numpy(dtype=np.float64))
        for column in (
            "feature_norm",
            "hidden_norm",
            "raw_delta",
            "margin_delta",
            "margin_delta_abs",
            "base_existence_margin",
            "gate_final_existence_margin",
            "mode_final_existence_margin",
            "gate_margin_change",
            "mode_margin_change",
        )
    }
    activation_summary.update(
        {
            "margin_delta_positive_pairs": int(
                (pair_audit["margin_delta"] > 0).sum()
            ),
            "margin_delta_positive_rate": float(
                (pair_audit["margin_delta"] > 0).mean()
            ),
            "margin_delta_negative_pairs": int(
                (pair_audit["margin_delta"] < 0).sum()
            ),
            "margin_delta_negative_rate": float(
                (pair_audit["margin_delta"] < 0).mean()
            ),
            "margin_delta_saturated_abs_ge_3_9_pairs": int(
                pair_audit["margin_delta_saturated_abs_ge_3_9"].sum()
            ),
            "margin_delta_saturated_abs_ge_3_9_rate": float(
                pair_audit["margin_delta_saturated_abs_ge_3_9"].mean()
            ),
        }
    )

    family_summary: list[dict[str, Any]] = []
    for family, group in pair_audit.groupby("pair_family_v4", dropna=False):
        family_summary.append(
            {
                "pair_family_v4": str(family),
                "pairs": int(len(group)),
                "images": int(group["global_image_key"].nunique()),
                "margin_delta_mean": float(group["margin_delta"].mean()),
                "margin_delta_median": float(group["margin_delta"].median()),
                "margin_delta_p95_abs": float(
                    group["margin_delta_abs"].quantile(0.95)
                ),
                "margin_delta_positive_rate": float(
                    (group["margin_delta"] > 0).mean()
                ),
                "margin_delta_saturation_rate_abs_ge_3_9": float(
                    group["margin_delta_saturated_abs_ge_3_9"].mean()
                ),
                "mode_final_existence_margin_mean": float(
                    group["mode_final_existence_margin"].mean()
                ),
            }
        )

    positive_equal = all(
        np.array_equal(
            pair_audit.filter(like="base_logit__").iloc[:, 1:].to_numpy(),
            pair_audit.filter(like="gate_final_logit__").iloc[:, 1:].to_numpy(),
        )
        for _ in [0]
    )
    checks = {
        "split_is_validation": args.split == EXPECTED_SPLIT,
        "model_fibe_enabled": bool(config.fibe.enabled),
        "mode_geometry_existence_gate": (
            str(config.fibe.mode) == "geometry_existence_gate"
        ),
        "test_cache_locked": config.fibe.test_cache is None,
        "token_residual_branch_absent": model.fibe_branch is None,
        "existence_gate_present": model.fibe_existence_gate is not None,
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
        "pair_audit_rows_complete": len(pair_audit) == args.expected_pairs,
        "pair_audit_duplicates_zero": pair_audit_duplicate_rows == 0,
        "pair_audit_nonfinite_zero": bool(
            np.isfinite(
                pair_audit.select_dtypes(include=[np.number]).to_numpy(
                    dtype=np.float64
                )
            ).all()
        ),
        "all_fibe_pairs_valid_when_required": (
            not args.require_all_fibe_valid
            or (
                total_fibe_valid == args.expected_pairs
                and total_fibe_invalid == 0
            )
        ),
        "shuffle_plan_present_only_for_shuffle": (
            (args.mode == "family_shuffle" and shuffle_summary is not None)
            or (args.mode != "family_shuffle" and shuffle_summary is None)
        ),
        "positive_logits_bitwise_unchanged": positive_equal,
        "invalid_control_rows_complete": (
            invalid_control_rows == args.expected_pairs
        ),
        "invalid_control_delta_exact_zero": (
            invalid_control_delta_nonzero == 0
        ),
        "invalid_control_output_exact_identity": (
            invalid_control_output_mismatches == 0
        ),
        "delta_bound_respected": bool(
            (pair_audit["margin_delta_abs"] <= float(gate.delta_max)).all()
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"

    report: dict[str, Any] = {
        "protocol": PROTOCOL,
        "status": status,
        "mode": args.mode,
        "scope": "frozen validation Water-HN only; final test locked",
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
        "pair_audit": str(pair_audit_path),
        "device": str(device),
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "workers": int(args.workers),
        "max_relations": int(args.max_relations),
        "entry_selection": {
            "authoritative_source": "annotation.test_image_ids",
            "loader_validation_entries": len(loader_validation_entries),
            "authoritative_validation_entries": len(validation_entries),
            "loader_omitted_authoritative_entries": (
                len(validation_entries) - len(loader_validation_entries)
            ),
        },
        "fibe": {
            "enabled": True,
            "mode": "geometry_existence_gate",
            "cache_path": str(fibe_cache.path),
            "selected_valid_pairs": int(total_fibe_valid),
            "selected_invalid_pairs": int(total_fibe_invalid),
            "delta_max": float(gate.delta_max),
            "counterfactual_definition": {
                "normal": "Unmodified trained D1-v2a existence gate.",
                "gate_bypass": (
                    "Run and record the trained gate, but return the exact "
                    "pre-gate base relation logits."
                ),
                "raw_feature_zero": (
                    "Set the selected raw 21D input tensor to exact zero while "
                    "keeping the trained geometry encoder and delta-head active."
                ),
                "family_shuffle": (
                    "Apply a deterministic bijective derangement of selected "
                    "21D tensors within each pair_family_v4."
                ),
            }[args.mode],
        },
        "shuffle": shuffle_summary,
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
            "pair_audit_rows": int(len(pair_audit)),
            "pair_audit_duplicate_rows": pair_audit_duplicate_rows,
            "invalid_control_rows": invalid_control_rows,
            "invalid_control_delta_nonzero": invalid_control_delta_nonzero,
            "invalid_control_output_mismatches": (
                invalid_control_output_mismatches
            ),
        },
        "input_semantic_sha256": input_digest.hexdigest(),
        "mode_fibe_feature_semantic_sha256": mode_feature_digest.hexdigest(),
        "activation_summary": activation_summary,
        "pair_family_activation_summary": family_summary,
        "relation_names": list(dataset.rel_names),
        "sha256": {
            "config": sha256_file(config_path),
            "checkpoint": sha256_file(checkpoint_path),
            "annotation": sha256_file(annotation_path),
            "hardneg": sha256_file(hardneg_path),
            "fibe_cache": sha256_file(fibe_cache.path),
        },
        "checks": checks,
        "first_missing_output_pairs": missing_output_pairs[:20],
        "first_unexpected_output_pairs": unexpected_output_pairs[:20],
    }

    if status != "PASS":
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        print()
        print("D1-v2a COUNTERFACTUAL SCORER V1: FAIL")
        raise SystemExit(1)

    atomic_pickle_dump(output_data, output_path)
    atomic_csv_dump(pair_audit, pair_audit_path)
    report["sha256"]["output_pickle"] = sha256_file(output_path)
    report["sha256"]["pair_audit_csv"] = sha256_file(pair_audit_path)
    atomic_json_dump(report, report_path)

    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    print()
    print("D1-v2a COUNTERFACTUAL SCORER V1: PASS")


if __name__ == "__main__":
    main()
