from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (PROJECT_ROOT / "data" / "floodpsg").resolve()

DEFAULT_FEATURE_DIR = (
    DATA_ROOT
    / "features"
    / "fibe_scalar_v1"
)
DEFAULT_STATS_DIR = (
    DATA_ROOT
    / "stats"
    / "dsformer_fibe_d1_v1"
)
DEFAULT_SPEC = (
    PROJECT_ROOT
    / "configs"
    / "floodpsg"
    / "fibe_scalar_v1.json"
)

FEATURE_COUNT = 21
RATIO_INDICES = (0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14)
ANGLE_INDICES = (15, 16)
FLAG_INDICES = (18, 19, 20)
ROBUST_INDICES = (7, 8, 9, 17)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit FIBE-Scalar train/validation feature caches, "
            "including tensor shapes, invalid-pair zeroing, "
            "canonical pair symmetry, ranges, hashes, and "
            "train-only robust-scaler counts."
        )
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=DEFAULT_FEATURE_DIR,
    )
    parser.add_argument(
        "--stats-dir",
        type=Path,
        default=DEFAULT_STATS_DIR,
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=DEFAULT_SPEC,
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Cache tag such as smoke5. Empty means formal cache names.",
    )
    parser.add_argument(
        "--expected-train-images",
        type=int,
        default=1500,
    )
    parser.add_argument(
        "--expected-validation-images",
        type=int,
        default=173,
    )
    parser.add_argument(
        "--expected-train-objects",
        type=int,
        default=0,
        help="0 skips the exact object-count assertion.",
    )
    parser.add_argument(
        "--expected-validation-objects",
        type=int,
        default=0,
        help="0 skips the exact object-count assertion.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
    )
    return parser.parse_args()


def tagged_name(base_name: str, tag: str) -> str:
    if not tag:
        return base_name
    path = Path(base_name)
    return f"{path.stem}_{tag}{path.suffix}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def tensor_range_violations(
    tensor: torch.Tensor,
    selection: torch.Tensor,
    *,
    lower: float,
    upper: float,
    atol: float,
) -> int:
    selected = tensor[selection]
    if selected.numel() == 0:
        return 0
    return int(
        (
            (selected < lower - atol)
            | (selected > upper + atol)
        ).sum().item()
    )


def audit_split(
    *,
    split: str,
    payload: dict[str, Any],
    expected_images: int,
    expected_objects: int,
    atol: float,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []

    if payload.get("version") != "FIBE_SCALAR_FEATURE_CACHE_V1":
        errors.append(
            f"{split}: unexpected cache version "
            f"{payload.get('version')!r}"
        )

    if payload.get("split") != split:
        errors.append(
            f"{split}: payload split is {payload.get('split')!r}"
        )

    if int(payload.get("feature_count", -1)) != FEATURE_COUNT:
        errors.append(
            f"{split}: feature_count={payload.get('feature_count')}"
        )

    images = payload.get("features_by_image")
    if not isinstance(images, dict):
        raise TypeError(
            f"{split}: features_by_image is not a dictionary"
        )

    if len(images) != expected_images:
        errors.append(
            f"{split}: image count {len(images)} != {expected_images}"
        )

    total_objects = 0
    total_matrix_cells = 0
    total_valid_pairs = 0
    apparent_valid_pairs = 0
    truncated_valid_pairs = 0

    nonfinite_values = 0
    invalid_nonzero_values = 0
    valid_flag_mismatches = 0
    diagonal_valid_pairs = 0
    diagonal_nonzero_values = 0
    nonsymmetric_valid_cells = 0
    nonsymmetric_feature_values = 0
    ratio_range_violations = 0
    angle_range_violations = 0
    flag_range_violations = 0
    robust_range_violations = 0
    dtype_errors = 0
    shape_errors = 0
    object_count_errors = 0

    for raw_image_id, entry in images.items():
        image_id = int(raw_image_id)
        features = entry["features"]
        valid_pairs = entry["valid_pairs"]
        num_objects = int(entry["num_objects"])

        if not isinstance(features, torch.Tensor):
            errors.append(f"{split}/{image_id}: features is not a tensor")
            continue

        if not isinstance(valid_pairs, torch.Tensor):
            errors.append(f"{split}/{image_id}: valid_pairs is not a tensor")
            continue

        if features.dtype != torch.float32:
            dtype_errors += 1

        if valid_pairs.dtype != torch.bool:
            dtype_errors += 1

        expected_feature_shape = (
            num_objects,
            num_objects,
            FEATURE_COUNT,
        )
        expected_valid_shape = (
            num_objects,
            num_objects,
        )

        if tuple(features.shape) != expected_feature_shape:
            shape_errors += 1
            continue

        if tuple(valid_pairs.shape) != expected_valid_shape:
            shape_errors += 1
            continue

        if num_objects < 1:
            object_count_errors += 1
            continue

        total_objects += num_objects
        total_matrix_cells += num_objects * num_objects
        total_valid_pairs += int(valid_pairs.sum().item())
        apparent_valid_pairs += int(
            (
                valid_pairs
                & (features[:, :, 18] > 0.5)
            ).sum().item()
        )
        truncated_valid_pairs += int(
            (
                valid_pairs
                & (features[:, :, 19] > 0.5)
            ).sum().item()
        )

        nonfinite_values += int(
            (~torch.isfinite(features)).sum().item()
        )

        invalid = ~valid_pairs
        invalid_nonzero_values += int(
            torch.count_nonzero(features[invalid]).item()
        )

        valid_flag_mismatches += int(
            (
                (features[:, :, 20] > 0.5)
                != valid_pairs
            ).sum().item()
        )

        diagonal = torch.eye(
            num_objects,
            dtype=torch.bool,
        )
        diagonal_valid_pairs += int(
            valid_pairs[diagonal].sum().item()
        )
        diagonal_nonzero_values += int(
            torch.count_nonzero(features[diagonal]).item()
        )

        nonsymmetric_valid_cells += int(
            (valid_pairs != valid_pairs.T).sum().item()
        )

        both_valid = valid_pairs & valid_pairs.T
        if both_valid.any():
            reverse = features.transpose(0, 1)
            delta = torch.abs(features - reverse)
            selected_delta = delta[both_valid]
            nonsymmetric_feature_values += int(
                (selected_delta > atol).sum().item()
            )

        for feature_index in RATIO_INDICES:
            ratio_range_violations += tensor_range_violations(
                features[:, :, feature_index],
                valid_pairs,
                lower=0.0,
                upper=1.0,
                atol=atol,
            )

        for feature_index in ANGLE_INDICES:
            angle_range_violations += tensor_range_violations(
                features[:, :, feature_index],
                valid_pairs,
                lower=-1.0,
                upper=1.0,
                atol=atol,
            )

        for feature_index in FLAG_INDICES:
            values = features[:, :, feature_index][valid_pairs]
            if values.numel():
                flag_range_violations += int(
                    (
                        ~(
                            torch.isclose(
                                values,
                                torch.zeros_like(values),
                                atol=atol,
                                rtol=0.0,
                            )
                            | torch.isclose(
                                values,
                                torch.ones_like(values),
                                atol=atol,
                                rtol=0.0,
                            )
                        )
                    ).sum().item()
                )

        for feature_index in ROBUST_INDICES:
            robust_selection = valid_pairs.clone()
            if feature_index == 17:
                robust_selection &= features[:, :, 18] > 0.5

            robust_range_violations += tensor_range_violations(
                features[:, :, feature_index],
                robust_selection,
                lower=-5.0,
                upper=5.0,
                atol=atol,
            )

            if feature_index == 17:
                invalid_apparent = (
                    valid_pairs
                    & ~(features[:, :, 18] > 0.5)
                )
                robust_range_violations += int(
                    torch.count_nonzero(
                        features[:, :, 17][invalid_apparent]
                    ).item()
                )

    if expected_objects > 0 and total_objects != expected_objects:
        errors.append(
            f"{split}: object count {total_objects} != {expected_objects}"
        )

    counters = {
        "images": int(len(images)),
        "objects": int(total_objects),
        "matrix_cells": int(total_matrix_cells),
        "valid_pairs": int(total_valid_pairs),
        "apparent_boundary_valid_pairs": int(apparent_valid_pairs),
        "truncated_valid_pairs": int(truncated_valid_pairs),
        "nonfinite_values": int(nonfinite_values),
        "invalid_pair_nonzero_values": int(invalid_nonzero_values),
        "valid_flag_mismatches": int(valid_flag_mismatches),
        "diagonal_valid_pairs": int(diagonal_valid_pairs),
        "diagonal_nonzero_values": int(diagonal_nonzero_values),
        "nonsymmetric_valid_cells": int(nonsymmetric_valid_cells),
        "nonsymmetric_feature_values": int(
            nonsymmetric_feature_values
        ),
        "ratio_range_violations": int(ratio_range_violations),
        "angle_range_violations": int(angle_range_violations),
        "flag_range_violations": int(flag_range_violations),
        "robust_range_violations": int(robust_range_violations),
        "dtype_errors": int(dtype_errors),
        "shape_errors": int(shape_errors),
        "object_count_errors": int(object_count_errors),
    }

    zero_required = (
        "nonfinite_values",
        "invalid_pair_nonzero_values",
        "valid_flag_mismatches",
        "diagonal_valid_pairs",
        "diagonal_nonzero_values",
        "nonsymmetric_valid_cells",
        "nonsymmetric_feature_values",
        "ratio_range_violations",
        "angle_range_violations",
        "flag_range_violations",
        "robust_range_violations",
        "dtype_errors",
        "shape_errors",
        "object_count_errors",
    )

    for name in zero_required:
        if counters[name] != 0:
            errors.append(
                f"{split}: {name}={counters[name]}"
            )

    return counters, errors


def main() -> None:
    args = parse_args()

    feature_dir = args.feature_dir.expanduser().resolve()
    stats_dir = args.stats_dir.expanduser().resolve()
    spec_path = args.spec.expanduser().resolve()

    train_path = feature_dir / tagged_name(
        "train_features.pt",
        args.tag,
    )
    validation_path = feature_dir / tagged_name(
        "validation_features.pt",
        args.tag,
    )
    scaler_path = feature_dir / tagged_name(
        "scaler_v1.json",
        args.tag,
    )
    build_summary_path = stats_dir / tagged_name(
        "FIBE_FEATURE_CACHE_BUILD_V1.json",
        args.tag,
    )
    audit_summary_path = stats_dir / tagged_name(
        "FIBE_FEATURE_CACHE_AUDIT_V1.json",
        args.tag,
    )

    for path in (
        train_path,
        validation_path,
        scaler_path,
        build_summary_path,
        spec_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    train_payload = torch.load(
        train_path,
        map_location="cpu",
    )
    validation_payload = torch.load(
        validation_path,
        map_location="cpu",
    )
    scaler = load_json(scaler_path)
    build_summary = load_json(build_summary_path)
    spec = load_json(spec_path)

    errors: list[str] = []

    if spec.get("version") != "FIBE_SCALAR_V1":
        errors.append(
            f"Unexpected spec version: {spec.get('version')!r}"
        )

    if int(spec.get("feature_count", -1)) != FEATURE_COUNT:
        errors.append(
            f"Unexpected spec feature_count: {spec.get('feature_count')}"
        )

    spec_sha256 = sha256_file(spec_path)

    for split, payload in (
        ("train", train_payload),
        ("validation", validation_payload),
    ):
        if payload.get("spec_sha256") != spec_sha256:
            errors.append(
                f"{split}: payload spec hash does not match current spec"
            )

    hash_fields = (
        "spec_sha256",
        "annotation_sha256",
        "manifest_sha256",
        "scaler_version",
    )
    for field in hash_fields:
        if train_payload.get(field) != validation_payload.get(field):
            errors.append(
                f"train/validation mismatch for {field}"
            )

    if scaler.get("version") != "FIBE_SCALAR_ROBUST_SCALER_V1":
        errors.append(
            f"Unexpected scaler version: {scaler.get('version')!r}"
        )

    scaler_indices = sorted(
        int(value)
        for value in scaler.get("features", {})
    )
    if scaler_indices != list(ROBUST_INDICES):
        errors.append(
            f"Scaler feature indexes are {scaler_indices}"
        )

    train_stats, train_errors = audit_split(
        split="train",
        payload=train_payload,
        expected_images=args.expected_train_images,
        expected_objects=args.expected_train_objects,
        atol=args.atol,
    )
    validation_stats, validation_errors = audit_split(
        split="validation",
        payload=validation_payload,
        expected_images=args.expected_validation_images,
        expected_objects=args.expected_validation_objects,
        atol=args.atol,
    )
    errors.extend(train_errors)
    errors.extend(validation_errors)

    for feature_index in (7, 8, 9):
        count = int(
            scaler["features"][str(feature_index)]["count"]
        )
        if count != train_stats["valid_pairs"]:
            errors.append(
                f"Scaler feature {feature_index} count={count}, "
                f"expected train valid_pairs={train_stats['valid_pairs']}"
            )

    feature17_count = int(
        scaler["features"]["17"]["count"]
    )
    if (
        feature17_count
        != train_stats["apparent_boundary_valid_pairs"]
    ):
        errors.append(
            f"Scaler feature 17 count={feature17_count}, expected "
            f"train apparent-valid pairs="
            f"{train_stats['apparent_boundary_valid_pairs']}"
        )

    expected_build_splits = build_summary.get("splits", {})
    for split, stats in (
        ("train", train_stats),
        ("validation", validation_stats),
    ):
        build_stats = expected_build_splits.get(split)
        if not isinstance(build_stats, dict):
            errors.append(
                f"Build summary has no {split} statistics"
            )
            continue

        for field in (
            "images",
            "objects",
            "valid_pairs",
            "apparent_boundary_valid_pairs",
            "truncated_valid_pairs",
        ):
            if int(build_stats.get(field, -1)) != int(stats[field]):
                errors.append(
                    f"{split}: build-summary {field}="
                    f"{build_stats.get(field)!r}, audited={stats[field]}"
                )

    summary = {
        "version": "FIBE_FEATURE_CACHE_AUDIT_V1",
        "tag": args.tag,
        "feature_dir": str(feature_dir),
        "spec_path": str(spec_path),
        "spec_sha256": spec_sha256,
        "train_cache": str(train_path),
        "train_cache_sha256": sha256_file(train_path),
        "validation_cache": str(validation_path),
        "validation_cache_sha256": sha256_file(validation_path),
        "scaler_path": str(scaler_path),
        "scaler_sha256": sha256_file(scaler_path),
        "build_summary_path": str(build_summary_path),
        "build_summary_sha256": sha256_file(build_summary_path),
        "train": train_stats,
        "validation": validation_stats,
        "scaler_counts": {
            key: int(value["count"])
            for key, value in scaler["features"].items()
        },
        "errors": errors,
    }

    audit_summary_path.parent.mkdir(parents=True, exist_ok=True)
    audit_summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 100)
    print("FIBE FEATURE CACHE AUDIT V1")
    print("=" * 100)
    print("tag:", args.tag or "(formal)")
    print("spec sha256:", spec_sha256)

    for split, stats in (
        ("train", train_stats),
        ("validation", validation_stats),
    ):
        print(f"\n{split.upper()}")
        for key, value in stats.items():
            print(f"{key}: {value}")

    print("\nSCALER COUNTS")
    for key, value in summary["scaler_counts"].items():
        print(f"feature {key}: {value}")

    print("\nHASHES")
    print("train:", summary["train_cache_sha256"])
    print("validation:", summary["validation_cache_sha256"])
    print("scaler:", summary["scaler_sha256"])
    print("build summary:", summary["build_summary_sha256"])

    print("\nOUTPUT")
    print("audit summary:", audit_summary_path)

    if errors:
        print("\nERRORS")
        for error in errors:
            print("-", error)
        raise SystemExit("FIBE FEATURE CACHE AUDIT: FAIL")

    print("\nFIBE FEATURE CACHE AUDIT: PASS")


if __name__ == "__main__":
    main()
