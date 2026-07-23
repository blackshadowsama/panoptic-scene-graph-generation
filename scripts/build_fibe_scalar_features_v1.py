from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any
import sys

# Support direct execution from the scripts directory.
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

import numpy as np
import pandas as pd
import torch

from fair_psgg.data.fibe_features import (
    FEATURE_COUNT,
    FIBESpec,
    apply_robust_scaler,
    build_object_geometry,
    compute_image_feature_matrix,
    fit_robust_scaler,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (PROJECT_ROOT / "data" / "floodpsg").resolve()

DEFAULT_ANNOTATION = (
    DATA_ROOT
    / "annotations"
    / "floodpsg_canonical_trainval_coarse8_groupstrict_v1.json"
)
DEFAULT_MANIFEST = (
    DATA_ROOT
    / "features"
    / "fibe_scalar_v1"
    / "fibe_canonical_mask_manifest_v1.csv"
)
DEFAULT_SPEC = (
    PROJECT_ROOT
    / "configs"
    / "floodpsg"
    / "fibe_scalar_v1.json"
)
DEFAULT_OUTPUT_DIR = (
    DATA_ROOT
    / "features"
    / "fibe_scalar_v1"
)
DEFAULT_SUMMARY = (
    DATA_ROOT
    / "stats"
    / "dsformer_fibe_d1_v1"
    / "FIBE_FEATURE_CACHE_BUILD_V1.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build offline train/validation FIBE-Scalar V1 feature caches."
        )
    )
    parser.add_argument("--annotation", type=Path, default=DEFAULT_ANNOTATION)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation"),
        default=("train", "validation"),
    )
    parser.add_argument(
        "--limit-images-per-split",
        type=int,
        default=0,
        help="0 means no limit. Positive values are for smoke tests only.",
    )
    parser.add_argument(
        "--output-tag",
        default="",
        help="Optional suffix, e.g. smoke5, to avoid overwriting formal caches.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_records(payload: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if isinstance(payload, list):
        return payload, {}
    if not isinstance(payload, dict):
        raise TypeError(f"Unexpected annotation type: {type(payload)!r}")
    for key in ("data", "images", "records"):
        if isinstance(payload.get(key), list):
            return payload[key], payload
    raise KeyError("Could not find records under data/images/records")


def resolve_mask_path(value: Any) -> Path:
    path = Path(str(value).strip()).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (DATA_ROOT / path).resolve()


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def tagged_name(base_name: str, tag: str) -> str:
    if not tag:
        return base_name
    path = Path(base_name)
    return f"{path.stem}_{tag}{path.suffix}"


def main() -> None:
    args = parse_args()

    annotation_path = args.annotation.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    spec_path = args.spec.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    summary_path = args.summary.expanduser().resolve()

    for path in (annotation_path, manifest_path, spec_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    annotation_payload = json.loads(
        annotation_path.read_text(encoding="utf-8")
    )
    records, annotation_root = load_records(annotation_payload)
    validation_ids = {
        int(value) for value in annotation_root.get("test_image_ids", [])
    }
    records_by_id = {int(record["image_id"]): record for record in records}

    manifest = pd.read_csv(
        manifest_path,
        encoding="utf-8-sig",
        low_memory=False,
    )
    manifest["image_id"] = pd.to_numeric(
        manifest["image_id"], errors="raise"
    ).astype(int)
    manifest["canonical_object_index"] = pd.to_numeric(
        manifest["canonical_object_index"], errors="raise"
    ).astype(int)

    spec_payload = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec_payload.get("version") != "FIBE_SCALAR_V1":
        raise RuntimeError(
            f"Unexpected spec version: {spec_payload.get('version')}"
        )
    if int(spec_payload.get("feature_count", -1)) != FEATURE_COUNT:
        raise RuntimeError("Unexpected FIBE feature count")

    spec = FIBESpec(
        r1_norm=float(spec_payload["calibration"]["r1_norm"]),
        r2_norm=float(spec_payload["calibration"]["r2_norm"]),
        truncation_margin_px=int(
            spec_payload["mask_processing"]["truncation_margin_px"]
        ),
        lower50_start_fraction=float(
            spec_payload["lower_interaction"]["lower50_start_fraction"]
        ),
        lower25_start_fraction=float(
            spec_payload["lower_interaction"]["lower25_start_fraction"]
        ),
        minimum_shared_columns=int(
            spec_payload["apparent_water_boundary"]["minimum_shared_columns"]
        ),
        minimum_x_coverage=float(
            spec_payload["apparent_water_boundary"]["minimum_x_coverage"]
        ),
        maximum_fit_columns=int(
            spec_payload["apparent_water_boundary"]["maximum_fit_columns"]
        ),
    )

    if not 0.0 <= spec.r1_norm < spec.r2_norm:
        raise RuntimeError(
            f"Invalid radii in spec: r1={spec.r1_norm}, r2={spec.r2_norm}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    raw_by_split: dict[str, dict[int, dict[str, Any]]] = {}
    split_stats: dict[str, dict[str, Any]] = {}

    for split in args.splits:
        split_manifest = manifest.loc[manifest["split"] == split].copy()
        image_ids = sorted(split_manifest["image_id"].unique().tolist())

        if args.limit_images_per_split > 0:
            image_ids = image_ids[: args.limit_images_per_split]

        features_by_image: dict[int, dict[str, Any]] = {}
        started = time.perf_counter()
        total_objects = 0
        total_pairs = 0
        valid_pairs_total = 0
        apparent_valid_total = 0
        truncated_valid_total = 0

        for position, image_id in enumerate(image_ids, start=1):
            record = records_by_id[int(image_id)]
            expected_split = (
                "validation" if int(image_id) in validation_ids else "train"
            )
            if expected_split != split:
                raise RuntimeError(
                    f"Split mismatch for image {image_id}: "
                    f"expected={expected_split}, requested={split}"
                )

            height = int(record["height"])
            width = int(record["width"])
            segments = record["segments_info"]
            group = (
                split_manifest.loc[split_manifest["image_id"] == image_id]
                .sort_values("canonical_object_index", kind="stable")
            )

            if len(group) != len(segments):
                raise RuntimeError(
                    f"Object-count mismatch for image {image_id}: "
                    f"manifest={len(group)}, annotation={len(segments)}"
                )

            objects = []
            for row in group.itertuples(index=False):
                object_index = int(row.canonical_object_index)
                segment = segments[object_index]
                manifest_category = str(row.category_name)
                annotation_category = str(segment["category_name"])

                if manifest_category != annotation_category:
                    raise RuntimeError(
                        f"Category mismatch image={image_id}, "
                        f"object={object_index}: manifest={manifest_category}, "
                        f"annotation={annotation_category}"
                    )

                objects.append(
                    build_object_geometry(
                        object_index=object_index,
                        category_name=manifest_category,
                        mask_path=resolve_mask_path(row.local_mask_path),
                        expected_height=height,
                        expected_width=width,
                        spec=spec,
                    )
                )

            features, valid_pairs = compute_image_feature_matrix(
                objects=objects,
                image_height=height,
                image_width=width,
                spec=spec,
            )

            num_objects = len(objects)
            features_by_image[int(image_id)] = {
                "features": features,
                "valid_pairs": valid_pairs,
                "num_objects": int(num_objects),
            }

            total_objects += num_objects
            total_pairs += num_objects * num_objects
            valid_pairs_total += int(valid_pairs.sum())
            apparent_valid_total += int(
                np.logical_and(
                    valid_pairs,
                    features[:, :, 18] > 0.5,
                ).sum()
            )
            truncated_valid_total += int(
                np.logical_and(
                    valid_pairs,
                    features[:, :, 19] > 0.5,
                ).sum()
            )

            if position % 50 == 0 or position == len(image_ids):
                elapsed = time.perf_counter() - started
                print(
                    f"{split}: processed {position}/{len(image_ids)} images, "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

        raw_by_split[split] = features_by_image
        elapsed = time.perf_counter() - started
        split_stats[split] = {
            "images": len(features_by_image),
            "objects": int(total_objects),
            "matrix_cells": int(total_pairs),
            "valid_pairs": int(valid_pairs_total),
            "apparent_boundary_valid_pairs": int(apparent_valid_total),
            "truncated_valid_pairs": int(truncated_valid_total),
            "elapsed_seconds_raw": float(elapsed),
        }

    scaler = None
    if "train" in raw_by_split:
        scaler = fit_robust_scaler(raw_by_split["train"])
    else:
        scaler_path = output_dir / "scaler_v1.json"
        if not scaler_path.is_file():
            raise FileNotFoundError(
                "Training split was not built and scaler_v1.json is absent"
            )
        scaler = json.loads(scaler_path.read_text(encoding="utf-8"))

    for split, features_by_image in raw_by_split.items():
        apply_robust_scaler(features_by_image, scaler)

        converted: dict[int, dict[str, Any]] = {}
        nonfinite = 0
        invalid_nonzero = 0
        valid_flag_mismatch = 0

        for image_id, entry in features_by_image.items():
            features = np.asarray(entry["features"], dtype=np.float32)
            valid_pairs = np.asarray(entry["valid_pairs"], dtype=bool)

            nonfinite += int((~np.isfinite(features)).sum())
            invalid_nonzero += int(np.count_nonzero(features[~valid_pairs]))
            valid_flag_mismatch += int(
                np.count_nonzero(
                    (features[:, :, 20] > 0.5) != valid_pairs
                )
            )

            converted[int(image_id)] = {
                "features": torch.from_numpy(features.copy()),
                "valid_pairs": torch.from_numpy(valid_pairs.copy()),
                "num_objects": int(entry["num_objects"]),
            }

        if nonfinite or invalid_nonzero or valid_flag_mismatch:
            raise RuntimeError(
                f"Cache audit failed for {split}: "
                f"nonfinite={nonfinite}, invalid_nonzero={invalid_nonzero}, "
                f"valid_flag_mismatch={valid_flag_mismatch}"
            )

        cache_name = tagged_name(f"{split}_features.pt", args.output_tag)
        cache_path = output_dir / cache_name

        if cache_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite {cache_path}; pass --overwrite"
            )

        payload = {
            "version": "FIBE_SCALAR_FEATURE_CACHE_V1",
            "split": split,
            "feature_count": FEATURE_COUNT,
            "spec_sha256": sha256_file(spec_path),
            "annotation_sha256": sha256_file(annotation_path),
            "manifest_sha256": sha256_file(manifest_path),
            "scaler_version": scaler["version"],
            "features_by_image": converted,
        }
        atomic_torch_save(payload, cache_path)
        split_stats[split]["cache_path"] = str(cache_path)
        split_stats[split]["cache_size_bytes"] = int(cache_path.stat().st_size)
        split_stats[split]["nonfinite_values"] = int(nonfinite)
        split_stats[split]["invalid_pair_nonzero_values"] = int(invalid_nonzero)
        split_stats[split]["valid_flag_mismatches"] = int(valid_flag_mismatch)

    scaler_name = tagged_name("scaler_v1.json", args.output_tag)
    scaler_path = output_dir / scaler_name
    if scaler_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {scaler_path}; pass --overwrite"
        )
    atomic_json_save(scaler, scaler_path)

    summary_name = tagged_name(summary_path.name, args.output_tag)
    actual_summary_path = summary_path.with_name(summary_name)
    summary = {
        "version": "FIBE_FEATURE_CACHE_BUILD_V1",
        "annotation_path": str(annotation_path),
        "annotation_sha256": sha256_file(annotation_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "spec_path": str(spec_path),
        "spec_sha256": sha256_file(spec_path),
        "output_tag": args.output_tag,
        "limit_images_per_split": int(args.limit_images_per_split),
        "splits": split_stats,
        "scaler_path": str(scaler_path),
        "scaler": scaler,
    }
    atomic_json_save(summary, actual_summary_path)

    print("\n" + "=" * 100)
    print("FIBE FEATURE CACHE BUILD V1")
    print("=" * 100)
    for split, stats in split_stats.items():
        print(
            f"{split}: images={stats['images']}, objects={stats['objects']}, "
            f"valid_pairs={stats['valid_pairs']}, "
            f"apparent_valid={stats['apparent_boundary_valid_pairs']}, "
            f"truncated_valid={stats['truncated_valid_pairs']}"
        )
        print(" cache:", stats["cache_path"])
        print(" bytes:", stats["cache_size_bytes"])
    print("scaler:", scaler_path)
    print("summary:", actual_summary_path)
    print("\nFIBE FEATURE CACHE BUILD: PASS")


if __name__ == "__main__":
    main()
