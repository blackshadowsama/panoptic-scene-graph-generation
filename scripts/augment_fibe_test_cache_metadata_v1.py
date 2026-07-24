#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


FORMAL_SOURCE_CACHE_SHA256 = (
    "7f8a6a45f8f47642873d2ad7bec88762e411a14d54512f003719c4c40276597a"
)
FORMAL_CONFIG_SHA256 = (
    "5701eb9f7ebfe6bf88a4db36ef9c97f20d242f4ba174ffcd103e3033f8003f25"
)
FORMAL_ANNOTATION_SHA256 = (
    "a2eddcebad0ff7b95589f031d71673e1df08dd449e36ec925e6c247908b12461"
)
FORMAL_MANIFEST_SHA256 = (
    "636a342f3589deafd3a54bafbf0ea3f1296ab722e3f42c151aa0162df8be6c35"
)
FORMAL_SCALER_SHA256 = (
    "f75d9608316c9d1934ab4bf6c39fd30bf322b63c5083e8554041df25e8b2364d"
)
FORMAL_FEATURE_CODE_COMMIT = (
    "b7ec3443208ada3ff3ae7c4ea88638df4c5c4705"
)

EXPECTED_IMAGES = 175
EXPECTED_OBJECTS = 1315
EXPECTED_FEATURE_COUNT = 21


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a versioned final-test FIBE cache copy with "
            "complete provenance metadata without recomputing "
            "features or modifying the original test cache."
        )
    )

    parser.add_argument(
        "--cache-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--scaler",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--feature-code-commit",
        required=True,
        help=(
            "Full Git commit used to generate the raw test "
            "feature cache."
        ),
    )
    parser.add_argument(
        "--augmentation-code-commit",
        required=True,
        help=(
            "Full Git commit containing this metadata "
            "augmentation implementation."
        ),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _update_semantic_hash(
    digest: "hashlib._Hash",
    value: Any,
) -> None:
    if isinstance(value, torch.Tensor):
        tensor = (
            value.detach()
            .cpu()
            .contiguous()
        )

        digest.update(b"TENSOR\0")
        digest.update(
            str(tensor.dtype).encode("utf-8")
        )
        digest.update(b"\0")
        digest.update(
            json.dumps(
                list(tensor.shape)
            ).encode("utf-8")
        )
        digest.update(b"\0")
        digest.update(
            tensor.view(torch.uint8)
            .numpy()
            .tobytes()
        )
        return

    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)

        digest.update(b"NDARRAY\0")
        digest.update(
            str(array.dtype).encode("utf-8")
        )
        digest.update(b"\0")
        digest.update(
            json.dumps(
                list(array.shape)
            ).encode("utf-8")
        )
        digest.update(b"\0")
        digest.update(array.tobytes())
        return

    if isinstance(value, dict):
        digest.update(b"DICT\0")

        for key in sorted(
            value,
            key=lambda item: (
                type(item).__name__,
                repr(item),
            ),
        ):
            _update_semantic_hash(
                digest,
                key,
            )
            _update_semantic_hash(
                digest,
                value[key],
            )

        digest.update(b"ENDDICT\0")
        return

    if isinstance(value, (list, tuple)):
        digest.update(
            b"LIST\0"
            if isinstance(value, list)
            else b"TUPLE\0"
        )

        for item in value:
            _update_semantic_hash(
                digest,
                item,
            )

        digest.update(b"ENDSEQ\0")
        return

    if isinstance(value, np.generic):
        _update_semantic_hash(
            digest,
            value.item(),
        )
        return

    if value is None:
        digest.update(b"NONE\0")
        return

    if isinstance(value, bool):
        digest.update(
            b"BOOL\0"
            + (
                b"1"
                if value
                else b"0"
            )
        )
        return

    if isinstance(value, int):
        digest.update(
            b"INT\0"
            + str(value).encode("utf-8")
            + b"\0"
        )
        return

    if isinstance(value, float):
        digest.update(
            b"FLOAT\0"
            + value.hex().encode("ascii")
            + b"\0"
        )
        return

    if isinstance(value, str):
        encoded = value.encode("utf-8")

        digest.update(
            b"STR\0"
            + str(len(encoded)).encode("ascii")
            + b"\0"
            + encoded
        )
        return

    raise TypeError(
        "Unsupported semantic-hash value: "
        f"{type(value)!r}"
    )


def semantic_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _update_semantic_hash(
        digest,
        value,
    )
    return digest.hexdigest()


def validate_commit(
    value: str,
    *,
    argument: str,
) -> str:
    commit = value.strip().lower()

    if (
        len(commit) != 40
        or any(
            character
            not in "0123456789abcdef"
            for character in commit
        )
    ):
        raise ValueError(
            f"{argument} must be a full "
            "40-character hexadecimal commit hash"
        )

    return commit


def atomic_json_save(
    payload: Any,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


def main() -> None:
    args = parse_args()

    cache_root = (
        args.cache_root
        .expanduser()
        .resolve()
    )
    config_path = (
        args.config
        .expanduser()
        .resolve()
    )
    annotation_path = (
        args.annotation
        .expanduser()
        .resolve()
    )
    manifest_path = (
        args.manifest
        .expanduser()
        .resolve()
    )
    scaler_path = (
        args.scaler
        .expanduser()
        .resolve()
    )
    summary_path = (
        args.summary
        .expanduser()
        .resolve()
    )

    feature_code_commit = validate_commit(
        args.feature_code_commit,
        argument="--feature-code-commit",
    )
    augmentation_code_commit = validate_commit(
        args.augmentation_code_commit,
        argument="--augmentation-code-commit",
    )

    source_path = (
        cache_root
        / "test_features.pt"
    )
    output_path = (
        cache_root
        / "test_features_metadata_v1.pt"
    )

    source_paths = {
        "config": config_path,
        "annotation": annotation_path,
        "manifest": manifest_path,
        "scaler": scaler_path,
        "source_cache": source_path,
    }

    for label, path in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"{label}: {path}"
            )

    if output_path.exists():
        raise FileExistsError(
            "Refusing to overwrite existing "
            f"enriched cache: {output_path}"
        )

    if summary_path.exists():
        raise FileExistsError(
            "Refusing to overwrite existing "
            f"summary: {summary_path}"
        )

    metadata = {
        "config_sha256": file_sha256(
            config_path
        ),
        "annotation_sha256": file_sha256(
            annotation_path
        ),
        "manifest_sha256": file_sha256(
            manifest_path
        ),
        "scaler_sha256": file_sha256(
            scaler_path
        ),
        "code_commit": feature_code_commit,
    }

    formal_metadata = {
        "config_sha256": FORMAL_CONFIG_SHA256,
        "annotation_sha256": (
            FORMAL_ANNOTATION_SHA256
        ),
        "manifest_sha256": (
            FORMAL_MANIFEST_SHA256
        ),
        "scaler_sha256": FORMAL_SCALER_SHA256,
        "code_commit": (
            FORMAL_FEATURE_CODE_COMMIT
        ),
    }

    if metadata != formal_metadata:
        raise AssertionError(
            "Formal metadata lock mismatch:\n"
            f"actual={metadata}\n"
            f"expected={formal_metadata}"
        )

    source_hash_before = file_sha256(
        source_path
    )

    if (
        source_hash_before
        != FORMAL_SOURCE_CACHE_SHA256
    ):
        raise AssertionError(
            "test_features.pt SHA-256 mismatch: "
            f"{source_hash_before} != "
            f"{FORMAL_SOURCE_CACHE_SHA256}"
        )

    annotation = json.loads(
        annotation_path.read_text(
            encoding="utf-8"
        )
    )

    records = annotation.get("data")

    if not isinstance(records, list):
        raise TypeError(
            "Annotation data is not a list"
        )

    records_by_id = {
        int(record["image_id"]): record
        for record in records
    }

    test_ids = {
        int(value)
        for value in annotation.get(
            "test_image_ids",
            [],
        )
    }

    if len(test_ids) != EXPECTED_IMAGES:
        raise AssertionError(
            "Unexpected final-test ID count: "
            f"{len(test_ids)}"
        )

    missing_records = sorted(
        test_ids - set(records_by_id)
    )

    if missing_records:
        raise AssertionError(
            "Final-test records missing: "
            f"{missing_records[:20]}"
        )

    expected_objects = sum(
        len(
            records_by_id[image_id]
            .get("segments_info", [])
        )
        for image_id in test_ids
    )

    if expected_objects != EXPECTED_OBJECTS:
        raise AssertionError(
            "Unexpected final-test object count: "
            f"{expected_objects}"
        )

    with manifest_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        manifest_rows = list(
            csv.DictReader(handle)
        )

    manifest_ids = {
        int(row["image_id"])
        for row in manifest_rows
    }
    manifest_splits = {
        str(row["split"])
        for row in manifest_rows
    }

    if len(manifest_rows) != EXPECTED_OBJECTS:
        raise AssertionError(
            "Unexpected manifest row count: "
            f"{len(manifest_rows)}"
        )

    if manifest_ids != test_ids:
        raise AssertionError(
            "Manifest/test image-ID mismatch"
        )

    if manifest_splits != {"test"}:
        raise AssertionError(
            "Unexpected manifest splits: "
            f"{sorted(manifest_splits)}"
        )

    payload = torch.load(
        source_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(payload, dict):
        raise TypeError(
            "test_features.pt payload is not a dict"
        )

    if payload.get("split") != "test":
        raise AssertionError(
            "Unexpected cache split: "
            f"{payload.get('split')!r}"
        )

    if (
        int(payload.get("feature_count", -1))
        != EXPECTED_FEATURE_COUNT
    ):
        raise AssertionError(
            "Unexpected feature count: "
            f"{payload.get('feature_count')!r}"
        )

    if "features_by_image" not in payload:
        raise KeyError(
            "test_features.pt is missing "
            "features_by_image"
        )

    if (
        payload.get("annotation_sha256")
        != metadata["annotation_sha256"]
    ):
        raise AssertionError(
            "Annotation SHA-256 mismatch "
            "inside raw test cache"
        )

    if (
        payload.get("manifest_sha256")
        != metadata["manifest_sha256"]
    ):
        raise AssertionError(
            "Manifest SHA-256 mismatch "
            "inside raw test cache"
        )

    spec_sha = payload.get("spec_sha256")

    if spec_sha not in (
        None,
        metadata["config_sha256"],
    ):
        raise AssertionError(
            "Raw cache spec SHA-256 does not "
            "match formal config SHA-256"
        )

    if (
        payload.get("scaler_version")
        != "FIBE_SCALAR_ROBUST_SCALER_V1"
    ):
        raise AssertionError(
            "Unexpected scaler version: "
            f"{payload.get('scaler_version')!r}"
        )

    feature_ids = {
        int(key)
        for key in payload[
            "features_by_image"
        ]
    }

    if feature_ids != test_ids:
        raise AssertionError(
            "Raw test cache image IDs do not "
            "match final-test annotation IDs"
        )

    if len(feature_ids) != EXPECTED_IMAGES:
        raise AssertionError(
            "Unexpected number of cached images"
        )

    before_semantic = semantic_sha256(
        payload["features_by_image"]
    )

    enriched = dict(payload)
    enriched.update(metadata)
    enriched["metadata_version"] = (
        "FIBE_CACHE_METADATA_V1"
    )
    enriched["source_cache_filename"] = (
        "test_features.pt"
    )
    enriched["source_cache_sha256"] = (
        source_hash_before
    )
    enriched["features_semantic_sha256"] = (
        before_semantic
    )

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    torch.save(
        enriched,
        temporary_path,
    )

    os.replace(
        temporary_path,
        output_path,
    )

    reloaded = torch.load(
        output_path,
        map_location="cpu",
        weights_only=False,
    )

    after_semantic = semantic_sha256(
        reloaded["features_by_image"]
    )

    if before_semantic != after_semantic:
        raise AssertionError(
            "Semantic feature hash changed "
            "during metadata augmentation"
        )

    for key, expected_value in metadata.items():
        if reloaded.get(key) != expected_value:
            raise AssertionError(
                "Metadata mismatch for "
                f"{key}"
            )

    expected_added_keys = {
        "code_commit",
        "config_sha256",
        "features_semantic_sha256",
        "metadata_version",
        "scaler_sha256",
        "source_cache_filename",
        "source_cache_sha256",
    }

    actual_added_keys = (
        set(reloaded)
        - set(payload)
    )

    if actual_added_keys != expected_added_keys:
        raise AssertionError(
            "Unexpected metadata key delta: "
            f"{sorted(actual_added_keys)}"
        )

    removed_keys = (
        set(payload)
        - set(reloaded)
    )

    if removed_keys:
        raise AssertionError(
            "Raw cache fields were removed: "
            f"{sorted(removed_keys)}"
        )

    for key, value in payload.items():
        if key == "features_by_image":
            continue

        if reloaded.get(key) != value:
            raise AssertionError(
                "Original raw-cache field changed: "
                f"{key}"
            )

    source_hash_after = file_sha256(
        source_path
    )
    scaler_hash_after = file_sha256(
        scaler_path
    )

    if source_hash_after != source_hash_before:
        raise AssertionError(
            "Original test cache was modified"
        )

    if (
        scaler_hash_after
        != FORMAL_SCALER_SHA256
    ):
        raise AssertionError(
            "Frozen training scaler was modified"
        )

    output_hash = file_sha256(
        output_path
    )
    script_path = Path(__file__).resolve()

    summary = {
        "version": (
            "FIBE_CACHE_METADATA_AUGMENTATION_V1"
        ),
        "split": "test",
        "source_files": {
            label: str(path)
            for label, path
            in source_paths.items()
        },
        "metadata": metadata,
        "augmentation": {
            "code_commit": (
                augmentation_code_commit
            ),
            "script_path": str(script_path),
            "script_sha256": file_sha256(
                script_path
            ),
        },
        "cache": {
            "source_path": str(source_path),
            "source_file_sha256": (
                source_hash_before
            ),
            "output_path": str(output_path),
            "output_file_sha256": (
                output_hash
            ),
            "features_semantic_sha256_before": (
                before_semantic
            ),
            "features_semantic_sha256_after": (
                after_semantic
            ),
            "cached_images": len(
                reloaded["features_by_image"]
            ),
            "feature_count": reloaded.get(
                "feature_count"
            ),
            "added_keys": sorted(
                actual_added_keys
            ),
            "removed_keys": sorted(
                removed_keys
            ),
        },
        "dataset": {
            "test_images": len(test_ids),
            "test_objects": expected_objects,
            "manifest_rows": len(
                manifest_rows
            ),
            "manifest_splits": sorted(
                manifest_splits
            ),
        },
    }

    atomic_json_save(
        summary,
        summary_path,
    )

    print(
        "PASS: test metadata augmentation"
    )
    print("  source:", source_path)
    print("  output:", output_path)
    print(
        "  raw cache SHA-256:",
        source_hash_before,
    )
    print(
        "  semantic SHA-256:",
        before_semantic,
    )
    print(
        "  output SHA-256:",
        output_hash,
    )
    print(
        "  feature-code commit:",
        feature_code_commit,
    )
    print(
        "  augmentation commit:",
        augmentation_code_commit,
    )
    print("  summary:", summary_path)
    print(
        "FIBE TEST CACHE METADATA "
        "AUGMENTATION: PASS"
    )


if __name__ == "__main__":
    main()
