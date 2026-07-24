#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


FORMAL_ORIGINAL_SHA256 = {
    "train_features.pt": "96b01c2bd6b64c34c767ad9c403303676b2dca17efac5c7274fff4478480f678",
    "validation_features.pt": "d78c6cfcf7bf019926b63546a974a6ad3af3a5ae0c281f438d2ad98ac793e8e4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create versioned FIBE cache copies with complete provenance metadata "
            "without recomputing features or modifying the original cache files."
        )
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="JSON summary path.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_semantic_hash(digest: "hashlib._Hash", value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"TENSOR\0")
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
        return

    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"NDARRAY\0")
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(b"\0")
        digest.update(json.dumps(list(array.shape)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(array.tobytes())
        return

    if isinstance(value, dict):
        digest.update(b"DICT\0")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _update_semantic_hash(digest, key)
            _update_semantic_hash(digest, value[key])
        digest.update(b"ENDDICT\0")
        return

    if isinstance(value, (list, tuple)):
        digest.update(b"LIST\0" if isinstance(value, list) else b"TUPLE\0")
        for item in value:
            _update_semantic_hash(digest, item)
        digest.update(b"ENDSEQ\0")
        return

    if isinstance(value, np.generic):
        _update_semantic_hash(digest, value.item())
        return

    if value is None:
        digest.update(b"NONE\0")
        return

    if isinstance(value, bool):
        digest.update(b"BOOL\0" + (b"1" if value else b"0"))
        return

    if isinstance(value, int):
        digest.update(b"INT\0" + str(value).encode("utf-8") + b"\0")
        return

    if isinstance(value, float):
        digest.update(b"FLOAT\0" + value.hex().encode("ascii") + b"\0")
        return

    if isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(b"STR\0" + str(len(encoded)).encode("ascii") + b"\0" + encoded)
        return

    raise TypeError(f"Unsupported semantic-hash value: {type(value)!r}")


def semantic_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _update_semantic_hash(digest, value)
    return digest.hexdigest()


def validate_commit(value: str) -> str:
    commit = value.strip().lower()
    if len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
        raise ValueError("--code-commit must be a full 40-character hexadecimal commit hash")
    return commit


def main() -> None:
    args = parse_args()

    cache_root = args.cache_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    annotation_path = args.annotation.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    scaler_path = args.scaler.expanduser().resolve()
    summary_path = args.summary.expanduser().resolve()
    code_commit = validate_commit(args.code_commit)

    source_paths = {
        "config": config_path,
        "annotation": annotation_path,
        "manifest": manifest_path,
        "scaler": scaler_path,
    }
    for label, path in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label}: {path}")

    metadata = {
        "config_sha256": file_sha256(config_path),
        "annotation_sha256": file_sha256(annotation_path),
        "manifest_sha256": file_sha256(manifest_path),
        "scaler_sha256": file_sha256(scaler_path),
        "code_commit": code_commit,
    }

    split_specs = (
        ("train_features.pt", "train_features_metadata_v1.pt", ("train",)),
        (
            "validation_features.pt",
            "validation_features_metadata_v1.pt",
            ("validation", "val"),
        ),
    )

    summary: dict[str, Any] = {
        "version": "FIBE_CACHE_METADATA_AUGMENTATION_V1",
        "source_files": {label: str(path) for label, path in source_paths.items()},
        "metadata": metadata,
        "caches": {},
    }

    for source_name, output_name, accepted_splits in split_specs:
        summary_split = "train" if source_name.startswith("train_") else "validation"
        source_path = cache_root / source_name
        output_path = cache_root / output_name
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        if output_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing enriched cache: {output_path}"
            )

        original_file_hash = file_sha256(source_path)
        expected_original_hash = FORMAL_ORIGINAL_SHA256[source_name]
        if original_file_hash != expected_original_hash:
            raise AssertionError(
                f"{source_name}: original SHA-256 mismatch: "
                f"{original_file_hash} != {expected_original_hash}"
            )

        payload = torch.load(source_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError(f"{source_name}: payload is not a dict")
        if payload.get("split") not in accepted_splits:
            raise AssertionError(
                f"{source_name}: split={payload.get('split')!r}, "
                f"expected one of {accepted_splits!r}"
            )
        if "features_by_image" not in payload:
            raise KeyError(f"{source_name}: missing features_by_image")

        if payload.get("annotation_sha256") != metadata["annotation_sha256"]:
            raise AssertionError(f"{source_name}: annotation SHA-256 mismatch")
        if payload.get("manifest_sha256") != metadata["manifest_sha256"]:
            raise AssertionError(f"{source_name}: manifest SHA-256 mismatch")

        spec_sha = payload.get("spec_sha256")
        if spec_sha not in (None, metadata["config_sha256"]):
            raise AssertionError(
                f"{source_name}: spec_sha256={spec_sha!r} does not match config SHA-256"
            )

        before_semantic = semantic_sha256(payload["features_by_image"])

        enriched = dict(payload)
        enriched.update(metadata)
        enriched["metadata_version"] = "FIBE_CACHE_METADATA_V1"
        enriched["source_cache_filename"] = source_name
        enriched["source_cache_sha256"] = original_file_hash
        enriched["features_semantic_sha256"] = before_semantic

        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        if temporary_path.exists():
            temporary_path.unlink()

        torch.save(enriched, temporary_path)
        os.replace(temporary_path, output_path)

        reloaded = torch.load(output_path, map_location="cpu", weights_only=False)
        after_semantic = semantic_sha256(reloaded["features_by_image"])
        if before_semantic != after_semantic:
            raise AssertionError(f"{output_name}: semantic feature hash changed")

        for key, expected_value in metadata.items():
            if reloaded.get(key) != expected_value:
                raise AssertionError(f"{output_name}: metadata mismatch for {key}")

        output_file_hash = file_sha256(output_path)
        summary["caches"][summary_split] = {
            "source_path": str(source_path),
            "source_file_sha256": original_file_hash,
            "output_path": str(output_path),
            "output_file_sha256": output_file_hash,
            "features_semantic_sha256_before": before_semantic,
            "features_semantic_sha256_after": after_semantic,
            "cached_images": len(reloaded["features_by_image"]),
            "feature_count": reloaded.get("feature_count"),
        }

        print(f"PASS: {summary_split} metadata augmentation")
        print(f"  source: {source_path}")
        print(f"  output: {output_path}")
        print(f"  semantic SHA-256: {before_semantic}")
        print(f"  output SHA-256: {output_file_hash}")

    train_semantic = summary["caches"]["train"]["features_semantic_sha256_after"]
    validation_semantic = summary["caches"]["validation"]["features_semantic_sha256_after"]
    if train_semantic == validation_semantic:
        raise AssertionError("Train and validation semantic hashes unexpectedly match")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"PASS: summary written: {summary_path}")
    print("FIBE CACHE METADATA AUGMENTATION: PASS")


if __name__ == "__main__":
    main()
