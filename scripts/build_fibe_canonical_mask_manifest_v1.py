from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (PROJECT_ROOT / "data" / "floodpsg").resolve()

DEFAULT_ANNOTATION = (
    DATA_ROOT
    / "annotations"
    / "floodpsg_canonical_trainval_coarse8_groupstrict_v1.json"
)

DEFAULT_OUTPUT_ROOT = (
    DATA_ROOT
    / "binary_masks_canonical_v1"
)

DEFAULT_MANIFEST = (
    DATA_ROOT
    / "features"
    / "fibe_scalar_v1"
    / "fibe_canonical_mask_manifest_v1.csv"
)

DEFAULT_SUMMARY = (
    DATA_ROOT
    / "stats"
    / "dsformer_fibe_d1_v1"
    / "FIBE_CANONICAL_MASK_STAGING_V1.json"
)

DEFAULT_FAILURES = (
    DATA_ROOT
    / "stats"
    / "dsformer_fibe_d1_v1"
    / "FIBE_CANONICAL_MASK_STAGING_FAILURES_V1.tsv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy the independent binary masks referenced by the "
            "FloodPSG canonical annotation into the WSL data root "
            "and build a one-to-one canonical object manifest."
        )
    )

    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help=(
            "Path to groundingdino_batch_output/full_psg, "
            "containing batch_001 ... batch_007."
        ),
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=DEFAULT_ANNOTATION,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
    )
    parser.add_argument(
        "--failures",
        type=Path,
        default=DEFAULT_FAILURES,
    )

    return parser.parse_args()


def file_sha256(
    path: Path,
    *,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def annotation_sha256(path: Path) -> str:
    return file_sha256(path)


def parse_validation_ids(
    annotation: dict[str, Any],
) -> set[int]:
    result: set[int] = set()

    for value in annotation.get(
        "test_image_ids",
        [],
    ):
        result.add(int(value))

    return result


def resolve_source_mask(
    raw_root: Path,
    source_mask_path: str,
) -> Path:
    normalized = source_mask_path.replace(
        "\\",
        "/",
    )

    marker = "/full_psg/"
    lower = normalized.lower()

    if marker not in lower:
        raise ValueError(
            "source_mask_path does not contain "
            f"{marker!r}: {source_mask_path}"
        )

    marker_index = lower.index(marker)

    suffix = normalized[
        marker_index + len(marker):
    ]

    return raw_root / Path(suffix)


def inspect_mask(
    path: Path,
) -> dict[str, Any]:
    with Image.open(path) as image:
        width, height = image.size
        mode = image.mode

        grayscale = np.asarray(
            image.convert("L"),
            dtype=np.uint8,
        )

    foreground = int(
        np.count_nonzero(grayscale)
    )

    unique_values = np.unique(grayscale)

    return {
        "width": int(width),
        "height": int(height),
        "mode": str(mode),
        "foreground_pixels": foreground,
        "unique_value_count": int(
            len(unique_values)
        ),
        "min_value": int(unique_values.min()),
        "max_value": int(unique_values.max()),
    }


def atomic_copy(
    source: Path,
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    if temporary.exists():
        temporary.unlink()

    shutil.copy2(
        source,
        temporary,
    )

    os.replace(
        temporary,
        destination,
    )


def write_failures(
    path: Path,
    failures: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "image_id",
        "global_image_key",
        "canonical_object_index",
        "canonical_object_key",
        "failure_type",
        "message",
        "source_mask_path",
        "resolved_source_path",
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(failures)


def main() -> None:
    args = parse_args()

    raw_root = args.raw_root.expanduser().resolve()
    annotation_path = (
        args.annotation.expanduser().resolve()
    )
    output_root = (
        args.output_root.expanduser().resolve()
    )
    manifest_path = (
        args.manifest.expanduser().resolve()
    )
    summary_path = (
        args.summary.expanduser().resolve()
    )
    failures_path = (
        args.failures.expanduser().resolve()
    )

    if not raw_root.is_dir():
        raise FileNotFoundError(
            f"Raw full_psg root not found: "
            f"{raw_root}"
        )

    if not annotation_path.is_file():
        raise FileNotFoundError(
            annotation_path
        )

    annotation = json.loads(
        annotation_path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        annotation.get("data"),
        list,
    ):
        raise RuntimeError(
            "Annotation has no valid data list"
        )

    validation_ids = parse_validation_ids(
        annotation
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    seen_destination_paths: set[str] = set()
    seen_canonical_keys: set[str] = set()

    split_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    batch_counts: Counter[str] = Counter()

    copied = 0
    reused = 0
    overwritten = 0
    total_source_bytes = 0

    for item in annotation["data"]:
        image_id = int(item["image_id"])
        global_image_key = str(
            item["global_image_key"]
        )
        image_width = int(item["width"])
        image_height = int(item["height"])

        split = (
            "validation"
            if image_id in validation_ids
            else "train"
        )

        batch_id = global_image_key.split(
            "__",
            1,
        )[0]

        segments = item.get(
            "segments_info",
            [],
        )

        for object_index, segment in enumerate(
            segments
        ):
            canonical_object_key = str(
                segment.get(
                    "canonical_object_key",
                    "",
                )
            )

            source_mask_path = str(
                segment.get(
                    "source_mask_path",
                    "",
                )
            ).strip()

            resolved_source = Path()

            try:
                if not canonical_object_key:
                    raise ValueError(
                        "Missing canonical_object_key"
                    )

                if (
                    canonical_object_key
                    in seen_canonical_keys
                ):
                    raise ValueError(
                        "Duplicate canonical_object_key"
                    )

                if not source_mask_path:
                    raise ValueError(
                        "Missing source_mask_path"
                    )

                resolved_source = (
                    resolve_source_mask(
                        raw_root,
                        source_mask_path,
                    )
                )

                if not resolved_source.is_file():
                    raise FileNotFoundError(
                        resolved_source
                    )

                local_relative = (
                    Path(
                        "binary_masks_canonical_v1"
                    )
                    / f"{image_id:06d}"
                    / (
                        f"object_"
                        f"{object_index:04d}.png"
                    )
                )

                destination = (
                    DATA_ROOT / local_relative
                ).resolve()

                destination_key = str(
                    destination
                )

                if (
                    destination_key
                    in seen_destination_paths
                ):
                    raise ValueError(
                        "Duplicate destination path"
                    )

                source_info = inspect_mask(
                    resolved_source
                )

                if (
                    source_info["width"]
                    != image_width
                    or source_info["height"]
                    != image_height
                ):
                    raise ValueError(
                        "Mask/image dimension mismatch: "
                        f"mask="
                        f"{source_info['width']}x"
                        f"{source_info['height']}, "
                        f"image="
                        f"{image_width}x"
                        f"{image_height}"
                    )

                if (
                    source_info[
                        "foreground_pixels"
                    ]
                    <= 0
                ):
                    raise ValueError(
                        "Binary mask has no "
                        "foreground pixels"
                    )

                source_hash = file_sha256(
                    resolved_source
                )

                destination_existed = (
                    destination.is_file()
                )

                if destination_existed:
                    destination_hash = (
                        file_sha256(destination)
                    )

                    if (
                        destination_hash
                        == source_hash
                    ):
                        reused += 1
                    else:
                        atomic_copy(
                            resolved_source,
                            destination,
                        )
                        overwritten += 1
                else:
                    atomic_copy(
                        resolved_source,
                        destination,
                    )
                    copied += 1

                destination_hash = file_sha256(
                    destination
                )

                if (
                    destination_hash
                    != source_hash
                ):
                    raise RuntimeError(
                        "Copied file SHA256 mismatch"
                    )

                destination_info = inspect_mask(
                    destination
                )

                if (
                    destination_info
                    != source_info
                ):
                    raise RuntimeError(
                        "Copied mask metadata mismatch"
                    )

                source_bytes = int(
                    resolved_source.stat().st_size
                )
                total_source_bytes += source_bytes

                category_name = str(
                    segment.get(
                        "category_name",
                        "",
                    )
                )

                row = {
                    "split": split,
                    "batch_id": batch_id,
                    "image_id": image_id,
                    "global_image_key": (
                        global_image_key
                    ),
                    "image_width": image_width,
                    "image_height": image_height,
                    "canonical_object_index": (
                        int(object_index)
                    ),
                    "segment_id": int(
                        segment["id"]
                    ),
                    "category_id": int(
                        segment["category_id"]
                    ),
                    "category_name": (
                        category_name
                    ),
                    "isthing": int(
                        bool(
                            segment.get(
                                "isthing",
                                False,
                            )
                        )
                    ),
                    "canonical_object_key": (
                        canonical_object_key
                    ),
                    "object_id": str(
                        segment.get(
                            "object_id",
                            "",
                        )
                    ),
                    "object_uid": str(
                        segment.get(
                            "object_uid",
                            "",
                        )
                    ),
                    "source_row_id": str(
                        segment.get(
                            "source_row_id",
                            "",
                        )
                    ),
                    "source_mask_path": (
                        source_mask_path
                    ),
                    "resolved_source_path": (
                        str(
                            resolved_source.resolve()
                        )
                    ),
                    "local_mask_path": (
                        local_relative.as_posix()
                    ),
                    "mask_width": (
                        source_info["width"]
                    ),
                    "mask_height": (
                        source_info["height"]
                    ),
                    "mask_mode": (
                        source_info["mode"]
                    ),
                    "foreground_pixels": (
                        source_info[
                            "foreground_pixels"
                        ]
                    ),
                    "canonical_visible_area": int(
                        segment.get(
                            "area",
                            0,
                        )
                    ),
                    "unique_value_count": (
                        source_info[
                            "unique_value_count"
                        ]
                    ),
                    "min_value": (
                        source_info["min_value"]
                    ),
                    "max_value": (
                        source_info["max_value"]
                    ),
                    "source_file_bytes": (
                        source_bytes
                    ),
                    "sha256": source_hash,
                }

                rows.append(row)

                seen_canonical_keys.add(
                    canonical_object_key
                )
                seen_destination_paths.add(
                    destination_key
                )

                split_counts[split] += 1
                category_counts[
                    category_name
                ] += 1
                batch_counts[batch_id] += 1

            except Exception as error:
                failures.append(
                    {
                        "image_id": image_id,
                        "global_image_key": (
                            global_image_key
                        ),
                        "canonical_object_index": (
                            object_index
                        ),
                        "canonical_object_key": (
                            canonical_object_key
                        ),
                        "failure_type": type(
                            error
                        ).__name__,
                        "message": str(error),
                        "source_mask_path": (
                            source_mask_path
                        ),
                        "resolved_source_path": (
                            str(resolved_source)
                            if str(resolved_source)
                            != "."
                            else ""
                        ),
                    }
                )

    write_failures(
        failures_path,
        failures,
    )

    expected_objects = sum(
        len(item.get("segments_info", []))
        for item in annotation["data"]
    )

    expected_images = len(
        annotation["data"]
    )

    if failures:
        raise SystemExit(
            f"FAIL: {len(failures)} mask "
            f"staging failures. See "
            f"{failures_path}"
        )

    if len(rows) != expected_objects:
        raise SystemExit(
            "FAIL: manifest row count mismatch: "
            f"{len(rows)} != {expected_objects}"
        )

    if (
        len(seen_canonical_keys)
        != expected_objects
    ):
        raise SystemExit(
            "FAIL: canonical key uniqueness "
            "check failed"
        )

    if (
        len(seen_destination_paths)
        != expected_objects
    ):
        raise SystemExit(
            "FAIL: local mask path uniqueness "
            "check failed"
        )

    rows.sort(
        key=lambda row: (
            int(row["image_id"]),
            int(
                row[
                    "canonical_object_index"
                ]
            ),
        )
    )

    fieldnames = list(rows[0].keys())

    temporary_manifest = (
        manifest_path.with_suffix(
            manifest_path.suffix + ".tmp"
        )
    )

    with temporary_manifest.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    os.replace(
        temporary_manifest,
        manifest_path,
    )

    summary = {
        "version": (
            "FIBE_CANONICAL_MASK_STAGING_V1"
        ),
        "project_root": str(PROJECT_ROOT),
        "data_root": str(DATA_ROOT),
        "raw_root": str(raw_root),
        "annotation_path": str(
            annotation_path
        ),
        "annotation_sha256": (
            annotation_sha256(
                annotation_path
            )
        ),
        "output_root": str(output_root),
        "manifest_path": str(
            manifest_path
        ),
        "failures_path": str(
            failures_path
        ),
        "canonical_images": (
            expected_images
        ),
        "canonical_objects": (
            expected_objects
        ),
        "manifest_rows": len(rows),
        "validation_image_ids": len(
            validation_ids
        ),
        "copied_files": copied,
        "reused_files": reused,
        "overwritten_files": overwritten,
        "failure_count": len(failures),
        "unique_canonical_keys": len(
            seen_canonical_keys
        ),
        "unique_local_paths": len(
            seen_destination_paths
        ),
        "total_source_bytes": (
            total_source_bytes
        ),
        "split_object_counts": dict(
            sorted(split_counts.items())
        ),
        "batch_object_counts": dict(
            sorted(batch_counts.items())
        ),
        "category_object_counts": dict(
            sorted(category_counts.items())
        ),
    }

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 100)
    print(
        "FIBE CANONICAL MASK STAGING V1"
    )
    print("=" * 100)
    print("raw root:", raw_root)
    print(
        "annotation:",
        annotation_path,
    )
    print(
        "canonical images:",
        expected_images,
    )
    print(
        "canonical objects:",
        expected_objects,
    )
    print(
        "manifest rows:",
        len(rows),
    )
    print("copied:", copied)
    print("reused:", reused)
    print(
        "overwritten:",
        overwritten,
    )
    print(
        "failures:",
        len(failures),
    )
    print(
        "unique canonical keys:",
        len(seen_canonical_keys),
    )
    print(
        "unique local paths:",
        len(seen_destination_paths),
    )

    print("\nSPLIT OBJECT COUNTS")

    for name, count in sorted(
        split_counts.items()
    ):
        print(f"{name}: {count}")

    print("\nBATCH OBJECT COUNTS")

    for name, count in sorted(
        batch_counts.items()
    ):
        print(f"{name}: {count}")

    print("\nCATEGORY OBJECT COUNTS")

    for name, count in sorted(
        category_counts.items()
    ):
        print(f"{name}: {count}")

    print("\nOUTPUTS")
    print("mask root:", output_root)
    print("manifest:", manifest_path)
    print("summary:", summary_path)
    print(
        "failure report:",
        failures_path,
    )

    print("\nFIBE CANONICAL MASK STAGING: PASS")


if __name__ == "__main__":
    main()
