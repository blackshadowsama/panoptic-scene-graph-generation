from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (PROJECT_ROOT / "data" / "floodpsg").resolve()

ANNOTATION_PATH = (
    DATA_ROOT
    / "annotations"
    / "floodpsg_canonical_trainval_coarse8_groupstrict_v1.json"
)
MANIFEST_PATH = (
    DATA_ROOT
    / "features"
    / "fibe_scalar_v1"
    / "fibe_canonical_mask_manifest_v1.csv"
)
SPEC_PATH = (
    PROJECT_ROOT
    / "configs"
    / "floodpsg"
    / "fibe_scalar_v1.json"
)
SUMMARY_PATH = (
    DATA_ROOT
    / "stats"
    / "dsformer_fibe_d1_v1"
    / "FIBE_CACHE_INPUT_AUDIT_V1.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_mask_path(value: Any) -> Path:
    path = Path(str(value).strip()).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (DATA_ROOT / path).resolve()


def load_annotation_records(payload: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if isinstance(payload, list):
        return payload, {}

    if not isinstance(payload, dict):
        raise TypeError(f"Unexpected annotation type: {type(payload)!r}")

    for key in ("data", "images", "records"):
        value = payload.get(key)
        if isinstance(value, list):
            return value, payload

    raise KeyError(
        "Could not find annotation image records under data/images/records"
    )


def main() -> None:
    for path in (ANNOTATION_PATH, MANIFEST_PATH, SPEC_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)

    annotation_payload = json.loads(
        ANNOTATION_PATH.read_text(encoding="utf-8")
    )
    records, annotation_root = load_annotation_records(annotation_payload)

    manifest = pd.read_csv(
        MANIFEST_PATH,
        encoding="utf-8-sig",
        low_memory=False,
    )
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))

    required_manifest_columns = {
        "image_id",
        "canonical_object_index",
        "category_name",
        "local_mask_path",
        "split",
    }
    missing_columns = required_manifest_columns - set(manifest.columns)
    if missing_columns:
        raise RuntimeError(
            f"Manifest missing columns: {sorted(missing_columns)}"
        )

    manifest["image_id"] = pd.to_numeric(
        manifest["image_id"], errors="raise"
    ).astype(int)
    manifest["canonical_object_index"] = pd.to_numeric(
        manifest["canonical_object_index"], errors="raise"
    ).astype(int)

    validation_ids = {
        int(value)
        for value in annotation_root.get("test_image_ids", [])
    }

    annotation_by_id: dict[int, dict[str, Any]] = {}
    annotation_object_counts: dict[int, int] = {}
    duplicate_annotation_ids: list[int] = []

    for record in records:
        image_id = int(record["image_id"])
        if image_id in annotation_by_id:
            duplicate_annotation_ids.append(image_id)
            continue

        segments = record.get("segments_info")
        if not isinstance(segments, list):
            raise TypeError(
                f"image_id={image_id} has no list-valued segments_info"
            )

        annotation_by_id[image_id] = record
        annotation_object_counts[image_id] = len(segments)

    manifest_image_ids = set(manifest["image_id"].tolist())
    annotation_image_ids = set(annotation_by_id)

    missing_manifest_images = sorted(annotation_image_ids - manifest_image_ids)
    extra_manifest_images = sorted(manifest_image_ids - annotation_image_ids)

    duplicate_keys = int(
        manifest.duplicated(
            subset=["image_id", "canonical_object_index"]
        ).sum()
    )
    duplicate_paths = int(manifest["local_mask_path"].duplicated().sum())

    split_counts = (
        manifest.groupby("split", observed=True)
        .agg(
            objects=("canonical_object_index", "size"),
            images=("image_id", "nunique"),
        )
        .reset_index()
    )

    object_count_mismatches: list[dict[str, int]] = []
    index_gap_images: list[dict[str, Any]] = []
    split_mismatches: list[dict[str, Any]] = []
    missing_masks: list[dict[str, Any]] = []

    for image_id, group in manifest.groupby("image_id", sort=True):
        image_id = int(image_id)
        expected_count = annotation_object_counts.get(image_id)
        actual_count = int(len(group))

        if expected_count is None:
            continue

        if actual_count != expected_count:
            object_count_mismatches.append(
                {
                    "image_id": image_id,
                    "annotation_objects": int(expected_count),
                    "manifest_objects": actual_count,
                }
            )

        actual_indexes = sorted(
            int(value) for value in group["canonical_object_index"]
        )
        expected_indexes = list(range(expected_count))

        if actual_indexes != expected_indexes:
            index_gap_images.append(
                {
                    "image_id": image_id,
                    "expected_indexes": expected_indexes,
                    "actual_indexes": actual_indexes,
                }
            )

        split_values = sorted(set(group["split"].astype(str)))
        expected_split = (
            "validation" if image_id in validation_ids else "train"
        )

        if split_values != [expected_split]:
            split_mismatches.append(
                {
                    "image_id": image_id,
                    "expected_split": expected_split,
                    "manifest_splits": split_values,
                }
            )

        for row in group.itertuples(index=False):
            resolved = resolve_mask_path(row.local_mask_path)
            if not resolved.is_file():
                missing_masks.append(
                    {
                        "image_id": image_id,
                        "canonical_object_index": int(
                            row.canonical_object_index
                        ),
                        "path": str(resolved),
                    }
                )

    category_counts = Counter(
        manifest["category_name"].astype(str).tolist()
    )

    first_record = records[0]
    first_segments = first_record.get("segments_info", [])
    first_segment = first_segments[0] if first_segments else {}

    errors = {
        "duplicate_annotation_ids": len(duplicate_annotation_ids),
        "missing_manifest_images": len(missing_manifest_images),
        "extra_manifest_images": len(extra_manifest_images),
        "duplicate_manifest_keys": duplicate_keys,
        "duplicate_manifest_paths": duplicate_paths,
        "object_count_mismatches": len(object_count_mismatches),
        "index_gap_images": len(index_gap_images),
        "split_mismatches": len(split_mismatches),
        "missing_masks": len(missing_masks),
    }

    summary = {
        "version": "FIBE_CACHE_INPUT_AUDIT_V1",
        "annotation_path": str(ANNOTATION_PATH),
        "annotation_sha256": sha256_file(ANNOTATION_PATH),
        "manifest_path": str(MANIFEST_PATH),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "spec_path": str(SPEC_PATH),
        "spec_sha256": sha256_file(SPEC_PATH),
        "spec_version": spec.get("version"),
        "feature_count": spec.get("feature_count"),
        "annotation_images": len(annotation_by_id),
        "annotation_objects": int(sum(annotation_object_counts.values())),
        "validation_ids": len(validation_ids),
        "manifest_rows": int(len(manifest)),
        "manifest_images": int(manifest["image_id"].nunique()),
        "split_counts": split_counts.to_dict(orient="records"),
        "category_counts": dict(sorted(category_counts.items())),
        "first_record_keys": sorted(first_record.keys()),
        "first_segment_keys": sorted(first_segment.keys()),
        "errors": errors,
        "previews": {
            "duplicate_annotation_ids": duplicate_annotation_ids[:20],
            "missing_manifest_images": missing_manifest_images[:20],
            "extra_manifest_images": extra_manifest_images[:20],
            "object_count_mismatches": object_count_mismatches[:20],
            "index_gap_images": index_gap_images[:10],
            "split_mismatches": split_mismatches[:20],
            "missing_masks": missing_masks[:20],
        },
    }

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 100)
    print("FIBE CACHE INPUT AUDIT V1")
    print("=" * 100)
    print("data root:", DATA_ROOT)
    print("annotation images:", len(annotation_by_id))
    print("annotation objects:", sum(annotation_object_counts.values()))
    print("validation IDs:", len(validation_ids))
    print("manifest rows:", len(manifest))
    print("manifest images:", manifest["image_id"].nunique())

    print("\nSPLIT COUNTS")
    print(split_counts.to_string(index=False))

    print("\nFIRST ANNOTATION RECORD KEYS")
    print(sorted(first_record.keys()))

    print("\nFIRST SEGMENT KEYS")
    print(sorted(first_segment.keys()))

    print("\nCATEGORY COUNTS")
    for name, count in sorted(category_counts.items()):
        print(f"{name}: {count}")

    print("\nERROR COUNTS")
    for name, count in errors.items():
        print(f"{name}: {count}")

    print("\nHASHES")
    print("annotation:", summary["annotation_sha256"])
    print("manifest:", summary["manifest_sha256"])
    print("spec:", summary["spec_sha256"])

    print("\nOUTPUT")
    print("summary:", SUMMARY_PATH)

    if any(errors.values()):
        raise SystemExit("FIBE CACHE INPUT AUDIT: FAIL")

    expected = {
        "annotation_images": 1673,
        "annotation_objects": 12409,
        "validation_ids": 173,
        "manifest_rows": 12409,
        "manifest_images": 1673,
    }
    actual = {
        key: summary[key]
        for key in expected
    }

    if actual != expected:
        raise SystemExit(
            "FIBE CACHE INPUT AUDIT: FAIL expected-count mismatch: "
            f"actual={actual}, expected={expected}"
        )

    expected_split = {
        "train": {"images": 1500, "objects": 11086},
        "validation": {"images": 173, "objects": 1323},
    }
    actual_split = {
        str(row["split"]): {
            "images": int(row["images"]),
            "objects": int(row["objects"]),
        }
        for row in summary["split_counts"]
    }

    if actual_split != expected_split:
        raise SystemExit(
            "FIBE CACHE INPUT AUDIT: FAIL split-count mismatch: "
            f"actual={actual_split}, expected={expected_split}"
        )

    print("\nFIBE CACHE INPUT AUDIT: PASS")


if __name__ == "__main__":
    main()
