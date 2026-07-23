from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (PROJECT_ROOT / "data" / "floodpsg").resolve()

DEFAULT_ANNOTATION = (
    DATA_ROOT
    / "annotations"
    / "floodpsg_canonical_trainval_coarse8_groupstrict_v1.json"
)
DEFAULT_FEATURE_DIR = (
    DATA_ROOT
    / "features"
    / "fibe_scalar_v1"
)
DEFAULT_OUTPUT = (
    DATA_ROOT
    / "stats"
    / "dsformer_fibe_d1_v1"
    / "FIBE_RELATION_CACHE_ALIGNMENT_V1.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit alignment between canonical relation endpoints and "
            "formal FIBE train/validation caches."
        )
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=DEFAULT_ANNOTATION,
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=DEFAULT_FEATURE_DIR,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--expected-train-relations",
        type=int,
        default=3232,
    )
    parser.add_argument(
        "--expected-validation-relations",
        type=int,
        default=381,
    )
    parser.add_argument(
        "--require-all-relations-valid",
        action="store_true",
    )
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
        records = payload.get(key)
        if isinstance(records, list):
            return records, payload
    raise KeyError("Could not find records under data/images/records")


def relation_triplet(value: Any) -> tuple[int, int, int]:
    if isinstance(value, (list, tuple)):
        if len(value) < 3:
            raise ValueError(f"Relation sequence is too short: {value!r}")
        return int(value[0]), int(value[1]), int(value[2])

    if not isinstance(value, dict):
        raise TypeError(f"Unsupported relation type: {type(value)!r}")

    subject_keys = (
        "subject_idx",
        "subject_index",
        "subject_id",
        "sub_idx",
        "sub_id",
        "subject",
    )
    object_keys = (
        "object_idx",
        "object_index",
        "object_id",
        "obj_idx",
        "obj_id",
        "object",
    )
    predicate_keys = (
        "predicate_idx",
        "predicate_index",
        "predicate_id",
        "predicate",
        "relation",
    )

    def first_present(keys: tuple[str, ...]) -> Any:
        for key in keys:
            if key in value:
                return value[key]
        raise KeyError(
            f"None of the expected keys {keys!r} appear in {value!r}"
        )

    return (
        int(first_present(subject_keys)),
        int(first_present(object_keys)),
        int(first_present(predicate_keys)),
    )


def predicate_name(
    predicate_id: int,
    root: dict[str, Any],
) -> str:
    for key in (
        "predicate_classes",
        "predicate_categories",
        "relation_classes",
    ):
        classes = root.get(key)
        if isinstance(classes, list) and 0 <= predicate_id < len(classes):
            return str(classes[predicate_id])
    return str(predicate_id)


def audit_split(
    *,
    split: str,
    records: list[dict[str, Any]],
    cache_payload: dict[str, Any],
    annotation_root: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    features_by_image = cache_payload["features_by_image"]

    relation_count = 0
    valid_relation_count = 0
    invalid_relation_count = 0
    endpoint_errors = 0
    missing_cache_images = 0
    object_count_mismatches = 0
    feature_flag_mismatches = 0
    relation_images = 0

    invalid_examples: list[dict[str, Any]] = []
    category_pair_counts: Counter[str] = Counter()
    invalid_category_pair_counts: Counter[str] = Counter()
    predicate_counts: Counter[str] = Counter()
    invalid_predicate_counts: Counter[str] = Counter()

    record_ids = {int(record["image_id"]) for record in records}
    cache_ids = {int(value) for value in features_by_image}
    missing_from_cache = sorted(record_ids - cache_ids)
    extra_in_cache = sorted(cache_ids - record_ids)

    missing_cache_images = len(missing_from_cache)

    for record in records:
        image_id = int(record["image_id"])
        relations = record.get("relations", [])
        segments = record.get("segments_info", [])

        if relations:
            relation_images += 1

        entry = features_by_image.get(image_id)
        if entry is None:
            entry = features_by_image.get(str(image_id))
        if entry is None:
            relation_count += len(relations)
            continue

        num_objects = int(entry["num_objects"])
        valid_pairs = entry["valid_pairs"]
        features = entry["features"]

        if num_objects != len(segments):
            object_count_mismatches += 1
            continue

        if tuple(valid_pairs.shape) != (num_objects, num_objects):
            raise RuntimeError(
                f"{split}/{image_id}: invalid valid_pairs shape "
                f"{tuple(valid_pairs.shape)}"
            )

        if tuple(features.shape) != (num_objects, num_objects, 21):
            raise RuntimeError(
                f"{split}/{image_id}: invalid feature shape "
                f"{tuple(features.shape)}"
            )

        for raw_relation in relations:
            relation_count += 1
            subject_idx, object_idx, predicate_id = relation_triplet(
                raw_relation
            )

            if not (
                0 <= subject_idx < num_objects
                and 0 <= object_idx < num_objects
            ):
                endpoint_errors += 1
                if len(invalid_examples) < 30:
                    invalid_examples.append(
                        {
                            "split": split,
                            "image_id": image_id,
                            "reason": "endpoint_out_of_range",
                            "relation": [
                                subject_idx,
                                object_idx,
                                predicate_id,
                            ],
                            "num_objects": num_objects,
                        }
                    )
                continue

            subject_category = str(
                segments[subject_idx]["category_name"]
            )
            object_category = str(
                segments[object_idx]["category_name"]
            )
            pair_name = f"{subject_category}->{object_category}"
            pred_name = predicate_name(predicate_id, annotation_root)

            category_pair_counts[pair_name] += 1
            predicate_counts[pred_name] += 1

            valid = bool(valid_pairs[subject_idx, object_idx].item())
            feature_flag = bool(
                features[subject_idx, object_idx, 20].item() > 0.5
            )

            if feature_flag != valid:
                feature_flag_mismatches += 1

            if valid:
                valid_relation_count += 1
            else:
                invalid_relation_count += 1
                invalid_category_pair_counts[pair_name] += 1
                invalid_predicate_counts[pred_name] += 1

                if len(invalid_examples) < 30:
                    invalid_examples.append(
                        {
                            "split": split,
                            "image_id": image_id,
                            "reason": "fibe_pair_invalid",
                            "relation": [
                                subject_idx,
                                object_idx,
                                predicate_id,
                            ],
                            "subject_category": subject_category,
                            "object_category": object_category,
                            "predicate": pred_name,
                        }
                    )

    summary = {
        "split": split,
        "records": len(records),
        "cache_images": len(features_by_image),
        "relation_images": relation_images,
        "relations": relation_count,
        "valid_relations": valid_relation_count,
        "invalid_relations": invalid_relation_count,
        "valid_relation_fraction": (
            float(valid_relation_count / relation_count)
            if relation_count
            else 0.0
        ),
        "endpoint_errors": endpoint_errors,
        "missing_cache_images": missing_cache_images,
        "extra_cache_images": len(extra_in_cache),
        "object_count_mismatches": object_count_mismatches,
        "feature_flag_mismatches": feature_flag_mismatches,
        "missing_cache_image_ids": missing_from_cache[:30],
        "extra_cache_image_ids": extra_in_cache[:30],
        "category_pair_counts": dict(
            sorted(category_pair_counts.items())
        ),
        "invalid_category_pair_counts": dict(
            sorted(invalid_category_pair_counts.items())
        ),
        "predicate_counts": dict(sorted(predicate_counts.items())),
        "invalid_predicate_counts": dict(
            sorted(invalid_predicate_counts.items())
        ),
    }
    return summary, invalid_examples


def main() -> None:
    args = parse_args()

    annotation_path = args.annotation.expanduser().resolve()
    feature_dir = args.feature_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    train_cache_path = feature_dir / "train_features.pt"
    validation_cache_path = feature_dir / "validation_features.pt"

    for path in (
        annotation_path,
        train_cache_path,
        validation_cache_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    annotation_payload = json.loads(
        annotation_path.read_text(encoding="utf-8")
    )
    records, annotation_root = load_records(annotation_payload)
    validation_ids = {
        int(value)
        for value in annotation_root.get("test_image_ids", [])
    }

    train_records = [
        record
        for record in records
        if int(record["image_id"]) not in validation_ids
    ]
    validation_records = [
        record
        for record in records
        if int(record["image_id"]) in validation_ids
    ]

    train_payload = torch.load(train_cache_path, map_location="cpu")
    validation_payload = torch.load(
        validation_cache_path,
        map_location="cpu",
    )

    train_summary, train_invalid_examples = audit_split(
        split="train",
        records=train_records,
        cache_payload=train_payload,
        annotation_root=annotation_root,
    )
    validation_summary, validation_invalid_examples = audit_split(
        split="validation",
        records=validation_records,
        cache_payload=validation_payload,
        annotation_root=annotation_root,
    )

    errors: list[str] = []

    if train_summary["relations"] != args.expected_train_relations:
        errors.append(
            "train relation count mismatch: "
            f"{train_summary['relations']} != "
            f"{args.expected_train_relations}"
        )

    if (
        validation_summary["relations"]
        != args.expected_validation_relations
    ):
        errors.append(
            "validation relation count mismatch: "
            f"{validation_summary['relations']} != "
            f"{args.expected_validation_relations}"
        )

    for split_summary in (train_summary, validation_summary):
        split = split_summary["split"]

        for field in (
            "endpoint_errors",
            "missing_cache_images",
            "extra_cache_images",
            "object_count_mismatches",
            "feature_flag_mismatches",
        ):
            if split_summary[field] != 0:
                errors.append(
                    f"{split}: {field}={split_summary[field]}"
                )

        if (
            args.require_all_relations_valid
            and split_summary["invalid_relations"] != 0
        ):
            errors.append(
                f"{split}: invalid_relations="
                f"{split_summary['invalid_relations']}"
            )

    output = {
        "version": "FIBE_RELATION_CACHE_ALIGNMENT_V1",
        "annotation_path": str(annotation_path),
        "annotation_sha256": sha256_file(annotation_path),
        "train_cache_path": str(train_cache_path),
        "train_cache_sha256": sha256_file(train_cache_path),
        "validation_cache_path": str(validation_cache_path),
        "validation_cache_sha256": sha256_file(
            validation_cache_path
        ),
        "require_all_relations_valid": bool(
            args.require_all_relations_valid
        ),
        "train": train_summary,
        "validation": validation_summary,
        "invalid_examples": (
            train_invalid_examples
            + validation_invalid_examples
        ),
        "errors": errors,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 100)
    print("FIBE RELATION/CACHE ALIGNMENT V1")
    print("=" * 100)

    for split_summary in (train_summary, validation_summary):
        print(f"\n{split_summary['split'].upper()}")
        print("records:", split_summary["records"])
        print("cache images:", split_summary["cache_images"])
        print("relation images:", split_summary["relation_images"])
        print("relations:", split_summary["relations"])
        print("valid relations:", split_summary["valid_relations"])
        print("invalid relations:", split_summary["invalid_relations"])
        print(
            "valid relation fraction:",
            split_summary["valid_relation_fraction"],
        )
        print("endpoint errors:", split_summary["endpoint_errors"])
        print(
            "missing cache images:",
            split_summary["missing_cache_images"],
        )
        print(
            "extra cache images:",
            split_summary["extra_cache_images"],
        )
        print(
            "object count mismatches:",
            split_summary["object_count_mismatches"],
        )
        print(
            "feature flag mismatches:",
            split_summary["feature_flag_mismatches"],
        )

        if split_summary["invalid_category_pair_counts"]:
            print("invalid category pairs:")
            for key, value in split_summary[
                "invalid_category_pair_counts"
            ].items():
                print(f"  {key}: {value}")

        if split_summary["invalid_predicate_counts"]:
            print("invalid predicates:")
            for key, value in split_summary[
                "invalid_predicate_counts"
            ].items():
                print(f"  {key}: {value}")

    print("\nOUTPUT")
    print("summary:", output_path)

    if errors:
        print("\nERRORS")
        for error in errors:
            print("-", error)
        raise SystemExit("FIBE RELATION/CACHE ALIGNMENT: FAIL")

    print("\nFIBE RELATION/CACHE ALIGNMENT: PASS")


if __name__ == "__main__":
    main()
