#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def replace_exact(
    text: str,
    old: str,
    new: str,
    *,
    label: str,
    expected: int = 1,
) -> str:
    count = text.count(old)
    if count != expected:
        raise RuntimeError(
            f"{label}: expected {expected} occurrences, found {count}"
        )
    return text.replace(old, new, expected)


def replace_regex(
    text: str,
    pattern: str,
    replacement,
    *,
    label: str,
) -> str:
    # Multiline regex literals below are formatted for readability.
    # Remove only their formatting newlines and leading indentation.
    # Spaces within quoted source strings remain unchanged.
    pattern = re.sub(r"\n[ \t]*", "", pattern)

    updated, count = re.subn(
        pattern,
        replacement,
        text,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise RuntimeError(
            f"{label}: expected one regex match, found {count}"
        )
    return updated


def build_manifest_test_script() -> Path:
    source = PROJECT_ROOT / "scripts/build_fibe_canonical_mask_manifest_v1.py"
    target = PROJECT_ROOT / "scripts/build_fibe_canonical_mask_manifest_test_v1.py"

    if not source.is_file():
        raise FileNotFoundError(source)
    if target.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing generated script: {target}"
        )

    text = source.read_text(encoding="utf-8")

    text = replace_exact(
        text,
        '"floodpsg_canonical_trainval_coarse8_groupstrict_v1.json"',
        '"floodpsg_canonical_full_coarse8_groupstrict_v1.json"',
        label="manifest default annotation",
    )

    # The source contains this literal twice: once for DEFAULT_OUTPUT_ROOT and
    # once for the relative local_mask_path prefix. Both must be isolated for
    # the final-test mask tree.
    text = replace_exact(
        text,
        '"binary_masks_canonical_v1"',
        '"binary_masks_canonical_test_v1"',
        label="manifest output/local-mask root",
        expected=2,
    )

    text = replace_exact(
        text,
        '"fibe_canonical_mask_manifest_v1.csv"',
        '"fibe_canonical_mask_manifest_test_v1.csv"',
        label="manifest filename",
    )
    text = replace_exact(
        text,
        '"FIBE_CANONICAL_MASK_STAGING_V1.json"',
        '"FIBE_CANONICAL_MASK_STAGING_TEST_V1.json"',
        label="manifest summary filename",
    )
    text = replace_exact(
        text,
        '"FIBE_CANONICAL_MASK_STAGING_FAILURES_V1.tsv"',
        '"FIBE_CANONICAL_MASK_STAGING_TEST_FAILURES_V1.tsv"',
        label="manifest failure filename",
    )

    selection_code = r'''
    record_ids = {
        int(item["image_id"])
        for item in annotation["data"]
    }

    missing_test_ids = sorted(validation_ids - record_ids)
    if missing_test_ids:
        raise RuntimeError(
            "Final-test IDs missing from annotation records: "
            f"{missing_test_ids[:20]}"
        )

    selected_data = [
        item
        for item in annotation["data"]
        if int(item["image_id"]) in validation_ids
    ]

    if len(selected_data) != len(validation_ids):
        raise RuntimeError(
            "Final-test record count mismatch: "
            f"records={len(selected_data)}, ids={len(validation_ids)}"
        )

    if len(selected_data) != 175:
        raise RuntimeError(
            "Expected 175 final-test images, got "
            f"{len(selected_data)}"
        )

    annotation = dict(annotation)
    annotation["data"] = selected_data
'''

    text = replace_regex(
        text,
        r'''(
            validation_ids\s*=\s*parse_validation_ids\(
            \s*annotation\s*
            \)
        )''',
        lambda match: match.group(1) + selection_code,
        label="insert final-test record selection",
    )

    text = replace_regex(
        text,
        r'''split\s*=\s*\(
            \s*"validation"\s*
            if\s+image_id\s+in\s+validation_ids\s*
            else\s+"train"\s*
            \)''',
        'split = "test"',
        label="manifest test split",
    )

    text = replace_exact(
        text,
        '"validation_image_ids": len(',
        '"test_image_ids": len(',
        label="manifest summary split key",
    )

    text = replace_exact(
        text,
        '"FIBE_CANONICAL_MASK_STAGING_V1"',
        '"FIBE_CANONICAL_MASK_STAGING_TEST_V1"',
        label="manifest summary version",
    )

    target.write_text(text, encoding="utf-8")
    return target


def build_feature_test_script() -> Path:
    source = PROJECT_ROOT / "scripts/build_fibe_scalar_features_v1.py"
    target = PROJECT_ROOT / "scripts/build_fibe_scalar_test_features_v1.py"

    if not source.is_file():
        raise FileNotFoundError(source)
    if target.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing generated script: {target}"
        )

    text = source.read_text(encoding="utf-8")

    text = replace_exact(
        text,
        '"floodpsg_canonical_trainval_coarse8_groupstrict_v1.json"',
        '"floodpsg_canonical_full_coarse8_groupstrict_v1.json"',
        label="feature default annotation",
    )
    text = replace_exact(
        text,
        '"fibe_canonical_mask_manifest_v1.csv"',
        '"fibe_canonical_mask_manifest_test_v1.csv"',
        label="feature default manifest",
    )

    text = replace_exact(
        text,
        "Build offline train/validation FIBE-Scalar V1 feature caches.",
        "Build the offline final-test FIBE-Scalar V1 feature cache.",
        label="feature description",
    )

    text = replace_regex(
        text,
        r'''choices\s*=\s*\(
            \s*"train"\s*,\s*"validation"\s*
            \)''',
        'choices=("test",)',
        label="feature split choices",
    )
    text = replace_regex(
        text,
        r'''default\s*=\s*\(
            \s*"train"\s*,\s*"validation"\s*
            \)''',
        'default=("test",)',
        label="feature default split",
    )

    text = replace_regex(
        text,
        r'''expected_split\s*=\s*\(
            \s*"validation"\s*
            if\s+int\(image_id\)\s+in\s+validation_ids\s*
            else\s+"train"\s*
            \)''',
        '''expected_split = (
                "test"
                if int(image_id) in validation_ids
                else "train"
            )''',
        label="feature expected test split",
    )

    text = replace_regex(
        text,
        r'''    scaler_name\s*=\s*tagged_name\(
            "scaler_v1\.json",\s*args\.output_tag
            \)
            \s*scaler_path\s*=\s*output_dir\s*/\s*scaler_name
            \s*if\s+scaler_path\.exists\(\)\s+and\s+not\s+args\.overwrite:
            \s*raise\s+FileExistsError\(
            \s*f"Refusing to overwrite \{scaler_path\}; pass --overwrite"
            \s*\)
            \s*atomic_json_save\(scaler,\s*scaler_path\)''',
        '''    scaler_name = tagged_name(
        "scaler_v1.json",
        args.output_tag,
    )
    scaler_path = output_dir / scaler_name

    if not scaler_path.is_file():
        raise FileNotFoundError(
            "Frozen training scaler is missing: "
            f"{scaler_path}"
        )

    # Final-test construction must never fit or rewrite the scaler.
    print(
        "Using frozen scaler without modification:",
        scaler_path,
    )''',
        label="disable test scaler rewrite",
    )

    if 'choices=("test",)' not in text:
        raise RuntimeError("Generated feature builder does not restrict split to test")
    if 'atomic_json_save(scaler, scaler_path)' in text:
        raise RuntimeError("Generated feature builder still rewrites the scaler")

    target.write_text(text, encoding="utf-8")
    return target


def main() -> None:
    manifest_target = build_manifest_test_script()
    feature_target = build_feature_test_script()

    print("Created:", manifest_target)
    print("Created:", feature_target)
    print("FIBE FINAL-TEST BUILDERS: CREATED")


if __name__ == "__main__":
    main()
