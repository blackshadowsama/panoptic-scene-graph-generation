from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (
    PROJECT_ROOT / "data" / "floodpsg"
).resolve()

DEFAULT_REVIEW = (
    DATA_ROOT
    / "features"
    / "fibe_scalar_v1"
    / "fibe_boundary_audit_review_v1.csv"
)

DEFAULT_KEY = (
    DATA_ROOT
    / "features"
    / "fibe_scalar_v1"
    / "fibe_boundary_audit_sample_key_v1.csv"
)

DEFAULT_LABEL_DIR = (
    DATA_ROOT
    / "features"
    / "fibe_scalar_v1"
    / "boundary_audit_labels_v1"
)

DEFAULT_FINAL_REVIEW = (
    DATA_ROOT
    / "features"
    / "fibe_scalar_v1"
    / "fibe_boundary_audit_review_final_v1.csv"
)

DEFAULT_UNBLINDED = (
    DATA_ROOT
    / "features"
    / "fibe_scalar_v1"
    / "fibe_boundary_audit_unblinded_v1.csv"
)

DEFAULT_SUMMARY = (
    DATA_ROOT
    / "stats"
    / "dsformer_fibe_d1_v1"
    / "FIBE_BOUNDARY_AUDIT_FINAL_V1.json"
)

DEFAULT_ERRORS = (
    DATA_ROOT
    / "stats"
    / "dsformer_fibe_d1_v1"
    / "FIBE_BOUNDARY_AUDIT_FINAL_ERRORS_V1.csv"
)


MANUAL_FIELDS = [
    "manual_review_status",
    "manual_mask_pair_quality",
    "manual_geometry_relation",
    "manual_semantic_contact",
    "manual_apparent_boundary_quality",
    "manual_failure_mode",
    "manual_confidence",
    "manual_notes",
    "manual_saved_at",
    "manual_annotator",
    "manual_revision",
]


CHOICES = {
    "manual_review_status": {
        "complete",
        "exclude",
    },
    "manual_mask_pair_quality": {
        "valid",
        "target_mask_error",
        "hazard_mask_error",
        "both_mask_error",
        "ambiguous",
    },
    "manual_geometry_relation": {
        "overlap",
        "touching",
        "near_gap",
        "far_gap",
        "ambiguous",
    },
    "manual_semantic_contact": {
        "contact",
        "not_contact",
        "ambiguous",
    },
    "manual_apparent_boundary_quality": {
        "valid",
        "invalid",
        "not_applicable",
        "ambiguous",
    },
    "manual_confidence": {
        "high",
        "medium",
        "low",
    },
}


STATIC_FIELDS = [
    "image_id",
    "global_image_key",
    "target_index",
    "hazard_index",
    "target_category",
    "hazard_category",
    "pair_family",
]


JSON_ANNOTATION_FIELDS = [
    "manual_review_status",
    "manual_mask_pair_quality",
    "manual_geometry_relation",
    "manual_semantic_contact",
    "manual_apparent_boundary_quality",
    "manual_failure_mode",
    "manual_confidence",
    "manual_notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate, freeze and unblind the completed "
            "FIBE boundary audit."
        )
    )

    parser.add_argument(
        "--review",
        type=Path,
        default=DEFAULT_REVIEW,
    )
    parser.add_argument(
        "--key",
        type=Path,
        default=DEFAULT_KEY,
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=DEFAULT_LABEL_DIR,
    )
    parser.add_argument(
        "--final-review",
        type=Path,
        default=DEFAULT_FINAL_REVIEW,
    )
    parser.add_argument(
        "--unblinded",
        type=Path,
        default=DEFAULT_UNBLINDED,
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
    )
    parser.add_argument(
        "--errors",
        type=Path,
        default=DEFAULT_ERRORS,
    )

    return parser.parse_args()


def clean_text(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def normalize_integer(value: Any) -> str:
    text = clean_text(value)

    if not text:
        return ""

    try:
        return str(int(float(text)))
    except ValueError:
        return text


def normalized_static(
    field: str,
    value: Any,
) -> str:
    if field in {
        "image_id",
        "target_index",
        "hazard_index",
    }:
        return normalize_integer(value)

    return clean_text(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def atomic_write_csv(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    frame.to_csv(
        temporary,
        index=False,
        encoding="utf-8-sig",
    )

    os.replace(
        temporary,
        path,
    )


def add_error(
    errors: list[dict[str, Any]],
    *,
    audit_pair_id: str,
    error_type: str,
    field: str = "",
    value: Any = "",
    detail: str = "",
) -> None:
    errors.append(
        {
            "audit_pair_id": audit_pair_id,
            "error_type": error_type,
            "field": field,
            "value": clean_text(value),
            "detail": detail,
        }
    )


def validate_review(
    review: pd.DataFrame,
    errors: list[dict[str, Any]],
) -> None:
    required_columns = {
        "audit_pair_id",
        "image_id",
        "global_image_key",
        "target_index",
        "hazard_index",
        "target_category",
        "hazard_category",
        "pair_family",
        *MANUAL_FIELDS,
    }

    missing = (
        required_columns
        - set(review.columns)
    )

    if missing:
        raise RuntimeError(
            "Review CSV missing columns: "
            f"{sorted(missing)}"
        )

    forbidden = {
        "source_type",
        "predicate_id",
        "predicate_name",
    }

    leaked = (
        forbidden
        & set(review.columns)
    )

    if leaked:
        raise RuntimeError(
            "Blinding violation in review CSV: "
            f"{sorted(leaked)}"
        )

    if len(review) != 240:
        raise RuntimeError(
            f"Expected 240 review rows, "
            f"found {len(review)}"
        )

    duplicated = review[
        review["audit_pair_id"].duplicated(
            keep=False
        )
    ]

    for pair_id in duplicated[
        "audit_pair_id"
    ].unique():
        add_error(
            errors,
            audit_pair_id=str(pair_id),
            error_type="duplicate_audit_pair_id",
        )

    for row in review.to_dict(
        orient="records"
    ):
        pair_id = clean_text(
            row["audit_pair_id"]
        )

        status = clean_text(
            row["manual_review_status"]
        )

        if status not in CHOICES[
            "manual_review_status"
        ]:
            add_error(
                errors,
                audit_pair_id=pair_id,
                error_type="invalid_status",
                field="manual_review_status",
                value=status,
                detail=(
                    "Status must be complete "
                    "or exclude."
                ),
            )

        for field, allowed in CHOICES.items():
            if field == "manual_review_status":
                continue

            value = clean_text(
                row.get(field)
            )

            if status == "exclude" and not value:
                continue

            if not value:
                add_error(
                    errors,
                    audit_pair_id=pair_id,
                    error_type="missing_manual_field",
                    field=field,
                )
            elif value not in allowed:
                add_error(
                    errors,
                    audit_pair_id=pair_id,
                    error_type="invalid_manual_value",
                    field=field,
                    value=value,
                )

        if status == "complete":
            required_complete = [
                "manual_mask_pair_quality",
                "manual_geometry_relation",
                "manual_semantic_contact",
                "manual_apparent_boundary_quality",
                "manual_confidence",
            ]

            for field in required_complete:
                if not clean_text(
                    row.get(field)
                ):
                    add_error(
                        errors,
                        audit_pair_id=pair_id,
                        error_type=(
                            "complete_row_missing_field"
                        ),
                        field=field,
                    )

        if status == "exclude":
            if not clean_text(
                row.get(
                    "manual_failure_mode"
                )
            ):
                add_error(
                    errors,
                    audit_pair_id=pair_id,
                    error_type=(
                        "exclude_without_failure_mode"
                    ),
                    field="manual_failure_mode",
                )

        hazard = clean_text(
            row["hazard_category"]
        )

        boundary_quality = clean_text(
            row[
                "manual_apparent_boundary_quality"
            ]
        )

        if status == "complete":
            if (
                hazard in {"water", "river"}
                and boundary_quality
                == "not_applicable"
            ):
                add_error(
                    errors,
                    audit_pair_id=pair_id,
                    error_type=(
                        "water_boundary_marked_"
                        "not_applicable"
                    ),
                    field=(
                        "manual_apparent_"
                        "boundary_quality"
                    ),
                    value=boundary_quality,
                )

            if (
                hazard
                in {"mud_debris", "barricade"}
                and boundary_quality
                != "not_applicable"
            ):
                add_error(
                    errors,
                    audit_pair_id=pair_id,
                    error_type=(
                        "nonwater_boundary_not_"
                        "marked_not_applicable"
                    ),
                    field=(
                        "manual_apparent_"
                        "boundary_quality"
                    ),
                    value=boundary_quality,
                )

        saved_at = clean_text(
            row.get("manual_saved_at")
        )

        annotator = clean_text(
            row.get("manual_annotator")
        )

        revision = clean_text(
            row.get("manual_revision")
        )

        if not saved_at:
            add_error(
                errors,
                audit_pair_id=pair_id,
                error_type="missing_saved_at",
                field="manual_saved_at",
            )

        if not annotator:
            add_error(
                errors,
                audit_pair_id=pair_id,
                error_type="missing_annotator",
                field="manual_annotator",
            )

        try:
            revision_number = int(
                float(revision)
            )
        except ValueError:
            revision_number = 0

        if revision_number < 1:
            add_error(
                errors,
                audit_pair_id=pair_id,
                error_type="invalid_revision",
                field="manual_revision",
                value=revision,
            )


def validate_key(
    review: pd.DataFrame,
    key: pd.DataFrame,
    errors: list[dict[str, Any]],
) -> None:
    if len(key) != 240:
        raise RuntimeError(
            f"Expected 240 key rows, "
            f"found {len(key)}"
        )

    if "audit_pair_id" not in key.columns:
        raise RuntimeError(
            "Locked key has no audit_pair_id"
        )

    if key[
        "audit_pair_id"
    ].duplicated().any():
        raise RuntimeError(
            "Locked key contains duplicate IDs"
        )

    review_ids = set(
        review[
            "audit_pair_id"
        ].astype(str)
    )

    key_ids = set(
        key[
            "audit_pair_id"
        ].astype(str)
    )

    missing_from_key = (
        review_ids - key_ids
    )

    missing_from_review = (
        key_ids - review_ids
    )

    for pair_id in sorted(
        missing_from_key
    ):
        add_error(
            errors,
            audit_pair_id=pair_id,
            error_type="missing_from_locked_key",
        )

    for pair_id in sorted(
        missing_from_review
    ):
        add_error(
            errors,
            audit_pair_id=pair_id,
            error_type="missing_from_review",
        )

    review_index = review.set_index(
        "audit_pair_id",
        drop=False,
    )

    key_index = key.set_index(
        "audit_pair_id",
        drop=False,
    )

    for pair_id in sorted(
        review_ids & key_ids
    ):
        for field in STATIC_FIELDS:
            review_value = normalized_static(
                field,
                review_index.at[
                    pair_id,
                    field,
                ],
            )

            key_value = normalized_static(
                field,
                key_index.at[
                    pair_id,
                    field,
                ],
            )

            if review_value != key_value:
                add_error(
                    errors,
                    audit_pair_id=pair_id,
                    error_type=(
                        "review_key_static_mismatch"
                    ),
                    field=field,
                    value=review_value,
                    detail=(
                        f"locked_key={key_value}"
                    ),
                )


def validate_label_jsons(
    review: pd.DataFrame,
    label_dir: Path,
    errors: list[dict[str, Any]],
) -> int:
    if not label_dir.is_dir():
        raise FileNotFoundError(
            label_dir
        )

    label_files = sorted(
        label_dir.glob(
            "FIBE-AUDIT-*.json"
        )
    )

    review_index = review.set_index(
        "audit_pair_id",
        drop=False,
    )

    expected_ids = set(
        review_index.index.astype(str)
    )

    found_ids = {
        path.stem
        for path in label_files
    }

    for pair_id in sorted(
        expected_ids - found_ids
    ):
        add_error(
            errors,
            audit_pair_id=pair_id,
            error_type="missing_pair_json",
        )

    for pair_id in sorted(
        found_ids - expected_ids
    ):
        add_error(
            errors,
            audit_pair_id=pair_id,
            error_type="unexpected_pair_json",
        )

    for path in label_files:
        pair_id = path.stem

        try:
            payload = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception as error:
            add_error(
                errors,
                audit_pair_id=pair_id,
                error_type="invalid_pair_json",
                detail=str(error),
            )
            continue

        payload_id = clean_text(
            payload.get(
                "audit_pair_id"
            )
        )

        if payload_id != pair_id:
            add_error(
                errors,
                audit_pair_id=pair_id,
                error_type=(
                    "pair_json_id_mismatch"
                ),
                value=payload_id,
            )

        annotations = payload.get(
            "annotations"
        )

        if not isinstance(
            annotations,
            dict,
        ):
            add_error(
                errors,
                audit_pair_id=pair_id,
                error_type=(
                    "pair_json_annotations_missing"
                ),
            )
            continue

        if pair_id not in review_index.index:
            continue

        for field in JSON_ANNOTATION_FIELDS:
            csv_value = clean_text(
                review_index.at[
                    pair_id,
                    field,
                ]
            )

            json_value = clean_text(
                annotations.get(
                    field
                )
            )

            if csv_value != json_value:
                add_error(
                    errors,
                    audit_pair_id=pair_id,
                    error_type=(
                        "csv_json_annotation_"
                        "mismatch"
                    ),
                    field=field,
                    value=csv_value,
                    detail=(
                        f"json={json_value}"
                    ),
                )

    return len(label_files)


def make_unblinded(
    review: pd.DataFrame,
    key: pd.DataFrame,
) -> pd.DataFrame:
    manual = review[
        [
            "audit_pair_id",
            *MANUAL_FIELDS,
        ]
    ].copy()

    unblinded = key.merge(
        manual,
        on="audit_pair_id",
        how="left",
        validate="one_to_one",
    )

    if len(unblinded) != 240:
        raise RuntimeError(
            "Unblinded row count mismatch"
        )

    return unblinded


def value_counts_dict(
    series: pd.Series,
) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in (
            series.value_counts(
                dropna=False
            ).items()
        )
    }


def crosstab_records(
    frame: pd.DataFrame,
    rows: list[str],
    column: str,
) -> list[dict[str, Any]]:
    table = (
        frame.groupby(
            [
                *rows,
                column,
            ],
            observed=True,
            dropna=False,
        )
        .size()
        .reset_index(name="count")
    )

    return table.to_dict(
        orient="records"
    )


def main() -> None:
    args = parse_args()

    review_path = (
        args.review.expanduser().resolve()
    )
    key_path = (
        args.key.expanduser().resolve()
    )
    label_dir = (
        args.label_dir
        .expanduser()
        .resolve()
    )
    final_review_path = (
        args.final_review
        .expanduser()
        .resolve()
    )
    unblinded_path = (
        args.unblinded
        .expanduser()
        .resolve()
    )
    summary_path = (
        args.summary
        .expanduser()
        .resolve()
    )
    errors_path = (
        args.errors.expanduser().resolve()
    )

    if not review_path.is_file():
        raise FileNotFoundError(
            review_path
        )

    if not key_path.is_file():
        raise FileNotFoundError(
            key_path
        )

    review = pd.read_csv(
        review_path,
        encoding="utf-8-sig",
        keep_default_na=False,
        dtype=str,
    )

    key = pd.read_csv(
        key_path,
        encoding="utf-8-sig",
        keep_default_na=False,
        low_memory=False,
    )

    errors: list[
        dict[str, Any]
    ] = []

    validate_review(
        review,
        errors,
    )

    validate_key(
        review,
        key,
        errors,
    )

    label_json_count = (
        validate_label_jsons(
            review,
            label_dir,
            errors,
        )
    )

    errors_frame = pd.DataFrame(
        errors,
        columns=[
            "audit_pair_id",
            "error_type",
            "field",
            "value",
            "detail",
        ],
    )

    errors_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    errors_frame.to_csv(
        errors_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 100)
    print(
        "FIBE BOUNDARY AUDIT FINAL V1"
    )
    print("=" * 100)
    print("review rows:", len(review))
    print("locked key rows:", len(key))
    print(
        "pair JSON files:",
        label_json_count,
    )
    print("errors:", len(errors))

    if errors:
        print("\nERROR COUNTS")
        print(
            errors_frame[
                "error_type"
            ]
            .value_counts()
            .to_string()
        )

        print("\nERROR PREVIEW")
        print(
            errors_frame
            .head(30)
            .to_string(index=False)
        )

        print("\nError report:")
        print(errors_path)

        raise SystemExit(
            "FAIL: manual audit has "
            "validation errors"
        )

    unblinded = make_unblinded(
        review,
        key,
    )

    atomic_write_csv(
        review,
        final_review_path,
    )

    atomic_write_csv(
        unblinded,
        unblinded_path,
    )

    complete = review[
        review[
            "manual_review_status"
        ]
        == "complete"
    ]

    excluded = review[
        review[
            "manual_review_status"
        ]
        == "exclude"
    ]

    valid_geometry = complete[
        (
            complete[
                "manual_mask_pair_quality"
            ]
            == "valid"
        )
        & (
            complete[
                "manual_geometry_relation"
            ]
            != "ambiguous"
        )
    ]

    valid_semantic = valid_geometry[
        valid_geometry[
            "manual_semantic_contact"
        ]
        != "ambiguous"
    ]

    summary = {
        "version": (
            "FIBE_BOUNDARY_AUDIT_"
            "FINAL_V1"
        ),
        "review_path": str(
            review_path
        ),
        "locked_key_path": str(
            key_path
        ),
        "label_dir": str(
            label_dir
        ),
        "final_review_path": str(
            final_review_path
        ),
        "unblinded_path": str(
            unblinded_path
        ),
        "review_rows": int(
            len(review)
        ),
        "locked_key_rows": int(
            len(key)
        ),
        "label_json_files": int(
            label_json_count
        ),
        "error_count": int(
            len(errors)
        ),
        "complete_rows": int(
            len(complete)
        ),
        "excluded_rows": int(
            len(excluded)
        ),
        "valid_geometry_rows": int(
            len(valid_geometry)
        ),
        "valid_semantic_rows": int(
            len(valid_semantic)
        ),
        "status_counts": (
            value_counts_dict(
                review[
                    "manual_review_status"
                ]
            )
        ),
        "mask_quality_counts": (
            value_counts_dict(
                review[
                    "manual_mask_pair_quality"
                ]
            )
        ),
        "geometry_counts": (
            value_counts_dict(
                review[
                    "manual_geometry_relation"
                ]
            )
        ),
        "semantic_contact_counts": (
            value_counts_dict(
                review[
                    "manual_semantic_contact"
                ]
            )
        ),
        "boundary_quality_counts": (
            value_counts_dict(
                review[
                    "manual_apparent_boundary_quality"
                ]
            )
        ),
        "confidence_counts": (
            value_counts_dict(
                review[
                    "manual_confidence"
                ]
            )
        ),
        "source_by_semantic_contact": (
            crosstab_records(
                unblinded[
                    unblinded[
                        "manual_review_status"
                    ]
                    == "complete"
                ],
                ["source_type"],
                "manual_semantic_contact",
            )
        ),
        "family_by_geometry": (
            crosstab_records(
                unblinded[
                    unblinded[
                        "manual_review_status"
                    ]
                    == "complete"
                ],
                ["pair_family"],
                "manual_geometry_relation",
            )
        ),
        "sha256": {
            "review_input": (
                sha256_file(
                    review_path
                )
            ),
            "locked_key": (
                sha256_file(
                    key_path
                )
            ),
            "final_review": (
                sha256_file(
                    final_review_path
                )
            ),
            "unblinded": (
                sha256_file(
                    unblinded_path
                )
            ),
        },
    }

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nSTATUS")
    print(
        review[
            "manual_review_status"
        ]
        .value_counts()
        .to_string()
    )

    print("\nMASK QUALITY")
    print(
        review[
            "manual_mask_pair_quality"
        ]
        .value_counts()
        .to_string()
    )

    print("\nGEOMETRY")
    print(
        review[
            "manual_geometry_relation"
        ]
        .value_counts()
        .to_string()
    )

    print("\nSEMANTIC CONTACT")
    print(
        review[
            "manual_semantic_contact"
        ]
        .value_counts()
        .to_string()
    )

    print("\nBOUNDARY QUALITY")
    print(
        review[
            "manual_apparent_boundary_quality"
        ]
        .value_counts()
        .to_string()
    )

    print("\nCONFIDENCE")
    print(
        review[
            "manual_confidence"
        ]
        .value_counts()
        .to_string()
    )

    print("\nSOURCE × SEMANTIC CONTACT")
    print(
        pd.crosstab(
            unblinded[
                "source_type"
            ],
            unblinded[
                "manual_semantic_contact"
            ],
        ).to_string()
    )

    print("\nELIGIBLE COUNTS")
    print(
        "complete:",
        len(complete),
    )
    print(
        "excluded:",
        len(excluded),
    )
    print(
        "valid geometry:",
        len(valid_geometry),
    )
    print(
        "valid semantic:",
        len(valid_semantic),
    )

    print("\nOUTPUTS")
    print(
        "final review:",
        final_review_path,
    )
    print(
        "unblinded audit:",
        unblinded_path,
    )
    print(
        "summary:",
        summary_path,
    )
    print(
        "error report:",
        errors_path,
    )

    print(
        "\nFIBE BOUNDARY AUDIT "
        "FINALIZATION: PASS"
    )


if __name__ == "__main__":
    main()
