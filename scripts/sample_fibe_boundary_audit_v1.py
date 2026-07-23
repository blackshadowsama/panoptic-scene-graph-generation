from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (
    PROJECT_ROOT / "data" / "floodpsg"
).resolve()

DEFAULT_INPUT = (
    DATA_ROOT
    / "features"
    / "fibe_scalar_v1"
    / "fibe_boundary_audit_pair_pool_train_geometry_v1.csv"
)

DEFAULT_KEY = (
    DATA_ROOT
    / "features"
    / "fibe_scalar_v1"
    / "fibe_boundary_audit_sample_key_v1.csv"
)

DEFAULT_REVIEW = (
    DATA_ROOT
    / "features"
    / "fibe_scalar_v1"
    / "fibe_boundary_audit_review_v1.csv"
)

DEFAULT_SUMMARY = (
    DATA_ROOT
    / "stats"
    / "dsformer_fibe_d1_v1"
    / "FIBE_BOUNDARY_AUDIT_SAMPLE_V1.json"
)

SELECTION_SEED = 3407
BLIND_ORDER_SEED = 3408

FAMILY_QUOTAS = {
    "human-water": 17,
    "vehicle-water": 17,
    "road-surface-water": 17,
    "building-water": 15,
    "bridge-water": 15,
    "drainage-water": 12,
    "road-debris": 15,
    "road-barricade": 12,
}

SOURCE_TYPES = (
    "positive",
    "hard_negative",
)

RARE_HN_ALL = {
    "drainage-water",
    "road-debris",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a deterministic and blinded "
            "240-pair manual boundary audit sample."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )
    parser.add_argument(
        "--key-output",
        type=Path,
        default=DEFAULT_KEY,
    )
    parser.add_argument(
        "--review-output",
        type=Path,
        default=DEFAULT_REVIEW,
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
    )
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=SELECTION_SEED,
    )
    parser.add_argument(
        "--blind-seed",
        type=int,
        default=BLIND_ORDER_SEED,
    )

    return parser.parse_args()


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


def as_text(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def candidate_score(
    row: pd.Series,
    topology_counts: Counter[str],
    scale_counts: Counter[str],
    truncation_counts: Counter[int],
    global_image_counts: Counter[int],
    random_jitter: float,
) -> float:
    topology = as_text(
        row["geometry_topology_proxy"]
    )
    scale = as_text(
        row["target_scale_tertile"]
    )
    truncated = int(
        row["pair_mask_truncated"]
    )
    image_id = int(row["image_id"])

    return (
        100.0
        * global_image_counts[image_id]
        + 12.0
        * topology_counts[topology]
        + 4.0
        * scale_counts[scale]
        + 2.0
        * truncation_counts[truncated]
        + random_jitter
    )


def balanced_select(
    *,
    subset: pd.DataFrame,
    mandatory_reasons: dict[int, str],
    quota: int,
    rng: np.random.Generator,
    global_image_counts: Counter[int],
) -> tuple[list[int], dict[int, str]]:
    if len(subset) < quota:
        raise RuntimeError(
            "Insufficient rows for quota: "
            f"available={len(subset)}, "
            f"quota={quota}"
        )

    mandatory_indices = sorted(
        mandatory_reasons
    )

    if len(mandatory_indices) > quota:
        raise RuntimeError(
            "Mandatory rows exceed quota: "
            f"mandatory={len(mandatory_indices)}, "
            f"quota={quota}"
        )

    selected = list(mandatory_indices)
    reasons = dict(mandatory_reasons)

    topology_counts: Counter[str] = (
        Counter()
    )
    scale_counts: Counter[str] = (
        Counter()
    )
    truncation_counts: Counter[int] = (
        Counter()
    )

    for index in selected:
        row = subset.loc[index]

        topology_counts[
            as_text(
                row[
                    "geometry_topology_proxy"
                ]
            )
        ] += 1

        scale_counts[
            as_text(
                row[
                    "target_scale_tertile"
                ]
            )
        ] += 1

        truncation_counts[
            int(
                row[
                    "pair_mask_truncated"
                ]
            )
        ] += 1

        global_image_counts[
            int(row["image_id"])
        ] += 1

    candidates = [
        int(index)
        for index in subset.index
        if int(index) not in reasons
    ]

    jitter = {
        index: float(rng.random())
        for index in candidates
    }

    while len(selected) < quota:
        if not candidates:
            raise RuntimeError(
                "Candidate pool exhausted "
                "before reaching quota"
            )

        best_index = min(
            candidates,
            key=lambda index: candidate_score(
                subset.loc[index],
                topology_counts,
                scale_counts,
                truncation_counts,
                global_image_counts,
                jitter[index],
            ),
        )

        selected.append(best_index)

        reasons[best_index] = (
            "balanced_geometry_"
            "scale_truncation"
        )

        candidates.remove(best_index)

        row = subset.loc[best_index]

        topology_counts[
            as_text(
                row[
                    "geometry_topology_proxy"
                ]
            )
        ] += 1

        scale_counts[
            as_text(
                row[
                    "target_scale_tertile"
                ]
            )
        ] += 1

        truncation_counts[
            int(
                row[
                    "pair_mask_truncated"
                ]
            )
        ] += 1

        global_image_counts[
            int(row["image_id"])
        ] += 1

    return selected, reasons


def main() -> None:
    args = parse_args()

    input_path = (
        args.input.expanduser().resolve()
    )
    key_output = (
        args.key_output
        .expanduser()
        .resolve()
    )
    review_output = (
        args.review_output
        .expanduser()
        .resolve()
    )
    summary_path = (
        args.summary
        .expanduser()
        .resolve()
    )

    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    frame = pd.read_csv(
        input_path,
        encoding="utf-8-sig",
        low_memory=False,
    )

    frame.reset_index(
        drop=True,
        inplace=True,
    )

    required = {
        "pool_pair_id",
        "source_type",
        "pair_family",
        "predicate_name",
        "image_id",
        "global_image_key",
        "target_index",
        "hazard_index",
        "target_category",
        "hazard_category",
        "target_mask_path",
        "hazard_mask_path",
        "geometry_topology_proxy",
        "target_scale_tertile",
        "pair_mask_truncated",
    }

    missing = required - set(
        frame.columns
    )

    if missing:
        raise RuntimeError(
            "Input missing columns: "
            f"{sorted(missing)}"
        )

    if frame[
        "pool_pair_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate pool_pair_id values"
        )

    unexpected_sources = (
        set(
            frame[
                "source_type"
            ].astype(str)
        )
        - set(SOURCE_TYPES)
    )

    if unexpected_sources:
        raise RuntimeError(
            "Unexpected source types: "
            f"{sorted(unexpected_sources)}"
        )

    expected_total = (
        2
        * sum(
            FAMILY_QUOTAS.values()
        )
    )

    if expected_total != 240:
        raise RuntimeError(
            "Quota definition does not "
            f"sum to 240: {expected_total}"
        )

    rng = np.random.default_rng(
        args.selection_seed
    )

    global_image_counts: Counter[int] = (
        Counter()
    )

    selected_indices: list[int] = []
    selection_reasons: dict[
        int,
        str,
    ] = {}

    for source_type in SOURCE_TYPES:
        for family, quota in (
            FAMILY_QUOTAS.items()
        ):
            subset = frame[
                (
                    frame["source_type"]
                    == source_type
                )
                & (
                    frame["pair_family"]
                    == family
                )
            ].copy()

            mandatory: dict[int, str] = {}

            if (
                source_type
                == "hard_negative"
                and family in RARE_HN_ALL
            ):
                for index in subset.index:
                    mandatory[int(index)] = (
                        "mandatory_all_rare_"
                        "hard_negative"
                    )

            if source_type == "positive":
                boat_mask = (
                    subset[
                        "predicate_name"
                    ].astype(str)
                    == "boat_water_context"
                )

                for index in subset.index[
                    boat_mask
                ]:
                    mandatory[int(index)] = (
                        "mandatory_all_"
                        "boat_positive"
                    )

            chosen, reasons = (
                balanced_select(
                    subset=subset,
                    mandatory_reasons=(
                        mandatory
                    ),
                    quota=quota,
                    rng=rng,
                    global_image_counts=(
                        global_image_counts
                    ),
                )
            )

            overlap = (
                set(chosen)
                & set(selected_indices)
            )

            if overlap:
                raise RuntimeError(
                    "Rows selected in "
                    "multiple strata: "
                    f"{sorted(overlap)[:10]}"
                )

            selected_indices.extend(
                chosen
            )
            selection_reasons.update(
                reasons
            )

    if len(selected_indices) != 240:
        raise RuntimeError(
            "Selected row count mismatch: "
            f"{len(selected_indices)}"
        )

    selected = frame.loc[
        selected_indices
    ].copy()

    selected[
        "selection_reason"
    ] = [
        selection_reasons[
            int(index)
        ]
        for index in selected.index
    ]

    if selected[
        "pool_pair_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate selected pool pairs"
        )

    quota_table = (
        selected.groupby(
            [
                "source_type",
                "pair_family",
            ],
            observed=True,
        )
        .size()
        .to_dict()
    )

    for source_type in SOURCE_TYPES:
        for family, quota in (
            FAMILY_QUOTAS.items()
        ):
            actual = int(
                quota_table.get(
                    (
                        source_type,
                        family,
                    ),
                    0,
                )
            )

            if actual != quota:
                raise RuntimeError(
                    "Quota mismatch for "
                    f"{source_type}/{family}: "
                    f"{actual} != {quota}"
                )

    all_boat = frame[
        (
            frame["source_type"]
            == "positive"
        )
        & (
            frame[
                "predicate_name"
            ].astype(str)
            == "boat_water_context"
        )
    ]

    selected_boat = selected[
        (
            selected["source_type"]
            == "positive"
        )
        & (
            selected[
                "predicate_name"
            ].astype(str)
            == "boat_water_context"
        )
    ]

    if set(
        all_boat["pool_pair_id"]
    ) != set(
        selected_boat[
            "pool_pair_id"
        ]
    ):
        raise RuntimeError(
            "Not all boat positives "
            "were selected"
        )

    for family in sorted(
        RARE_HN_ALL
    ):
        all_rare = frame[
            (
                frame["source_type"]
                == "hard_negative"
            )
            & (
                frame["pair_family"]
                == family
            )
        ]

        chosen_rare = selected[
            (
                selected["source_type"]
                == "hard_negative"
            )
            & (
                selected["pair_family"]
                == family
            )
        ]

        if set(
            all_rare["pool_pair_id"]
        ) != set(
            chosen_rare["pool_pair_id"]
        ):
            raise RuntimeError(
                "Not all rare HN rows "
                f"were selected for {family}"
            )

    blind_rng = np.random.default_rng(
        args.blind_seed
    )

    blind_order = blind_rng.permutation(
        len(selected)
    )

    blinded = (
        selected.iloc[blind_order]
        .copy()
        .reset_index(drop=True)
    )

    blinded.insert(
        0,
        "audit_pair_id",
        [
            f"FIBE-AUDIT-{index:04d}"
            for index in range(
                1,
                len(blinded) + 1,
            )
        ],
    )

    key_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    review_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    blinded.to_csv(
        key_output,
        index=False,
        encoding="utf-8-sig",
    )

    review_columns = [
        "audit_pair_id",
        "image_id",
        "global_image_key",
        "image_path",
        "image_width",
        "image_height",
        "target_index",
        "hazard_index",
        "target_category",
        "hazard_category",
        "pair_family",
        "target_mask_path",
        "hazard_mask_path",
    ]

    missing_review = [
        column
        for column in review_columns
        if column not in blinded.columns
    ]

    if missing_review:
        raise RuntimeError(
            "Missing review columns: "
            f"{missing_review}"
        )

    review = blinded[
        review_columns
    ].copy()

    manual_columns = [
        "manual_review_status",
        "manual_mask_pair_quality",
        "manual_geometry_relation",
        "manual_semantic_contact",
        "manual_apparent_boundary_quality",
        "manual_failure_mode",
        "manual_confidence",
        "manual_notes",
    ]

    for column in manual_columns:
        review[column] = ""

    review.to_csv(
        review_output,
        index=False,
        encoding="utf-8-sig",
    )

    image_pair_counts = selected[
        "image_id"
    ].value_counts()

    source_family_counts = (
        selected.groupby(
            [
                "source_type",
                "pair_family",
            ],
            observed=True,
        )
        .size()
        .reset_index(name="count")
        .to_dict(orient="records")
    )

    topology_counts = (
        selected.groupby(
            [
                "source_type",
                "pair_family",
                "geometry_topology_proxy",
            ],
            observed=True,
        )
        .size()
        .reset_index(name="count")
        .to_dict(orient="records")
    )

    scale_counts = (
        selected.groupby(
            [
                "source_type",
                "pair_family",
                "target_scale_tertile",
            ],
            observed=True,
        )
        .size()
        .reset_index(name="count")
        .to_dict(orient="records")
    )

    truncation_counts = (
        selected.groupby(
            [
                "source_type",
                "pair_mask_truncated",
            ],
            observed=True,
        )
        .size()
        .reset_index(name="count")
        .to_dict(orient="records")
    )

    summary = {
        "version": (
            "FIBE_BOUNDARY_AUDIT_"
            "SAMPLE_V1"
        ),
        "input_path": str(
            input_path
        ),
        "input_sha256": (
            sha256_file(input_path)
        ),
        "key_output": str(
            key_output
        ),
        "review_output": str(
            review_output
        ),
        "selection_seed": int(
            args.selection_seed
        ),
        "blind_order_seed": int(
            args.blind_seed
        ),
        "selected_rows": int(
            len(selected)
        ),
        "selected_positive": int(
            (
                selected["source_type"]
                == "positive"
            ).sum()
        ),
        "selected_hard_negative": int(
            (
                selected["source_type"]
                == "hard_negative"
            ).sum()
        ),
        "unique_images": int(
            selected[
                "image_id"
            ].nunique()
        ),
        "max_pairs_per_image": int(
            image_pair_counts.max()
        ),
        "images_with_multiple_pairs": int(
            (
                image_pair_counts > 1
            ).sum()
        ),
        "boat_positive_total": int(
            len(all_boat)
        ),
        "boat_positive_selected": int(
            len(selected_boat)
        ),
        "family_quotas_per_source": (
            FAMILY_QUOTAS
        ),
        "source_family_counts": (
            source_family_counts
        ),
        "topology_counts": (
            topology_counts
        ),
        "scale_counts": (
            scale_counts
        ),
        "truncation_counts": (
            truncation_counts
        ),
        "manual_label_schema": {
            "manual_review_status": [
                "pending",
                "complete",
                "exclude",
            ],
            "manual_mask_pair_quality": [
                "valid",
                "target_mask_error",
                "hazard_mask_error",
                "both_mask_error",
                "ambiguous",
            ],
            "manual_geometry_relation": [
                "overlap",
                "touching",
                "near_gap",
                "far_gap",
                "ambiguous",
            ],
            "manual_semantic_contact": [
                "contact",
                "not_contact",
                "ambiguous",
            ],
            "manual_apparent_boundary_quality": [
                "valid",
                "invalid",
                "not_applicable",
                "ambiguous",
            ],
            "manual_confidence": [
                "high",
                "medium",
                "low",
            ],
        },
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
        "FIBE BOUNDARY AUDIT "
        "SAMPLE V1"
    )
    print("=" * 100)
    print("input rows:", len(frame))
    print(
        "selected rows:",
        len(selected),
    )
    print(
        "positive:",
        int(
            (
                selected[
                    "source_type"
                ]
                == "positive"
            ).sum()
        ),
    )
    print(
        "hard negative:",
        int(
            (
                selected[
                    "source_type"
                ]
                == "hard_negative"
            ).sum()
        ),
    )
    print(
        "unique images:",
        selected[
            "image_id"
        ].nunique(),
    )
    print(
        "max pairs per image:",
        int(
            image_pair_counts.max()
        ),
    )
    print(
        "images with multiple pairs:",
        int(
            (
                image_pair_counts > 1
            ).sum()
        ),
    )
    print(
        "boat positives:",
        f"{len(selected_boat)}/"
        f"{len(all_boat)}",
    )

    print("\nSOURCE × FAMILY")
    print(
        pd.crosstab(
            selected["pair_family"],
            selected["source_type"],
        ).to_string()
    )

    print("\nSOURCE × TOPOLOGY")
    print(
        pd.crosstab(
            selected["source_type"],
            selected[
                "geometry_topology_proxy"
            ],
        ).to_string()
    )

    print("\nSOURCE × SCALE")
    print(
        pd.crosstab(
            selected["source_type"],
            selected[
                "target_scale_tertile"
            ],
        ).to_string()
    )

    print(
        "\nSOURCE × "
        "INDEPENDENT-MASK TRUNCATION"
    )
    print(
        pd.crosstab(
            selected["source_type"],
            selected[
                "pair_mask_truncated"
            ],
        ).to_string()
    )

    print("\nSELECTION REASONS")
    print(
        selected[
            "selection_reason"
        ]
        .value_counts()
        .to_string()
    )

    print("\nOUTPUTS")
    print(
        "locked key:",
        key_output,
    )
    print(
        "blinded review:",
        review_output,
    )
    print(
        "summary:",
        summary_path,
    )

    print(
        "\nFIBE BOUNDARY AUDIT "
        "SAMPLE: PASS"
    )


if __name__ == "__main__":
    main()
