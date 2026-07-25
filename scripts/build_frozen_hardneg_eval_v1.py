#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build a frozen FloodPSG hard-negative subset without import-time side effects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


WATER_FAMILIES = {
    "person-water",
    "vehicle-water",
    "building-water",
    "road-water",
    "sidewalk-water",
    "underpass_bridge-water",
    "bridge-water",
    "manhole-water",
    "drain-water",
    "boat-water",
}

EXPECTED_WATER_COUNTS = {
    "train": {
        "pairs": 2198,
        "images": 527,
    },
    "validation": {
        "pairs": 322,
        "images": 75,
    },
    "final_test": {
        "pairs": 363,
        "images": 72,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one frozen FloodPSG hard-negative split. "
            "No input is read and no output is written when --help is used."
        ),
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Canonical hard-negative audit CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output directory; existing paths are refused.",
    )
    parser.add_argument(
        "--split",
        choices=(
            "train",
            "validation",
            "final_test",
        ),
        required=True,
        help="Frozen split to build.",
    )
    parser.add_argument(
        "--expected-water-pairs",
        type=int,
        default=None,
        help=(
            "Expected number of unique water HN pairs. "
            "Defaults to the frozen protocol count for the selected split."
        ),
    )
    parser.add_argument(
        "--expected-water-images",
        type=int,
        default=None,
        help=(
            "Expected number of unique water HN images. "
            "Defaults to the frozen protocol count for the selected split."
        ),
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Compute and validate the subset without writing files.",
    )
    parser.add_argument(
        "--allow-final-test",
        action="store_true",
        help=(
            "Explicitly permit final_test construction. "
            "Do not use this during D1 development/model selection."
        ),
    )

    return parser.parse_args()


def bool_series(
    series: pd.Series,
) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "1.0": True,
            "0.0": False,
        })
        .fillna(False)
        .astype(bool)
    )


def group_summary(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "pair_family_v4",
                "unique_pairs",
                "unique_images",
            ]
        )

    return (
        frame.groupby(
            "pair_family_v4",
            dropna=False,
        )
        .agg(
            unique_pairs=(
                "canonical_pair_key",
                "nunique",
            ),
            unique_images=(
                "global_image_key",
                "nunique",
            ),
        )
        .reset_index()
        .sort_values(
            "unique_pairs",
            ascending=False,
        )
    )


def json_safe(
    value: Any,
) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def main() -> None:
    args = parse_args()

    input_path = (
        args.input.expanduser().resolve()
    )
    output_path = (
        args.output.expanduser().resolve()
    )

    if args.split == "final_test" and not args.allow_final_test:
        raise SystemExit(
            "ERROR: final_test is locked. "
            "Pass --allow-final-test only after all validation gates are frozen."
        )

    if not input_path.is_file():
        raise FileNotFoundError(
            input_path
        )

    if output_path.exists() and not args.verify_only:
        raise FileExistsError(
            "Refusing to overwrite output path: "
            f"{output_path}"
        )

    expected = EXPECTED_WATER_COUNTS[
        args.split
    ]

    expected_pairs = (
        int(args.expected_water_pairs)
        if args.expected_water_pairs is not None
        else int(expected["pairs"])
    )

    expected_images = (
        int(args.expected_water_images)
        if args.expected_water_images is not None
        else int(expected["images"])
    )

    frame = pd.read_csv(
        input_path,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required = {
        "global_image_key",
        "audit_split_v2",
        "audit_pair_status",
        "subject_index_v4",
        "object_index_v4",
        "canonical_pair_legal_v4",
        "canonical_pair_positive_v4",
        "canonical_label_consistent_v4",
        "is_high_risk_v4",
        "pair_family_v4",
    }

    missing = sorted(
        required - set(frame.columns)
    )

    if missing:
        raise KeyError(
            f"Missing columns: {missing}"
        )

    resolved = (
        frame["subject_index_v4"].notna()
        & frame["object_index_v4"].notna()
    )

    legal = bool_series(
        frame["canonical_pair_legal_v4"]
    )

    canonical_positive = bool_series(
        frame["canonical_pair_positive_v4"]
    )

    consistent = bool_series(
        frame[
            "canonical_label_consistent_v4"
        ]
    )

    high_risk = bool_series(
        frame["is_high_risk_v4"]
    )

    verified_negative = (
        frame["audit_pair_status"]
        .fillna("")
        .astype(str)
        .eq("verified_negative")
    )

    pair_key = [
        "global_image_key",
        "subject_index_v4",
        "object_index_v4",
    ]

    eligible = frame[
        resolved
        & legal
        & ~canonical_positive
        & consistent
        & high_risk
        & verified_negative
    ].copy()

    eligible["subject_index_v4"] = (
        eligible[
            "subject_index_v4"
        ].astype(int)
    )

    eligible["object_index_v4"] = (
        eligible[
            "object_index_v4"
        ].astype(int)
    )

    eligible["canonical_pair_key"] = (
        eligible[
            "global_image_key"
        ].astype(str)
        + "__s"
        + eligible[
            "subject_index_v4"
        ].astype(str)
        + "__o"
        + eligible[
            "object_index_v4"
        ].astype(str)
    )

    duplicate_rows = eligible[
        eligible.duplicated(
            subset=pair_key,
            keep=False,
        )
    ].copy()

    unique_pairs = (
        eligible.sort_values(
            pair_key
        )
        .drop_duplicates(
            subset=pair_key,
            keep="first",
        )
        .copy()
    )

    split_all = unique_pairs[
        unique_pairs["audit_split_v2"]
        == args.split
    ].copy()

    split_water = split_all[
        split_all[
            "pair_family_v4"
        ].isin(WATER_FAMILIES)
    ].copy()

    actual_pairs = int(
        len(split_water)
    )

    actual_images = int(
        split_water[
            "global_image_key"
        ].nunique()
    )

    duplicate_pair_count = int(
        split_water.duplicated(
            subset=pair_key,
            keep=False,
        ).sum()
    )

    split_values = sorted(
        str(value)
        for value in split_water[
            "audit_split_v2"
        ].dropna().unique()
    )

    checks = {
        "input_exists": True,
        "eligible_duplicate_rows_zero": (
            len(duplicate_rows) == 0
        ),
        "selected_pair_duplicates_zero": (
            duplicate_pair_count == 0
        ),
        "selected_split_exact": (
            split_values == [args.split]
        ),
        "water_pair_count_matches": (
            actual_pairs == expected_pairs
        ),
        "water_image_count_matches": (
            actual_images == expected_images
        ),
    }

    status = (
        "PASS"
        if all(checks.values())
        else "FAIL"
    )

    summary = {
        "protocol": (
            "FloodPSG frozen hard-negative "
            "evaluation subset v1"
        ),
        "mode": (
            "verify_only"
            if args.verify_only
            else "write"
        ),
        "split": args.split,
        "final_test_explicitly_allowed": bool(
            args.allow_final_test
        ),
        "input": str(input_path),
        "output": str(output_path),
        "source_rows": int(
            len(frame)
        ),
        "eligible_rows_before_dedup": int(
            len(eligible)
        ),
        "duplicate_eligible_rows": int(
            len(duplicate_rows)
        ),
        "unique_eligible_pairs": int(
            len(unique_pairs)
        ),
        "selected": {
            "all_hardneg_pairs": int(
                len(split_all)
            ),
            "water_hardneg_pairs": (
                actual_pairs
            ),
            "water_hardneg_images": (
                actual_images
            ),
            "expected_water_hardneg_pairs": (
                expected_pairs
            ),
            "expected_water_hardneg_images": (
                expected_images
            ),
            "duplicate_water_pair_rows": (
                duplicate_pair_count
            ),
            "audit_split_values": (
                split_values
            ),
        },
        "checks": checks,
        "status": status,
    }

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )

    print()
    print(
        f"{args.split} water hard negatives:"
    )
    print(
        group_summary(
            split_water
        ).to_string(index=False)
    )

    if status != "PASS":
        print()
        print(
            "FROZEN HARD-NEGATIVE "
            "SUBSET V1: FAIL"
        )
        raise SystemExit(1)

    if args.verify_only:
        print()
        print(
            "FROZEN HARD-NEGATIVE "
            "SUBSET V1 VERIFY-ONLY: PASS"
        )
        return

    output_path.mkdir(
        parents=True,
        exist_ok=False,
    )

    duplicate_rows.to_csv(
        output_path
        / "00_duplicate_eligible_rows.csv",
        index=False,
        encoding="utf-8-sig",
    )

    split_all.to_csv(
        output_path
        / f"01_{args.split}_hardneg_all.csv",
        index=False,
        encoding="utf-8-sig",
    )

    split_water.to_csv(
        output_path
        / f"02_{args.split}_hardneg_water.csv",
        index=False,
        encoding="utf-8-sig",
    )

    group_summary(
        split_water
    ).to_csv(
        output_path
        / f"03_{args.split}_water_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with (
        output_path / "summary.json"
    ).open(
        "x",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=json_safe,
        )
        handle.write("\n")

    print()
    print(
        "FROZEN HARD-NEGATIVE "
        "SUBSET V1: PASS"
    )
    print(
        "Saved to:",
        output_path,
    )


if __name__ == "__main__":
    main()
