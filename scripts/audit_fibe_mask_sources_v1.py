from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path, PureWindowsPath
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (PROJECT_ROOT / "data" / "floodpsg").resolve()

ANNOTATION_PATH = (
    DATA_ROOT
    / "annotations"
    / "floodpsg_canonical_trainval_coarse8_groupstrict_v1.json"
)

OBJECT_TABLE_PATH = (
    DATA_ROOT
    / "tables"
    / "mask_objects_all.csv"
)

HN_PATHS = [
    DATA_ROOT
    / "stats"
    / "frozen_hardneg_eval_v1"
    / "01_train_hardneg_all.csv",

    DATA_ROOT
    / "stats"
    / "frozen_hardneg_eval_v1"
    / "03_validation_hardneg_all.csv",

    DATA_ROOT
    / "stats"
    / "frozen_hardneg_eval_v1"
    / "04_validation_hardneg_water.csv",
]

OUTPUT_DIR = (
    DATA_ROOT
    / "stats"
    / "dsformer_fibe_d1_v1"
)

OUTPUT_PATH = OUTPUT_DIR / "FIBE_MASK_SOURCE_AUDIT_V1.json"


def clean_text(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(value).strip()

    if text.lower() in {
        "",
        "nan",
        "none",
        "null",
        "<na>",
    }:
        return ""

    return text


def unique_paths(paths: list[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()

    for path in paths:
        key = str(path)

        if key in seen:
            continue

        seen.add(key)
        result.append(path)

    return result


def candidate_paths(raw_value: Any) -> list[Path]:
    value = clean_text(raw_value)

    if not value:
        return []

    candidates: list[Path] = []

    # 原始Linux路径或相对路径。
    native = Path(value).expanduser()

    if native.is_absolute():
        candidates.append(native)
    else:
        candidates.extend(
            [
                PROJECT_ROOT / native,
                DATA_ROOT / native,
            ]
        )

    # Windows绝对路径转换为WSL挂载路径。
    if re.match(r"^[A-Za-z]:[\\/]", value):
        windows_path = PureWindowsPath(value)
        drive = windows_path.drive[0].lower()

        relative_parts = windows_path.parts[1:]

        candidates.append(
            Path("/mnt")
            / drive
            / Path(*relative_parts)
        )

    # 如果路径包含数据包目录名，尝试将后缀映射到当前DATA_ROOT。
    normalized = value.replace("\\", "/")
    marker = "FloodPSG_DSFormer_package/"

    marker_index = normalized.lower().find(marker.lower())

    if marker_index >= 0:
        suffix = normalized[
            marker_index + len(marker):
        ]

        candidates.append(
            DATA_ROOT / Path(suffix)
        )

    return unique_paths(candidates)


def resolve_existing(raw_value: Any) -> Path | None:
    for candidate in candidate_paths(raw_value):
        if candidate.is_file():
            return candidate.resolve()

    return None


def audit_path_column(
    frame: pd.DataFrame,
    column: str,
    *,
    sample_limit: int = 8,
) -> dict[str, Any]:
    nonempty_values = [
        clean_text(value)
        for value in frame[column].tolist()
        if clean_text(value)
    ]

    resolved_count = 0
    examples: list[dict[str, Any]] = []

    for value in nonempty_values:
        resolved = resolve_existing(value)

        if resolved is not None:
            resolved_count += 1

        if len(examples) < sample_limit:
            examples.append(
                {
                    "raw": value,
                    "candidate_paths": [
                        str(path)
                        for path in candidate_paths(value)
                    ],
                    "resolved": (
                        str(resolved)
                        if resolved is not None
                        else None
                    ),
                }
            )

    return {
        "column": column,
        "total_rows": int(len(frame)),
        "nonempty_rows": int(len(nonempty_values)),
        "unique_nonempty_values": int(
            len(set(nonempty_values))
        ),
        "resolved_existing_rows": int(resolved_count),
        "resolved_rate": (
            float(resolved_count / len(nonempty_values))
            if nonempty_values
            else None
        ),
        "examples": examples,
    }


def relevant_columns(
    frame: pd.DataFrame,
) -> list[str]:
    keywords = (
        "mask",
        "path",
        "image",
        "filename",
        "batch",
        "object",
        "segment",
        "uid",
        "global",
        "bbox",
        "category",
        "group",
    )

    return [
        column
        for column in frame.columns
        if any(
            keyword in column.lower()
            for keyword in keywords
        )
    ]


def path_columns(
    frame: pd.DataFrame,
) -> list[str]:
    result = []

    for column in frame.columns:
        lower = column.lower()

        if (
            "path" in lower
            and (
                "mask" in lower
                or "image" in lower
                or "file" in lower
            )
        ):
            result.append(column)

    return result


def audit_canonical_annotation() -> dict[str, Any]:
    if not ANNOTATION_PATH.is_file():
        raise FileNotFoundError(ANNOTATION_PATH)

    data = json.loads(
        ANNOTATION_PATH.read_text(
            encoding="utf-8"
        )
    )

    segment_key_counts: Counter[str] = Counter()
    annotation_key_counts: Counter[str] = Counter()

    total_segments = 0
    total_annotations = 0

    example_segments: list[dict[str, Any]] = []
    example_relations_info: list[dict[str, Any]] = []

    for item in data["data"]:
        annotations = item.get("annotations", [])
        segments = item.get("segments_info", [])
        relations_info = item.get(
            "relations_info",
            [],
        )

        total_annotations += len(annotations)
        total_segments += len(segments)

        for annotation in annotations:
            annotation_key_counts.update(
                annotation.keys()
            )

        for segment in segments:
            segment_key_counts.update(
                segment.keys()
            )

            if len(example_segments) < 10:
                example_segments.append(segment)

        for relation_info in relations_info:
            if len(example_relations_info) < 10:
                example_relations_info.append(
                    relation_info
                )

    return {
        "path": str(ANNOTATION_PATH),
        "num_images": int(len(data["data"])),
        "num_validation_ids": int(
            len(data.get("test_image_ids", []))
        ),
        "total_annotations": int(total_annotations),
        "total_segments": int(total_segments),
        "annotation_key_counts": dict(
            sorted(annotation_key_counts.items())
        ),
        "segment_key_counts": dict(
            sorted(segment_key_counts.items())
        ),
        "example_segments_info": example_segments,
        "example_relations_info": (
            example_relations_info
        ),
    }


def audit_object_table() -> dict[str, Any]:
    if not OBJECT_TABLE_PATH.is_file():
        raise FileNotFoundError(
            OBJECT_TABLE_PATH
        )

    frame = pd.read_csv(
        OBJECT_TABLE_PATH,
        encoding="utf-8-sig",
        low_memory=False,
    )

    detected_path_columns = path_columns(frame)
    relevant = relevant_columns(frame)

    preview_columns = relevant[:40]

    preview = (
        frame[preview_columns]
        .head(10)
        .fillna("")
        .to_dict(orient="records")
        if preview_columns
        else []
    )

    path_audits = {
        column: audit_path_column(
            frame,
            column,
        )
        for column in detected_path_columns
    }

    return {
        "path": str(OBJECT_TABLE_PATH),
        "rows": int(len(frame)),
        "columns": frame.columns.tolist(),
        "relevant_columns": relevant,
        "detected_path_columns": (
            detected_path_columns
        ),
        "path_audits": path_audits,
        "preview": preview,
    }


def audit_hard_negative_table(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "exists": False,
        }

    frame = pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False,
    )

    detected_path_columns = [
        column
        for column in (
            "subject_mask_path",
            "object_mask_path",
        )
        if column in frame.columns
    ]

    index_columns = [
        column
        for column in (
            "global_image_key",
            "image_id",
            "subject_index_v4",
            "object_index_v4",
            "subject_group",
            "object_group",
            "pair_family_v4",
        )
        if column in frame.columns
    ]

    index_preview = (
        frame[index_columns]
        .head(10)
        .fillna("")
        .to_dict(orient="records")
        if index_columns
        else []
    )

    return {
        "path": str(path),
        "exists": True,
        "rows": int(len(frame)),
        "detected_path_columns": (
            detected_path_columns
        ),
        "path_audits": {
            column: audit_path_column(
                frame,
                column,
            )
            for column
            in detected_path_columns
        },
        "index_preview": index_preview,
    }


def print_path_audit(
    title: str,
    audit: dict[str, Any],
) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    print("column:", audit["column"])
    print("total rows:", audit["total_rows"])
    print("nonempty rows:", audit["nonempty_rows"])
    print(
        "unique values:",
        audit["unique_nonempty_values"],
    )
    print(
        "resolved existing rows:",
        audit["resolved_existing_rows"],
    )
    print(
        "resolved rate:",
        audit["resolved_rate"],
    )

    print("\nexamples:")

    for example in audit["examples"]:
        print("- raw:", example["raw"])
        print(
            "  resolved:",
            example["resolved"],
        )

        for candidate in example[
            "candidate_paths"
        ]:
            print(
                "  candidate:",
                candidate,
            )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    canonical = audit_canonical_annotation()
    object_table = audit_object_table()

    hard_negative_tables = [
        audit_hard_negative_table(path)
        for path in HN_PATHS
    ]

    report = {
        "version": "FIBE_MASK_SOURCE_AUDIT_V1",
        "project_root": str(PROJECT_ROOT),
        "data_root": str(DATA_ROOT),
        "canonical": canonical,
        "object_table": object_table,
        "hard_negative_tables": (
            hard_negative_tables
        ),
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print("=" * 100)
    print("FIBE MASK SOURCE AUDIT V1")
    print("=" * 100)
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("DATA_ROOT:", DATA_ROOT)
    print("ANNOTATION_PATH:", ANNOTATION_PATH)
    print("OBJECT_TABLE_PATH:", OBJECT_TABLE_PATH)

    print("\nCANONICAL")
    print("images:", canonical["num_images"])
    print(
        "annotations:",
        canonical["total_annotations"],
    )
    print(
        "segments:",
        canonical["total_segments"],
    )
    print(
        "annotation keys:",
        canonical["annotation_key_counts"],
    )
    print(
        "segment keys:",
        canonical["segment_key_counts"],
    )

    print("\nEXAMPLE SEGMENTS_INFO")

    for record in canonical[
        "example_segments_info"
    ][:5]:
        print(record)

    print("\nOBJECT TABLE")
    print("rows:", object_table["rows"])
    print("columns:")
    print(object_table["columns"])
    print(
        "detected path columns:",
        object_table[
            "detected_path_columns"
        ],
    )

    print("\nOBJECT TABLE PREVIEW")

    for record in object_table["preview"][:5]:
        print(record)

    for column, audit in object_table[
        "path_audits"
    ].items():
        print_path_audit(
            f"OBJECT TABLE PATH: {column}",
            audit,
        )

    for table_audit in hard_negative_tables:
        print("\n" + "=" * 100)
        print(
            "HN TABLE:",
            table_audit["path"],
        )
        print("=" * 100)

        if not table_audit["exists"]:
            print("MISSING")
            continue

        print("rows:", table_audit["rows"])
        print(
            "path columns:",
            table_audit[
                "detected_path_columns"
            ],
        )

        for record in table_audit[
            "index_preview"
        ][:5]:
            print(record)

        for column, audit in table_audit[
            "path_audits"
        ].items():
            print_path_audit(
                f"HN PATH: {column}",
                audit,
            )

    print("\n" + "=" * 100)
    print("REPORT WRITTEN")
    print("=" * 100)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
