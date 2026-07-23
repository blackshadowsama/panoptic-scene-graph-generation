from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (
    PROJECT_ROOT / "data" / "floodpsg"
).resolve()

DEFAULT_PAIR_POOL = (
    DATA_ROOT
    / "features"
    / "fibe_scalar_v1"
    / "fibe_boundary_audit_pair_pool_train_v1.csv"
)

DEFAULT_OUTPUT = (
    DATA_ROOT
    / "features"
    / "fibe_scalar_v1"
    / "fibe_boundary_audit_pair_pool_train_geometry_v1.csv"
)

DEFAULT_SUMMARY = (
    DATA_ROOT
    / "stats"
    / "dsformer_fibe_d1_v1"
    / "FIBE_PAIR_GEOMETRY_PROXY_AUDIT_V1.json"
)


@dataclass
class MaskData:
    path: Path
    mask: np.ndarray
    boundary: np.ndarray
    boundary_coords: np.ndarray
    area: int
    boundary_pixels: int
    height: int
    width: int
    bbox_xyxy: tuple[int, int, int, int]
    touches_border: bool
    _tree: cKDTree | None = None

    @property
    def tree(self) -> cKDTree:
        if self._tree is None:
            if len(self.boundary_coords) == 0:
                raise RuntimeError(
                    f"Empty boundary: {self.path}"
                )

            self._tree = cKDTree(
                self.boundary_coords
            )

        return self._tree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute independent-mask geometry proxies "
            "for the training-only FIBE boundary audit pool."
        )
    )

    parser.add_argument(
        "--pair-pool",
        type=Path,
        default=DEFAULT_PAIR_POOL,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
    )
    parser.add_argument(
        "--border-margin",
        type=int,
        default=1,
        help=(
            "A mask is truncated when foreground "
            "appears within this many pixels of an "
            "image border."
        ),
    )

    return parser.parse_args()


def resolve_mask_path(value: Any) -> Path:
    text = str(value).strip()
    path = Path(text)

    if path.is_absolute():
        return path.resolve()

    return (DATA_ROOT / path).resolve()


def mask_touches_border(
    mask: np.ndarray,
    margin: int,
) -> bool:
    if margin < 0:
        raise ValueError(
            "border margin must be non-negative"
        )

    band = margin + 1

    return bool(
        mask[:band, :].any()
        or mask[-band:, :].any()
        or mask[:, :band].any()
        or mask[:, -band:].any()
    )


def mask_bbox(
    mask: np.ndarray,
) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)

    if len(xs) == 0:
        raise ValueError(
            "Mask has no foreground pixels"
        )

    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def load_mask(
    path: Path,
    *,
    border_margin: int,
) -> MaskData:
    if not path.is_file():
        raise FileNotFoundError(path)

    with Image.open(path) as image:
        array = np.asarray(
            image.convert("L"),
            dtype=np.uint8,
        )

    mask = array > 0
    area = int(mask.sum())

    if area <= 0:
        raise ValueError(
            f"Empty mask: {path}"
        )

    structure = np.ones(
        (3, 3),
        dtype=bool,
    )

    eroded = binary_erosion(
        mask,
        structure=structure,
        border_value=0,
    )

    boundary = np.logical_and(
        mask,
        np.logical_not(eroded),
    )

    if not boundary.any():
        boundary = mask.copy()

    boundary_coords = np.argwhere(
        boundary
    ).astype(np.float32)

    height, width = mask.shape

    return MaskData(
        path=path,
        mask=mask,
        boundary=boundary,
        boundary_coords=boundary_coords,
        area=area,
        boundary_pixels=int(
            boundary.sum()
        ),
        height=int(height),
        width=int(width),
        bbox_xyxy=mask_bbox(mask),
        touches_border=mask_touches_border(
            mask,
            border_margin,
        ),
    )


def intersection_pixels(
    first: MaskData,
    second: MaskData,
) -> int:
    ax1, ay1, ax2, ay2 = (
        first.bbox_xyxy
    )
    bx1, by1, bx2, by2 = (
        second.bbox_xyxy
    )

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    if x1 >= x2 or y1 >= y2:
        return 0

    return int(
        np.logical_and(
            first.mask[y1:y2, x1:x2],
            second.mask[y1:y2, x1:x2],
        ).sum()
    )


def minimum_boundary_distance(
    first: MaskData,
    second: MaskData,
) -> float:
    first_coords = (
        first.boundary_coords
    )
    second_coords = (
        second.boundary_coords
    )

    if len(first_coords) <= len(
        second_coords
    ):
        distances, _ = (
            second.tree.query(
                first_coords,
                k=1,
            )
        )
    else:
        distances, _ = (
            first.tree.query(
                second_coords,
                k=1,
            )
        )

    return float(
        np.min(distances)
    )


def finite_float(value: Any) -> float:
    result = float(value)

    if not math.isfinite(result):
        raise ValueError(
            f"Non-finite value: {value}"
        )

    return result


def main() -> None:
    args = parse_args()

    pair_pool_path = (
        args.pair_pool
        .expanduser()
        .resolve()
    )
    output_path = (
        args.output
        .expanduser()
        .resolve()
    )
    summary_path = (
        args.summary
        .expanduser()
        .resolve()
    )

    if not pair_pool_path.is_file():
        raise FileNotFoundError(
            pair_pool_path
        )

    frame = pd.read_csv(
        pair_pool_path,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required_columns = {
        "pool_pair_id",
        "image_id",
        "source_type",
        "pair_family",
        "target_mask_path",
        "hazard_mask_path",
    }

    missing_columns = (
        required_columns
        - set(frame.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Pair pool missing columns: "
            f"{sorted(missing_columns)}"
        )

    if frame["pool_pair_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate pool_pair_id values"
        )

    result_rows: list[
        dict[str, Any]
    ] = []

    missing_masks = 0
    shape_mismatches = 0
    invalid_pairs = 0

    grouped = frame.groupby(
        "image_id",
        sort=True,
    )

    total_images = int(
        frame["image_id"].nunique()
    )

    for group_number, (
        image_id,
        group,
    ) in enumerate(grouped, start=1):
        mask_cache: dict[
            str,
            MaskData,
        ] = {}

        unique_paths = sorted(
            set(
                group[
                    "target_mask_path"
                ].astype(str)
            )
            | set(
                group[
                    "hazard_mask_path"
                ].astype(str)
            )
        )

        for raw_path in unique_paths:
            resolved = resolve_mask_path(
                raw_path
            )

            try:
                mask_cache[raw_path] = (
                    load_mask(
                        resolved,
                        border_margin=(
                            args.border_margin
                        ),
                    )
                )
            except FileNotFoundError:
                missing_masks += 1
                raise

        for row in group.to_dict(
            orient="records"
        ):
            target = mask_cache[
                str(row["target_mask_path"])
            ]
            hazard = mask_cache[
                str(row["hazard_mask_path"])
            ]

            if (
                target.height
                != hazard.height
                or target.width
                != hazard.width
            ):
                shape_mismatches += 1
                raise RuntimeError(
                    "Target/hazard shape mismatch "
                    f"for {row['pool_pair_id']}: "
                    f"{target.width}x"
                    f"{target.height} versus "
                    f"{hazard.width}x"
                    f"{hazard.height}"
                )

            intersection = (
                intersection_pixels(
                    target,
                    hazard,
                )
            )

            union = (
                target.area
                + hazard.area
                - intersection
            )

            if union <= 0:
                invalid_pairs += 1
                raise RuntimeError(
                    "Invalid union area for "
                    f"{row['pool_pair_id']}"
                )

            iou = intersection / union
            target_overlap = (
                intersection / target.area
            )
            hazard_overlap = (
                intersection / hazard.area
            )

            if intersection > 0:
                minimum_distance = 0.0
                base_topology = "overlap"
            else:
                minimum_distance = (
                    minimum_boundary_distance(
                        target,
                        hazard,
                    )
                )

                if (
                    minimum_distance
                    <= math.sqrt(2.0)
                    + 1e-6
                ):
                    base_topology = (
                        "pixel_touching"
                    )
                else:
                    base_topology = (
                        "nonoverlap"
                    )

            diagonal = math.hypot(
                target.width,
                target.height,
            )

            normalized_distance = (
                minimum_distance / diagonal
                if diagonal > 0
                else 0.0
            )

            output = dict(row)

            output.update(
                {
                    "target_mask_area": (
                        target.area
                    ),
                    "hazard_mask_area": (
                        hazard.area
                    ),
                    "target_boundary_pixels": (
                        target.boundary_pixels
                    ),
                    "hazard_boundary_pixels": (
                        hazard.boundary_pixels
                    ),
                    "intersection_pixels": (
                        intersection
                    ),
                    "union_pixels": union,
                    "mask_iou": finite_float(
                        iou
                    ),
                    "target_overlap_fraction": (
                        finite_float(
                            target_overlap
                        )
                    ),
                    "hazard_overlap_fraction": (
                        finite_float(
                            hazard_overlap
                        )
                    ),
                    "minimum_boundary_distance_px": (
                        finite_float(
                            minimum_distance
                        )
                    ),
                    "minimum_boundary_distance_norm": (
                        finite_float(
                            normalized_distance
                        )
                    ),
                    "geometry_base_topology": (
                        base_topology
                    ),
                    "target_mask_truncated": int(
                        target.touches_border
                    ),
                    "hazard_mask_truncated": int(
                        hazard.touches_border
                    ),
                    "pair_mask_truncated": int(
                        target.touches_border
                        or hazard.touches_border
                    ),
                }
            )

            result_rows.append(output)

        if (
            group_number % 50 == 0
            or group_number == total_images
        ):
            print(
                f"processed images: "
                f"{group_number}/"
                f"{total_images}",
                flush=True,
            )

    result = pd.DataFrame(
        result_rows
    )

    if len(result) != len(frame):
        raise RuntimeError(
            "Output row count mismatch: "
            f"{len(result)} != "
            f"{len(frame)}"
        )

    result[
        "geometry_topology_proxy"
    ] = result[
        "geometry_base_topology"
    ].copy()

    distance_thresholds: list[
        dict[str, Any]
    ] = []

    nonoverlap = result[
        result["geometry_base_topology"]
        == "nonoverlap"
    ]

    for (
        source_type,
        family,
    ), indexes in nonoverlap.groupby(
        [
            "source_type",
            "pair_family",
        ],
        sort=True,
    ).groups.items():
        index_list = list(indexes)

        distances = result.loc[
            index_list,
            "minimum_boundary_distance_norm",
        ].to_numpy(dtype=float)

        if len(distances) >= 3:
            q33 = float(
                np.quantile(
                    distances,
                    1 / 3,
                )
            )
            q67 = float(
                np.quantile(
                    distances,
                    2 / 3,
                )
            )
        elif len(distances) == 2:
            q33 = float(
                np.min(distances)
            )
            q67 = float(
                np.max(distances)
            )
        elif len(distances) == 1:
            q33 = float(distances[0])
            q67 = float(distances[0])
        else:
            continue

        distance_thresholds.append(
            {
                "source_type": (
                    str(source_type)
                ),
                "pair_family": (
                    str(family)
                ),
                "nonoverlap_rows": int(
                    len(distances)
                ),
                "distance_norm_q33": (
                    q33
                ),
                "distance_norm_q67": (
                    q67
                ),
            }
        )

        for index in index_list:
            value = float(
                result.at[
                    index,
                    "minimum_boundary_distance_norm",
                ]
            )

            if value <= q33:
                label = (
                    "nonoverlap_near"
                )
            elif value <= q67:
                label = (
                    "nonoverlap_mid"
                )
            else:
                label = (
                    "nonoverlap_far"
                )

            result.at[
                index,
                "geometry_topology_proxy",
            ] = label

    result.sort_values(
        by=[
            "source_type",
            "pair_family",
            "image_id",
            "target_index",
            "hazard_index",
        ],
        inplace=True,
        kind="stable",
    )

    result.reset_index(
        drop=True,
        inplace=True,
    )

    numeric_columns = [
        "mask_iou",
        "target_overlap_fraction",
        "hazard_overlap_fraction",
        "minimum_boundary_distance_px",
        "minimum_boundary_distance_norm",
    ]

    nonfinite_values = 0

    for column in numeric_columns:
        values = result[
            column
        ].to_numpy(dtype=float)

        nonfinite_values += int(
            (~np.isfinite(values)).sum()
        )

    if nonfinite_values:
        raise RuntimeError(
            "Non-finite geometry values: "
            f"{nonfinite_values}"
        )

    if (
        result["pool_pair_id"]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Duplicate pair IDs after geometry "
            "computation"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    topology_counts = (
        result.groupby(
            [
                "source_type",
                "pair_family",
                "geometry_topology_proxy",
            ],
            observed=True,
        )
        .size()
        .reset_index(name="count")
    )

    truncation_counts = (
        result.groupby(
            [
                "source_type",
                "pair_mask_truncated",
            ],
            observed=True,
        )
        .size()
        .reset_index(name="count")
    )

    old_new_truncation = None

    if "pair_truncated" in result.columns:
        old_new_truncation = (
            pd.crosstab(
                result["pair_truncated"],
                result[
                    "pair_mask_truncated"
                ],
            )
            .rename_axis(
                index="canonical_bbox_flag",
                columns="independent_mask_flag",
            )
            .reset_index()
            .to_dict(orient="records")
        )

    summary = {
        "version": (
            "FIBE_PAIR_GEOMETRY_PROXY_AUDIT_V1"
        ),
        "pair_pool_path": str(
            pair_pool_path
        ),
        "output_path": str(
            output_path
        ),
        "border_margin": int(
            args.border_margin
        ),
        "rows": int(len(result)),
        "images": int(
            result["image_id"].nunique()
        ),
        "missing_masks": int(
            missing_masks
        ),
        "shape_mismatches": int(
            shape_mismatches
        ),
        "invalid_pairs": int(
            invalid_pairs
        ),
        "nonfinite_values": int(
            nonfinite_values
        ),
        "distance_thresholds": (
            distance_thresholds
        ),
        "topology_counts": (
            topology_counts.to_dict(
                orient="records"
            )
        ),
        "truncation_counts": (
            truncation_counts.to_dict(
                orient="records"
            )
        ),
        "old_new_truncation_crosstab": (
            old_new_truncation
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

    print("\n" + "=" * 100)
    print(
        "FIBE PAIR GEOMETRY PROXY "
        "AUDIT V1"
    )
    print("=" * 100)
    print("rows:", len(result))
    print(
        "images:",
        result["image_id"].nunique(),
    )
    print(
        "missing masks:",
        missing_masks,
    )
    print(
        "shape mismatches:",
        shape_mismatches,
    )
    print(
        "invalid pairs:",
        invalid_pairs,
    )
    print(
        "non-finite values:",
        nonfinite_values,
    )

    print("\nSOURCE × TOPOLOGY")
    print(
        pd.crosstab(
            result["source_type"],
            result[
                "geometry_topology_proxy"
            ],
        ).to_string()
    )

    print("\nFAMILY × SOURCE × TOPOLOGY")
    print(
        pd.crosstab(
            [
                result["pair_family"],
                result["source_type"],
            ],
            result[
                "geometry_topology_proxy"
            ],
        ).to_string()
    )

    print("\nINDEPENDENT MASK TRUNCATION")
    print(
        pd.crosstab(
            result["source_type"],
            result[
                "pair_mask_truncated"
            ],
        ).to_string()
    )

    if "pair_truncated" in result.columns:
        print(
            "\nCANONICAL BBOX FLAG × "
            "INDEPENDENT MASK FLAG"
        )
        print(
            pd.crosstab(
                result[
                    "pair_truncated"
                ],
                result[
                    "pair_mask_truncated"
                ],
                rownames=[
                    "canonical_bbox_flag"
                ],
                colnames=[
                    "independent_mask_flag"
                ],
            ).to_string()
        )

    print("\nOUTPUTS")
    print("geometry pool:", output_path)
    print("summary:", summary_path)

    print(
        "\nFIBE PAIR GEOMETRY "
        "PROXY AUDIT: PASS"
    )


if __name__ == "__main__":
    main()
