from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_PATH = (
    PROJECT_ROOT / "data" / "floodpsg" / "stats"
    / "dsformer_fibe_d1_v1"
    / "FIBE_CONTACT_RADII_CALIBRATION_V1.json"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "configs" / "floodpsg" / "fibe_scalar_v1.json"
)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def feature(index: int, name: str, group: str, kind: str,
            scaling: str, definition: str) -> dict[str, Any]:
    return {
        "index": index,
        "name": name,
        "group": group,
        "kind": kind,
        "scaling": scaling,
        "definition": definition,
    }

def main() -> None:
    if not CALIBRATION_PATH.is_file():
        raise FileNotFoundError(CALIBRATION_PATH)

    calibration = json.loads(
        CALIBRATION_PATH.read_text(encoding="utf-8")
    )
    r1 = float(calibration["r1"]["normalized_radius"])
    r2 = float(calibration["r2"]["normalized_radius"])

    if not 0.0 <= r1 < r2:
        raise RuntimeError(f"Invalid radii: r1={r1}, r2={r2}")

    features = [
        feature(
            0,
            'mask_iou',
            'overlap_soft_topology',
            'ratio',
            'identity_0_1',
            'intersection(target,hazard) / union(target,hazard)',
        ),
        feature(
            1,
            'target_overlap_fraction',
            'overlap_soft_topology',
            'ratio',
            'identity_0_1',
            'intersection(target,hazard) / area(target)',
        ),
        feature(
            2,
            'hazard_overlap_fraction',
            'overlap_soft_topology',
            'ratio',
            'identity_0_1',
            'intersection(target,hazard) / area(hazard)',
        ),
        feature(
            3,
            'target_boundary_within_r1_fraction',
            'overlap_soft_topology',
            'ratio',
            'identity_0_1',
            'fraction of target-boundary pixels with distance to hazard boundary <= r1 * image diagonal',
        ),
        feature(
            4,
            'hazard_boundary_within_r1_fraction',
            'overlap_soft_topology',
            'ratio',
            'identity_0_1',
            'fraction of hazard-boundary pixels with distance to target boundary <= r1 * image diagonal',
        ),
        feature(
            5,
            'target_boundary_within_r2_fraction',
            'overlap_soft_topology',
            'ratio',
            'identity_0_1',
            'fraction of target-boundary pixels with distance to hazard boundary <= r2 * image diagonal',
        ),
        feature(
            6,
            'hazard_boundary_within_r2_fraction',
            'overlap_soft_topology',
            'ratio',
            'identity_0_1',
            'fraction of hazard-boundary pixels with distance to target boundary <= r2 * image diagonal',
        ),
        feature(
            7,
            'minimum_boundary_distance_norm',
            'distance',
            'distance',
            'robust_train_only',
            'minimum Euclidean target-hazard boundary distance divided by image diagonal',
        ),
        feature(
            8,
            'target_to_hazard_boundary_q10_norm',
            'distance',
            'distance',
            'robust_train_only',
            '10th percentile target-boundary distance to hazard boundary divided by image diagonal',
        ),
        feature(
            9,
            'hazard_to_target_boundary_q10_norm',
            'distance',
            'distance',
            'robust_train_only',
            '10th percentile hazard-boundary distance to target boundary divided by image diagonal',
        ),
        feature(
            10,
            'target_lower50_overlap_fraction',
            'lower_interaction',
            'ratio',
            'identity_0_1',
            'hazard overlap fraction inside the lower 50 percent of the independent target-mask bounding box',
        ),
        feature(
            11,
            'target_lower25_overlap_fraction',
            'lower_interaction',
            'ratio',
            'identity_0_1',
            'hazard overlap fraction inside the lower 25 percent of the independent target-mask bounding box',
        ),
        feature(
            12,
            'target_lower50_boundary_within_r1_fraction',
            'lower_interaction',
            'ratio',
            'identity_0_1',
            'fraction of lower-50-percent target-boundary pixels whose hazard-boundary distance is <= r1',
        ),
        feature(
            13,
            'target_lower50_boundary_within_r2_fraction',
            'lower_interaction',
            'ratio',
            'identity_0_1',
            'fraction of lower-50-percent target-boundary pixels whose hazard-boundary distance is <= r2',
        ),
        feature(
            14,
            'apparent_boundary_x_coverage',
            'apparent_water_boundary',
            'ratio',
            'identity_0_1',
            'fraction of target-support columns containing an observable water or river apparent upper boundary',
        ),
        feature(
            15,
            'apparent_boundary_slope_sin',
            'apparent_water_boundary',
            'angle',
            'identity_minus1_1',
            'sine of the fitted apparent water-boundary angle',
        ),
        feature(
            16,
            'apparent_boundary_slope_cos',
            'apparent_water_boundary',
            'angle',
            'identity_minus1_1',
            'cosine of the fitted apparent water-boundary angle',
        ),
        feature(
            17,
            'target_bottom_offset_from_apparent_boundary_norm',
            'apparent_water_boundary',
            'signed_distance',
            'robust_train_only',
            'median target-bottom minus apparent water-boundary vertical coordinate divided by image diagonal',
        ),
        feature(
            18,
            'apparent_boundary_valid',
            'validity_flags',
            'flag',
            'identity_binary',
            '1 when a local apparent water or river boundary can be computed, otherwise 0',
        ),
        feature(
            19,
            'pair_truncated',
            'validity_flags',
            'flag',
            'identity_binary',
            '1 when either independent mask reaches the image border within one pixel',
        ),
        feature(
            20,
            'fibe_pair_valid',
            'validity_flags',
            'flag',
            'identity_binary',
            '1 only for a supported canonical target-hazard pair with valid independent masks',
        ),
    ]

    spec = {
        "version": "FIBE_SCALAR_V1",
        "feature_count": 21,
        "dtype": "float32",
        "calibration": {
            "summary_path": (
                "data/floodpsg/stats/dsformer_fibe_d1_v1/"
                "FIBE_CONTACT_RADII_CALIBRATION_V1.json"
            ),
            "summary_sha256": sha256_file(CALIBRATION_PATH),
            "eligible_input_sha256": calibration["input_sha256"],
            "calibration_rows": int(calibration["rows"]),
            "calibration_images": int(calibration["images"]),
            "r1_norm": r1,
            "r2_norm": r2,
            "r1_bootstrap_95": calibration["r1"]["cluster_bootstrap_interval"],
            "r2_bootstrap_95": calibration["r2"]["cluster_bootstrap_interval"],
            "distance_normalization": "full_image_diagonal",
            "pixel_radius_rule": (
                "float_radius_equals_normalized_radius_times_image_diagonal"
            ),
            "integer_rounding": False,
            "threshold_comparison": (
                "euclidean_distance_less_than_or_equal_to_float_radius"
            ),
        },
        "pair_orientation": {
            "hazard_categories": [
                "water", "river", "mud_debris", "barricade"
            ],
            "target_categories": [
                "person", "vehicle", "road", "sidewalk", "ground",
                "building", "underpass_bridge", "drain", "manhole"
            ],
            "orientation_source": "canonical_category_name_only",
            "predicate_labels_used": False,
            "hard_negative_labels_used": False,
        },
        "mask_processing": {
            "mask_source": "independent_overlapping_binary_masks",
            "panoptic_mask_fallback": False,
            "foreground_rule": "pixel_value_greater_than_0",
            "boundary_rule": "mask AND NOT binary_erosion",
            "erosion_kernel": "3x3_ones",
            "connectivity": 8,
            "erosion_border_value": 0,
            "distance_transform": "scipy_ndimage_distance_transform_edt",
            "truncation_margin_px": 1,
        },
        "lower_interaction": {
            "reference": "independent_target_mask_bbox",
            "lower50_start_fraction": 0.5,
            "lower25_start_fraction": 0.75,
        },
        "apparent_water_boundary": {
            "supported_hazard_categories": ["water", "river"],
            "target_support": "columns_containing_target_foreground",
            "shared_support": "target_support_intersection_hazard_support",
            "water_boundary_per_column": "topmost_hazard_foreground_pixel",
            "target_bottom_per_column": "bottommost_target_foreground_pixel",
            "minimum_shared_columns": 8,
            "minimum_x_coverage": 0.1,
            "maximum_fit_columns": 256,
            "fit_method": (
                "centered_least_squares_after_single_mad_outlier_rejection"
            ),
            "invalid_numeric_policy": (
                "set_all_four_apparent_boundary_features_to_zero"
            ),
        },
        "scaling": {
            "fit_split": "train_only",
            "robust_feature_indices": [7, 8, 9, 17],
            "robust_formula": (
                "(value - train_median) / max(train_iqr, 1e-6)"
            ),
            "robust_clip": [-5.0, 5.0],
            "ratio_features_remain_unscaled": True,
            "angle_features_remain_unscaled": True,
            "flags_remain_unscaled": True,
        },
        "invalid_pair_policy": {
            "all_feature_values": 0.0,
            "fibe_pair_valid": 0.0,
            "gate_multiplier": 0.0,
            "residual_multiplier": 0.0,
        },
        "features": features,
    }

    if len(features) != 21:
        raise RuntimeError(f"Expected 21 features, found {len(features)}")
    if [item["index"] for item in features] != list(range(21)):
        raise RuntimeError("Feature indexes are not 0..20")
    if len({item["name"] for item in features}) != 21:
        raise RuntimeError("Feature names are not unique")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = OUTPUT_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(spec, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(OUTPUT_PATH)

    print("Wrote:", OUTPUT_PATH)
    print("r1:", r1)
    print("r2:", r2)
    print("features:", len(features))
    print("config_sha256:", sha256_file(OUTPUT_PATH))

if __name__ == "__main__":
    main()
