from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt


FEATURE_COUNT = 21
ROBUST_FEATURE_INDICES = (7, 8, 9, 17)

WATER_HAZARDS = frozenset({"water", "river"})
DEBRIS_HAZARDS = frozenset({"mud_debris"})
BARRICADE_HAZARDS = frozenset({"barricade"})
HAZARD_CATEGORIES = WATER_HAZARDS | DEBRIS_HAZARDS | BARRICADE_HAZARDS

WATER_TARGETS = frozenset(
    {
        "person",
        "vehicle",
        "road",
        "sidewalk",
        "ground",
        "building",
        "underpass_bridge",
        "drain",
        "manhole",
    }
)
ROAD_TARGETS = frozenset({"road", "sidewalk", "ground"})


@dataclass(frozen=True)
class FIBESpec:
    r1_norm: float
    r2_norm: float
    truncation_margin_px: int = 1
    lower50_start_fraction: float = 0.5
    lower25_start_fraction: float = 0.75
    minimum_shared_columns: int = 8
    minimum_x_coverage: float = 0.1
    maximum_fit_columns: int = 256


@dataclass
class ObjectGeometry:
    object_index: int
    category_name: str
    mask_path: Path
    mask: np.ndarray
    boundary: np.ndarray
    distance_to_boundary: np.ndarray
    area: int
    boundary_pixels: int
    bbox_xyxy: tuple[int, int, int, int]
    lower50_mask: np.ndarray
    lower25_mask: np.ndarray
    lower50_boundary: np.ndarray
    touches_border: bool
    x_support: np.ndarray
    top_y_by_x: np.ndarray
    bottom_y_by_x: np.ndarray


def load_binary_mask(path: Path) -> np.ndarray:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.uint8)

    mask = array > 0
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {mask.shape}: {path}")
    if not mask.any():
        raise ValueError(f"Mask has no foreground: {path}")
    return mask


def extract_boundary(mask: np.ndarray) -> np.ndarray:
    eroded = binary_erosion(
        mask,
        structure=np.ones((3, 3), dtype=bool),
        border_value=0,
    )
    boundary = mask & ~eroded
    if not boundary.any():
        boundary = mask.copy()
    return boundary


def mask_bbox_xyxy(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Cannot compute bbox for an empty mask")
    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def mask_touches_border(mask: np.ndarray, margin_px: int) -> bool:
    if margin_px < 0:
        raise ValueError("margin_px must be non-negative")
    band = margin_px + 1
    return bool(
        mask[:band, :].any()
        or mask[-band:, :].any()
        or mask[:, :band].any()
        or mask[:, -band:].any()
    )


def column_extrema(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = mask.shape
    support = mask.any(axis=0)

    top = np.full(width, -1, dtype=np.int32)
    bottom = np.full(width, -1, dtype=np.int32)

    columns = np.flatnonzero(support)
    for x in columns:
        ys = np.flatnonzero(mask[:, x])
        top[x] = int(ys[0])
        bottom[x] = int(ys[-1])

    return support, top, bottom


def build_object_geometry(
    *,
    object_index: int,
    category_name: str,
    mask_path: Path,
    expected_height: int,
    expected_width: int,
    spec: FIBESpec,
) -> ObjectGeometry:
    mask = load_binary_mask(mask_path)

    if mask.shape != (expected_height, expected_width):
        raise RuntimeError(
            "Mask/image shape mismatch: "
            f"object={object_index}, mask={mask.shape}, "
            f"expected={(expected_height, expected_width)}, path={mask_path}"
        )

    boundary = extract_boundary(mask)
    distance_to_boundary = distance_transform_edt(~boundary).astype(
        np.float32,
        copy=False,
    )
    area = int(mask.sum())
    boundary_pixels = int(boundary.sum())
    bbox = mask_bbox_xyxy(mask)
    x1, y1, x2, y2 = bbox
    bbox_height = y2 - y1

    lower50_y = y1 + int(math.floor(bbox_height * spec.lower50_start_fraction))
    lower25_y = y1 + int(math.floor(bbox_height * spec.lower25_start_fraction))
    lower50_y = max(y1, min(y2 - 1, lower50_y))
    lower25_y = max(y1, min(y2 - 1, lower25_y))

    lower50_mask = np.zeros_like(mask, dtype=bool)
    lower25_mask = np.zeros_like(mask, dtype=bool)
    lower50_mask[lower50_y:y2, x1:x2] = mask[lower50_y:y2, x1:x2]
    lower25_mask[lower25_y:y2, x1:x2] = mask[lower25_y:y2, x1:x2]
    lower50_boundary = boundary & lower50_mask

    x_support, top_y_by_x, bottom_y_by_x = column_extrema(mask)

    return ObjectGeometry(
        object_index=object_index,
        category_name=str(category_name),
        mask_path=mask_path.resolve(),
        mask=mask,
        boundary=boundary,
        distance_to_boundary=distance_to_boundary,
        area=area,
        boundary_pixels=boundary_pixels,
        bbox_xyxy=bbox,
        lower50_mask=lower50_mask,
        lower25_mask=lower25_mask,
        lower50_boundary=lower50_boundary,
        touches_border=mask_touches_border(
            mask,
            spec.truncation_margin_px,
        ),
        x_support=x_support,
        top_y_by_x=top_y_by_x,
        bottom_y_by_x=bottom_y_by_x,
    )


def supported_target_hazard(
    first_category: str,
    second_category: str,
) -> tuple[bool, bool]:
    """Return (is_supported, first_is_target).

    The decision uses category names only. Predicate labels and hard-negative
    labels are never consulted.
    """
    first = str(first_category)
    second = str(second_category)

    if second in WATER_HAZARDS and first in WATER_TARGETS:
        return True, True
    if first in WATER_HAZARDS and second in WATER_TARGETS:
        return True, False

    if second in DEBRIS_HAZARDS and first in ROAD_TARGETS:
        return True, True
    if first in DEBRIS_HAZARDS and second in ROAD_TARGETS:
        return True, False

    if second in BARRICADE_HAZARDS and first in ROAD_TARGETS:
        return True, True
    if first in BARRICADE_HAZARDS and second in ROAD_TARGETS:
        return True, False

    return False, False


def _safe_fraction(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _boundary_distance_values(
    source_boundary: np.ndarray,
    destination: ObjectGeometry,
) -> np.ndarray:
    values = destination.distance_to_boundary[source_boundary]
    if values.size == 0:
        return np.zeros(1, dtype=np.float32)
    return values.astype(np.float32, copy=False)


def _q10(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.quantile(values, 0.1))


def _deterministic_column_subsample(
    columns: np.ndarray,
    maximum_columns: int,
) -> np.ndarray:
    if len(columns) <= maximum_columns:
        return columns
    positions = np.linspace(
        0,
        len(columns) - 1,
        num=maximum_columns,
        dtype=np.int64,
    )
    return columns[positions]


def apparent_water_boundary_features(
    *,
    target: ObjectGeometry,
    hazard: ObjectGeometry,
    image_diagonal: float,
    spec: FIBESpec,
) -> tuple[float, float, float, float, bool]:
    if hazard.category_name not in WATER_HAZARDS:
        return 0.0, 0.0, 0.0, 0.0, False

    target_columns = np.flatnonzero(target.x_support)
    if len(target_columns) == 0:
        return 0.0, 0.0, 0.0, 0.0, False

    shared_mask = target.x_support & hazard.x_support
    shared_columns = np.flatnonzero(shared_mask)
    x_coverage = _safe_fraction(len(shared_columns), len(target_columns))

    if (
        len(shared_columns) < spec.minimum_shared_columns
        or x_coverage < spec.minimum_x_coverage
    ):
        return 0.0, 0.0, 0.0, 0.0, False

    fit_columns = _deterministic_column_subsample(
        shared_columns,
        spec.maximum_fit_columns,
    )
    x = fit_columns.astype(np.float64)
    y = hazard.top_y_by_x[fit_columns].astype(np.float64)

    if len(x) < spec.minimum_shared_columns:
        return 0.0, 0.0, 0.0, 0.0, False

    def fit_line(x_values: np.ndarray, y_values: np.ndarray) -> tuple[float, float]:
        x_center = float(x_values.mean())
        centered = x_values - x_center
        denominator = float(np.dot(centered, centered))
        slope = (
            float(np.dot(centered, y_values - y_values.mean()) / denominator)
            if denominator > 0
            else 0.0
        )
        intercept = float(y_values.mean() - slope * x_center)
        return slope, intercept

    slope, intercept = fit_line(x, y)
    residual = y - (slope * x + intercept)
    residual_median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - residual_median)))

    if mad > 0:
        robust_scale = 1.4826 * mad
        keep = np.abs(residual - residual_median) <= 3.0 * robust_scale
        if int(keep.sum()) >= spec.minimum_shared_columns:
            x = x[keep]
            y = y[keep]
            slope, intercept = fit_line(x, y)

    if len(x) < spec.minimum_shared_columns:
        return 0.0, 0.0, 0.0, 0.0, False

    angle = math.atan(slope)
    fitted_boundary = slope * x + intercept
    target_bottom = target.bottom_y_by_x[x.astype(np.int64)].astype(np.float64)
    valid_bottom = target_bottom >= 0

    if int(valid_bottom.sum()) < spec.minimum_shared_columns:
        return 0.0, 0.0, 0.0, 0.0, False

    offset_norm = float(
        np.median(target_bottom[valid_bottom] - fitted_boundary[valid_bottom])
        / image_diagonal
    )

    values = (
        float(x_coverage),
        float(math.sin(angle)),
        float(math.cos(angle)),
        offset_norm,
        True,
    )
    if not np.isfinite(np.asarray(values[:4], dtype=np.float64)).all():
        return 0.0, 0.0, 0.0, 0.0, False
    return values


def compute_pair_features(
    *,
    first: ObjectGeometry,
    second: ObjectGeometry,
    image_diagonal: float,
    spec: FIBESpec,
) -> tuple[np.ndarray, bool]:
    result = np.zeros(FEATURE_COUNT, dtype=np.float32)

    is_supported, first_is_target = supported_target_hazard(
        first.category_name,
        second.category_name,
    )
    if not is_supported or first.object_index == second.object_index:
        return result, False

    target = first if first_is_target else second
    hazard = second if first_is_target else first

    intersection = int(np.logical_and(target.mask, hazard.mask).sum())
    union = target.area + hazard.area - intersection

    result[0] = _safe_fraction(intersection, union)
    result[1] = _safe_fraction(intersection, target.area)
    result[2] = _safe_fraction(intersection, hazard.area)

    target_to_hazard = _boundary_distance_values(target.boundary, hazard)
    hazard_to_target = _boundary_distance_values(hazard.boundary, target)

    r1_px = spec.r1_norm * image_diagonal
    r2_px = spec.r2_norm * image_diagonal

    result[3] = float(np.mean(target_to_hazard <= r1_px))
    result[4] = float(np.mean(hazard_to_target <= r1_px))
    result[5] = float(np.mean(target_to_hazard <= r2_px))
    result[6] = float(np.mean(hazard_to_target <= r2_px))

    result[7] = (
        0.0
        if intersection > 0
        else float(
            min(target_to_hazard.min(), hazard_to_target.min())
            / image_diagonal
        )
    )
    result[8] = float(_q10(target_to_hazard) / image_diagonal)
    result[9] = float(_q10(hazard_to_target) / image_diagonal)

    lower50_area = int(target.lower50_mask.sum())
    lower25_area = int(target.lower25_mask.sum())
    result[10] = _safe_fraction(
        np.logical_and(target.lower50_mask, hazard.mask).sum(),
        lower50_area,
    )
    result[11] = _safe_fraction(
        np.logical_and(target.lower25_mask, hazard.mask).sum(),
        lower25_area,
    )

    lower50_distances = _boundary_distance_values(
        target.lower50_boundary,
        hazard,
    )
    result[12] = float(np.mean(lower50_distances <= r1_px))
    result[13] = float(np.mean(lower50_distances <= r2_px))

    (
        result[14],
        result[15],
        result[16],
        result[17],
        apparent_valid,
    ) = apparent_water_boundary_features(
        target=target,
        hazard=hazard,
        image_diagonal=image_diagonal,
        spec=spec,
    )

    result[18] = 1.0 if apparent_valid else 0.0
    result[19] = 1.0 if (target.touches_border or hazard.touches_border) else 0.0
    result[20] = 1.0

    if not np.isfinite(result).all():
        raise RuntimeError(
            "Non-finite FIBE feature vector: "
            f"first={first.object_index}, second={second.object_index}"
        )
    return result, True


def compute_image_feature_matrix(
    *,
    objects: Iterable[ObjectGeometry],
    image_height: int,
    image_width: int,
    spec: FIBESpec,
) -> tuple[np.ndarray, np.ndarray]:
    object_list = list(objects)
    num_objects = len(object_list)
    features = np.zeros(
        (num_objects, num_objects, FEATURE_COUNT),
        dtype=np.float32,
    )
    valid_pairs = np.zeros((num_objects, num_objects), dtype=bool)
    diagonal = math.hypot(image_width, image_height)

    if diagonal <= 0:
        raise ValueError(
            f"Invalid image size: {image_width}x{image_height}"
        )

    for first_index, first in enumerate(object_list):
        if first.object_index != first_index:
            raise RuntimeError(
                "Object list must be sorted and contiguous by canonical index: "
                f"position={first_index}, object_index={first.object_index}"
            )

        for second_index, second in enumerate(object_list):
            if first_index == second_index:
                continue

            vector, valid = compute_pair_features(
                first=first,
                second=second,
                image_diagonal=diagonal,
                spec=spec,
            )
            features[first_index, second_index] = vector
            valid_pairs[first_index, second_index] = valid

    invalid_values = features[~valid_pairs]
    if invalid_values.size and np.any(invalid_values != 0):
        raise RuntimeError("Invalid pairs contain non-zero FIBE features")

    if not np.array_equal(
        features[:, :, 20] > 0.5,
        valid_pairs,
    ):
        raise RuntimeError(
            "fibe_pair_valid feature disagrees with valid_pairs"
        )

    return features, valid_pairs


def fit_robust_scaler(
    train_features_by_image: dict[int, dict[str, Any]],
    *,
    epsilon: float = 1e-6,
) -> dict[str, Any]:
    statistics: dict[str, Any] = {
        "version": "FIBE_SCALAR_ROBUST_SCALER_V1",
        "epsilon": float(epsilon),
        "clip": [-5.0, 5.0],
        "features": {},
    }

    for feature_index in ROBUST_FEATURE_INDICES:
        values: list[np.ndarray] = []

        for entry in train_features_by_image.values():
            matrix = np.asarray(entry["features"], dtype=np.float32)
            valid_pairs = np.asarray(entry["valid_pairs"], dtype=bool)
            selection = valid_pairs.copy()

            if feature_index == 17:
                selection &= matrix[:, :, 18] > 0.5

            selected = matrix[:, :, feature_index][selection]
            if selected.size:
                values.append(selected.astype(np.float64, copy=False))

        if not values:
            raise RuntimeError(
                f"No training values available for robust feature {feature_index}"
            )

        merged = np.concatenate(values)
        median = float(np.median(merged))
        q25 = float(np.percentile(merged, 25))
        q75 = float(np.percentile(merged, 75))
        iqr = float(q75 - q25)
        denominator = max(iqr, epsilon)

        statistics["features"][str(feature_index)] = {
            "count": int(len(merged)),
            "median": median,
            "q25": q25,
            "q75": q75,
            "iqr": iqr,
            "denominator": denominator,
        }

    return statistics


def apply_robust_scaler(
    features_by_image: dict[int, dict[str, Any]],
    scaler: dict[str, Any],
) -> None:
    clip_min, clip_max = (
        float(scaler["clip"][0]),
        float(scaler["clip"][1]),
    )

    for entry in features_by_image.values():
        matrix = np.asarray(entry["features"], dtype=np.float32)
        valid_pairs = np.asarray(entry["valid_pairs"], dtype=bool)

        for index_text, feature_stats in scaler["features"].items():
            feature_index = int(index_text)
            selection = valid_pairs.copy()

            if feature_index == 17:
                selection &= matrix[:, :, 18] > 0.5

            values = matrix[:, :, feature_index]
            values[selection] = np.clip(
                (
                    values[selection]
                    - float(feature_stats["median"])
                )
                / float(feature_stats["denominator"]),
                clip_min,
                clip_max,
            )
            values[~selection] = 0.0

        matrix[~valid_pairs] = 0.0
        matrix[:, :, 20] = valid_pairs.astype(np.float32)

        if not np.isfinite(matrix).all():
            raise RuntimeError("Scaled FIBE cache contains NaN or Inf")
