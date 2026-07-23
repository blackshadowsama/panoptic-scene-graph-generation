from __future__ import annotations

import argparse
import html
import json
import os
import re
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.ndimage import binary_erosion


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

DEFAULT_OUTPUT_DIR = (
    DATA_ROOT
    / "features"
    / "fibe_scalar_v1"
    / "boundary_audit_visuals_v1"
)

DEFAULT_MANIFEST = (
    DATA_ROOT
    / "features"
    / "fibe_scalar_v1"
    / "fibe_boundary_audit_visual_manifest_v1.csv"
)

DEFAULT_SUMMARY = (
    DATA_ROOT
    / "stats"
    / "dsformer_fibe_d1_v1"
    / "FIBE_BOUNDARY_AUDIT_VISUALS_V1.json"
)

DEFAULT_FAILURES = (
    DATA_ROOT
    / "stats"
    / "dsformer_fibe_d1_v1"
    / "FIBE_BOUNDARY_AUDIT_VISUAL_FAILURES_V1.tsv"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

SKIP_IMAGE_INDEX_PARTS = {
    "binary_masks_canonical_v1",
    "panoptic",
    "panoptic_seg",
    "features",
    "stats",
    "annotations",
}

TARGET_COLOR = np.array(
    [255, 55, 55],
    dtype=np.float32,
)

HAZARD_COLOR = np.array(
    [20, 160, 255],
    dtype=np.float32,
)


CATEGORY_ZH = {
    "person": "人员",
    "vehicle": "车辆",
    "road": "道路",
    "sidewalk": "人行道",
    "ground": "地面",
    "building": "建筑物",
    "underpass_bridge": "桥梁或地下通道",
    "drain": "排水口",
    "manhole": "检查井",
    "water": "积水或水体",
    "river": "河流",
    "mud_debris": "泥沙或碎屑",
    "barricade": "路障",
    "vegetation": "植被",
    "sign": "标志牌",
    "sky": "天空",
}


FAMILY_ZH = {
    "human-water": "人员—水体",
    "vehicle-water": "车辆—水体",
    "road-surface-water": "道路表面—水体",
    "building-water": "建筑物—水体",
    "bridge-water": "桥梁或地下通道—水体",
    "drainage-water": "排水设施—水体",
    "road-debris": "道路—泥沙碎屑",
    "road-barricade": "道路—路障",
}


def zh_category(value: Any) -> str:
    text = str(value)
    return CATEGORY_ZH.get(text, text)


def zh_family(value: Any) -> str:
    text = str(value)
    return FAMILY_ZH.get(text, text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render blinded four-panel visualizations "
            "for the frozen FIBE boundary audit sample."
        )
    )

    parser.add_argument(
        "--review",
        type=Path,
        default=DEFAULT_REVIEW,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
    )
    parser.add_argument(
        "--failures",
        type=Path,
        default=DEFAULT_FAILURES,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
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


def windows_to_wsl_path(
    value: str,
) -> Path | None:
    if not re.match(
        r"^[A-Za-z]:[\\/]",
        value,
    ):
        return None

    windows_path = PureWindowsPath(value)
    drive = windows_path.drive[0].lower()

    return (
        Path("/mnt")
        / drive
        / Path(*windows_path.parts[1:])
    )


def unique_paths(
    paths: list[Path],
) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()

    for path in paths:
        key = str(path)

        if key in seen:
            continue

        seen.add(key)
        result.append(path)

    return result


def resolve_data_path(
    value: Any,
) -> Path:
    text = clean_text(value)

    if not text:
        raise ValueError(
            "Empty data path"
        )

    raw_path = Path(text).expanduser()

    candidates: list[Path] = []

    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend(
            [
                DATA_ROOT / raw_path,
                PROJECT_ROOT / raw_path,
            ]
        )

    converted = windows_to_wsl_path(
        text
    )

    if converted is not None:
        candidates.append(converted)

    for candidate in unique_paths(
        candidates
    ):
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "Cannot resolve data path: "
        f"{text}; candidates="
        f"{[str(path) for path in candidates]}"
    )


def build_image_index(
    root: Path,
) -> tuple[
    dict[str, list[Path]],
    list[Path],
]:
    by_basename: dict[
        str,
        list[Path],
    ] = {}

    all_images: list[Path] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if (
            path.suffix.lower()
            not in IMAGE_EXTENSIONS
        ):
            continue

        lowered_parts = {
            part.lower()
            for part in path.parts
        }

        if (
            lowered_parts
            & SKIP_IMAGE_INDEX_PARTS
        ):
            continue

        resolved = path.resolve()
        all_images.append(resolved)

        by_basename.setdefault(
            path.name.lower(),
            [],
        ).append(resolved)

    return by_basename, all_images


def direct_image_candidates(
    raw_value: str,
    global_image_key: str,
) -> list[Path]:
    candidates: list[Path] = []

    raw_path = Path(raw_value).expanduser()

    if raw_path.is_absolute():
        candidates.append(raw_path)
    elif raw_value:
        candidates.extend(
            [
                DATA_ROOT / raw_path,
                DATA_ROOT / "images" / raw_path,
                DATA_ROOT
                / "images"
                / raw_path.name,
                PROJECT_ROOT / raw_path,
            ]
        )

    converted = windows_to_wsl_path(
        raw_value
    )

    if converted is not None:
        candidates.append(converted)

    batch_id = ""
    filename = ""

    if "__" in global_image_key:
        batch_id, filename = (
            global_image_key.split(
                "__",
                1,
            )
        )

    if filename:
        candidates.extend(
            [
                DATA_ROOT
                / "images"
                / filename,
                DATA_ROOT
                / "images"
                / batch_id
                / filename,
                DATA_ROOT
                / batch_id
                / "images"
                / filename,
                DATA_ROOT
                / batch_id
                / filename,
            ]
        )

    return unique_paths(candidates)


def resolve_image_path(
    raw_value: Any,
    global_image_key: Any,
    basename_index: dict[
        str,
        list[Path],
    ],
    all_images: list[Path],
) -> Path:
    raw_text = clean_text(raw_value)
    global_key = clean_text(
        global_image_key
    )

    for candidate in direct_image_candidates(
        raw_text,
        global_key,
    ):
        if candidate.is_file():
            return candidate.resolve()

    batch_id = ""
    global_filename = ""

    if "__" in global_key:
        batch_id, global_filename = (
            global_key.split(
                "__",
                1,
            )
        )

    requested_names = []

    if raw_text:
        requested_names.append(
            Path(raw_text).name
        )

    if global_filename:
        requested_names.append(
            Path(global_filename).name
        )

    requested_names = list(
        dict.fromkeys(
            name.lower()
            for name in requested_names
            if name
        )
    )

    exact_matches: list[Path] = []

    for name in requested_names:
        exact_matches.extend(
            basename_index.get(
                name,
                [],
            )
        )

    exact_matches = unique_paths(
        exact_matches
    )

    if batch_id:
        batch_matches = [
            path
            for path in exact_matches
            if batch_id.lower()
            in str(path).lower()
        ]

        if len(batch_matches) == 1:
            return batch_matches[0]

    if len(exact_matches) == 1:
        return exact_matches[0]

    suffix_matches: list[Path] = []

    for requested_name in requested_names:
        suffix_matches.extend(
            path
            for path in all_images
            if path.name.lower().endswith(
                requested_name
            )
        )

    suffix_matches = unique_paths(
        suffix_matches
    )

    if batch_id:
        batch_suffix_matches = [
            path
            for path in suffix_matches
            if batch_id.lower()
            in str(path).lower()
        ]

        if len(batch_suffix_matches) == 1:
            return batch_suffix_matches[0]

    if len(suffix_matches) == 1:
        return suffix_matches[0]

    raise FileNotFoundError(
        "Unable to uniquely resolve image: "
        f"raw={raw_text!r}, "
        f"global_key={global_key!r}, "
        f"exact_matches={len(exact_matches)}, "
        f"suffix_matches={len(suffix_matches)}"
    )


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(
            image.convert("L"),
            dtype=np.uint8,
        )

    mask = array > 0

    if not mask.any():
        raise ValueError(
            f"Empty mask: {path}"
        )

    return mask


def mask_boundary(
    mask: np.ndarray,
) -> np.ndarray:
    eroded = binary_erosion(
        mask,
        structure=np.ones(
            (3, 3),
            dtype=bool,
        ),
        border_value=0,
    )

    boundary = (
        mask
        & ~eroded
    )

    if not boundary.any():
        return mask.copy()

    return boundary


def crop_box_from_masks(
    target: np.ndarray,
    hazard: np.ndarray,
) -> tuple[int, int, int, int]:
    union = target | hazard

    ys, xs = np.nonzero(union)

    if len(xs) == 0:
        raise ValueError(
            "Target/hazard union is empty"
        )

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1

    object_width = x2 - x1
    object_height = y2 - y1

    padding = max(
        24,
        int(
            0.18
            * max(
                object_width,
                object_height,
            )
        ),
    )

    height, width = union.shape

    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(width, x2 + padding),
        min(height, y2 + padding),
    )


def make_overlay(
    image: Image.Image,
    target_mask: np.ndarray,
    hazard_mask: np.ndarray,
    *,
    show_target: bool,
    show_hazard: bool,
    grayscale_context: bool = False,
    alpha: float = 0.42,
) -> Image.Image:
    base = np.asarray(
        image.convert("RGB"),
        dtype=np.uint8,
    ).astype(np.float32)

    if grayscale_context:
        luminance = (
            0.299 * base[:, :, 0]
            + 0.587 * base[:, :, 1]
            + 0.114 * base[:, :, 2]
        )

        base = np.stack(
            [
                luminance,
                luminance,
                luminance,
            ],
            axis=-1,
        )

    if show_target:
        base[target_mask] = (
            (1.0 - alpha)
            * base[target_mask]
            + alpha
            * TARGET_COLOR
        )

    if show_hazard:
        base[hazard_mask] = (
            (1.0 - alpha)
            * base[hazard_mask]
            + alpha
            * HAZARD_COLOR
        )

    target_boundary = mask_boundary(
        target_mask
    )
    hazard_boundary = mask_boundary(
        hazard_mask
    )

    if show_target:
        base[target_boundary] = (
            TARGET_COLOR
        )

    if show_hazard:
        base[hazard_boundary] = (
            HAZARD_COLOR
        )

    return Image.fromarray(
        np.clip(
            base,
            0,
            255,
        ).astype(np.uint8),
        mode="RGB",
    )


def load_font(
    size: int,
) -> ImageFont.ImageFont:
    candidates = [
        Path(
            "/mnt/c/Windows/Fonts/"
            "msyh.ttc"
        ),
        Path(
            "/mnt/c/Windows/Fonts/"
            "msyhbd.ttc"
        ),
        Path(
            "/usr/share/fonts/opentype/"
            "noto/NotoSansCJK-Regular.ttc"
        ),
        Path(
            "/usr/share/fonts/opentype/"
            "noto/NotoSansCJKsc-Regular.otf"
        ),
        Path(
            "/usr/share/fonts/truetype/"
            "dejavu/DejaVuSans.ttf"
        ),
        Path(
            "/usr/share/fonts/truetype/"
            "liberation2/"
            "LiberationSans-Regular.ttf"
        ),
    ]

    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(
                str(path),
                size=size,
            )

    return ImageFont.load_default()


def fit_panel(
    image: Image.Image,
    *,
    width: int,
    height: int,
    label: str,
    font: ImageFont.ImageFont,
) -> Image.Image:
    panel = Image.new(
        "RGB",
        (width, height),
        (245, 245, 245),
    )

    label_height = 42
    available_height = (
        height - label_height
    )

    fitted = ImageOps.contain(
        image,
        (
            width - 10,
            available_height - 10,
        ),
        method=Image.Resampling.LANCZOS,
    )

    x = (
        width - fitted.width
    ) // 2

    y = label_height + (
        available_height
        - fitted.height
    ) // 2

    panel.paste(
        fitted,
        (x, y),
    )

    draw = ImageDraw.Draw(panel)

    draw.rectangle(
        (
            0,
            0,
            width - 1,
            height - 1,
        ),
        outline=(100, 100, 100),
        width=2,
    )

    draw.rectangle(
        (
            0,
            0,
            width,
            label_height,
        ),
        fill=(230, 230, 230),
    )

    draw.text(
        (12, 8),
        label,
        fill=(20, 20, 20),
        font=font,
    )

    return panel


def render_record(
    row: dict[str, Any],
    *,
    image_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    with Image.open(image_path) as raw_image:
        image = ImageOps.exif_transpose(
            raw_image
        ).convert("RGB")

    target_path = resolve_data_path(
        row["target_mask_path"]
    )
    hazard_path = resolve_data_path(
        row["hazard_mask_path"]
    )

    target_mask = load_mask(
        target_path
    )
    hazard_mask = load_mask(
        hazard_path
    )

    image_width, image_height = (
        image.size
    )

    if target_mask.shape != (
        image_height,
        image_width,
    ):
        raise RuntimeError(
            "Target mask/image shape mismatch "
            f"for {row['audit_pair_id']}: "
            f"mask={target_mask.shape}, "
            f"image="
            f"{image_width}x{image_height}"
        )

    if hazard_mask.shape != (
        image_height,
        image_width,
    ):
        raise RuntimeError(
            "Hazard mask/image shape mismatch "
            f"for {row['audit_pair_id']}: "
            f"mask={hazard_mask.shape}, "
            f"image="
            f"{image_width}x{image_height}"
        )

    crop_box = crop_box_from_masks(
        target_mask,
        hazard_mask,
    )

    x1, y1, x2, y2 = crop_box

    full_overlay = make_overlay(
        image,
        target_mask,
        hazard_mask,
        show_target=True,
        show_hazard=True,
    )

    local_image = image.crop(
        crop_box
    )

    local_target = target_mask[
        y1:y2,
        x1:x2,
    ]

    local_hazard = hazard_mask[
        y1:y2,
        x1:x2,
    ]

    local_both = make_overlay(
        local_image,
        local_target,
        local_hazard,
        show_target=True,
        show_hazard=True,
    )

    local_target_panel = make_overlay(
        local_image,
        local_target,
        local_hazard,
        show_target=True,
        show_hazard=False,
        grayscale_context=True,
        alpha=0.52,
    )

    local_hazard_panel = make_overlay(
        local_image,
        local_target,
        local_hazard,
        show_target=False,
        show_hazard=True,
        grayscale_context=True,
        alpha=0.52,
    )

    title_font = load_font(30)
    panel_font = load_font(24)
    legend_font = load_font(22)

    panel_width = 860
    panel_height = 520
    gap = 20
    margin = 20
    title_height = 78
    legend_height = 54

    canvas_width = (
        margin * 2
        + panel_width * 2
        + gap
    )

    canvas_height = (
        title_height
        + panel_height * 2
        + gap
        + legend_height
        + margin
    )

    canvas = Image.new(
        "RGB",
        (
            canvas_width,
            canvas_height,
        ),
        "white",
    )

    draw = ImageDraw.Draw(canvas)

    title = (
        f"{row['audit_pair_id']}   "
        f"目标对象="
        f"{zh_category(row['target_category'])}   "
        f"灾害对象="
        f"{zh_category(row['hazard_category'])}   "
        f"关系族="
        f"{zh_family(row['pair_family'])}"
    )

    draw.text(
        (margin, 20),
        title,
        fill=(15, 15, 15),
        font=title_font,
    )

    panels = [
        fit_panel(
            full_overlay,
            width=panel_width,
            height=panel_height,
            label=(
                "A. 全图："
                "目标对象与灾害对象Mask"
            ),
            font=panel_font,
        ),
        fit_panel(
            local_both,
            width=panel_width,
            height=panel_height,
            label=(
                "B. 局部交互区域"
            ),
            font=panel_font,
        ),
        fit_panel(
            local_target_panel,
            width=panel_width,
            height=panel_height,
            label=(
                "C. 目标对象Mask及上下文"
            ),
            font=panel_font,
        ),
        fit_panel(
            local_hazard_panel,
            width=panel_width,
            height=panel_height,
            label=(
                "D. 灾害对象Mask及上下文"
            ),
            font=panel_font,
        ),
    ]

    positions = [
        (
            margin,
            title_height,
        ),
        (
            margin
            + panel_width
            + gap,
            title_height,
        ),
        (
            margin,
            title_height
            + panel_height
            + gap,
        ),
        (
            margin
            + panel_width
            + gap,
            title_height
            + panel_height
            + gap,
        ),
    ]

    for panel, position in zip(
        panels,
        positions,
    ):
        canvas.paste(
            panel,
            position,
        )

    legend_y = (
        title_height
        + panel_height * 2
        + gap
        + 15
    )

    draw.rectangle(
        (
            margin,
            legend_y + 4,
            margin + 28,
            legend_y + 28,
        ),
        fill=tuple(
            TARGET_COLOR.astype(int)
        ),
    )

    draw.text(
        (
            margin + 40,
            legend_y,
        ),
        (
            f"目标对象："
            f"{zh_category(row['target_category'])}"
        ),
        fill=(20, 20, 20),
        font=legend_font,
    )

    second_x = margin + 340

    draw.rectangle(
        (
            second_x,
            legend_y + 4,
            second_x + 28,
            legend_y + 28,
        ),
        fill=tuple(
            HAZARD_COLOR.astype(int)
        ),
    )

    draw.text(
        (
            second_x + 40,
            legend_y,
        ),
        (
            f"灾害对象："
            f"{zh_category(row['hazard_category'])}"
        ),
        fill=(20, 20, 20),
        font=legend_font,
    )

    draw.text(
        (
            margin + 720,
            legend_y,
        ),
        (
            f"图像编号：{row['image_id']}   "
            f"对象索引："
            f"{row['target_index']} / "
            f"{row['hazard_index']}"
        ),
        fill=(50, 50, 50),
        font=legend_font,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    canvas.save(
        temporary,
        format="PNG",
        optimize=True,
    )

    os.replace(
        temporary,
        output_path,
    )

    intersection = int(
        np.logical_and(
            target_mask,
            hazard_mask,
        ).sum()
    )

    return {
        "audit_pair_id": (
            row["audit_pair_id"]
        ),
        "image_id": int(
            row["image_id"]
        ),
        "global_image_key": (
            row["global_image_key"]
        ),
        "target_index": int(
            row["target_index"]
        ),
        "hazard_index": int(
            row["hazard_index"]
        ),
        "target_category": (
            row["target_category"]
        ),
        "hazard_category": (
            row["hazard_category"]
        ),
        "pair_family": (
            row["pair_family"]
        ),
        "resolved_image_path": str(
            image_path
        ),
        "resolved_target_mask_path": str(
            target_path
        ),
        "resolved_hazard_mask_path": str(
            hazard_path
        ),
        "output_visual_path": str(
            output_path
        ),
        "image_width": image_width,
        "image_height": image_height,
        "crop_x1": x1,
        "crop_y1": y1,
        "crop_x2": x2,
        "crop_y2": y2,
        "target_pixels": int(
            target_mask.sum()
        ),
        "hazard_pixels": int(
            hazard_mask.sum()
        ),
        "intersection_pixels": (
            intersection
        ),
    }


def write_html_index(
    records: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    index_path = (
        output_dir / "index.html"
    )

    cards = []

    for record in records:
        filename = Path(
            record[
                "output_visual_path"
            ]
        ).name

        cards.append(
            f"""
            <article class="card">
              <a href="{html.escape(filename)}">
                <img
                  src="{html.escape(filename)}"
                  loading="lazy"
                  alt="{html.escape(record['audit_pair_id'])}"
                >
              </a>
              <div class="meta">
                <strong>
                  {html.escape(record['audit_pair_id'])}
                </strong><br>
                目标对象={
                    html.escape(
                        zh_category(
                            record['target_category']
                        )
                    )
                }，
                灾害对象={
                    html.escape(
                        zh_category(
                            record['hazard_category']
                        )
                    )
                }<br>
                关系族={
                    html.escape(
                        zh_family(
                            record['pair_family']
                        )
                    )
                }，
                图像编号={record['image_id']}
              </div>
            </article>
            """
        )

    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>FIBE 边界人工审查 V1</title>
<style>
body {{
  font-family: Arial, sans-serif;
  margin: 20px;
  background: #f3f3f3;
}}
h1 {{
  margin-bottom: 6px;
}}
.notice {{
  margin-bottom: 20px;
  color: #444;
}}
.grid {{
  display: grid;
  grid-template-columns:
    repeat(auto-fill, minmax(360px, 1fr));
  gap: 18px;
}}
.card {{
  background: white;
  border: 1px solid #ccc;
  padding: 10px;
  box-shadow: 0 1px 4px rgba(0,0,0,.12);
}}
.card img {{
  width: 100%;
  height: auto;
  display: block;
}}
.meta {{
  padding-top: 8px;
  line-height: 1.45;
}}
</style>
</head>
<body>
<h1>FIBE 边界人工审查 V1</h1>
<div class="notice">
共240个盲审对象对。红色表示目标对象，
蓝色表示灾害对象。本页面不显示正负关系标签
或谓词标签。
</div>
<div class="grid">
{''.join(cards)}
</div>
</body>
</html>
"""

    index_path.write_text(
        document,
        encoding="utf-8",
    )

    return index_path


def main() -> None:
    args = parse_args()

    review_path = (
        args.review.expanduser().resolve()
    )
    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )
    manifest_path = (
        args.manifest
        .expanduser()
        .resolve()
    )
    summary_path = (
        args.summary
        .expanduser()
        .resolve()
    )
    failures_path = (
        args.failures
        .expanduser()
        .resolve()
    )

    if not review_path.is_file():
        raise FileNotFoundError(
            review_path
        )

    review = pd.read_csv(
        review_path,
        encoding="utf-8-sig",
        keep_default_na=False,
    )

    required = {
        "audit_pair_id",
        "image_id",
        "global_image_key",
        "image_path",
        "target_index",
        "hazard_index",
        "target_category",
        "hazard_category",
        "pair_family",
        "target_mask_path",
        "hazard_mask_path",
    }

    missing = required - set(
        review.columns
    )

    if missing:
        raise RuntimeError(
            "Review CSV missing columns: "
            f"{sorted(missing)}"
        )

    if len(review) != 240:
        raise RuntimeError(
            f"Expected 240 rows, found "
            f"{len(review)}"
        )

    if review[
        "audit_pair_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate audit_pair_id values"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    failures_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Building original-image index...",
        flush=True,
    )

    basename_index, all_images = (
        build_image_index(DATA_ROOT)
    )

    print(
        "Indexed source images:",
        len(all_images),
        flush=True,
    )

    records: list[
        dict[str, Any]
    ] = []

    failures: list[
        dict[str, Any]
    ] = []

    reused = 0
    rendered = 0

    for index, row in enumerate(
        review.to_dict(
            orient="records"
        ),
        start=1,
    ):
        audit_pair_id = clean_text(
            row["audit_pair_id"]
        )

        output_path = (
            output_dir
            / f"{audit_pair_id}.png"
        )

        try:
            image_path = resolve_image_path(
                row["image_path"],
                row["global_image_key"],
                basename_index,
                all_images,
            )

            if (
                output_path.is_file()
                and not args.overwrite
            ):
                reused += 1

            record = render_record(
                row,
                image_path=image_path,
                output_path=output_path,
            )

            records.append(record)
            rendered += 1

        except Exception as error:
            failures.append(
                {
                    "audit_pair_id": (
                        audit_pair_id
                    ),
                    "image_id": (
                        row.get(
                            "image_id",
                            "",
                        )
                    ),
                    "global_image_key": (
                        row.get(
                            "global_image_key",
                            "",
                        )
                    ),
                    "failure_type": (
                        type(error).__name__
                    ),
                    "message": str(error),
                }
            )

        if (
            index % 20 == 0
            or index == len(review)
        ):
            print(
                f"processed: "
                f"{index}/{len(review)}",
                flush=True,
            )

    failure_frame = pd.DataFrame(
        failures,
        columns=[
            "audit_pair_id",
            "image_id",
            "global_image_key",
            "failure_type",
            "message",
        ],
    )

    failure_frame.to_csv(
        failures_path,
        sep="\t",
        index=False,
        encoding="utf-8",
    )

    if failures:
        raise SystemExit(
            f"FAIL: {len(failures)} visual "
            f"rendering errors. See "
            f"{failures_path}"
        )

    records.sort(
        key=lambda record: (
            record["audit_pair_id"]
        )
    )

    record_frame = pd.DataFrame(
        records
    )

    record_frame.to_csv(
        manifest_path,
        index=False,
        encoding="utf-8-sig",
    )

    index_path = write_html_index(
        records,
        output_dir,
    )

    output_png_count = len(
        list(
            output_dir.glob(
                "FIBE-AUDIT-*.png"
            )
        )
    )

    if output_png_count != 240:
        raise RuntimeError(
            "Output PNG count mismatch: "
            f"{output_png_count}"
        )

    summary = {
        "version": (
            "FIBE_BOUNDARY_AUDIT_"
            "VISUALS_V1"
        ),
        "review_path": str(
            review_path
        ),
        "output_dir": str(
            output_dir
        ),
        "manifest_path": str(
            manifest_path
        ),
        "html_index": str(
            index_path
        ),
        "input_rows": int(
            len(review)
        ),
        "rendered_rows": int(
            rendered
        ),
        "reused_existing": int(
            reused
        ),
        "failure_count": int(
            len(failures)
        ),
        "output_png_count": int(
            output_png_count
        ),
        "unique_audit_ids": int(
            record_frame[
                "audit_pair_id"
            ].nunique()
        ),
        "unique_images": int(
            record_frame[
                "image_id"
            ].nunique()
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
        "FIBE BOUNDARY AUDIT "
        "VISUALS V1"
    )
    print("=" * 100)
    print(
        "input rows:",
        len(review),
    )
    print(
        "rendered rows:",
        rendered,
    )
    print(
        "output PNGs:",
        output_png_count,
    )
    print(
        "unique images:",
        record_frame[
            "image_id"
        ].nunique(),
    )
    print(
        "failures:",
        len(failures),
    )
    print(
        "visual directory:",
        output_dir,
    )
    print(
        "HTML index:",
        index_path,
    )
    print(
        "visual manifest:",
        manifest_path,
    )
    print(
        "summary:",
        summary_path,
    )

    print(
        "\nFIBE BOUNDARY AUDIT "
        "VISUALS: PASS"
    )


if __name__ == "__main__":
    main()
