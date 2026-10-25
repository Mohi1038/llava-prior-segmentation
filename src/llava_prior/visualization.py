"""Create a polished, high-contrast visual summary of saved predictions."""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

_COLORS = (
    (255, 91, 105),
    (43, 213, 255),
    (255, 197, 61),
    (77, 231, 155),
    (197, 126, 255),
    (255, 145, 64),
    (100, 155, 255),
    (255, 118, 207),
)
_BACKGROUND = (15, 23, 42)
_CARD = (25, 36, 59)
_TEXT = (245, 248, 255)
_MUTED = (165, 180, 205)
_PANEL_WIDTH = 960
_HEADER_HEIGHT = 90
_PANEL_TITLE_HEIGHT = 58
_PANEL_FOOTER_HEIGHT = 48
_PANEL_GAP = 24
_OUTER_PADDING = 32
_OVERLAY_ALPHA = 0.52


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        (r"C:\Windows\Fonts\segoeuib.ttf", r"C:\Windows\Fonts\arialbd.ttf")
        if bold
        else (r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf")
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _overlay(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    source = np.asarray(image, dtype=np.float32)
    colored = source.copy()
    selected = mask.astype(bool)
    tint = np.asarray(color, dtype=np.float32)
    colored[selected] = source[selected] * (1 - _OVERLAY_ALPHA) + tint * _OVERLAY_ALPHA
    result = Image.fromarray(np.clip(colored, 0, 255).astype(np.uint8))

    binary = Image.fromarray(selected.astype(np.uint8) * 255)
    interior = np.asarray(binary.filter(ImageFilter.MinFilter(3))) > 0
    outline = selected & ~interior
    result_array = np.asarray(result).copy()
    result_array[outline] = (255, 255, 255)
    return Image.fromarray(result_array)


def _panel(
    title: str,
    detail: str,
    image: Image.Image,
    color: tuple[int, int, int] | None = None,
) -> tuple[str, str, Image.Image, tuple[int, int, int] | None]:
    return title, detail, image, color


def save_prediction_visualization(image_path: str | Path, output_dir: str | Path) -> Path:
    """Save a high-resolution contact sheet with readable, distinct class overlays."""
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    image = Image.open(image_path).convert("RGB")
    metadata = json.loads((output_dir / "predictions.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, list):
        raise TypeError("predictions.json must contain a list of predictions")

    panel_height = max(1, round(image.height * _PANEL_WIDTH / image.width))
    resized = image.resize((_PANEL_WIDTH, panel_height), Image.Resampling.LANCZOS)
    panels = [
        _panel("Original image", "Unmodified input", resized),
    ]
    masks: list[np.ndarray] = []
    labels: list[tuple[str, tuple[int, int, int], np.ndarray]] = []
    for index, prediction in enumerate(metadata):
        class_name = prediction["class"]
        mask_path = output_dir / Path(prediction["mask"]).name
        mask = np.load(mask_path).astype(bool)
        if mask.shape != (image.height, image.width):
            raise ValueError(f"mask for {class_name!r} does not match the input image dimensions")
        masks.append(mask)
        color = _COLORS[index % len(_COLORS)]
        labels.append((class_name, color, mask))
        coverage = float(mask.mean() * 100)
        overlay = _overlay(resized, np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (_PANEL_WIDTH, panel_height), Image.Resampling.NEAREST
            )
        ) > 0, color)
        panels.append(
            _panel(
                class_name,
                f"{int(mask.sum()):,} pixels  |  {coverage:.1f}% of image",
                overlay,
                color,
            )
        )

    semantic = np.load(output_dir / "semantic_mask.npy")
    if semantic.shape != (image.height, image.width):
        raise ValueError("semantic mask does not match the input image dimensions")
    semantic_small = np.asarray(
        Image.fromarray(semantic.astype(np.int32), mode="I").resize(
            (_PANEL_WIDTH, panel_height), Image.Resampling.NEAREST
        )
    )
    combined = np.asarray(resized).copy()
    for index, (_, color, _) in enumerate(labels, start=1):
        selected = semantic_small == index
        tint = np.asarray(color, dtype=np.float32)
        combined[selected] = (
            combined[selected].astype(np.float32) * (1 - _OVERLAY_ALPHA)
            + tint * _OVERLAY_ALPHA
        ).astype(np.uint8)
        boundary = selected & ~np.asarray(
            Image.fromarray(selected.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(3))
        ).astype(bool)
        combined[boundary] = (255, 255, 255)
    panels.insert(
        1,
        _panel(
            "Combined prediction",
            "Bright outlines separate predicted regions",
            Image.fromarray(combined),
        ),
    )

    columns = 2
    rows = math.ceil(len(panels) / columns)
    cell_height = _PANEL_TITLE_HEIGHT + panel_height + _PANEL_FOOTER_HEIGHT
    cell_width = _PANEL_WIDTH
    sheet_width = _OUTER_PADDING * 2 + columns * cell_width + (columns - 1) * _PANEL_GAP
    sheet_height = (
        _OUTER_PADDING * 2
        + _HEADER_HEIGHT
        + rows * cell_height
        + (rows - 1) * _PANEL_GAP
    )
    sheet = Image.new("RGB", (sheet_width, sheet_height), _BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    title_font = _font(30, bold=True)
    subtitle_font = _font(17)
    panel_font = _font(22, bold=True)
    detail_font = _font(16)
    draw.text(
        (_OUTER_PADDING, _OUTER_PADDING),
        "Segmentation prediction",
        fill=_TEXT,
        font=title_font,
    )
    draw.text(
        (_OUTER_PADDING, _OUTER_PADDING + 44),
        f"{image_path.name}   |   {image.width} × {image.height} px",
        fill=_MUTED,
        font=subtitle_font,
    )

    for index, (title, detail, panel_image, color) in enumerate(panels):
        column, row = index % columns, index // columns
        x = _OUTER_PADDING + column * (cell_width + _PANEL_GAP)
        y = _OUTER_PADDING + _HEADER_HEIGHT + row * (cell_height + _PANEL_GAP)
        draw.rounded_rectangle(
            (x, y, x + cell_width, y + cell_height),
            radius=12,
            fill=_CARD,
        )
        marker_color = color or (119, 147, 190)
        draw.rounded_rectangle(
            (x + 16, y + 17, x + 22, y + 41),
            radius=3,
            fill=marker_color,
        )
        draw.text((x + 34, y + 14), title, fill=_TEXT, font=panel_font)
        panel_image = panel_image.resize((_PANEL_WIDTH, panel_height), Image.Resampling.LANCZOS)
        sheet.paste(panel_image, (x, y + _PANEL_TITLE_HEIGHT))
        draw.text(
            (x + 16, y + _PANEL_TITLE_HEIGHT + panel_height + 13),
            detail,
            fill=_MUTED,
            font=detail_font,
        )

    if labels:
        legend_y = sheet_height - _OUTER_PADDING + 4
        legend_x = _OUTER_PADDING
        for name, color, _ in labels:
            draw.ellipse((legend_x, legend_y, legend_x + 13, legend_y + 13), fill=color)
            draw.text((legend_x + 20, legend_y - 4), name, fill=_TEXT, font=detail_font)
            legend_x += 42 + int(draw.textlength(name, font=detail_font))

    result_path = output_dir / "visualization.png"
    sheet.save(result_path, optimize=True)
    return result_path
