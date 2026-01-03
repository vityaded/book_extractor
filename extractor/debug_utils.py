from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


def draw_overlays(
    image: Image.Image,
    boxes: Iterable[dict],
    output_path: Path,
    label_key: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None
    for entry in boxes:
        x, y, w, h = entry["bbox_px"]
        draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
        label = entry.get(label_key, "")
        if label:
            draw.text((x + 4, y + 4), label, fill="red", font=font)
    overlay.save(output_path)


def save_layout_json(data: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
