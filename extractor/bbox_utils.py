from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Iterable

from PIL import Image


def clamp_bbox(bbox: Iterable[float], img_w: int, img_h: int) -> tuple[int, int, int, int] | None:
    x, y, w, h = bbox
    x0 = max(0.0, x)
    y0 = max(0.0, y)
    x1 = min(float(img_w), x + w)
    y1 = min(float(img_h), y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    return (int(x0), int(y0), int(x1 - x0), int(y1 - y0))


def safe_crop(
    image: Image.Image,
    bbox: Iterable[float],
    *,
    context: str = "",
    log_path: Path | None = None,
) -> Image.Image | None:
    clamped = clamp_bbox(bbox, image.width, image.height)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        original = tuple(bbox)
        if clamped is None or original != clamped:
            timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
            status = "REJECTED" if clamped is None else f"clamped={clamped}"
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"{timestamp} context={context} bbox={original} image={image.width}x{image.height} {status}\n"
                )
    if clamped is None:
        return None
    x0, y0, w, h = clamped
    return image.crop((x0, y0, x0 + w, y0 + h))
