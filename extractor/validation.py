from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def load_json(path: Path):
    return json.loads(path.read_text())


def validate_schema_files(out_dir: Path) -> None:
    required = ["manifest.json", "rules.json", "exercises.json", "answer_key.json"]
    for filename in required:
        path = out_dir / filename
        if not path.exists():
            raise AssertionError(f"Missing {filename}")
        json.loads(path.read_text())


def validate_referenced_images(out_dir: Path) -> None:
    manifest = load_json(out_dir / "manifest.json")
    rules = load_json(out_dir / "rules.json")
    exercises = load_json(out_dir / "exercises.json")
    referenced = []
    for unit in rules:
        referenced.append(unit["page_image"])
        for section in unit["sections"]:
            referenced.append(section["crop_image"])
    for unit in exercises:
        referenced.append(unit["page_image"])
        for section in unit["sections"]:
            referenced.append(section["crop_image"])
            for region in section.get("boxed_regions", []):
                referenced.append(region["crop_image"])
    for rel_path in referenced:
        path = out_dir / rel_path
        if not path.exists():
            raise AssertionError(f"Missing image {path}")

    for unit in manifest.get("units", []):
        for suffix in ("rule", "ex"):
            overlay = out_dir / "debug" / f"unit{unit['unit_id']}_{suffix}_overlay.png"
            if not overlay.exists():
                raise AssertionError(f"Missing debug overlay {overlay}")


def validate_bboxes_in_bounds(out_dir: Path) -> None:
    rules = load_json(out_dir / "rules.json")
    exercises = load_json(out_dir / "exercises.json")
    for unit in rules:
        page_image = out_dir / unit["page_image"]
        if not page_image.exists():
            continue
        width, height = _get_image_size(page_image)
        for section in unit["sections"]:
            _assert_bbox_inside(section["bbox_px"], width, height)
    for unit in exercises:
        page_image = out_dir / unit["page_image"]
        if not page_image.exists():
            continue
        width, height = _get_image_size(page_image)
        for section in unit["sections"]:
            _assert_bbox_inside(section["bbox_px"], width, height)
            for region in section.get("boxed_regions", []):
                _assert_bbox_inside(region["bbox_px"], width, height)


def _get_image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as img:
        return img.size


def _assert_bbox_inside(bbox_px: Iterable[int], width: int, height: int) -> None:
    x, y, w, h = bbox_px
    if x < 0 or y < 0 or x + w > width or y + h > height:
        raise AssertionError(f"BBox {bbox_px} outside image bounds {width}x{height}")
