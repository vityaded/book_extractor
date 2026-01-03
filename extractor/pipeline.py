from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import fitz
from PIL import Image

from extractor.bbox_utils import safe_crop
from extractor.debug_utils import draw_overlays, save_layout_json
from extractor.layout import Section, build_sections, union_bbox
from extractor.ocr_utils import ocr_text
from extractor.pdf_utils import PdfDocument, extract_text_blocks, extract_text_lines, render_page

UNIT_RE = re.compile(r"\bUnit\s+(\d{1,3})\b", re.IGNORECASE)
EXERCISES_RE = re.compile(r"\bExercises\b", re.IGNORECASE)
LABEL_RE = re.compile(r"^(\d+\.\d+|\d+)\b")
UNIT_RE_NORM = re.compile(r"\bunit\s+(\d{1,3})\b")
EXERCISES_LABEL_RE = re.compile(r"\b(?:n\.\d+|\d+\.\d+)\b")
UNDERSCORE_RE = re.compile(r"_{3,}")


@dataclass(frozen=True)
class ExtractionConfig:
    pdf_path: Path
    out_dir: Path
    dpi: int
    ocr_enabled: bool
    unit_start: int
    unit_end: int
    debug_enabled: bool
    strict_units: bool = False
    extract_keys: bool = True


def run_extraction(config: ExtractionConfig) -> None:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    (config.out_dir / "assets" / "pages").mkdir(parents=True, exist_ok=True)
    (config.out_dir / "assets" / "rules").mkdir(parents=True, exist_ok=True)
    (config.out_dir / "assets" / "exercises").mkdir(parents=True, exist_ok=True)
    (config.out_dir / "debug").mkdir(parents=True, exist_ok=True)

    doc = PdfDocument(config.pdf_path)
    try:
        units = detect_units(doc, config.unit_start, config.unit_end, config)
        answer_key_pages = detect_answer_key_pages(doc)
        rules_payload = []
        exercises_payload = []
        if config.extract_keys:
            answer_key_payload = extract_answer_keys(doc, answer_key_pages, config)
        else:
            answer_key_payload = []
        answer_key_lookup = {(item["unit_id"], item["label"]): item for item in answer_key_payload}

        for unit in units:
            rule_page = doc.page(unit["rule_page_index"])
            ex_page = doc.page(unit["exercise_page_index"])

            rule_render = render_page(rule_page, config.dpi)
            ex_render = render_page(ex_page, config.dpi)

            unit_id = unit["unit_id"]
            rule_page_image = config.out_dir / "assets" / "pages" / f"unit{unit_id}_rule.png"
            ex_page_image = config.out_dir / "assets" / "pages" / f"unit{unit_id}_ex.png"
            rule_render.image.save(rule_page_image)
            ex_render.image.save(ex_page_image)

            rule_sections = extract_rule_sections(rule_page, rule_render.image, config, unit_id)
            rules_payload.append(
                {
                    "unit_id": unit_id,
                    "unit_title": unit["title"],
                    "rule_page_index": unit["rule_page_index"],
                    "page_image": str(rule_page_image.relative_to(config.out_dir)),
                    "sections": rule_sections,
                }
            )

            exercise_sections = extract_exercise_sections(
                ex_page, ex_render.image, config, unit_id, unit["title"]
            )
            for section in exercise_sections:
                key = (unit_id, section["label"])
                if key in answer_key_lookup:
                    section["answer_key_text"] = answer_key_lookup[key]["answer_text"]
            exercises_payload.append(
                {
                    "unit_id": unit_id,
                    "unit_title": unit["title"],
                    "exercise_page_index": unit["exercise_page_index"],
                    "page_image": str(ex_page_image.relative_to(config.out_dir)),
                    "sections": exercise_sections,
                }
            )

            if config.debug_enabled:
                rule_overlay_path = config.out_dir / "debug" / f"unit{unit_id}_rule_overlay.png"
                ex_overlay_path = config.out_dir / "debug" / f"unit{unit_id}_ex_overlay.png"
                draw_overlays(rule_render.image, rule_sections, rule_overlay_path, "label")
                draw_overlays(ex_render.image, exercise_sections, ex_overlay_path, "label")
                save_layout_json(
                    {"unit_id": unit_id, "sections": rule_sections},
                    config.out_dir / "debug" / f"unit{unit_id}_rule_layout.json",
                )
                save_layout_json(
                    {"unit_id": unit_id, "sections": exercise_sections},
                    config.out_dir / "debug" / f"unit{unit_id}_ex_layout.json",
                )

        manifest = {
            "pdf": config.pdf_path.name,
            "dpi": config.dpi,
            "units": [
                {
                    "unit_id": unit["unit_id"],
                    "title": unit["title"],
                    "rule_page_index": unit["rule_page_index"],
                    "exercise_page_index": unit["exercise_page_index"],
                }
                for unit in units
            ],
            "answer_key_pages": answer_key_pages,
        }

        (config.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (config.out_dir / "rules.json").write_text(json.dumps(rules_payload, indent=2))
        (config.out_dir / "exercises.json").write_text(json.dumps(exercises_payload, indent=2))
        (config.out_dir / "answer_key.json").write_text(json.dumps(answer_key_payload, indent=2))
    finally:
        doc.close()


def detect_units(doc: PdfDocument, start: int, end: int, config: ExtractionConfig) -> list[dict]:
    units: dict[int, dict] = {}
    page_cache: dict[int, dict] = {}
    page_count = len(doc)
    min_chars = 40

    def normalized_page_text(page: fitz.Page) -> dict:
        if page.number in page_cache:
            return page_cache[page.number]
        text_pdf = page.get_text("text") or ""
        text_source = text_pdf
        used_ocr = False
        if len(text_pdf.strip()) < min_chars and config.ocr_enabled:
            rendered = render_page(page, config.dpi)
            text_ocr = ocr_text(rendered.image)
            if text_ocr.strip():
                text_source = text_ocr
                used_ocr = True
        normalized = normalize_text(text_source)
        data = {
            "normalized": normalized,
            "used_ocr": used_ocr,
            "source_text": text_source,
        }
        page_cache[page.number] = data
        return data

    for page in doc.iter_pages():
        page_text = normalized_page_text(page)
        match = UNIT_RE_NORM.search(page_text["normalized"])
        if not match:
            continue
        unit_num = int(match.group(1))
        if unit_num < start or unit_num > end:
            continue
        title = extract_unit_title(page_text["source_text"], unit_num)
        entry = units.setdefault(
            unit_num,
            {
                "unit_id": f"{unit_num:03d}",
                "title": title,
                "rule_page_index": None,
                "exercise_page_index": None,
            },
        )
        if entry["rule_page_index"] is None:
            entry["rule_page_index"] = page.number
        if entry["title"].startswith("Unit") and title:
            entry["title"] = title

    results = []
    for unit_num in range(start, end + 1):
        entry = units.setdefault(
            unit_num,
            {
                "unit_id": f"{unit_num:03d}",
                "title": f"Unit {unit_num}",
                "rule_page_index": None,
                "exercise_page_index": None,
            },
        )
        rule_idx = entry["rule_page_index"]
        exercise_idx = None
        if rule_idx is not None:
            candidate_indices = [
                idx
                for idx in range(rule_idx + 1, min(rule_idx + 4, page_count))
            ]
            for idx in candidate_indices:
                page_text = normalized_page_text(doc.page(idx))
                if EXERCISES_RE.search(page_text["normalized"]):
                    exercise_idx = idx
                    break
            if exercise_idx is None:
                for idx in candidate_indices:
                    page_text = normalized_page_text(doc.page(idx))
                    if EXERCISES_LABEL_RE.search(page_text["normalized"]) or UNDERSCORE_RE.search(
                        page_text["normalized"]
                    ):
                        exercise_idx = idx
                        break
        entry["exercise_page_index"] = exercise_idx
        if rule_idx is None or exercise_idx is None:
            log_missing_unit(unit_num, rule_idx, config, doc, normalized_page_text)
            if config.strict_units:
                raise ValueError(f"Unit {unit_num} missing rule or exercise page.")
            logging.warning(
                "Unit %s missing rule or exercise page (rule=%s, ex=%s)",
                unit_num,
                rule_idx,
                exercise_idx,
            )
            continue
        results.append(entry)

    return results


def normalize_text(text: str) -> str:
    normalized = text.lower()
    normalized = normalized.replace("unlt", "unit")
    normalized = normalized.replace("exerclses", "exercises")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def log_missing_unit(
    unit_num: int,
    rule_idx: int | None,
    config: ExtractionConfig,
    doc: PdfDocument,
    page_text_fn,
) -> None:
    debug_dir = config.out_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    if rule_idx is not None:
        unit_re = re.compile(rf"\\bunit\\s+{unit_num}\\b")
        for idx in range(rule_idx, min(rule_idx + 4, len(doc))):
            page_text = page_text_fn(doc.page(idx))
            normalized = page_text["normalized"]
            candidates.append(
                {
                    "page_index": idx,
                    "unit_match": bool(unit_re.search(normalized)),
                    "exercise_match": bool(EXERCISES_RE.search(normalized)),
                    "text_source": "ocr" if page_text["used_ocr"] else "pdf",
                    "text_preview": normalized[:300],
                }
            )
    payload = {
        "unit": unit_num,
        "rule_page_index": rule_idx,
        "candidates": candidates,
    }
    out_path = debug_dir / f"unit{unit_num:03d}_missing.json"
    out_path.write_text(json.dumps(payload, indent=2))


def extract_unit_title(text: str, unit_num: int) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines:
        if line.lower().startswith("unit") and str(unit_num) in line:
            parts = line.split(" ", 2)
            if len(parts) >= 3:
                return parts[2].strip()
    return f"Unit {unit_num}"


def detect_answer_key_pages(doc: PdfDocument) -> list[int]:
    pages = []
    for page in doc.iter_pages():
        text = page.get_text("text")
        if re.search(r"\bAnswer\s+Key\b", text, re.IGNORECASE) or re.search(
            r"\bAnswers\b", text, re.IGNORECASE
        ):
            pages.append(page.number)
    return pages


def scale_bbox(bbox: tuple[float, float, float, float], scale: float) -> list[int]:
    x0, y0, x1, y1 = bbox
    return [int(x0 * scale), int(y0 * scale), int((x1 - x0) * scale), int((y1 - y0) * scale)]


def extract_rule_sections(
    page: fitz.Page, image: Image.Image, config: ExtractionConfig, unit_id: str
) -> list[dict]:
    blocks = [block for block in extract_text_blocks(page) if block.text.strip()]
    sections = build_sections(blocks, page.rect.height)
    if not sections:
        raise ValueError(f"No rule sections detected for unit {unit_id}")
    scale = config.dpi / 72.0
    debug_log = config.out_dir / "debug" / "bbox_warnings.log"
    output_sections: list[dict] = []
    for idx, section in enumerate(sections):
        label = chr(ord("A") + idx)
        bbox_px = scale_bbox(section.bbox, scale)
        crop_path = config.out_dir / "assets" / "rules" / f"unit{unit_id}_{label}.png"
        crop = safe_crop(
            image,
            bbox_px,
            context=f"rules/unit{unit_id}/{label}",
            log_path=debug_log,
        )
        if crop is None:
            continue
        crop.save(crop_path)
        text_pdf = section.text.strip()
        text_ocr = ocr_text(crop) if config.ocr_enabled else ""
        text_best = text_pdf or text_ocr
        if not text_best:
            raise ValueError(f"Empty text for rule section unit {unit_id} {label}")
        split_reason = section.split_reason
        output_sections.append(
            {
                "rule_section_id": f"unit{unit_id}_{label}",
                "label": label,
                "bbox_px": bbox_px,
                "crop_image": str(crop_path.relative_to(config.out_dir)),
                "text_pdf": text_pdf,
                "text_ocr": text_ocr,
                "text_best": text_best,
                "split_reason": split_reason,
                "candidate_exercise_section_ids": [],
            }
        )
    return output_sections


def extract_exercise_sections(
    page: fitz.Page,
    image: Image.Image,
    config: ExtractionConfig,
    unit_id: str,
    unit_title: str,
) -> list[dict]:
    lines = extract_text_lines(page)
    labels = []
    for line in lines:
        match = LABEL_RE.match(line["text"])
        if match:
            labels.append({"label": match.group(1), "bbox": line["bbox"]})
    if not labels:
        raise ValueError(f"No exercise sections detected for unit {unit_id}")
    labels = sorted(labels, key=lambda item: item["bbox"][1])
    content_blocks = [block for block in extract_text_blocks(page) if block.text.strip()]
    content_bbox = union_bbox(content_blocks)
    scale = config.dpi / 72.0
    debug_log = config.out_dir / "debug" / "bbox_warnings.log"
    sections: list[dict] = []
    for idx, label_info in enumerate(labels):
        start_y = label_info["bbox"][1]
        end_y = labels[idx + 1]["bbox"][1] if idx + 1 < len(labels) else content_bbox[3]
        section_blocks = [
            block
            for block in content_blocks
            if block.y0 >= start_y and block.y1 <= end_y + 1
        ]
        bbox = union_bbox(section_blocks) if section_blocks else (content_bbox[0], start_y, content_bbox[2], end_y)
        bbox_px = scale_bbox(bbox, scale)
        label = label_info["label"]
        label_id = label.replace(".", "_")
        crop_path = config.out_dir / "assets" / "exercises" / f"unit{unit_id}_{label_id}.png"
        crop = safe_crop(
            image,
            bbox_px,
            context=f"exercises/unit{unit_id}/{label_id}",
            log_path=debug_log,
        )
        if crop is None:
            continue
        crop.save(crop_path)
        text_pdf = "\n".join(block.text.strip() for block in section_blocks if block.text.strip())
        text_ocr = ocr_text(crop) if config.ocr_enabled else ""
        text_best = text_pdf or text_ocr
        if not text_best:
            raise ValueError(f"Empty text for exercise section unit {unit_id} {label}")
        boxed_regions = detect_boxed_regions(
            page=page,
            section_bbox=bbox,
            section_label=f"unit{unit_id}_{label_id}",
            image=image,
            scale=scale,
            config=config,
        )
        sections.append(
            {
                "exercise_section_id": f"unit{unit_id}_{label_id}",
                "label": label,
                "bbox_px": bbox_px,
                "crop_image": str(crop_path.relative_to(config.out_dir)),
                "text_pdf": text_pdf,
                "text_ocr": text_ocr,
                "text_best": text_best,
                "boxed_regions": boxed_regions,
                "answer_key_ref": {"unit_id": unit_id, "label": label},
                "answer_key_text": None,
                "candidate_rule_section_ids": [],
            }
        )
    return sections


def detect_boxed_regions(
    *,
    page: fitz.Page,
    section_bbox: tuple[float, float, float, float],
    section_label: str,
    image: Image.Image,
    scale: float,
    config: ExtractionConfig,
) -> list[dict]:
    sx0, sy0, sx1, sy1 = section_bbox
    section_area = max((sx1 - sx0) * (sy1 - sy0), 1.0)
    section_width = sx1 - sx0
    section_height = sy1 - sy0
    regions: list[dict] = []
    index = 1
    debug_log = config.out_dir / "debug" / "bbox_warnings.log"

    drawings = page.get_drawings()
    for drawing in drawings:
        rect = drawing.get("rect")
        if not rect:
            continue
        rx0, ry0, rx1, ry1 = rect
        if rx1 < sx0 or rx0 > sx1 or ry1 < sy0 or ry0 > sy1:
            continue
        area = (rx1 - rx0) * (ry1 - ry0)
        width = rx1 - rx0
        height = ry1 - ry0
        text = page.get_textbox(fitz.Rect(rx0, ry0, rx1, ry1))
        if not region_passes_threshold(area, width, height, section_area, section_width, text):
            continue
        kind = "boxed"
        if len(drawing.get("items", [])) >= 8:
            kind = "table"
        bbox_px = scale_bbox((rx0, ry0, rx1, ry1), scale)
        crop_path = (
            config.out_dir
            / "assets"
            / "exercises"
            / f"{section_label}_box{index:02d}.png"
        )
        crop = safe_crop(
            image,
            bbox_px,
            context=f"{section_label}/box{index:02d}",
            log_path=debug_log,
        )
        if crop is None:
            continue
        crop.save(crop_path)
        regions.append(
            {
                "boxed_region_id": f"{section_label}_box{index:02d}",
                "bbox_px": bbox_px,
                "crop_image": str(crop_path.relative_to(config.out_dir)),
                "kind": kind,
            }
        )
        index += 1

    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 1:
            continue
        bbox = block.get("bbox", [0, 0, 0, 0])
        rx0, ry0, rx1, ry1 = bbox
        if rx1 < sx0 or rx0 > sx1 or ry1 < sy0 or ry0 > sy1:
            continue
        area = (rx1 - rx0) * (ry1 - ry0)
        width = rx1 - rx0
        height = ry1 - ry0
        if not region_passes_threshold(area, width, height, section_area, section_width, ""):
            continue
        bbox_px = scale_bbox((rx0, ry0, rx1, ry1), scale)
        crop_path = (
            config.out_dir
            / "assets"
            / "exercises"
            / f"{section_label}_box{index:02d}.png"
        )
        crop = safe_crop(
            image,
            bbox_px,
            context=f"{section_label}/box{index:02d}",
            log_path=debug_log,
        )
        if crop is None:
            continue
        crop.save(crop_path)
        regions.append(
            {
                "boxed_region_id": f"{section_label}_box{index:02d}",
                "bbox_px": bbox_px,
                "crop_image": str(crop_path.relative_to(config.out_dir)),
                "kind": "image",
            }
        )
        index += 1
    return regions


def region_passes_threshold(
    area: float,
    width: float,
    height: float,
    section_area: float,
    section_width: float,
    text: str,
) -> bool:
    if area >= 0.03 * section_area:
        return True
    if width >= 0.25 * section_width and height >= 0.12 * (section_area / section_width):
        return True
    if len(text.strip()) >= 15:
        return True
    return False


def extract_answer_keys(
    doc: PdfDocument, answer_key_pages: Iterable[int], config: ExtractionConfig
) -> list[dict]:
    results: list[dict] = []
    for page_index in answer_key_pages:
        page = doc.page(page_index)
        lines = extract_text_lines(page)
        labels = []
        for line in lines:
            match = LABEL_RE.match(line["text"])
            if match:
                labels.append({"label": match.group(1), "bbox": line["bbox"]})
        labels = sorted(labels, key=lambda item: item["bbox"][1])
        if not labels:
            continue
        blocks = extract_text_blocks(page)
        content_bbox = union_bbox(blocks)
        debug_log = config.out_dir / "debug" / "bbox_warnings.log"
        rendered_page = render_page(page, config.dpi)
        for idx, label_info in enumerate(labels):
            start_y = label_info["bbox"][1]
            end_y = labels[idx + 1]["bbox"][1] if idx + 1 < len(labels) else content_bbox[3]
            section_blocks = [
                block
                for block in blocks
                if block.y0 >= start_y and block.y1 <= end_y + 1
            ]
            bbox = union_bbox(section_blocks) if section_blocks else (content_bbox[0], start_y, content_bbox[2], end_y)
            unit_id = derive_unit_id(page, label_info["label"])
            label = label_info["label"]
            scale = config.dpi / 72.0
            bbox_px = scale_bbox(bbox, scale)
            text_pdf = "\n".join(block.text.strip() for block in section_blocks if block.text.strip())
            crop = safe_crop(
                rendered_page.image,
                bbox_px,
                context=f"answer_keys/page{page_index}/{label}",
                log_path=debug_log,
            )
            if crop is None:
                continue
            text_ocr = ocr_text(crop) if config.ocr_enabled else ""
            text_best = text_pdf or text_ocr
            results.append(
                {
                    "unit_id": unit_id,
                    "label": label,
                    "answer_text": text_best,
                    "source_page_index": page_index,
                    "source_bbox_px": bbox_px,
                    "text_pdf": text_pdf,
                    "text_ocr": text_ocr,
                    "text_best": text_best,
                }
            )
    return results


def derive_unit_id(page: fitz.Page, label: str) -> str:
    text = page.get_text("text")
    match = UNIT_RE.search(text)
    if match:
        unit_num = int(match.group(1))
        return f"{unit_num:03d}"
    match = re.search(r"\bUnit\s+(\d{1,3})\b", label)
    if match:
        return f"{int(match.group(1)):03d}"
    return "000"
