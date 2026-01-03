from __future__ import annotations

from pathlib import Path

import pytest

from extractor.pipeline import (
    ExtractionConfig,
    detect_exercise_labels,
    select_exercise_text,
)


def test_detect_exercise_labels_prefers_decimal():
    lines = [
        {"text": "1.4 Something", "bbox": [10, 100, 80, 120]},
        {"text": "4 Something else", "bbox": [300, 200, 340, 220]},
        {"text": "2 Practice", "bbox": [12, 300, 60, 320]},
    ]
    labels = detect_exercise_labels(lines, page_width=600)
    label_values = [label["label"] for label in labels]
    assert "1.4" in label_values
    assert "2" in label_values
    assert "4" not in label_values


def test_select_exercise_text_allows_empty_when_not_strict(tmp_path):
    config = ExtractionConfig(
        pdf_path=Path("dummy.pdf"),
        out_dir=tmp_path,
        dpi=72,
        ocr_enabled=True,
        unit_start=1,
        unit_end=1,
        debug_enabled=True,
        strict_units=False,
        strict_text=False,
        extract_keys=True,
    )
    crop_path = tmp_path / "assets" / "exercises" / "unit001_1_4.png"
    text_best = select_exercise_text(
        text_pdf="",
        text_ocr="",
        config=config,
        unit_id="001",
        label="1.4",
        exercise_section_id="unit001_1_4",
        bbox_px=(0, 0, 10, 10),
        crop_path=crop_path,
    )
    assert text_best == ""
    assert (tmp_path / "debug" / "empty_text_sections.jsonl").exists()


def test_select_exercise_text_strict_raises(tmp_path):
    config = ExtractionConfig(
        pdf_path=Path("dummy.pdf"),
        out_dir=tmp_path,
        dpi=72,
        ocr_enabled=True,
        unit_start=1,
        unit_end=1,
        debug_enabled=True,
        strict_units=False,
        strict_text=True,
        extract_keys=True,
    )
    crop_path = tmp_path / "assets" / "exercises" / "unit001_1_4.png"
    with pytest.raises(ValueError):
        select_exercise_text(
            text_pdf="",
            text_ocr="",
            config=config,
            unit_id="001",
            label="1.4",
            exercise_section_id="unit001_1_4",
            bbox_px=(0, 0, 10, 10),
            crop_path=crop_path,
        )
