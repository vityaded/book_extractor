from __future__ import annotations

from pathlib import Path

import pytest

from extractor.pipeline import (
    ExtractionConfig,
    detect_answer_key_pages,
    detect_units,
    extract_answer_keys,
    run_extraction,
)
from extractor.validation import validate_bboxes_in_bounds, validate_referenced_images, validate_schema_files

PDF_PATH = Path("/mnt/data/English Grammar in Use 5th Edition - 2019.pdf")


@pytest.mark.skipif(not PDF_PATH.exists(), reason="PDF not available")
def test_detect_units_1_3():
    config = ExtractionConfig(
        pdf_path=PDF_PATH,
        out_dir=Path("/tmp/unused"),
        dpi=72,
        ocr_enabled=False,
        unit_start=1,
        unit_end=3,
        debug_enabled=False,
    )
    from extractor.pdf_utils import PdfDocument

    doc = PdfDocument(config.pdf_path)
    try:
        units = detect_units(doc, config.unit_start, config.unit_end)
    finally:
        doc.close()
    assert len(units) == 3
    assert units[0]["unit_id"] == "001"


@pytest.mark.skipif(not PDF_PATH.exists(), reason="PDF not available")
def test_rule_and_exercise_sections_unit1(tmp_path):
    config = ExtractionConfig(
        pdf_path=PDF_PATH,
        out_dir=tmp_path,
        dpi=72,
        ocr_enabled=False,
        unit_start=1,
        unit_end=1,
        debug_enabled=True,
    )
    run_extraction(config)
    rules = (tmp_path / "rules.json").read_text()
    exercises = (tmp_path / "exercises.json").read_text()
    assert "unit001" in rules
    assert "unit001" in exercises


@pytest.mark.skipif(not PDF_PATH.exists(), reason="PDF not available")
def test_answer_key_contains_unit1():
    from extractor.pdf_utils import PdfDocument

    doc = PdfDocument(PDF_PATH)
    try:
        pages = detect_answer_key_pages(doc)
        config = ExtractionConfig(
            pdf_path=PDF_PATH,
            out_dir=Path("/tmp/unused"),
            dpi=72,
            ocr_enabled=False,
            unit_start=1,
            unit_end=1,
            debug_enabled=False,
        )
        keys = extract_answer_keys(doc, pages, config)
    finally:
        doc.close()
    assert any(item["unit_id"] == "001" for item in keys)


@pytest.mark.skipif(not PDF_PATH.exists(), reason="PDF not available")
def test_golden_units_1_2(tmp_path):
    config = ExtractionConfig(
        pdf_path=PDF_PATH,
        out_dir=tmp_path,
        dpi=72,
        ocr_enabled=False,
        unit_start=1,
        unit_end=2,
        debug_enabled=True,
    )
    run_extraction(config)
    validate_schema_files(tmp_path)
    validate_referenced_images(tmp_path)
    validate_bboxes_in_bounds(tmp_path)
