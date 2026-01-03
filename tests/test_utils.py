from __future__ import annotations

from PIL import Image

from extractor.bbox_utils import clamp_bbox
from extractor.ocr_utils import ocr_text


def test_clamp_bbox():
    assert clamp_bbox((-5, -5, 10, 10), 100, 100) == (0, 0, 5, 5)
    assert clamp_bbox((90, 90, 20, 20), 100, 100) == (90, 90, 10, 10)
    assert clamp_bbox((10, 10, 0, 5), 100, 100) is None
    assert clamp_bbox((10, 10, -5, 5), 100, 100) is None
    assert clamp_bbox((200, 200, 10, 10), 100, 100) is None


def test_ocr_text_handles_none_and_tiny():
    assert ocr_text(None) == ""
    assert ocr_text(Image.new("RGB", (1, 1))) == ""
