from __future__ import annotations

import logging

from PIL import Image
import pytesseract


def ocr_text(image: Image.Image | None, lang: str = "eng") -> str:
    if image is None:
        return ""
    if image.width < 2 or image.height < 2:
        return ""
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    image = image.copy()
    try:
        return pytesseract.image_to_string(image, lang=lang).strip()
    except Exception:  # noqa: BLE001
        logging.exception("OCR failed")
        return ""
