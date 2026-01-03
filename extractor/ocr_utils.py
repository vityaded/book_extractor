from __future__ import annotations

from PIL import Image
import pytesseract


def ocr_text(image: Image.Image, lang: str = "eng") -> str:
    return pytesseract.image_to_string(image, lang=lang).strip()
