from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import fitz
from PIL import Image


@dataclass(frozen=True)
class PageTextBlock:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    block_type: int


@dataclass(frozen=True)
class RenderedPage:
    index: int
    width: int
    height: int
    image: Image.Image


class PdfDocument:
    def __init__(self, path: Path):
        self.path = path
        self._doc = fitz.open(path)

    def __len__(self) -> int:
        return self._doc.page_count

    def page(self, index: int) -> fitz.Page:
        return self._doc.load_page(index)

    def close(self) -> None:
        self._doc.close()

    def iter_pages(self) -> Iterable[fitz.Page]:
        for page_index in range(self._doc.page_count):
            yield self._doc.load_page(page_index)



def render_page(page: fitz.Page, dpi: int) -> RenderedPage:
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return RenderedPage(index=page.number, width=pix.width, height=pix.height, image=image)


def extract_text_blocks(page: fitz.Page) -> list[PageTextBlock]:
    blocks = []
    for block in page.get_text("blocks"):
        if len(block) < 6:
            continue
        x0, y0, x1, y1, text, block_type = block[:6]
        blocks.append(PageTextBlock(x0=x0, y0=y0, x1=x1, y1=y1, text=text, block_type=block_type))
    return blocks


def extract_text_lines(page: fitz.Page) -> list[dict]:
    data = page.get_text("dict")
    lines: list[dict] = []
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            line_text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
            if not line_text:
                continue
            bbox = line.get("bbox", [0, 0, 0, 0])
            lines.append({"text": line_text, "bbox": bbox})
    return lines
