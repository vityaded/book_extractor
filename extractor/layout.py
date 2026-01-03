from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from extractor.pdf_utils import PageTextBlock


@dataclass
class Section:
    bbox: tuple[float, float, float, float]
    blocks: list[PageTextBlock]
    text: str
    split_reason: str


def union_bbox(blocks: Iterable[PageTextBlock]) -> tuple[float, float, float, float]:
    xs0, ys0, xs1, ys1 = [], [], [], []
    for block in blocks:
        xs0.append(block.x0)
        ys0.append(block.y0)
        xs1.append(block.x1)
        ys1.append(block.y1)
    if not xs0:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(xs0), min(ys0), max(xs1), max(ys1))


def group_blocks_by_gaps(blocks: list[PageTextBlock], gap_ratio: float, page_height: float) -> list[list[PageTextBlock]]:
    if not blocks:
        return []
    sorted_blocks = sorted(blocks, key=lambda b: (b.y0, b.x0))
    groups = [[sorted_blocks[0]]]
    for block in sorted_blocks[1:]:
        prev = groups[-1][-1]
        gap = block.y0 - prev.y1
        if gap > gap_ratio * page_height:
            groups.append([block])
        else:
            groups[-1].append(block)
    return groups


def needs_split(section: Section, content_height: float) -> bool:
    x0, y0, x1, y1 = section.bbox
    height = y1 - y0
    return height > 0.42 * content_height or len(section.text) > 1200 or len(section.blocks) > 8


def split_section(section: Section, content_height: float) -> list[Section]:
    if not needs_split(section, content_height):
        return [section]
    blocks = section.blocks
    if len(blocks) <= 1:
        return [section]
    sorted_blocks = sorted(blocks, key=lambda b: (b.y0, b.x0))
    # split at largest gap
    gaps = []
    for idx in range(1, len(sorted_blocks)):
        gap = sorted_blocks[idx].y0 - sorted_blocks[idx - 1].y1
        gaps.append((gap, idx))
    if not gaps:
        return [section]
    _, split_idx = max(gaps, key=lambda item: item[0])
    first_blocks = sorted_blocks[:split_idx]
    second_blocks = sorted_blocks[split_idx:]
    sections = []
    for blocks_subset in (first_blocks, second_blocks):
        bbox = union_bbox(blocks_subset)
        text = "\n".join(block.text.strip() for block in blocks_subset if block.text.strip())
        sections.append(Section(bbox=bbox, blocks=blocks_subset, text=text, split_reason="oversize_subsplit"))
    # Recursively ensure size constraints
    final_sections: list[Section] = []
    for section_item in sections:
        if needs_split(section_item, content_height):
            final_sections.extend(split_section(section_item, content_height))
        else:
            final_sections.append(section_item)
    return final_sections


def build_sections(blocks: list[PageTextBlock], page_height: float) -> list[Section]:
    groups = group_blocks_by_gaps(blocks, gap_ratio=0.03, page_height=page_height)
    sections = []
    for group in groups:
        text = "\n".join(block.text.strip() for block in group if block.text.strip())
        bbox = union_bbox(group)
        sections.append(Section(bbox=bbox, blocks=group, text=text, split_reason="layout_blocks"))
    if not sections:
        return []
    content_y0 = min(section.bbox[1] for section in sections)
    content_y1 = max(section.bbox[3] for section in sections)
    content_height = max(content_y1 - content_y0, 1)
    final_sections: list[Section] = []
    for section in sections:
        if needs_split(section, content_height):
            final_sections.extend(split_section(section, content_height))
        else:
            final_sections.append(section)
    return final_sections
