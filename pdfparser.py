"""Parses a PDF into per-page text, tables, and image crops.

Design: pdfplumber gives us native text + table bounding boxes for digital
PDFs "for free". For pages with no extractable text (scans), we fall back to
OCR (see ocr.py). Anything on the page that isn't text and isn't a clean
table (charts, photos, diagrams) is cropped out as an image region so it can
be captioned later by the vision model instead of silently dropped.
"""
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber
from PIL import Image

from backend.ingestion.ocr import ocr_page_image, page_to_image


@dataclass
class TableRegion:
    page: int
    bbox: list[float]
    rows: list[list[str]]


@dataclass
class ImageRegion:
    page: int
    bbox: list[float]
    image: Image.Image


@dataclass
class PageContent:
    page: int
    text: str
    tables: list[TableRegion] = field(default_factory=list)
    images: list[ImageRegion] = field(default_factory=list)
    used_ocr: bool = False


def parse_pdf(pdf_path: str) -> list[PageContent]:
    pages: list[PageContent] = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_num = i + 1
            native_text = (page.extract_text() or "").strip()

            tables = _extract_tables(page, page_num)
            table_bboxes = [t.bbox for t in tables]

            images = _extract_image_regions(page, page_num, exclude_bboxes=table_bboxes)

            used_ocr = False
            text = native_text
            if not native_text:
                # Scanned page: no text layer, fall back to OCR
                rendered = page_to_image(pdf_path, page_num)
                text = ocr_page_image(rendered)
                used_ocr = True

            pages.append(
                PageContent(
                    page=page_num,
                    text=text,
                    tables=tables,
                    images=images,
                    used_ocr=used_ocr,
                )
            )

    return pages


def _extract_tables(page, page_num: int) -> list[TableRegion]:
    tables: list[TableRegion] = []
    try:
        found = page.find_tables()
    except Exception:
        found = []

    for t in found:
        rows = t.extract() or []
        rows = [[c or "" for c in row] for row in rows]
        if not rows:
            continue
        tables.append(
            TableRegion(page=page_num, bbox=list(t.bbox), rows=rows)
        )
    return tables


def _extract_image_regions(
    page, page_num: int, exclude_bboxes: list[list[float]]
) -> list[ImageRegion]:
    """Crop embedded raster images (charts, photos) that aren't inside a
    detected table region."""
    regions: list[ImageRegion] = []
    page_image = None

    for img in page.images:
        bbox = [img["x0"], img["top"], img["x1"], img["bottom"]]
        if _overlaps_any(bbox, exclude_bboxes):
            continue
        # Skip tiny images (icons, bullet glyphs) — not worth captioning
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width < 40 or height < 40:
            continue

        try:
            if page_image is None:
                page_image = page.to_image(resolution=150).original
            scale = page_image.width / page.width
            crop_box = tuple(int(v * scale) for v in bbox)
            cropped = page_image.crop(crop_box)
            regions.append(ImageRegion(page=page_num, bbox=bbox, image=cropped))
        except Exception:
            continue

    return regions


def _overlaps_any(bbox: list[float], others: list[list[float]]) -> bool:
    for o in others:
        if _iou(bbox, o) > 0.3:
            return True
    return False


def _iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0