"""OCR fallback for scanned PDF pages (no extractable text layer).

Requires system packages: tesseract-ocr, poppler-utils (for pdf2image).
"""
from PIL import Image
from pdf2image import convert_from_path
import pytesseract


def page_to_image(pdf_path: str, page_num: int, dpi: int = 200) -> Image.Image:
    """Rasterize a single PDF page to a PIL image."""
    images = convert_from_path(
        pdf_path, dpi=dpi, first_page=page_num, last_page=page_num
    )
    return images[0]


def ocr_page_image(image: Image.Image) -> str:
    """Run Tesseract OCR on a page image and return extracted text."""
    return pytesseract.image_to_string(image).strip()