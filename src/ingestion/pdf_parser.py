"""
PDF Parser Module - Dual-path extraction (PyMuPDF + pytesseract OCR)

Handles both digital/typed PDFs and scanned/image PDFs across
English, Hindi, and Gujarati court judgment documents.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF
from PIL import Image
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic result models
# ---------------------------------------------------------------------------

class PageResult(BaseModel):
    """Extraction result for a single PDF page."""
    page_no: int = Field(..., description="1-indexed page number")
    text: str = Field(default="", description="Extracted text content")
    is_ocr: bool = Field(default=False, description="True if OCR was used")
    ocr_confidence: Optional[float] = Field(default=None, description="OCR confidence 0-100")
    char_count: int = Field(default=0, description="Character count of extracted text")


class DocumentResult(BaseModel):
    """Extraction result for an entire PDF document."""
    file_path: str
    file_name: str
    total_pages: int
    full_text: str
    pages: List[PageResult]
    has_ocr_pages: bool = False
    primary_extraction_method: str = "digital"  # "digital" or "ocr" or "mixed"


# ---------------------------------------------------------------------------
# OCR Configuration
# ---------------------------------------------------------------------------

def _configure_tesseract() -> None:
    """Configure pytesseract with the correct Tesseract executable path."""
    import pytesseract

    # Check .env or environment variable first
    tesseract_cmd = os.getenv("TESSERACT_CMD")
    if tesseract_cmd and os.path.isfile(tesseract_cmd):
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        return

    # Common Windows install paths
    common_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Tesseract-OCR\tesseract.exe"),
    ]
    for path in common_paths:
        if os.path.isfile(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return

    logger.warning(
        "Tesseract executable not found at common paths. "
        "OCR will fail for scanned pages. Install Tesseract or set TESSERACT_CMD."
    )


def _ocr_page_image(pix: fitz.Pixmap, lang: str = "eng+hin+guj") -> tuple[str, float]:
    """
    Run OCR on a PyMuPDF Pixmap using pytesseract.

    Returns:
        Tuple of (extracted_text, confidence_score).
    """
    import pytesseract

    _configure_tesseract()

    # Convert Pixmap to PIL Image
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Get OCR text
    text = pytesseract.image_to_string(img, lang=lang)

    # Get confidence data
    try:
        data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)
        confidences = [int(c) for c in data.get("conf", []) if str(c).strip() and int(c) >= 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    except Exception:
        avg_confidence = 0.0

    return text.strip(), avg_confidence


# ---------------------------------------------------------------------------
# Main extraction functions
# ---------------------------------------------------------------------------

MIN_CHARS_THRESHOLD = 50  # Below this, page is considered scanned/image


def extract_page(page: fitz.Page, page_no: int, ocr_lang: str = "eng+hin+guj") -> PageResult:
    """
    Extract text from a single PDF page.
    Uses direct text extraction first; falls back to OCR if char count < 50.
    """
    # Try digital extraction first
    text = page.get_text("text").strip()
    char_count = len(text)

    if char_count >= MIN_CHARS_THRESHOLD:
        # Digital/typed page - text extraction succeeded
        return PageResult(
            page_no=page_no,
            text=text,
            is_ocr=False,
            ocr_confidence=None,
            char_count=len(text),
        )

    # Scanned/image page - fall back to OCR
    logger.info(f"Page {page_no}: char_count={char_count} < {MIN_CHARS_THRESHOLD}, using OCR")
    try:
        # Render page at 300 DPI
        mat = fitz.Matrix(300 / 72, 300 / 72)  # 300 DPI scale factor
        pix = page.get_pixmap(matrix=mat)

        ocr_text, confidence = _ocr_page_image(pix, lang=ocr_lang)

        return PageResult(
            page_no=page_no,
            text=ocr_text,
            is_ocr=True,
            ocr_confidence=round(confidence, 2),
            char_count=len(ocr_text),
        )
    except Exception as e:
        logger.error(f"OCR failed for page {page_no}: {e}")
        # Return whatever digital text we got (even if sparse)
        return PageResult(
            page_no=page_no,
            text=text,
            is_ocr=False,
            ocr_confidence=None,
            char_count=char_count,
        )


def extract_pdf(pdf_path: str | Path) -> DocumentResult:
    """
    Extract text from all pages of a PDF document.
    Automatically detects digital vs. scanned pages and applies OCR as needed.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        DocumentResult with full text and per-page results.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    pages: List[PageResult] = []

    for i in range(len(doc)):
        page = doc[i]
        page_result = extract_page(page, page_no=i + 1)
        pages.append(page_result)

    doc.close()

    # Aggregate results
    full_text = "\n\n".join(p.text for p in pages if p.text)
    has_ocr = any(p.is_ocr for p in pages)
    all_ocr = all(p.is_ocr for p in pages)

    if all_ocr:
        method = "ocr"
    elif has_ocr:
        method = "mixed"
    else:
        method = "digital"

    return DocumentResult(
        file_path=str(pdf_path),
        file_name=pdf_path.name,
        total_pages=len(pages),
        full_text=full_text,
        pages=pages,
        has_ocr_pages=has_ocr,
        primary_extraction_method=method,
    )
