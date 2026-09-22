"""
Phase 1 Tests: Ingestion & Language Detection

Tests PDF extraction (digital and OCR) and language detection
against the sample court judgment PDFs.
"""

import os
import sys
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.ingestion.pdf_parser import extract_pdf, extract_page, DocumentResult, PageResult
from src.ingestion.lang_detector import detect_language, detect_scripts, LanguageResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_DIR = PROJECT_ROOT / "data" / "sample_judgments"


def get_pdf_path(filename: str) -> Path:
    path = SAMPLE_DIR / filename
    if not path.exists():
        pytest.skip(f"Sample PDF not found: {path}")
    return path


# ---------------------------------------------------------------------------
# PDF Extraction Tests
# ---------------------------------------------------------------------------

class TestPDFExtraction:
    """Test PDF text extraction for digital and scanned documents."""

    def test_extract_english_pdf(self):
        """English digital PDF should return non-empty text."""
        path = get_pdf_path("order-pdf-english.pdf")
        result = extract_pdf(path)

        assert isinstance(result, DocumentResult)
        assert result.total_pages > 0
        assert len(result.full_text) > 100, "Extracted text should be substantial"
        assert result.file_name == "order-pdf-english.pdf"
        assert len(result.pages) == result.total_pages

    def test_extract_hindi_pdf(self):
        """Hindi digital PDF should return non-empty text."""
        path = get_pdf_path("order-pdf-hindi.pdf")
        result = extract_pdf(path)

        assert isinstance(result, DocumentResult)
        assert len(result.full_text) > 50, "Hindi PDF should contain extractable text"

    def test_extract_gujarati_pdf(self):
        """Gujarati digital PDF should return non-empty text."""
        path = get_pdf_path("order-pdf-gujrati.pdf")
        result = extract_pdf(path)

        assert isinstance(result, DocumentResult)
        assert len(result.full_text) > 50, "Gujarati PDF should contain extractable text"

    def test_extract_karnataka_pdf(self):
        """Karnataka PDF should return non-empty text."""
        path = get_pdf_path("order-pdf-karnataka.pdf")
        result = extract_pdf(path)

        assert isinstance(result, DocumentResult)
        assert len(result.full_text) > 50

    def test_extract_scanned_pdf(self):
        """Scanned/image PDF should use OCR and return non-empty text."""
        path = get_pdf_path("image-order-pdf.pdf")
        result = extract_pdf(path)

        assert isinstance(result, DocumentResult)
        assert result.total_pages > 0
        # Scanned PDFs should trigger OCR on at least some pages
        assert result.has_ocr_pages or len(result.full_text) > 50, \
            "Scanned PDF should either trigger OCR or have some extractable text"

    def test_page_results_structure(self):
        """Each page result should have correct structure."""
        path = get_pdf_path("order-pdf-english.pdf")
        result = extract_pdf(path)

        for page in result.pages:
            assert isinstance(page, PageResult)
            assert page.page_no >= 1
            assert isinstance(page.text, str)
            assert isinstance(page.is_ocr, bool)
            assert page.char_count == len(page.text)

    def test_file_not_found_raises(self):
        """Non-existent PDF should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            extract_pdf("nonexistent_file.pdf")


# ---------------------------------------------------------------------------
# Language Detection Tests
# ---------------------------------------------------------------------------

class TestLanguageDetection:
    """Test language detection for English, Hindi, and Gujarati."""

    def test_detect_english(self):
        """English text should be detected correctly."""
        text = (
            "The Hon'ble Supreme Court of India hereby dismisses the appeal "
            "filed by the petitioner under Section 302 of the Indian Penal Code. "
            "The lower court's judgment is upheld."
        )
        result = detect_language(text)

        assert isinstance(result, LanguageResult)
        assert result.dominant_language == "en"
        assert result.dominant_language_name == "English"
        assert result.dominant_confidence > 0.5

    def test_detect_hindi(self):
        """Hindi text should be detected correctly."""
        text = (
            "माननीय न्यायालय ने याचिकाकर्ता की अपील खारिज कर दी है। "
            "निचली अदालत का फैसला बरकरार रखा गया है।"
        )
        result = detect_language(text)

        assert isinstance(result, LanguageResult)
        assert result.dominant_language == "hi"
        assert result.dominant_language_name == "Hindi"

    def test_detect_gujarati(self):
        """Gujarati text should be detected correctly."""
        text = (
            "માનનીય ન્યાયાલયે અરજદારની અપીલ ફગાવી દીધી છે। "
            "નીચલી અદાલતનો ચુકાદો યથાવત રાખવામાં આવ્યો છે।"
        )
        result = detect_language(text)

        assert isinstance(result, LanguageResult)
        assert result.dominant_language == "gu"
        assert result.dominant_language_name == "Gujarati"

    def test_detect_scripts_devanagari(self):
        """Devanagari script should be detected in Hindi text."""
        text = "माननीय न्यायालय ने याचिकाकर्ता की अपील खारिज कर दी है।"
        scripts = detect_scripts(text)

        assert "Devanagari" in scripts

    def test_detect_scripts_gujarati(self):
        """Gujarati script should be detected in Gujarati text."""
        text = "માનનીય ન્યાયાલયે અરજદારની અપીલ ફગાવી દીધી છે।"
        scripts = detect_scripts(text)

        assert "Gujarati" in scripts

    def test_detect_scripts_latin(self):
        """Latin script should be detected in English text."""
        text = "The Supreme Court of India has dismissed the appeal."
        scripts = detect_scripts(text)

        assert "Latin" in scripts

    def test_detect_from_pdf_content(self):
        """Language detection should work on actual PDF extracted text."""
        path = get_pdf_path("order-pdf-english.pdf")
        result = extract_pdf(path)
        lang_result = detect_language(result.full_text)

        assert lang_result.dominant_language in ["en", "hi", "gu", "kn"]
        assert lang_result.dominant_confidence > 0.3

    def test_empty_text_handling(self):
        """Empty text should return unknown language gracefully."""
        result = detect_language("")
        assert result.dominant_language == "unknown"
        assert result.dominant_confidence == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
