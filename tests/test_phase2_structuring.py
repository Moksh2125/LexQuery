"""
Phase 2 Tests: Information Structuring (Pydantic + Gemini)

Tests schema validation and Gemini-based metadata extraction
from court judgment documents.
"""

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.structuring.schemas import (
    CourtMetadata,
    DisposalOutcome,
    Parties,
    DocumentMetadata,
)
from src.structuring.extractor import (
    extract_metadata,
    extract_preamble,
    extract_disposition,
    map_vernacular_outcome,
)
from src.ingestion.pdf_parser import extract_pdf

SAMPLE_DIR = PROJECT_ROOT / "data" / "sample_judgments"


def get_pdf_path(filename: str) -> Path:
    path = SAMPLE_DIR / filename
    if not path.exists():
        pytest.skip(f"Sample PDF not found: {path}")
    return path


# ---------------------------------------------------------------------------
# Schema Validation Tests
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    """Test Pydantic schema validation for CourtMetadata."""

    def test_valid_metadata_creation(self):
        """Valid metadata should create without errors."""
        metadata = CourtMetadata(
            case_number="Criminal Appeal No. 123/2024",
            court_name="High Court of Gujarat",
            parties=Parties(
                petitioner=["John Doe"],
                respondent=["State of Gujarat"],
            ),
            judge_names=["Hon. Justice A.B. Sharma"],
            decision_date="15/03/2024",
            disposal_outcome=DisposalOutcome.DISMISSED,
            raw_outcome_verbatim="The appeal is hereby dismissed.",
        )

        assert metadata.case_number == "Criminal Appeal No. 123/2024"
        assert metadata.court_name == "High Court of Gujarat"
        assert len(metadata.parties.petitioner) == 1
        assert len(metadata.judge_names) == 1
        assert metadata.disposal_outcome == "Dismissed"

    def test_metadata_json_roundtrip(self):
        """Metadata should serialize and deserialize correctly."""
        original = CourtMetadata(
            case_number="SCA 456/2023",
            court_name="Supreme Court of India",
            parties=Parties(
                petitioner=["Ram Kumar", "Shyam Kumar"],
                respondent=["State of Rajasthan"],
            ),
            judge_names=["Justice X", "Justice Y"],
            decision_date="01/01/2024",
            disposal_outcome=DisposalOutcome.ALLOWED,
            raw_outcome_verbatim="Appeal allowed.",
        )

        json_str = original.model_dump_json()
        restored = CourtMetadata.model_validate_json(json_str)

        assert restored.case_number == original.case_number
        assert restored.court_name == original.court_name
        assert len(restored.parties.petitioner) == 2

    def test_minimal_metadata(self):
        """Metadata with only required fields should be valid."""
        metadata = CourtMetadata(
            case_number="Unknown",
            court_name="Unknown",
        )
        assert metadata.case_number == "Unknown"
        assert metadata.disposal_outcome == "Unknown"
        assert metadata.cnr_number is None
        assert metadata.decision_date is None

    def test_all_disposal_outcomes(self):
        """All disposal outcome enum values should be valid."""
        for outcome in DisposalOutcome:
            metadata = CourtMetadata(
                case_number="Test",
                court_name="Test Court",
                disposal_outcome=outcome,
            )
            assert metadata.disposal_outcome == outcome.value


# ---------------------------------------------------------------------------
# Vernacular Outcome Mapping Tests
# ---------------------------------------------------------------------------

class TestVernacularMapping:
    """Test vernacular phrase to outcome mapping."""

    def test_hindi_dismissed(self):
        """Hindi dismissal phrase should map correctly."""
        result = map_vernacular_outcome("अपील खारिज की जाती है")
        assert result == DisposalOutcome.DISMISSED

    def test_gujarati_allowed(self):
        """Gujarati allowed phrase should map correctly."""
        result = map_vernacular_outcome("અપીલ મંજૂર")
        assert result == DisposalOutcome.ALLOWED

    def test_english_bail_granted(self):
        """English bail granted phrase should map correctly."""
        result = map_vernacular_outcome("bail granted")
        assert result == DisposalOutcome.BAIL_GRANTED

    def test_unknown_phrase(self):
        """Unknown phrases should return UNKNOWN."""
        result = map_vernacular_outcome("some random text")
        assert result == DisposalOutcome.UNKNOWN


# ---------------------------------------------------------------------------
# Preamble/Disposition Extraction Tests
# ---------------------------------------------------------------------------

class TestTextWindowing:
    """Test preamble and disposition extraction."""

    def test_preamble_extraction(self):
        """Should extract first 2 pages of text."""
        pages = ["Page 1 content", "Page 2 content", "Page 3 content", "Page 4 content"]
        preamble = extract_preamble(pages, n=2)
        assert "Page 1 content" in preamble
        assert "Page 2 content" in preamble
        assert "Page 3 content" not in preamble

    def test_disposition_extraction(self):
        """Should extract last 2 pages of text."""
        pages = ["Page 1 content", "Page 2 content", "Page 3 content", "Page 4 content"]
        disposition = extract_disposition(pages, n=2)
        assert "Page 3 content" in disposition
        assert "Page 4 content" in disposition
        assert "Page 1 content" not in disposition


# ---------------------------------------------------------------------------
# Gemini Extraction Tests (require API key)
# ---------------------------------------------------------------------------

class TestGeminiExtraction:
    """Test Gemini-based metadata extraction from actual PDFs."""

    @pytest.fixture(autouse=True)
    def check_api_key(self):
        if not os.getenv("GEMINI_API_KEY"):
            pytest.skip("GEMINI_API_KEY not set")

    def test_extract_english_metadata(self):
        """Extract metadata from English court judgment PDF."""
        path = get_pdf_path("order-pdf-english.pdf")
        doc = extract_pdf(path)
        pages_text = [p.text for p in doc.pages]

        metadata = extract_metadata(doc.full_text, pages_text)

        assert isinstance(metadata, CourtMetadata)
        assert metadata.case_number != "Unknown", "Should extract a case number"
        assert metadata.court_name != "Unknown", "Should extract court name"
        assert len(metadata.judge_names) >= 0  # May or may not have judges

    def test_extract_hindi_metadata(self):
        """Extract metadata from Hindi court judgment PDF."""
        path = get_pdf_path("order-pdf-hindi.pdf")
        doc = extract_pdf(path)
        pages_text = [p.text for p in doc.pages]

        metadata = extract_metadata(doc.full_text, pages_text)

        assert isinstance(metadata, CourtMetadata)
        assert metadata.case_number != "Unknown"

    def test_extract_gujarati_metadata(self):
        """Extract metadata from Gujarati court judgment PDF."""
        path = get_pdf_path("order-pdf-gujrati.pdf")
        doc = extract_pdf(path)
        pages_text = [p.text for p in doc.pages]

        metadata = extract_metadata(doc.full_text, pages_text)

        assert isinstance(metadata, CourtMetadata)
        assert metadata.case_number != "Unknown"

    def test_metadata_validates_fully(self):
        """Extracted metadata should validate against CourtMetadata schema."""
        path = get_pdf_path("order-pdf-english.pdf")
        doc = extract_pdf(path)
        pages_text = [p.text for p in doc.pages]

        metadata = extract_metadata(doc.full_text, pages_text)

        # Validate by round-tripping through JSON
        json_data = metadata.model_dump()
        restored = CourtMetadata(**json_data)

        assert restored.case_number == metadata.case_number
        assert restored.court_name == metadata.court_name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
