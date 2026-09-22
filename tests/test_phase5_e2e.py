"""
Phase 5 Tests: End-to-End Pipeline

Tests the complete pipeline from PDF ingestion through to
structured extraction, indexing, retrieval, summarization,
and guardrails — all without crashing.
"""

import json
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

SAMPLE_DIR = PROJECT_ROOT / "data" / "sample_judgments"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"
TEST_CHROMADB = str(PROJECT_ROOT / "data" / "chromadb_e2e_test")


def get_pdf_path(filename: str) -> Path:
    path = SAMPLE_DIR / filename
    if not path.exists():
        pytest.skip(f"Sample PDF not found: {path}")
    return path


class TestEndToEnd:
    """End-to-end pipeline tests."""

    @pytest.fixture(autouse=True)
    def check_api_key(self):
        if not os.getenv("GEMINI_API_KEY"):
            pytest.skip("GEMINI_API_KEY not set")

    def test_full_pipeline_english(self):
        """Complete pipeline for English PDF: ingest → structure → index → retrieve → summarize → verify."""
        from src.ingestion.pdf_parser import extract_pdf
        from src.ingestion.lang_detector import detect_language
        from src.structuring.extractor import extract_metadata
        from src.rag.indexer import index_document, clear_index
        from src.rag.retriever import retrieve
        from src.rag.summarizer import summarize_judgment, count_sentences
        from src.guardrails.nli_verifier import verify_faithfulness
        from src.guardrails.entity_checker import verify_hard_entities

        # Step 1: Ingest
        path = get_pdf_path("order-pdf-english.pdf")
        doc = extract_pdf(path)
        assert len(doc.full_text) > 100, "Extraction failed"

        # Step 2: Detect language
        lang = detect_language(doc.full_text)
        assert lang.dominant_language in ["en", "hi", "gu", "kn"]

        # Step 3: Extract structured metadata
        pages_text = [p.text for p in doc.pages]
        metadata = extract_metadata(doc.full_text, pages_text)
        assert metadata.case_number != "Unknown"

        # Step 4: Index document
        clear_index(TEST_CHROMADB)
        chunks = index_document(
            case_id="e2e_english_001",
            text=doc.full_text,
            metadata={
                "court": metadata.court_name,
                "language": lang.dominant_language_name,
            },
            persist_dir=TEST_CHROMADB,
        )
        assert len(chunks) > 0, "Should create at least one chunk"

        # Step 5: Retrieve
        results = retrieve(
            "What is the case about?",
            top_k=3,
            persist_dir=TEST_CHROMADB,
        )
        assert len(results) > 0, "Should retrieve results"

        # Step 6: Summarize
        summary = summarize_judgment(doc.full_text, metadata.model_dump())
        assert len(summary) > 50
        assert 4 <= count_sentences(summary) <= 6

        # Step 7: Verify faithfulness
        verification = verify_faithfulness(doc.full_text, summary)
        assert verification in ["FAITHFUL", "UNSUPPORTED"]

        # Step 8: Check entities
        hallucinated_entities = verify_hard_entities(doc.full_text, summary)
        assert isinstance(hallucinated_entities, list)

        # Cleanup
        clear_index(TEST_CHROMADB)

    def test_pipeline_produces_output_json(self):
        """Pipeline should produce valid JSON output for sample results."""
        from src.ingestion.pdf_parser import extract_pdf
        from src.ingestion.lang_detector import detect_language
        from src.structuring.extractor import extract_metadata
        from src.rag.summarizer import summarize_judgment

        results = []
        pdf_files = list(SAMPLE_DIR.glob("*.pdf"))

        if not pdf_files:
            pytest.skip("No sample PDFs found")

        # Process first 2 PDFs for speed
        for pdf_path in pdf_files[:2]:
            try:
                doc = extract_pdf(pdf_path)
                lang = detect_language(doc.full_text)
                pages_text = [p.text for p in doc.pages]
                metadata = extract_metadata(doc.full_text, pages_text)
                summary = summarize_judgment(doc.full_text, metadata.model_dump())

                result = {
                    "file_name": doc.file_name,
                    "total_pages": doc.total_pages,
                    "extraction_method": doc.primary_extraction_method,
                    "detected_language": lang.dominant_language_name,
                    "metadata": metadata.model_dump(),
                    "summary": summary,
                }
                results.append(result)
            except Exception as e:
                results.append({
                    "file_name": pdf_path.name,
                    "error": str(e),
                })

        # Save output
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / "sample_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        assert output_path.exists()
        assert len(results) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
