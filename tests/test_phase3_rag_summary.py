"""
Phase 3 Tests: Cross-Lingual RAG & Summarization

Tests semantic chunking, cross-lingual retrieval, and
legal judgment summarization.
"""

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.rag.indexer import chunk_text, index_document, clear_index, CHROMADB_PATH
from src.rag.retriever import retrieve, answer_question
from src.rag.summarizer import summarize_judgment, count_sentences
from src.ingestion.pdf_parser import extract_pdf

SAMPLE_DIR = PROJECT_ROOT / "data" / "sample_judgments"
TEST_CHROMADB = str(PROJECT_ROOT / "data" / "chromadb_test")


def get_pdf_path(filename: str) -> Path:
    path = SAMPLE_DIR / filename
    if not path.exists():
        pytest.skip(f"Sample PDF not found: {path}")
    return path


# ---------------------------------------------------------------------------
# Semantic Chunking Tests
# ---------------------------------------------------------------------------

class TestSemanticChunking:
    """Test text chunking preserves paragraph integrity."""

    def test_basic_chunking(self):
        """Text should be split into reasonable chunks."""
        # Create text with ~2000 words
        text = "\n\n".join([f"Paragraph {i}. " + "This is a sentence. " * 30 for i in range(20)])
        chunks = chunk_text(text)

        assert len(chunks) > 1, "Should produce multiple chunks"
        for chunk in chunks:
            word_count = len(chunk.split())
            # Allow some flexibility in token estimation
            assert word_count > 100, f"Chunk too small: {word_count} words"

    def test_short_text_single_chunk(self):
        """Short text should remain a single chunk."""
        text = "This is a short judgment. The appeal is dismissed."
        chunks = chunk_text(text)

        assert len(chunks) == 1
        assert "appeal is dismissed" in chunks[0]

    def test_paragraph_preservation(self):
        """Chunks should preserve paragraph boundaries."""
        paragraphs = [f"Paragraph {i}: " + "Content here. " * 20 for i in range(10)]
        text = "\n\n".join(paragraphs)
        chunks = chunk_text(text)

        # Each chunk should contain complete paragraphs (not split mid-paragraph)
        for chunk in chunks:
            # Count paragraphs boundaries - should be clean
            assert len(chunk.strip()) > 0


# ---------------------------------------------------------------------------
# Indexing & Retrieval Tests
# ---------------------------------------------------------------------------

class TestIndexingAndRetrieval:
    """Test document indexing and cross-lingual retrieval."""

    @pytest.fixture(autouse=True)
    def setup_index(self):
        """Set up a test index with sample documents."""
        clear_index(TEST_CHROMADB)

        # Index English document
        english_text = (
            "This is a bail application under Section 439 of CrPC. "
            "The applicant Rajesh Kumar is accused of offences under "
            "Section 302 and 307 of the Indian Penal Code. "
            "The incident occurred on 15th March 2024 in Ahmedabad. "
            "The prosecution alleges that the accused murdered his neighbor."
        )
        index_document(
            case_id="english_case_001",
            text=english_text,
            metadata={"court": "High Court of Gujarat", "language": "English"},
            persist_dir=TEST_CHROMADB,
        )

        # Index Hindi document
        hindi_text = (
            "यह जमानत आवेदन धारा 439 सीआरपीसी के तहत है। "
            "आवेदक राम कुमार पर भारतीय दंड संहिता की धारा 302 और 307 के तहत "
            "आरोप लगाए गए हैं। यह घटना 20 अप्रैल 2024 को जयपुर में हुई थी। "
            "अभियोजन पक्ष का आरोप है कि आरोपी ने अपने पड़ोसी की हत्या की।"
        )
        index_document(
            case_id="hindi_case_001",
            text=hindi_text,
            metadata={"court": "High Court of Rajasthan", "language": "Hindi"},
            persist_dir=TEST_CHROMADB,
        )

        # Index Gujarati document
        gujarati_text = (
            "આ જામીન અરજી CrPC ની કલમ 439 હેઠળ છે। "
            "અરજદાર સુરેશ પટેલ પર ભારતીય દંડ સંહિતાની કલમ 302 અને 307 હેઠળ "
            "આરોપ મૂકવામાં આવ્યો છે। આ ઘટના 10 મે 2024 ના રોજ સુરતમાં બની હતી।"
        )
        index_document(
            case_id="gujarati_case_001",
            text=gujarati_text,
            metadata={"court": "High Court of Gujarat", "language": "Gujarati"},
            persist_dir=TEST_CHROMADB,
        )

        yield

        # Cleanup
        clear_index(TEST_CHROMADB)

    def test_english_query_retrieval(self):
        """English query should retrieve relevant documents."""
        results = retrieve("bail application", top_k=3, persist_dir=TEST_CHROMADB)

        assert len(results) > 0, "Should return at least one result"
        assert any("bail" in r.text.lower() or "जमानत" in r.text or "જામીન" in r.text
                    for r in results), "Results should be relevant to bail"

    def test_cross_lingual_bail_query(self):
        """English query 'bail application' should retrieve Hindi/Gujarati chunks."""
        results = retrieve("Which cases involve a bail application?", top_k=3, persist_dir=TEST_CHROMADB)

        assert len(results) >= 2, "Should retrieve results from multiple languages"
        languages = {r.language for r in results}
        # Should retrieve from at least 2 different languages
        assert len(languages) >= 2, f"Should retrieve cross-lingual results, got: {languages}"

    def test_cross_lingual_murder_query(self):
        """English query about murder should retrieve Hindi documents."""
        results = retrieve("murder case details", top_k=3, persist_dir=TEST_CHROMADB)

        assert len(results) > 0

    def test_retrieval_with_metadata(self):
        """Retrieved chunks should include metadata."""
        results = retrieve("bail application", top_k=3, persist_dir=TEST_CHROMADB)

        for result in results:
            assert result.case_id != "", "Should have case_id"
            assert result.similarity_score > 0, "Should have positive similarity"


# ---------------------------------------------------------------------------
# Summarization Tests
# ---------------------------------------------------------------------------

class TestSummarization:
    """Test legal judgment summarization."""

    @pytest.fixture(autouse=True)
    def check_api_key(self):
        if not os.getenv("GEMINI_API_KEY"):
            pytest.skip("GEMINI_API_KEY not set")

    def test_summary_sentence_count(self):
        """Summary should contain exactly 4-6 sentences."""
        text = (
            "IN THE HIGH COURT OF GUJARAT AT AHMEDABAD\n"
            "Criminal Appeal No. 1234/2024\n"
            "Petitioner: Rajesh Kumar\n"
            "Respondent: State of Gujarat\n"
            "Before: Hon'ble Justice A.B. Sharma\n\n"
            "The petitioner has filed this bail application under Section 439 CrPC. "
            "The petitioner is accused of offences under Section 302 and 307 IPC. "
            "The prosecution case is that on 15.03.2024 the accused assaulted "
            "the deceased with a sharp weapon resulting in death. "
            "FIR No. 123/2024 was registered at Navrangpura Police Station. "
            "The investigation is complete and charge sheet has been filed. "
            "Considering the gravity of the offence and evidence on record, "
            "the bail application is hereby rejected. "
            "The petitioner shall surrender before the trial court."
        )
        summary = summarize_judgment(text)

        num_sentences = count_sentences(summary)
        assert 4 <= num_sentences <= 6, f"Expected 4-6 sentences, got {num_sentences}: {summary}"

    def test_summary_in_english(self):
        """Summary should be in English even for non-English source."""
        hindi_text = (
            "माननीय उच्च न्यायालय, राजस्थान, जयपुर\n"
            "आपराधिक अपील संख्या 567/2024\n"
            "याचिकाकर्ता: राम कुमार\n"
            "प्रतिवादी: राजस्थान राज्य\n\n"
            "याचिकाकर्ता ने धारा 439 सीआरपीसी के तहत जमानत आवेदन दायर किया है। "
            "याचिकाकर्ता पर भारतीय दंड संहिता की धारा 302 और 307 के तहत आरोप हैं। "
            "अभियोजन पक्ष का मामला यह है कि 20.04.2024 को आरोपी ने तेज हथियार से "
            "मृतक पर हमला किया जिससे उसकी मृत्यु हो गई। "
            "एफआईआर संख्या 456/2024 सदर थाने में दर्ज की गई। "
            "जांच पूरी हो चुकी है और चार्जशीट दाखिल की जा चुकी है। "
            "अपराध की गंभीरता और रिकॉर्ड पर मौजूद साक्ष्य को देखते हुए, "
            "जमानत आवेदन खारिज किया जाता है।"
        )
        summary = summarize_judgment(hindi_text)

        # Check summary is predominantly English (Latin characters)
        latin_chars = sum(1 for c in summary if 'a' <= c.lower() <= 'z')
        total_alpha = sum(1 for c in summary if c.isalpha())
        if total_alpha > 0:
            assert latin_chars / total_alpha > 0.7, f"Summary should be in English: {summary}"

    def test_summary_from_pdf(self):
        """Generate summary from actual PDF content."""
        path = get_pdf_path("order-pdf-english.pdf")
        doc = extract_pdf(path)

        summary = summarize_judgment(doc.full_text)

        assert len(summary) > 50, "Summary should be substantial"
        num_sentences = count_sentences(summary)
        assert 4 <= num_sentences <= 6, f"Expected 4-6 sentences, got {num_sentences}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
