"""
LexQuery — Multilingual Legal Intelligence Platform
Streamlit Frontend Application

Three-tab interface:
1. 📤 Upload & Ingest — PDF upload with OCR/language detection
2. 📋 Triage Dashboard — Structured metadata cards with summaries
3. 💬 Cross-Lingual Q&A — Chat-based legal Q&A with faithfulness scores
"""

import json
import os
import sys
import time
from pathlib import Path

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.ingestion.pdf_parser import extract_pdf
from src.ingestion.lang_detector import detect_language
from src.structuring.extractor import extract_metadata
from src.structuring.schemas import CourtMetadata, DocumentMetadata
from src.rag.indexer import index_document, clear_index
from src.rag.retriever import answer_question
from src.rag.summarizer import summarize_judgment
from src.guardrails.nli_verifier import verify_faithfulness
from src.guardrails.entity_checker import verify_hard_entities

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LexQuery — Legal Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for premium look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main .block-container { padding-top: 1rem; max-width: 1200px; }

    /* Header */
    .lex-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.15);
    }
    .lex-header h1 {
        color: #e8e8ff;
        font-weight: 700;
        font-size: 1.8rem;
        margin: 0;
    }
    .lex-header p {
        color: #a8a8d8;
        font-size: 0.95rem;
        margin: 0.3rem 0 0 0;
    }

    /* Cards */
    .case-card {
        background: #ffffff;
        border: 1px solid #e8eaf6;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .case-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    .case-card h3 {
        color: #1a1a2e;
        font-size: 1.05rem;
        margin-bottom: 0.5rem;
    }

    /* Outcome badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .badge-allowed { background: #e8f5e9; color: #2e7d32; }
    .badge-dismissed { background: #ffebee; color: #c62828; }
    .badge-disposed { background: #fff3e0; color: #e65100; }
    .badge-quashed { background: #e3f2fd; color: #1565c0; }
    .badge-bail-granted { background: #e8f5e9; color: #1b5e20; }
    .badge-bail-rejected { background: #fce4ec; color: #b71c1c; }
    .badge-pending { background: #f3e5f5; color: #6a1b9a; }
    .badge-unknown { background: #f5f5f5; color: #616161; }

    /* Language badges */
    .lang-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 500;
        margin-right: 0.3rem;
    }
    .lang-english { background: #e3f2fd; color: #1565c0; }
    .lang-hindi { background: #fff3e0; color: #e65100; }
    .lang-gujarati { background: #e8f5e9; color: #2e7d32; }
    .lang-kannada { background: #f3e5f5; color: #7b1fa2; }
    .lang-ocr { background: #fce4ec; color: #c62828; }
    .lang-digital { background: #e8eaf6; color: #283593; }

    /* Faithfulness meter */
    .faith-meter {
        height: 8px;
        border-radius: 4px;
        background: #f0f0f0;
        margin-top: 0.5rem;
        overflow: hidden;
    }
    .faith-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s;
    }

    /* Chat */
    .chat-msg {
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 0.75rem;
    }
    .chat-user {
        background: #e3f2fd;
        border-left: 4px solid #1565c0;
        color: #1a1a2e !important;
    }
    .chat-bot {
        background: #f5f5f5;
        border-left: 4px solid #4caf50;
        color: #1a1a2e !important;
    }

    /* Source expander */
    .source-snippet {
        background: #fafafa;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 0.75rem;
        font-size: 0.85rem;
        margin: 0.5rem 0;
    }

    div[data-testid="stTabs"] button {
        font-weight: 600;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "documents" not in st.session_state:
    st.session_state.documents = {}  # {filename: DocumentMetadata dict}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = set()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_outcome_badge(outcome: str) -> str:
    """Return HTML badge for disposal outcome."""
    css_class = {
        "Allowed": "badge-allowed",
        "Dismissed": "badge-dismissed",
        "Disposed": "badge-disposed",
        "Quashed": "badge-quashed",
        "Bail Granted": "badge-bail-granted",
        "Bail Rejected": "badge-bail-rejected",
        "Pending": "badge-pending",
    }.get(outcome, "badge-unknown")
    return f'<span class="badge {css_class}">{outcome}</span>'


def get_lang_badge(language: str) -> str:
    """Return HTML badge for detected language."""
    css_class = {
        "English": "lang-english",
        "Hindi": "lang-hindi",
        "Gujarati": "lang-gujarati",
        "Kannada": "lang-kannada",
    }.get(language, "lang-english")
    return f'<span class="lang-badge {css_class}">{language}</span>'


def get_method_badge(method: str) -> str:
    """Return HTML badge for extraction method."""
    if method == "ocr" or method == "mixed":
        return '<span class="lang-badge lang-ocr">🔍 OCR</span>'
    return '<span class="lang-badge lang-digital">📄 Digital</span>'


def get_faith_color(score: float) -> str:
    """Return color for faithfulness score."""
    if score >= 0.7:
        return "#4caf50"
    elif score >= 0.4:
        return "#ff9800"
    return "#f44336"


def save_results():
    """Save all processed results to data/outputs/sample_results.json."""
    output_dir = PROJECT_ROOT / "data" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for fname, doc_data in st.session_state.documents.items():
        results.append(doc_data)

    output_path = output_dir / "sample_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="lex-header">
    <h1>⚖️ LexQuery</h1>
    <p>Multilingual Legal Intelligence Platform — Indian Court Judgments</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_upload, tab_triage, tab_qa = st.tabs([
    "📤 Upload & Ingest",
    "📋 Triage Dashboard",
    "💬 Cross-Lingual Q&A",
])


# ===========================================================================
# TAB 1: Upload & Ingest
# ===========================================================================
with tab_upload:
    st.markdown("### Upload Court Judgment PDFs")
    st.markdown("Upload typed or scanned PDFs in English, Hindi, or Gujarati. "
                "The system will automatically detect the language and extract structured metadata.")

    uploaded_files = st.file_uploader(
        "Drag and drop PDF files here",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
    )

    if uploaded_files:
        process_btn = st.button("🚀 Process All Documents", type="primary", use_container_width=True)

        if process_btn:
            progress_bar = st.progress(0, text="Initializing...")
            status_container = st.container()
            total = len(uploaded_files)

            for idx, uploaded_file in enumerate(uploaded_files):
                fname = uploaded_file.name
                progress_pct = (idx) / total
                progress_bar.progress(progress_pct, text=f"Processing {fname}... ({idx+1}/{total})")

                with status_container:
                    with st.status(f"Processing: {fname}", expanded=True) as status:
                        try:
                            # Save uploaded file temporarily
                            temp_dir = PROJECT_ROOT / "data" / "temp"
                            temp_dir.mkdir(parents=True, exist_ok=True)
                            temp_path = temp_dir / fname
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getvalue())

                            # Step 1: Extract text
                            st.write("📄 Extracting text from PDF...")
                            doc = extract_pdf(temp_path)

                            # Step 2: Detect language
                            st.write("🌐 Detecting language...")
                            lang_result = detect_language(doc.full_text)

                            # Step 3: Extract metadata
                            st.write("🏛️ Extracting structured metadata via Gemini...")
                            pages_text = [p.text for p in doc.pages]
                            metadata = extract_metadata(doc.full_text, pages_text)

                            # Step 4: Generate summary
                            st.write("📝 Generating legal summary...")
                            summary = summarize_judgment(
                                doc.full_text,
                                metadata.model_dump(),
                            )

                            # Step 5: Index for RAG
                            st.write("🔍 Indexing for cross-lingual search...")
                            index_document(
                                case_id=fname,
                                text=doc.full_text,
                                metadata={
                                    "court": metadata.court_name,
                                    "language": lang_result.dominant_language_name,
                                },
                            )
                            st.session_state.indexed_docs.add(fname)

                            # Store results
                            doc_data = {
                                "file_name": fname,
                                "total_pages": doc.total_pages,
                                "extraction_method": doc.primary_extraction_method,
                                "has_ocr": doc.has_ocr_pages,
                                "detected_language": lang_result.dominant_language_name,
                                "detected_scripts": lang_result.detected_scripts,
                                "language_confidence": lang_result.dominant_confidence,
                                "metadata": metadata.model_dump(),
                                "summary": summary,
                                "full_text": doc.full_text[:5000],  # Truncate for storage
                            }
                            st.session_state.documents[fname] = doc_data

                            status.update(label=f"✅ {fname} — processed successfully!", state="complete")

                        except Exception as e:
                            st.error(f"Error processing {fname}: {str(e)}")
                            st.session_state.documents[fname] = {
                                "file_name": fname,
                                "error": str(e),
                                "processing_status": "error",
                            }
                            status.update(label=f"❌ {fname} — error", state="error")

                        finally:
                            # Clean up temp file
                            if temp_path.exists():
                                temp_path.unlink()

            progress_bar.progress(1.0, text="✅ All documents processed!")

            # Save results
            save_results()
            st.success(f"Processed {total} documents. Results saved to `data/outputs/sample_results.json`.")

    # Show already processed documents
    if st.session_state.documents:
        st.markdown("---")
        st.markdown("### 📁 Processed Documents")
        for fname, doc_data in st.session_state.documents.items():
            if "error" in doc_data:
                st.error(f"❌ {fname}: {doc_data['error']}")
                continue

            lang = doc_data.get("detected_language", "Unknown")
            method = doc_data.get("extraction_method", "digital")
            pages = doc_data.get("total_pages", 0)

            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            with col1:
                st.markdown(f"**{fname}**")
            with col2:
                st.markdown(get_lang_badge(lang), unsafe_allow_html=True)
            with col3:
                st.markdown(get_method_badge(method), unsafe_allow_html=True)
            with col4:
                st.caption(f"{pages} pages")


# ===========================================================================
# TAB 2: Triage Dashboard
# ===========================================================================
with tab_triage:
    if not st.session_state.documents:
        st.info("📤 No documents processed yet. Upload PDFs in the Upload tab to see the triage dashboard.")
    else:
        st.markdown("### 📋 Case Triage Dashboard")

        # Filter options
        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            all_outcomes = set()
            for doc_data in st.session_state.documents.values():
                if "metadata" in doc_data:
                    all_outcomes.add(doc_data["metadata"].get("disposal_outcome", "Unknown"))
            outcome_filter = st.multiselect(
                "Filter by Outcome",
                options=sorted(all_outcomes) if all_outcomes else ["All"],
                default=[],
            )
        with col_filter2:
            all_languages = set()
            for doc_data in st.session_state.documents.values():
                if "detected_language" in doc_data:
                    all_languages.add(doc_data.get("detected_language", "Unknown"))
            lang_filter = st.multiselect(
                "Filter by Language",
                options=sorted(all_languages) if all_languages else ["All"],
                default=[],
            )

        # Display cards
        for fname, doc_data in st.session_state.documents.items():
            if "error" in doc_data or "metadata" not in doc_data:
                continue

            meta = doc_data["metadata"]
            outcome = meta.get("disposal_outcome", "Unknown")
            lang = doc_data.get("detected_language", "Unknown")

            # Apply filters
            if outcome_filter and outcome not in outcome_filter:
                continue
            if lang_filter and lang not in lang_filter:
                continue

            # Build card
            parties = meta.get("parties", {})
            petitioners = ", ".join(parties.get("petitioner", ["Unknown"]))
            respondents = ", ".join(parties.get("respondent", ["Unknown"]))
            judges = ", ".join(meta.get("judge_names", ["Unknown"]))
            case_num = meta.get("case_number", "Unknown")
            court = meta.get("court_name", "Unknown")
            date = meta.get("decision_date", "N/A")
            summary = doc_data.get("summary", "No summary available.")

            outcome_badge = get_outcome_badge(outcome)
            lang_badge = get_lang_badge(lang)

            st.markdown(f"""
            <div class="case-card">
                <div style="display: flex; justify-content: space-between; align-items: start;">
                    <h3>{case_num}</h3>
                    <div>{outcome_badge} {lang_badge}</div>
                </div>
                <p style="color: #666; font-size: 0.88rem; margin: 0.3rem 0;">
                    🏛️ {court} &nbsp;|&nbsp; 📅 {date}
                </p>
                <p style="font-size: 0.88rem; margin: 0.3rem 0;">
                    <strong>Petitioner:</strong> {petitioners}<br>
                    <strong>Respondent:</strong> {respondents}<br>
                    <strong>Judge(s):</strong> {judges}
                </p>
                <hr style="margin: 0.5rem 0; border: none; border-top: 1px solid #eee;">
                <p style="font-size: 0.88rem; color: #333; line-height: 1.5;">
                    📝 {summary}
                </p>
            </div>
            """, unsafe_allow_html=True)

            with st.expander(f"📄 View Raw Outcome — {fname}"):
                st.text(meta.get("raw_outcome_verbatim", "N/A"))


# ===========================================================================
# TAB 3: Cross-Lingual Q&A
# ===========================================================================
with tab_qa:
    st.markdown("### 💬 Cross-Lingual Legal Q&A")
    st.markdown("Ask questions in English about any uploaded document — "
                "even if the source is in Hindi, Gujarati, or Kannada.")

    if not st.session_state.indexed_docs:
        st.info("📤 No documents indexed yet. Upload and process PDFs first.")
    else:
        st.caption(f"📚 {len(st.session_state.indexed_docs)} document(s) indexed for search")

        # Chat input
        user_query = st.chat_input("Ask a question about the court documents...")

        if user_query:
            st.session_state.chat_history.append({"role": "user", "content": user_query})

            with st.spinner("🔍 Searching across documents and generating answer..."):
                try:
                    # Get answer from RAG pipeline
                    result = answer_question(user_query, top_k=5)

                    # Verify faithfulness
                    if result.source_chunks:
                        source_text = "\n".join([c.text for c in result.source_chunks])
                        verification = verify_faithfulness(source_text, result.answer)
                        hallucinated_entities = verify_hard_entities(source_text, result.answer)
                    else:
                        verification = None
                        hallucinated_entities = []

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result.answer,
                        "source_chunks": [c.model_dump() for c in result.source_chunks],
                        "confidence": result.confidence,
                        "verification": verification,
                        "hallucinated_entities": hallucinated_entities,
                    })

                except Exception as e:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"Error generating answer: {str(e)}",
                        "source_chunks": [],
                        "confidence": 0.0,
                    })

        # Display chat history
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-msg chat-user">
                    <strong>🧑 You</strong><br>{msg['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                # Assistant message
                answer = msg["content"]
                confidence = msg.get("confidence", 0.0)
                verification = msg.get("verification")
                hallucinated_entities = msg.get("hallucinated_entities", [])

                if verification == "FAITHFUL":
                    faith_color = "#2e7d32"
                    faith_verdict = "FAITHFUL"
                elif verification == "UNSUPPORTED":
                    faith_color = "#c62828"
                    faith_verdict = "UNSUPPORTED"
                else:
                    faith_color = "#ff9800"
                    faith_verdict = "UNKNOWN"

                st.markdown(f"""
                <div class="chat-msg chat-bot">
                    <strong>⚖️ LexQuery</strong><br>
                    {answer}
                    <div style="margin-top: 0.75rem; display: flex; align-items: center; gap: 0.5rem;">
                        <span style="font-size: 0.78rem; color: #666;">
                            Faithfulness: <strong style="color: {faith_color};">{faith_verdict}</strong>
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Source snippets expander
                source_chunks = msg.get("source_chunks", [])
                if source_chunks:
                    with st.expander(f"📑 View {len(source_chunks)} Source Snippets"):
                        for i, chunk in enumerate(source_chunks, 1):
                            lang = chunk.get("language", "Unknown")
                            case_id = chunk.get("case_id", "Unknown")
                            score = chunk.get("similarity_score", 0)
                            st.markdown(f"""
                            <div class="source-snippet">
                                <div style="display: flex; justify-content: space-between;">
                                    <strong>Source {i}</strong>
                                    <span style="font-size: 0.75rem; color: #999;">
                                        📄 {case_id} | 🌐 {lang} | Score: {score:.2f}
                                    </span>
                                </div>
                                <p style="font-size: 0.82rem; color: #444; margin-top: 0.5rem;">
                                    {chunk.get('text', '')[:500]}...
                                </p>
                            </div>
                            """, unsafe_allow_html=True)

                # Entity check results
                if hallucinated_entities:
                    st.warning(f"⚠️ Entity check found {len(hallucinated_entities)} "
                               f"potentially hallucinated legal references: {', '.join(hallucinated_entities)}")
                elif verification is not None:
                    st.success(f"✅ All legal references verified against source")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    st.markdown("### 📊 Statistics")
    total_docs = len(st.session_state.documents)
    indexed_docs = len(st.session_state.indexed_docs)
    st.metric("Documents Uploaded", total_docs)
    st.metric("Documents Indexed", indexed_docs)

    if st.session_state.documents:
        outcomes = {}
        for doc_data in st.session_state.documents.values():
            if "metadata" in doc_data:
                outcome = doc_data["metadata"].get("disposal_outcome", "Unknown")
                outcomes[outcome] = outcomes.get(outcome, 0) + 1

        if outcomes:
            st.markdown("### 📈 Outcome Distribution")
            for outcome, count in sorted(outcomes.items()):
                st.markdown(f"{get_outcome_badge(outcome)} × {count}", unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🗑️ Clear All Data", type="secondary"):
        st.session_state.documents = {}
        st.session_state.chat_history = []
        st.session_state.indexed_docs = set()
        clear_index()
        st.rerun()

    if st.button("💾 Export Results", type="secondary"):
        save_results()
        st.success("Results saved to `data/outputs/sample_results.json`")

    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.75rem; color: #888;">
        <strong>LexQuery v1.0</strong><br>
        Powered by Google Gemini, bge-m3 (Local)<br>
        © 2024 LexQuery Legal Intelligence
    </div>
    """, unsafe_allow_html=True)
