# ⚖️ LexQuery — Multilingual Legal Intelligence Platform

A comprehensive AI-powered platform for processing Indian court judgments across multiple languages (English, Hindi, Gujarati). LexQuery ingests, structures, indexes, and enables intelligent Q&A over court documents.

## 🚀 Features

- **📄 Dual-Path PDF Extraction** — PyMuPDF for digital PDFs, Tesseract OCR for scanned documents
- **🌐 Multilingual Support** — English, Hindi, Gujarati, and Kannada court judgments
- **🏛️ Structured Metadata Extraction** — Case number, court, parties, judges, outcome via Gemini AI
- **🔍 Cross-Lingual Q&A** — Ask questions in English, get answers from any language document
- **📝 Legal Summarization** — 4-6 sentence neutral English summaries of each judgment
- **🛡️ Faithfulness Guardrails** — NLI-based entailment checking + entity cross-validation
- **💻 Interactive Streamlit UI** — Upload, triage, and Q&A interface

## 📦 Prerequisites

- Python 3.10+
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) with `eng`, `hin`, `guj` language packs
- [Google Gemini API Key](https://aistudio.google.com/app/apikey)

## 🔧 Installation

```bash
# Clone / navigate to project
cd lexquery

# Install Python dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY and TESSERACT_CMD path
```

## ⚡ Quick Start

```bash
# Run the Streamlit application
streamlit run app.py
```

Then:
1. Open `http://localhost:8501` in your browser
2. Go to **Upload & Ingest** tab — upload your court judgment PDFs
3. Click **Process All Documents** — watch the progress bar
4. Switch to **Triage Dashboard** — view structured metadata cards
5. Go to **Cross-Lingual Q&A** — ask questions about the documents

## 🏗️ Architecture

```
lexquery/
├── app.py                     # Streamlit frontend
├── src/
│   ├── ingestion/             # PDF extraction + language detection
│   ├── structuring/           # Pydantic schemas + Gemini extraction
│   ├── rag/                   # Chunking, indexing, retrieval, summarization
│   └── guardrails/            # NLI verification + entity checking
├── tests/                     # Phase-wise test suites
├── data/
│   ├── sample_judgments/      # Input PDFs
│   └── outputs/               # Structured results (JSON)
└── design_doc.md              # Architecture & design decisions
```

## 🧪 Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run by phase
python -m pytest tests/test_phase1_ingestion.py -v     # PDF + language
python -m pytest tests/test_phase2_structuring.py -v   # Schemas + Gemini
python -m pytest tests/test_phase3_rag_summary.py -v   # RAG + summarization
python -m pytest tests/test_phase4_guardrails.py -v    # Faithfulness
python -m pytest tests/test_phase5_e2e.py -v           # End-to-end
```

## 🛠️ Tech Stack

| Component        | Technology                     |
|------------------|--------------------------------|
| LLM              | Google Gemini 1.5 Flash        |
| PDF Extraction   | PyMuPDF (fitz)                 |
| OCR              | Tesseract (pytesseract)        |
| Language Detect   | FastText (lid.176.bin)         |
| Embeddings       | BAAI/bge-m3                    |
| Vector Store     | ChromaDB (persistent)          |
| Data Validation  | Pydantic v2                    |
| Guardrails       | cross-encoder/nli-deberta-v3-small |
| Frontend         | Streamlit                      |

## 📄 License

This project is built with open-source tools for educational and assessment purposes.
