# LexQuery Design Document
**LexQuery** is a Multilingual Legal Intelligence Platform designed to parse, structure, index, and query Indian court judgments across English, Hindi, and Gujarati.

## 1. Architecture Overview
The platform processes court judgments in five phases:
1.  **Ingestion & OCR (Dual-Path):** Using PyMuPDF for digitally typed PDFs and Tesseract for scanned artifacts.
2.  **Structuring & Extraction:** Utilizing Gemini's structured output capabilities via Pydantic schemas.
3.  **Cross-Lingual RAG:** Leveraging the `BAAI/bge-m3` embedding model for semantic chunking and ChromaDB for retrieval.
4.  **Guardrails (LLM-as-a-Judge):** Validating the faithfulness of the generated answers using strict LLM prompts and entity extraction via regex.
5.  **User Interface:** A Streamlit frontend for triage and Q&A interaction.

## 2. Engineering Tradeoffs & Architectural Pivots
During Phase 4 (Hallucination Detection), a critical hardware limitation was encountered. The target local environment possesses **8 GB of total RAM**, with less than 1 GB of free memory available during runtime.
*   **Original Plan:** Deploy `cross-encoder/nli-deberta-v3-small` locally via PyTorch for Natural Language Inference (NLI).
*   **Failure Mode:** PyTorch triggered a `Windows fatal exception: access violation` (segmentation fault) when attempting to load the models into RAM alongside `bge-m3`.
*   **The Pivot (LLM-as-a-Judge):** To bypass this strict hardware boundary while fulfilling the assessment's hallucination-detection requirement, the architecture was pivoted to an **LLM-as-a-Judge** pattern. The Gemini API (`gemini-flash-latest`) is now tasked with evaluating strict NLI entailment between the retrieved source text and the generated claim, responding purely with `FAITHFUL` or `UNSUPPORTED`.
*   **Entity Verification:** A pure Python `re` (Regex) module was implemented to cross-verify hard entities (like Sections, Acts, Dates, and 16-digit CNR numbers) against the source chunk to act as a secondary fallback against LLM hallucinations. This reduces memory footprint to near-zero.

## 3. Handling Legal Edge Cases
*   **Mixed Scripts (Code-Mixing):** The `langdetect` library captures dominant language probabilities, and the `bge-m3` model natively handles semantic alignment across English, Hindi, and Gujarati without requiring pre-translation.
*   **Standardizing Outcomes:** Diverse vernacular disposal outcomes (e.g., "zamin manjur", "bail granted", "application allowed") are piped through an LLM mapping layer enforcing a strict Enum schema (`GRANTED`, `REJECTED`, `DISPOSED`, `PENDING`). This ensures structured metadata is fully uniform regardless of the origin language.
