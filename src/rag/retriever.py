"""
Cross-Lingual Retriever Module.

Accepts English natural-language queries and retrieves relevant chunks
from the ChromaDB collection, even if the source text is in Hindi,
Gujarati, or Kannada. Uses bge-m3 for cross-lingual dense search
and Gemini for generating grounded English answers.
"""

import logging
import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.rag.indexer import (
    CHROMADB_PATH,
    COLLECTION_NAME,
    embed_texts,
    get_chromadb_client,
)

load_dotenv()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

class RetrievedChunk(BaseModel):
    """A retrieved chunk from the vector store."""
    chunk_id: str
    text: str
    case_id: str = ""
    court: str = ""
    language: str = ""
    page_no: int = 0
    chunk_index: int = 0
    similarity_score: float = 0.0


class AnswerResult(BaseModel):
    """Result of a Q&A query over the document corpus."""
    query: str
    answer: str
    source_chunks: List[RetrievedChunk] = Field(default_factory=list)
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Retrieval functions
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    top_k: int = 5,
    persist_dir: str = CHROMADB_PATH,
    filter_metadata: Optional[Dict] = None,
) -> List[RetrievedChunk]:
    """
    Retrieve the most relevant chunks for an English query.

    Performs cross-lingual dense search using bge-m3 embeddings,
    returning chunks regardless of their source language.

    Args:
        query: English natural-language query.
        top_k: Number of top results to return.
        persist_dir: Path to ChromaDB storage.
        filter_metadata: Optional metadata filter dict.

    Returns:
        List of RetrievedChunk objects ranked by relevance.
    """
    # Embed the query
    query_embedding = embed_texts([query])[0]

    # Query ChromaDB
    client = get_chromadb_client(persist_dir)
    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
        )
    except Exception:
        logger.warning(f"Collection '{COLLECTION_NAME}' not found. Return empty results.")
        return []

    query_params = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }

    if filter_metadata:
        query_params["where"] = filter_metadata

    results = collection.query(**query_params)

    # Parse results
    chunks = []
    if results and results["ids"] and results["ids"][0]:
        for i, chunk_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results["distances"] else 1.0
            # Convert cosine distance to similarity score
            similarity = 1.0 - distance

            chunk = RetrievedChunk(
                chunk_id=chunk_id,
                text=results["documents"][0][i],
                case_id=metadata.get("case_id", ""),
                court=metadata.get("court", ""),
                language=metadata.get("language", ""),
                page_no=metadata.get("page_no", 0),
                chunk_index=metadata.get("chunk_index", 0),
                similarity_score=round(similarity, 4),
            )
            chunks.append(chunk)

    return chunks


# ---------------------------------------------------------------------------
# Q&A with Gemini
# ---------------------------------------------------------------------------

QA_PROMPT = """You are a legal research assistant specializing in Indian court judgments.
Answer the user's question based ONLY on the provided source excerpts.

RULES:
1. Base your answer ONLY on the provided source excerpts. Do not use external knowledge.
2. Cite the source case ID and page number when making claims.
3. If the answer cannot be found in the sources, say "The provided documents do not contain information to answer this question."
4. Answer in clear, professional English even if sources are in Hindi, Gujarati, or Kannada.
5. Be precise and factual. Do not speculate.

USER QUESTION: {query}

SOURCE EXCERPTS:
{sources}

Provide a clear, well-structured answer:"""


def answer_question(
    query: str,
    top_k: int = 5,
    persist_dir: str = CHROMADB_PATH,
) -> AnswerResult:
    """
    Answer an English question using cross-lingual RAG.

    Retrieves relevant chunks from the vector store and generates
    a grounded English answer using Gemini.

    Args:
        query: English natural-language question.
        top_k: Number of chunks to retrieve.
        persist_dir: Path to ChromaDB storage.

    Returns:
        AnswerResult with answer and source chunks.
    """
    # Retrieve relevant chunks
    chunks = retrieve(query, top_k=top_k, persist_dir=persist_dir)

    if not chunks:
        return AnswerResult(
            query=query,
            answer="No documents have been indexed yet. Please upload and process documents first.",
            source_chunks=[],
            confidence=0.0,
        )

    # Format sources for the prompt
    sources_text = ""
    for i, chunk in enumerate(chunks, 1):
        sources_text += f"\n--- Source {i} (Case: {chunk.case_id}, Language: {chunk.language}, Page: {chunk.page_no}) ---\n"
        sources_text += chunk.text[:1500]  # Limit per chunk to manage token count
        sources_text += "\n"

    # Build prompt
    prompt = QA_PROMPT.format(query=query, sources=sources_text)

    # Call Gemini
    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-flash-lite-latest")

    import time
    for attempt in range(3):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=1024,
                ),
            )
            answer = response.text.strip()
            if answer:
                break
            elif attempt < 2:
                time.sleep(35)
        except Exception as e:
            if attempt == 2:
                raise e
            time.sleep(35)

    answer_text = response.text.strip()

    # Calculate average confidence from retrieval scores
    avg_confidence = sum(c.similarity_score for c in chunks) / len(chunks) if chunks else 0.0

    return AnswerResult(
        query=query,
        answer=answer_text,
        source_chunks=chunks,
        confidence=round(avg_confidence, 4),
    )
