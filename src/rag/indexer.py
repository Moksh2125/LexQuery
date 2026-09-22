"""
RAG Indexer Module - Semantic chunking + bge-m3 embedding + ChromaDB storage.

Splits court judgment text into semantic chunks, vectorizes them using
the BAAI/bge-m3 multilingual model, and stores them in a persistent
ChromaDB collection.
"""

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHUNK_MIN_TOKENS = 400
CHUNK_MAX_TOKENS = 800
CHUNK_TARGET_TOKENS = 600
CHROMADB_PATH = str(Path(__file__).resolve().parent.parent.parent / "data" / "chromadb")
COLLECTION_NAME = "lexquery_judgments"

# ---------------------------------------------------------------------------
# Chunk model
# ---------------------------------------------------------------------------

class TextChunk(BaseModel):
    """A semantic chunk of text from a court judgment."""
    chunk_id: str
    text: str
    case_id: str
    court: str = ""
    language: str = ""
    page_no: int = 0
    chunk_index: int = 0
    token_count: int = 0


# ---------------------------------------------------------------------------
# Embedding model (lazy-loaded singleton)
# ---------------------------------------------------------------------------

_embedding_model = None


def _get_embedding_model():
    """Lazily load the bge-m3 sentence transformer model."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading BAAI/bge-m3 embedding model...")
        _embedding_model = SentenceTransformer("BAAI/bge-m3")
        logger.info("bge-m3 model loaded successfully.")
    return _embedding_model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts using bge-m3."""
    model = _get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return embeddings.tolist()


# ---------------------------------------------------------------------------
# Semantic chunking
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Rough token count estimation (words * 1.3 for multilingual)."""
    return int(len(text.split()) * 1.3)


def _split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs, preserving integrity."""
    # Split on double newlines or numbered paragraphs
    paragraphs = re.split(r'\n\s*\n|\n(?=\d+[\.\)]\s)', text)
    return [p.strip() for p in paragraphs if p.strip()]


def chunk_text(
    text: str,
    min_tokens: int = CHUNK_MIN_TOKENS,
    max_tokens: int = CHUNK_MAX_TOKENS,
    target_tokens: int = CHUNK_TARGET_TOKENS,
) -> List[str]:
    """
    Split text into semantic chunks of 600-800 tokens, preserving paragraph boundaries.

    Strategy:
    1. Split into paragraphs.
    2. Accumulate paragraphs until reaching target token count.
    3. If a single paragraph exceeds max_tokens, split it at sentence boundaries.
    """
    paragraphs = _split_into_paragraphs(text)
    chunks = []
    current_chunk = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _estimate_tokens(para)

        # If single paragraph exceeds max, split it by sentences
        if para_tokens > max_tokens:
            # Flush current chunk first
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_tokens = 0

            # Split large paragraph by sentences
            sentences = re.split(r'(?<=[.!?।])\s+', para)
            sent_chunk = []
            sent_tokens = 0

            for sent in sentences:
                sent_tok = _estimate_tokens(sent)
                if sent_tokens + sent_tok > max_tokens and sent_chunk:
                    chunks.append(" ".join(sent_chunk))
                    sent_chunk = []
                    sent_tokens = 0
                sent_chunk.append(sent)
                sent_tokens += sent_tok

            if sent_chunk:
                # Add remaining sentences to current_chunk for potential merging
                current_chunk.append(" ".join(sent_chunk))
                current_tokens += sent_tokens
            continue

        # Check if adding this paragraph would exceed max
        if current_tokens + para_tokens > max_tokens and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_tokens = 0

        current_chunk.append(para)
        current_tokens += para_tokens

        # If we've reached target, start a new chunk
        if current_tokens >= target_tokens:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_tokens = 0

    # Don't forget the last chunk
    if current_chunk:
        last_text = "\n\n".join(current_chunk)
        # If last chunk is too small, merge with previous
        if chunks and _estimate_tokens(last_text) < min_tokens:
            chunks[-1] = chunks[-1] + "\n\n" + last_text
        else:
            chunks.append(last_text)

    return chunks


# ---------------------------------------------------------------------------
# ChromaDB collection management
# ---------------------------------------------------------------------------

def _get_collection(persist_dir: str = CHROMADB_PATH) -> chromadb.Collection:
    """Get or create the ChromaDB collection."""
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def get_chromadb_client(persist_dir: str = CHROMADB_PATH):
    """Get the ChromaDB persistent client."""
    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)


# ---------------------------------------------------------------------------
# Document indexing
# ---------------------------------------------------------------------------

def index_document(
    case_id: str,
    text: str,
    metadata: Optional[Dict] = None,
    persist_dir: str = CHROMADB_PATH,
) -> List[TextChunk]:
    """
    Chunk, embed, and index a court judgment document into ChromaDB.

    Args:
        case_id: Unique identifier for the case/document.
        text: Full text of the judgment.
        metadata: Optional dict with 'court', 'language', 'page_no' etc.
        persist_dir: Path to ChromaDB persistent storage.

    Returns:
        List of TextChunk objects that were indexed.
    """
    if not text or not text.strip():
        logger.warning(f"Empty text for case {case_id}, skipping indexing.")
        return []

    metadata = metadata or {}
    court = metadata.get("court", "")
    language = metadata.get("language", "")

    # Chunk the text
    raw_chunks = chunk_text(text)
    logger.info(f"Case {case_id}: Split into {len(raw_chunks)} chunks")

    if not raw_chunks:
        return []

    # Create chunk objects
    chunks = []
    for i, chunk_text_content in enumerate(raw_chunks):
        chunk_id = hashlib.md5(f"{case_id}_{i}_{chunk_text_content[:100]}".encode()).hexdigest()
        chunk = TextChunk(
            chunk_id=chunk_id,
            text=chunk_text_content,
            case_id=case_id,
            court=court,
            language=language,
            page_no=metadata.get("page_no", 0),
            chunk_index=i,
            token_count=_estimate_tokens(chunk_text_content),
        )
        chunks.append(chunk)

    # Embed chunks
    texts_to_embed = [c.text for c in chunks]
    embeddings = embed_texts(texts_to_embed)

    # Store in ChromaDB
    collection = _get_collection(persist_dir)

    ids = [c.chunk_id for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [
        {
            "case_id": c.case_id,
            "court": c.court,
            "language": c.language,
            "page_no": c.page_no,
            "chunk_index": c.chunk_index,
            "token_count": c.token_count,
        }
        for c in chunks
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    logger.info(f"Indexed {len(chunks)} chunks for case {case_id}")
    return chunks


def clear_index(persist_dir: str = CHROMADB_PATH) -> None:
    """Clear the entire ChromaDB collection."""
    client = get_chromadb_client(persist_dir)
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info(f"Cleared collection '{COLLECTION_NAME}'")
    except Exception:
        logger.info(f"Collection '{COLLECTION_NAME}' does not exist, nothing to clear")
