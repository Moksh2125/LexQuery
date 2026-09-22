"""
Legal Judgment Summarizer Module.

Generates concise, neutral 4-6 sentence English summaries of court
judgments using the Gemini API. Summaries cover jurisdiction, factual
dispute, legal reasoning, and operative order.
"""

import logging
import os
import re
from typing import Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Summarization prompt
# ---------------------------------------------------------------------------

SUMMARY_PROMPT = """You are a legal summarization expert specializing in Indian court judgments.
Summarize the following court judgment.

STRICT RULES:
1. The summary MUST be EXACTLY 4 to 6 sentences long. No more, no less.
2. Do NOT output any metadata, headers, or bullet points. Write ONLY the summary paragraph in English.
3. Write in clear, professional English even if the source is in Hindi, Gujarati, or Kannada.
4. The summary MUST cover these four aspects in order:
   - Sentence 1: Forum/jurisdiction (which court, case type, case number)
   - Sentence 2: Factual dispute (what the case is about)
   - Sentences 3-4: Legal reasoning (key legal arguments and provisions cited)
   - Sentence 5-6: Operative order (final decision/disposal)
5. Be neutral and objective. Do not express opinions.
6. Preserve the legal outcome and key facts faithfully.
7. Include specific legal provisions (e.g., IPC sections, CrPC sections) if mentioned.

{metadata_context}

--- JUDGMENT TEXT ---
{judgment_text}

Generate the summary (4-6 sentences only):"""


# ---------------------------------------------------------------------------
# Main summarization function
# ---------------------------------------------------------------------------

def summarize_judgment(
    text: str,
    metadata: Optional[dict] = None,
    max_input_chars: int = 12000,
) -> str:
    """
    Generate a 4-6 sentence neutral English summary of a court judgment.

    Args:
        text: Full text of the judgment (can be in any supported language).
        metadata: Optional dict with court_name, case_number, etc.
        max_input_chars: Max characters to send to Gemini (truncated from both ends).

    Returns:
        Summary string (4-6 sentences in English).
    """
    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-flash-lite-latest")

    # Build metadata context if available
    metadata_context = ""
    if metadata:
        parts = []
        if metadata.get("case_number"):
            parts.append(f"Case Number: {metadata['case_number']}")
        if metadata.get("court_name"):
            parts.append(f"Court: {metadata['court_name']}")
        if metadata.get("decision_date"):
            parts.append(f"Date: {metadata['decision_date']}")
        if parts:
            metadata_context = "KNOWN METADATA:\n" + "\n".join(parts)

    # Truncate text intelligently — keep beginning and end
    if len(text) > max_input_chars:
        half = max_input_chars // 2
        truncated = text[:half] + "\n\n[... middle portion omitted for brevity ...]\n\n" + text[-half:]
    else:
        truncated = text

    # Build prompt
    prompt = SUMMARY_PROMPT.format(
        metadata_context=metadata_context,
        judgment_text=truncated,
    )

    import time
    from google.api_core.exceptions import GoogleAPIError

    # Call Gemini with retry
    for attempt in range(3):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=512,
                ),
            )
            summary = response.text.strip()
            
            # Simple check if output was completely cut off
            if len(summary) > 50 and summary.endswith((".", "!", "?")):
                break
            elif attempt < 2:
                time.sleep(35)  # wait for rate limits
        except Exception as e:
            if attempt == 2:
                raise e
            time.sleep(35)

    # Post-process: validate sentence count
    summary = _enforce_sentence_count(summary)

    return summary


def _enforce_sentence_count(summary: str, min_sentences: int = 4, max_sentences: int = 6) -> str:
    """
    Ensure the summary contains exactly 4-6 sentences.
    If too many, truncate. If too few, return as-is with a warning.
    """
    # Split by sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', summary.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) > max_sentences:
        # Keep first max_sentences
        sentences = sentences[:max_sentences]
        # Ensure last sentence ends with period
        if not sentences[-1].endswith('.'):
            sentences[-1] = sentences[-1].rstrip() + '.'
        return ' '.join(sentences)

    if len(sentences) < min_sentences:
        logger.warning(
            f"Summary has only {len(sentences)} sentences (expected {min_sentences}-{max_sentences}). "
            f"Returning as-is."
        )

    return ' '.join(sentences)


def count_sentences(text: str) -> int:
    """Count the number of sentences in text."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return len([s for s in sentences if s.strip()])
