"""
Structured Metadata Extractor using Google Gemini API.

Extracts court metadata from judgment preambles and disposition sections
using Gemini 1.5 Flash with JSON mode and Pydantic schema enforcement.
"""

import json
import logging
import os
from typing import List, Optional

from dotenv import load_dotenv

from src.structuring.schemas import (
    CourtMetadata,
    DisposalOutcome,
    Parties,
    VERNACULAR_OUTCOME_MAP,
)

load_dotenv()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gemini client setup
# ---------------------------------------------------------------------------

def _get_gemini_model():
    """Initialize and return the Gemini generative model."""
    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not set. Please set it in your .env file or environment."
        )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-flash-lite-latest")
    return model


# ---------------------------------------------------------------------------
# Text windowing functions
# ---------------------------------------------------------------------------

def extract_preamble(pages_text: List[str], n: int = 2) -> str:
    """Extract the preamble (first n pages) of a judgment."""
    preamble_pages = pages_text[:n]
    return "\n\n".join(preamble_pages)


def extract_disposition(pages_text: List[str], n: int = 2) -> str:
    """Extract the disposition/order section (last n pages) of a judgment."""
    disposition_pages = pages_text[-n:]
    return "\n\n".join(disposition_pages)


# ---------------------------------------------------------------------------
# Vernacular outcome mapping
# ---------------------------------------------------------------------------

def map_vernacular_outcome(raw_text: str) -> DisposalOutcome:
    """
    Map vernacular outcome phrases to standardized DisposalOutcome enum.
    Checks Hindi, Gujarati, and English phrases.
    """
    text_lower = raw_text.lower().strip()

    # Check exact and substring matches
    for phrase, outcome in VERNACULAR_OUTCOME_MAP.items():
        if phrase.lower() in text_lower or phrase in raw_text:
            return outcome

    return DisposalOutcome.UNKNOWN


# ---------------------------------------------------------------------------
# Gemini structured extraction
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """You are a legal metadata extraction expert specializing in Indian court judgments.
You must analyze the provided court judgment text (which may be in English, Hindi, Gujarati, or Kannada)
and extract structured metadata.

IMPORTANT RULES:
1. Extract ALL fields accurately from the text. Do not fabricate information.
2. For parties, list each petitioner and respondent separately.
3. For the disposal outcome, determine the final operative order and map it to one of:
   "Allowed", "Dismissed", "Disposed", "Quashed", "Bail Granted", "Bail Rejected", "Pending", or "Unknown".
4. Include the verbatim text of the operative order in raw_outcome_verbatim.
5. For dates, use the format as found in the document.
6. If a field cannot be determined, use null for optional fields or empty lists for list fields.

Return a valid JSON object with exactly these fields:
{{
    "case_number": "string - primary case number",
    "cnr_number": "string or null - CNR number if available",
    "court_name": "string - full court name",
    "parties": {{
        "petitioner": ["list of petitioner names"],
        "respondent": ["list of respondent names"]
    }},
    "judge_names": ["list of judge names"],
    "decision_date": "string or null - date of judgment",
    "disposal_outcome": "one of: Allowed, Dismissed, Disposed, Quashed, Bail Granted, Bail Rejected, Pending, Unknown",
    "raw_outcome_verbatim": "string - verbatim operative order text"
}}

--- JUDGMENT PREAMBLE ---
{preamble}

--- JUDGMENT DISPOSITION ---
{disposition}
"""


def extract_metadata(
    full_text: str,
    pages_text: Optional[List[str]] = None,
) -> CourtMetadata:
    """
    Extract structured court metadata from a judgment document using Gemini.

    Args:
        full_text: Complete text of the judgment document.
        pages_text: Optional list of per-page text strings.
                    If not provided, full_text is split into approximate pages.

    Returns:
        Validated CourtMetadata instance.
    """
    # Split into pages if not provided
    if pages_text is None:
        # Approximate page splitting by character count (~3000 chars per page)
        page_size = 3000
        pages_text = [
            full_text[i:i + page_size]
            for i in range(0, len(full_text), page_size)
        ]

    # Extract windowed sections
    preamble = extract_preamble(pages_text, n=2)
    disposition = extract_disposition(pages_text, n=2)

    # Build prompt
    prompt = EXTRACTION_PROMPT.format(
        preamble=preamble[:4000],  # Limit to avoid token overflow
        disposition=disposition[:4000],
    )

    # Call Gemini API
    model = _get_gemini_model()

    try:
        import google.generativeai as genai
        import time
        from google.api_core.exceptions import GoogleAPIError

        for attempt in range(3):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        temperature=0.1,
                    ),
                )
                raw_json = response.text.strip()
                
                # Simple check if output was empty
                if raw_json and "{" in raw_json:
                    break
                elif attempt < 2:
                    time.sleep(35)  # API asks to wait 30 seconds
            except Exception as e:
                if attempt == 2:
                    raise e
                time.sleep(35)

        # Parse response
        response_text = response.text.strip()

        # Clean up potential markdown code block wrapping
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1])

        raw_data = json.loads(response_text, strict=False)

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Gemini JSON response: {e}")
        logger.error(f"Raw response: {response_text[:500]}")
        raise ValueError(f"Gemini returned invalid JSON: {e}")
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        raise

    # Post-process and validate
    # Handle parties
    parties_data = raw_data.get("parties", {})
    if isinstance(parties_data, dict):
        parties = Parties(
            petitioner=parties_data.get("petitioner", []),
            respondent=parties_data.get("respondent", []),
        )
    else:
        parties = Parties()

    # Handle disposal outcome
    raw_outcome = raw_data.get("disposal_outcome", "Unknown")
    raw_verbatim = raw_data.get("raw_outcome_verbatim", "")

    # Try to map the outcome
    try:
        outcome = DisposalOutcome(raw_outcome)
    except ValueError:
        # Try vernacular mapping
        outcome = map_vernacular_outcome(raw_outcome)
        if outcome == DisposalOutcome.UNKNOWN and raw_verbatim:
            # Try mapping the verbatim text
            outcome = map_vernacular_outcome(raw_verbatim)

    # Build validated metadata
    metadata = CourtMetadata(
        case_number=raw_data.get("case_number", "Unknown"),
        cnr_number=raw_data.get("cnr_number"),
        court_name=raw_data.get("court_name", "Unknown"),
        parties=parties,
        judge_names=raw_data.get("judge_names", []),
        decision_date=raw_data.get("decision_date"),
        disposal_outcome=outcome,
        raw_outcome_verbatim=raw_verbatim,
    )

    return metadata


def extract_metadata_batch(
    documents: list[dict],
) -> list[CourtMetadata]:
    """
    Extract metadata from multiple documents.

    Args:
        documents: List of dicts with 'full_text' and optional 'pages_text' keys.

    Returns:
        List of CourtMetadata instances.
    """
    results = []
    for doc in documents:
        try:
            metadata = extract_metadata(
                full_text=doc["full_text"],
                pages_text=doc.get("pages_text"),
            )
            results.append(metadata)
        except Exception as e:
            logger.error(f"Failed to extract metadata: {e}")
            # Return a minimal metadata object on failure
            results.append(CourtMetadata(
                case_number="EXTRACTION_FAILED",
                court_name="Unknown",
                raw_outcome_verbatim=str(e),
            ))
    return results
