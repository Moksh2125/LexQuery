"""
Language Detection Module - FastText-based multilingual language identification.

Detects dominant and secondary languages in court judgment text,
supporting English, Hindi, Gujarati, and other Indian languages.
"""

import logging
import os
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

class LanguageResult(BaseModel):
    """Language detection result for a text sample."""
    dominant_language: str = Field(..., description="ISO 639-1 code of dominant language")
    dominant_language_name: str = Field(..., description="Human-readable language name")
    dominant_confidence: float = Field(..., description="Confidence score 0-1")
    secondary_languages: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of {language_code: confidence} for secondary languages"
    )
    detected_scripts: List[str] = Field(
        default_factory=list,
        description="Scripts detected in the text (Latin, Devanagari, Gujarati, etc.)"
    )


# ---------------------------------------------------------------------------
# Language code mappings
# ---------------------------------------------------------------------------

LANG_CODE_TO_NAME = {
    "en": "English",
    "hi": "Hindi",
    "gu": "Gujarati",
    "mr": "Marathi",
    "kn": "Kannada",
    "ta": "Tamil",
    "te": "Telugu",
    "bn": "Bengali",
    "pa": "Punjabi",
    "ur": "Urdu",
    "ml": "Malayalam",
    "or": "Odia",
    "sa": "Sanskrit",
}

# Unicode block ranges for script detection
SCRIPT_RANGES = {
    "Devanagari": (0x0900, 0x097F),
    "Gujarati": (0x0A80, 0x0AFF),
    "Latin": (0x0041, 0x024F),
    "Kannada": (0x0C80, 0x0CFF),
    "Tamil": (0x0B80, 0x0BFF),
    "Telugu": (0x0C00, 0x0C7F),
    "Bengali": (0x0980, 0x09FF),
    "Gurmukhi": (0x0A00, 0x0A7F),
    "Malayalam": (0x0D00, 0x0D7F),
    "Odia": (0x0B00, 0x0B7F),
}


# ---------------------------------------------------------------------------
# Script detection (rule-based fallback)
# ---------------------------------------------------------------------------

def detect_scripts(text: str) -> List[str]:
    """Detect Unicode scripts present in the text using character ranges."""
    script_counts: Dict[str, int] = {}

    for char in text:
        code_point = ord(char)
        for script_name, (start, end) in SCRIPT_RANGES.items():
            if start <= code_point <= end:
                script_counts[script_name] = script_counts.get(script_name, 0) + 1
                break

    # Return scripts with significant presence (> 5% of detected chars)
    total = sum(script_counts.values())
    if total == 0:
        return ["Unknown"]

    return [
        script for script, count in sorted(script_counts.items(), key=lambda x: -x[1])
        if count / total > 0.05
    ]


# ---------------------------------------------------------------------------
# FastText model management
# ---------------------------------------------------------------------------

MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "models"
MODEL_PATH = MODEL_DIR / "lid.176.bin"

_ft_model = None


def _download_model() -> Path:
    """Download the FastText language identification model if not present."""
    if MODEL_PATH.exists():
        logger.info(f"FastText model already cached at {MODEL_PATH}")
        return MODEL_PATH

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading FastText lid.176.bin to {MODEL_PATH}...")

    urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
    logger.info("FastText model download complete.")
    return MODEL_PATH


def _get_model():
    """Get or lazily initialize the FastText model."""
    global _ft_model
    if _ft_model is None:
        import fasttext

        model_path = _download_model()
        # Suppress FastText warnings about deprecated loading
        _ft_model = fasttext.load_model(str(model_path))
    return _ft_model


# ---------------------------------------------------------------------------
# Main detection function
# ---------------------------------------------------------------------------

def detect_language(text: str, top_k: int = 3) -> LanguageResult:
    """
    Detect the dominant and secondary languages in the given text.

    Args:
        text: Input text to analyze.
        top_k: Number of top language predictions to return.

    Returns:
        LanguageResult with dominant language, secondaries, and scripts.
    """
    if not text or not text.strip():
        return LanguageResult(
            dominant_language="unknown",
            dominant_language_name="Unknown",
            dominant_confidence=0.0,
            secondary_languages=[],
            detected_scripts=[],
        )

    # Clean text for FastText (single line, no newlines)
    clean_text = " ".join(text.split())

    # Truncate to first 5000 chars for efficiency
    if len(clean_text) > 5000:
        clean_text = clean_text[:5000]

    # Kruti Dev / Legacy Hindi heuristic detection
    # Legacy fonts map Hindi to English ASCII. FastText detects these as "English".
    kruti_dev_keywords = ["U;k;k/kh'k", "izdj.k", "vkns'k", "U;k;ky;", "jkT;", "nhokuh", "ewy", "la[;k"]
    if any(keyword in clean_text for keyword in kruti_dev_keywords):
        return LanguageResult(
            dominant_language="hi",
            dominant_language_name="Hindi",
            dominant_confidence=0.99,
            secondary_languages=[],
            detected_scripts=["Latin (Kruti Dev Legacy)"],
        )

    try:
        model = _get_model()
        predictions = model.predict(clean_text, k=top_k)

        labels = predictions[0]  # e.g., ['__label__en', '__label__hi']
        scores = predictions[1]  # e.g., [0.95, 0.03]

        # Parse dominant language
        dominant_code = labels[0].replace("__label__", "")
        dominant_name = LANG_CODE_TO_NAME.get(dominant_code, dominant_code.upper())
        dominant_conf = float(scores[0])

        # Parse secondary languages
        secondaries = []
        for i in range(1, len(labels)):
            code = labels[i].replace("__label__", "")
            conf = float(scores[i])
            if conf > 0.01:  # Only include if > 1% confidence
                secondaries.append({
                    "code": code,
                    "name": LANG_CODE_TO_NAME.get(code, code.upper()),
                    "confidence": round(conf, 4),
                })

    except Exception as e:
        logger.error(f"FastText detection failed: {e}. Falling back to script detection.")
        # Fallback: use script detection to infer language
        scripts = detect_scripts(text)
        if "Devanagari" in scripts:
            dominant_code, dominant_name = "hi", "Hindi"
        elif "Gujarati" in scripts:
            dominant_code, dominant_name = "gu", "Gujarati"
        elif "Kannada" in scripts:
            dominant_code, dominant_name = "kn", "Kannada"
        elif "Latin" in scripts:
            dominant_code, dominant_name = "en", "English"
        else:
            dominant_code, dominant_name = "unknown", "Unknown"

        return LanguageResult(
            dominant_language=dominant_code,
            dominant_language_name=dominant_name,
            dominant_confidence=0.5,  # Low confidence for fallback
            secondary_languages=[],
            detected_scripts=scripts,
        )

    # Detect scripts independently
    scripts = detect_scripts(text)

    return LanguageResult(
        dominant_language=dominant_code,
        dominant_language_name=dominant_name,
        dominant_confidence=round(dominant_conf, 4),
        secondary_languages=secondaries,
        detected_scripts=scripts,
    )


def detect_language_simple(text: str) -> str:
    """
    Simple convenience function that returns just the dominant language name.
    """
    result = detect_language(text)
    return result.dominant_language_name
