"""
Pydantic Schemas for Court Judgment Metadata.

Defines validated data models for structured extraction of Indian court
judgment metadata across English, Hindi, and Gujarati documents.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class DisposalOutcome(str, Enum):
    """Standardized disposal outcomes for Indian court judgments."""
    ALLOWED = "Allowed"
    DISMISSED = "Dismissed"
    DISPOSED = "Disposed"
    QUASHED = "Quashed"
    BAIL_GRANTED = "Bail Granted"
    BAIL_REJECTED = "Bail Rejected"
    PENDING = "Pending"
    UNKNOWN = "Unknown"


# Mapping of vernacular phrases to disposal outcomes
VERNACULAR_OUTCOME_MAP = {
    # Hindi phrases
    "अपील स्वीकार": DisposalOutcome.ALLOWED,
    "अपील मंजूर": DisposalOutcome.ALLOWED,
    "याचिका स्वीकार": DisposalOutcome.ALLOWED,
    "अपील खारिज": DisposalOutcome.DISMISSED,
    "याचिका खारिज": DisposalOutcome.DISMISSED,
    "खारिज की जाती है": DisposalOutcome.DISMISSED,
    "निस्तारित": DisposalOutcome.DISPOSED,
    "निरस्त": DisposalOutcome.QUASHED,
    "जमानत मंजूर": DisposalOutcome.BAIL_GRANTED,
    "जमानत स्वीकार": DisposalOutcome.BAIL_GRANTED,
    "जमानत अस्वीकार": DisposalOutcome.BAIL_REJECTED,
    "जमानत खारिज": DisposalOutcome.BAIL_REJECTED,
    "लंबित": DisposalOutcome.PENDING,

    # Gujarati phrases
    "અપીલ મંજૂર": DisposalOutcome.ALLOWED,
    "અરજી મંજૂર": DisposalOutcome.ALLOWED,
    "અપીલ નામંજૂર": DisposalOutcome.DISMISSED,
    "અરજી ફગાવવામાં": DisposalOutcome.DISMISSED,
    "આથી હુકમ કરવામાં આવે છે": DisposalOutcome.DISPOSED,
    "રદ કરવામાં આવે છે": DisposalOutcome.QUASHED,
    "જામીન મંજૂર": DisposalOutcome.BAIL_GRANTED,
    "જામીન નામંજૂર": DisposalOutcome.BAIL_REJECTED,

    # English phrases (case-insensitive matching done in code)
    "appeal allowed": DisposalOutcome.ALLOWED,
    "petition allowed": DisposalOutcome.ALLOWED,
    "appeal dismissed": DisposalOutcome.DISMISSED,
    "petition dismissed": DisposalOutcome.DISMISSED,
    "disposed of": DisposalOutcome.DISPOSED,
    "disposed off": DisposalOutcome.DISPOSED,
    "quashed": DisposalOutcome.QUASHED,
    "set aside": DisposalOutcome.QUASHED,
    "bail granted": DisposalOutcome.BAIL_GRANTED,
    "bail rejected": DisposalOutcome.BAIL_REJECTED,
    "bail refused": DisposalOutcome.BAIL_REJECTED,
}


class Parties(BaseModel):
    """Parties involved in the court case."""
    petitioner: List[str] = Field(
        default_factory=list,
        description="Names of petitioner(s) / appellant(s)"
    )
    respondent: List[str] = Field(
        default_factory=list,
        description="Names of respondent(s)"
    )


class CourtMetadata(BaseModel):
    """Structured metadata extracted from a court judgment document."""
    case_number: Optional[str] = Field(
        default="Unknown",
        description="Primary case number (e.g., 'Criminal Appeal No. 123/2024')"
    )
    cnr_number: Optional[str] = Field(
        default=None,
        description="CNR (Case Number Record) number if available"
    )
    court_name: Optional[str] = Field(
        default="Unknown",
        description="Name of the court (e.g., 'High Court of Gujarat')"
    )
    parties: Parties = Field(
        default_factory=Parties,
        description="Petitioner and respondent parties"
    )
    judge_names: List[str] = Field(
        default_factory=list,
        description="Names of presiding judge(s)"
    )
    decision_date: Optional[str] = Field(
        default=None,
        description="Date of judgment/order (DD/MM/YYYY or as found)"
    )
    disposal_outcome: DisposalOutcome = Field(
        default=DisposalOutcome.UNKNOWN,
        description="Standardized disposal outcome"
    )
    raw_outcome_verbatim: str = Field(
        default="",
        description="Verbatim text of the operative order/disposal from the document"
    )

    model_config = {"use_enum_values": True}


class DocumentMetadata(BaseModel):
    """Combined metadata for a processed document."""
    file_name: str
    file_path: str
    total_pages: int
    detected_language: str
    detected_scripts: List[str] = Field(default_factory=list)
    extraction_method: str  # "digital", "ocr", or "mixed"
    court_metadata: Optional[CourtMetadata] = None
    summary: Optional[str] = None
    processing_status: str = "pending"  # "pending", "processing", "completed", "error"
    error_message: Optional[str] = None
