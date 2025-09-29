from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChecklistStatus(str, Enum):
    PENDING = "PENDING"
    YES = "YES"
    NO = "NO"
    NA = "NA"


class AdequacyLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ChecklistSection(BaseModel):
    id: str
    title: str
    description: Optional[str] = None
    items: List[Dict[str, Any]]
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    class Config:
        from_attributes = True


class ChecklistItem(BaseModel):
    id: str
    document_id: str
    section: str
    question: str
    reference: Optional[str] = None
    status: ChecklistStatus = ChecklistStatus.PENDING
    adequacy: AdequacyLevel = AdequacyLevel.LOW
    confidence: float = 0.0
    explanation: Optional[str] = None
    evidence: Optional[str] = None
    user_comment: Optional[str] = None
    suggestion: Optional[str] = None
    resolved: bool = False
    deleted: bool = False
    deletion_comment: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    class Config:
        from_attributes = True


class ChecklistItemCreate(BaseModel):
    section: str
    question: str
    reference: Optional[str] = None
    status: ChecklistStatus = ChecklistStatus.PENDING
    adequacy: AdequacyLevel = AdequacyLevel.LOW
    confidence: float = 0.0
    explanation: Optional[str] = None
    evidence: Optional[str] = None


class ChecklistItemUpdate(BaseModel):
    status: Optional[ChecklistStatus] = None
    adequacy: Optional[AdequacyLevel] = None
    confidence: Optional[float] = None
    explanation: Optional[str] = None
    evidence: Optional[str] = None
    user_comment: Optional[str] = None
    resolved: Optional[bool] = None
    deleted: Optional[bool] = None
    deletion_comment: Optional[str] = None


class ChecklistResponse(BaseModel):
    document_id: str
    items: List[ChecklistItem]
    overall_score: float
    compliance_score: float
    adequacy_score: float
    created_at: datetime
    updated_at: datetime
