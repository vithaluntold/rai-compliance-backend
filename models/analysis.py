from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AnalysisStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    AWAITING_FRAMEWORK = "awaiting_framework_selection"


class ExportFormat(str, Enum):
    JSON = "JSON"
    EXCEL = "EXCEL"


class AnalysisType(str, Enum):
    COMPLIANCE = "COMPLIANCE"
    ADEQUACY = "ADEQUACY"
    RISK = "RISK"
    RECOMMENDATIONS = "RECOMMENDATIONS"


class AnalysisResult(BaseModel):
    id: str
    document_id: str
    result_type: AnalysisType
    content: Dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.now)
    status: AnalysisStatus = AnalysisStatus.PENDING
    error_message: Optional[str] = None

    class Config:
        from_attributes = True


class AnalysisRequest(BaseModel):
    document_id: str
    analysis_types: List[AnalysisType] = [AnalysisType.COMPLIANCE]
    options: Optional[Dict[str, Any]] = None


class AnalysisResponse(BaseModel):
    document_id: str
    results: List[AnalysisResult]
    overall_score: float
    compliance_score: float
    adequacy_score: float
    risk_score: float
    recommendations: List[str]
    created_at: datetime
    updated_at: datetime


class AnalysisProgress(BaseModel):
    document_id: str
    status: AnalysisStatus
    progress: float
    current_step: str
    message: str
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
