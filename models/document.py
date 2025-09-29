from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel


class DocumentStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class DocumentMetadata(BaseModel):
    title: str
    author: Optional[str] = "Unknown"
    creation_date: str
    modification_date: str
    page_count: int
    file_size: int
    file_hash: str
    company_name: Optional[str] = None
    industry: Optional[str] = None
    geography: Optional[str] = None
    reporting_period: Optional[str] = None
    document_type: Optional[str] = None
    language: Optional[str] = None

    class Config:
        from_attributes = True


class Document(BaseModel):
    id: str
    filename: str
    uploaded_at: datetime
    analysis_status: DocumentStatus
    metadata: Optional[DocumentMetadata] = None
    file_hash: str

    class Config:
        from_attributes = True


class DocumentCreate(BaseModel):
    filename: str
    file_hash: str
    metadata: Optional[DocumentMetadata] = None


class DocumentUpdate(BaseModel):
    analysis_status: Optional[DocumentStatus] = None
    metadata: Optional[DocumentMetadata] = None


class DocumentResponse(BaseModel):
    id: str
    filename: str
    uploaded_at: datetime
    analysis_status: DocumentStatus
    metadata: Optional[DocumentMetadata] = None
    file_hash: str
    analysis_results: Optional[Dict[str, Any]] = None
