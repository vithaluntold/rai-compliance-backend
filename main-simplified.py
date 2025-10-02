import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

# Load environment variables FIRST
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import asyncio
import json
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Create required directories
required_dirs = [
    "uploads", "sessions", "analysis_results", 
    "checklist_data", "vector_indices", "logs", "data"
]

for dir_name in required_dirs:
    try:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created/verified directory: {dir_name}")
    except Exception as e:
        logger.warning(f"Could not create directory {dir_name}: {e}")

# Pydantic models for API
class DocumentUploadResponse(BaseModel):
    document_id: str
    status: str
    message: str

class AnalysisResponse(BaseModel):
    status: str
    document_id: str
    progress: int
    message: str

class ChecklistItem(BaseModel):
    question: str
    reference: str
    status: str
    explanation: str = ""
    evidence: str = ""

class ChecklistSection(BaseModel):
    title: str
    items: List[ChecklistItem]

# Global state for mock analysis
analysis_state: Dict[str, Dict] = {}

async def mock_document_analysis(document_id: str):
    """Mock document analysis that simulates the real process"""
    await asyncio.sleep(2)  # Simulate processing time
    
    # Mock analysis results
    mock_results = {
        "status": "COMPLETED",
        "sections": [
            {
                "title": "Financial Reporting Standards",
                "items": [
                    {
                        "question": "Are financial statements prepared in accordance with applicable accounting standards?",
                        "reference": "IAS 1.15",
                        "status": "COMPLIANT",
                        "explanation": "Financial statements are properly prepared according to IAS standards.",
                        "evidence": "Statement of financial position follows IAS 1 presentation requirements."
                    },
                    {
                        "question": "Are all material items disclosed appropriately?",
                        "reference": "IAS 1.29",
                        "status": "COMPLIANT", 
                        "explanation": "All material items are properly disclosed in the notes.",
                        "evidence": "Notes 1-15 provide comprehensive disclosures of material items."
                    }
                ]
            },
            {
                "title": "Revenue Recognition",
                "items": [
                    {
                        "question": "Is revenue recognized in accordance with IFRS 15?",
                        "reference": "IFRS 15.1",
                        "status": "COMPLIANT",
                        "explanation": "Revenue recognition follows the 5-step model of IFRS 15.",
                        "evidence": "Revenue recognition policy described in Note 2.3."
                    }
                ]
            }
        ],
        "overall_score": 95,
        "compliance_score": 98,
        "adequacy_score": 92,
        "metadata": {
            "company_name": {"value": "Sample Corporation Ltd"},
            "nature_of_business": {"value": "Financial Services"},
            "operational_demographics": {"value": "Multi-national operations"},
            "_overall_status": "COMPLETED"
        }
    }
    
    analysis_state[document_id] = mock_results
    return mock_results

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("RAi Compliance Engine starting up...")
    logger.info("✅ All systems initialized successfully")
    yield
    # Shutdown
    logger.info("Shutting down FastAPI application")

# Initialize FastAPI app
app = FastAPI(
    title="RAi Compliance Engine",
    description="Advanced AI-powered financial compliance analysis platform",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Health check endpoint
@app.get("/")
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "RAi Compliance Engine",
        "version": "1.0.0",
        "timestamp": "2025-09-18T00:00:00Z"
    }

# Document upload endpoint
@app.post("/api/v1/analysis/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Generate document ID
        import uuid
        document_id = str(uuid.uuid4())
        
        # Save file
        file_path = f"uploads/{document_id}_{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Start mock analysis
        asyncio.create_task(mock_document_analysis(document_id))
        
        # Initialize analysis state
        analysis_state[document_id] = {
            "status": "PROCESSING",
            "progress": 10,
            "message": "Document uploaded successfully. Starting analysis..."
        }
        
        logger.info(f"Document uploaded: {document_id}")
        
        return DocumentUploadResponse(
            document_id=document_id,
            status="uploaded",
            message="Document uploaded successfully. Analysis started."
        )
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Analysis status endpoint
@app.get("/api/v1/analysis/status/{document_id}")
async def get_analysis_status(document_id: str):
    try:
        if document_id not in analysis_state:
            raise HTTPException(status_code=404, detail="Document not found")
        
        state = analysis_state[document_id]
        
        return AnalysisResponse(
            status=state.get("status", "PROCESSING"),
            document_id=document_id,
            progress=state.get("progress", 0),
            message=state.get("message", "Processing...")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Analysis results endpoint
@app.get("/api/v1/analysis/results/{document_id}")
async def get_analysis_results(document_id: str):
    try:
        if document_id not in analysis_state:
            raise HTTPException(status_code=404, detail="Document not found")
        
        state = analysis_state[document_id]
        
        if state.get("status") != "COMPLETED":
            raise HTTPException(status_code=202, detail="Analysis still in progress")
        
        return state
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Results error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Documents list endpoint
@app.get("/api/v1/documents")
async def list_documents():
    try:
        documents = []
        for doc_id, state in analysis_state.items():
            documents.append({
                "id": doc_id,
                "status": state.get("status", "UNKNOWN"),
                "created_at": "2025-09-18T00:00:00Z",  # Mock timestamp
                "filename": f"document_{doc_id[:8]}.pdf"
            })
        
        return {"documents": documents}
        
    except Exception as e:
        logger.error(f"List documents error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "path": str(request.url.path),
            "method": request.method,
        },
    )

# Export app for Vercel
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting RAi Compliance Engine server...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)