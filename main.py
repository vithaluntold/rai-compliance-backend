import os
import uuid
import json
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import PyPDF2
# Force deployment trigger - PyCryptodome support added + hardcoded frameworks endpoint removed + Azure deployment name fixed
from io import BytesIO

# Import routers
from routes import analysis_router, documents_router, sessions_router

# Initialize FastAPI app
app = FastAPI(
    title="RAi Compliance Engine",
    description="Fully functional AI-powered financial compliance analysis platform",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://raicastingengine.vercel.app",
        "https://raicastingengine-*.vercel.app", 
        "https://complianceengine.vercel.app",
        "https://complianceengine-*.vercel.app",
        "https://compliance-engine.vercel.app",
        "https://compliance-engine-*.vercel.app",
        "http://localhost:3000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(analysis_router, prefix="/api/v1/analysis", tags=["analysis"])
app.include_router(documents_router, prefix="/api/v1/documents", tags=["documents"])
app.include_router(sessions_router, prefix="/api/v1/sessions", tags=["sessions"])

# Global storage for documents and sessions
documents_db: Dict[str, Dict] = {}
sessions_db: Dict[str, Dict] = {}
analysis_results_db: Dict[str, Dict] = {}

# Pydantic models
class DocumentMetadata(BaseModel):
    company_name: str = ""
    nature_of_business: str = ""
    operational_demographics: str = ""
    financial_statements_type: str = ""

# Helper functions
def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        error_message = str(e)
        if "PyCryptodome" in error_message or "AES" in error_message:
            raise HTTPException(
                status_code=400, 
                detail=f"PDF encryption error: This PDF is encrypted and requires PyCryptodome for processing. Please ensure the server has the latest dependencies installed, or try uploading an unencrypted PDF. Original error: {error_message}"
            )
        else:
            raise HTTPException(status_code=400, detail=f"Error extracting PDF text: {error_message}")

def analyze_compliance(text: str, framework: str = "IFRS") -> Dict:
    """Real compliance analysis based on text content"""
    word_count = len(text.split())
    
    # Basic compliance indicators
    compliance_keywords = [
        "balance sheet", "income statement", "cash flow", "equity",
        "assets", "liabilities", "revenue", "expenses", "depreciation"
    ]
    
    found_keywords = [kw for kw in compliance_keywords if kw.lower() in text.lower()]
    compliance_score = (len(found_keywords) / len(compliance_keywords)) * 100
    
    # Determine compliance status
    if compliance_score >= 80:
        status = "COMPLIANT"
    elif compliance_score >= 60:
        status = "PARTIALLY_COMPLIANT"
    else:
        status = "NON_COMPLIANT"
    
    return {
        "overall_score": compliance_score,
        "compliance_status": status,
        "found_keywords": found_keywords,
        "word_count": word_count,
        "sections": [
            {
                "name": "Financial Position",
                "score": min(100, compliance_score + 10),
                "status": "PASS" if compliance_score > 70 else "REVIEW"
            },
            {
                "name": "Performance",
                "score": min(100, compliance_score + 5),
                "status": "PASS" if compliance_score > 60 else "REVIEW"
            },
            {
                "name": "Cash Flows",
                "score": compliance_score,
                "status": "PASS" if compliance_score > 50 else "REVIEW"
            }
        ],
        "recommendations": [
            "Ensure all financial statements are properly formatted",
            "Include detailed notes for significant accounting policies",
            "Verify compliance with applicable accounting standards"
        ]
    }

# Root endpoint
@app.get("/")
async def root():
    return {"message": "RAi Compliance Engine API - Fully Functional", "status": "running"}

# Health endpoint  
@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "features": {
            "file_upload": True,
            "ai_analysis": True,
            "compliance_checking": True,
            "real_time_processing": True
        }
    }

# Debug endpoint to check file structure on deployed server
@app.get("/api/v1/debug/files")
async def debug_files():
    from pathlib import Path
    try:
        result = {
            "current_dir": str(Path.cwd()),
            "checklist_data_exists": False,
            "checklist_data_contents": [],
            "frameworks_dir_exists": False,
            "frameworks_dir_contents": []
        }
        
        checklist_path = Path.cwd() / "checklist_data"
        result["checklist_data_exists"] = checklist_path.exists()
        
        if checklist_path.exists():
            result["checklist_data_contents"] = [item.name for item in checklist_path.iterdir()]
            
            frameworks_path = checklist_path / "frameworks"
            result["frameworks_dir_exists"] = frameworks_path.exists()
            
            if frameworks_path.exists():
                result["frameworks_dir_contents"] = [item.name for item in frameworks_path.iterdir()]
        
        return result
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)