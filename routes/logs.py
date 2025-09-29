"""
API endpoint for receiving comprehensive analysis pipeline logs from frontend
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import logging
import json
from datetime import datetime

# Setup router
router = APIRouter(prefix="/api/logs", tags=["logging"])

# Setup logger for API logs
logger = logging.getLogger("pipeline_logs_api")

class FrontendLogEntry(BaseModel):
    sessionId: str
    documentId: Optional[str] = None
    fileName: Optional[str] = None
    currentStep: str
    overallStatus: str
    steps: List[Dict[str, Any]]
    criticalErrors: List[str]
    warnings: List[str]
    totalDuration: Optional[int] = None
    metadata: Dict[str, Any]

class ProceedAnalysisLog(BaseModel):
    timestamp: str
    documentId: Optional[str]
    selectedFramework: Optional[str]
    selectedStandards: List[str]
    standardCount: int
    isProcessing: bool
    currentStep: Optional[str]
    environment: str
    renderInstance: str

@router.post("/analysis-pipeline")
async def receive_analysis_pipeline_log(log_entry: FrontendLogEntry, request: Request):
    """
    Receive comprehensive analysis pipeline logs from frontend
    """
    try:
        # Log the received data
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "client_ip": client_ip,
            "user_agent": user_agent,
            "log_entry": log_entry.dict()
        }
        
        # Determine log level based on critical errors
        if log_entry.criticalErrors:
            logger.error(f"FRONTEND_CRITICAL_PIPELINE_LOG: {json.dumps(log_data, default=str)}")
        elif log_entry.warnings:
            logger.warning(f"FRONTEND_WARNING_PIPELINE_LOG: {json.dumps(log_data, default=str)}")
        else:
            logger.info(f"FRONTEND_INFO_PIPELINE_LOG: {json.dumps(log_data, default=str)}")
        
        # Print to console for immediate Render visibility
        print(f"📊 FRONTEND PIPELINE LOG - {log_entry.overallStatus.upper()}")
        print(f"Session: {log_entry.sessionId}")
        print(f"Document: {log_entry.documentId}")
        print(f"Current Step: {log_entry.currentStep}")
        print(f"Steps Completed: {len([s for s in log_entry.steps if s.get('status') == 'completed'])}")
        print(f"Critical Errors: {len(log_entry.criticalErrors)}")
        
        if log_entry.criticalErrors:
            print("🚨 CRITICAL ERRORS:")
            for error in log_entry.criticalErrors:
                print(f"  - {error}")
        
        if log_entry.warnings:
            print("⚠️ WARNINGS:")
            for warning in log_entry.warnings:
                print(f"  - {warning}")
        
        return {
            "status": "success", 
            "message": "Pipeline log received and processed",
            "log_level": "critical" if log_entry.criticalErrors else "warning" if log_entry.warnings else "info"
        }
        
    except Exception as e:
        logger.error(f"Failed to process frontend pipeline log: {str(e)}")
        print(f"🚨 ERROR processing frontend log: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process log: {str(e)}")

@router.post("/proceed-analysis-click")  
async def log_proceed_analysis_click(log_entry: ProceedAnalysisLog, request: Request):
    """
    Log when Proceed to Analysis button is clicked with full context
    """
    try:
        client_ip = request.client.host if request.client else "unknown"
        
        click_log = {
            "event": "PROCEED_TO_ANALYSIS_CLICKED",
            "timestamp": log_entry.timestamp,
            "client_ip": client_ip,
            "document_id": log_entry.documentId,
            "framework": log_entry.selectedFramework,
            "standards": log_entry.selectedStandards,
            "standard_count": log_entry.standardCount,
            "is_processing": log_entry.isProcessing,
            "current_step": log_entry.currentStep,
            "environment": log_entry.environment,
            "render_instance": log_entry.renderInstance
        }
        
        # Always log as INFO level for button clicks
        logger.info(f"PROCEED_ANALYSIS_BUTTON_CLICK: {json.dumps(click_log, default=str)}")
        
        # Print to console for immediate visibility
        print(f"🚀 PROCEED TO ANALYSIS BUTTON CLICKED")
        print(f"Document ID: {log_entry.documentId}")
        print(f"Framework: {log_entry.selectedFramework}")  
        print(f"Standards: {log_entry.standardCount} selected ({', '.join(log_entry.selectedStandards[:3])}{'...' if len(log_entry.selectedStandards) > 3 else ''})")
        print(f"Current Step: {log_entry.currentStep}")
        print(f"Processing: {log_entry.isProcessing}")
        print(f"Environment: {log_entry.environment}")
        
        # Check for potential issues
        issues = []
        if not log_entry.documentId:
            issues.append("Missing document ID")
        if not log_entry.selectedFramework:
            issues.append("Missing framework selection")
        if log_entry.standardCount == 0:
            issues.append("No standards selected")
        if log_entry.isProcessing:
            issues.append("Already processing (potential duplicate click)")
            
        if issues:
            print("⚠️ POTENTIAL ISSUES DETECTED:")
            for issue in issues:
                print(f"  - {issue}")
                
        return {
            "status": "success",
            "message": "Proceed to Analysis click logged",
            "issues_detected": issues
        }
        
    except Exception as e:
        logger.error(f"Failed to log proceed analysis click: {str(e)}")
        print(f"🚨 ERROR logging proceed click: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to log click: {str(e)}")

@router.get("/pipeline-status/{document_id}")
async def get_pipeline_status(document_id: str):
    """
    Get current pipeline status for a document
    """
    try:
        # This would typically query your database for the document status
        # For now, return a placeholder response
        return {
            "document_id": document_id,
            "status": "processing",
            "message": "Pipeline status endpoint - implement with your database"
        }
        
    except Exception as e:
        logger.error(f"Failed to get pipeline status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@router.post("/debug-export")
async def export_debug_logs(request: Request):
    """
    Export comprehensive debug logs for troubleshooting
    """
    try:
        request_data = await request.json()
        
        debug_export = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "request_data": request_data,
            "client_info": {
                "ip": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", "unknown")
            }
        }
        
        logger.info(f"DEBUG_EXPORT: {json.dumps(debug_export, default=str)}")
        
        print("📋 DEBUG EXPORT RECEIVED")
        print(f"Data keys: {list(request_data.keys()) if request_data else 'None'}")
        
        return {
            "status": "success",
            "message": "Debug logs exported successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to export debug logs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))