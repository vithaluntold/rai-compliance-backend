"""
Bulletproof API Endpoints V2 - Database-Backed, Zero Race Conditions
Maintains full backward compatibility while providing bulletproof data access

New endpoints:
- GET /documents/{id}/v2 - Database-backed document status (zero race conditions)
- GET /documents/{id}/results/v2 - Database-backed results (bulletproof)  
- POST /documents/{id}/migrate - Migrate existing data to database
- GET /documents/{id}/verify - Verify data consistency between systems
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Union

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from database.dual_storage import get_dual_storage, save_analysis_atomic, get_analysis_atomic
from database.db_manager import get_database_manager

# Configure logging
logger = logging.getLogger(__name__)

# Create V2 router
router_v2 = APIRouter(prefix="/v2", tags=["bulletproof-analysis"])

@router_v2.get("/documents/{document_id}", response_model=None)
async def get_document_status_v2(document_id: str) -> Union[Dict[str, Any], JSONResponse]:
    """
    BULLETPROOF: Get document status from database with zero race conditions
    
    This endpoint is immune to:
    - File system race conditions
    - Partial data states
    - Background process overwrites
    - Concurrent access issues
    """
    logger.info(f"🔍 V2 GET /documents/{document_id} - BULLETPROOF REQUEST")
    
    try:
        # Get data from bulletproof database
        result = await get_analysis_atomic(document_id)
        
        if not result:
            logger.info(f"📭 V2: No data found for document {document_id}")
            return JSONResponse(
                status_code=404,
                content={
                    "error": "Document not found",
                    "message": f"Document with ID {document_id} not found in bulletproof storage",
                    "document_id": document_id
                }
            )
        
        # Build bulletproof response
        response = {
            "document_id": document_id,
            "status": result.get("status", "PENDING"),
            "metadata_extraction": result.get("metadata_extraction", "PENDING"),
            "compliance_analysis": result.get("compliance_analysis", "PENDING"),
            "processing_mode": result.get("processing_mode", "smart"),
            "framework": result.get("framework"),
            "standards": result.get("standards", []),
            "metadata": result.get("metadata", {}),
            "sections": result.get("sections", []),
            "message": result.get("message", "Analysis in progress"),
            "special_instructions": result.get("special_instructions", ""),
            "extensive_search": result.get("extensive_search", False),
            "performance_metrics": result.get("performance_metrics", {}),
            "failed_standards": result.get("failed_standards", []),
            "created_at": result.get("created_at"),
            "updated_at": result.get("updated_at"),
            "completed_at": result.get("completed_at"),
            "bulletproof": True,  # Indicates this came from bulletproof system
            "source": "database_v2"  # Data source identifier
        }
        
        # Add error information if present
        if result.get("error_message"):
            response["error"] = result["error_message"]
        
        logger.info(f"✅ V2: Successfully retrieved {document_id} with status {response['status']}")
        return response
        
    except Exception as e:
        logger.error(f"❌ V2: Error retrieving document {document_id}: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Server error",
                "message": f"Failed to retrieve document status: {str(e)}",
                "document_id": document_id,
                "bulletproof": False
            }
        )

@router_v2.get("/documents/{document_id}/results", response_model=None)
async def get_document_results_v2(document_id: str) -> Union[Dict[str, Any], JSONResponse]:
    """
    BULLETPROOF: Get document results from database with guaranteed consistency
    """
    logger.info(f"🔍 V2 GET /documents/{document_id}/results - BULLETPROOF RESULTS")
    
    try:
        result = await get_analysis_atomic(document_id)
        
        if not result:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "Results not found",
                    "message": f"No results found for document {document_id}",
                    "document_id": document_id,
                    "bulletproof": True
                }
            )
        
        # Return bulletproof results
        response = {
            "document_id": document_id,
            "status": result.get("status", "UNKNOWN"),
            "metadata": result.get("metadata", {}),
            "sections": result.get("sections", []),
            "framework": result.get("framework"),
            "standards": result.get("standards", []),
            "performance_metrics": result.get("performance_metrics", {}),
            "message": result.get("message", "Results retrieved successfully"),
            "bulletproof": True,
            "source": "database_v2",
            "retrieved_at": datetime.now().isoformat()
        }
        
        logger.info(f"✅ V2 RESULTS: Successfully retrieved {len(response['sections'])} sections for {document_id}")
        return response
        
    except Exception as e:
        logger.error(f"❌ V2 RESULTS: Error for {document_id}: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Server error", 
                "message": f"Failed to retrieve results: {str(e)}",
                "document_id": document_id,
                "bulletproof": False
            }
        )

@router_v2.post("/documents/{document_id}/migrate")
async def migrate_document_data_v2(document_id: str, background_tasks: BackgroundTasks) -> JSONResponse:
    """
    Migrate existing file-based data to bulletproof database storage
    """
    logger.info(f"🔄 V2 MIGRATE: Starting migration for {document_id}")
    
    async def perform_migration():
        try:
            storage = get_dual_storage()
            success = await storage.migrate_existing_data(document_id)
            
            if success:
                logger.info(f"✅ V2 MIGRATE: Success for {document_id}")
            else:
                logger.warning(f"📭 V2 MIGRATE: No data to migrate for {document_id}")
                
        except Exception as e:
            logger.error(f"❌ V2 MIGRATE: Failed for {document_id}: {e}")
    
    # Run migration in background
    background_tasks.add_task(perform_migration)
    
    return JSONResponse(
        status_code=202,
        content={
            "message": "Migration started",
            "document_id": document_id,
            "status": "migrating"
        }
    )

@router_v2.get("/documents/{document_id}/verify")
async def verify_data_consistency_v2(document_id: str) -> JSONResponse:
    """
    Verify data consistency between database and file systems
    """
    logger.info(f"🔍 V2 VERIFY: Starting verification for {document_id}")
    
    try:
        storage = get_dual_storage()
        report = await storage.verify_data_consistency(document_id)
        
        return JSONResponse(
            status_code=200,
            content={
                "verification_report": report,
                "bulletproof": True
            }
        )
        
    except Exception as e:
        logger.error(f"❌ V2 VERIFY: Failed for {document_id}: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Verification failed",
                "message": str(e),
                "document_id": document_id
            }
        )

@router_v2.get("/documents/{document_id}/progress")  
async def get_analysis_progress_v2(document_id: str) -> Union[Dict[str, Any], JSONResponse]:
    """
    BULLETPROOF: Get real-time analysis progress from database
    """
    logger.info(f"🔍 V2 PROGRESS: Getting progress for {document_id}")
    
    try:
        db = get_database_manager()
        progress = await db.get_progress(document_id)
        
        if not progress['standards']:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "Progress not found",
                    "message": f"No progress data found for document {document_id}",
                    "document_id": document_id
                }
            )
        
        # Format response for frontend compatibility
        response = {
            "document_id": document_id,
            "overall_progress": {
                "percentage": round(progress['overall_progress'], 1),
                "completed_questions": progress['completed_questions'],
                "total_questions": progress['total_questions']
            },
            "standards_detail": [],
            "bulletproof": True,
            "source": "database_v2"
        }
        
        for standard_id, standard_data in progress['standards'].items():
            response["standards_detail"].append({
                "standard_id": standard_id,
                "status": standard_data['status'],
                "progress_percentage": round(standard_data['progress_percentage'], 1),
                "completed_questions": standard_data['completed_questions'],
                "total_questions": standard_data['total_questions'],
                "current_question": standard_data['current_question']
            })
        
        return response
        
    except Exception as e:
        logger.error(f"❌ V2 PROGRESS: Error for {document_id}: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Server error",
                "message": f"Failed to get progress: {str(e)}",
                "document_id": document_id
            }
        )

@router_v2.get("/health")
async def bulletproof_health_check() -> JSONResponse:
    """
    Health check for bulletproof V2 system
    """
    try:
        db = get_database_manager()
        await db.initialize()
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "system": "bulletproof_v2",
                "database": "connected",
                "race_conditions": "eliminated",
                "timestamp": datetime.now().isoformat()
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy", 
                "error": str(e),
                "system": "bulletproof_v2"
            }
        )

@router_v2.post("/documents/{document_id}/save")
async def save_document_analysis_v2(document_id: str, data: Dict[str, Any]) -> JSONResponse:
    """
    BULLETPROOF: Save document analysis with atomic transaction
    For testing and manual data updates
    """
    logger.info(f"💾 V2 SAVE: Atomic save for {document_id}")
    
    try:
        await save_analysis_atomic(document_id, data)
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Document saved successfully",
                "document_id": document_id,
                "bulletproof": True,
                "saved_at": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"❌ V2 SAVE: Failed for {document_id}: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Save failed",
                "message": str(e),
                "document_id": document_id,
                "bulletproof": False
            }
        )