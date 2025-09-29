#!/usr/bin/env python3
"""
AI Parser FastAPI Routes - AI tagging system integration
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import asyncio

from ai_parser.file_processor import get_queue_manager
from ai_parser import AITaggingEngine

logger = logging.getLogger(__name__)

# Create AI Parser router
router = APIRouter()

# Initialize queue manager and AI engine
queue_manager = get_queue_manager()
ai_engine = AITaggingEngine()

@router.post("/upload")
async def upload_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """Handle multiple file uploads and add to processing queue"""
    try:
        job_ids = []
        
        for file in files:
            if not file.filename:
                continue
                
            if not file.filename.lower().endswith('.json'):
                continue
            
            # Read and parse JSON file
            try:
                file_content = await file.read()
                # Handle UTF-8 BOM by trying utf-8-sig first, then fallback to utf-8
                try:
                    text_content = file_content.decode('utf-8-sig')
                except UnicodeDecodeError:
                    text_content = file_content.decode('utf-8')
                
                file_data = json.loads(text_content)
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                logger.error(f"Error parsing file {file.filename}: {e}")
                continue
            
            # Add to processing queue
            job_id = queue_manager.add_job(file.filename, file_data)
            job_ids.append(job_id)
            
            logger.info(f"Added file {file.filename} to queue with job_id {job_id}")
        
        return {
            'success': True,
            'job_ids': job_ids,
            'message': f'Successfully queued {len(job_ids)} files for processing'
        }
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get the current status of a processing job"""
    try:
        status = queue_manager.get_job_status(job_id)
        if not status:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download/{job_id}")
async def download_file(job_id: str):
    """Download the enhanced JSON file for a completed job"""
    try:
        download_info = queue_manager.get_download_info(job_id)
        if not download_info:
            raise HTTPException(status_code=404, detail="Download not available")
        
        download_id = download_info['download_id']
        filename = download_info['filename']
        
        filepath = Path("ai_parser_downloads") / f"{download_id}.json"
        
        if not filepath.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            path=str(filepath),
            filename=filename,
            media_type='application/json'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/queue/status")
async def queue_status():
    """Get overall queue status and statistics"""
    try:
        stats = queue_manager.get_queue_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Queue status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint for AI Parser"""
    return {
        'status': 'healthy',
        'service': 'ai-parser',
        'queue_stats': queue_manager.get_queue_stats()
    }

@router.post("/classify-single")
async def classify_single_question(question_data: Dict[str, Any]):
    """Classify a single question for testing purposes"""
    try:
        question_text = question_data.get('question', '')
        context = question_data.get('context', '')
        
        if not question_text:
            raise HTTPException(status_code=400, detail="Question text is required")
        
        result = ai_engine.classify_question(question_text, context)
        
        return {
            'success': result.success,
            'classification': result.classification,
            'error': result.error
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Single classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions
def get_ai_parser_stats() -> Dict[str, int]:
    """Get AI Parser statistics for main dashboard"""
    try:
        return queue_manager.get_queue_stats()
    except Exception:
        return {'total_jobs': 0, 'completed_jobs': 0, 'processing_jobs': 0, 'queued_jobs': 0, 'failed_jobs': 0}

async def cleanup_old_downloads(days_old: int = 7):
    """Cleanup old download files"""
    try:
        downloads_dir = Path("ai_parser_downloads")
        if not downloads_dir.exists():
            return
        
        import time
        current_time = time.time()
        cutoff_time = current_time - (days_old * 24 * 60 * 60)
        
        for file_path in downloads_dir.glob("*.json"):
            if file_path.stat().st_mtime < cutoff_time:
                file_path.unlink()
                logger.info(f"Cleaned up old download file: {file_path}")
                
    except Exception as e:
        logger.error(f"Error cleaning up downloads: {e}")

# Initialize on module load
def init_ai_parser():
    """Initialize AI Parser components"""
    # Ensure downloads directory exists
    Path("ai_parser_downloads").mkdir(exist_ok=True)
    logger.info("AI Parser routes initialized")

# Initialize when module is imported
init_ai_parser()