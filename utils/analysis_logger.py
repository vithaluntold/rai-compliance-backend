"""
Analysis Pipeline Backend Logger
Comprehensive logging for the entire analysis pipeline on the backend side
Designed to identify bottlenecks and failures in production (Render)
"""

import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import os
import traceback

class PipelineStatus(Enum):
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"

class AnalysisBackendLogger:
    def __init__(self, document_id: str, session_id: str = None):
        self.document_id = document_id
        self.session_id = session_id or f"backend_{int(time.time())}"
        self.start_time = datetime.utcnow()
        self.steps: List[Dict[str, Any]] = []
        self.critical_errors: List[str] = []
        self.warnings: List[str] = []
        self.current_step = "initialization"
        
        # Setup logger
        self.logger = logging.getLogger(f"analysis_pipeline_{document_id}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - ANALYSIS_PIPELINE - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Log initialization
        self.log_step("backend_initialization", PipelineStatus.STARTED, {
            "document_id": document_id,
            "session_id": session_id,
            "render_instance": os.environ.get("RENDER_INSTANCE_ID", "local"),
            "python_version": os.environ.get("PYTHON_VERSION", "unknown"),
            "environment": os.environ.get("ENVIRONMENT", "production")
        })

    def log_step(self, step_name: str, status: PipelineStatus, data: Dict[str, Any] = None):
        """Log a pipeline step with comprehensive metadata"""
        timestamp = datetime.utcnow().isoformat()
        step_data = {
            "step": step_name,
            "status": status.value,
            "timestamp": timestamp,
            "document_id": self.document_id,
            "session_id": self.session_id,
            "data": data or {},
            "memory_usage": self._get_memory_usage(),
            "step_duration": self._calculate_step_duration(step_name)
        }
        
        self.steps.append(step_data)
        self.current_step = step_name
        
        # Log to console/file based on status
        log_message = f"STEP: {step_name} | STATUS: {status.value} | DOC: {self.document_id}"
        if data:
            log_message += f" | DATA: {json.dumps(data, default=str)}"
        
        if status == PipelineStatus.FAILED:
            self.logger.error(log_message)
            self.critical_errors.append(f"[{timestamp}] {step_name}: {data.get('error', 'Unknown error')}")
        elif status == PipelineStatus.COMPLETED:
            self.logger.info(log_message)
        else:
            self.logger.debug(log_message)

    # UPLOAD PROCESSING PIPELINE
    def log_upload_received(self, file_info: Dict[str, Any]):
        """Log when file upload is received on backend"""
        self.log_step("upload_received", PipelineStatus.STARTED, {
            "filename": file_info.get("filename"),
            "file_size": file_info.get("size"),
            "content_type": file_info.get("content_type"),
            "upload_method": file_info.get("method", "unknown")
        })

    def log_upload_validation(self, validation_result: Dict[str, Any]):
        """Log file validation results"""
        status = PipelineStatus.COMPLETED if validation_result.get("valid") else PipelineStatus.FAILED
        self.log_step("upload_validation", status, validation_result)

    def log_upload_storage(self, storage_result: Dict[str, Any]):
        """Log file storage results"""
        status = PipelineStatus.COMPLETED if storage_result.get("success") else PipelineStatus.FAILED
        self.log_step("upload_storage", status, storage_result)

    # METADATA EXTRACTION PIPELINE
    def log_metadata_extraction_start(self, extraction_config: Dict[str, Any]):
        """Log start of metadata extraction"""
        self.log_step("metadata_extraction_start", PipelineStatus.STARTED, {
            "extraction_type": extraction_config.get("type", "enhanced"),
            "ai_models": extraction_config.get("models", []),
            "geographic_detection": extraction_config.get("geographic_detection", False)
        })

    def log_document_parsing(self, parsing_result: Dict[str, Any]):
        """Log document parsing results"""
        status = PipelineStatus.COMPLETED if parsing_result.get("success") else PipelineStatus.FAILED
        self.log_step("document_parsing", status, {
            "pages_processed": parsing_result.get("pages"),
            "text_length": parsing_result.get("text_length"),
            "parsing_engine": parsing_result.get("engine"),
            "parsing_time": parsing_result.get("duration"),
            "errors": parsing_result.get("errors", [])
        })

    def log_ai_extraction_attempt(self, attempt_data: Dict[str, Any]):
        """Log AI extraction attempt"""
        self.log_step("ai_extraction_attempt", PipelineStatus.STARTED, {
            "model": attempt_data.get("model"),
            "prompt_type": attempt_data.get("prompt_type"),
            "attempt_number": attempt_data.get("attempt", 1),
            "input_length": attempt_data.get("input_length")
        })

    def log_ai_extraction_result(self, extraction_result: Dict[str, Any]):
        """Log AI extraction results"""
        status = PipelineStatus.COMPLETED if extraction_result.get("success") else PipelineStatus.FAILED
        extracted_fields = extraction_result.get("extracted_data", {})
        
        self.log_step("ai_extraction_result", status, {
            "success": extraction_result.get("success"),
            "model_used": extraction_result.get("model"),
            "extraction_time": extraction_result.get("duration"),
            "fields_extracted": list(extracted_fields.keys()) if extracted_fields else [],
            "company_name_found": bool(extracted_fields.get("company_name")),
            "business_nature_found": bool(extracted_fields.get("nature_of_business")),
            "demographics_found": bool(extracted_fields.get("operational_demographics")),
            "confidence_scores": extraction_result.get("confidence_scores", {}),
            "error": extraction_result.get("error")
        })

    def log_metadata_storage(self, storage_result: Dict[str, Any]):
        """Log metadata storage to database"""
        status = PipelineStatus.COMPLETED if storage_result.get("success") else PipelineStatus.FAILED
        self.log_step("metadata_storage", status, storage_result)

    # FRAMEWORK & ANALYSIS PIPELINE
    def log_framework_request(self, request_data: Dict[str, Any]):
        """Log framework selection request"""
        self.log_step("framework_request_received", PipelineStatus.STARTED, {
            "framework": request_data.get("framework"),
            "standards_count": len(request_data.get("standards", [])),
            "standards": request_data.get("standards", []),
            "has_custom_instructions": bool(request_data.get("specialInstructions")),
            "processing_mode": request_data.get("processingMode")
        })

    def log_framework_validation(self, validation_result: Dict[str, Any]):
        """Log framework validation results"""
        status = PipelineStatus.COMPLETED if validation_result.get("valid") else PipelineStatus.FAILED
        self.log_step("framework_validation", status, validation_result)

    def log_analysis_job_creation(self, job_data: Dict[str, Any]):
        """Log analysis job creation"""
        status = PipelineStatus.COMPLETED if job_data.get("success") else PipelineStatus.FAILED
        self.log_step("analysis_job_created", status, {
            "job_id": job_data.get("job_id"),
            "queue_position": job_data.get("queue_position"),
            "estimated_duration": job_data.get("estimated_duration"),
            "worker_assigned": job_data.get("worker_id"),
            "priority": job_data.get("priority")
        })

    def log_analysis_processing_start(self, processing_data: Dict[str, Any]):
        """Log start of actual analysis processing"""
        self.log_step("analysis_processing_start", PipelineStatus.STARTED, {
            "worker_id": processing_data.get("worker_id"),
            "document_chunks": processing_data.get("chunk_count"),
            "total_standards": processing_data.get("total_standards"),
            "processing_strategy": processing_data.get("strategy")
        })

    def log_analysis_progress(self, progress_data: Dict[str, Any]):
        """Log analysis progress updates"""
        self.log_step("analysis_progress", PipelineStatus.STARTED, {
            "percentage": progress_data.get("percentage"),
            "current_standard": progress_data.get("current_standard"),
            "completed_standards": progress_data.get("completed_standards"),
            "estimated_remaining": progress_data.get("estimated_remaining"),
            "worker_status": progress_data.get("worker_status")
        })

    def log_analysis_completion(self, completion_data: Dict[str, Any]):
        """Log analysis completion"""
        status = PipelineStatus.COMPLETED if completion_data.get("success") else PipelineStatus.FAILED
        total_duration = (datetime.utcnow() - self.start_time).total_seconds()
        
        self.log_step("analysis_completed", status, {
            "success": completion_data.get("success"),
            "total_duration_seconds": total_duration,
            "results_generated": completion_data.get("results_count"),
            "compliance_score": completion_data.get("compliance_score"),
            "issues_found": completion_data.get("issues_count"),
            "worker_id": completion_data.get("worker_id"),
            "final_status": completion_data.get("status")
        })

    # ERROR AND FAILURE TRACKING
    def log_critical_error(self, error: Exception, context: str = ""):
        """Log critical errors that prevent analysis"""
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "traceback": traceback.format_exc(),
            "current_step": self.current_step
        }
        
        self.log_step(f"critical_error_{context}", PipelineStatus.FAILED, error_data)
        self.critical_errors.append(f"[{datetime.utcnow().isoformat()}] CRITICAL: {context}: {str(error)}")

    def log_timeout_error(self, operation: str, timeout_duration: int):
        """Log timeout errors"""
        self.log_step(f"timeout_{operation}", PipelineStatus.TIMEOUT, {
            "operation": operation,
            "timeout_duration_seconds": timeout_duration,
            "current_step": self.current_step
        })
        self.critical_errors.append(f"TIMEOUT: {operation} after {timeout_duration}s")

    # API COMMUNICATION TRACKING
    def log_api_call(self, endpoint: str, method: str, request_data: Dict[str, Any] = None):
        """Log API calls made during processing"""
        self.log_step("api_call", PipelineStatus.STARTED, {
            "endpoint": endpoint,
            "method": method,
            "request_size": len(json.dumps(request_data)) if request_data else 0,
            "has_auth": "authorization" in str(request_data).lower() if request_data else False
        })

    def log_api_response(self, endpoint: str, response_data: Dict[str, Any]):
        """Log API response"""
        status = PipelineStatus.COMPLETED if response_data.get("success") else PipelineStatus.FAILED
        self.log_step("api_response", status, {
            "endpoint": endpoint,
            "status_code": response_data.get("status_code"),
            "response_size": response_data.get("response_size"),
            "duration": response_data.get("duration"),
            "success": response_data.get("success")
        })

    # DIAGNOSTIC AND EXPORT METHODS
    def diagnose_pipeline_issues(self) -> List[str]:
        """Diagnose common pipeline issues"""
        issues = []
        
        # Check for upload issues
        upload_steps = [s for s in self.steps if "upload" in s["step"]]
        failed_uploads = [s for s in upload_steps if s["status"] == "failed"]
        if failed_uploads:
            issues.append(f"UPLOAD_FAILURE: {len(failed_uploads)} upload operations failed")

        # Check for metadata extraction issues
        metadata_steps = [s for s in self.steps if "metadata" in s["step"] or "extraction" in s["step"]]
        if not metadata_steps:
            issues.append("METADATA_MISSING: No metadata extraction steps found")
        
        failed_extractions = [s for s in metadata_steps if s["status"] == "failed"]
        if failed_extractions:
            issues.append(f"EXTRACTION_FAILURE: {len(failed_extractions)} extraction operations failed")

        # Check for analysis issues
        analysis_steps = [s for s in self.steps if "analysis" in s["step"]]
        if not analysis_steps:
            issues.append("ANALYSIS_NOT_STARTED: No analysis steps found")
        
        # Check for timeouts
        timeout_steps = [s for s in self.steps if s["status"] == "timeout"]
        if timeout_steps:
            issues.append(f"TIMEOUTS: {len(timeout_steps)} operations timed out")

        # Check for critical errors
        if self.critical_errors:
            issues.append(f"CRITICAL_ERRORS: {len(self.critical_errors)} critical errors occurred")

        return issues

    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get comprehensive pipeline summary"""
        total_duration = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            "document_id": self.document_id,
            "session_id": self.session_id,
            "total_duration_seconds": total_duration,
            "current_step": self.current_step,
            "total_steps": len(self.steps),
            "completed_steps": len([s for s in self.steps if s["status"] == "completed"]),
            "failed_steps": len([s for s in self.steps if s["status"] == "failed"]),
            "critical_errors": self.critical_errors,
            "warnings": self.warnings,
            "render_instance": os.environ.get("RENDER_INSTANCE_ID", "local"),
            "environment": os.environ.get("ENVIRONMENT", "production")
        }

    def export_full_log(self) -> Dict[str, Any]:
        """Export complete log for debugging"""
        return {
            "pipeline_summary": self.get_pipeline_summary(),
            "all_steps": self.steps,
            "diagnosed_issues": self.diagnose_pipeline_issues(),
            "environment": {
                "render_instance": os.environ.get("RENDER_INSTANCE_ID"),
                "python_version": os.environ.get("PYTHON_VERSION"),
                "memory_limit": os.environ.get("MEMORY_LIMIT"),
                "cpu_count": os.cpu_count(),
                "environment": os.environ.get("ENVIRONMENT", "production")
            }
        }

    def send_to_render_logs(self):
        """Send logs to Render's logging system"""
        try:
            summary = self.get_pipeline_summary()
            issues = self.diagnose_pipeline_issues()
            
            # Critical errors or failures should be logged as ERROR level
            if self.critical_errors or issues:
                self.logger.error(f"PIPELINE_FAILURE_SUMMARY: {json.dumps(summary, default=str)}")
                self.logger.error(f"DIAGNOSED_ISSUES: {json.dumps(issues)}")
            else:
                self.logger.info(f"PIPELINE_SUCCESS_SUMMARY: {json.dumps(summary, default=str)}")
                
        except Exception as e:
            self.logger.error(f"Failed to send logs to Render: {str(e)}")

    # UTILITY METHODS
    def _get_memory_usage(self) -> Optional[Dict[str, Any]]:
        """Get current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            return {
                "memory_mb": process.memory_info().rss / 1024 / 1024,
                "cpu_percent": process.cpu_percent()
            }
        except ImportError:
            return None

    def _calculate_step_duration(self, step_name: str) -> Optional[float]:
        """Calculate duration for a step"""
        # Find the last occurrence of this step being started
        started_step = None
        for step in reversed(self.steps):
            if step["step"] == step_name and step["status"] == "started":
                started_step = step
                break
        
        if started_step:
            start_time = datetime.fromisoformat(started_step["timestamp"])
            return (datetime.utcnow() - start_time).total_seconds()
        
        return None

# Global logger instance management
_active_loggers: Dict[str, AnalysisBackendLogger] = {}

def get_logger(document_id: str, session_id: str = None) -> AnalysisBackendLogger:
    """Get or create a logger for a document"""
    if document_id not in _active_loggers:
        _active_loggers[document_id] = AnalysisBackendLogger(document_id, session_id)
    return _active_loggers[document_id]

def cleanup_logger(document_id: str):
    """Clean up logger and send final logs"""
    if document_id in _active_loggers:
        logger = _active_loggers[document_id]
        logger.send_to_render_logs()
        del _active_loggers[document_id]

# Decorator for automatic function logging
def log_pipeline_step(step_name: str):
    """Decorator to automatically log pipeline steps"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Extract document_id from args or kwargs
            document_id = kwargs.get('document_id') or (args[0] if args else None)
            if not document_id:
                return func(*args, **kwargs)
                
            logger = get_logger(document_id)
            logger.log_step(f"{step_name}_start", PipelineStatus.STARTED)
            
            try:
                result = func(*args, **kwargs)
                logger.log_step(f"{step_name}_completed", PipelineStatus.COMPLETED, 
                              {"result_type": type(result).__name__})
                return result
            except Exception as e:
                logger.log_critical_error(e, step_name)
                raise
                
        return wrapper
    return decorator