import asyncio
import json
import logging
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# NO PSUTIL IMPORT - DEPLOYMENT FIX
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from services.ai import AIService, get_ai_service
from services.ai_prompts import AIPrompts
from services.checklist_utils import (
    get_available_frameworks,
    is_standard_available,
    load_checklist,
)
from services.document_chunker import document_chunker
from services.smart_metadata_extractor import SmartMetadataExtractor
from services.vector_store import generate_document_id, get_vector_store

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Track performance metrics for different processing modes"""

    def __init__(self, mode: str):
        self.mode = mode
        self.start_time = None
        self.end_time = None
        self.token_usage = 0
        self.questions_processed = 0
        self.api_calls_made = 0
        self.accuracy_scores = []
        self.memory_usage = []

    def start_tracking(self):
        self.start_time = time.time()

    def end_tracking(self):
        self.end_time = time.time()

    def get_metrics(self) -> Dict:
        processing_time = (
            (self.end_time - self.start_time)
            if self.end_time and self.start_time
            else 0
        )
        avg_accuracy = (
            sum(self.accuracy_scores) / len(self.accuracy_scores)
            if self.accuracy_scores
            else 0
        )

        return {
            "processing_time_seconds": round(processing_time, 2),
            "tokens_consumed": self.token_usage,
            "questions_processed": self.questions_processed,
            "api_calls_made": self.api_calls_made,
            "avg_accuracy": round(avg_accuracy, 3),
            "efficiency_score": self.calculate_efficiency(),
        }

    def calculate_efficiency(self) -> float:
        """Calculate efficiency score based on time, tokens, and accuracy"""
        if not self.start_time or not self.end_time or self.token_usage == 0:
            return 0.0

        processing_time = self.end_time - self.start_time
        avg_accuracy = (
            sum(self.accuracy_scores) / len(self.accuracy_scores)
            if self.accuracy_scores
            else 0.5
        )

        # Efficiency = (Accuracy * Questions) / (Time * Tokens)
        efficiency = (avg_accuracy * self.questions_processed) / (
            processing_time * (self.token_usage / 1000)
        )
        return round(efficiency, 3)


try:
    from docx import Document as DocxDocument
except ImportError:
    logger.warning("python-docx not installed, DOCX support disabled")
    DocxDocument = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the absolute path to the backend directory
BACKEND_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create required directories using Path objects
UPLOADS_DIR = BACKEND_DIR / "uploads"
ANALYSIS_RESULTS_DIR = BACKEND_DIR / "analysis_results"
CHECKLIST_DATA_DIR = BACKEND_DIR / "checklist_data"

# Create directories
for directory in [UPLOADS_DIR, ANALYSIS_RESULTS_DIR, CHECKLIST_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directory created/verified: {directory}")

router = APIRouter()

# Initialize services
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv(
    "AZURE_OPENAI_ENDPOINT",
    "https://vitha-maxu94mf-eastus2.cognitiveservices.azure.com/",
)
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "model-router")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"
)
AZURE_OPENAI_EMBEDDING_API_VERSION = os.getenv(
    "AZURE_OPENAI_EMBEDDING_API_VERSION", "2023-05-15"
)


# Initialize smart metadata extractor
smart_metadata_extractor = SmartMetadataExtractor()


def save_analysis_results(document_id: str, results: Dict[str, Any]) -> None:
    """Save analysis results to JSON file."""
    try:
        results_path = ANALYSIS_RESULTS_DIR / f"{document_id}.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved analysis results for document {document_id}")
    except Exception as e:
        logger.error(f"Error saving analysis results: {str(e)}")
        raise


# Utility to get the file path for a document_id, regardless of extension
def get_document_file_path(document_id: str) -> Optional[Path]:
    for ext in [".pdf", ".docx"]:
        candidate = UPLOADS_DIR / f"{document_id}{ext}"
        if candidate.exists():
            return candidate
    return None


# Rate limiting for uploads
UPLOAD_RATE_LIMIT = {}  # IP -> last_upload_time
UPLOAD_COOLDOWN_SECONDS = 1  # Minimum seconds between uploads per IP


def _check_upload_rate_limit(client_ip: str) -> bool:
    """Check if client can upload based on rate limiting."""
    # Rate limiting disabled for development - allow all uploads
    return True


def _check_processing_locks(document_id: str) -> bool:
    """
    Check if processing is already in progress or completed.
    Returns True if should skip.
    """
    processing_lock_file = ANALYSIS_RESULTS_DIR / f"{document_id}.processing"
    metadata_lock_file = ANALYSIS_RESULTS_DIR / f"{document_id}.metadata_completed"

    if processing_lock_file.exists():
        logger.info(
            f"[PATCH] Processing already in progress for {document_id}, "
            "skipping duplicate trigger."
        )
        return True
    if metadata_lock_file.exists():
        logger.info(
            f"[PATCH] Metadata extraction already completed for "
            f"{document_id}, skipping duplicate trigger."
        )
        return True
    return False


def _initialize_processing_results(
    document_id: str, processing_mode: str = "smart"
) -> dict:
    """Initialize empty results file with proper structure."""
    initial_results = {
        "status": "PROCESSING",
        "document_id": document_id,
        "timestamp": datetime.now().isoformat(),
        "processing_mode": processing_mode,
        "metadata": {},
        "sections": [],
        "metadata_extraction": "PENDING",
        "compliance_analysis": "PENDING",
        "message": "Document processing started",
    }
    save_analysis_results(document_id, initial_results)
    return initial_results


def _process_document_chunks(document_id: str) -> list:
    """Process document chunks based on file type."""
    logger.info(f"Starting document chunking for {document_id}")
    file_path = get_document_file_path(document_id)
    if not file_path:
        logger.error(f"No file found for document_id: {document_id}")
        raise ValueError("No file found for document_id")

    ext = file_path.suffix.lower()
    if ext == ".pdf":
        chunks = document_chunker.chunk_pdf(str(file_path), document_id)
    elif ext == ".docx":
        chunks = document_chunker.chunk_docx(str(file_path), document_id)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    if not chunks:
        raise ValueError("No chunks generated from document")

    logger.info(f"Generated {len(chunks)} chunks for document {document_id}")
    return chunks


async def _create_vector_index(document_id: str, chunks: list) -> None:
    """Index chunks in vector store."""
    logger.info(f"Starting vector store indexing for document {document_id}")
    vs_svc = get_vector_store()
    if not vs_svc:
        raise ValueError("Vector store service not initialized")

    index_created = await vs_svc.create_index(document_id, chunks)
    if not index_created:
        raise ValueError("Failed to create vector index")

    logger.info(f"Vector store indexing completed for document {document_id}")


async def _extract_document_metadata(document_id: str, chunks: list) -> dict:
    """Extract metadata from document chunks using optimized smart extraction."""
    logger.info(f"🚀 Starting OPTIMIZED metadata extraction for document {document_id}")
    logger.info(f"🔍 Using SmartMetadataExtractor - this should be DIFFERENT from old extraction!")
    
    # Use smart metadata extractor for 80% token reduction
    metadata_result = await smart_metadata_extractor.extract_metadata_optimized(document_id, chunks)
    logger.info(f"✅ Smart metadata extraction completed for document {document_id}")
    logger.info(f"💰 Token usage: {metadata_result.get('optimization_metrics', {}).get('tokens_used', 'N/A')}")
    return metadata_result


def _transform_metadata_for_frontend(metadata_result: dict) -> dict:
    """Transform metadata from smart extractor format to enhanced frontend presentation."""
    transformed_metadata = {}

    # Fields that need to be transformed
    metadata_fields = ["company_name", "nature_of_business", "operational_demographics", "financial_statements_type"]

    for field in metadata_fields:
        if field in metadata_result:
            value = metadata_result[field]
            # Smart extractor returns objects with value, confidence, extraction_method, context
            if isinstance(value, dict) and "value" in value:
                transformed_field = {
                    "value": value.get("value", ""),
                    "confidence": value.get("confidence", 0.0),
                    "extraction_method": value.get("extraction_method", "unknown"),
                    "context": value.get("context", ""),
                    "confidence_level": _get_confidence_level(value.get("confidence", 0.0)),
                    "presentation": _format_field_presentation(field, value)
                }
                
                # Special handling for operational_demographics to extract geography
                if field == "operational_demographics":
                    # Extract geography-specific information for result page
                    geography_parts = []
                    
                    # Add primary location if available
                    if value.get("primary_location"):
                        geography_parts.append(value["primary_location"])
                    
                    # Add regions detected
                    if value.get("regions_detected") and isinstance(value["regions_detected"], list):
                        geography_parts.extend(value["regions_detected"])
                    
                    # Add geographical entities
                    if value.get("geographical_entities") and isinstance(value["geographical_entities"], list):
                        for entity in value["geographical_entities"]:
                            if isinstance(entity, dict):
                                location_name = (
                                    entity.get("name") or 
                                    entity.get("location") or 
                                    entity.get("place") or 
                                    entity.get("country") or 
                                    entity.get("region")
                                )
                                if location_name and location_name not in geography_parts:
                                    geography_parts.append(str(location_name))
                    
                    # Create geography_of_operations field
                    if geography_parts:
                        # Remove duplicates while preserving order
                        unique_parts = []
                        for part in geography_parts:
                            part_clean = str(part).strip()
                            if part_clean and part_clean not in unique_parts:
                                unique_parts.append(part_clean)
                        geography_value = ", ".join(unique_parts)
                    else:
                        geography_value = value.get("value", "Geographic operations not specified")
                    
                    transformed_field["geography_of_operations"] = geography_value
                
                transformed_metadata[field] = transformed_field
            else:
                # Handle legacy string format
                transformed_metadata[field] = {
                    "value": str(value) if value is not None else "",
                    "confidence": 0.85,
                    "extraction_method": "legacy",
                    "context": "",
                    "confidence_level": "High",
                    "presentation": str(value) if value else "Not specified"
                }
                
                # Add geography field for legacy format too
                if field == "operational_demographics":
                    transformed_metadata[field]["geography_of_operations"] = str(value) if value else "Geographic operations not specified"
        else:
            # Provide default structure for missing fields
            default_field = {
                "value": "", 
                "confidence": 0.0,
                "extraction_method": "none",
                "context": "",
                "confidence_level": "None",
                "presentation": "Not specified"
            }
            
            # Add geography field for operational demographics
            if field == "operational_demographics":
                default_field["geography_of_operations"] = "Geographic operations not specified"
            
            transformed_metadata[field] = default_field

    # Copy optimization metrics and other fields
    for field, value in metadata_result.items():
        if field not in metadata_fields:
            transformed_metadata[field] = value

    # Add enhanced presentation summary
    transformed_metadata["extraction_summary"] = _create_extraction_summary(transformed_metadata)

    return transformed_metadata


def _get_confidence_level(confidence: float) -> str:
    """Convert confidence score to human-readable level."""
    if confidence >= 0.9:
        return "Very High"
    elif confidence >= 0.7:
        return "High"
    elif confidence >= 0.5:
        return "Medium"
    elif confidence > 0.0:
        return "Low"
    else:
        return "None"


def _format_field_presentation(field_name: str, field_data: dict) -> str:
    """Format field data for enhanced presentation."""
    value = field_data.get("value", "")
    confidence = field_data.get("confidence", 0.0)
    context = field_data.get("context", "")
    method = field_data.get("extraction_method", "")
    
    if not value:
        return "Not specified"
    
    # Create formatted presentation based on field type
    if field_name == "operational_demographics" and context:
        # For demographics, show value with context
        return f"{value} (Context: {context[:200]}...)" if len(context) > 200 else f"{value} (Context: {context})"
    elif field_name == "nature_of_business" and context:
        # For business nature, provide detailed description
        return f"{value}. Details: {context[:300]}..." if len(context) > 300 else f"{value}. Details: {context}"
    elif context:
        # For other fields with context
        return f"{value} (Source: {context[:150]}...)" if len(context) > 150 else f"{value} (Source: {context})"
    else:
        return value


def _create_extraction_summary(metadata: dict) -> dict:
    """Create a summary table of extraction results."""
    summary = {
        "extraction_table": [],
        "confidence_metrics": {
            "average_confidence": 0.0,
            "total_fields_extracted": 0,
            "high_confidence_fields": 0
        }
    }
    
    metadata_fields = ["company_name", "nature_of_business", "operational_demographics", "financial_statements_type"]
    total_confidence = 0.0
    extracted_fields = 0
    high_confidence_count = 0
    
    for field in metadata_fields:
        if field in metadata and isinstance(metadata[field], dict):
            field_data = metadata[field]
            confidence = field_data.get("confidence", 0.0)
            
            # Add to extraction table
            summary["extraction_table"].append({
                "field": field.replace("_", " ").title(),
                "value": field_data.get("value", "Not specified"),
                "confidence": f"{confidence:.1%}",
                "confidence_level": field_data.get("confidence_level", "None"),
                "method": field_data.get("extraction_method", "unknown"),
                "has_context": bool(field_data.get("context", ""))
            })
            
            # Update metrics
            if confidence > 0:
                total_confidence += confidence
                extracted_fields += 1
                if confidence >= 0.7:
                    high_confidence_count += 1
    
    # Calculate summary metrics
    summary["confidence_metrics"]["average_confidence"] = total_confidence / extracted_fields if extracted_fields > 0 else 0.0
    summary["confidence_metrics"]["total_fields_extracted"] = extracted_fields
    summary["confidence_metrics"]["high_confidence_fields"] = high_confidence_count
    
    return summary


def _finalize_processing_results(document_id: str, metadata_result: dict) -> dict:
    """Create final results after successful processing."""
    # Transform metadata for frontend compatibility
    transformed_metadata = _transform_metadata_for_frontend(metadata_result)

    final_results = {
        "status": "awaiting_framework_selection",
        "document_id": document_id,
        "timestamp": datetime.now().isoformat(),
        "metadata": transformed_metadata,
        "sections": [],
        "metadata_extraction": "COMPLETED",
        "compliance_analysis": "PENDING",
        "message": (
            "Metadata extraction completed. Please select a framework "
            "and standard for compliance analysis."
        ),
    }
    save_analysis_results(document_id, final_results)

    return final_results


def _cleanup_uploaded_file(document_id: str) -> None:
    """Auto-delete uploaded file after vectorization and chunking."""
    file_path = get_document_file_path(document_id)
    try:
        if file_path and file_path.exists():
            file_path.unlink()
            logger.info(f"Deleted uploaded file after vectorization: {file_path}")
    except Exception as cleanup_err:
        logger.warning(f"Failed to delete uploaded file: {file_path} ({cleanup_err})")


def _handle_processing_error(document_id: str, error: Exception) -> None:
    """Handle processing errors and save error state."""
    logger.error(
        f"Error processing document {document_id}: {str(error)}", exc_info=True
    )
    error_results = {
        "status": "error",
        "document_id": document_id,
        "timestamp": datetime.now().isoformat(),
        "error": str(error),
        "error_timestamp": datetime.now().isoformat(),
        "metadata": {},
        "sections": [],
        "metadata_extraction": "ERROR",
        "compliance_analysis": "PENDING",
        "message": f"Document processing failed: {str(error)}",
    }
    save_analysis_results(document_id, error_results)

    # Create error lock file
    error_lock_file = ANALYSIS_RESULTS_DIR / f"{document_id}.error"
    error_lock_file.touch()


async def process_upload_tasks(
    document_id: str, ai_svc: AIService, text: str = "", processing_mode: str = "smart"
) -> None:
    """Run document processing tasks up to metadata extraction."""
    # Check for duplicate processing
    if _check_processing_locks(document_id):
        return

    # Create processing lock
    processing_lock_file = ANALYSIS_RESULTS_DIR / f"{document_id}.processing"
    processing_lock_file.touch()

    try:
        # Initialize processing with processing mode
        _initialize_processing_results(document_id, processing_mode)

        # Process document chunks
        chunks = _process_document_chunks(document_id)

        # Create vector index
        await _create_vector_index(document_id, chunks)

        # Extract metadata
        metadata_result = await _extract_document_metadata(document_id, chunks)

        # Finalize results
        _finalize_processing_results(document_id, metadata_result)

        # Create metadata completion lock file
        metadata_lock_file = ANALYSIS_RESULTS_DIR / f"{document_id}.metadata_completed"
        metadata_lock_file.touch()

        # Remove processing lock file
        if processing_lock_file.exists():
            processing_lock_file.unlink()

        # Cleanup uploaded file
        _cleanup_uploaded_file(document_id)

        logger.info(f"Metadata extraction completed for document {document_id}")

    except Exception as e:
        # Handle error
        _handle_processing_error(document_id, e)

        # Remove processing lock file if it exists
        if processing_lock_file.exists():
            processing_lock_file.unlink()

        raise


@router.post("/upload", response_model=None)
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    processing_mode: str = Form(default="smart"),
    ai_svc: AIService = Depends(get_ai_service),
) -> Union[Dict[str, Any], JSONResponse]:
    """Upload and process a document."""
    try:
        # Log request details
        logger.info(f"Received upload request for file: {file.filename}")

        # Get client IP for rate limiting
        client_ip = request.client.host if request.client else "unknown"

        # Check rate limiting
        if not _check_upload_rate_limit(client_ip):
            response = {
                "status": "error",
                "error": "Rate limit exceeded",
                "message": f"Please wait {UPLOAD_COOLDOWN_SECONDS} seconds between uploads",
            }
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return JSONResponse(status_code=429, content=response)

        # Validate file type
        allowed_exts = [".pdf", ".docx"]
        if file.filename is None:
            response = {
                "status": "error",
                "error": "Invalid file",
                "message": "No filename provided",
            }
            logger.error("Upload failed: No filename provided")
            return JSONResponse(status_code=400, content=response)

        ext = file.filename.lower().split(".")[-1]
        if f".{ext}" not in allowed_exts:
            response = {
                "status": "error",
                "error": "Invalid file type",
                "message": "Only PDF and DOCX files are supported",
            }
            logger.info(
                f"Returning response for invalid file type: {json.dumps(response)}"
            )
            return response

        # Read file content
        try:
            content = await file.read()
        except Exception as e:
            logger.error(f"Error reading file content: {str(e)}")
            response = {
                "status": "error",
                "error": "File processing failed",
                "message": "Failed to read uploaded file content",
            }
            return response

        # Generate unique document ID
        document_id = generate_document_id()
        logger.info(f"Processing document upload with ID: {document_id}")

        # Save uploaded file with original extension
        upload_ext = f".{ext}"
        upload_path = UPLOADS_DIR / f"{document_id}{upload_ext}"
        try:
            with open(upload_path, "wb") as f:
                f.write(content)
            logger.info(f"Saved uploaded file to: {upload_path}")
        except Exception as e:
            logger.error(f"Error saving uploaded file: {str(e)}")
            response = {
                "status": "error",
                "error": "File save failed",
                "message": "Failed to save uploaded file",
            }
            logger.info(
                f"Returning response for file save error: {json.dumps(response)}"
            )
            return response

        # Log the processing mode
        logger.info(f"Upload received with processing mode: {processing_mode}")

        # Validate processing mode
        valid_modes = ["zap", "smart", "comparison"]
        if processing_mode not in valid_modes:
            logger.warning(
                f"Invalid processing mode '{processing_mode}', defaulting to 'smart'"
            )
            processing_mode = "smart"

        # No need to chunk here, just start background processing
        try:
            background_tasks.add_task(
                process_upload_tasks,
                document_id=document_id,
                ai_svc=ai_svc,
                processing_mode=processing_mode,
            )
            response = {
                "status": "processing",
                "document_id": document_id,
                "processing_mode": processing_mode,
                "message": f"Document uploaded with {processing_mode} mode, processing started",
            }
            logger.info(f"Returning success response: {json.dumps(response)}")
            return response
        except Exception as e:
            logger.error(f"Error starting background processing: {str(e)}")
            response = {
                "status": "error",
                "error": "Processing failed",
                "message": "Failed to start document processing",
            }
            logger.info(
                f"Returning response for processing error: {json.dumps(response)}"
            )
            return response

    except Exception as e:
        logger.error(f"Error processing document upload: {str(e)}")
        logger.error(traceback.format_exc())
        response = {
            "status": "error",
            "error": "Upload failed",
            "message": f"Error processing document: {str(e)}",
        }
        logger.info(f"Returning response for general error: {json.dumps(response)}")
        return response


@router.get("/checklist")
async def get_checklist() -> JSONResponse:
    """Get the default IAS 40 checklist questions."""
    try:
        checklist = load_checklist()
        return JSONResponse(status_code=200, content=checklist)
    except Exception as e:
        logger.error(f"Error loading checklist: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to load checklist", "message": str(e)},
        )


@router.get("/frameworks", response_model=None)
async def get_frameworks() -> Union[Dict[str, Any], JSONResponse]:
    """
    Get the list of available frameworks and their standards, dynamically
    filtered to only those with checklist files present.
    """
    try:
        # Load frameworks from frameworks.json
        frameworks_data = get_available_frameworks()
        filtered_frameworks = []
        checklist_base = Path(__file__).parent.parent / "checklist_data" / "frameworks"
        for fw in frameworks_data.get("frameworks", []):
            fw_id = fw["id"]
            fw_dir = checklist_base / fw_id
            if not fw_dir.exists() or not fw_dir.is_dir():
                continue  # Skip frameworks with no directory
            filtered_standards = []
            for std in fw.get("standards", []):
                std_id = std["id"]
                std_file = fw_dir / f"{std_id}.json"

                # Note: All standards (IFRS and IAS) are now consolidated in IFRS
                # directory

                if std_file.exists():
                    filtered_standards.append(std)
            if filtered_standards:
                fw_copy = fw.copy()
                fw_copy["standards"] = filtered_standards
                filtered_frameworks.append(fw_copy)
        return {"frameworks": filtered_frameworks}
    except Exception as e:
        logger.error(f"Error getting frameworks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.post("/suggest-standards", response_model=None)
async def suggest_accounting_standards(request: Dict[str, Any]) -> Union[Dict[str, Any], JSONResponse]:
    """
    Suggest relevant accounting standards based on company metadata and selected framework.
    
    Expected request body:
    {
        "framework": "IFRS",
        "company_name": "ALDAR Properties PJSC", 
        "nature_of_business": "Real estate development...",
        "operational_demographics": "United Arab Emirates, Egypt",
        "financial_statements_type": "Consolidated"
    }
    """
    try:
        # Validate required fields
        required_fields = ["framework", "company_name", "nature_of_business", 
                          "operational_demographics", "financial_statements_type"]
        
        for field in required_fields:
            if field not in request:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        framework = request["framework"]
        company_name = request["company_name"] 
        nature_of_business = request["nature_of_business"]
        operational_demographics = request["operational_demographics"]
        financial_statements_type = request["financial_statements_type"]
        
        # Get available standards for the framework
        frameworks_data = get_available_frameworks()
        framework_data = None
        for fw in frameworks_data.get("frameworks", []):
            if fw["id"] == framework:
                framework_data = fw
                break
        
        if not framework_data:
            raise HTTPException(status_code=400, detail=f"Framework '{framework}' not found")
        
        # Extract available standard IDs and names
        available_standards = []
        standards_map = {}  # Map ID to name for lookup
        for std in framework_data.get("standards", []):
            available_standards.append(std["id"])
            standards_map[std["id"]] = std.get("name", std["id"])
        
        if not available_standards:
            raise HTTPException(status_code=400, detail=f"No standards available for framework '{framework}'")
        
        # Create AI prompt for standards suggestion
        # Create the prompt directly since there's an issue with the AIPrompts method
        available_standards_list = "\\n".join([f"- {std} ({standards_map[std]})" for std in available_standards])
        
        system_prompt = (
            "You are a financial standards recommendation AI. You MUST respond with valid JSON only - no text before or after the JSON.\\n\\n"
            "Your task: Analyze company profiles and return JSON with recommended accounting standards.\\n\\n"
            "CRITICAL: Your response must be valid JSON that starts with { and ends with }. No markdown, no explanations, no code blocks. Keep reasoning concise (max 80 characters)."
        )
        
        user_prompt = (
            f"Company: {company_name}\\n"
            f"Business: {nature_of_business}\\n"
            f"Location: {operational_demographics}\\n"
            f"Framework: {framework}\\n"
            f"Statement Type: {financial_statements_type}\\n\\n"
            f"Available Standards: {available_standards_list}\\n\\n"
            "INSTRUCTIONS:\\n"
            "1. Analyze the company profile and suggest 6-10 most relevant standards\\n"
            "2. Include core universal standards (IAS 1, IAS 7) that apply to all companies\\n"
            "3. Add industry-specific standards based on business nature\\n"
            "4. Use EXACT standard IDs from Available Standards list\\n"
            "5. Include the full standard title in your response\\n"
            "6. Keep reasoning brief and specific (max 80 characters per reason)\\n\\n"
            "Return JSON with this exact structure:\\n"
            "{\\n"
            '  "suggested_standards": [\\n'
            '    {"standard_id": "IAS 1", "standard_title": "IAS 1 - Presentation of Financial Statements", "relevance_score": 0.95, "reasoning": "Financial statement presentation - mandatory"},\\n'
            '    {"standard_id": "IAS 7", "standard_title": "IAS 7 - Statement of Cash Flows", "relevance_score": 0.95, "reasoning": "Cash flow statements - mandatory"}\\n'
            '  ],\\n'
            '  "priority_level": "high",\\n'
            '  "business_context": "Brief analysis summary"\\n'
            "}"
        )
        
        prompt_data = {
            "system": system_prompt,
            "user": user_prompt
        }
        
        # Call AI service to get suggestions
        ai_service = get_ai_service()
        
        # Use the OpenAI client directly for this simple call
        response = ai_service.openai_client.chat.completions.create(
            model=ai_service.deployment_name,
            messages=[
                {"role": "system", "content": prompt_data["system"]},
                {"role": "user", "content": prompt_data["user"]}
            ],
            max_completion_tokens=8000  # Much higher token limit to prevent truncation
        )
        
        ai_response = response.choices[0].message.content
        
        # Parse AI response - ensure it's not None
        if not ai_response:
            raise HTTPException(status_code=500, detail="AI response is empty")
        
        logger.info(f"Raw AI response: {ai_response}")
        
        # Clean the response - remove any markdown code blocks
        cleaned_response = ai_response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response.replace("```json", "").replace("```", "").strip()
        elif cleaned_response.startswith("```"):
            cleaned_response = cleaned_response.replace("```", "").strip()
            
        logger.info(f"Cleaned AI response: {cleaned_response}")
        
        # Parse JSON - no fallback, just fail if it doesn't work
        suggestions_data = json.loads(cleaned_response)
        
        # Validate the response structure
        if "suggested_standards" not in suggestions_data:
            raise HTTPException(status_code=500, detail="AI response missing 'suggested_standards' field")
        
        # Filter suggestions to only include available standards and add titles if missing
        valid_suggestions = []
        for suggestion in suggestions_data["suggested_standards"]:
            standard_id = suggestion.get("standard_id")
            if standard_id in available_standards:
                # Ensure we have a title - use AI provided title or lookup from standards_map
                standard_title = suggestion.get("standard_title") or standards_map.get(standard_id, standard_id)
                
                valid_suggestion = {
                    "standard_id": standard_id,
                    "standard_title": standard_title,
                    "relevance_score": suggestion.get("relevance_score", 0.8),
                    "reasoning": suggestion.get("reasoning", "Recommended for your business profile")
                }
                valid_suggestions.append(valid_suggestion)
        
        result = {
            "framework": framework,
            "metadata_used": {
                "company_name": company_name,
                "nature_of_business": nature_of_business[:100] + "..." if len(nature_of_business) > 100 else nature_of_business,
                "operational_demographics": operational_demographics,
                "financial_statements_type": financial_statements_type
            },
            "suggested_standards": valid_suggestions,
            "priority_level": suggestions_data.get("priority_level", "medium"),
            "business_context": suggestions_data.get("business_context", ""),
            "total_available_standards": len(available_standards),
            "suggestions_count": len(valid_suggestions)
        }
        
        logger.info(f"Generated {len(valid_suggestions)} accounting standards suggestions for {framework} framework")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error suggesting accounting standards: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.get("/progress/{document_id}", response_model=None)
async def get_analysis_progress(
    document_id: str,
) -> Union[Dict[str, Any], JSONResponse]:
    """
    Get real-time analysis progress for a document including question counts and elapsed time.
    """
    try:
        # FIRST: Check if analysis is completed (highest priority)
        completed_file = (
            Path(__file__).parent.parent
            / "analysis_results"
            / f"{document_id}.completed"
        )
        results_file = (
            Path(__file__).parent.parent / "analysis_results" / f"{document_id}.json"
        )

        if completed_file.exists() and results_file.exists():
            # Analysis is completed, cleanup any stale progress data and return
            # completed status
            from services.progress_tracker import get_progress_tracker

            tracker = get_progress_tracker()
            tracker.cleanup_analysis(document_id)

            return {
                "document_id": document_id,
                "status": "COMPLETED",
                "overall_progress": {
                    "percentage": 100.0,
                    "elapsed_time_seconds": 0.0,
                    "elapsed_time_formatted": "Complete",
                    "completed_standards": 2,
                    "total_standards": 2,
                    "current_standard": "Analysis Complete",
                },
                "percentage": 100,  # For backwards compatibility
                "currentStandard": "Analysis Complete",
                "completedStandards": 2,
                "totalStandards": 2,
                "completed": True,
            }

        # SECOND: Check progress tracker for active analysis
        from services.progress_tracker import get_progress_tracker

        tracker = get_progress_tracker()
        progress = tracker.get_progress(document_id)

        if not progress:
            # Check if analysis is running by looking for lock file
            processing_lock_file = (
                Path(__file__).parent.parent
                / "analysis_results"
                / f"{document_id}_processing.lock"
            )

            if processing_lock_file.exists():
                # Analysis is running but progress tracker not working, return fallback
                # data
                return {
                    "document_id": document_id,
                    "status": "PROCESSING",
                    "overall_progress": {
                        "percentage": 15.0,  # Default progress
                        "elapsed_time_seconds": 60.0,
                        "elapsed_time_formatted": "1m 0s",
                        "completed_standards": 0,
                        "total_standards": 2,
                        "current_standard": "Analysis in progress...",
                    },
                    "percentage": 15,  # For backwards compatibility
                    "currentStandard": "Analysis in progress...",
                    "completedStandards": 0,
                    "totalStandards": 2,
                }
            else:
                return JSONResponse(
                    status_code=404,
                    content={
                        "error": "Progress not found",
                        "detail": f"No progress found for document {document_id}",
                    },
                )

        # Format response with detailed progress information
        response_data = {
            "document_id": document_id,
            "status": progress.status,
            "processing_mode": progress.processing_mode,  # Add processing mode to response
            "overall_progress": {
                "percentage": round(progress.overall_progress_percentage, 1),
                "elapsed_time_seconds": round(progress.overall_elapsed_time, 1),
                "elapsed_time_formatted": _format_elapsed_time(
                    progress.overall_elapsed_time
                ),
                "completed_standards": progress.completed_standards,
                "total_standards": progress.total_standards,
                "current_standard": progress.current_standard,
            },
            "questions": {
                "total": progress.total_questions,
                "completed": progress.completed_questions,
                "remaining": progress.total_questions - progress.completed_questions,
            },
            "standards_detail": [],
        }

        # Add detailed progress for each standard
        if progress.standards_progress:
            for std_id, std_progress in progress.standards_progress.items():
                standard_detail = {
                    "standard_id": std_id,
                    "standard_name": std_progress.standard_name,
                    "status": std_progress.status,
                    "progress_percentage": round(std_progress.progress_percentage, 1),
                    "completed_questions": std_progress.completed_questions,
                    "total_questions": std_progress.total_questions,
                    "current_question": std_progress.current_question,
                    "elapsed_time_seconds": round(std_progress.elapsed_time, 1),
                    "elapsed_time_formatted": _format_elapsed_time(
                        std_progress.elapsed_time
                    ),
                    "questions_progress": [],
                }

                # Add individual question progress with tick marks
                if std_progress.questions_progress:
                    for q_id, q_progress in std_progress.questions_progress.items():
                        question_detail = {
                            "id": q_progress.question_id,
                            "section": q_progress.section,
                            "question": q_progress.question_text,
                            "status": q_progress.status,  # pending, processing, completed, failed
                            "completed_at": q_progress.completed_at,
                            "tick_mark": (
                                "✅"
                                if q_progress.status == "completed"
                                else (
                                    "🔄"
                                    if q_progress.status == "processing"
                                    else (
                                        "❌" if q_progress.status == "failed" else "⏳"
                                    )
                                )
                            ),
                        }
                        standard_detail["questions_progress"].append(question_detail)

                response_data["standards_detail"].append(standard_detail)

        # Add backward compatibility fields
        response_data["percentage"] = response_data["overall_progress"]["percentage"]
        response_data["currentStandard"] = response_data["overall_progress"][
            "current_standard"
        ]
        response_data["completedStandards"] = response_data["overall_progress"][
            "completed_standards"
        ]
        response_data["totalStandards"] = response_data["overall_progress"][
            "total_standards"
        ]

        return response_data

    except Exception as e:
        logger.error(f"Error getting analysis progress for {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.get("/rate-limit-status", response_model=None)
async def get_rate_limit_status() -> Dict[str, Any]:
    """
    Get current rate limiting status and system health metrics.
    """
    try:
        from services.ai import get_rate_limit_status

        status = get_rate_limit_status()

        # Add system health information
        health_status = "healthy"
        if status["circuit_breaker_open"]:
            health_status = "circuit_breaker_open"
        elif status["consecutive_failures"] > 5:
            health_status = "degraded"
        elif status["requests_used"] / status["requests_limit"] > 0.8:
            health_status = "near_limit"

        return {
            "timestamp": datetime.now().isoformat(),
            "health_status": health_status,
            "rate_limiting": {
                "requests": {
                    "used": status["requests_used"],
                    "limit": status["requests_limit"],
                    "percentage": round(
                        (status["requests_used"] / status["requests_limit"]) * 100, 1
                    ),
                    "remaining": status["requests_limit"] - status["requests_used"],
                },
                "tokens": {
                    "used": status["tokens_used"],
                    "limit": status["tokens_limit"],
                    "percentage": round(
                        (status["tokens_used"] / status["tokens_limit"]) * 100, 1
                    ),
                    "remaining": status["tokens_limit"] - status["tokens_used"],
                },
                "window": {
                    "elapsed_seconds": round(status["window_elapsed"], 1),
                    "remaining_seconds": round(status["window_remaining"], 1),
                },
            },
            "circuit_breaker": {
                "is_open": status["circuit_breaker_open"],
                "consecutive_failures": status["consecutive_failures"],
                "failure_threshold": 10,  # From our configuration
            },
            "processing": {
                "processed_questions_count": status["processed_questions_count"],
                "duplicate_prevention_active": True,
            },
        }

    except Exception as e:
        logger.error(f"Error getting rate limit status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


def _format_elapsed_time(seconds: float) -> str:
    """Format elapsed time in a human-readable format"""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes}m {remaining_seconds}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        return f"{hours}h {remaining_minutes}m"


def _validate_framework_exists(framework: str) -> Optional[JSONResponse]:
    """Validate that the framework exists."""
    frameworks = get_available_frameworks()
    framework_exists = False
    for fw in frameworks.get("frameworks", []):
        if fw["id"] == framework:
            framework_exists = True
            break

    if not framework_exists:
        logger.error(f"Framework {framework} does not exist")
        return JSONResponse(
            status_code=400,
            content={
                "error": "Framework not available",
                "detail": f"Framework {framework} is not available",
            },
        )
    return None


def _validate_standards_available(
    framework: str, standards: list
) -> Optional[JSONResponse]:
    """Validate that all standards are available for the framework."""
    unavailable_standards = []
    for std in standards:
        if not is_standard_available(framework, std):
            unavailable_standards.append(std)

    if unavailable_standards:
        logger.error(f"Standards not available: {unavailable_standards}")
        return JSONResponse(
            status_code=400,
            content={
                "error": "Standard(s) not available",
                "detail": (
                    f"The following standard(s) are not available for "
                    f"framework {framework}: "
                    f"{', '.join(unavailable_standards)}"
                ),
            },
        )
    return None


def _validate_document_exists(document_id: str) -> Optional[JSONResponse]:
    """Validate that the document exists."""
    results_path = ANALYSIS_RESULTS_DIR / f"{document_id}.json"
    if not results_path.exists():
        logger.error(f"Document {document_id} not found")
        return JSONResponse(
            status_code=404,
            content={
                "error": "Document not found",
                "detail": f"Document with ID {document_id} not found",
            },
        )
    return None


def _extract_document_text(document_id: str) -> Union[str, JSONResponse]:
    """Extract text from document file or chunks."""
    file_path = get_document_file_path(document_id)

    if file_path and file_path.exists():
        return _extract_text_from_file(file_path)
    else:
        return _extract_text_from_chunks(document_id)


def _extract_text_from_file(file_path: Path) -> Union[str, JSONResponse]:
    """Extract text from PDF or DOCX file."""
    if file_path.suffix.lower() == ".pdf":
        # Use PyMuPDF (fitz) instead of document_extractor for consistency
        try:
            import fitz
            text = ""
            with fitz.open(str(file_path)) as doc:
                for page in doc:
                    page_text = page.get_text()  # type: ignore[attr-defined]
                    if page_text:
                        text += page_text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": "PDF text extraction failed", "detail": str(e)}
            )
    elif file_path.suffix.lower() == ".docx":
        if DocxDocument is None:
            logger.error("DOCX support not available (python-docx not installed)")
            return JSONResponse(
                status_code=400,
                content={
                    "error": "DOCX support not available",
                    "detail": "python-docx package is not installed",
                },
            )
        text = "\n".join(
            [p.text for p in DocxDocument(str(file_path)).paragraphs if p.text.strip()]
        )
        return text
    else:
        logger.error(f"Unsupported file extension for document: {file_path.suffix}")
        return JSONResponse(
            status_code=400,
            content={
                "error": "Unsupported file type",
                "detail": f"File extension {file_path.suffix} is not supported",
            },
        )


def _extract_text_from_chunks(document_id: str) -> Union[str, JSONResponse]:
    """Extract text from chunk data."""
    chunks_path = Path("vector_indices") / f"{document_id}_chunks.json"
    if chunks_path.exists():
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        text = "\n".join(chunk["text"] for chunk in chunks if "text" in chunk)
        return text
    else:
        logger.error(
            f"Document file and chunk data not found for document_id: {document_id}"
        )
        return JSONResponse(
            status_code=404,
            content={
                "error": "File not found",
                "detail": "Document file and chunk data not found",
            },
        )


# Define framework mappings
FRAMEWORK_MAP = {
    "International Financial Reporting Standards (IFRS)": {
        "id": "IFRS",
        "name": "International Financial Reporting Standards",
        "description": "Global accounting standards issued by the IASB.",
    },
    "International Public Sector Accounting Standards (IPSAS)": {
        "id": "IPSAS",
        "name": "International Public Sector Accounting Standards",
        "description": (
            "Global public sector accounting standards issued by the " "IPSAS Board."
        ),
    },
}
FRAMEWORK_MAP = {
    "International Financial Reporting Standards (IFRS)": {
        "id": "IFRS",
        "name": "International Financial Reporting Standards",
        "description": "Global accounting standards issued by the IASB.",
    },
    "International Public Sector Accounting Standards (IPSAS)": {
        "id": "IPSAS",
        "name": "International Public Sector Accounting Standards",
        "description": (
            "Global public sector accounting standards issued by the " "IPSAS Board."
        ),
    },
}


@router.get("/checklist/{framework}/{standard}", response_model=None)
async def get_framework_checklist(
    framework: str, standard: str
) -> Union[Dict[str, Any], JSONResponse]:
    """Get the checklist for a specific framework and standard."""
    try:
        # Check if standard is available
        if not is_standard_available(framework, standard):
            return JSONResponse(
                status_code=404,
                content={
                    "error": "Standard not available",
                    "message": (
                        f"The standard {standard} for framework {framework} "
                        "is not available"
                    ),
                },
            )

        checklist = load_checklist(framework, standard)
        return JSONResponse(status_code=200, content=checklist)
    except FileNotFoundError as e:
        logger.error(f"Checklist not found: {str(e)}")
        return JSONResponse(
            status_code=404, content={"error": "Checklist not found", "message": str(e)}
        )
    except Exception as e:
        logger.error(f"Error loading checklist: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to load checklist", "message": str(e)},
        )


@router.get("/metadata/fields")
async def get_metadata_fields() -> JSONResponse:
    """Get the metadata extraction fields."""
    try:
        metadata_path = os.path.join(CHECKLIST_DATA_DIR, "company_metadata.json")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return JSONResponse(status_code=200, content=metadata["metadata_fields"])
    except Exception as e:
        logger.error(f"Error loading metadata fields: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to load metadata fields", "message": str(e)},
        )


@router.get("/documents/{document_id}", response_model=None)
async def get_document_status(document_id: str) -> Union[Dict[str, Any], JSONResponse]:
    """Get document status and metadata."""
    logger.info(f"🔍 GET /documents/{document_id} - Starting request")
    try:
        logger.info(f"🔍 Checking results path for document: {document_id}")
        # Check if analysis results exist
        results_path = os.path.join(ANALYSIS_RESULTS_DIR, f"{document_id}.json")
        logger.info(f"🔍 Results path: {results_path}")
        if os.path.exists(results_path):
            # Read results
            with open(results_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            # Check for errors
            if "error" in results:
                return {
                    "document_id": document_id,
                    "status": "FAILED",
                    "error": results.get("error", "Unknown error"),
                    "message": results.get("message", "Analysis failed"),
                    "metadata": results.get("metadata", {}),
                }
            # Normalize status values
            status = results.get("status", "PROCESSING")
            if status == "awaiting_framework_selection":
                status = "awaiting_framework_selection"
            elif status.lower() == "completed":
                status = "COMPLETED"
            elif status.lower() == "failed" or status.lower() == "error":
                status = "FAILED"
            else:
                status = "PROCESSING"
            metadata_extraction = results.get("metadata_extraction", "PENDING")
            compliance_analysis = results.get("compliance_analysis", "PENDING")
            return {
                "document_id": document_id,
                "status": status,
                "metadata_extraction": metadata_extraction,
                "compliance_analysis": compliance_analysis,
                "processing_mode": results.get("processing_mode", "smart"),
                "metadata": results.get("metadata", {}),
                "sections": results.get("sections", []),
                "progress": results.get("progress", {}),
                "framework": results.get("framework"),
                "standards": results.get("standards", []),
                "specialInstructions": results.get("specialInstructions"),
                "extensiveSearch": results.get("extensiveSearch", False),
                "message": results.get("message", "Analysis in progress"),
            }
        # If no results, check if the document was even uploaded
        file_path = get_document_file_path(document_id)
        if not file_path:
            logger.warning(f"Document file not found: {document_id}")
            return JSONResponse(
                status_code=404,
                content={
                    "error": "Document not found",
                    "message": f"Document with ID {document_id} not found",
                },
            )
        # Document uploaded but no analysis results yet
        return {
            "document_id": document_id,
            "status": "PENDING",
            "metadata_extraction": "PENDING",
            "compliance_analysis": "PENDING",
            "metadata": {},
            "message": "Document uploaded, analysis not started yet",
        }
    except Exception as e:
        logger.error(f"Error getting document status: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Server error",
                "message": f"Failed to get document status: {str(e)}",
            },
        )


@router.get("/documents/{document_id}/results")
async def get_document_results(document_id: str) -> Dict[str, Any]:
    """Get the results of a document analysis."""
    try:
        results_path = ANALYSIS_RESULTS_DIR / f"{document_id}.json"
        if not os.path.exists(results_path):
            return {
                "status": "not_found",
                "document_id": document_id,
                "message": "No results found for the specified document",
            }

        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        # Ensure consistent response structure
        return {
            "status": results.get("status", "unknown"),
            "document_id": document_id,
            "metadata": results.get("metadata", {}),
            "sections": results.get("sections", []),
            "message": (
                "Document analysis completed"
                if results.get("status") == "completed"
                else "Document analysis in progress"
            ),
        }
    except Exception as e:
        logger.error(f"Error retrieving document results: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "document_id": document_id,
            "message": f"Error retrieving document results: {str(e)}",
        }


class ChecklistItemUpdate(BaseModel):
    status: str
    comments: Optional[str] = None


@router.patch("/documents/{document_id}/items/{item_id}")
async def update_compliance_item(
    document_id: str, item_id: str, update: ChecklistItemUpdate
):
    """Update a compliance checklist item."""
    try:
        results_path = ANALYSIS_RESULTS_DIR / f"{document_id}.json"
        if not os.path.exists(results_path):
            return JSONResponse(
                status_code=404,
                content={
                    "error": "Document not found",
                    "message": "No results found for the specified document",
                },
            )

        # Read current results
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        # Update the specific item
        item_updated = False
        for section in results.get("sections", []):
            for item in section.get("items", []):
                if item.get("id") == item_id:
                    item["status"] = update.status
                    if update.comments is not None:
                        item["comments"] = update.comments
                    item_updated = True
                    break
            if item_updated:
                break

        if not item_updated:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "Item not found",
                    "message": "No item found with the specified ID",
                },
            )

        # Save updated results
        save_analysis_results(document_id, results)

        return JSONResponse(
            status_code=200, content={"message": "Item updated successfully"}
        )
    except Exception as e:
        logger.error(f"Error updating compliance item: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "error": "Update failed",
                "message": f"Error updating compliance item: {str(e)}",
            },
        )


class FrameworkSelectionRequest(BaseModel):
    framework: str
    standards: list[str]  # Accept a list of standards
    specialInstructions: str = ""  # Optional special instructions from user
    extensiveSearch: bool = False  # Optional flag for extensive analysis
    processingMode: str = "smart"  # Processing mode: "zap" | "smart" | "comparison"


class ProcessingModeRequest(BaseModel):
    processing_mode: str  # "zap" | "smart" | "comparison"
    comparison_settings: Optional[Dict[str, Any]] = None
    user_preferences: Optional[Dict[str, float]] = None


class ComplianceAnalysisRequest(BaseModel):
    mode: str = "smart"  # "zap" | "smart" | "comparison"
    special_instructions: str = ""
    comparison_config: Optional[Dict[str, Any]] = None


@router.post("/documents/{document_id}/select-framework", response_model=None)
async def select_framework(
    document_id: str,
    request: FrameworkSelectionRequest,
    background_tasks: BackgroundTasks,
    ai_svc: AIService = Depends(get_ai_service),
) -> Union[Dict[str, Any], JSONResponse]:
    try:
        logger.info(
            f"Selecting framework {request.framework} and standards "
            f"{request.standards} for document {document_id}"
        )

        # Validate framework exists
        framework_error = _validate_framework_exists(request.framework)
        if framework_error:
            return framework_error

        # Validate standards are available
        standards_error = _validate_standards_available(
            request.framework, request.standards
        )
        if standards_error:
            return standards_error

        # Validate document exists
        document_error = _validate_document_exists(document_id)
        if document_error:
            return document_error

        # Read and update analysis results
        results_path = ANALYSIS_RESULTS_DIR / f"{document_id}.json"
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        # Update results with framework selection
        results.update(
            {
                "framework": request.framework,
                "standards": request.standards,
                "specialInstructions": request.specialInstructions,
                "extensiveSearch": request.extensiveSearch,
                "compliance_analysis": "PENDING",
                "status": "PROCESSING",
            }
        )

        # Build status message
        instructions_msg = (
            f" with special instructions: {request.specialInstructions}"
            if request.specialInstructions
            else ""
        )
        extensive_msg = " (extensive search enabled)" if request.extensiveSearch else ""
        standards_list = ", ".join(request.standards)
        results["message"] = (
            f"Framework {request.framework} and standards {standards_list} "
            f"selected{instructions_msg}{extensive_msg}, compliance analysis pending"
        )

        save_analysis_results(document_id, results)

        # Extract document text
        text = _extract_document_text(document_id)
        if isinstance(text, JSONResponse):
            return text

        # Start compliance analysis in the background
        background_tasks.add_task(
            process_compliance_analysis,
            document_id,
            text,
            request.framework,
            request.standards,
            request.specialInstructions,
            request.extensiveSearch,
            ai_svc,
            request.processingMode,  # Pass the processing mode
        )

        return {
            "status": "PROCESSING",
            "document_id": document_id,
            "framework": request.framework,
            "standards": request.standards,
            "specialInstructions": request.specialInstructions,
            "extensiveSearch": request.extensiveSearch,
            "message": (
                "Framework selection successful, compliance analysis started "
                "for all selected standards"
            ),
        }
    except Exception as e:
        logger.error(f"Error selecting framework: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Server error",
                "detail": f"Error selecting framework: {str(e)}",
            },
        )


# Add an alias endpoint for framework selection to match the frontend call
@router.post("/documents/{document_id}/framework")
async def select_framework_alias(
    document_id: str,
    request: FrameworkSelectionRequest,
    background_tasks: BackgroundTasks,
    ai_svc: AIService = Depends(get_ai_service),
):
    """
    Alternative endpoint for framework selection.
    Calls the existing select_framework function.
    """
    return await select_framework(document_id, request, background_tasks, ai_svc)


@router.post("/documents/{document_id}/select-processing-mode")
async def select_processing_mode(
    document_id: str,
    request: ProcessingModeRequest,
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Select processing mode for compliance analysis.
    Modes: zap (fast), smart (intelligent), comparison (benchmark)
    """
    try:
        logger.info(
            f"Selecting processing mode {request.processing_mode} for "
            f"document {document_id}"
        )

        # Validate processing mode
        valid_modes = ["zap", "smart", "comparison"]
        if request.processing_mode not in valid_modes:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Invalid processing mode",
                    "detail": f"Mode must be one of: {', '.join(valid_modes)}",
                },
            )

        # Load existing results
        results_path = ANALYSIS_RESULTS_DIR / f"{document_id}.json"
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                results = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return JSONResponse(
                status_code=404,
                content={
                    "error": "Document not found",
                    "detail": f"Document {document_id} not found or results corrupted",
                },
            )

        # Update results with processing mode selection
        results["processing_mode"] = request.processing_mode
        results["comparison_settings"] = request.comparison_settings or {}
        results["user_preferences"] = request.user_preferences or {}
        results["mode_selection_timestamp"] = datetime.now().isoformat()

        # If metadata extraction is not completed, trigger it now
        if results.get("metadata_extraction") != "COMPLETED":
            logger.info(f"Triggering metadata extraction for document {document_id}")
            results["status"] = "PROCESSING"
            results["metadata_extraction"] = "PROCESSING"
            results["message"] = (
                f"Processing mode '{
                    request.processing_mode}' selected, starting metadata extraction"
            )

            # Save updated results
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            # Start metadata extraction in background
            background_tasks = BackgroundTasks()
            ai_svc = get_ai_service()
            background_tasks.add_task(
                process_upload_tasks, document_id=document_id, ai_svc=ai_svc
            )

            mode_descriptions = {
                "zap": "Lightning fast analysis with 16 parallel workers",
                "smart": (
                    "AI-powered intelligent semantic processing with cost "
                    "optimization"
                ),
                "comparison": "Performance benchmark running both Zap and Smart modes",
            }

            return {
                "success": True,
                "processing_mode": request.processing_mode,
                "description": mode_descriptions[request.processing_mode],
                "status": "PROCESSING",
                "message": (
                    f"Processing mode '{request.processing_mode}' selected, "
                    "metadata extraction started"
                ),
            }
        else:
            # Metadata already extracted, ready for framework selection
            results["status"] = "awaiting_framework_selection"
            results["message"] = (
                f"Processing mode '{
                    request.processing_mode}' selected, ready for framework selection"
            )

            # Save updated results
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            mode_descriptions = {
                "zap": "Lightning fast analysis with 16 parallel workers",
                "smart": (
                    "AI-powered intelligent semantic processing with cost "
                    "optimization"
                ),
                "comparison": "Performance benchmark running both Zap and Smart modes",
            }

            return {
                "success": True,
                "processing_mode": request.processing_mode,
                "description": mode_descriptions[request.processing_mode],
                "status": "awaiting_framework_selection",
                "message": (
                    f"Processing mode '{request.processing_mode}' selected, "
                    "ready for framework selection"
                ),
            }

    except Exception as e:
        logger.error(f"Error selecting processing mode: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Server error",
                "detail": f"Error selecting processing mode: {str(e)}",
            },
        )


@router.post("/documents/{document_id}/start-compliance")
async def start_compliance_analysis(
    document_id: str,
    request: ComplianceAnalysisRequest,
    background_tasks: BackgroundTasks,
    ai_svc: AIService = Depends(get_ai_service),
):
    """
    Start compliance analysis with specified processing mode.
    This endpoint allows explicit control over when analysis starts.
    """
    try:
        logger.info(
            f"Starting compliance analysis for document {document_id} "
            f"with mode {request.mode}"
        )

        # Load existing results
        results_path = ANALYSIS_RESULTS_DIR / f"{document_id}.json"
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                results = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return JSONResponse(
                status_code=404,
                content={
                    "error": "Document not found",
                    "detail": f"Document {document_id} not found or results corrupted",
                },
            )

        # Check if framework and standards are selected
        if "framework" not in results or "standards" not in results:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Framework not selected",
                    "detail": (
                        "Please select framework and standards before "
                        "starting analysis"
                    ),
                },
            )

        # Update processing mode if provided
        processing_mode = request.mode or results.get("processing_mode", "smart")
        results["processing_mode"] = processing_mode
        results["special_instructions"] = request.special_instructions
        results["comparison_config"] = request.comparison_config or {}

        # Start compliance analysis based on processing mode
        framework = results["framework"]
        standards = results["standards"]
        text_content = results.get("text", "")

        if processing_mode == "comparison":
            # For comparison mode, run both Smart and Zap modes
            background_tasks.add_task(
                process_compliance_comparison,
                document_id,
                text_content,
                framework,
                standards,
                request.special_instructions,
                results.get("extensiveSearch", False),
                ai_svc,
            )
        else:
            # For single mode (zap or smart)
            background_tasks.add_task(
                process_compliance_analysis,
                document_id,
                text_content,
                framework,
                standards,
                request.special_instructions,
                results.get("extensiveSearch", False),
                ai_svc,
                processing_mode,  # Pass processing mode parameter
            )

        return {
            "success": True,
            "processing_mode": processing_mode,
            "message": f"Compliance analysis started with {processing_mode} mode",
        }

    except Exception as e:
        logger.error(f"Error starting compliance analysis: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Server error",
                "detail": f"Error starting compliance analysis: {str(e)}",
            },
        )


async def _run_smart_mode_comparison(
    document_id: str,
    text: str,
    framework: str,
    standards: list,
    special_instructions: str,
    extensive_search: bool,
    ai_svc: AIService,
) -> tuple[dict, dict]:
    """Run Smart Mode analysis and return metrics and results."""
    logger.info(f"Running Smart Mode for comparison - document {document_id}")
    start_time = time.time()

    try:
        await process_compliance_analysis_internal(
            document_id + "_smart_comparison",
            text,
            framework,
            standards,
            special_instructions,
            extensive_search,
            ai_svc,
            "smart",
        )
        end_time = time.time()
        processing_time = end_time - start_time

        # Load Smart Mode results
        results_path = ANALYSIS_RESULTS_DIR / f"{document_id}_smart_comparison.json"
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                results = json.load(f)
        except FileNotFoundError:
            results = {"sections": []}

        # Extract Smart Mode metrics
        metrics = {
            "processing_time_seconds": round(processing_time, 2),
            "questions_processed": len(results.get("sections", [])),
            "success": True,
            "sections_analyzed": len(results.get("sections", [])),
            "error": None,
        }

        return metrics, results

    except Exception as e:
        logger.error(f"Smart mode failed in comparison: {str(e)}")
        end_time = time.time()
        processing_time = end_time - start_time

        metrics = {
            "processing_time_seconds": round(processing_time, 2),
            "questions_processed": 0,
            "success": False,
            "sections_analyzed": 0,
            "error": str(e),
        }
        results = {"sections": []}

        return metrics, results


async def _run_zap_mode_comparison(
    document_id: str,
    text: str,
    framework: str,
    standards: list,
    special_instructions: str,
    extensive_search: bool,
    ai_svc: AIService,
) -> tuple[dict, dict]:
    """Run Zap Mode analysis and return metrics and results."""
    logger.info(f"Running Zap Mode for comparison - document {document_id}")
    start_time = time.time()

    try:
        await process_compliance_analysis_internal(
            document_id + "_zap_comparison",
            text,
            framework,
            standards,
            special_instructions,
            extensive_search,
            ai_svc,
            "zap",
        )
        end_time = time.time()
        processing_time = end_time - start_time

        # Load Zap Mode results
        results_path = ANALYSIS_RESULTS_DIR / f"{document_id}_zap_comparison.json"
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                results = json.load(f)
        except FileNotFoundError:
            results = {"sections": []}

        # Extract Zap Mode metrics
        metrics = {
            "processing_time_seconds": round(processing_time, 2),
            "questions_processed": len(results.get("sections", [])),
            "success": True,
            "sections_analyzed": len(results.get("sections", [])),
            "error": None,
        }

        return metrics, results

    except Exception as e:
        logger.error(f"Zap mode failed in comparison: {str(e)}")
        end_time = time.time()
        processing_time = end_time - start_time

        metrics = {
            "processing_time_seconds": round(processing_time, 2),
            "questions_processed": 0,
            "success": False,
            "sections_analyzed": 0,
            "error": str(e),
        }
        results = {"sections": []}

        return metrics, results


def _calculate_speed_improvement(smart_metrics: dict, zap_metrics: dict) -> str:
    """Calculate speed improvement between Smart and Zap modes."""
    if zap_metrics["processing_time_seconds"] > 0:
        speed_ratio = (
            smart_metrics["processing_time_seconds"]
            / zap_metrics["processing_time_seconds"]
        )
        if speed_ratio > 1:
            return f"{speed_ratio:.1f}x faster (Zap vs Smart)"
        else:
            return f"{1 / speed_ratio:.1f}x faster (Smart vs Zap)"
    else:
        return "Unable to calculate"


def _determine_mode_recommendation(
    smart_metrics: dict, zap_metrics: dict
) -> tuple[str, str]:
    """Determine which mode to recommend based on performance."""
    smart_time = smart_metrics["processing_time_seconds"]
    zap_time = zap_metrics["processing_time_seconds"]

    if smart_time < zap_time:
        return "smart", "Smart mode was faster"
    elif zap_time < smart_time:
        return "zap", "Zap mode was faster"
    else:
        return "equivalent", "Both modes performed similarly"


def _handle_failed_modes(
    smart_metrics: dict, zap_metrics: dict
) -> tuple[str, str, str]:
    """Handle cases where one or both modes failed."""
    speed_improvement = "Analysis failed"
    if smart_metrics["success"] and not zap_metrics["success"]:
        return speed_improvement, "smart", "Only Smart mode succeeded"
    elif zap_metrics["success"] and not smart_metrics["success"]:
        return speed_improvement, "zap", "Only Zap mode succeeded"
    else:
        return speed_improvement, "neither", "Both modes failed"


def _calculate_comparison_metrics(
    smart_metrics: dict, zap_metrics: dict
) -> tuple[str, str, str]:
    """Calculate performance comparison metrics between Smart and Zap modes."""
    if smart_metrics["success"] and zap_metrics["success"]:
        speed_improvement = _calculate_speed_improvement(smart_metrics, zap_metrics)
        recommendation, reason = _determine_mode_recommendation(
            smart_metrics, zap_metrics
        )
        return speed_improvement, recommendation, reason
    else:
        return _handle_failed_modes(smart_metrics, zap_metrics)


def _build_comparison_results(
    document_id: str,
    framework: str,
    standards: list,
    smart_metrics: dict,
    zap_metrics: dict,
    smart_results: dict,
    zap_results: dict,
    speed_improvement: str,
    recommendation: str,
    reason: str,
) -> dict:
    """Build the final comparison results structure."""
    # Merge results (use Smart mode results as primary if available)
    primary_results = smart_results if smart_metrics["success"] else zap_results
    if not smart_metrics["success"] and not zap_metrics["success"]:
        primary_results = {"sections": [], "status": "FAILED"}

    # Build final comparison results
    results = {
        "document_id": document_id,
        "status": (
            "COMPLETED"
            if (smart_metrics["success"] or zap_metrics["success"])
            else "FAILED"
        ),
        "processing_mode": "comparison",
        "framework": framework,
        "standards": standards,
        "sections": primary_results.get("sections", []),
        "message": f"Comparison analysis completed - {recommendation} mode recommended",
        "comparison_results": {
            "enabled": True,
            "modes_compared": ["smart", "zap"],
            "smart_mode": smart_metrics,
            "zap_mode": zap_metrics,
            "performance_metrics": {
                "speed_improvement": speed_improvement,
                "recommendation": recommendation,
                "recommendation_reason": reason,
                "total_analysis_time": round(
                    smart_metrics["processing_time_seconds"]
                    + zap_metrics["processing_time_seconds"],
                    2,
                ),
            },
        },
    }

    return results


async def process_compliance_comparison(
    document_id: str,
    text: str,
    framework: str,
    standards: list,
    special_instructions: str,
    extensive_search: bool,
    ai_svc: AIService,
) -> None:
    """
    Process compliance analysis using both Smart and Zap modes for comparison.
    Executes both modes sequentially and compares their performance metrics.
    """
    try:
        logger.info(f"Starting comparison mode analysis for document {document_id}")

        # Initialize base results
        initial_results = {
            "document_id": document_id,
            "status": "PROCESSING",
            "processing_mode": "comparison",
            "framework": framework,
            "standards": standards,
            "message": "Running Smart and Zap mode comparison",
            "comparison_results": {
                "enabled": True,
                "modes_compared": ["smart", "zap"],
            },
        }

        # Save initial progress
        results_path = ANALYSIS_RESULTS_DIR / f"{document_id}.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(initial_results, f, indent=2, ensure_ascii=False)

        # Run both modes and gather results
        smart_metrics, smart_results = await _run_smart_mode_comparison(
            document_id,
            text,
            framework,
            standards,
            special_instructions,
            extensive_search,
            ai_svc,
        )

        zap_metrics, zap_results = await _run_zap_mode_comparison(
            document_id,
            text,
            framework,
            standards,
            special_instructions,
            extensive_search,
            ai_svc,
        )

        # Calculate comparison metrics
        speed_improvement, recommendation, reason = _calculate_comparison_metrics(
            smart_metrics, zap_metrics
        )

        # Build final results
        final_results = _build_comparison_results(
            document_id,
            framework,
            standards,
            smart_metrics,
            zap_metrics,
            smart_results,
            zap_results,
            speed_improvement,
            recommendation,
            reason,
        )

        # Save final results
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Comparison analysis completed for document {document_id} - "
            f"{recommendation} mode recommended"
        )

    except Exception as e:
        logger.error(f"Error in comparison analysis: {str(e)}", exc_info=True)
        # Update results with error
        try:
            results_path = ANALYSIS_RESULTS_DIR / f"{document_id}.json"
            error_results = {
                "document_id": document_id,
                "status": "FAILED",
                "error": str(e),
            }
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(error_results, f, indent=2, ensure_ascii=False)
        except Exception as save_error:
            # Failed to save error state - log but continue
            logger.error(
                f"Failed to save error state for {document_id}: {str(save_error)}"
            )


async def process_compliance_analysis_internal(
    document_id: str,
    text: str,
    framework: str,
    standards: list,
    special_instructions: str,
    extensive_search: bool,
    ai_svc: AIService,
    processing_mode: str = "smart",
) -> None:
    """
    Internal function to handle the actual compliance analysis.
    This wraps the existing analysis logic with processing mode awareness.
    """
    # Call the main analysis function with mode parameter
    return await process_compliance_analysis(
        document_id,
        text,
        framework,
        standards,
        special_instructions,
        extensive_search,
        ai_svc,
        processing_mode,
    )


async def process_smart_mode_analysis(
    checklist: Dict[str, Any],
    text: str,
    document_id: str,
    ai_svc: AIService,
    standard: str,
    progress_tracker=None,
) -> List[Dict[str, Any]]:
    """
    Process compliance analysis using Smart Mode with intelligent semantic
    processing and optimized AI usage.

    Smart Mode features:
    - Semantic content analysis for better accuracy
    - Question prioritization based on document content
    - Cost-optimized AI processing
    - Enhanced context understanding
    """
    logger.info(f"Starting Smart Mode analysis for standard {standard}")
    start_time = time.time()

    # Preprocess checklist to add standard numbers to questions
    processed_checklist = _preprocess_checklist_with_standard_numbers(checklist)

    # Set progress tracker for question-level tracking
    if progress_tracker:
        ai_svc.progress_tracker = progress_tracker

        # Initialize question-level tracking for Smart Mode
        all_questions_data = []
        for section in processed_checklist.get("sections", []):
            for item in section.get("items", []):
                all_questions_data.append(
                    {
                        "id": item.get("id"),
                        "section": item.get("section", standard),
                        "question": item.get("question", ""),
                    }
                )
        progress_tracker.initialize_questions(document_id, standard, all_questions_data)

    try:

        # Step 1: Extract and analyze all questions from the checklist
        all_questions = []
        section_question_map = {}

        for section_idx, section in enumerate(processed_checklist.get("sections", [])):
            section_questions = []
            for item in section.get("items", []):
                question = item.get("question", "")
                if question:
                    all_questions.append(question)
                    section_questions.append(question)
            section_question_map[section_idx] = section_questions

        logger.info(f"Smart Mode: Analyzing {len(all_questions)} questions")

        # Step 2: Intelligent text segmentation for semantic analysis
        text_segments = _create_semantic_segments(text)
        logger.info(f"Smart Mode: Created {len(text_segments)} semantic segments")

        # Step 3: Question-content mapping for optimal processing
        question_priorities = _prioritize_questions_by_content(
            all_questions, text_segments
        )

        # Step 4: Process sections with smart AI optimization
        processed_sections = []
        total_tokens_used = 0
        completed_questions = 0

        for section_idx, section in enumerate(processed_checklist.get("sections", [])):
            section_start = time.time()

            # Get questions for this section
            section_questions = section_question_map.get(section_idx, [])

            # Find most relevant text segments for this section's questions
            relevant_segments = _find_relevant_segments(
                section_questions, text_segments, question_priorities
            )

            # Optimize context for AI processing
            optimized_context = _optimize_context_for_ai(relevant_segments, text)

            # Process section with enhanced context
            ai_svc.current_document_id = document_id

            # Enhanced section processing with Smart Mode features
            enhanced_section = await _process_section_smart_mode(
                section, optimized_context, ai_svc, document_id, standard
            )

            # Add Smart Mode metadata
            enhanced_section["processing_mode"] = "smart"
            enhanced_section["processing_time"] = time.time() - section_start
            enhanced_section["segments_analyzed"] = len(relevant_segments)

            processed_sections.append(enhanced_section)

            # Update progress tracking
            completed_questions += len(section_questions)
            if progress_tracker:
                progress_tracker.update_question_progress(
                    document_id,
                    standard,
                    f"Processing section {
                        section_idx + 1}/{
                        len(
                            processed_checklist.get(
                                'sections',
                                []))}",
                    completed_questions,
                )

            # Track token usage (estimated)
            estimated_tokens = len(optimized_context.split()) * 1.3
            total_tokens_used += estimated_tokens

        processing_time = time.time() - start_time
        logger.info(
            f"Smart Mode completed for {standard}: {processing_time:.2f}s, "
            f"~{total_tokens_used:.0f} tokens"
        )

        return processed_sections

    except Exception as e:
        logger.error(f"Smart Mode analysis failed for {standard}: {str(e)}")
        # Fallback to standard processing
        logger.info("Falling back to standard processing mode")
        try:
            section_tasks = []
            for section in processed_checklist.get("sections", []):
                ai_svc.current_document_id = document_id
                section_tasks.append(
                    ai_svc._process_section(
                        section, text, document_id, standard_id=standard
                    )
                )
            return await asyncio.gather(*section_tasks)
        except Exception as fallback_error:
            logger.error(f"Fallback processing also failed: {str(fallback_error)}")
            # Return minimal fallback results
            return [
                {
                    "section_name": section.get("name", f"Section {i + 1}"),
                    "analysis_result": "Analysis unavailable due to processing error",
                    "processing_mode": "fallback",
                    "error": True,
                }
                for i, section in enumerate(processed_checklist.get("sections", []))
            ]


def _create_semantic_segments(text: str) -> List[str]:
    """Create semantic segments from text for intelligent analysis"""
    # Split text into logical segments (paragraphs, sections)
    segments = []

    # Split by double newlines (paragraph breaks)
    paragraphs = text.split("\n\n")

    current_segment = ""
    max_segment_length = 2000  # Optimal size for semantic analysis

    for paragraph in paragraphs:
        if len(current_segment) + len(paragraph) < max_segment_length:
            current_segment += paragraph + "\n\n"
        else:
            if current_segment.strip():
                segments.append(current_segment.strip())
            current_segment = paragraph + "\n\n"

    # Add final segment
    if current_segment.strip():
        segments.append(current_segment.strip())

    return segments


def _prioritize_questions_by_content(
    questions: List[str], segments: List[str]
) -> Dict[str, float]:
    """Prioritize questions based on content relevance"""
    priorities = {}

    # Simple keyword-based prioritization
    for question in questions:
        question_lower = question.lower()
        total_relevance = 0

        for segment in segments:
            segment_lower = segment.lower()

            # Check for direct keyword matches
            question_words = set(question_lower.split())
            segment_words = set(segment_lower.split())
            overlap = len(question_words.intersection(segment_words))

            if overlap > 0:
                total_relevance += overlap / len(question_words)

        priorities[question] = total_relevance

    return priorities


def _find_relevant_segments(
    section_questions: List[str], segments: List[str], priorities: Dict[str, float]
) -> List[str]:
    """Find most relevant text segments for section questions"""
    if not section_questions:
        return segments[:3]  # Return first 3 segments if no questions

    relevant_segments = []
    max_segments = min(5, len(segments))  # Limit for efficiency

    # Score segments based on question relevance
    segment_scores = []
    for i, segment in enumerate(segments):
        score = 0
        for question in section_questions:
            question_priority = priorities.get(question, 0)
            # Simple text overlap scoring
            question_words = set(question.lower().split())
            segment_words = set(segment.lower().split())
            overlap = len(question_words.intersection(segment_words))
            score += overlap * question_priority

        segment_scores.append((score, i, segment))

    # Sort by score and take top segments
    segment_scores.sort(reverse=True)
    for score, _, segment in segment_scores[:max_segments]:
        relevant_segments.append(segment)

    return relevant_segments


def _optimize_context_for_ai(segments: List[str], full_text: str) -> str:
    """Optimize context for AI processing to reduce token usage"""
    if not segments:
        # Fallback to beginning of document
        return full_text[:4000]

    # Combine relevant segments with smart truncation
    combined_context = "\n\n".join(segments)

    # Limit total context size for cost optimization
    max_context_length = 6000
    if len(combined_context) > max_context_length:
        combined_context = combined_context[:max_context_length] + "..."

    return combined_context


def _preprocess_checklist_with_standard_numbers(
    checklist: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Preprocess checklist to add standard numbers to all questions.
    This ensures both Smart Mode and Zap Mode display standard numbers in questions.
    """
    try:
        # Create a copy to avoid modifying the original
        processed_checklist = json.loads(json.dumps(checklist))

        for section in processed_checklist.get("sections", []):
            # Get the standard number from the section itself
            section_standard = section.get("section", "")

            for item in section.get("items", []):
                question = item.get("question", "")
                if (
                    question
                    and section_standard
                    and not question.startswith(section_standard)
                ):
                    item["question"] = f"{section_standard}: {question}"

        return processed_checklist
    except Exception as e:
        logger.error(f"Error preprocessing checklist with standard numbers: {str(e)}")
        return checklist  # Return original on error


async def _process_section_smart_mode(
    section: Dict[str, Any],
    optimized_context: str,
    ai_svc: AIService,
    document_id: str,
    standard_id: str,
) -> Dict[str, Any]:
    """Process a section using Smart Mode with optimized context"""
    # Use the existing _process_section but with optimized context and standard_id
    return await ai_svc._process_section(
        section, optimized_context, document_id, standard_id
    )


async def process_zap_mode_analysis(
    checklist: Dict[str, Any],
    text: str,
    document_id: str,
    ai_svc: AIService,
    standard: str,
    progress_tracker=None,
) -> List[Dict[str, Any]]:
    """
    Process compliance analysis using Zap Mode with 16 concurrent workers
    for maximum speed.

    Zap Mode features:
    - 16 parallel workers for maximum throughput
    - Direct processing without semantic optimization
    - Speed-first approach with acceptable accuracy trade-offs
    - Minimal context processing for fastest results
    """
    logger.info(f"Starting Zap Mode analysis for standard {standard}")
    start_time = time.time()

    # Preprocess checklist to add standard numbers to questions
    processed_checklist = _preprocess_checklist_with_standard_numbers(checklist)

    # Set progress tracker for question-level tracking
    if progress_tracker:
        ai_svc.progress_tracker = progress_tracker

        # Initialize question-level tracking for Zap Mode
        all_questions_data = []
        for section in processed_checklist.get("sections", []):
            for item in section.get("items", []):
                all_questions_data.append(
                    {
                        "id": item.get("id"),
                        "section": item.get("section", standard),
                        "question": item.get("question", ""),
                    }
                )
        progress_tracker.initialize_questions(document_id, standard, all_questions_data)

    try:

        sections = processed_checklist.get("sections", [])
        if not sections:
            logger.warning(f"No sections found in checklist for standard {standard}")
            return []

        # SEMAPHORE REMOVED - Process all 217 questions without worker limits
        # semaphore = asyncio.Semaphore(16)

        async def process_section_unlimited(section):
            # NO SEMAPHORE - Process all sections concurrently
            # Set document ID for this worker
            ai_svc.current_document_id = document_id

            # NO RETRY LIMITS - Process until completion
            max_retries = 10  # Increased from 3 to ensure completion
            for attempt in range(max_retries):
                try:
                    # Process section with minimal optimization for speed
                    result = await ai_svc._process_section(
                        section, text, document_id, standard_id=standard
                    )

                    # Add Zap Mode metadata
                    result["processing_mode"] = "zap"
                    current_task = asyncio.current_task()
                    result["worker_id"] = (
                        current_task.get_name() if current_task else "unknown"
                    )
                    return result

                except Exception as e:
                    error_str = str(e).lower()
                    if (
                        "rate limit" in error_str or "429" in error_str
                    ) and attempt < max_retries - 1:
                        # Simple retry without staggered backoff - just power through
                        logger.warning(
                            f"Zap Mode worker error, retrying immediately (attempt {
                                attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(0.1)  # Minimal delay
                        continue
                    else:
                        # Log error and return error result instead of failing
                        # completely
                        logger.error(
                            f"Zap Mode worker failed after "
                            f"{attempt + 1} attempts: {str(e)}"
                        )
                        current_task = asyncio.current_task()
                        return {
                            "processing_mode": "zap",
                            "section": section.get("section", "unknown"),
                            "title": section.get("title", ""),
                            "items": [],
                            "error": str(e),
                            "worker_id": (
                                current_task.get_name() if current_task else "unknown"
                            ),
                            "retry_attempts": attempt + 1,
                        }

            # Should never reach here, but just in case
            current_task = asyncio.current_task()
            return {
                "processing_mode": "zap",
                "section": section.get("section", "unknown"),
                "title": section.get("title", ""),
                "items": [],
                "error": "Max retries exceeded",
                "worker_id": current_task.get_name() if current_task else "unknown",
            }

        # Create tasks for all sections - NO LIMITS
        logger.info(
            f"Zap Mode: Processing {len(sections)} sections with NO WORKER LIMITS"
        )
        section_tasks = [process_section_unlimited(section) for section in sections]

        # Execute all tasks concurrently - UNLIMITED PROCESSING
        processed_sections = await asyncio.gather(
            *section_tasks, return_exceptions=True
        )

        # Filter out exceptions and log them
        valid_sections = []
        failed_count = 0
        completed_questions = 0

        for i, result in enumerate(processed_sections):
            if isinstance(result, Exception):
                logger.error(f"Zap Mode: Section {i} failed: {str(result)}")
                failed_count += 1
            else:
                valid_sections.append(result)
                # Count questions in this section
                if isinstance(result, dict) and "items" in result:
                    completed_questions += len(result["items"])

                # Update progress tracker
                if progress_tracker:
                    progress_tracker.update_question_progress(
                        document_id,
                        standard,
                        f"Processing section {i + 1}/{len(sections)}",
                        completed_questions,
                    )

        processing_time = time.time() - start_time

        logger.info(
            f"Zap Mode completed for {standard}: {processing_time:.2f}s, "
            f"{len(valid_sections)}/{len(sections)} sections successful, "
            f"{failed_count} failures"
        )

        # Add performance metadata to each section
        for section in valid_sections:
            section["zap_mode_stats"] = {
                "total_processing_time": processing_time,
                "concurrent_workers": 16,
                "success_rate": len(valid_sections) / len(sections),
                "sections_processed": len(sections),
            }

        return valid_sections

    except Exception as e:
        logger.error(f"Zap Mode analysis failed for {standard}: {str(e)}")
        # Fallback to standard processing
        logger.info("Falling back to standard processing mode")
        section_tasks = []
        for section in processed_checklist.get("sections", []):
            ai_svc.current_document_id = document_id
            section_tasks.append(
                ai_svc._process_section(section, text, document_id, standard)
            )
        return await asyncio.gather(*section_tasks)


async def _initialize_analysis_tracking(
    document_id: str,
    framework: str,
    standards: list,
    special_instructions: str,
    extensive_search: bool,
) -> tuple[dict, Path, Path, Path]:
    """Initialize analysis tracking with results structure and lock files."""
    logger.info(
        f"Starting compliance analysis for {document_id} with framework "
        f"{framework} and standards {', '.join(standards)}"
    )

    results_path = ANALYSIS_RESULTS_DIR / f"{document_id}.json"
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            results: dict = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        results: dict = {
            "document_id": document_id,
            "status": "PROCESSING",
            "message": "Initializing compliance analysis",
        }

    results["status"] = "PROCESSING"
    results["compliance_analysis"] = "PROCESSING"
    results["framework"] = framework
    results["standards"] = standards
    results["specialInstructions"] = special_instructions
    results["extensiveSearch"] = extensive_search

    instructions_msg = (
        f" with special instructions: {special_instructions}"
        if special_instructions
        else ""
    )
    extensive_msg = " (extensive search enabled)" if extensive_search else ""
    results["message"] = (
        f"Compliance analysis in progress using {framework} for "
        f"standards: {', '.join(standards)}{instructions_msg}"
        f"{extensive_msg}"
    )
    results["sections"] = []

    if "error" in results:
        del results["error"]
    if "error_timestamp" in results:
        del results["error_timestamp"]

    # Setup lock files
    processing_lock_file = ANALYSIS_RESULTS_DIR / f"{document_id}.compliance_processing"
    processing_lock_file.touch()

    error_lock_file = ANALYSIS_RESULTS_DIR / f"{document_id}.error"
    if error_lock_file.exists():
        error_lock_file.unlink()

    save_analysis_results(document_id, results)

    return results, results_path, processing_lock_file, error_lock_file


async def _process_standards_sequentially(
    standards: list,
    document_id: str,
    text: str,
    framework: str,
    ai_svc: AIService,
    processing_mode: str,
    results: dict,
    progress_tracker=None,
) -> tuple[list, list]:
    """
    Process each standard sequentially and return all sections and failed standards.
    CRITICAL: This function MUST only process the user-selected standards.
    """
    # STRICT VALIDATION: Ensure we only process user-selected standards
    logger.info(f"🔒 STRICT STANDARDS VALIDATION for document {document_id}")
    logger.info(f"🔒 Processing EXACTLY these user-selected standards: {standards}")
    logger.info(
        "🔒 Will NOT process any other standards regardless of document content"
    )

    if not standards or len(standards) == 0:
        raise ValueError("No standards provided for sequential processing")

    all_sections = []
    failed_standards = []
    total_standards = len(standards)
    completed_standards = 0

    for i, standard in enumerate(standards):
        try:
            logger.info(
                f"🎯 PROCESSING STANDARD {
                    i + 1}/{total_standards}: {standard} (USER-SELECTED ONLY)"
            )
            logger.info(
                f"🎯 Document {document_id} - Current: {standard}, "
                f"Remaining: {standards[i + 1:] if i + 1 < len(standards) else 'None'}"
            )

            # Get checklist to determine total questions
            checklist_data = load_checklist(framework, standard)
            total_questions = len(checklist_data)

            # Start tracking this standard
            if progress_tracker:
                progress_tracker.start_standard(document_id, standard, total_questions)

                # Initialize question-level tracking
                all_questions_data = []
                for section in checklist_data.get("sections", []):
                    for item in section.get("items", []):
                        all_questions_data.append(
                            {
                                "id": item.get("id"),
                                "section": item.get("section", standard),
                                "question": item.get("question", ""),
                            }
                        )
                progress_tracker.initialize_questions(
                    document_id, standard, all_questions_data
                )

            # Update progress status
            progress_percent = int((i / total_standards) * 100)
            results["progress"] = {
                "current_standard": standard,
                "completed_standards": completed_standards,
                "total_standards": total_standards,
                "progress_percent": progress_percent,
                "status": f"Processing {standard}...",
            }
            results["message"] = (
                f"Processing standard {i + 1}/{total_standards}: {standard}"
            )
            save_analysis_results(document_id, results)

            checklist = load_checklist(framework, standard)

            # Set progress tracker for question-level tracking
            ai_svc.progress_tracker = progress_tracker

            # Choose processing approach based on mode
            if processing_mode == "smart":
                # Smart Mode: Use intelligent semantic processing
                standard_sections = await process_smart_mode_analysis(
                    checklist, text, document_id, ai_svc, standard, progress_tracker
                )
            elif processing_mode == "zap":
                # Zap Mode: High-speed processing with 16 concurrent workers
                standard_sections = await process_zap_mode_analysis(
                    checklist, text, document_id, ai_svc, standard, progress_tracker
                )
            else:
                # Standard Mode: Balanced parallel processing
                section_tasks = []
                for section in checklist.get("sections", []):
                    ai_svc.current_document_id = (
                        document_id  # Ensure it's set before each section
                    )
                    section_tasks.append(
                        ai_svc._process_section(
                            section, text, document_id, standard_id=standard
                        )
                    )
                standard_sections = await asyncio.gather(*section_tasks)

            # Add metadata to identify which standard these sections belong to
            for section in standard_sections:
                section["standard"] = standard
            all_sections.extend(standard_sections)

            # Mark standard as completed in progress tracker
            if progress_tracker:
                progress_tracker.complete_standard(document_id, standard)

            # Update progress after completing standard
            completed_standards += 1
            progress_percent = int((completed_standards / total_standards) * 100)
            results["progress"] = {
                "current_standard": None,
                "completed_standards": completed_standards,
                "total_standards": total_standards,
                "progress_percent": progress_percent,
                "status": f"Completed {standard}",
            }
            results["message"] = (
                f"Completed standard {completed_standards}/"
                f"{total_standards}: {standard}"
            )
            save_analysis_results(document_id, results)
            logger.info(
                f"Added {len(standard_sections)} sections for standard {standard}"
            )
        except Exception as e:
            logger.error(
                f"Error processing standard {standard}: {str(e)}", exc_info=True
            )
            failed_standards.append({"standard": standard, "error": str(e)})

        save_analysis_results(
            document_id, {**results, "sections": all_sections}
        )  # Save after each standard

    return all_sections, failed_standards


def _handle_analysis_completion(
    document_id: str,
    results: dict,
    all_sections: list,
    failed_standards: list,
    standards: list,
    performance_tracker,
    processing_mode: str,
) -> dict:
    """Handle the completion of analysis and build final results."""
    # Update the results with the compiled data
    results_path = ANALYSIS_RESULTS_DIR / f"{document_id}.json"
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    if len(failed_standards) == len(standards):
        results["status"] = "FAILED"
        results["compliance_analysis"] = "FAILED"
        results["error"] = "All standards failed to process"
        results["failed_standards"] = failed_standards
        results["error_timestamp"] = datetime.now().isoformat()
        results["message"] = "Compliance analysis failed for all standards"
        error_lock_file = ANALYSIS_RESULTS_DIR / f"{document_id}.error"
        error_lock_file.touch()
    elif len(failed_standards) > 0:
        results["status"] = "COMPLETED_WITH_ERRORS"
        results["compliance_analysis"] = "COMPLETED_WITH_ERRORS"
        results["sections"] = all_sections
        results["failed_standards"] = failed_standards
        results["completed_at"] = datetime.now().isoformat()
        results["message"] = (
            f"Compliance analysis completed with errors for "
            f"{len(failed_standards)} of {len(standards)} standards"
        )
        completion_lock_file = ANALYSIS_RESULTS_DIR / f"{document_id}.completed"
        completion_lock_file.touch()
    else:
        results["status"] = "COMPLETED"
        results["compliance_analysis"] = "COMPLETED"
        results["sections"] = all_sections
        results["completed_at"] = datetime.now().isoformat()
        results["message"] = (
            f"Compliance analysis completed for all {len(standards)} standards"
        )
        results["progress"] = {
            "current_standard": None,
            "completed_standards": len(standards),
            "total_standards": len(standards),
            "progress_percent": 100,
            "status": "Analysis completed successfully",
        }
        completion_lock_file = ANALYSIS_RESULTS_DIR / f"{document_id}.completed"
        completion_lock_file.touch()

    # End performance tracking and add metrics
    performance_tracker.end_tracking()
    performance_tracker.questions_processed = len(all_sections)
    performance_metrics = performance_tracker.get_metrics()

    # Add performance data to results
    results["performance_metrics"] = performance_metrics
    results["processing_mode"] = processing_mode

    return results


def _handle_analysis_error(
    document_id: str, error: Exception, performance_tracker, processing_mode: str
) -> None:
    """Handle analysis errors and update error state."""
    logger.error(f"Error processing compliance analysis: {str(error)}", exc_info=True)

    # End performance tracking even on error
    try:
        if performance_tracker is not None:
            performance_tracker.end_tracking()
            error_performance_metrics = performance_tracker.get_metrics()
        else:
            error_performance_metrics = {"error": "Performance tracker not initialized"}
    except Exception:
        error_performance_metrics = {"error": "Performance tracking failed"}

    try:
        results_path = ANALYSIS_RESULTS_DIR / f"{document_id}.json"
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                results = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            results = {
                "document_id": document_id,
                "status": "FAILED",
                "message": "Failed to process compliance analysis",
            }
        results["status"] = "FAILED"
        results["compliance_analysis"] = "FAILED"
        results["error"] = str(error)
        results["error_timestamp"] = datetime.now().isoformat()
        results["message"] = f"Compliance analysis failed: {str(error)}"
        results["performance_metrics"] = error_performance_metrics  # type: ignore
        results["processing_mode"] = processing_mode
        save_analysis_results(document_id, results)
        error_lock_file = ANALYSIS_RESULTS_DIR / f"{document_id}.error"
        error_lock_file.touch()
        processing_lock_file = (
            ANALYSIS_RESULTS_DIR / f"{document_id}.compliance_processing"
        )
        if processing_lock_file.exists():
            processing_lock_file.unlink()
    except Exception as inner_e:
        logger.error(f"Error updating error status: {str(inner_e)}")


async def process_compliance_analysis(
    document_id: str,
    text: str,
    framework: str,
    standards: list,  # Accept a list of standards
    special_instructions: str,  # User's special instructions
    extensive_search: bool,  # Extensive search flag
    ai_svc: AIService,
    processing_mode: str = "smart",  # New parameter for processing mode
) -> None:
    """
    Process compliance analysis in the background for multiple standards
    sequentially, checklist items in parallel.
    """
    performance_tracker = None  # Initialize to avoid NameError in exception handler

    try:
        # Clear any previously processed questions for this document
        from services.ai import clear_document_questions

        clear_document_questions(document_id)

        # CRITICAL: Log exactly which standards the user selected for compliance
        # analysis
        logger.info(f"🎯 COMPLIANCE ANALYSIS STARTING for document {document_id}")
        logger.info(f"🎯 USER-SELECTED STANDARDS ONLY: {standards}")
        logger.info(f"🎯 Framework: {framework}, Processing Mode: {processing_mode}")
        logger.info(f"🎯 Total standards to analyze: {len(standards)}")

        # Validate standards list is not empty and contains only user selections
        if not standards or len(standards) == 0:
            raise ValueError("No standards provided for compliance analysis")

        # Initialize progress tracking
        from services.progress_tracker import get_progress_tracker

        progress_tracker = get_progress_tracker()
        progress_tracker.start_analysis(
            document_id, framework, standards, processing_mode
        )

        # Initialize analysis tracking
        tracking_result = await _initialize_analysis_tracking(
            document_id, framework, standards, special_instructions, extensive_search
        )
        results, results_path, processing_lock_file, error_lock_file = tracking_result

        # Initialize performance tracker
        performance_tracker = PerformanceTracker(processing_mode)
        performance_tracker.start_tracking()

        # Process all standards sequentially
        all_sections, failed_standards = await _process_standards_sequentially(
            standards,
            document_id,
            text,
            framework,
            ai_svc,
            processing_mode,
            results,
            progress_tracker,
        )

        # Handle completion and build final results
        final_results = _handle_analysis_completion(
            document_id,
            results,
            all_sections,
            failed_standards,
            standards,
            performance_tracker,
            processing_mode,
        )

        # Mark progress as completed
        progress_tracker.cleanup_analysis(document_id)

        # Save final results and cleanup
        save_analysis_results(document_id, final_results)
        if processing_lock_file.exists():
            processing_lock_file.unlink()
        logger.info(f"Completed compliance analysis process for document {document_id}")

    except Exception as e:
        # Mark progress as failed
        from services.progress_tracker import get_progress_tracker

        progress_tracker = get_progress_tracker()
        progress_tracker.fail_analysis(document_id, str(e))

        _handle_analysis_error(document_id, e, performance_tracker, processing_mode)
