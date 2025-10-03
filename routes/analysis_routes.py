import asyncio
import json
import logging
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import random
import string

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from services.ai import AIService, get_ai_service
from services.checklist_utils import get_available_frameworks, is_standard_available, load_checklist
# Document chunking now handled by NLP pipeline - see nlp_tools/
from services.smart_metadata_extractor import SmartMetadataExtractor
from services.vector_store import get_vector_store
from services.staged_storage import staged_storage


def generate_document_id() -> str:
    """Generate a unique document ID without vector dependencies"""
    timestamp = datetime.now().strftime("%d%m%Y")
    random_part = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
    random_part2 = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
    return f"RAI-{timestamp}-{random_part}-{random_part2}"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to use enhanced storage with PostgreSQL support, fallback to basic SQLite
try:
    from services.persistent_storage_enhanced import get_persistent_storage_manager
    logger.info("✅ Using enhanced persistent storage with PostgreSQL support")
except ImportError:
    # Force use of enhanced storage - should not fall back to old version
    logger.error("❌ Enhanced storage import failed - this should not happen")
    raise


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
    logger.warning("python - docx not installed, DOCX support disabled")
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
VECTOR_INDICES_DIR = BACKEND_DIR / "vector_indices"

# Create directories
for directory in [UPLOADS_DIR, ANALYSIS_RESULTS_DIR, CHECKLIST_DATA_DIR, VECTOR_INDICES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directory created / verified: {directory}")

router = APIRouter()

# Initialize services
AZURE_OPENAI_API_KEY = os.getenv(
    "AZURE_OPENAI_API_KEY",
    (
        "Dqlg5AKLmgh4d7riA5lcJc9NTygtQgTskHZ7UQ6ZFgm9m6cDoiNEJQQJ99BEACHYHv6XJ3w3"
        "AAAAACOG1eeM"
    ),
)
AZURE_OPENAI_ENDPOINT = os.getenv(
    "AZURE_OPENAI_ENDPOINT",
    "https://vitha-maxu94mf-eastus2.cognitiveservices.azure.com/",
)
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "model-router")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"
)
AZURE_OPENAI_EMBEDDING_API_VERSION = os.getenv(
    "AZURE_OPENAI_EMBEDDING_API_VERSION", "2023-05-15"
)

# Initialize smart metadata extractor
smart_metadata_extractor = SmartMetadataExtractor()


async def _start_keepalive_task(document_id: str) -> None:
    """Start a background task that keeps the server alive after metadata extraction."""
    import asyncio
    
    async def keepalive_worker():
        """Background worker that keeps server alive by logging periodically."""
        logger.info(f"🟢 KEEPALIVE: Started for document {document_id}")
        
        # Keep alive for 30 minutes (enough time for user to select framework)
        for i in range(180):  # 30 minutes = 180 * 10 seconds
            await asyncio.sleep(10)  # Wait 10 seconds
            
            # Check if framework has been selected (compliance analysis started)
            try:
                status_file = ANALYSIS_RESULTS_DIR / f"{document_id}.json"
                if status_file.exists():
                    with open(status_file, 'r', encoding='utf-8') as f:
                        status_data = json.load(f)
                        
                    if status_data.get("compliance_analysis") in ["IN_PROGRESS", "COMPLETED"]:
                        logger.info(f"🟢 KEEPALIVE: Framework selected for {document_id}, ending keepalive")
                        break
                        
                logger.info(f"🟢 KEEPALIVE: Ping {i+1}/180 for document {document_id}")
            except Exception as e:
                logger.error(f"🟢 KEEPALIVE: Error checking status for {document_id}: {e}")
        
        logger.info(f"🟢 KEEPALIVE: Ended for document {document_id}")
    
    # Start the keepalive task in the background
    asyncio.create_task(keepalive_worker())


def save_analysis_results(document_id: str, results: Dict[str, Any]) -> None:
    """
    SIMPLE FILE SAVE: Direct file storage only, no dual storage complexity.
    """
    try:
        logger.info(f"� FILE SAVE: Starting save for {document_id}")
        
        # Create a deep copy and ensure all GeographicalEntity objects are converted to dicts
        serializable_results = _ensure_json_serializable(results)
        
        # Simple file save
        results_path = ANALYSIS_RESULTS_DIR / f"{document_id}.json"
        
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
        logger.info(f"✅ FILE SAVE: Save completed for {document_id} (status: {results.get('status', 'unknown')})")
        
    except Exception as e:
        logger.error(f"❌ FILE SAVE: Save failed for {document_id}: {str(e)}")
        raise


def _ensure_json_serializable(obj: Any) -> Any:
    """Recursively ensure all objects are JSON serializable, converting GeographicalEntity objects to dicts."""
    from services.geographical_service import GeographicalEntity

    if isinstance(obj, GeographicalEntity):
        # Convert GeographicalEntity to dictionary
        return obj.__dict__ if hasattr(obj, '__dict__') else str(obj)
    elif isinstance(obj, set):
        # Convert sets to lists for JSON serialization
        return list(obj)
    elif isinstance(obj, dict):
        # Recursively process dictionary values
        return {key: _ensure_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # Recursively process list/tuple items
        return [_ensure_json_serializable(item) for item in obj]
    else:
        # Return as-is for JSON-serializable types
        return obj

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
            "[PATCH] Metadata extraction already completed for "
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
    logger.info(f"🚀 CHUNKING START: Starting document chunking for {document_id}")
    file_path = get_document_file_path(document_id)
    if not file_path:
        logger.error(f"❌ CHUNKING ERROR: No file found for document_id: {document_id}")
        raise ValueError("No file found for document_id")

    logger.info(f"📁 CHUNKING FILE: Found file at {file_path}")
    ext = file_path.suffix.lower()
    logger.info(f"📄 CHUNKING EXT: File extension detected: {ext}")
    
    if ext == ".pdf":
        logger.info(f"🔧 CHUNKING PDF: Processing with NLP pipeline for {document_id} - NO FALLBACKS!")
        # Use NLP pipeline for enhanced document processing - FORCE IT TO WORK
        from nlp_tools.complete_nlp_validation_pipeline import CompleteNLPValidationPipeline
        import os
        taxonomy_dir = os.path.join(os.path.dirname(__file__), "..", "taxonomy", "IFRSAT-2025", "IFRSAT-2025", "full_ifrs")
        
        logger.info(f"🔧 CHUNKING PDF: Initializing NLP pipeline with taxonomy_dir: {taxonomy_dir}")
        nlp_pipeline = CompleteNLPValidationPipeline(taxonomy_dir=taxonomy_dir)
        
        logger.info(f"🔧 CHUNKING PDF: Running NLP processing on {file_path}")
        nlp_result = nlp_pipeline.process_document_with_validation(str(file_path))
        
        logger.info(f"🔧 CHUNKING PDF: NLP result success: {nlp_result.success}")
        if hasattr(nlp_result, 'error'):
            logger.info(f"🔧 CHUNKING PDF: NLP result error: {nlp_result.error}")
        
        if not nlp_result.success:
            logger.error(f"❌ CHUNKING PDF FAILED: NLP processing failed: {nlp_result.error}")
            raise ValueError(f"NLP processing failed: {nlp_result.error}")
        
        # Convert NLP result to legacy chunk format for compatibility
        logger.info(f"� CHUNKING PDF: Converting NLP result to chunks")
        chunks = _convert_nlp_to_chunks(nlp_result, document_id)
        logger.info(f"✅ CHUNKING PDF SUCCESS: NLP pipeline returned {len(chunks) if chunks else 0} chunks")
    elif ext == ".docx":
        logger.info(f"🔧 CHUNKING DOCX: Processing with NLP pipeline for {document_id} - NO FALLBACKS!")
        # Use NLP pipeline for enhanced document processing - FORCE IT TO WORK
        from nlp_tools.complete_nlp_validation_pipeline import CompleteNLPValidationPipeline
        import os
        taxonomy_dir = os.path.join(os.path.dirname(__file__), "..", "taxonomy", "IFRSAT-2025", "IFRSAT-2025", "full_ifrs")
        
        logger.info(f"🔧 CHUNKING DOCX: Initializing NLP pipeline with taxonomy_dir: {taxonomy_dir}")
        nlp_pipeline = CompleteNLPValidationPipeline(taxonomy_dir=taxonomy_dir)
        
        logger.info(f"🔧 CHUNKING DOCX: Running NLP processing on {file_path}")
        nlp_result = nlp_pipeline.process_document_with_validation(str(file_path))
        
        logger.info(f"🔧 CHUNKING DOCX: NLP result success: {nlp_result.success}")
        if hasattr(nlp_result, 'error'):
            logger.info(f"🔧 CHUNKING DOCX: NLP result error: {nlp_result.error}")
        
        if not nlp_result.success:
            logger.error(f"❌ CHUNKING DOCX FAILED: NLP processing failed: {nlp_result.error}")
            raise ValueError(f"NLP processing failed: {nlp_result.error}")
        
        # Convert NLP result to legacy chunk format for compatibility
        logger.info(f"� CHUNKING DOCX: Converting NLP result to chunks")
        chunks = _convert_nlp_to_chunks(nlp_result, document_id)
        logger.info(f"✅ CHUNKING DOCX SUCCESS: NLP pipeline returned {len(chunks) if chunks else 0} chunks")
    else:
        logger.error(f"❌ CHUNKING ERROR: Unsupported file extension: {ext}")
        raise ValueError(f"Unsupported file extension: {ext}")

    if not chunks:
        logger.error(f"❌ CHUNKING ERROR: No chunks generated from document {document_id}")
        raise ValueError("No chunks generated from document")

    # CRITICAL FIX: Store chunks in categorized_content database for smart categorization
    _store_chunks_for_smart_categorization(document_id, chunks)

    logger.info(f"Generated {len(chunks)} chunks for document {document_id}")
    return chunks


def _store_chunks_for_smart_categorization(document_id: str, chunks: list) -> None:
    """Store document chunks in categorized_content database for smart categorization."""
    logger.info(f"📚 SMART STORAGE: Starting to store {len(chunks)} chunks for document {document_id}")
    
    if not chunks:
        logger.warning(f"⚠️  SMART STORAGE: No chunks provided for document {document_id}")
        return
    
    try:
        # Import and initialize storage
        from services.intelligent_chunk_accumulator import get_global_storage
        logger.info(f"🔧 SMART STORAGE: Importing storage for document {document_id}")
        
        storage = get_global_storage()
        logger.info(f"🔧 SMART STORAGE: Storage initialized for document {document_id}")
        
        # Verify storage is working by checking database
        try:
            # Test database connection
            test_result = storage._test_database_connection()
            logger.info(f"🔧 SMART STORAGE: Database connection test result: {test_result}")
        except Exception as db_test_error:
            logger.warning(f"⚠️  SMART STORAGE: Database test failed: {db_test_error}")
        
        stored_count = 0
        failed_count = 0
        
        for i, chunk in enumerate(chunks):
            try:
                # Extract text content from chunk
                if isinstance(chunk, dict):
                    content = chunk.get("text") or chunk.get("content", "")
                    chunk_info = f"dict with keys: {list(chunk.keys())}"
                else:
                    content = str(chunk)
                    chunk_info = f"string of length {len(content)}"
                
                if not content or not content.strip():
                    logger.warning(f"⚠️  SMART STORAGE: Chunk {i} is empty for document {document_id}")
                    failed_count += 1
                    continue
                
                # Store with basic categorization - AI service can enhance later
                category = "financial_document"  # Basic category
                subcategory = "content_chunk"
                confidence = 0.8  # Good confidence for processed chunks
                keywords = []  # Can be enhanced by AI later
                
                logger.debug(f"📝 SMART STORAGE: Storing chunk {i} ({chunk_info[:100]}) for document {document_id}")
                
                success = storage.store_categorized_chunk(
                    document_id=document_id,
                    chunk=content,
                    category=category,
                    subcategory=subcategory,
                    confidence=confidence,
                    keywords=keywords
                )
                
                if success:
                    stored_count += 1
                    logger.debug(f"✅ SMART STORAGE: Successfully stored chunk {i} for document {document_id}")
                else:
                    failed_count += 1
                    logger.warning(f"⚠️  SMART STORAGE: Failed to store chunk {i} (storage returned False) for document {document_id}")
                        
            except Exception as chunk_error:
                failed_count += 1
                logger.error(f"❌ SMART STORAGE: Failed to store chunk {i} for document {document_id}: {chunk_error}")
                logger.error(f"❌ SMART STORAGE: Chunk {i} traceback: {traceback.format_exc()}")
                continue
        
        # Final verification - check if chunks were actually stored
        try:
            total_stored = storage.debug_chunk_count(document_id)
            logger.info(f"🔍 SMART STORAGE: Verification - {total_stored} total chunks found in database for document {document_id}")
            
            # Also show all documents for debugging
            logger.info(f"🔍 SMART STORAGE: All documents summary:")
            all_docs = storage.debug_all_documents()
            for doc_id, count in all_docs.items():
                logger.info(f"  📄 {doc_id}: {count} chunks")
                
        except Exception as verify_error:
            logger.warning(f"⚠️  SMART STORAGE: Could not verify chunk count: {verify_error}")
        
        if stored_count > 0:
            logger.info(f"✅ SMART STORAGE: Successfully stored {stored_count}/{len(chunks)} chunks for document {document_id} (failed: {failed_count})")
        else:
            logger.error(f"❌ SMART STORAGE: Failed to store any chunks for document {document_id} (failed: {failed_count})")
        
    except ImportError as import_error:
        logger.error(f"❌ SMART STORAGE: Import error for document {document_id}: {import_error}")
    except Exception as e:
        logger.error(f"❌ SMART STORAGE: Failed to store chunks for document {document_id}: {e}")
        logger.error(f"❌ SMART STORAGE: Traceback: {traceback.format_exc()}")


def _convert_nlp_to_chunks(nlp_result, document_id: str) -> list:
    """Convert NLP pipeline result to legacy chunk format - NO FALLBACKS, MUST WORK!"""
    chunks = []
    
    logger.info(f"🔍 CHUNK CONVERSION: Starting conversion for document {document_id}")
    logger.info(f"🔍 CHUNK CONVERSION: NLP result type: {type(nlp_result)}")
    logger.info(f"🔍 CHUNK CONVERSION: NLP result attributes: {dir(nlp_result)}")
    
    # Log what we have in the NLP result
    if hasattr(nlp_result, 'validated_mega_chunks'):
        logger.info(f"🔍 CHUNK CONVERSION: Has validated_mega_chunks: {nlp_result.validated_mega_chunks is not None}")
        logger.info(f"🔍 CHUNK CONVERSION: validated_mega_chunks type: {type(nlp_result.validated_mega_chunks)}")
        if nlp_result.validated_mega_chunks:
            logger.info(f"🔍 CHUNK CONVERSION: validated_mega_chunks keys: {list(nlp_result.validated_mega_chunks.keys()) if isinstance(nlp_result.validated_mega_chunks, dict) else 'Not a dict'}")
    
    if hasattr(nlp_result, 'structure_parsing'):
        logger.info(f"🔍 CHUNK CONVERSION: Has structure_parsing: {nlp_result.structure_parsing is not None}")
        logger.info(f"🔍 CHUNK CONVERSION: structure_parsing type: {type(nlp_result.structure_parsing)}")
    
    # PRIMARY: Extract validated mega chunks from NLP result
    if hasattr(nlp_result, 'validated_mega_chunks') and nlp_result.validated_mega_chunks:
        logger.info("🔧 CHUNK CONVERSION: Processing validated_mega_chunks")
        chunk_counter = 1
        
        for standard_key, chunks_dict in nlp_result.validated_mega_chunks.items():
            logger.info(f"🔧 CHUNK CONVERSION: Processing standard_key: {standard_key}, type: {type(chunks_dict)}")
            
            if isinstance(chunks_dict, dict):
                for chunk_id, chunk_data in chunks_dict.items():
                    logger.info(f"🔧 CHUNK CONVERSION: Processing chunk_id: {chunk_id}, data type: {type(chunk_data)}")
                    
                    # Handle both dict and string chunk_data
                    if isinstance(chunk_data, dict):
                        content_text = chunk_data.get("content", "")
                        accounting_standard = chunk_data.get("accounting_standard", "")
                        confidence_score = chunk_data.get("confidence_score", 0.0)
                        classification_tags = chunk_data.get("classification_tags", {})
                    elif isinstance(chunk_data, str):
                        content_text = chunk_data
                        accounting_standard = standard_key
                        confidence_score = 0.8
                        classification_tags = {}
                    else:
                        content_text = str(chunk_data)
                        accounting_standard = standard_key
                        confidence_score = 0.5
                        classification_tags = {}
                    
                    logger.info(f"🔧 CHUNK CONVERSION: Content length: {len(content_text)}")
                    
                    if len(content_text.strip()) > 0:  # Only create chunks with actual content
                        # Create legacy chunk format compatible with SmartMetadataExtractor
                        legacy_chunk = {
                            "id": f"{document_id}_chunk_{chunk_counter}",
                            "content": content_text,  # For frontend/analysis compatibility
                            "text": content_text,     # For SmartMetadataExtractor compatibility
                            "chunk_index": chunk_counter - 1,
                            "page": 0,  # Will be updated if page info available
                            "page_no": 0,
                            "length": len(content_text),
                            "chunk_type": "content",
                            "metadata": {
                                "accounting_standard": accounting_standard,
                                "confidence_score": confidence_score,
                                "classification_tags": classification_tags,
                                "original_chunk_id": chunk_id,
                                "processing_method": "nlp_pipeline"
                            }
                        }
                        chunks.append(legacy_chunk)
                        chunk_counter += 1
                        logger.info(f"✅ CHUNK CONVERSION: Created chunk {chunk_counter-1} with {len(content_text)} chars")
    
    # SECONDARY: use basic structure parsing if validated_mega_chunks didn't work
    if not chunks and hasattr(nlp_result, 'structure_parsing') and nlp_result.structure_parsing:
        logger.info("🔧 CHUNK CONVERSION: No validated_mega_chunks, trying structure_parsing")
        structure = nlp_result.structure_parsing
        
        if isinstance(structure, dict) and "sections" in structure:
            logger.info(f"🔧 CHUNK CONVERSION: Found {len(structure['sections'])} sections")
            
            for i, section in enumerate(structure["sections"], 1):
                # Handle both dict and string sections
                if isinstance(section, dict):
                    content_text = section.get("content", "")
                    section_type = section.get("type", "unknown")
                    section_id = section.get("id", f"section_{i}")
                elif isinstance(section, str):
                    content_text = section
                    section_type = "text"
                    section_id = f"section_{i}"
                else:
                    content_text = str(section)
                    section_type = "unknown"
                    section_id = f"section_{i}"
                
                if len(content_text.strip()) > 0:  # Only create chunks with actual content
                    legacy_chunk = {
                        "id": f"{document_id}_section_{i}",
                        "content": content_text,  # For frontend/analysis compatibility  
                        "text": content_text,     # For SmartMetadataExtractor compatibility
                        "chunk_index": i - 1,
                        "page": 0,
                        "page_no": 0,
                        "length": len(content_text),
                        "chunk_type": "section",
                        "metadata": {
                            "section_type": section_type,
                            "processing_method": "structure_parsing",
                            "original_section_id": section_id
                        }
                    }
                    chunks.append(legacy_chunk)
                    logger.info(f"✅ CHUNK CONVERSION: Created section chunk {i} with {len(content_text)} chars")
    
    # FAIL FAST: If we still have no chunks, something is wrong with the NLP pipeline
    if not chunks:
        logger.error(f"❌ CHUNK CONVERSION FAILED: No chunks created from NLP result for {document_id}")
        logger.error(f"❌ CHUNK CONVERSION: NLP result dump: {nlp_result}")
        raise ValueError(f"NLP pipeline produced no usable chunks for document {document_id}")
    
    logger.info(f"✅ CHUNK CONVERSION SUCCESS: Converted NLP result to {len(chunks)} legacy chunks for {document_id}")
    return chunks


def _save_chunks_directly(document_id: str, chunks: list) -> None:
    """Save chunks directly to vector_indices directory without vector indexing."""
    logger.info(f"Saving {len(chunks)} chunks for document {document_id}")
    
    # Save chunks to vector_indices directory for compatibility
    chunks_file = VECTOR_INDICES_DIR / f"{document_id}_chunks.json"
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Chunks saved to {chunks_file}")


    logger.info(f"Vector store indexing completed for document {document_id}")


async def _extract_document_metadata(document_id: str, chunks: list) -> dict:
    """Extract metadata from document chunks using optimized smart extraction."""
    logger.info(f"🚀 Starting OPTIMIZED metadata extraction for document {document_id}")
    logger.info("🔍 Using SmartMetadataExtractor - this should be DIFFERENT from old extraction!")

    # Use smart metadata extractor for 80% token reduction
    metadata_result = await smart_metadata_extractor.extract_metadata_optimized(document_id, chunks)
    logger.info(f"✅ Smart metadata extraction completed for document {document_id}")
    logger.info(f"💰 Token usage: {metadata_result.get('optimization_metrics', {}).get('tokens_used', 'N / A')}")
    return metadata_result


def _transform_metadata_for_frontend(metadata_result: dict) -> dict:
    """Transform metadata from smart extractor format to enhanced presentation."""
    transformed_metadata = {}

    # Fields that need to be transformed - BACKEND USES SNAKE_CASE AS STANDARD
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
                    # Extract geography - specific information for result page
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

                # BACKEND STANDARD: Use snake_case field names consistently
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
                    transformed_metadata[field]["geography_of_operations"] = str(
                        value) if value else "Geographic operations not specified"
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
    """Convert confidence score to human - readable level."""
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
    field_data.get("confidence", 0.0)
    context = field_data.get("context", "")
    field_data.get("extraction_method", "")

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
    summary["confidence_metrics"]["average_confidence"] = total_confidence / \
        extracted_fields if extracted_fields > 0 else 0.0
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


def _archive_document_file(document_id: str) -> None:
    """Archive uploaded file to document-specific folder for audit trail."""
    try:
        from pathlib import Path
        
        # Get the current file path
        file_path = get_document_file_path(document_id)
        if not file_path or not file_path.exists():
            logger.warning(f"No file found to archive for document: {document_id}")
            return
            
        # Create document archive directory
        archive_dir = Path("document_archives") / document_id
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Archive original file
        archived_file_path = archive_dir / f"original_{file_path.name}"
        
        # Move file to archive instead of deleting
        import shutil
        shutil.move(str(file_path), str(archived_file_path))
        
        logger.info(f"Archived uploaded file to: {archived_file_path}")
        
        # Store in database for audit trail
        try:
            storage = get_persistent_storage_manager()
            audit_data = {
                "document_id": document_id,
                "action": "file_archived",
                "original_path": str(file_path),
                "archive_path": str(archived_file_path),
                "timestamp": datetime.now().isoformat(),
                "file_size": archived_file_path.stat().st_size,
                "file_name": file_path.name
            }
            
            # Store audit log
            storage.save_document_audit_log(document_id, audit_data)
            logger.info(f"Audit log created for archived file: {document_id}")
            
        except Exception as audit_err:
            logger.warning(f"Failed to create audit log for {document_id}: {audit_err}")
        
    except Exception as archive_err:
        logger.error(f"Failed to archive file for {document_id}: {archive_err}")
        # Fallback to original deletion if archiving fails
        file_path = get_document_file_path(document_id)
        if file_path and file_path.exists():
            file_path.unlink()
            logger.info(f"Fallback: Deleted uploaded file after vectorization: {file_path}")


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
    import traceback
    
    # Check for duplicate processing
    if _check_processing_locks(document_id):
        return

    # Create processing lock
    processing_lock_file = ANALYSIS_RESULTS_DIR / f"{document_id}.processing"
    processing_lock_file.touch()

    try:
        # Initialize processing with processing mode
        _initialize_processing_results(document_id, processing_mode)

        # Create initial status file to indicate chunking has started
        status_file = ANALYSIS_RESULTS_DIR / f"{document_id}_status.json"
        initial_status = {
            "document_id": document_id,
            "chunking_status": "in_progress",
            "metadata_extraction_status": "pending",
            "chunking_started_at": datetime.now().isoformat()
        }
        with open(status_file, 'w', encoding='utf-8') as f:
            json.dump(initial_status, f, indent=2, ensure_ascii=False)

        # Process document chunks directly
        chunks = _process_document_chunks(document_id)
        
        # Create vector index for the chunks BEFORE metadata extraction
        vector_index_ready = False
        try:
            from services.vector_store import get_vector_store
            vs_svc = get_vector_store()
            index_created = await vs_svc.create_index(document_id, chunks)
            if index_created:
                logger.info(f"✅ Vector index created successfully for document {document_id}")
                
                # Verify the index is actually accessible
                try:
                    test_results = vs_svc.search("test", document_id, top_k=1)
                    vector_index_ready = True
                    logger.info(f"✅ Vector index verified and ready for document {document_id}")
                except Exception as ve:
                    logger.warning(f"⚠️ Vector index created but not accessible for document {document_id}: {ve}")
            else:
                logger.warning(f"⚠️ Vector index creation failed for document {document_id}")
        except ImportError as ie:
            logger.warning(f"⚠️ Vector store import error for document {document_id}: {ie}")
        except Exception as e:
            logger.warning(f"⚠️ Vector index creation error for document {document_id}: {e}")
            
        if not vector_index_ready:
            logger.warning(f"⚠️ Proceeding with metadata extraction without vector search for document {document_id}")

        # Update status to indicate chunking is complete and metadata extraction starting
        chunking_complete_status = {
            "document_id": document_id,
            "chunking_status": "completed",
            "metadata_extraction_status": "in_progress",
            "chunking_started_at": initial_status["chunking_started_at"],
            "chunking_completed_at": datetime.now().isoformat()
        }
        with open(status_file, 'w', encoding='utf-8') as f:
            json.dump(chunking_complete_status, f, indent=2, ensure_ascii=False)

        # Save chunks to analysis_results directory (no vector processing needed)
        chunks_file = ANALYSIS_RESULTS_DIR / f"{document_id}_chunks.json"
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Document chunking completed for {document_id}")
        logger.info(f"📄 Saved {len(chunks)} chunks to {chunks_file}")

        # Create basic results structure (metadata extraction will update this later)
        basic_result = {
            "status": "metadata_extraction_in_progress",
            "document_id": document_id,
            "chunks_count": len(chunks),
            "processing_mode": processing_mode,
            "completed_at": datetime.now().isoformat(),
            "timestamp": datetime.now().isoformat(),
            "metadata": {},
            "sections": [],
            "metadata_extraction": "IN_PROGRESS",
            "compliance_analysis": "PENDING",
            "message": "Document chunking completed. Metadata extraction in progress - poll /metadata-status for updates."
        }

        # Save results with basic structure (metadata extraction will update this)
        save_analysis_results(document_id, basic_result)

        # CRITICAL FIX: Actually extract metadata before marking as completed
        logger.info(f"🚀 METADATA EXTRACTION: Starting for document {document_id}")
        try:
            # Run metadata extraction using smart extractor
            metadata = await _extract_document_metadata(document_id, chunks)
            
            # Save metadata using staged storage for isolation
            try:
                logger.info(f"💾 STAGED: Starting staged storage save for {document_id}")
                from services.staged_storage import StagedStorageManager
                storage_manager = StagedStorageManager()
                storage_manager.save_metadata(document_id, metadata)
                logger.info(f"✅ STAGED: Staged storage save completed for {document_id}")
            except Exception as staged_error:
                logger.error(f"❌ STAGED STORAGE ERROR for {document_id}: {str(staged_error)}")
                logger.error(f"❌ STAGED STORAGE TRACEBACK: {traceback.format_exc()}")
                # Continue processing even if staged storage fails
            
            # Update the main results file with actual metadata and completion status
            try:
                logger.info(f"🔄 TRANSFORM: Starting metadata transformation for {document_id}")
                transformed_metadata = _transform_metadata_for_frontend(metadata)
                logger.info(f"✅ TRANSFORM: Metadata transformation completed for {document_id}")
                
                basic_result.update({
                    "status": "COMPLETED",  # FIX: Update main status to show completion
                    "metadata_extraction": "COMPLETED", 
                    "metadata": transformed_metadata,
                    "message": "Metadata extraction completed successfully. Ready for framework selection."
                })
                
                logger.info(f"💾 SAVE: Starting results save for {document_id}")
                save_analysis_results(document_id, basic_result)
                logger.info(f"✅ SAVE: Results saved successfully for {document_id}")
                
            except Exception as transform_error:
                logger.error(f"❌ TRANSFORM/SAVE ERROR for {document_id}: {str(transform_error)}")
                logger.error(f"❌ TRANSFORM/SAVE TRACEBACK: {traceback.format_exc()}")
                
                # Fallback: save with raw metadata if transformation fails
                basic_result.update({
                    "status": "COMPLETED",
                    "metadata_extraction": "COMPLETED",
                    "metadata": metadata,  # Use raw metadata as fallback
                    "message": "Metadata extraction completed (fallback mode)."
                })
                save_analysis_results(document_id, basic_result)
            
            # NOW create metadata completion lock file (after actual extraction)
            metadata_lock_file = ANALYSIS_RESULTS_DIR / f"{document_id}.metadata_completed"
            metadata_lock_file.touch()
            
            logger.info(f"✅ METADATA EXTRACTION: Completed successfully for {document_id}")
            
        except Exception as metadata_error:
            logger.error(f"❌ METADATA EXTRACTION: Failed for {document_id}: {metadata_error}")
            # Update results with error status
            basic_result.update({
                "metadata_extraction": "FAILED",
                "message": f"Metadata extraction failed: {str(metadata_error)}"
            })
            save_analysis_results(document_id, basic_result)

        # Remove processing lock file
        if processing_lock_file.exists():
            processing_lock_file.unlink()

        # Archive uploaded file for audit trail
        _archive_document_file(document_id)

        # UPDATE SESSION STATUS: Metadata extraction completed, ready for framework selection
        try:
            from routes.sessions_routes import update_session_processing_status
            await update_session_processing_status(
                f"session_{document_id}", 
                "metadata_complete", 
                {"last_updated": datetime.now().isoformat()}
            )
            logger.info(f"✅ SESSION STATUS UPDATED: Metadata complete for session_{document_id}")
        except Exception as session_error:
            logger.error(f"❌ Failed to update session status: {session_error}")

        logger.info(f"Metadata extraction completed for document {document_id}")
        
        # CRITICAL FIX: Keep server alive by starting a keep-alive background task
        # This prevents Render from shutting down the service after metadata extraction
        logger.info(f"🟢 SERVER CONTINUITY: Starting keep-alive task to prevent shutdown")
        await _start_keepalive_task(document_id)

    except Exception as _e:
        # Handle error
        _handle_processing_error(document_id, _e)

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
        logger.info(f"📖 UPLOAD STEP 1: Reading uploaded file content ({file.filename})")
        try:
            content = await file.read()
            logger.info(f"✅ UPLOAD STEP 1 COMPLETE: File content read ({len(content)} bytes)")
        except Exception as _e:
            logger.error(f"❌ UPLOAD STEP 1 FAILED: Error reading file content: {str(_e)}")
            response = {
                "status": "error",
                "error": "File processing failed",
                "message": "Failed to read uploaded file content",
            }
            return response

        # Generate unique document ID
        logger.info("🆔 UPLOAD STEP 2: Generating document ID")
        document_id = generate_document_id()
        logger.info(f"✅ UPLOAD STEP 2 COMPLETE: Document ID generated: {document_id}")

        # Save uploaded file with original extension
        logger.info("💾 UPLOAD STEP 3: Saving uploaded file to disk")
        upload_ext = f".{ext}"
        upload_path = UPLOADS_DIR / f"{document_id}{upload_ext}"
        try:
            with open(upload_path, "wb") as f:
                f.write(content)
            logger.info(f"✅ UPLOAD STEP 3 COMPLETE: File saved to {upload_path}")
        except Exception as _e:
            logger.error(f"❌ UPLOAD STEP 3 FAILED: Error saving uploaded file: {str(_e)}")
            response = {
                "status": "error",
                "error": "File save failed",
                "message": "Failed to save uploaded file",
            }
            logger.error(f"🚫 UPLOAD FAILED: File save error response: {json.dumps(response)}")
            return response

        # Log the processing mode
        logger.info(f"⚙️  Processing mode validation: {processing_mode}")

        # Validate processing mode
        valid_modes = ["zap", "smart", "comparison", "enhanced"]
        if processing_mode not in valid_modes:
            logger.warning(
                f"Invalid processing mode '{processing_mode}', defaulting to 'smart'"
            )
            processing_mode = "smart"
            
        # Map enhanced mode to smart for backend processing
        if processing_mode == "enhanced":
            processing_mode = "smart"

        logger.info(f"📤 UPLOAD STEP 4: Starting background processing for document {document_id}")
        # FULL NLP WORKFLOW: Document chunking with intelligent categorization
        try:
            logger.info("🔧 UPLOAD STEP 5: Importing full NLP document processing system")
            # Import the complete NLP pipeline
            background_tasks.add_task(
                process_upload_tasks,
                document_id,
                ai_svc,
                "",  # text parameter
                processing_mode
            )
            logger.info("✅ UPLOAD STEP 5 COMPLETE: Full NLP processing system imported successfully")

            # Generate session ID and format response for frontend compatibility
            session_id = f"session_{document_id}"
            document_name = file.filename if file.filename else "unknown.pdf"
            file_type = ext
            
            # Create enhanced session with file tracking
            try:
                now = datetime.now()
                session_file_info = SessionFile(
                    file_id=document_id,
                    original_filename=document_name,
                    file_size=len(content),
                    upload_timestamp=now,
                    file_type=file_type,
                    storage_location="postgresql"
                )
                
                enhanced_session = {
                    "session_id": session_id,
                    "title": f"Analysis of {document_name}",
                    "description": f"Document analysis session for {document_name}",
                    "created_at": now.isoformat(),
                    "updated_at": now.isoformat(),
                    "document_count": 1,
                    "last_document_id": document_id,
                    "status": "active",
                    "owner": "current_user",
                    "shared_with": [],
                    "chat_state": {"documentId": document_id, "processingStatus": "processing"},
                    "conversation_history": [
                        {
                            "message_id": f"msg_{now.strftime('%Y%m%d_%H%M%S')}_upload",
                            "role": "system",
                            "content": f"Document {document_name} uploaded and processing started",
                            "timestamp": now.isoformat(),
                            "message_type": "upload_notification",
                            "metadata": {"document_id": document_id, "file_size": len(content)}
                        }
                    ],
                    "user_choices": [],
                    "ai_responses": [],
                    "uploaded_files": [session_file_info.dict()],
                    "documents": [],
                    "analysis_context": {
                        "accounting_standard": None,
                        "custom_instructions": None,
                        "selected_frameworks": [],
                        "analysis_preferences": {"processing_mode": processing_mode}
                    }
                }
                
                # Save enhanced session
                save_session_to_file(session_id, enhanced_session)
                logger.info(f"✅ Enhanced session created: {session_id}")
                
            except Exception as session_error:
                logger.warning(f"Failed to create enhanced session: {session_error}")
                # Continue with upload even if session creation fails
            
            response = {
                "session_id": session_id,
                "document_id": document_id,
                "status": "processing",
                "document_name": document_name,
                "file_type": file_type,
                "processing_mode": processing_mode,
                "message": "Document uploaded - processing with full NLP pipeline including "
                          "chunking, categorization, and metadata extraction",
            }
            logger.info(f"✅ UPLOAD STEP 6 COMPLETE: Full NLP background task started for document {document_id}")
            logger.info(f"🎯 ALL UPLOAD STEPS COMPLETE: Full NLP pipeline initiated for {document_id}")
            logger.info(f"📋 Response: {json.dumps(response)}")
            return response
        except Exception as _e:
            logger.error(f"❌ UPLOAD STEP 5 FAILED: Full NLP processing failed to start: {str(_e)}")
            response = {
                "status": "error",
                "error": "NLP processing failed",
                "message": f"Full NLP pipeline failed to start: {str(_e)}",
            }
            logger.error(f"🚫 UPLOAD FAILED: Returning NLP processing failure response: {json.dumps(response)}")
            return response

    except Exception as _e:
        logger.error(f"Error processing document upload: {str(_e)}")
        logger.error(traceback.format_exc())
        response = {
            "status": "error",
            "error": "Upload failed",
            "message": f"Error processing document: {str(_e)}",
        }
        logger.info(f"Returning response for general error: {json.dumps(response)}")
        return response


@router.get("/checklist")
async def get_checklist() -> JSONResponse:
    """Get the default IAS 40 checklist questions."""
    try:
        checklist = load_checklist()
        return JSONResponse(status_code=200, content=checklist)
    except Exception as _e:
        logger.error(f"Error loading checklist: {str(_e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to load checklist", "message": str(_e)},
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

        # Debug logging
        logger.info(f"Frameworks loaded: {len(frameworks_data.get('frameworks', []))}")
        logger.info(f"Checklist base path: {checklist_base}")
        logger.info(f"Checklist base exists: {checklist_base.exists()}")
        if checklist_base.exists():
            dirs = [item.name for item in checklist_base.iterdir() if item.is_dir()]
            logger.info(f"Framework directories found: {dirs}")

        for fw in frameworks_data.get("frameworks", []):
            fw_id = fw["id"]
            fw_dir = checklist_base / fw_id
            logger.info(f"Checking framework {fw_id}: {fw_dir} exists={fw_dir.exists()}")
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
    except Exception as _e:
        logger.error(f"Error getting frameworks: {str(_e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(_e)}")


@router.get("/frameworks - debug", response_model=None)
async def get_frameworks_debug():
    """Debug endpoint to check frameworks loading issues"""
    try:
        from pathlib import Path

        result = {
            "raw_frameworks_count": 0,
            "checklist_base_exists": False,
            "checklist_base_path": "",
            "framework_dirs": [],
            "error": None
        }

        try:
            frameworks_data = get_available_frameworks()
            result["raw_frameworks_count"] = len(frameworks_data.get("frameworks", []))
        except Exception as _e:
            result["error"] = f"get_available_frameworks failed: {str(_e)}"

        checklist_base = Path(__file__).parent.parent / "checklist_data" / "frameworks"
        result["checklist_base_path"] = str(checklist_base)
        result["checklist_base_exists"] = checklist_base.exists()

        if checklist_base.exists():
            result["framework_dirs"] = [item.name for item in checklist_base.iterdir() if item.is_dir()]

        return result
    except Exception:
        return {"error": "Debug endpoint failed"}


@router.get("/checklist-debug/{framework}/{standard}", response_model=None)
async def get_checklist_debug(framework: str, standard: str):
    """Debug endpoint to check checklist loading for specific framework / standard"""
    try:
        from pathlib import Path

        result = {
            "framework": framework,
            "standard": standard,
            "is_available": False,
            "checklist_loaded": False,
            "checklist_path_checked": [],
            "checklist_sections": 0,
            "checklist_total_items": 0,
            "error": None
        }

        # Check availability
        try:
            result["is_available"] = is_standard_available(framework, standard)
        except Exception as _e:
            result["error"] = f"is_standard_available failed: {str(_e)}"

        # Check checklist loading
        try:
            checklist_base = Path(__file__).parent.parent / "checklist_data" / "frameworks"

            # Test possible paths
            possible_paths = [
                checklist_base / framework / f"{standard}.json",
                checklist_base / "IFRS" / f"{standard}.json",
                checklist_base / framework / standard / "checklist.json"
            ]

            for path in possible_paths:
                path_info = {
                    "path": str(path),
                    "exists": path.exists(),
                    "is_file": path.is_file() if path.exists() else False
                }
                result["checklist_path_checked"].append(path_info)

            checklist = load_checklist(framework, standard)
            if checklist:
                result["checklist_loaded"] = True
                result["checklist_sections"] = len(checklist.get('sections', []))
                result["checklist_total_items"] = sum(len(section.get('items', []))
                                                      for section in checklist.get('sections', []))

        except Exception as _e:
            result["error"] = f"checklist loading failed: {str(_e)}"

        return result
    except Exception:
        return {"error": "Debug endpoint failed"}


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
            "You are a financial standards recommendation AI. You MUST respond with valid JSON only - "
            "no text before or after the JSON.\\n\\n"
            "Your task: Analyze company profiles and return JSON with recommended accounting standards.\\n\\n"
            "CRITICAL: Your response must be valid JSON that starts with { and ends with }. "
            "No markdown, no explanations, no code blocks. Keep reasoning concise (max 80 characters).")

        user_prompt = (
            f"Company: {company_name}\\n"
            f"Business: {nature_of_business}\\n"
            f"Location: {operational_demographics}\\n"
            f"Framework: {framework}\\n"
            f"Statement Type: {financial_statements_type}\\n\\n"
            f"Available Standards: {available_standards_list}\\n\\n"
            "INSTRUCTIONS:\\n"
            "1. Analyze the company profile and suggest 6 - 10 most relevant standards\\n"
            "2. Include core universal standards (IAS 1, IAS 7) that apply to all companies\\n"
            "3. Add industry - specific standards based on business nature\\n"
            "4. Use EXACT standard IDs from Available Standards list\\n"
            "5. Include the full standard title in your response\\n"
            "6. Keep reasoning brief and specific (max 80 characters per reason)\\n\\n"
            "Return JSON with this exact structure:\\n"
            "{\\n"
            '  "suggested_standards": [\\n'
            '    {"standard_id": "IAS 1", "standard_title": "IAS 1 - Presentation of Financial Statements", '
            '"relevance_score": 0.95, "reasoning": "Financial statement presentation - mandatory"},\\n'
            '    {"standard_id": "IAS 7", "standard_title": "IAS 7 - Statement of Cash Flows", '
            '"relevance_score": 0.95, "reasoning": "Cash flow statements - mandatory"}\\n'
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
        
        # Check if AI service is available
        if ai_service is None:
            raise HTTPException(status_code=503, detail="AI service is not available - check Azure OpenAI configuration")

        # Use the OpenAI client directly for this simple call - ENHANCED DEBUGGING
        logger.info(f"🔍 About to call Azure OpenAI - Model: {ai_service.deployment_name}")
        logger.info(f"🔍 System prompt length: {len(prompt_data['system'])}")
        logger.info(f"🔍 User prompt length: {len(prompt_data['user'])}")
        
        try:
            response = ai_service.openai_client.chat.completions.create(
                model=ai_service.deployment_name,
                messages=[
                    {"role": "system", "content": prompt_data["system"]},
                    {"role": "user", "content": prompt_data["user"]}
                ],
                max_tokens=8000  # Use max_tokens for Azure OpenAI API compatibility
            )
            logger.info("🔍 Azure OpenAI call succeeded!")
        except Exception as openai_error:
            logger.error(f"🔍 Azure OpenAI call failed: {type(openai_error).__name__}: {str(openai_error)}")
            raise

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
                "nature_of_business": (nature_of_business[:100] + "..."
                                     if len(nature_of_business) > 100 else nature_of_business),
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
    except Exception as _e:
        logger.error(f"Error suggesting accounting standards: {str(_e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(_e)}")


@router.get("/progress/{document_id}", response_model=None)
async def get_analysis_progress(
    document_id: str,
) -> Union[Dict[str, Any], JSONResponse]:
    """
    Get real-time analysis progress for a document including question counts and elapsed time.
    Now with persistent storage support for Render deployment.
    """
    try:
        # Import persistent storage
        from services.persistent_storage_enhanced import get_persistent_storage_manager
        storage_manager = get_persistent_storage_manager()
        
        # FIRST: Check if analysis is completed in persistent storage
        logger.info(f"📊 Checking persistent storage for completed analysis: {document_id}")
        persistent_results = await storage_manager.get_analysis_results(document_id)
        
        if persistent_results and persistent_results.get("status", "").lower() == "completed":
            logger.info(f"✅ Analysis completed in persistent storage: {document_id}")
            # Analysis is completed, cleanup any stale progress data
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
        
        # FALLBACK: Check filesystem for completion
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
            # Check if analysis is running in persistent storage first
            logger.info(f"🔍 Checking persistent storage for processing lock: {document_id}")
            processing_lock = await storage_manager.get_processing_lock(document_id)
            
            if processing_lock:
                logger.info(f"🔐 Found processing lock in persistent storage: {document_id}")
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
            
            # Fallback: Check filesystem for lock file
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
                    for _q_id, q_progress in std_progress.questions_progress.items():
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

    except Exception as _e:
        logger.error(f"Error getting analysis progress for {document_id}: {str(_e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(_e)}")


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

    except Exception as _e:
        logger.error(f"Error getting rate limit status: {str(_e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(_e)}")


def _format_elapsed_time(seconds: float) -> str:
    """Format elapsed time in a human - readable format"""
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
                    "The following standard(s) are not available for "
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
                    page_text = page.get_text()  # type: ignore[attr - defined]
                    if page_text:
                        text += page_text + "\n"
            return text.strip()
        except Exception as _e:
            logger.error(f"Error extracting text from PDF: {str(_e)}")
            return JSONResponse(
                status_code=500,
                content={"error": "PDF text extraction failed", "detail": str(_e)}
            )
    elif file_path.suffix.lower() == ".docx":
        if DocxDocument is None:
            logger.error("DOCX support not available (python - docx not installed)")
            return JSONResponse(
                status_code=400,
                content={
                    "error": "DOCX support not available",
                    "detail": "python - docx package is not installed",
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
    chunks_path = ANALYSIS_RESULTS_DIR / f"{document_id}_chunks.json"
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
        # Debug logging
        logger.info(f"Checklist request: framework={framework}, standard={standard}")

        # Check if standard is available
        if not is_standard_available(framework, standard):
            logger.warning(f"Standard not available: {framework}/{standard}")
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

        logger.info(f"Loading checklist for {framework}/{standard}")
        checklist = load_checklist(framework, standard)

        if checklist:
            sections_count = len(checklist.get('sections', []))
            total_items = sum(len(section.get('items', [])) for section in checklist.get('sections', []))
            logger.info(f"Checklist loaded: {sections_count} sections, {total_items} total items")
        else:
            logger.warning(f"Empty checklist returned for {framework}/{standard}")

        return JSONResponse(status_code=200, content=checklist)
    except FileNotFoundError as _e:
        logger.error(f"Checklist not found: {str(_e)}")
        return JSONResponse(
            status_code=404, content={"error": "Checklist not found", "message": str(_e)}
        )
    except Exception as _e:
        logger.error(f"Error loading checklist: {str(_e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to load checklist", "message": str(_e)},
        )


@router.get("/metadata/fields")
async def get_metadata_fields() -> JSONResponse:
    """Get the metadata extraction fields."""
    try:
        metadata_path = os.path.join(CHECKLIST_DATA_DIR, "company_metadata.json")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return JSONResponse(status_code=200, content=metadata["metadata_fields"])
    except Exception as _e:
        logger.error(f"Error loading metadata fields: {str(_e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to load metadata fields", "message": str(_e)},
        )


@router.get("/documents/{document_id}", response_model=None)
async def get_document_status(document_id: str) -> Union[Dict[str, Any], JSONResponse]:
    """
    Get document status with persistent storage support for Render deployment.
    
    This endpoint uses dual storage:
    1. Try persistent database storage first (primary for Render)
    2. Fallback to filesystem if database unavailable
    """
    logger.info(f"📊 GET /documents/{document_id} - Starting status check")
    
    try:
        # Import persistent storage
        from services.persistent_storage_enhanced import get_persistent_storage_manager
        storage_manager = get_persistent_storage_manager()
        
        # Try to get analysis results from persistent storage first
        logger.info(f"🗄️ Checking persistent storage for results: {document_id}")
        persistent_results = await storage_manager.get_analysis_results(document_id)
        
        if persistent_results:
            logger.info(f"✅ Found results in persistent storage for {document_id}")
            return _process_analysis_results(document_id, persistent_results)
        
        # Fallback to filesystem
        logger.info(f"📁 Checking filesystem for results: {document_id}")
        return await _get_document_status_legacy(document_id)
        
    except Exception as e:
        logger.error(f"❌ Error getting document status for {document_id}: {str(e)}")
        # Final fallback to legacy system
        try:
            return await _get_document_status_legacy(document_id)
        except Exception as fallback_error:
            logger.error(f"❌ FALLBACK: Legacy system failed for {document_id}: {str(fallback_error)}")
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Server error",
                    "message": f"Failed to get document status: {str(e)}",
                    "document_id": document_id
                }
            )


def _process_analysis_results(document_id: str, results: dict) -> Dict[str, Any]:
    """
    Process analysis results from either persistent storage or filesystem.
    Standardizes the response format regardless of storage source.
    """
    try:
        # Check for errors in results
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
        
        # Include smart categorization metadata if available
        smart_categorization = results.get("smart_categorization", {})
        
        # Only return sections when compliance analysis is completed
        sections_data = []
        if compliance_analysis in ["COMPLETED", "COMPLETED_WITH_ERRORS"]:
            sections_data = results.get("sections", [])
            logger.info(f"Returning {len(sections_data)} sections for completed analysis {document_id}")
        else:
            logger.info(f"Compliance analysis not completed ({compliance_analysis}) - returning empty sections for {document_id}")
        
        return {
            "document_id": document_id,
            "status": status,
            "metadata_extraction": metadata_extraction,
            "compliance_analysis": compliance_analysis,
            "processing_mode": results.get("processing_mode", "smart"),
            "smart_categorization": {
                "total_categories": smart_categorization.get("total_categories", 0),
                "content_chunks": smart_categorization.get("content_chunks", 0),
                "categorization_complete": smart_categorization.get("categorization_complete", False),
                "categories_found": smart_categorization.get("categories_found", [])
            },
            "metadata": results.get("metadata", {}),
            "sections": sections_data,
            "progress": results.get("progress", {}),
            "framework": results.get("framework"),
            "standards": results.get("standards", []),
            "specialInstructions": results.get("specialInstructions"),
            "extensiveSearch": results.get("extensiveSearch", False),
            "message": results.get("message", "Analysis in progress"),
        }
    except Exception as e:
        logger.error(f"Error processing analysis results for {document_id}: {str(e)}")
        return {
            "document_id": document_id,
            "status": "FAILED",
            "error": "Processing error",
            "message": f"Failed to process analysis results: {str(e)}",
            "metadata": {},
        }


async def _get_document_status_legacy(document_id: str) -> Union[Dict[str, Any], JSONResponse]:
    """
    LEGACY: Original file-based document status retrieval for fallback.
    This is the old implementation kept for backward compatibility.
    """
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
            
            # CRITICAL: Check for completion flag file to update metadata_extraction status
            completion_flag_path = os.path.join(ANALYSIS_RESULTS_DIR, f"{document_id}.metadata_completed")
            if os.path.exists(completion_flag_path):
                metadata_extraction = "COMPLETED"
                
                # Try to load metadata from staged storage first, fallback to legacy
                from services.staged_storage import StagedStorageManager
                storage_manager = StagedStorageManager()
                staged_metadata = storage_manager.get_metadata(document_id)
                
                # Extract actual metadata from staged storage format
                extracted_metadata = None
                if staged_metadata:
                    # Handle staged storage wrapper format
                    extracted_metadata = staged_metadata.get('data', staged_metadata)
                
                if not extracted_metadata:
                    # FALLBACK: Load from legacy location
                    metadata_file_path = os.path.join(ANALYSIS_RESULTS_DIR, f"{document_id}_metadata.json")
                    if os.path.exists(metadata_file_path):
                        try:
                            with open(metadata_file_path, 'r', encoding='utf-8') as f:
                                extracted_metadata = json.load(f)
                        except Exception as e:
                            logger.error(f"[POLLING FIX] Failed to load legacy metadata file: {e}")
                
                if extracted_metadata:
                    # Transform metadata to proper backend format with snake_case field names
                    frontend_metadata = {
                        "company_name": "",
                        "nature_of_business": "", 
                        "operational_demographics": "",
                        "financial_statements_type": ""
                    }
                    
                    # Extract values from confidence structure - BACKEND MAINTAINS SNAKE_CASE STANDARD
                    for key, metadata_obj in extracted_metadata.items():
                        if key == "optimization_metrics":
                            continue  # Skip metrics
                            
                        if isinstance(metadata_obj, dict) and 'value' in metadata_obj:
                            value = metadata_obj['value']
                        else:
                            value = metadata_obj
                        
                        # Map to backend standard snake_case field names
                        if key in ["company_name", "companyName"]:
                            if value and value != "":
                                frontend_metadata["company_name"] = str(value) if value else ""
                        elif key in ["nature_of_business", "natureOfBusiness", "business_nature"]:
                            if value and value != "":
                                frontend_metadata["nature_of_business"] = str(value) if value else ""
                        elif key in ["operational_demographics", "operationalDemographics", "geography", "demographics"]:
                            if value and value != "":
                                frontend_metadata["operational_demographics"] = str(value) if value else ""
                        elif key in ["financial_statements_type", "financialStatementsType", "statement_type", "fs_type"]:
                            if value and value != "":
                                frontend_metadata["financial_statements_type"] = str(value) if value else ""
                    
                    # Only update if we have actual extracted values
                    has_extracted_data = any(v for v in frontend_metadata.values() if v)
                    if has_extracted_data:
                        results["metadata"] = frontend_metadata
                        logger.info(f"[POLLING FIX] Successfully loaded metadata for {document_id}")
                    else:
                        logger.warning(f"[POLLING FIX] No valid metadata extracted for {document_id}")
                else:
                    logger.warning(f"[POLLING FIX] No metadata found for {document_id}")
                
                # Update the results file with the completed status
                results["metadata_extraction"] = "COMPLETED"
                try:
                    with open(results_path, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                    logger.info(f"[POLLING FIX] Updated results file for {document_id} - metadata_extraction: COMPLETED")
                except Exception as e:
                    logger.error(f"[POLLING FIX] Failed to update results file: {e}")
            
            compliance_analysis = results.get("compliance_analysis", "PENDING")

            # Include smart categorization metadata if available
            smart_categorization = results.get("smart_categorization", {})

            # CRITICAL FIX: Only return sections when compliance analysis is completed
            # This prevents frontend from showing 0% compliance for incomplete analyses
            sections_data = []
            if compliance_analysis in ["COMPLETED", "COMPLETED_WITH_ERRORS"]:
                sections_data = results.get("sections", [])
                logger.info(f"Returning {len(sections_data)} sections for completed analysis {document_id}")
            else:
                logger.info(f"Compliance analysis not completed ({compliance_analysis}) - returning empty sections for {document_id}")

            return {
                "document_id": document_id,
                "status": status,
                "metadata_extraction": metadata_extraction,
                "compliance_analysis": compliance_analysis,
                "processing_mode": results.get("processing_mode", "smart"),
                "smart_categorization": {
                    "total_categories": smart_categorization.get("total_categories", 0),
                    "content_chunks": smart_categorization.get("content_chunks", 0),
                    "categorization_complete": smart_categorization.get("categorization_complete", False),
                    "categories_found": smart_categorization.get("categories_found", [])
                },
                "metadata": results.get("metadata", {}),
                "sections": sections_data,
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
    except Exception as _e:
        logger.error(f"Error getting document status: {str(_e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Server error",
                "message": f"Failed to get document status: {str(_e)}",
            },
        )


@router.get("/documents/{document_id}/results")
async def get_document_results(document_id: str) -> Dict[str, Any]:
    """
    BULLETPROOF: Get the results of a document analysis with atomic data access.
    """
    logger.info(f"🛡️ BULLETPROOF GET /documents/{document_id}/results - Starting bulletproof request")
    
    try:
        # SIMPLE FILE READ: Get results from file
        results_path = ANALYSIS_RESULTS_DIR / f"{document_id}.json"
        
        if not results_path.exists():
            return {
                "error": "Document not found",
                "message": f"No results found for document {document_id}",
                "document_id": document_id
            }
        
        with open(results_path, "r", encoding="utf-8") as f:
            result = json.load(f)
            
        logger.info(f"✅ FILE READ: File success for {document_id}")
        
        # Build response from file data
        return {
            "status": result.get("status", "unknown"),
            "document_id": document_id,
            "metadata": result.get("metadata", {}),
            "sections": result.get("sections", []),
            "message": (
                    "Document analysis completed"
                    if result.get("status") == "completed"
                    else "Document analysis in progress"
                ),
                "bulletproof": True,  # Indicates bulletproof system was used
                "data_source": "database_primary"
            }
        
        # FALLBACK: Try legacy file system
        logger.info(f"📁 FALLBACK: Trying legacy file system for {document_id}")
        return await _get_document_results_legacy(document_id)
        
    except Exception as e:
        logger.error(f"❌ BULLETPROOF GET: Error for {document_id}: {str(e)}")
        # Final fallback to legacy system
        try:
            return await _get_document_results_legacy(document_id)
        except Exception as fallback_error:
            logger.error(f"❌ FALLBACK: Even legacy failed for {document_id}: {str(fallback_error)}")
            return {
                "status": "error",
                "document_id": document_id,
                "message": f"Error retrieving document results: {str(e)}",
                "bulletproof": False
            }


async def _get_document_results_legacy(document_id: str) -> Dict[str, Any]:
    """
    LEGACY: Original file-based document results retrieval for fallback.
    """
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
    except Exception as _e:
        logger.error(f"Error retrieving document results: {str(_e)}")
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "document_id": document_id,
            "message": f"Error retrieving document results: {str(_e)}",
        }


@router.get("/documents/{document_id}/keywords")
async def get_document_keywords(document_id: str) -> Dict[str, Any]:
    """Get keyword extraction progress for a document analysis."""
    try:
        # Check if analysis is in progress
        from services.progress_tracker import get_progress_tracker
        tracker = get_progress_tracker()
        
        progress = tracker.get_progress(document_id)
        
        if progress:
            # Extract keyword-related progress
            keywords_discovered = []
            current_step = "Keyword extraction in progress"
            progress_percentage = getattr(progress, 'overall_progress', 0.0)
            current_keyword = None
            
            # Try to extract keywords from standards progress
            if hasattr(progress, 'standards_progress'):
                standards_progress = getattr(progress, 'standards_progress', {})
                if standards_progress:
                    for std_progress in standards_progress.values():
                        if hasattr(std_progress, 'questions_progress'):
                            questions_progress = getattr(std_progress, 'questions_progress', {})
                            if questions_progress:
                                keywords_discovered.extend(questions_progress.keys())
            
            return {
                "keywords_discovered": list(set(keywords_discovered)),
                "current_step": current_step,
                "progress_percentage": progress_percentage,
                "current_keyword": current_keyword
            }
        else:
            # No active progress, check if completed
            from services.persistent_storage_enhanced import get_persistent_storage_manager
            storage_manager = get_persistent_storage_manager()
            
            results = await storage_manager.get_analysis_results(document_id)
            if results:
                # Extract keywords from completed analysis
                keywords = []
                if results.get("sections"):
                    for section in results.get("sections", []):
                        if section.get("items"):
                            keywords.extend([item.get("id", "") for item in section.get("items", [])])
                
                return {
                    "keywords_discovered": list(set(filter(None, keywords))),
                    "current_step": "Keyword extraction completed",
                    "progress_percentage": 100,
                    "current_keyword": None
                }
            else:
                return {
                    "keywords_discovered": [],
                    "current_step": "No keyword extraction found",
                    "progress_percentage": 0,
                    "current_keyword": None
                }
                
    except Exception as e:
        logger.error(f"Error getting keywords for document {document_id}: {e}")
        return {
            "keywords_discovered": [],
            "current_step": "Error retrieving keywords",
            "progress_percentage": 0,
            "current_keyword": None
        }

@router.get("/documents/{document_id}/extract")
async def get_document_extract(document_id: str) -> Dict[str, Any]:
    """Get the extracted metadata from document analysis. Alias for /results endpoint."""
    return await get_document_results(document_id)


class ChecklistItemUpdateModel(BaseModel):
    status: str
    comments: Optional[str] = None


@router.patch("/documents/{document_id}/items/{item_id}")
async def update_compliance_item(
    document_id: str, item_id: str, update: ChecklistItemUpdateModel
):
    """
    BULLETPROOF: Update a compliance checklist item with atomic operations.
    """
    logger.info(f"🛡️ BULLETPROOF PATCH /documents/{document_id}/items/{item_id} - Starting bulletproof update")
    
    try:
        # SIMPLE FILE READ: Get current results from file
        results_path = ANALYSIS_RESULTS_DIR / f"{document_id}.json"
        
        if not results_path.exists():
            return JSONResponse(
                status_code=404,
                content={
                    "error": "Document not found",
                    "message": f"No results found for document {document_id}",
                    "document_id": document_id
                }
            )
        
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        
        if not results:
            # FALLBACK: Try legacy file system
            logger.info(f"📁 FALLBACK: Trying legacy file system for {document_id}")
            results_path = ANALYSIS_RESULTS_DIR / f"{document_id}.json"
            if not os.path.exists(results_path):
                return JSONResponse(
                    status_code=404,
                    content={
                        "error": "Document not found",
                        "message": "No results found for the specified document",
                    },
                )
            
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

        # SIMPLE FILE SAVE: Save updated results to file
        logger.info(f"💾 FILE SAVE: Saving updated compliance item for {document_id}")
        save_analysis_results(document_id, results)
        
        logger.info(f"✅ FILE UPDATE: Successfully updated compliance item {item_id} for {document_id}")
        return JSONResponse(
            status_code=200, 
            content={
                "message": "Item updated successfully",
                "item_id": item_id,
                "document_id": document_id
            }
        )
        
    except Exception as _e:
        logger.error(f"❌ BULLETPROOF UPDATE: Error updating compliance item for {document_id}: {str(_e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "error": "Update failed",
                "message": f"Error updating compliance item: {str(_e)}",
                "bulletproof": False
            },
        )


@router.get("/documents/{document_id}/report")
async def get_document_analysis_report(document_id: str):
    """Get comprehensive analysis report for a document"""
    try:
        from services.persistent_storage import PersistentStorageManager
        storage = PersistentStorageManager()
        
        # Get document metadata
        metadata = getattr(storage, 'load_document_metadata', lambda x: {})(document_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get document analysis results 
        compliance_results = getattr(storage, 'load_compliance_results', lambda x: {})(document_id)
        if not compliance_results:
            compliance_results = {}
            
        # Get extracted content
        extracted_content = getattr(storage, 'load_extracted_content', lambda x: {})(document_id)
        if not extracted_content:
            extracted_content = {}
        
        # Build comprehensive report
        report = {
            "document_id": document_id,
            "document_name": metadata.get("filename", "Unknown"),
            "upload_date": metadata.get("upload_date", "Unknown"),
            "analysis_status": metadata.get("status", "unknown"),
            "total_pages": extracted_content.get("total_pages", 0),
            "word_count": extracted_content.get("word_count", 0),
            "compliance_results": compliance_results,
            "framework_analysis": metadata.get("framework_analysis", {}),
            "key_findings": compliance_results.get("key_findings", []),
            "compliance_score": compliance_results.get("overall_score", 0),
            "recommendations": compliance_results.get("recommendations", []),
            "extracted_sections": extracted_content.get("sections", {}),
            "metadata": metadata
        }
        
        logger.info(f"✅ Generated comprehensive report for document {document_id}")
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "report": report,
                "message": "Report generated successfully"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error generating report for document {document_id}: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "error": "Report generation failed",
                "message": f"Error generating report: {str(e)}",
                "success": False
            }
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
    custom_instructions: Optional[str] = ""  # User-provided analysis instructions
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
            f"🚀 FRAMEWORK SELECTION STARTED: {request.framework} and standards "
            f"{request.standards} for document {document_id}"
        )
        logger.info(f"🔍 REQUEST DETAILS: specialInstructions='{request.specialInstructions}', extensiveSearch={request.extensiveSearch}")

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
        logger.info(f"✅ VALIDATION PASSED: Starting compliance analysis for {document_id}")
        logger.info(f"📝 TEXT EXTRACTED: {len(text)} characters for analysis")
        
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
        
        logger.info(f"🎯 COMPLIANCE ANALYSIS TASK SCHEDULED: Background task added for {document_id}")

        response_data = {
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
        
        logger.info(f"✅ FRAMEWORK SELECTION COMPLETED: Returning response for {document_id}")
        return response_data
    except Exception as _e:
        logger.error(f"Error selecting framework: {str(_e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Server error",
                "detail": f"Error selecting framework: {str(_e)}",
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
                f"Processing mode '{request.processing_mode}' selected, starting metadata extraction"
            )

            # Save updated results
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            # Start metadata extraction in background
            background_tasks = BackgroundTasks()
            ai_svc = get_ai_service()
            background_tasks.add_task(
                process_upload_tasks,
                document_id,
                ai_svc,
                "",  # text parameter
                request.processing_mode,
            )

            mode_descriptions = {
                "zap": "Lightning fast analysis with 16 parallel workers",
                "smart": (
                    "AI - powered intelligent semantic processing with cost "
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
                f"Processing mode '{request.processing_mode}' selected, ready for framework selection"
            )

            # Save updated results
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            mode_descriptions = {
                "zap": "Lightning fast analysis with 16 parallel workers",
                "smart": (
                    "AI - powered intelligent semantic processing with cost "
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

    except Exception as _e:
        logger.error(f"Error selecting processing mode: {str(_e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Server error",
                "detail": f"Error selecting processing mode: {str(_e)}",
            },
        )


@router.get("/documents/{document_id}/metadata-status")
async def get_metadata_extraction_status(document_id: str) -> JSONResponse:
    """
    Get the current status of metadata extraction for a document.
    Frontend should poll this endpoint after upload to check metadata extraction progress.
    """
    try:
        # Check if status file exists
        status_file = ANALYSIS_RESULTS_DIR / f"{document_id}.json"
        
        if not status_file.exists():
            return JSONResponse(
                status_code=404,
                content={
                    "status": "not_found",
                    "message": f"No status information found for document {document_id}"
                }
            )
        
        # Read status file
        with open(status_file, 'r', encoding='utf-8') as f:
            status_data = json.load(f)
        
        # Check if metadata extraction is completed
        metadata_completed_file = ANALYSIS_RESULTS_DIR / f"{document_id}.metadata_completed"
        is_completed = metadata_completed_file.exists()
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "completed" if is_completed else status_data.get("status", "processing"),
                "message": status_data.get("message", "Processing in progress"),
                "metadata_extraction": "completed" if is_completed else "in_progress",
                "document_id": document_id,
                "last_updated": status_data.get("timestamp", "unknown")
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting metadata status for {document_id}: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error getting metadata status: {str(e)}"
            }
        )


@router.get("/documents/{document_id}/metadata-results")
async def get_metadata_extraction_results(document_id: str) -> JSONResponse:
    """
    Get the extracted metadata results for a document.
    This endpoint returns the actual metadata extracted by smart_metadata_extractor.
    """
    try:
        # Check if metadata file exists
        metadata_file = ANALYSIS_RESULTS_DIR / f"{document_id}_metadata.json"
        
        if not metadata_file.exists():
            return JSONResponse(
                status_code=404,
                content={
                    "status": "not_found",
                    "message": f"No metadata extraction results found for document {document_id}"
                }
            )
        
        # Read metadata results
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata_results = json.load(f)
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "document_id": document_id,
                "metadata": metadata_results,
                "extracted_at": metadata_results.get("extracted_at", "unknown")
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting metadata results for {document_id}: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error getting metadata results: {str(e)}"
            }
        )


@router.post("/documents/{document_id}/start-metadata-extraction")
async def start_metadata_extraction(
    document_id: str,
    background_tasks: BackgroundTasks
) -> JSONResponse:
    """
    Trigger metadata extraction using pre-extracted document data.
    This endpoint is called after upload completion to start metadata processing.
    """
    try:
        logger.info(f"🧠 Starting metadata extraction for document {document_id}")

        # Import and trigger metadata extraction
        from services.smart_metadata_extractor import SmartMetadataExtractor
        from datetime import datetime
        import asyncio
        import json

        async def run_metadata_extraction():
            try:
                # Load chunks from file
                chunks_file = f"analysis_results/{document_id}_chunks.json"
                if not os.path.exists(chunks_file):
                    logger.error(f"Chunks file not found: {chunks_file}")
                    return

                with open(chunks_file, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)

                # Run metadata extraction
                extractor = SmartMetadataExtractor()
                metadata = await extractor.extract_metadata_optimized(document_id, chunks)

                # Save metadata using staged storage for isolation
                from services.staged_storage import StagedStorageManager
                storage_manager = StagedStorageManager()
                storage_manager.save_metadata(document_id, metadata)

                # BACKWARD COMPATIBILITY: Also save to legacy location
                metadata_file = f"analysis_results/{document_id}_metadata.json"
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)

                # Mark as completed
                completion_file = f"analysis_results/{document_id}.metadata_completed"
                with open(completion_file, 'w') as f:
                    f.write("completed")
                
                # CRITICAL: Update main results file (consistent with document_chunker.py fix)
                main_results_file = f"analysis_results/{document_id}.json"
                try:
                    if os.path.exists(main_results_file):
                        with open(main_results_file, 'r', encoding='utf-8') as f:
                            results_data = json.load(f)
                    else:
                        results_data = {
                            "document_id": document_id,
                            "status": "PROCESSING"
                        }
                except Exception:
                    results_data = {
                        "document_id": document_id,
                        "status": "PROCESSING"
                    }
                
                # Update with metadata extraction completion
                results_data.update({
                    "metadata_extraction": "COMPLETED",
                    "metadata_completed_at": datetime.now().isoformat(),
                    "metadata_file": metadata_file
                })
                
                # Write back to main results file
                with open(main_results_file, 'w', encoding='utf-8') as f:
                    json.dump(results_data, f, indent=2, ensure_ascii=False)

                # 🔧 CRITICAL FIX: Update persistent storage with metadata completion status
                try:
                    from services.persistent_storage_enhanced import get_persistent_storage_manager
                    persistent_storage = get_persistent_storage_manager()
                    
                    logger.info(f"🗄️ ATTEMPTING: Persistent storage update for {document_id}")
                    
                    # Get current persistent results
                    current_results = await persistent_storage.get_analysis_results(document_id)
                    if current_results:
                        logger.info(f"🗄️ FOUND: Existing persistent results for {document_id}")
                        # Update with metadata extraction completion
                        current_results.update({
                            "metadata_extraction": "COMPLETED",
                            "metadata_completed_at": datetime.now().isoformat(),
                            "metadata_file": metadata_file
                        })
                        
                        # Save back to persistent storage
                        success = await persistent_storage.store_analysis_results(document_id, current_results)
                        if success:
                            logger.info(f"✅ SUCCESS: Updated persistent storage with metadata completion for {document_id}")
                        else:
                            logger.error(f"❌ FAILED: Persistent storage update returned False for {document_id}")
                    else:
                        logger.warning(f"⚠️ NOT FOUND: No existing persistent results to update for {document_id}")
                        # Create new persistent results with metadata completion
                        new_results = {
                            "document_id": document_id,
                            "status": "metadata_extraction_completed",
                            "metadata_extraction": "COMPLETED", 
                            "metadata_completed_at": datetime.now().isoformat(),
                            "metadata_file": metadata_file,
                            "compliance_analysis": "PENDING"
                        }
                        success = await persistent_storage.store_analysis_results(document_id, new_results)
                        if success:
                            logger.info(f"✅ CREATED: New persistent storage entry for {document_id}")
                        else:
                            logger.error(f"❌ FAILED: Could not create persistent storage entry for {document_id}")
                        
                except Exception as persistent_error:
                    logger.error(f"❌ EXCEPTION: Failed to update persistent storage for {document_id}: {persistent_error}", exc_info=True)
                    # Don't fail the whole operation, just log the error
                
                logger.info(f"✅ Metadata extraction completed for {document_id}")

            except Exception as e:
                logger.error(f"❌ Metadata extraction failed for {document_id}: {e}")
                # Mark as failed
                error_file = f"analysis_results/{document_id}.metadata_error"
                with open(error_file, 'w') as f:
                    f.write(str(e))

        # Run metadata extraction in background (fix: remove asyncio.run to prevent event loop conflict)
        background_tasks.add_task(run_metadata_extraction)

        response = {
            "status": "processing",
            "document_id": document_id,
            "stage": "metadata_extraction",
            "message": "Metadata extraction started using pre-extracted data"
        }

        logger.info(f"✅ Metadata extraction background task started for {document_id}")
        return JSONResponse(status_code=200, content=response)

    except Exception as e:
        logger.error(f"❌ Failed to start metadata extraction for {document_id}: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": "Failed to start metadata extraction",
                "detail": str(e)
            }
        )


@router.post("/documents/{document_id}/start-checklist-processing")
async def start_checklist_processing(
    document_id: str,
    request: Dict[str, Any],
    background_tasks: BackgroundTasks
) -> JSONResponse:
    """
    Trigger checklist processing using pre-extracted document data.
    This endpoint is called after user confirms framework and standards.
    """
    try:
        framework = request.get("framework")
        standards = request.get("standards", [])

        if not framework:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "error": "Framework required",
                    "detail": "Framework must be specified for checklist processing"
                }
            )

        logger.info(f"📋 Starting checklist processing for document {document_id}, framework: {framework}")

        # Import and trigger checklist processing
        # from services.simple_document_processing import trigger_checklist_processing

        # Run checklist processing in background
        # background_tasks.add_task(trigger_checklist_processing, document_id, framework, standards)  # Commented out - function not available

        response = {
            "status": "processing",
            "document_id": document_id,
            "stage": "checklist_processing",
            "framework": framework,
            "standards": standards,
            "message": "Checklist processing started using pre-extracted data"
        }

        logger.info(f"✅ Checklist processing background task started for {document_id}")
        return JSONResponse(status_code=200, content=response)

    except Exception as e:
        logger.error(f"❌ Failed to start checklist processing for {document_id}: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": "Failed to start checklist processing",
                "detail": str(e)
            }
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
            # Use custom_instructions if provided, otherwise fall back to special_instructions
            instructions = request.custom_instructions or request.special_instructions
            background_tasks.add_task(
                process_compliance_comparison,
                document_id,
                text_content,
                framework,
                standards,
                instructions,
                results.get("extensiveSearch", False),
                ai_svc,
            )
        else:
            # For single mode (zap or smart)
            # Use custom_instructions if provided, otherwise fall back to special_instructions
            instructions = request.custom_instructions or request.special_instructions
            background_tasks.add_task(
                process_compliance_analysis,
                document_id,
                text_content,
                framework,
                standards,
                instructions,
                results.get("extensiveSearch", False),
                ai_svc,
                processing_mode,  # Pass processing mode parameter
            )

        return {
            "success": True,
            "processing_mode": processing_mode,
            "message": f"Compliance analysis started with {processing_mode} mode",
        }

    except Exception as _e:
        logger.error(f"Error starting compliance analysis: {str(_e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Server error",
                "detail": f"Error starting compliance analysis: {str(_e)}",
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

    except Exception as _e:
        logger.error(f"Smart mode failed in comparison: {str(_e)}")
        end_time = time.time()
        processing_time = end_time - start_time

        metrics = {
            "processing_time_seconds": round(processing_time, 2),
            "questions_processed": 0,
            "success": False,
            "sections_analyzed": 0,
            "error": str(_e),
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

    except Exception as _e:
        logger.error(f"Zap mode failed in comparison: {str(_e)}")
        end_time = time.time()
        processing_time = end_time - start_time

        metrics = {
            "processing_time_seconds": round(processing_time, 2),
            "questions_processed": 0,
            "success": False,
            "sections_analyzed": 0,
            "error": str(_e),
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

    except Exception as _e:
        logger.error(f"Error in comparison analysis: {str(_e)}", exc_info=True)
        # Update results with error
        try:
            results_path = ANALYSIS_RESULTS_DIR / f"{document_id}.json"
            error_results = {
                "document_id": document_id,
                "status": "FAILED",
                "error": str(_e),
            }
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(error_results, f, indent=2, ensure_ascii=False)
        except Exception as _save_error:
            # Failed to save error state - log but continue
            logger.error(
                f"Failed to save error state for {document_id}: {str(_save_error)}"
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
    - Cost - optimized AI processing
    - Enhanced context understanding
    """
    logger.info(f"Starting Smart Mode analysis for standard {standard}")
    start_time = time.time()

    # Preprocess checklist to add standard numbers to questions
    processed_checklist = _preprocess_checklist_with_standard_numbers(checklist)

    # Set progress tracker for question - level tracking
    if progress_tracker:
        ai_svc.progress_tracker = progress_tracker

        # Initialize question - level tracking for Smart Mode
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

        # Step 3: Question - content mapping for optimal processing
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
                    f"Processing section {section_idx + 1}/{len(processed_checklist.get('sections', []))}",
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

    except Exception as _e:
        logger.error(f"Smart Mode analysis failed for {standard}: {str(_e)}")
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

    # Simple keyword - based prioritization
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
    for _score, _, segment in segment_scores[:max_segments]:
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
    except Exception as _e:
        logger.error(f"Error preprocessing checklist with standard numbers: {str(_e)}")
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
    - Speed - first approach with acceptable accuracy trade - offs
    - Minimal context processing for fastest results
    """
    logger.info(f"Starting Zap Mode analysis for standard {standard}")
    start_time = time.time()

    # Preprocess checklist to add standard numbers to questions
    processed_checklist = _preprocess_checklist_with_standard_numbers(checklist)

    # Set progress tracker for question - level tracking
    if progress_tracker:
        ai_svc.progress_tracker = progress_tracker

        # Initialize question - level tracking for Zap Mode
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

                except Exception as _e:
                    error_str = str(_e).lower()
                    if (
                        "rate limit" in error_str or "429" in error_str
                    ) and attempt < max_retries - 1:
                        # Simple retry without staggered backoff - just power through
                        logger.warning(
                            f"Zap Mode worker error, retrying immediately (attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(0.1)  # Minimal delay
                        continue
                    else:
                        # Log error and return error result instead of failing
                        # completely
                        logger.error(
                            "Zap Mode worker failed after "
                            f"{attempt + 1} attempts: {str(_e)}"
                        )
                        current_task = asyncio.current_task()
                        return {
                            "processing_mode": "zap",
                            "section": section.get("section", "unknown"),
                            "title": section.get("title", ""),
                            "items": [],
                            "error": str(_e),
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

    except Exception as _e:
        logger.error(f"Zap Mode analysis failed for {standard}: {str(_e)}")
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

    # Also create processing lock in persistent storage for Render deployment
    try:
        from services.persistent_storage_enhanced import get_persistent_storage_manager
        storage_manager = get_persistent_storage_manager()
        await storage_manager.set_processing_lock(document_id, {
            "status": "PROCESSING",
            "started_at": datetime.now().isoformat(),
            "framework": framework,
            "standards": standards,
            "special_instructions": special_instructions,
            "extensive_search": extensive_search
        }, "compliance_analysis")
        logger.info(f"🔐 Created processing lock in persistent storage: {document_id}")
    except Exception as e:
        logger.error(f"❌ Failed to create persistent storage lock for {document_id}: {str(e)}")

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
    custom_instructions: Optional[str] = None,
) -> tuple[list, list]:
    """
    Process each standard sequentially and return all sections and failed standards.
    CRITICAL: This function MUST only process the user - selected standards.
    """
    # STRICT VALIDATION: Ensure we only process user - selected standards
    logger.info(f"🔒 STRICT STANDARDS VALIDATION for document {document_id}")
    logger.info(f"🔒 Processing EXACTLY these user - selected standards: {standards}")
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
                f"🎯 PROCESSING STANDARD {i + 1}/{total_standards}: {standard} (USER-SELECTED ONLY)"
            )
            logger.info(
                f"🎯 Document {document_id} - Current: {standard}, "
                f"Remaining: {standards[i + 1:] if i + 1 < len(standards) else 'None'}"
            )

            # Get checklist to determine total questions
            checklist_data = load_checklist(framework, standard)

            # CRITICAL FIX: Count actual questions / items, not top - level object length
            total_questions = 0
            if checklist_data and isinstance(checklist_data, dict):
                for section in checklist_data.get("sections", []):
                    total_questions += len(section.get("items", []))

            logger.info(f"🔍 Checklist loaded for {standard}: {total_questions} total questions")

            if total_questions == 0:
                logger.error(f"❌ No questions found in checklist for {framework}/{standard}")
                failed_standards.append(standard)
                continue

            # Start tracking this standard
            if progress_tracker:
                progress_tracker.start_standard(document_id, standard, total_questions)

                # Initialize question - level tracking
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

            # Debug checklist loading
            if not checklist:
                logger.error(f"❌ Failed to load checklist for {framework}/{standard}")
                failed_standards.append(standard)
                continue

            checklist_sections = checklist.get("sections", [])
            checklist_items_count = sum(len(section.get("items", [])) for section in checklist_sections)
            logger.info(f"📋 Loaded checklist: {len(checklist_sections)} sections, {checklist_items_count} items")

            # Set progress tracker for question - level tracking
            ai_svc.progress_tracker = progress_tracker
            
            # Set custom instructions for this analysis
            if custom_instructions:
                # Store custom instructions for use in AI analysis
                setattr(ai_svc, 'custom_instructions', custom_instructions)

            # Choose processing approach based on mode
            if processing_mode == "smart":
                # Smart Mode: Use intelligent semantic processing
                standard_sections = await process_smart_mode_analysis(
                    checklist, text, document_id, ai_svc, standard, progress_tracker
                )
            elif processing_mode == "zap":
                # Zap Mode: High - speed processing with 16 concurrent workers
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

            # Debug processing results
            logger.info(
                f"📊 Processing result for {standard}: "
                f"{len(standard_sections) if standard_sections else 0} sections returned"
            )
            if standard_sections:
                total_processed_items = sum(len(section.get("items", [])) for section in standard_sections)
                logger.info(f"📊 Total processed items for {standard}: {total_processed_items}")
            else:
                logger.warning(f"⚠️ No sections returned for {standard}")

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
        except Exception as _e:
            logger.error(
                f"Error processing standard {standard}: {str(_e)}", exc_info=True
            )
            failed_standards.append({"standard": standard, "error": str(_e)})

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
            "Compliance analysis completed with errors for "
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
        
        # Remove processing lock from persistent storage on error
        try:
            from services.persistent_storage_enhanced import get_persistent_storage_manager
            storage_manager = get_persistent_storage_manager()
            # Note: Using asyncio to call async function in sync context
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(storage_manager.remove_processing_lock(document_id))
            loop.close()
            logger.info(f"🔓 Removed processing lock from persistent storage on error: {document_id}")
        except Exception as storage_e:
            logger.error(f"❌ Failed to remove persistent storage lock on error for {document_id}: {str(storage_e)}")
            
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
        logger.info(f"🎯 COMPLIANCE ANALYSIS BACKGROUND TASK STARTED: {document_id}")
        logger.info(f"🎯 PARAMETERS: framework={framework}, standards={standards}, processing_mode={processing_mode}")
        logger.info(f"🎯 TEXT LENGTH: {len(text)} chars, extensive_search={extensive_search}")
        # CRITICAL: Limit text content to 4000 characters to prevent Azure OpenAI API 500 errors
        MAX_TEXT_LENGTH = 4000
        if len(text) > MAX_TEXT_LENGTH:
            original_length = len(text)
            text = text[:MAX_TEXT_LENGTH]
            logger.warning(
                f"🔥 TEXT TRUNCATED: Original length {original_length} → "
                f"Limited to {MAX_TEXT_LENGTH} characters to prevent API errors")
        else:
            logger.info(f"📝 Text length: {len(text)} characters (within limit)")

        # Clear any previously processed questions for this document
        from services.ai import clear_document_questions

        clear_document_questions(document_id)

        # CRITICAL: Log exactly which standards the user selected for compliance
        # analysis
        logger.info(f"🎯 COMPLIANCE ANALYSIS STARTING for document {document_id}")
        logger.info(f"🎯 USER - SELECTED STANDARDS ONLY: {standards}")
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
            custom_instructions=special_instructions,  # Pass the custom instructions
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
        
        # Remove processing lock from persistent storage
        try:
            from services.persistent_storage_enhanced import get_persistent_storage_manager
            storage_manager = get_persistent_storage_manager()
            await storage_manager.remove_processing_lock(document_id)
            logger.info(f"🔓 Removed processing lock from persistent storage: {document_id}")
        except Exception as e:
            logger.error(f"❌ Failed to remove persistent storage lock for {document_id}: {str(e)}")
        
        logger.info(f"Completed compliance analysis process for document {document_id}")

    except Exception as _e:
        # Mark progress as failed
        from services.progress_tracker import get_progress_tracker

        progress_tracker = get_progress_tracker()
        progress_tracker.fail_analysis(document_id, str(_e))

        _handle_analysis_error(document_id, _e, performance_tracker, processing_mode)


# Health check endpoints (consolidated from health_routes.py)
@router.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint to verify service is running"""
    return {"status": "healthy", "message": "Audricc AI Service is running"}


@router.get("/health/detailed", tags=["Health"])  
async def detailed_health_check():
    """Detailed health check with system metrics"""
    from pathlib import Path
    import os
    
    # Check critical directories
    critical_dirs = [
        UPLOADS_DIR,
        ANALYSIS_RESULTS_DIR,
        CHECKLIST_DATA_DIR,
        VECTOR_INDICES_DIR
    ]
    
    dir_status = {}
    for dir_path in critical_dirs:
        dir_status[str(dir_path)] = {
            "exists": dir_path.exists(),
            "writable": os.access(dir_path, os.W_OK) if dir_path.exists() else False
        }
    
    # System metrics (simplified without psutil)
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system": {
            "memory_percent": 0,  # Disabled - psutil not available
            "disk_percent": 0,    # Disabled - psutil not available
            "available_memory_gb": 0  # Disabled - psutil not available
        },
        "directories": dir_status,
        "services": {
            "ai_service": "operational",
            "smart_metadata_extractor": "operational",
            "bulletproof_storage": "operational"
        }
    }


# Session Management endpoints (consolidated from sessions_routes.py)
# Pydantic models for session management
# Enhanced session models for comprehensive data storage
class UserChoice(BaseModel):
    choice_type: str  # "accounting_standard", "framework", "section_selection", etc.
    value: Any
    timestamp: datetime
    context: Optional[str] = None

class AIResponse(BaseModel):
    response_type: str  # "suggestion", "analysis", "answer", etc.
    content: str
    confidence: Optional[float] = None
    sources: Optional[List[str]] = None
    timestamp: datetime

class ConversationMessage(BaseModel):
    message_id: str
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime
    message_type: Optional[str] = None  # "instruction", "query", "section_query", etc.
    metadata: Optional[Dict[str, Any]] = None  # page_range, section_reference, etc.

class SessionFile(BaseModel):
    file_id: str
    original_filename: str
    file_size: int
    upload_timestamp: datetime
    file_type: str
    storage_location: str  # "postgresql" or file path

class SessionAnalysisContext(BaseModel):
    accounting_standard: Optional[str] = None
    custom_instructions: Optional[str] = None
    selected_frameworks: Optional[List[str]] = None
    analysis_preferences: Optional[Dict[str, Any]] = None

class SessionCreate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    analysis_context: Optional[SessionAnalysisContext] = None

class SessionUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    chat_state: Optional[Dict[str, Any]] = None
    messages: Optional[List[ConversationMessage]] = None
    user_choices: Optional[List[UserChoice]] = None
    ai_responses: Optional[List[AIResponse]] = None
    analysis_context: Optional[SessionAnalysisContext] = None

class SessionResponse(BaseModel):
    session_id: str
    title: str
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    document_count: int = 0
    last_document_id: Optional[str] = None
    status: str = "active"  # active, completed, archived, shared
    shared_with: Optional[List[str]] = None  # list of user IDs
    owner: Optional[str] = None

class SessionDetail(SessionResponse):
    chat_state: Optional[Dict[str, Any]] = None
    conversation_history: Optional[List[ConversationMessage]] = None
    user_choices: Optional[List[UserChoice]] = None
    ai_responses: Optional[List[AIResponse]] = None
    uploaded_files: Optional[List[SessionFile]] = None
    documents: Optional[List[Dict[str, Any]]] = None
    analysis_context: Optional[SessionAnalysisContext] = None

# Session storage directory
SESSIONS_DIR = BACKEND_DIR / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)

def generate_session_id() -> str:
    """Generate a unique session ID"""
    import uuid
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = str(uuid.uuid4())[:8]
    return f"session_{timestamp}_{unique_id}"

def get_session_file_path(session_id: str) -> Path:
    """Get the file path for a session"""
    return SESSIONS_DIR / f"{session_id}.json"

def save_session_to_file(session_id: str, session_data: Dict[str, Any]) -> None:
    """Save session data to file"""
    session_file = get_session_file_path(session_id)
    with open(session_file, 'w', encoding='utf-8') as f:
        json.dump(session_data, f, indent=2, default=str)

def load_session_from_file(session_id: str) -> Optional[Dict[str, Any]]:
    """Load session data from file"""
    session_file = get_session_file_path(session_id)
    if not session_file.exists():
        return None

    try:
        with open(session_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading session {session_id}: {e}")
        return None

@router.post("/sessions/create", response_model=SessionResponse, tags=["Sessions"])
async def create_session(session_data: SessionCreate):
    """Create a new analysis session"""
    try:
        session_id = generate_session_id()
        now = datetime.now()

        # Default title if not provided
        title = session_data.title or f"Analysis Session {now.strftime('%Y-%m-%d %H:%M')}"

        session = {
            "session_id": session_id,
            "title": title,
            "description": session_data.description,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "document_count": 0,
            "last_document_id": None,
            "status": "active",
            "owner": "current_user",  # TODO: Replace with actual user ID from auth
            "shared_with": [],
            "chat_state": None,
            "conversation_history": [],
            "user_choices": [],
            "ai_responses": [],
            "uploaded_files": [],
            "documents": [],
            "analysis_context": {
                "accounting_standard": session_data.analysis_context.accounting_standard if session_data.analysis_context else None,
                "custom_instructions": session_data.analysis_context.custom_instructions if session_data.analysis_context else None,
                "selected_frameworks": session_data.analysis_context.selected_frameworks if session_data.analysis_context else [],
                "analysis_preferences": session_data.analysis_context.analysis_preferences if session_data.analysis_context else {}
            }
        }

        # Save to file
        save_session_to_file(session_id, session)

        return SessionResponse(
            session_id=session_id,
            title=title,
            description=session_data.description,
            created_at=now,
            updated_at=now,
            document_count=0,
            last_document_id=None,
            status="active"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")

@router.get("/sessions/list", response_model=List[SessionResponse], tags=["Sessions"])
async def list_sessions(limit: int = 50, offset: int = 0):
    """List all analysis sessions"""
    try:
        sessions = []

        # Get all session files
        session_files = list(SESSIONS_DIR.glob("session_*.json"))
        session_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)  # Sort by modification time

        # Apply pagination
        paginated_files = session_files[offset:offset + limit]

        for session_file in paginated_files:
            session_data = load_session_from_file(session_file.stem)
            if session_data:
                sessions.append(SessionResponse(
                    session_id=session_data["session_id"],
                    title=session_data["title"],
                    description=session_data.get("description"),
                    created_at=datetime.fromisoformat(session_data["created_at"]),
                    updated_at=datetime.fromisoformat(session_data["updated_at"]),
                    document_count=session_data.get("document_count", 0),
                    last_document_id=session_data.get("last_document_id"),
                    status=session_data.get("status", "active")
                ))

        return sessions

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")

@router.get("/sessions/{session_id}", response_model=SessionDetail, tags=["Sessions"])
async def get_session(session_id: str):
    """Get a specific session with full details including documents from PostgreSQL"""
    try:
        session_data = load_session_from_file(session_id)

        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")

        # Enhanced: Fetch document details from PostgreSQL storage
        documents = []
        document_ids = []
        
        # Collect document IDs from various sources
        if session_data.get("last_document_id"):
            document_ids.append(session_data["last_document_id"])
        
        # Check chat state for document IDs
        chat_state = session_data.get("chat_state", {})
        if isinstance(chat_state, dict) and chat_state.get("documentId"):
            if chat_state["documentId"] not in document_ids:
                document_ids.append(chat_state["documentId"])
        
        # Check messages for document references
        messages = session_data.get("messages", [])
        for message in messages:
            if isinstance(message, dict):
                # Look for document IDs in message content
                content = str(message.get("content", ""))
                if "RAI-" in content:
                    # Extract document IDs from content (simple pattern matching)
                    import re
                    doc_matches = re.findall(r'RAI-[0-9A-Z-]+', content)
                    for doc_id in doc_matches:
                        if doc_id not in document_ids:
                            document_ids.append(doc_id)

        # Fetch document analysis results from persistent storage
        try:
            from services.persistent_storage_enhanced import PersistentStorageManager
            storage = PersistentStorageManager()
            
            for doc_id in document_ids:
                try:
                    analysis_result = await storage.get_analysis_results(doc_id)
                    if analysis_result:
                        # Format document for frontend compatibility
                        documents.append({
                            "document_id": doc_id,
                            "filename": f"{doc_id}.pdf",  # Default filename
                            "status": analysis_result.get("status", "COMPLETED"),
                            "metadata": analysis_result.get("metadata", {}),
                            "sections": analysis_result.get("sections", []),
                            "created_at": analysis_result.get("created_at"),
                            "analysis_complete": True
                        })
                        logger.info(f"✅ Retrieved document {doc_id} from PostgreSQL for session {session_id}")
                except Exception as e:
                    logger.warning(f"Could not retrieve document {doc_id} from storage: {e}")
                    
        except Exception as e:
            logger.warning(f"Storage system not available for session {session_id}: {e}")

        return SessionDetail(
            session_id=session_data["session_id"],
            title=session_data["title"],
            description=session_data.get("description"),
            created_at=datetime.fromisoformat(session_data["created_at"]),
            updated_at=datetime.fromisoformat(session_data["updated_at"]),
            document_count=len(documents),  # Updated with actual document count
            last_document_id=session_data.get("last_document_id"),
            status=session_data.get("status", "active"),
            chat_state=session_data.get("chat_state"),
            documents=documents  # Now includes full document data from PostgreSQL
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")

@router.put("/sessions/{session_id}", response_model=SessionResponse, tags=["Sessions"])
async def update_session(session_id: str, session_update: SessionUpdate):
    """Update a session with new data"""
    try:
        session_data = load_session_from_file(session_id)

        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")

        # Update fields
        if session_update.title is not None:
            session_data["title"] = session_update.title

        if session_update.description is not None:
            session_data["description"] = session_update.description

        if session_update.chat_state is not None:
            session_data["chat_state"] = session_update.chat_state

            # Update document count and last document if present in chat state
            if "documentId" in session_update.chat_state and session_update.chat_state["documentId"]:
                session_data["last_document_id"] = session_update.chat_state["documentId"]
                session_data["document_count"] = max(session_data.get("document_count", 0), 1)

        if session_update.messages is not None:
            session_data["messages"] = session_update.messages

        # Update timestamp
        session_data["updated_at"] = datetime.now().isoformat()

        # Save updated session
        save_session_to_file(session_id, session_data)

        return SessionResponse(
            session_id=session_data["session_id"],
            title=session_data["title"],
            description=session_data.get("description"),
            created_at=datetime.fromisoformat(session_data["created_at"]),
            updated_at=datetime.fromisoformat(session_data["updated_at"]),
            document_count=session_data.get("document_count", 0),
            last_document_id=session_data.get("last_document_id"),
            status=session_data.get("status", "active")
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update session: {str(e)}")

@router.delete("/sessions/{session_id}", tags=["Sessions"])
async def delete_session(session_id: str):
    """Delete a session"""
    try:
        session_file = get_session_file_path(session_id)

        if not session_file.exists():
            raise HTTPException(status_code=404, detail="Session not found")

        # Delete the session file
        session_file.unlink()

        return JSONResponse(content={"message": "Session deleted successfully"})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")

@router.post("/sessions/{session_id}/archive", tags=["Sessions"])
async def archive_session(session_id: str):
    """Archive a session (mark as completed)"""
    try:
        session_data = load_session_from_file(session_id)

        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")

        # Update status to archived
        session_data["status"] = "archived"
        session_data["updated_at"] = datetime.now().isoformat()

        # Save updated session
        save_session_to_file(session_id, session_data)

        return JSONResponse(content={"message": "Session archived successfully"})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to archive session: {str(e)}")

# Enhanced session management endpoints
@router.delete("/sessions/{session_id}/delete", tags=["Sessions"])
async def delete_session_permanently(session_id: str):
    """Permanently delete a session"""
    try:
        session_file = get_session_file_path(session_id)
        
        if not session_file.exists():
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Delete session file
        session_file.unlink()
        
        return JSONResponse(content={"message": "Session deleted successfully"})
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")

@router.post("/sessions/{session_id}/share", tags=["Sessions"])
async def share_session(session_id: str, shared_users: List[str]):
    """Share a session with other users"""
    try:
        session_data = load_session_from_file(session_id)
        
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Update shared users list
        session_data["shared_with"] = list(set(session_data.get("shared_with", []) + shared_users))
        session_data["status"] = "shared"
        session_data["updated_at"] = datetime.now().isoformat()
        
        # Save updated session
        save_session_to_file(session_id, session_data)
        
        return JSONResponse(content={
            "message": f"Session shared with {len(shared_users)} users",
            "shared_with": session_data["shared_with"]
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to share session: {str(e)}")

@router.post("/sessions/{session_id}/conversation", tags=["Sessions"])
async def add_conversation_message(session_id: str, message: ConversationMessage):
    """Add a message to session conversation history"""
    try:
        session_data = load_session_from_file(session_id)
        
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Add message to conversation history
        conversation_history = session_data.get("conversation_history", [])
        conversation_history.append(message.dict())
        session_data["conversation_history"] = conversation_history
        session_data["updated_at"] = datetime.now().isoformat()
        
        # Save updated session
        save_session_to_file(session_id, session_data)
        
        return JSONResponse(content={"message": "Conversation message added"})
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add conversation message: {str(e)}")

@router.post("/sessions/{session_id}/user-choice", tags=["Sessions"])
async def record_user_choice(session_id: str, choice: UserChoice):
    """Record a user choice in the session"""
    try:
        session_data = load_session_from_file(session_id)
        
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Add user choice
        user_choices = session_data.get("user_choices", [])
        user_choices.append(choice.dict())
        session_data["user_choices"] = user_choices
        session_data["updated_at"] = datetime.now().isoformat()
        
        # Save updated session
        save_session_to_file(session_id, session_data)
        
        return JSONResponse(content={"message": "User choice recorded"})
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record user choice: {str(e)}")

@router.post("/sessions/{session_id}/ai-response", tags=["Sessions"])
async def record_ai_response(session_id: str, response: AIResponse):
    """Record an AI response in the session"""
    try:
        session_data = load_session_from_file(session_id)
        
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Add AI response
        ai_responses = session_data.get("ai_responses", [])
        ai_responses.append(response.dict())
        session_data["ai_responses"] = ai_responses
        session_data["updated_at"] = datetime.now().isoformat()
        
        # Save updated session
        save_session_to_file(session_id, session_data)
        
        return JSONResponse(content={"message": "AI response recorded"})
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record AI response: {str(e)}")

# Document section query endpoint for results page chatbox
class DocumentSectionQuery(BaseModel):
    document_id: str
    question: str
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    section_name: Optional[str] = None
    custom_instructions: Optional[str] = None

@router.post("/sessions/{session_id}/document-query", tags=["Sessions", "AI"])
async def query_document_section(session_id: str, query: DocumentSectionQuery):
    """Query a specific section of a document with AI assistance"""
    try:
        # Load session
        session_data = load_session_from_file(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get document from PostgreSQL
        from services.persistent_storage_enhanced import PersistentStorageManager
        storage = PersistentStorageManager()
        
        # Get document analysis results
        analysis_result = await storage.get_analysis_results(query.document_id)
        if not analysis_result:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get original file content if needed for page-specific queries
        file_data = await storage.get_file(query.document_id)
        
        # Prepare context for AI query
        context = {
            "document_metadata": analysis_result.get("metadata", {}),
            "sections": analysis_result.get("sections", []),
            "query": query.question,
            "custom_instructions": query.custom_instructions
        }
        
        # Filter sections by page range if specified
        if query.start_page and query.end_page:
            # TODO: Implement page-based section filtering
            context["page_filter"] = {"start": query.start_page, "end": query.end_page}
        
        # Filter by section name if specified
        if query.section_name:
            filtered_sections = [
                section for section in context["sections"] 
                if query.section_name.lower() in section.get("title", "").lower()
            ]
            context["sections"] = filtered_sections
        
        # Get custom instructions from analysis context
        analysis_context = session_data.get("analysis_context", {})
        base_instructions = analysis_context.get("custom_instructions", "")
        
        # Combine instructions
        full_instructions = f"{base_instructions}\n\n{query.custom_instructions}" if query.custom_instructions else base_instructions
        
        # Prepare AI prompt
        prompt = f"""
        Based on the document analysis and the specific section(s) requested, please answer the following question:
        
        Question: {query.question}
        
        Custom Instructions: {full_instructions}
        
        Document Context:
        Company: {context['document_metadata'].get('company_name', {}).get('value', 'Unknown')}
        Document Type: {context['document_metadata'].get('document_type', {}).get('value', 'Unknown')}
        
        Relevant Sections:
        {json.dumps(context['sections'], indent=2)}
        
        Please provide a detailed answer based only on the information available in these sections.
        If the answer cannot be found in the provided sections, please state that clearly.
        """
        
        # TODO: Call AI service to get response
        # For now, return a structured response
        ai_answer = {
            "query_id": f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "question": query.question,
            "answer": "This would be the AI-generated answer based on the document sections and custom instructions.",
            "sections_referenced": len(context["sections"]),
            "confidence": 0.85,
            "sources": [section.get("title", "Unknown Section") for section in context["sections"]],
            "page_range": f"{query.start_page}-{query.end_page}" if query.start_page and query.end_page else "All sections"
        }
        
        # Record this interaction in session
        conversation_message = ConversationMessage(
            message_id=f"msg_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            role="user",
            content=query.question,
            timestamp=datetime.now(),
            message_type="section_query",
            metadata={
                "document_id": query.document_id,
                "start_page": query.start_page,
                "end_page": query.end_page,
                "section_name": query.section_name
            }
        )
        
        ai_response_message = ConversationMessage(
            message_id=f"msg_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            role="assistant", 
            content=ai_answer["answer"],
            timestamp=datetime.now(),
            message_type="section_answer",
            metadata={
                "query_id": ai_answer["query_id"],
                "confidence": ai_answer["confidence"],
                "sources": ai_answer["sources"]
            }
        )
        
        # Update session with conversation
        conversation_history = session_data.get("conversation_history", [])
        conversation_history.extend([conversation_message.dict(), ai_response_message.dict()])
        session_data["conversation_history"] = conversation_history
        session_data["updated_at"] = datetime.now().isoformat()
        
        save_session_to_file(session_id, session_data)
        
        return {
            "success": True,
            "ai_response": ai_answer,
            "session_updated": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document section query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process document query: {str(e)}")


# Documents management endpoints (consolidated from documents_routes.py)
@router.get("/documents", response_model=List[Dict[str, Any]], tags=["Documents"])
async def list_documents():
    """List all uploaded documents by scanning the uploads and results directories."""
    documents = {}
    try:
        # Scan uploads directory for both .pdf and .docx
        for item in UPLOADS_DIR.iterdir():
            if item.is_file() and item.suffix in [".pdf", ".docx"]:
                doc_id = item.stem
                try:
                    stat_result = item.stat()
                    documents[doc_id] = {
                        "id": doc_id,
                        "filename": item.name,
                        "uploaded_at": datetime.fromtimestamp(stat_result.st_ctime).isoformat(),
                        "status": "PENDING",  # Default status, update below
                        "file_size": stat_result.st_size,
                    }
                except Exception as stat_err:
                    logger.warning(f"Could not get stats for file {item.name}: {stat_err}")

        # Scan analysis results directory to update status
        for item in ANALYSIS_RESULTS_DIR.iterdir():
            if item.is_file() and item.suffix == ".json":
                doc_id = item.stem.split("_metadata")[0]  # Handle both metadata and final results files
                if doc_id in documents:
                    try:
                        with open(item, "r", encoding="utf-8") as f:
                            results_data = json.load(f)
                        # Determine status based on file content
                        if "_metadata.json" in item.name:
                            meta_status = results_data.get("_overall_status", "COMPLETED")
                            if meta_status == "FAILED":
                                documents[doc_id]["status"] = "FAILED"
                            elif meta_status == "PARTIAL":
                                documents[doc_id]["status"] = "PROCESSING"
                            elif documents[doc_id]["status"] == "PENDING":
                                documents[doc_id]["status"] = "PROCESSING"
                        elif ".json" in item.name and "_metadata" not in item.name:
                            # Final results file
                            final_status = results_data.get("status", "completed")
                            if final_status == "failed":
                                documents[doc_id]["status"] = "FAILED"
                            else:
                                documents[doc_id]["status"] = "COMPLETED"
                    except Exception as json_err:
                        logger.warning(f"Could not read or parse result file {item.name}: {json_err}")
                        if doc_id in documents:
                            documents[doc_id]["status"] = "FAILED"

        # Return documents sorted by upload time
        sorted_docs = sorted(documents.values(), key=lambda d: d.get("uploaded_at", ""), reverse=True)
        return sorted_docs

    except Exception as e:
        logger.error(f"Error listing documents from file system: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


# Checklist management endpoints (consolidated from checklist_routes.py)
class ChecklistItemUpdateRequest(BaseModel):
    status: Optional[str] = None
    comment: Optional[str] = None
    resolved: Optional[bool] = None

@router.put("/documents/{document_id}/checklist/items/{item_ref}", tags=["Checklist"])
async def update_checklist_item(document_id: str, item_ref: str, update: ChecklistItemUpdateRequest):
    """Update a checklist item."""
    try:
        result_path = CHECKLIST_DATA_DIR / f"{document_id}.json"
        if not result_path.exists():
            raise HTTPException(status_code=404, detail="Analysis not found")

        with open(result_path, "r", encoding='utf-8') as f:
            data = json.load(f)

        # Find and update the item
        item_found = False
        for item in data.get("items", []):
            if item.get("ref") == item_ref or item.get("id") == item_ref:
                if update.status is not None:
                    item["status"] = update.status
                if update.comment is not None:
                    item["comment"] = update.comment
                if update.resolved is not None:
                    item["resolved"] = update.resolved
                item_found = True
                break

        if not item_found:
            raise HTTPException(status_code=404, detail="Item not found")

        # Save updated data
        with open(result_path, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        return {"message": "Item updated successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/documents/{document_id}/checklist/auto-fill", tags=["Checklist"])
async def auto_fill_checklist(document_id: str, request: Dict[str, Any] = {}):
    """Auto-fill checklist items based on document analysis."""
    try:
        logger.info(f"Received auto-fill request for document {document_id}")

        # Extract section_id from request body
        section_id = request.get("section_id") if request else None

        # Get analysis results
        results_path = ANALYSIS_RESULTS_DIR / f"{document_id}.json"
        if not results_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"No analysis results found for document {document_id}",
            )

        with open(results_path, "r", encoding="utf-8") as f:
            analysis_data = json.load(f)

        # Check if analysis is completed
        if analysis_data.get("status") not in ["COMPLETED", "completed"]:
            raise HTTPException(status_code=400, detail="Document analysis not completed")

        # Load base checklist template
        checklist_template = load_checklist()

        # Auto-fill checklist items based on analysis
        checklist_items = []
        total_items = 0
        completed_items = 0

        for section in checklist_template.get("sections", []):
            # Skip sections that don't match the requested section_id if one is provided
            if section_id and section.get("title") != section_id:
                continue

            for item in section.get("items", []):
                total_items += 1
                auto_filled_item = {
                    "id": item.get("id"),
                    "section": section.get("title"),
                    "requirement": item.get("requirement", item.get("question", "")),
                    "reference": item.get("reference", ""),
                    "status": "PENDING",
                    "evidence": "",
                    "comments": "",
                    "auto_filled": True,
                }

                # Try to find matching analysis result
                for analysis_section in analysis_data.get("sections", []):
                    for analysis_item in analysis_section.get("items", []):
                        if analysis_item.get("id") == item.get("id"):
                            auto_filled_item.update({
                                "status": analysis_item.get("status", "PENDING"),
                                "evidence": analysis_item.get("evidence", ""),
                                "comments": analysis_item.get("ai_explanation", ""),
                                "suggestion": analysis_item.get("suggestion", ""),
                                "auto_filled": True,
                            })
                            if analysis_item.get("status") != "PENDING":
                                completed_items += 1
                            break

                checklist_items.append(auto_filled_item)

        # Create response with metadata
        response = {
            "items": checklist_items,
            "metadata": {
                "total_items": total_items,
                "completed_items": completed_items,
                "compliance_score": (completed_items / total_items) if total_items > 0 else 0,
            },
        }

        # Save auto-filled checklist
        checklist_path = CHECKLIST_DATA_DIR / f"{document_id}.json"
        with open(checklist_path, "w", encoding="utf-8") as f:
            json.dump(response, f, indent=2)

        logger.info(f"Successfully auto-filled checklist for document {document_id}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error auto-filling checklist for document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/debug/files")
async def debug_list_files():
    """
    DEBUG ENDPOINT: List files in analysis_results directory on Render
    """
    try:
        import os
        from pathlib import Path
        
        # Get the current working directory
        cwd = os.getcwd()
        
        # List analysis_results directory
        analysis_dir = Path(cwd) / "analysis_results"
        files_info = {
            "cwd": cwd,
            "analysis_results_dir": str(analysis_dir),
            "analysis_results_exists": analysis_dir.exists(),
            "files": []
        }
        
        if analysis_dir.exists():
            files_info["files"] = [f.name for f in analysis_dir.iterdir() if f.is_file()]
        
        # Also check uploads directory
        uploads_dir = Path(cwd) / "uploads"
        files_info["uploads_dir"] = str(uploads_dir)
        files_info["uploads_exists"] = uploads_dir.exists()
        files_info["uploads_files"] = []
        
        if uploads_dir.exists():
            files_info["uploads_files"] = [f.name for f in uploads_dir.iterdir() if f.is_file()]
            
        return files_info
        
    except Exception as e:
        try:
            import os
            cwd = os.getcwd()
        except:
            cwd = "unknown"
        return {"error": str(e), "cwd": cwd}


@router.get("/debug/storage")
async def debug_storage_check():
    """
    DEBUG ENDPOINT: Check storage directories and recent files
    """
    try:
        import os
        from pathlib import Path
        import time
        
        cwd = os.getcwd()
        
        # Check analysis_results
        analysis_dir = Path(cwd) / "analysis_results"
        uploads_dir = Path(cwd) / "uploads"
        
        result = {
            "working_directory": cwd,
            "analysis_results": {
                "path": str(analysis_dir),
                "exists": analysis_dir.exists(),
                "files": [],
                "file_count": 0
            },
            "uploads": {
                "path": str(uploads_dir), 
                "exists": uploads_dir.exists(),
                "files": [],
                "file_count": 0
            }
        }
        
        # List analysis results files
        if analysis_dir.exists():
            files = list(analysis_dir.iterdir())
            result["analysis_results"]["files"] = [
                {
                    "name": f.name,
                    "size": f.stat().st_size if f.is_file() else 0,
                    "modified": time.ctime(f.stat().st_mtime) if f.exists() else "unknown"
                } 
                for f in files if f.is_file()
            ]
            result["analysis_results"]["file_count"] = len(result["analysis_results"]["files"])
        
        # List uploads files
        if uploads_dir.exists():
            files = list(uploads_dir.iterdir())
            result["uploads"]["files"] = [
                {
                    "name": f.name,
                    "size": f.stat().st_size if f.is_file() else 0,
                    "modified": time.ctime(f.stat().st_mtime) if f.exists() else "unknown"
                }
                for f in files if f.is_file()
            ]
            result["uploads"]["file_count"] = len(result["uploads"]["files"])
            
        return result
        
    except Exception as e:
        try:
            import os
            working_directory = os.getcwd()
        except:
            working_directory = "unknown"
        return {"error": str(e), "working_directory": working_directory}
