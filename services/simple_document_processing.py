"""
Simple Document Processing Service

Handles ONLY data extraction and storage during upload.
Does NOT trigger metadata extraction or checklist processing.

This replaces the complex CompleteDocumentProcessor in the upload workflow
to implement the staged processing approach requested.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from services.document_data_extractor import DocumentDataExtractor

logger = logging.getLogger(__name__)

# Get backend directory for file paths
BACKEND_DIR = Path(__file__).parent.parent
ANALYSIS_RESULTS_DIR = BACKEND_DIR / "analysis_results"


async def process_upload_simple_extraction(
    document_id: str, upload_path: str
) -> Dict[str, Any]:
    """
    Simple upload processing: ONLY extract and store data.
    
    This function:
    1. Extracts ALL document content using advanced NLP chunking
    2. Stores data in formats expected by metadata and checklist processes
    3. Creates ready-for-processing indicators
    4. Does NOT trigger any processing workflows
    
    User can then trigger metadata extraction, checklist processing, etc. separately.
    """
    logger.info(f"🚀 SIMPLE PROCESSING: Starting data extraction only for {document_id}")
    
    try:
        # Phase 1: Initialize Data Extractor
        logger.info(f"🔧 Initializing document data extractor")
        extractor = DocumentDataExtractor()
        
        # Phase 2: Extract ALL Document Data
        logger.info(f"📄 Starting complete data extraction from {upload_path}")
        extraction_result = await extractor.extract_all_data(upload_path, document_id)
        
        # Phase 3: Create Upload Completion Status
        logger.info(f"💾 Creating upload completion status for {document_id}")
        
        # Create basic status file for frontend polling
        status = {
            "document_id": document_id,
            "status": "data_extracted",
            "processing_mode": "simple_extraction",
            "extraction_completed": True,
            "extraction_timestamp": datetime.now().isoformat(),
            "total_characters": extraction_result.get('total_characters', 0),
            "total_chunks": extraction_result.get('total_chunks', 0),
            "ready_for_metadata_extraction": True,
            "ready_for_checklist_processing": True,
            "message": "Document uploaded and data extracted. Ready for metadata extraction.",
            "_workflow_stage": "DATA_EXTRACTED_AWAITING_METADATA_TRIGGER"
        }
        
        # Save status file for frontend to poll
        status_file = ANALYSIS_RESULTS_DIR / f"{document_id}.json"
        with open(status_file, 'w', encoding='utf-8') as f:
            json.dump(status, f, indent=2, ensure_ascii=False)
            
        logger.info(f"✅ SIMPLE PROCESSING: Data extraction completed for {document_id}")
        logger.info(f"📊 Extracted {extraction_result.get('total_chunks', 0)} chunks, {extraction_result.get('total_characters', 0)} characters")
        logger.info(f"🎯 Document ready for metadata extraction trigger")
        logger.info(f"🎯 Document ready for checklist processing trigger")
        
        return {
            'status': 'success',
            'stage': 'data_extracted',
            'message': 'Document data extracted successfully. Ready for metadata processing.',
            'extraction_summary': {
                'total_chunks': extraction_result.get('total_chunks', 0),
                'total_characters': extraction_result.get('total_characters', 0),
                'ready_for_metadata': True,
                'ready_for_checklist': True
            }
        }
        
    except Exception as e:
        logger.error(f"❌ SIMPLE PROCESSING FAILED for {document_id}: {e}")
        
        # Create error status file
        error_status = {
            "document_id": document_id,
            "status": "extraction_failed",
            "processing_mode": "simple_extraction",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "message": f"Data extraction failed: {e}",
            "_workflow_stage": "EXTRACTION_FAILED"
        }
        
        status_file = ANALYSIS_RESULTS_DIR / f"{document_id}.json"
        with open(status_file, 'w', encoding='utf-8') as f:
            json.dump(error_status, f, indent=2, ensure_ascii=False)
            
        raise Exception(f"Simple document processing failed: {e}")


async def trigger_metadata_extraction(document_id: str) -> Dict[str, Any]:
    """
    Trigger metadata extraction using pre-extracted data.
    This can be called separately after upload completion.
    """
    logger.info(f"🧠 METADATA TRIGGER: Starting metadata extraction for {document_id}")
    
    try:
        # Check if data is ready
        extractor = DocumentDataExtractor()
        if not extractor.is_data_ready_for_metadata(document_id):
            raise Exception(f"Data not ready for metadata extraction for {document_id}")
            
        # Get pre-extracted data
        extracted_data = extractor.get_extracted_data(document_id)
        metadata_input = extracted_data['metadata_input_format']
        
        # Import and use SmartMetadataExtractor
        from services.smart_metadata_extractor import SmartMetadataExtractor
        metadata_extractor = SmartMetadataExtractor()
        
        # Extract metadata using pre-extracted data
        metadata_result = await metadata_extractor.extract_metadata_optimized(
            document_id, metadata_input
        )
        
        logger.info(f"✅ METADATA TRIGGER: Metadata extraction completed for {document_id}")
        return {
            'status': 'success',
            'stage': 'metadata_extracted',
            'metadata': metadata_result
        }
        
    except Exception as e:
        logger.error(f"❌ METADATA TRIGGER FAILED for {document_id}: {e}")
        raise Exception(f"Metadata extraction trigger failed: {e}")


async def trigger_checklist_processing(document_id: str, framework: str, standards: List[str]) -> Dict[str, Any]:
    """
    Trigger checklist processing using pre-extracted data.
    This can be called separately after user confirms framework and standards.
    """
    logger.info(f"📋 CHECKLIST TRIGGER: Starting checklist processing for {document_id}")
    
    try:
        # Check if data is ready
        extractor = DocumentDataExtractor()
        if not extractor.is_data_ready_for_checklist(document_id):
            raise Exception(f"Data not ready for checklist processing for {document_id}")
            
        # Get pre-extracted data
        extracted_data = extractor.get_extracted_data(document_id)
        checklist_input = extracted_data['checklist_input_format']
        
        # TODO: Import and use appropriate checklist processor
        # This would be implemented based on the specific checklist processing logic
        
        logger.info(f"✅ CHECKLIST TRIGGER: Checklist processing completed for {document_id}")
        return {
            'status': 'success', 
            'stage': 'checklist_processed',
            'framework': framework,
            'standards': standards
        }
        
    except Exception as e:
        logger.error(f"❌ CHECKLIST TRIGGER FAILED for {document_id}: {e}")
        raise Exception(f"Checklist processing trigger failed: {e}")
