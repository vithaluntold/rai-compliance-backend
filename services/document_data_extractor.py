"""
Document Data Extractor Service

Extracts ALL document content using advanced NLP chunking for later use by:
1. SmartMetadataExtractor (expects List[Union[str, Dict[str, Any]]] with text content)  
2. Checklist processes (expect structured document content)

This service ONLY extracts and stores data - does NOT trigger processing workflows.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Union
from datetime import datetime

from services.document_chunker import DocumentChunker

logger = logging.getLogger(__name__)

# Get backend directory for file paths
BACKEND_DIR = Path(__file__).parent.parent
UPLOADS_DIR = BACKEND_DIR / "uploads"
EXTRACTED_DATA_DIR = BACKEND_DIR / "exported_dataset"


class DocumentDataExtractor:
    """
    Extracts ALL document content for metadata and checklist processing.
    
    Uses advanced NLP chunking to extract:
    - Full text content for metadata extraction
    - Structured chunks for checklist processing  
    - Page-level metadata for citation tracking
    """
    
    def __init__(self):
        self.chunker = DocumentChunker(min_chunk_length=30)
        
        # Ensure directories exist
        EXTRACTED_DATA_DIR.mkdir(exist_ok=True)
        
    async def extract_all_data(self, pdf_path: str, document_id: str) -> Dict[str, Any]:
        """
        Extract ALL document data for later use by metadata and checklist processes.
        
        Returns:
        - full_text: Complete document text for metadata extraction
        - structured_chunks: Detailed chunks for checklist processing
        - metadata: Document metadata and extraction info
        """
        logger.info(f"🔍 DATA EXTRACTOR: Starting complete data extraction for {document_id}")
        
        try:
            # Phase 1: Advanced NLP Chunking
            logger.info(f"📄 Extracting structured chunks from PDF")
            chunks = self.chunker.chunk_pdf(pdf_path, document_id)
            
            if not chunks:
                raise Exception("Failed to extract any content from PDF")
                
            logger.info(f"✅ Extracted {len(chunks)} structured chunks")
            
            # Phase 2: Generate Full Text for Metadata Extraction
            logger.info(f"📝 Generating full text for metadata processing")
            full_text = ""
            content_chunks = []
            metadata_chunk = None
            
            for chunk in chunks:
                chunk_text = chunk.get('text', '')
                if chunk_text:
                    full_text += chunk_text + "\n"
                    
                    # Separate metadata chunk for special handling
                    if chunk.get('chunk_type') == 'metadata':
                        metadata_chunk = chunk
                    else:
                        content_chunks.append(chunk)
            
            logger.info(f"✅ Generated {len(full_text)} characters of full text")
            
            # Phase 3: Structure Data for Different Consumers
            
            # For SmartMetadataExtractor (expects List[Union[str, Dict[str, Any]]])
            metadata_input = chunks  # Pass structured chunks with text keys
            
            # For checklist processes (expects structured document data)
            checklist_input = {
                'document_id': document_id,
                'full_text': full_text,
                'structured_content': content_chunks,
                'metadata_section': metadata_chunk,
                'total_pages': max([chunk.get('page', 0) for chunk in chunks]) + 1 if chunks else 0,
                'total_chunks': len(chunks)
            }
            
            # Phase 4: Store Extracted Data
            extraction_result = {
                'document_id': document_id,
                'extraction_timestamp': datetime.now().isoformat(),
                'status': 'completed',
                'full_text': full_text,
                'total_characters': len(full_text),
                'total_chunks': len(chunks),
                'metadata_input_format': metadata_input,  # For SmartMetadataExtractor
                'checklist_input_format': checklist_input,  # For checklist processes
                'raw_chunks': chunks  # Original chunk data
            }
            
            # Save to extracted data directory
            data_file = EXTRACTED_DATA_DIR / f"{document_id}_extracted_data.json"
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(extraction_result, f, indent=2, ensure_ascii=False)
                
            logger.info(f"💾 Saved extracted data to {data_file}")
            
            # Phase 5: Create Ready-for-Processing Indicators
            
            # Create indicators that data is ready for metadata extraction
            metadata_ready_file = EXTRACTED_DATA_DIR / f"{document_id}_metadata_ready.json"
            with open(metadata_ready_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'document_id': document_id,
                    'data_ready': True,
                    'extraction_completed': datetime.now().isoformat(),
                    'metadata_input_path': str(data_file),
                    'ready_for_metadata_extraction': True
                }, f, indent=2)
                
            # Create indicators that data is ready for checklist processing  
            checklist_ready_file = EXTRACTED_DATA_DIR / f"{document_id}_checklist_ready.json"
            with open(checklist_ready_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'document_id': document_id,
                    'data_ready': True,
                    'extraction_completed': datetime.now().isoformat(),
                    'checklist_input_path': str(data_file),
                    'ready_for_checklist_processing': True
                }, f, indent=2)
                
            logger.info(f"✅ DATA EXTRACTOR: Complete data extraction finished for {document_id}")
            logger.info(f"📊 Summary: {len(chunks)} chunks, {len(full_text)} characters extracted")
            logger.info(f"🎯 Data ready for metadata extraction and checklist processing")
            
            return extraction_result
            
        except Exception as e:
            logger.error(f"❌ DATA EXTRACTOR FAILED for {document_id}: {e}")
            raise Exception(f"Data extraction failed: {e}")
    
    def get_extracted_data(self, document_id: str) -> Dict[str, Any]:
        """Retrieve previously extracted data for a document."""
        data_file = EXTRACTED_DATA_DIR / f"{document_id}_extracted_data.json"
        
        if not data_file.exists():
            raise Exception(f"No extracted data found for document {document_id}")
            
        with open(data_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def is_data_ready_for_metadata(self, document_id: str) -> bool:
        """Check if data is ready for metadata extraction."""
        ready_file = EXTRACTED_DATA_DIR / f"{document_id}_metadata_ready.json"
        return ready_file.exists()
    
    def is_data_ready_for_checklist(self, document_id: str) -> bool:
        """Check if data is ready for checklist processing.""" 
        ready_file = EXTRACTED_DATA_DIR / f"{document_id}_checklist_ready.json"
        return ready_file.exists()
