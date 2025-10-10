"""
Basic document chunker for processing documents into chunks.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def process_document_chunks(document_id: str, file_path: Path) -> List[Dict[str, Any]]:
    """Process a document into basic chunks."""
    logger.info(f"🔄 Processing document {document_id} into chunks")
    
    try:
        # Simple implementation - just return a basic chunk
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # For now, create a single chunk with the document
        chunks = [{
            "id": f"{document_id}_chunk_1",
            "text": "Document content processed",
            "chunk_number": 1,
            "document_id": document_id,
            "metadata": {
                "length": len(content),
                "type": "document_chunk"
            }
        }]
        
        logger.info(f"✅ Created {len(chunks)} chunks for document {document_id}")
        return chunks
        
    except Exception as e:
        logger.error(f"❌ Failed to process document {document_id}: {str(e)}")
        raise
