#!/usr/bin/env python3
"""
Test the smart categorization fix with Phoenix document
"""

import os
import sys
import logging

# Add the render-backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def test_chunking_and_storage():
    """Test that chunks are properly stored in categorized_content database"""
    logger.info("🚀 TESTING: Smart categorization fix with Phoenix document")
    
    try:
        # Import the functions we need to test
        from routes.analysis_routes import _process_document_chunks, generate_document_id
        from services.intelligent_chunk_accumulator import get_global_storage
        
        # Generate a test document ID
        document_id = generate_document_id()
        logger.info(f"📋 Generated test document ID: {document_id}")
        
        # First, let's check if the Phoenix PDF exists
        phoenix_path = "c:\\Users\\saivi\\OneDrive\\Documents\\Audricc all\\phoenix-financial-statement.pdf"
        if not os.path.exists(phoenix_path):
            logger.error(f"❌ Phoenix document not found at: {phoenix_path}")
            return False
        
        logger.info(f"✅ Found Phoenix document at: {phoenix_path}")
        
        # Copy the Phoenix document to uploads directory for processing
        uploads_dir = os.path.join(os.path.dirname(__file__), "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        
        import shutil
        target_path = os.path.join(uploads_dir, f"{document_id}.pdf")
        shutil.copy2(phoenix_path, target_path)
        logger.info(f"📁 Copied Phoenix document to: {target_path}")
        
        # Test chunk processing
        logger.info("🔧 TESTING: Document chunk processing...")
        chunks = _process_document_chunks(document_id)
        
        if not chunks:
            logger.error("❌ NO CHUNKS GENERATED - this is the problem!")
            return False
        
        logger.info(f"✅ Generated {len(chunks)} chunks successfully")
        
        # Test chunk storage verification
        logger.info("🔍 TESTING: Checking if chunks were stored in database...")
        storage = get_global_storage()
        
        # Query the database to see if chunks were stored - use broader search
        stored_chunks = storage.search_relevant_content(document_id, ["phoenix", "financial"], max_chunks=100)
        
        if not stored_chunks:
            logger.error("❌ NO CHUNKS FOUND IN DATABASE - storage failed!")
            return False
        
        logger.info(f"✅ Found {len(stored_chunks)} chunks in database for document {document_id}")
        
        # Show sample content to verify it's working
        if stored_chunks:
            sample_content = stored_chunks[0].get('content', '')[:150]
            logger.info(f"📄 Sample chunk content: {sample_content}...")
        
        # Test smart categorization query
        logger.info("🧠 TESTING: Smart categorization with real query...")
        test_queries = [
            "earnings per share",
            "financial statements", 
            "revenue",
            "profit and loss"
        ]
        
        for query in test_queries:
            relevant_chunks = storage.search_relevant_content(document_id, [query], max_chunks=5)
            logger.info(f"📊 Query '{query}': Found {len(relevant_chunks)} relevant chunks")
            
            if relevant_chunks:
                # Show first chunk content (truncated)
                first_chunk = relevant_chunks[0]
                content = first_chunk.get('content', '')[:200] + "..." if len(first_chunk.get('content', '')) > 200 else first_chunk.get('content', '')
                logger.info(f"   Sample content: {content}")
        
        logger.info("🎯 SUCCESS: Smart categorization is working correctly!")
        return True
        
    except Exception as e:
        logger.error(f"❌ ERROR during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_chunking_and_storage()
    if success:
        print("\n✅ ALL TESTS PASSED - Smart categorization fix is working!")
    else:
        print("\n❌ TESTS FAILED - There are still issues with smart categorization")
    
    sys.exit(0 if success else 1)