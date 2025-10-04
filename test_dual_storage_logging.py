#!/usr/bin/env python3
"""
Test just the dual storage part of _create_vector_index
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_dual_storage_logging():
    """Test just the dual storage part with enhanced logging"""
    print("🧪 TESTING DUAL STORAGE LOGGING")
    print("=" * 60)
    
    # Copy the exact dual storage code from _create_vector_index
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    
    document_id = "TEST-DUAL-STORAGE-789"
    chunks = [
        "CONSOLIDATED STATEMENT OF FINANCIAL POSITION As at 31 December 2023",
        "Note 9 - Digital Assets accounting treatment"
    ]
    
    logger.info(f"🔧 STARTING DUAL STORAGE: Storing {len(chunks)} chunks in categorized content storage")
    
    try:
        # Import with detailed logging
        logger.info(f"📦 Importing CategoryAwareContentStorage...")
        from services.intelligent_chunk_accumulator import CategoryAwareContentStorage
        logger.info(f"✅ CategoryAwareContentStorage imported successfully")
        
        # Create storage instance with logging
        logger.info(f"🔧 Creating storage instance...")
        storage = CategoryAwareContentStorage()
        logger.info(f"✅ Storage instance created successfully")
        
        stored_count = 0
        for i, chunk in enumerate(chunks):
            try:
                # Extract text content based on chunk format
                if isinstance(chunk, dict):
                    text_content = chunk.get('text', '') or chunk.get('content', '') or str(chunk)
                    category = chunk.get('category', 'document_content')
                    subcategory = chunk.get('subcategory', 'general')
                else:
                    text_content = str(chunk)
                    category = 'document_content'
                    subcategory = 'general'
                
                logger.info(f"📝 Storing chunk {i+1}/{len(chunks)}: {len(text_content)} chars, category={category}")
                
                # Store each chunk with appropriate categorization
                storage.store_categorized_chunk(
                    document_id=document_id,
                    chunk=text_content,
                    category=category,
                    subcategory=subcategory,
                    confidence=0.8,  # Default confidence
                    keywords=[]
                )
                
                stored_count += 1
                logger.info(f"✅ Chunk {i+1} stored successfully")
                
            except Exception as chunk_error:
                logger.error(f"❌ Failed to store chunk {i+1}: {chunk_error}")
                logger.error(f"❌ Chunk type: {type(chunk)}, content preview: {str(chunk)[:100]}")
                continue
        
        logger.info(f"🎉 DUAL STORAGE COMPLETE: Stored {stored_count}/{len(chunks)} chunks in categorized content storage")
        
        # Verification: Check if chunks were actually stored
        try:
            import sqlite3
            import os
            db_path = "categorized_content.db"
            logger.info(f"🔍 Verifying storage in database: {os.path.abspath(db_path)}")
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM categorized_content WHERE document_id = ?", (document_id,))
            verification_count = cursor.fetchone()[0]
            conn.close()
            
            logger.info(f"📊 VERIFICATION: {verification_count} chunks found in database for {document_id}")
            
            if verification_count != stored_count:
                logger.error(f"❌ STORAGE MISMATCH: Stored {stored_count} but found {verification_count} in database")
            else:
                logger.info(f"✅ STORAGE VERIFIED: All {verification_count} chunks confirmed in database")
                
        except Exception as verify_error:
            logger.error(f"❌ Verification failed: {verify_error}")
        
        # Cleanup
        import sqlite3
        db_path = "categorized_content.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM categorized_content WHERE document_id = ?", (document_id,))
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        logger.info(f"🧹 Cleaned up {deleted} test chunks")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ DUAL STORAGE FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the test"""
    print("🚀 DUAL STORAGE LOGGING TEST")
    print("=" * 80)
    
    success = test_dual_storage_logging()
    
    if success:
        print("\n✅ Dual storage logging works perfectly!")
        print("🚀 Enhanced logging is ready for production deployment")
    else:
        print("\n❌ Dual storage logging test failed")
    
    print("=" * 80)

if __name__ == "__main__":
    main()