#!/usr/bin/env python3
"""
Debug the database contents to see what's actually stored
"""

import sqlite3
import sys
import os

# Add the render-backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def debug_database():
    """Debug what's actually in the categorized_content database"""
    db_path = "categorized_content.db"
    
    print("🔍 DEBUGGING: Categorized content database")
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Check table structure
            cursor.execute("PRAGMA table_info(categorized_content)")
            columns = cursor.fetchall()
            print(f"📋 Table structure: {len(columns)} columns")
            for col in columns:
                print(f"   - {col[1]} ({col[2]})")
            
            # Count total records
            cursor.execute("SELECT COUNT(*) FROM categorized_content")
            total_count = cursor.fetchone()[0]
            print(f"📊 Total records: {total_count}")
            
            if total_count > 0:
                # Show recent records
                cursor.execute("SELECT document_id, category, subcategory, confidence_score, SUBSTR(content_chunk, 1, 100) as content_preview FROM categorized_content ORDER BY created_at DESC LIMIT 10")
                records = cursor.fetchall()
                
                print(f"📄 Recent records:")
                for i, record in enumerate(records):
                    doc_id, category, subcategory, confidence, content_preview = record
                    print(f"   {i+1}. Doc: {doc_id}")
                    print(f"      Category: {category}, Subcategory: {subcategory}")
                    print(f"      Confidence: {confidence}")
                    print(f"      Content: {content_preview}...")
                    print()
                
                # Test search function
                print("🔍 Testing search function...")
                from services.intelligent_chunk_accumulator import get_global_storage
                storage = get_global_storage()
                
                # Get a recent document_id to test with
                recent_doc_id = records[0][0] if records else None
                if recent_doc_id:
                    print(f"Testing search for document: {recent_doc_id}")
                    
                    # Test different search terms
                    test_terms = ["financial", "phoenix", "revenue", "statement"]
                    
                    for term in test_terms:
                        results = storage.search_relevant_content(recent_doc_id, [term], max_chunks=5)
                        print(f"   Search '{term}': {len(results)} results")
                        if results:
                            first_result = results[0]
                            print(f"      Sample: {first_result.get('content', '')[:100]}...")
                
            else:
                print("❌ No records found in database!")
                
    except Exception as e:
        print(f"❌ Error accessing database: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_database()