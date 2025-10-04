"""
Chunk Quality Diagnostic Script
Investigates why questions are getting "No" answers when financial statements are present
"""

import sqlite3
import sys
import os
from pathlib import Path

# Add backend services to path
backend_path = Path(__file__).parent / "services"
sys.path.append(str(backend_path))

from services.intelligent_chunk_accumulator import CategoryAwareContentStorage, IntelligentChunkAccumulator

def diagnose_chunk_quality():
    """Diagnose chunk quality issues"""
    
    print("🔍 CHUNK QUALITY DIAGNOSTIC")
    print("=" * 60)
    
    # Initialize storage
    try:
        storage = CategoryAwareContentStorage()
        accumulator = IntelligentChunkAccumulator(storage)
        print("✅ Storage and accumulator initialized")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    # Check database contents
    print("\n📊 DATABASE ANALYSIS:")
    print("-" * 30)
    
    try:
        doc_summary = storage.debug_all_documents()
        print(f"Total documents in database: {len(doc_summary)}")
        
        for doc_id, chunk_count in doc_summary.items():
            print(f"  📄 Document '{doc_id}': {chunk_count} chunks")
            
            # Get document summary
            summary = storage.get_document_summary(doc_id)
            print(f"    Categories: {summary['total_categories']}")
            print(f"    Avg confidence: {summary['avg_confidence']:.3f}")
            print(f"    Category distribution: {summary['category_distribution']}")
            
    except Exception as e:
        print(f"❌ Database analysis failed: {e}")
    
    # Test financial statement question
    print("\n🏦 FINANCIAL STATEMENT QUESTION TEST:")
    print("-" * 40)
    
    test_question = "Are the financial statements identified clearly (using an unambiguous title) and distinguished from other information in the same published document?"
    
    # Get all available documents
    try:
        doc_summary = storage.debug_all_documents()
        
        if not doc_summary:
            print("❌ No documents found in database")
            return
            
        # Test with the first available document
        test_doc_id = list(doc_summary.keys())[0]
        print(f"🎯 Testing with document: {test_doc_id}")
        
        # Extract keywords
        keywords = accumulator.extract_question_keywords(test_question)
        print(f"🔍 Extracted keywords: {keywords}")
        
        # Search for relevant content
        relevant_chunks = storage.search_relevant_content(
            document_id=test_doc_id,
            keywords=keywords,
            max_chunks=3  # Get top 3 for analysis
        )
        
        print(f"📄 Found {len(relevant_chunks)} relevant chunks:")
        
        for i, chunk in enumerate(relevant_chunks):
            print(f"\n--- CHUNK {i+1} ---")
            print(f"Category: {chunk['category']}")
            print(f"Subcategory: {chunk.get('subcategory', 'N/A')}")
            print(f"Confidence: {chunk['confidence']:.3f}")
            print(f"Keywords: {chunk.get('keywords', [])}")
            print(f"Length: {len(chunk['content'])} chars")
            print(f"Content preview: {chunk['content'][:200]}...")
            
            # Check if this chunk contains financial statement titles
            content_lower = chunk['content'].lower()
            financial_indicators = [
                'consolidated statement of financial position',
                'balance sheet',
                'statement of profit or loss',
                'statement of comprehensive income',
                'statement of changes in equity',
                'statement of cash flows',
                'cash flow statement'
            ]
            
            found_indicators = [ind for ind in financial_indicators if ind in content_lower]
            if found_indicators:
                print(f"🎯 FINANCIAL INDICATORS FOUND: {found_indicators}")
            else:
                print("❌ NO FINANCIAL INDICATORS in this chunk")
        
        # Test accumulation
        print(f"\n🎯 CONTENT ACCUMULATION TEST:")
        print("-" * 30)
        
        accumulated = accumulator.accumulate_relevant_content(
            question=test_question,
            document_id=test_doc_id
        )
        
        print(f"Total chunks used: {accumulated['total_chunks']}")
        print(f"Confidence: {accumulated['confidence']:.3f}")
        print(f"Categories: {accumulated['categories']}")
        print(f"Content length: {len(accumulated['content'])} chars")
        print(f"Content preview: {accumulated['content'][:300]}...")
        
        # Check if accumulated content has financial statement info
        content_lower = accumulated['content'].lower()
        financial_indicators = [
            'consolidated statement of financial position',
            'balance sheet', 
            'statement of profit or loss',
            'statement of comprehensive income',
            'statement of changes in equity',
            'statement of cash flows'
        ]
        
        found_indicators = [ind for ind in financial_indicators if ind in content_lower]
        if found_indicators:
            print(f"✅ ACCUMULATED CONTENT HAS FINANCIAL INDICATORS: {found_indicators}")
        else:
            print("❌ ACCUMULATED CONTENT MISSING FINANCIAL INDICATORS")
            print("🔍 This explains why questions get 'No' answers!")
            
    except Exception as e:
        print(f"❌ Question test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    diagnose_chunk_quality()