#!/usr/bin/env python3
"""
Quick integration test for smart categorization system
Tests the complete workflow from document processing to question answering
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.smart_document_integration import (
    CompleteDocumentProcessor,
    CategoryAwareContentStorage,
    IntelligentChunkAccumulator
)
from services.ai import AIService

def test_smart_integration():
    """Test the complete smart categorization workflow"""
    print("🧪 Testing Smart Categorization Integration")
    print("=" * 50)
    
    # Test 1: Document Processor Initialization
    try:
        processor = CompleteDocumentProcessor()
        print("✅ CompleteDocumentProcessor initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize CompleteDocumentProcessor: {e}")
        return False
    
    # Test 2: Content Storage Initialization  
    try:
        storage = CategoryAwareContentStorage()
        print("✅ CategoryAwareContentStorage initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize CategoryAwareContentStorage: {e}")
        return False
    
    # Test 3: Chunk Accumulator Initialization
    try:
        accumulator = IntelligentChunkAccumulator(storage)
        print("✅ IntelligentChunkAccumulator initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize IntelligentChunkAccumulator: {e}")
        return False
    
    # Test 4: AI Service Integration
    try:
        ai_service = AIService()
        print("✅ AIService initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize AIService: {e}")
        return False
    
    # Test 5: Test Sample Content Processing
    try:
        sample_content = """
        BALANCE SHEET
        Assets: $1,000,000
        Liabilities: $600,000  
        Equity: $400,000
        
        INCOME STATEMENT
        Revenue: $500,000
        Expenses: $300,000
        Net Income: $200,000
        """
        
        document_id = "test_doc_001"
        
        # Simulate document processing
        categorized_content = processor.process_complete_document(
            content=sample_content,
            document_id=document_id
        )
        
        print(f"✅ Document processing completed: {len(categorized_content)} categories")
        
        # Store categorized content
        for category, content_list in categorized_content.items():
            for content_item in content_list:
                storage.store_categorized_content(
                    document_id=document_id,
                    category=category,
                    content=content_item["content"],
                    context=content_item.get("context", ""),
                    page_number=content_item.get("page_number", 1),
                    chunk_index=content_item.get("chunk_index", 0)
                )
        
        print("✅ Content stored in CategoryAwareContentStorage")
        
        # Test question processing
        test_question = {
            "question": "What is the total amount of assets?",
            "requirement": "Balance sheet disclosure"
        }
        
        relevant_content = accumulator.accumulate_relevant_content(
            document_id=document_id,
            question_dict=test_question
        )
        
        print(f"✅ Question processing completed: {len(relevant_content)} relevant chunks found")
        
        if relevant_content:
            print(f"📄 Sample relevant content: {relevant_content[0]['content'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Content processing test failed: {e}")
        return False

def test_fallback_mechanism():
    """Test that fallback to vector search works"""
    print("\n🔄 Testing Fallback Mechanism")
    print("=" * 30)
    
    try:
        ai_service = AIService()
        
        # Test fallback with sample data
        result = ai_service._fallback_vector_search(
            document_id="test_fallback",
            question_dict={"question": "Test question"}
        )
        
        if "status" in result and "confidence" in result:
            print("✅ Fallback mechanism working correctly")
            print(f"📊 Fallback result status: {result['status']}")
            return True
        else:
            print("❌ Fallback mechanism returned invalid result")
            return False
            
    except Exception as e:
        print(f"❌ Fallback test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Smart Categorization Integration Tests")
    print("=" * 60)
    
    # Run tests
    integration_success = test_smart_integration()
    fallback_success = test_fallback_mechanism()
    
    print("\n📋 Test Summary")
    print("=" * 20)
    print(f"Integration Test: {'✅ PASS' if integration_success else '❌ FAIL'}")
    print(f"Fallback Test: {'✅ PASS' if fallback_success else '❌ FAIL'}")
    
    if integration_success and fallback_success:
        print("\n🎉 All tests passed! Smart categorization integration is ready.")
        print("💡 You can now upload documents and they will be processed with smart categorization.")
        print("🔄 If smart processing fails, the system will automatically fallback to vector search.")
    else:
        print("\n⚠️ Some tests failed. Please check the error messages above.")
        print("🔧 The system will still work with vector search fallback.")
    
    print("\n📚 Next Steps:")
    print("1. Start the server: python main.py")
    print("2. Upload a document via POST /api/v1/analysis/upload")
    print("3. Check document status via GET /api/v1/analysis/documents/{id}")
    print("4. Ask questions via POST /api/v1/analysis/analyze-chunk")