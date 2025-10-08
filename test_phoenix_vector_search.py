#!/usr/bin/env python3
"""
Test Phoenix Vector Search
Test if vector search can find cash flow content for IAS 7 questions.
"""

import json
import os
import sys
from pathlib import Path

# Add the parent directory to Python path
sys.path.insert(0, os.path.abspath('.'))

def test_phoenix_vector_search():
    """Test vector search for Phoenix document with IAS 7 cash flow questions."""
    
    document_id = "RAI-02102025-E1JZP-HV77U"
    
    # Test cash flow related queries
    test_queries = [
        "cash flows from operating activities",
        "net cash used in operating activities", 
        "statement of cash flows",
        "investing activities cash flows",
        "financing activities cash flows",
        "IAS 7 cash flow statement classification",
        "operating investing financing activities"
    ]
    
    print(f"🔍 Testing vector search for document: {document_id}")
    
    # Check if index files exist
    vector_indices_dir = Path("vector_indices")
    faiss_file = vector_indices_dir / f"{document_id}_index.faiss"
    chunks_file = vector_indices_dir / f"{document_id}_chunks.json"
    
    if not faiss_file.exists() or not chunks_file.exists():
        print(f"❌ Vector index files not found!")
        print(f"   - FAISS file exists: {faiss_file.exists()}")
        print(f"   - Chunks file exists: {chunks_file.exists()}")
        return False
    
    print(f"✅ Vector index files found")
    
    # Load chunks directly to test content
    print(f"\n📊 Loading chunks to verify cash flow content...")
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"📝 Total chunks: {len(chunks)}")
    
    # Search for cash flow content in chunks
    cash_flow_chunks = []
    for i, chunk in enumerate(chunks):
        content = chunk.get('content', '').lower()
        if any(keyword in content for keyword in ['cash flows', 'operating activities', 'investing activities', 'financing activities']):
            cash_flow_chunks.append((i, chunk))
    
    print(f"💰 Found {len(cash_flow_chunks)} chunks with cash flow content")
    
    # Display cash flow chunks
    for i, (chunk_idx, chunk) in enumerate(cash_flow_chunks):
        content = chunk.get('content', '')
        preview = content[:200] + "..." if len(content) > 200 else content
        print(f"\n--- Cash Flow Chunk {i+1} (Index {chunk_idx}) ---")
        print(preview)
        
        # Look for specific cash flow numbers
        if 'net cash' in content.lower():
            lines = content.split('\n')
            for line in lines:
                if 'net cash' in line.lower() and any(char.isdigit() for char in line):
                    print(f"🔢 Found cash flow figure: {line.strip()}")
    
    # Now test with actual vector store if possible
    try:
        print(f"\n🔧 Testing actual vector store search...")
        
        # Try to import and test vector store
        from services.vector_store import VectorStore
        from services import get_vector_store
        
        # This will fail if Azure OpenAI is not configured, but let's try
        try:
            vs = get_vector_store()
            print(f"✅ Vector store initialized")
            
            # Test index_exists
            exists = vs.index_exists(document_id)
            print(f"📋 index_exists({document_id}): {exists}")
            
            if exists:
                print(f"\n🔍 Testing search queries...")
                for query in test_queries[:3]:  # Test first 3 queries
                    print(f"\nQuery: '{query}'")
                    try:
                        results = vs.search(query=query, document_id=document_id, top_k=2)
                        print(f"   Results: {len(results)} chunks found")
                        
                        for j, result in enumerate(results):
                            score = result.get('score', 0)
                            text_preview = result.get('text', '')[:100] + "..."
                            print(f"   {j+1}. Score: {score:.3f} - {text_preview}")
                            
                    except Exception as e:
                        print(f"   ❌ Search failed: {e}")
                        
            else:
                print(f"❌ Vector store says index doesn't exist!")
                
        except Exception as e:
            print(f"⚠️  Could not test vector store search: {e}")
            print(f"💡 This is expected if Azure OpenAI environment variables are not set")
            
    except ImportError as e:
        print(f"⚠️  Could not import vector store: {e}")
    
    return len(cash_flow_chunks) > 0


if __name__ == "__main__":
    print("🚀 Starting Phoenix Vector Search Test...")
    
    success = test_phoenix_vector_search()
    
    if success:
        print("\n🎉 SUCCESS: Cash flow content found in vector index!")
        print("The vector search should now be able to find relevant cash flow data for IAS 7 questions.")
    else:
        print("\n💥 FAILED: No cash flow content found in vector index")
        sys.exit(1)