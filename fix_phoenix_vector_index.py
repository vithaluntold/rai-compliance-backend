#!/usr/bin/env python3
"""
Fix Phoenix Vector Index
Creates the missing vector index for Phoenix document using existing extracted text.
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the parent directory to Python path so we can import services
sys.path.insert(0, os.path.abspath('.'))

from services.vector_store import get_vector_store


async def create_phoenix_vector_index():
    """Create vector index for Phoenix document using extracted text."""
    
    document_id = "RAI-02102025-E1JZP-HV77U"
    
    # Read the extracted Phoenix text
    phoenix_text_file = "phoenix_extracted_text.txt"
    if not os.path.exists(phoenix_text_file):
        print(f"❌ ERROR: {phoenix_text_file} not found!")
        return False
    
    print(f"📖 Reading Phoenix extracted text from {phoenix_text_file}...")
    with open(phoenix_text_file, 'r', encoding='utf-8') as f:
        full_text = f.read()
    
    print(f"📊 Text length: {len(full_text):,} characters")
    
    # Create vector store instance
    print("🔧 Initializing vector store...")
    vs = get_vector_store()
    
    # Create the vector index
    print(f"🏗️  Creating vector index for document {document_id}...")
    success = await vs.create_index_from_text(document_id, full_text)
    
    if success:
        print(f"✅ Vector index created successfully!")
        
        # Verify index exists
        if vs.index_exists(document_id):
            print(f"✅ Vector index verified - index_exists() returns True")
            
            # Check if .faiss file was created
            faiss_file = Path("vector_indices") / f"{document_id}_index.faiss"
            if faiss_file.exists():
                print(f"✅ FAISS file created: {faiss_file}")
                print(f"📁 File size: {faiss_file.stat().st_size:,} bytes")
            else:
                print(f"❌ ERROR: FAISS file not found at {faiss_file}")
                return False
                
        else:
            print(f"❌ ERROR: index_exists() still returns False")
            return False
            
    else:
        print(f"❌ ERROR: Failed to create vector index")
        return False
    
    return True


if __name__ == "__main__":
    print("🚀 Starting Phoenix Vector Index Fix...")
    
    # Run the async function
    success = asyncio.run(create_phoenix_vector_index())
    
    if success:
        print("\n🎉 SUCCESS: Phoenix vector index created!")
        print("Now the AI service should use Phoenix financial data for IAS 7 compliance questions.")
    else:
        print("\n💥 FAILED: Could not create Phoenix vector index")
        sys.exit(1)