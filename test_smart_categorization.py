#!/usr/bin/env python3
"""
Test script to diagnose smart categorization issues
"""

import sys
import os
import logging
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_categorizer_imports():
    """Test if all categorizer dependencies can be imported"""
    print("🔍 Testing categorizer imports...")
    
    try:
        from services.contextual_content_categorizer import ContextualContentCategorizer
        print("✅ ContextualContentCategorizer imported successfully")
    except Exception as e:
        print(f"❌ Failed to import ContextualContentCategorizer: {e}")
        import traceback
        print(traceback.format_exc())
        return False
    
    try:
        from services.intelligent_chunk_accumulator import IntelligentChunkAccumulator, CategoryAwareContentStorage
        print("✅ IntelligentChunkAccumulator imported successfully")
    except Exception as e:
        print(f"❌ Failed to import IntelligentChunkAccumulator: {e}")
        import traceback
        print(traceback.format_exc())
        return False
    
    return True

def test_categorizer_initialization():
    """Test if categorizer can be initialized"""
    print("\n🔧 Testing categorizer initialization...")
    
    try:
        from services.contextual_content_categorizer import ContextualContentCategorizer
        categorizer = ContextualContentCategorizer()
        print("✅ ContextualContentCategorizer initialized successfully")
        return categorizer
    except Exception as e:
        print(f"❌ Failed to initialize ContextualContentCategorizer: {e}")
        import traceback
        print(traceback.format_exc())
        return None

def test_accumulator_initialization():
    """Test if accumulator can be initialized"""
    print("\n🔧 Testing accumulator initialization...")
    
    try:
        from services.intelligent_chunk_accumulator import IntelligentChunkAccumulator, CategoryAwareContentStorage
        storage = CategoryAwareContentStorage()
        accumulator = IntelligentChunkAccumulator(storage)
        print("✅ IntelligentChunkAccumulator initialized successfully")
        return accumulator
    except Exception as e:
        print(f"❌ Failed to initialize IntelligentChunkAccumulator: {e}")
        import traceback
        print(traceback.format_exc())
        return None

def test_categorization():
    """Test the categorization process with sample data"""
    print("\n📝 Testing categorization with sample data...")
    
    categorizer = test_categorizer_initialization()
    if not categorizer:
        return False
    
    # Sample page texts like what document_chunker would provide
    sample_page_texts = [
        {
            'page_num': 1,
            'text': 'FINANCIAL STATEMENTS\n\nTo: The Shareholders of Test Corporation PLC\n\nWe have audited the consolidated financial statements of Test Corporation PLC.',
            'length': 140
        },
        {
            'page_num': 2,
            'text': 'The company is engaged in software development and consulting services.\n\nOperations are conducted primarily in the United States and Canada.',
            'length': 135
        }
    ]
    
    try:
        categorized_content = categorizer.categorize_page_texts(sample_page_texts, None)
        print(f"✅ Categorization successful! Generated {len(categorized_content)} categorized pieces")
        
        if categorized_content:
            print("📋 Sample categorized content:")
            for i, piece in enumerate(categorized_content[:2]):  # Show first 2
                print(f"   {i+1}. Category: {piece.get('category', 'N/A')}, Topic: {piece.get('topic', 'N/A')}")
                print(f"      Content: {piece.get('content', '')[:100]}...")
        else:
            print("⚠️  Categorization returned empty results")
        
        return len(categorized_content) > 0
    except Exception as e:
        print(f"❌ Categorization failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_spacy_model():
    """Test if spaCy model is available"""
    print("\n🧠 Testing spaCy model...")
    
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("Test sentence for spaCy processing.")
        print(f"✅ spaCy model loaded successfully. Processed: '{doc.text}'")
        return True
    except Exception as e:
        print(f"❌ spaCy model failed: {e}")
        return False

def main():
    """Run all diagnostic tests"""
    print("🚀 SMART CATEGORIZATION DIAGNOSTICS")
    print("=" * 60)
    
    # Test 1: Imports
    if not test_categorizer_imports():
        print("\n❌ FATAL: Import failures detected")
        return
    
    # Test 2: spaCy model
    if not test_spacy_model():
        print("\n❌ FATAL: spaCy model issues detected")
        return
    
    # Test 3: Initialization
    categorizer = test_categorizer_initialization()
    accumulator = test_accumulator_initialization()
    
    if not categorizer:
        print("\n❌ FATAL: Categorizer initialization failed")
        return
    
    if not accumulator:
        print("\n❌ FATAL: Accumulator initialization failed")
        return
    
    # Test 4: Categorization process
    if not test_categorization():
        print("\n❌ FATAL: Categorization process failed")
        return
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("✅ Smart categorization pipeline appears to be working locally")
    print("❗ The issue may be deployment-specific or dependency-related on Render")

if __name__ == "__main__":
    main()