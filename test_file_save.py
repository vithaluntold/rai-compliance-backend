#!/usr/bin/env python3
"""
Test file save functionality directly
"""
import os
import sys
import json
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_file_save():
    """Test saving files to analysis_results directory"""
    
    # Get the backend directory
    BACKEND_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
    ANALYSIS_RESULTS_DIR = BACKEND_DIR / "analysis_results"
    
    print(f"Backend dir: {BACKEND_DIR}")
    print(f"Analysis results dir: {ANALYSIS_RESULTS_DIR}")
    print(f"Directory exists: {ANALYSIS_RESULTS_DIR.exists()}")
    
    # Test document ID
    document_id = "RAI-02102025-E1JZP-HV77U"
    results_path = ANALYSIS_RESULTS_DIR / f"{document_id}.json"
    
    print(f"Target file path: {results_path}")
    
    # Test data
    test_data = {
        "document_id": document_id,
        "status": "COMPLETED",
        "metadata_extraction": "COMPLETED",
        "metadata": {
            "company_name": "Phoenix Group PLC",
            "nature_of_business": "Cryptocurrency mining",
            "operational_demographics": "United Arab Emirates",
            "financial_statements_type": "Consolidated"
        }
    }
    
    # Try to save
    try:
        print("Attempting to save file...")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
        
        print("✅ File save successful!")
        
        # Check if file exists
        if results_path.exists():
            print("✅ File exists after save")
            
            # Read it back
            with open(results_path, 'r', encoding='utf-8') as f:
                read_data = json.load(f)
            
            print("✅ File read successful")
            print(f"Data: {json.dumps(read_data, indent=2)}")
            
        else:
            print("❌ File does not exist after save!")
            
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        import traceback
        print(traceback.format_exc())
        
    # Also test completion flag
    flag_path = ANALYSIS_RESULTS_DIR / f"{document_id}.metadata_completed"
    try:
        print(f"\nTesting completion flag: {flag_path}")
        flag_path.touch()
        if flag_path.exists():
            print("✅ Completion flag created successfully")
        else:
            print("❌ Completion flag not found after creation")
    except Exception as e:
        print(f"❌ Error creating completion flag: {e}")

if __name__ == "__main__":
    test_file_save()