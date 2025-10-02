#!/usr/bin/env python3
"""
Test what metadata the frontend is actually getting vs what's stored.
"""
import os
import sys
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.staged_storage import StagedStorageManager

def test_document_metadata():
    """Test document RAI-02102025-E1JZP-HV77U metadata"""
    document_id = "RAI-02102025-E1JZP-HV77U"
    
    print(f"🔍 Testing metadata for document: {document_id}")
    
    # Check staged storage
    print("\n📦 STAGED STORAGE:")
    try:
        storage_manager = StagedStorageManager()
        staged_metadata = storage_manager.get_metadata(document_id)
        
        if staged_metadata:
            print("✅ Found staged metadata:")
            print(json.dumps(staged_metadata, indent=2))
            
            # Extract actual metadata
            extracted_metadata = staged_metadata.get('data', staged_metadata)
            print(f"\n📋 EXTRACTED DATA:")
            print(json.dumps(extracted_metadata, indent=2))
            
            # Test frontend transformation
            print(f"\n🖥️ FRONTEND TRANSFORMATION:")
            frontend_metadata = {
                "company_name": "",
                "nature_of_business": "", 
                "operational_demographics": "",
                "financial_statements_type": ""
            }
            
            for key, metadata_obj in extracted_metadata.items():
                if key == "optimization_metrics":
                    continue
                    
                if isinstance(metadata_obj, dict) and 'value' in metadata_obj:
                    value = metadata_obj['value']
                else:
                    value = metadata_obj
                
                print(f"Processing {key}: {value}")
                
                if key in ["company_name", "companyName"]:
                    if value and value != "":
                        frontend_metadata["company_name"] = value
                elif key in ["nature_of_business", "natureOfBusiness", "business_nature"]:
                    if value and value != "":
                        frontend_metadata["nature_of_business"] = value
                elif key in ["operational_demographics", "operationalDemographics", "geography", "demographics"]:
                    if value and value != "":
                        frontend_metadata["operational_demographics"] = value
                elif key in ["financial_statements_type", "financialStatementsType", "statement_type", "fs_type"]:
                    if value and value != "":
                        frontend_metadata["financial_statements_type"] = value
                        
            print("Final frontend metadata:")
            print(json.dumps(frontend_metadata, indent=2))
            
        else:
            print("❌ No staged metadata found")
    except Exception as e:
        print(f"❌ Error accessing staged storage: {e}")
    
    # Check results file
    print(f"\n📁 RESULTS FILE:")
    results_dir = "analysis_results"
    results_path = os.path.join(results_dir, f"{document_id}.json")
    
    if os.path.exists(results_path):
        print("✅ Found results file:")
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
            
        print(f"Status: {results.get('status', 'N/A')}")
        print(f"Metadata extraction: {results.get('metadata_extraction', 'N/A')}")
        print(f"Metadata: {results.get('metadata', 'N/A')}")
    else:
        print(f"❌ No results file found at {results_path}")
    
    # Check completion flag
    print(f"\n🏁 COMPLETION FLAG:")
    completion_flag_path = os.path.join(results_dir, f"{document_id}.metadata_completed")
    if os.path.exists(completion_flag_path):
        print("✅ Metadata completion flag exists")
    else:
        print("❌ No completion flag found")

if __name__ == "__main__":
    test_document_metadata()