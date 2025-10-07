#!/usr/bin/env python3
"""
Test Live API Response
Test what the actual API endpoint returns for the current document
"""

import json

def extract_value(field_data):
    """Extract simple string value from field data (same logic as analysis_routes.py)"""
    if isinstance(field_data, str):
        return field_data
    elif isinstance(field_data, dict):
        return field_data.get('value', str(field_data.get('company_name', '')))
    return str(field_data) if field_data is not None else ""

def test_live_api_response():
    """Test what the API would actually return"""
    
    print("🌐 TESTING LIVE API RESPONSE")
    print("=" * 50)
    
    # Load the current document
    document_id = "RAI-02102025-E1JZP-HV77U"
    results_file = f"analysis_results/{document_id}.json"
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        print(f"📄 Testing document: {document_id}")
        
        # Simulate the exact logic from analysis_routes.py get_document_status
        metadata = results.get('metadata', {})
        status = results.get('status', 'unknown')
        metadata_extraction = results.get('metadata_extraction', 'PENDING')
        
        # Extract values using the same extract_value function
        company_name = extract_value(metadata.get('company_name', ''))
        nature_of_business = extract_value(metadata.get('nature_of_business', ''))
        operational_demo = extract_value(metadata.get('operational_demographics', ''))
        financial_type = extract_value(metadata.get('financial_statements_type', 'Standalone'))
        
        print(f"🔍 EXTRACTED VALUES:")
        print(f"   company_name: {repr(company_name)}")
        print(f"   nature_of_business: {repr(nature_of_business[:50])}...")
        print(f"   operational_demo: {repr(operational_demo)}")
        print(f"   financial_type: {repr(financial_type)}")
        
        # Apply our geography array fix
        geography_list = []
        if operational_demo:
            geography_list = [country.strip() for country in operational_demo.split(',') if country.strip()]
        
        # Create the exact company_metadata structure sent to frontend
        company_metadata = {
            "company_name": company_name,
            "nature_of_business": nature_of_business,
            "geography_of_operations": geography_list,
            "financial_statement_type": financial_type,
            "confidence_score": 90
        }
        
        print(f"\n🏢 API RESPONSE - company_metadata:")
        print(json.dumps(company_metadata, indent=2))
        
        # Test frontend processing
        frontend_geo = company_metadata['geography_of_operations']
        if isinstance(frontend_geo, list) and frontend_geo:
            display = ', '.join(frontend_geo)
            print(f"\n✅ FRONTEND SHOULD DISPLAY: '{display}'")
        else:
            print(f"\n❌ FRONTEND WILL SHOW: 'Not specified'")
            print(f"   Reason: {repr(frontend_geo)}")
        
        # Full API response structure
        api_response = {
            "document_id": document_id,
            "status": status,
            "metadata_extraction": metadata_extraction,
            "company_metadata": company_metadata,
            "message": "Document analysis completed"
        }
        
        print(f"\n🌐 FULL API RESPONSE STRUCTURE:")
        print(f"   Keys: {list(api_response.keys())}")
        print(f"   company_metadata keys: {list(company_metadata.keys())}")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    test_live_api_response()