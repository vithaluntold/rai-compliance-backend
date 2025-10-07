#!/usr/bin/env python3
"""
Test the newer document with complex operational_demographics structure
"""

import json

def extract_value(field_data):
    """Extract simple string value from field data (same logic as analysis_routes.py)"""
    if isinstance(field_data, str):
        return field_data
    elif isinstance(field_data, dict):
        return field_data.get('value', str(field_data.get('company_name', '')))
    return str(field_data) if field_data is not None else ""

def test_newer_document():
    """Test the newer document that the frontend is probably using"""
    
    print("🔍 TESTING NEWER DOCUMENT")
    print("=" * 50)
    
    # Load the newer document
    document_id = "RAI-02102025-LTI88-AQ3JZ"
    results_file = f"../Audricc all/render-backend/analysis_results/{document_id}.json"
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        print(f"📄 Testing newer document: {document_id}")
        
        metadata = results.get('metadata', {})
        
        # Check the operational_demographics structure
        op_demo_raw = metadata.get('operational_demographics', {})
        print(f"🔍 RAW operational_demographics:")
        print(f"   Type: {type(op_demo_raw)}")
        print(f"   Value: {repr(op_demo_raw)}")
        
        if isinstance(op_demo_raw, dict):
            print(f"   Keys: {list(op_demo_raw.keys())}")
            print(f"   'value' field: {repr(op_demo_raw.get('value'))}")
            print(f"   'geography_of_operations' field: {repr(op_demo_raw.get('geography_of_operations'))}")
        
        # Test our extract_value function
        operational_demo = extract_value(op_demo_raw)
        print(f"\n🔧 EXTRACT_VALUE RESULT:")
        print(f"   Extracted: {repr(operational_demo)}")
        
        # Apply our geography array fix
        geography_list = []
        if operational_demo:
            geography_list = [country.strip() for country in operational_demo.split(',') if country.strip()]
        
        print(f"\n🌍 GEOGRAPHY ARRAY FIX:")
        print(f"   Input string: {repr(operational_demo)}")
        print(f"   Output array: {geography_list}")
        print(f"   Array length: {len(geography_list)}")
        
        # Create company_metadata
        company_metadata = {
            "geography_of_operations": geography_list
        }
        
        print(f"\n🏢 COMPANY_METADATA:")
        print(f"   geography_of_operations: {repr(company_metadata['geography_of_operations'])}")
        
        # Test frontend display
        frontend_geo = company_metadata['geography_of_operations']
        if isinstance(frontend_geo, list) and frontend_geo:
            display = ', '.join(frontend_geo)
            print(f"\n✅ FRONTEND SHOULD DISPLAY: '{display}'")
        else:
            print(f"\n❌ FRONTEND WILL SHOW: 'Not specified'")
            print(f"   Reason: {repr(frontend_geo)}")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_newer_document()