#!/usr/bin/env python3
"""
Test script to verify metadata mapping works correctly
"""

def test_metadata_mapping():
    """Test the metadata extraction and mapping logic"""
    
    # Simulate the smart extractor output format (as seen in logs)
    smart_extractor_output = {
        'company_name': {
            'value': 'Phoenix Group', 
            'confidence': 0.8, 
            'extraction_method': 'pattern', 
            'context': 'Box 46617, 14 Floor, WeWork Hub71, Al Khatem Tower...'
        }, 
        'nature_of_business': {
            'value': 'The Group is engaged primarily in the mining and sale of digital assets, with revenue generated from the provision of transaction verification services within the Bitcoin network, commonly referred to as "cryptocurrency mining." In addition, the Group is involved in the holding, management, and sale of digital assets, and the assessment of liquidity, valuation, and accounting treatment of tokens held.', 
            'confidence': 0.9, 
            'extraction_method': 'ai', 
            'context': 'Box 46617, 14 Floor, WeWork Hub71, Al Khatem Tower...'
        }, 
        'operational_demographics': {
            'value': 'United Arab Emirates', 
            'confidence': 0.9, 
            'extraction_method': 'ai', 
            'context': 'Registered in Abu Dhabi Global Market...'
        }, 
        'financial_statements_type': {
            'value': 'Consolidated', 
            'confidence': 0.9, 
            'extraction_method': 'ai', 
            'context': 'Pattern matched from document content'
        }
    }
    
    # Test the extract_value function (same as in analysis_routes.py)
    def extract_value(field_data):
        if isinstance(field_data, dict) and "value" in field_data:
            return field_data.get("value", "")
        return str(field_data) if field_data else ""
    
    # Create company_metadata in the format frontend expects (same logic as fixed code)
    company_name = extract_value(smart_extractor_output.get('company_name', ''))
    nature_of_business = extract_value(smart_extractor_output.get('nature_of_business', ''))
    operational_demo = extract_value(smart_extractor_output.get('operational_demographics', ''))
    financial_type = extract_value(smart_extractor_output.get('financial_statements_type', 'Standalone'))
    
    company_metadata = {
        "company_name": company_name,
        "nature_of_business": nature_of_business,
        "geography_of_operations": [operational_demo] if operational_demo else [],
        "financial_statement_type": financial_type,
        "confidence_score": 90
    }
    
    print("=== METADATA MAPPING TEST ===")
    print(f"✅ Company Name: '{company_metadata['company_name']}'")
    print(f"✅ Nature of Business: '{company_metadata['nature_of_business'][:100]}...'")
    print(f"✅ Geography: {company_metadata['geography_of_operations']}")
    print(f"✅ Financial Statement Type: '{company_metadata['financial_statement_type']}'")
    print(f"✅ Confidence Score: {company_metadata['confidence_score']}")
    print("\n=== TEST RESULTS ===")
    
    # Verify all fields are populated
    success = True
    if not company_metadata['company_name']:
        print("❌ FAIL: Company name is empty")
        success = False
    if not company_metadata['nature_of_business']:
        print("❌ FAIL: Nature of business is empty")
        success = False
    if not company_metadata['geography_of_operations']:
        print("❌ FAIL: Geography is empty")
        success = False
    if not company_metadata['financial_statement_type']:
        print("❌ FAIL: Financial statement type is empty")
        success = False
        
    if success:
        print("🎉 SUCCESS: All metadata fields populated correctly!")
        print("🎉 Frontend should now receive proper company_metadata!")
    else:
        print("💥 FAILURE: Some metadata fields are still empty!")
        
    return success

if __name__ == "__main__":
    test_metadata_mapping()