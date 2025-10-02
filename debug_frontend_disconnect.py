#!/usr/bin/env python3
"""
Simulate exact frontend polling behavior and identify the disconnect
"""
import asyncio
import aiohttp
import json
import time

async def simulate_frontend_workflow():
    """Simulate exact frontend workflow with detailed analysis"""
    
    # Use the document from production logs
    document_id = "RAI-02102025-E1JZP-HV77U"
    base_url = "https://rai-compliance-backend.onrender.com/api/v1/analysis"
    
    async with aiohttp.ClientSession() as session:
        print("🔍 FRONTEND WORKFLOW SIMULATION")
        print("=" * 50)
        
        # Step 1: Check document status (main polling endpoint)
        print("\n📡 STEP 1: Frontend polls document status")
        async with session.get(f"{base_url}/documents/{document_id}") as response:
            if response.status == 200:
                result = await response.json()
                
                print(f"✅ Response received (200 OK)")
                print(f"📊 Raw Response Size: {len(json.dumps(result))} bytes")
                
                # Analyze exactly what frontend should see
                print(f"\n🔍 FRONTEND DATA ANALYSIS:")
                print(f"Status: '{result.get('status')}'")
                print(f"Metadata Extraction: '{result.get('metadata_extraction')}'")
                print(f"Message: '{result.get('message')}'")
                
                # Check metadata structure 
                metadata = result.get('metadata', {})
                print(f"\n📋 METADATA STRUCTURE:")
                print(f"Type: {type(metadata)}")
                print(f"Keys: {list(metadata.keys()) if isinstance(metadata, dict) else 'Not a dict'}")
                
                # Check each expected field
                expected_fields = ['company_name', 'nature_of_business', 'operational_demographics', 'financial_statements_type']
                
                print(f"\n🎯 FIELD-BY-FIELD ANALYSIS:")
                for field in expected_fields:
                    if field in metadata:
                        value = metadata[field]
                        print(f"✅ {field}: Present")
                        print(f"   Type: {type(value)}")
                        print(f"   Value: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
                        
                        # Check if it's in expected format (frontend might expect objects with .value)
                        if isinstance(value, dict) and 'value' in value:
                            print(f"   Nested Value: {value['value'][:100]}{'...' if len(str(value['value'])) > 100 else ''}")
                        
                    else:
                        print(f"❌ {field}: MISSING")
                
                # Check what triggers frontend should look for
                print(f"\n🎮 FRONTEND TRIGGER CHECK:")
                
                # Check if status indicates ready for framework selection
                if result.get('status') == 'COMPLETED' and result.get('metadata_extraction') == 'COMPLETED':
                    print("✅ Status indicates metadata complete - frontend should show framework selection")
                else:
                    print("❌ Status does not indicate completion")
                    
                # Check if metadata has actual content
                metadata_has_content = False
                if isinstance(metadata, dict):
                    for field, value in metadata.items():
                        if value and str(value).strip():
                            metadata_has_content = True
                            break
                            
                if metadata_has_content:
                    print("✅ Metadata has actual content - frontend should display it")
                else:
                    print("❌ Metadata is empty or invalid")
                    
                # Common frontend issues to check
                print(f"\n🔧 COMMON FRONTEND ISSUES:")
                
                # Issue 1: Frontend expects different field names
                print("1. Field name mapping:")
                common_variants = {
                    'companyName': metadata.get('companyName'),
                    'company_name': metadata.get('company_name'),
                    'natureOfBusiness': metadata.get('natureOfBusiness'),
                    'nature_of_business': metadata.get('nature_of_business')
                }
                for variant, value in common_variants.items():
                    if value:
                        print(f"   ✅ {variant}: {str(value)[:50]}...")
                    else:
                        print(f"   ❌ {variant}: Not found")
                
                # Issue 2: Frontend expects nested structure
                print("2. Nested structure check:")
                if any(isinstance(v, dict) and 'value' in v for v in metadata.values()):
                    print("   ✅ Metadata uses nested {value: ...} structure")
                else:
                    print("   ❌ Metadata uses flat structure")
                
                # Issue 3: Response format issues
                print("3. Response format:")
                if 'error' in result:
                    print(f"   ❌ Response contains error: {result['error']}")
                else:
                    print("   ✅ No error in response")
                    
            else:
                print(f"❌ Request failed: {response.status}")
                error_text = await response.text()
                print(f"Error: {error_text}")

        # Step 2: Test alternative endpoints that frontend might be using
        print(f"\n📡 STEP 2: Test alternative endpoints")
        
        alternative_endpoints = [
            f"/documents/{document_id}/results",
            f"/documents/{document_id}/metadata-results", 
            f"/documents/{document_id}/extract"
        ]
        
        for endpoint in alternative_endpoints:
            try:
                async with session.get(f"{base_url}{endpoint}") as response:
                    if response.status == 200:
                        result = await response.json()
                        metadata = result.get('metadata', {})
                        if metadata:
                            print(f"✅ {endpoint}: Has metadata ({len(metadata)} fields)")
                        else:
                            print(f"⚠️  {endpoint}: No metadata")
                    else:
                        print(f"❌ {endpoint}: Failed ({response.status})")
            except Exception as e:
                print(f"❌ {endpoint}: Exception - {str(e)}")

if __name__ == "__main__":
    asyncio.run(simulate_frontend_workflow())