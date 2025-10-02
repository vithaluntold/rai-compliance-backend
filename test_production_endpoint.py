#!/usr/bin/env python3
"""
Test production endpoint response directly 
"""
import asyncio
import aiohttp
import json

async def test_production_endpoint():
    """Test the production endpoint directly"""
    document_id = "RAI-02102025-E1JZP-HV77U"  # From the logs
    
    base_url = "https://rai-compliance-backend.onrender.com/api/v1/analysis"
    
    async with aiohttp.ClientSession() as session:
        print(f"🔍 Testing production endpoint for document: {document_id}")
        
        async with session.get(f"{base_url}/documents/{document_id}") as response:
            if response.status == 200:
                result = await response.json()
                
                print(f"📊 Response Status: {response.status}")
                print(f"📊 Response Headers: {dict(response.headers)}")
                print(f"📊 Response Body:")
                print(json.dumps(result, indent=2))
                
                # Analyze the response
                print(f"\n🔍 ANALYSIS:")
                print(f"Status: {result.get('status')}")
                print(f"Metadata Extraction: {result.get('metadata_extraction')}")
                print(f"Compliance Analysis: {result.get('compliance_analysis')}")
                
                metadata = result.get('metadata', {})
                print(f"Metadata Fields Count: {len(metadata)}")
                
                if metadata:
                    print(f"Metadata Fields Present:")
                    for key, value in metadata.items():
                        if isinstance(value, dict) and 'value' in value:
                            print(f"  {key}: '{value['value']}'")
                        else:
                            print(f"  {key}: '{value}'")
                else:
                    print("❌ NO METADATA IN RESPONSE!")
                    
            else:
                print(f"❌ Request failed with status: {response.status}")
                error_text = await response.text()
                print(f"Error: {error_text}")

if __name__ == "__main__":
    asyncio.run(test_production_endpoint())