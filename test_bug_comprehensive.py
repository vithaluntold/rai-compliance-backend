#!/usr/bin/env python3
"""
Test script to demonstrate the standards selection bug.
This will show that when we select ONLY IAS 7, the backend processes ALL standards instead.
"""
import requests
import json
import sys
import time

# Configuration
BASE_URL = "http://localhost:8000"  # Backend is running on port 8000
TEST_DOCUMENT_ID = "RAI-02102025-E1JZP-HV77U"  # Using existing document with analysis results

def test_standards_bug():
    """Test framework selection with only IAS 7 - should NOT process all standards"""
    
    print("🚨 TESTING CRITICAL BUG: Standards Selection Override")
    print("=" * 60)
    
    # Test request with ONLY IAS 7
    test_payload = {
        "framework": "IFRS",
        "standards": ["IAS 7"],  # ONLY IAS 7 - critical test
        "specialInstructions": "🚨 BUG TEST: Process ONLY IAS 7 - Cash Flow Statements. DO NOT process any other standards!",
        "extensiveSearch": False,
        "processingMode": "smart"
    }
    
    print(f"📋 Test Document ID: {TEST_DOCUMENT_ID}")
    print(f"📤 Sending ONLY these standards: {test_payload['standards']}")
    print(f"📊 Standards count: {len(test_payload['standards'])}")
    print(f"⚠️  Expected behavior: Process ONLY IAS 7")
    print(f"🐛 Bug behavior: Will process ALL IFRS/IAS standards")
    print("\n📤 Full request payload:")
    print(json.dumps(test_payload, indent=2))
    
    # Make the API call
    try:
        url = f"{BASE_URL}/api/v1/analysis/documents/{TEST_DOCUMENT_ID}/select-framework"
        print(f"\n🌐 Sending POST to: {url}")
        
        response = requests.post(url, json=test_payload, timeout=30)
        
        print(f"📨 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Request accepted. Session ID: {result.get('session_id', 'N/A')}")
            
            # Check the debugging logs (look for our emoji markers)
            print("\n🔍 Look in server logs for debugging output with these markers:")
            print("   🎯 Framework selection debugging")
            print("   📤 Standards received from frontend")  
            print("   💾 Standards saved to storage")
            print("   ✅ Verification checks")
            
            return True
            
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        print("💡 Make sure the backend server is running on localhost:5000")
        return False

def check_results():
    """Check if analysis results show the bug"""
    print("\n🔍 CHECKING ANALYSIS RESULTS FOR BUG EVIDENCE")
    print("=" * 50)
    
    results_file = f"/Users/apple/Downloads/Audricc all 091025/render-backend/analysis_results/{TEST_DOCUMENT_ID}.json"
    
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
            
        # Check what standards were actually processed
        standards_processed = []
        
        # Look for standards in various places
        if 'framework_config' in data and 'standards' in data['framework_config']:
            standards_processed = data['framework_config']['standards']
            
        print(f"📄 Results file: {results_file}")
        print(f"📊 Standards found in results: {len(standards_processed)}")
        
        if len(standards_processed) == 1 and standards_processed[0] == "IAS 7":
            print("✅ SUCCESS: Only IAS 7 was processed (bug is fixed)")
        elif len(standards_processed) > 1:
            print(f"🐛 BUG CONFIRMED: {len(standards_processed)} standards processed instead of 1")
            print("🐛 Standards processed:")
            for std in standards_processed[:10]:  # Show first 10
                print(f"   - {std}")
            if len(standards_processed) > 10:
                print(f"   ... and {len(standards_processed) - 10} more")
        else:
            print("⚠️  Could not determine processed standards from results")
            
    except FileNotFoundError:
        print(f"⚠️  Results file not found yet: {results_file}")
        print("   Analysis may still be running...")
    except Exception as e:
        print(f"❌ Error reading results: {e}")

if __name__ == "__main__":
    print("🚨 RAI COMPLIANCE ENGINE - STANDARDS SELECTION BUG TEST")
    print("=" * 70)
    
    # Run the test
    success = test_standards_bug()
    
    if success:
        print("\n⏱️  Waiting 5 seconds for analysis to start...")
        time.sleep(5)
        check_results()
        
        print("\n📋 NEXT STEPS:")
        print("1. Check server logs for debugging output (look for emoji markers)")  
        print("2. Wait for analysis to complete")
        print("3. Check results file for evidence of bug")
        print("4. If bug confirmed, implement fix in framework selection pipeline")
    
    print("\n" + "=" * 70)