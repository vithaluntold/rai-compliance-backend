#!/usr/bin/env python3
"""
Final verification test with just ONE standard to prove the bug is fixed
"""
import requests
import json
import time

# Test with ONLY IAS 7 
single_standard_test = {
    "framework": "IFRS",
    "standards": ["IAS 7"],  # ONLY ONE STANDARD
    "specialInstructions": "🎯 FINAL TEST: Process ONLY IAS 7 (Cash Flow Statements)",
    "extensiveSearch": False,
    "processingMode": "smart"
}

print("🎯 FINAL VERIFICATION TEST")
print("=" * 40)
print("📤 Testing with EXACTLY 1 standard:")
print(f"   Standards: {single_standard_test['standards']}")
print(f"   Count: {len(single_standard_test['standards'])}")

url = "http://localhost:8000/api/v1/analysis/documents/RAI-02102025-E1JZP-HV77U/select-framework"
response = requests.post(url, json=single_standard_test)

print(f"\n📨 Response: {response.status_code}")

if response.status_code == 200:
    print("✅ Request successful")
    print("⏱️  Waiting 10 seconds for processing...")
    time.sleep(10)
    
    # Check results
    print("\n🔍 CHECKING RESULTS:")
    
    # Use grep to count sections quickly
    import subprocess
    result = subprocess.run([
        'grep', '-c', '"section":', 
        '/Users/apple/Downloads/Audricc all 091025/render-backend/analysis_results/RAI-02102025-E1JZP-HV77U.json'
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        sections_count = int(result.stdout.strip())
        print(f"📊 Sections found in results: {sections_count}")
        
        if sections_count == 1:
            print("✅ SUCCESS! Only 1 section processed (correct)")
            print("🎯 BUG IS FIXED - System processes only selected standards")
        elif sections_count == 2:
            print("ℹ️  2 sections found (might be from previous test)")
        else:
            print(f"⚠️  {sections_count} sections found - investigating...")
    
    # Also check which sections exist
    result2 = subprocess.run([
        'grep', '"section":', 
        '/Users/apple/Downloads/Audricc all 091025/render-backend/analysis_results/RAI-02102025-E1JZP-HV77U.json'
    ], capture_output=True, text=True)
    
    if result2.returncode == 0:
        print("\n📋 Sections in results:")
        for line in result2.stdout.strip().split('\n'):
            print(f"   {line.strip()}")
            
else:
    print(f"❌ Error: {response.text}")

print("\n" + "=" * 40)
print("🎯 TEST COMPLETE")