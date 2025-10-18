#!/usr/bin/env python3
"""
Simple test to trigger debugging logs
"""
import requests
import json

# Test with just 2 standards to make it obvious when bug occurs
test_payload = {
    "framework": "IFRS", 
    "standards": ["IAS 1", "IAS 7"],  # Only 2 standards
    "specialInstructions": "🚨 DEBUGGING TEST: Only analyze IAS 1 and IAS 7",
    "extensiveSearch": False,
    "processingMode": "smart"
}

print("🎯 Testing with exactly 2 standards:")
print(json.dumps(test_payload, indent=2))

url = "http://localhost:8000/api/v1/analysis/documents/RAI-02102025-E1JZP-HV77U/select-framework"
response = requests.post(url, json=test_payload)

print(f"\n📨 Response: {response.status_code}")
if response.status_code == 200:
    print("✅ Check server console for debugging logs with emoji markers")
else:
    print(f"❌ Error: {response.text}")