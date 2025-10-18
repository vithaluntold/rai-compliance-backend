import requests
import json

# Test framework selection with ONLY IAS 7
test_request = {
    "framework": "IFRS",
    "standards": ["IAS 7"],  # ONLY IAS 7 - no other standards
    "specialInstructions": "TEST BUG: ONLY analyze IAS 7 - DO NOT analyze any other standards!",
    "extensiveSearch": False,
    "processingMode": "smart"
}

print("🚨 TESTING STANDARDS SELECTION BUG")
print(f"📤 Sending request with ONLY: {test_request['standards']}")
print(f"📤 Standards count: {len(test_request['standards'])}")

# Make the API call (you'll need to replace DOCUMENT_ID with an actual document ID)
print("\n🔧 This script demonstrates the intended request.")
print("🔧 Use this exact payload to test against an actual document ID:")
print(json.dumps(test_request, indent=2))
