#!/usr/bin/env python3
"""
Test the FastAPI backend endpoints locally using a test server
"""

import os
import json
import requests
import uvicorn
import threading
import time
from datetime import datetime

# Set test mode
os.environ["TEST_MODE"] = "true"

# Import the FastAPI app
from main_fastapi import app

def start_test_server():
    """Start the test server in a separate thread"""
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="error")

def test_shipment_endpoints():
    """Test the shipment creation and retrieval endpoints"""
    
    base_url = "http://127.0.0.1:8001"
    
    # Wait for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(2)
    
    # Test health endpoint first
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"✅ Health check: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Test shipment creation
    shipment_data = {
        "senderName": "VISSAMSETTI GOPI SIVA KRISHNA",
        "senderPhone": "+918977277424",
        "senderAddress": "1-94,OPPOSITE MRO OFFICE ,NALLAJERLA,534112, NALLAJERLA, Andhra Pradesh 534112",
        "receiverName": "VISSAMSETTI GOPI SIVA KRISHNA",
        "receiverPhone": "+918977277424",
        "receiverAddress": "1-94,OPPOSITE MRO OFFICE ,NALLAJERLA,534112, NALLAJERLA, Andhra Pradesh 534112",
        "packageDetails": "Test package for API testing",
        "weight": 5,
        "dimensions": {
            "length": 30,
            "width": 20,
            "height": 14
        },
        "serviceType": "express",
        "cost": 78
    }
    
    print("\n🚀 Testing POST /api/shipments...")
    try:
        response = requests.post(
            f"{base_url}/api/shipments", 
            json=shipment_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Success! Created shipment with tracking number: {result.get('trackingNumber')}")
            print(f"   📋 Full response: {json.dumps(result, indent=2)}")
        else:
            print(f"   ❌ Error: {response.status_code}")
            print(f"   Error details: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Request failed: {e}")
    
    # Test getting shipments
    print("\n🔍 Testing GET /api/shipments...")
    try:
        response = requests.get(f"{base_url}/api/shipments", timeout=5)
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Success! Found {result.get('count', 0)} shipments")
            if result.get('shipments'):
                print(f"   📋 First shipment: {json.dumps(result['shipments'][0], indent=2)}")
        else:
            print(f"   ❌ Error: {response.status_code}")
            print(f"   Error details: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Request failed: {e}")

if __name__ == "__main__":
    print("🧪 BlueCart ERP - FastAPI Backend Test")
    print("=" * 50)
    print("Starting test server on port 8001...")
    
    # Start server in background thread
    server_thread = threading.Thread(target=start_test_server, daemon=True)
    server_thread.start()
    
    # Run tests
    test_shipment_endpoints()
    
    print("\n🏁 Test completed!")