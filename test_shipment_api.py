#!/usr/bin/env python3
"""
Test the shipment creation API endpoint
"""

import json
import requests
from datetime import datetime

BASE_URL = "http://localhost:8000"

def test_create_shipment():
    """Test creating a new shipment"""
    
    # Test data (matching the frontend format)
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
    
    print("🚀 Testing shipment creation API...")
    print(f"URL: {BASE_URL}/api/shipments")
    print(f"Data: {json.dumps(shipment_data, indent=2)}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/shipments", 
            json=shipment_data,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"\n📊 Response Status: {response.status_code}")
        print(f"📋 Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Created shipment:")
            print(json.dumps(result, indent=2))
            return result
        else:
            print(f"❌ Error: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"Error details: {json.dumps(error_detail, indent=2)}")
            except:
                print(f"Error text: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the FastAPI server is running on localhost:8000")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

def test_get_shipments():
    """Test getting all shipments"""
    print("\n🔍 Testing get shipments API...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/shipments")
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Found {result.get('count', 0)} shipments")
            if result.get('shipments'):
                print("First shipment:")
                print(json.dumps(result['shipments'][0], indent=2))
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🧪 BlueCart ERP - Shipment API Test")
    print("=" * 50)
    
    # Test creating a shipment
    created_shipment = test_create_shipment()
    
    # Test getting shipments
    test_get_shipments()
    
    print("\n🏁 Test completed!")