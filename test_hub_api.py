import requests
import json

# Test hub creation via API
def test_hub_creation():
    print("🧪 Testing hub creation API...")
    
    # API endpoint
    url = "http://localhost:8000/api/hubs"
    
    # Test hub data
    hub_data = {
        "name": "VISSAMSETTI GOPI SIVA KRISHNA",
        "code": "bgg66",
        "address": "1-94,OPPOSITE MRO OFFICE ,NALLAJERLA,534112",
        "city": "NALLAJERLA",
        "state": "Andhra Pradesh",
        "pincode": "534112",
        "phone": "+918977277424",
        "manager": "Rajesh Kumar",
        "capacity": 600,
        "status": "active"
    }
    
    try:
        print(f"📤 Sending POST request to: {url}")
        print(f"📋 Hub data: {json.dumps(hub_data, indent=2)}")
        
        response = requests.post(url, json=hub_data, timeout=10)
        
        print(f"📡 Response Status: {response.status_code}")
        print(f"📄 Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Hub created successfully!")
            print(f"📋 Created hub: {json.dumps(result, indent=2, default=str)}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"📄 Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection error: Make sure the backend server is running on localhost:8000")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_hub_creation()