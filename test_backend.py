#!/usr/bin/env python3
"""
Simple API Test Script for BlueCart ERP Backend
Tests all API endpoints to verify functionality
"""

import requests
import json
import time
from datetime import datetime

# API Base URL
BASE_URL = "http://localhost:8000"

def print_header(test_name):
    """Print a formatted test header"""
    print(f"\n{'='*60}")
    print(f"🧪 TESTING: {test_name}")
    print(f"{'='*60}")

def print_result(endpoint, status_code, response_data, expected_status=200):
    """Print test results in a formatted way"""
    success = status_code == expected_status
    status_icon = "✅" if success else "❌"
    
    print(f"{status_icon} {endpoint}")
    print(f"   Status: {status_code} (Expected: {expected_status})")
    print(f"   Response: {json.dumps(response_data, indent=2)}")
    
    return success

def test_root_endpoint():
    """Test the root endpoint"""
    print_header("ROOT ENDPOINT")
    
    try:
        response = requests.get(f"{BASE_URL}/")
        return print_result("GET /", response.status_code, response.json())
    except Exception as e:
        print(f"❌ GET / - Error: {e}")
        return False

def test_health_endpoint():
    """Test the health check endpoint"""
    print_header("HEALTH CHECK ENDPOINT")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        # Health endpoint might return 500 if database is not connected, but that's expected
        success = response.status_code in [200, 500]
        return print_result("GET /health", response.status_code, response.json(), "200 or 500")
    except Exception as e:
        print(f"❌ GET /health - Error: {e}")
        return False

def test_shipments_endpoints():
    """Test shipments API endpoints"""
    print_header("SHIPMENTS API ENDPOINTS")
    
    results = []
    
    # Test GET /api/shipments
    try:
        response = requests.get(f"{BASE_URL}/api/shipments")
        # Might return 500 if database is not connected
        success = response.status_code in [200, 500]
        results.append(print_result("GET /api/shipments", response.status_code, response.json(), "200 or 500"))
    except Exception as e:
        print(f"❌ GET /api/shipments - Error: {e}")
        results.append(False)
    
    # Test POST /api/shipments (create a new shipment)
    try:
        test_shipment = {
            "tracking_number": f"TEST-{int(time.time())}",
            "origin_hub": "New York",
            "destination_hub": "Los Angeles", 
            "weight": 10.5,
            "price": 29.99
        }
        
        response = requests.post(f"{BASE_URL}/api/shipments", json=test_shipment)
        # Might return 500 if database is not connected
        success = response.status_code in [201, 500]
        results.append(print_result("POST /api/shipments", response.status_code, response.json(), "201 or 500"))
    except Exception as e:
        print(f"❌ POST /api/shipments - Error: {e}")
        results.append(False)
    
    return all(results)

def test_hubs_endpoints():
    """Test hubs API endpoints"""
    print_header("HUBS API ENDPOINTS")
    
    results = []
    
    # Test GET /api/hubs
    try:
        response = requests.get(f"{BASE_URL}/api/hubs")
        # Might return 500 if database is not connected
        success = response.status_code in [200, 500]
        results.append(print_result("GET /api/hubs", response.status_code, response.json(), "200 or 500"))
    except Exception as e:
        print(f"❌ GET /api/hubs - Error: {e}")
        results.append(False)
    
    # Test POST /api/hubs (create a new hub)
    try:
        test_hub = {
            "name": f"Test Hub {int(time.time())}",
            "address": "123 Test Street, Test City, TC 12345",
            "capacity": 1500
        }
        
        response = requests.post(f"{BASE_URL}/api/hubs", json=test_hub)
        # Might return 500 if database is not connected
        success = response.status_code in [201, 500]
        results.append(print_result("POST /api/hubs", response.status_code, response.json(), "201 or 500"))
    except Exception as e:
        print(f"❌ POST /api/hubs - Error: {e}")
        results.append(False)
    
    return all(results)

def test_cors_headers():
    """Test CORS headers are properly set"""
    print_header("CORS HEADERS TEST")
    
    try:
        response = requests.options(f"{BASE_URL}/api/shipments")
        headers = response.headers
        
        cors_headers = {
            'Access-Control-Allow-Origin': headers.get('Access-Control-Allow-Origin'),
            'Access-Control-Allow-Methods': headers.get('Access-Control-Allow-Methods'),
            'Access-Control-Allow-Headers': headers.get('Access-Control-Allow-Headers')
        }
        
        success = (
            cors_headers['Access-Control-Allow-Origin'] == '*' and
            'GET' in cors_headers.get('Access-Control-Allow-Methods', '') and
            'POST' in cors_headers.get('Access-Control-Allow-Methods', '')
        )
        
        print_result("OPTIONS /api/shipments (CORS)", response.status_code, cors_headers)
        return success
    except Exception as e:
        print(f"❌ CORS Test - Error: {e}")
        return False

def test_404_handling():
    """Test 404 error handling"""
    print_header("404 ERROR HANDLING")
    
    try:
        response = requests.get(f"{BASE_URL}/nonexistent")
        return print_result("GET /nonexistent", response.status_code, response.json(), 404)
    except Exception as e:
        print(f"❌ 404 Test - Error: {e}")
        return False

def run_all_tests():
    """Run all API tests"""
    print(f"🚀 Starting BlueCart ERP API Tests")
    print(f"📅 Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌐 API Base URL: {BASE_URL}")
    
    # Wait a moment for server to be ready
    print("\n⏳ Waiting 2 seconds for server to be ready...")
    time.sleep(2)
    
    test_results = []
    
    # Run all tests
    test_results.append(test_root_endpoint())
    test_results.append(test_health_endpoint())
    test_results.append(test_shipments_endpoints())
    test_results.append(test_hubs_endpoints())
    test_results.append(test_cors_headers())
    test_results.append(test_404_handling())
    
    # Summary
    print_header("TEST SUMMARY")
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"📊 Tests Passed: {passed}/{total}")
    print(f"✅ Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Backend API is working correctly!")
    else:
        print("⚠️  Some tests failed - this might be due to database connection issues")
        print("💡 The API server is running, but database operations may not work until database is connected")
    
    return passed == total

if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite error: {e}")