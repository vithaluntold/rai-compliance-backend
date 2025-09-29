#!/usr/bin/env python3

import requests
import json
import time
import sys

def test_health_endpoint():
    """Test basic health endpoint"""
    try:
        print("Testing health endpoint...")
        response = requests.get("http://127.0.0.1:8000/api/v1/health", timeout=10)
        print(f"Health Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Health Response: {response.json()}")
            return True
        else:
            print(f"Health Failed: {response.text}")
            return False
    except Exception as e:
        print(f"Health endpoint error: {e}")
        return False

def test_session_creation():
    """Test simple session creation"""
    try:
        print("Testing session creation...")
        session_data = {
            "title": "Simple Test Session",
            "description": "Basic session test",
        }
        
        response = requests.post(
            "http://127.0.0.1:8000/api/v1/sessions/create",
            json=session_data,
            timeout=10
        )
        
        print(f"Session Creation Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Created session: {result.get('session_id')}")
            return result.get('session_id')
        else:
            print(f"Session creation failed: {response.text}")
            return None
    except Exception as e:
        print(f"Session creation error: {e}")
        return None

def test_enhanced_endpoints(session_id):
    """Test enhanced session endpoints one by one"""
    if not session_id:
        print("No session ID to test enhanced endpoints")
        return
    
    # Test user choice endpoint
    try:
        print(f"Testing user choice endpoint for session {session_id}...")
        choice_data = {
            "choice_type": "framework_selection",
            "choice_value": "IND AS",
            "timestamp": "2025-01-22T10:00:00Z"
        }
        
        response = requests.post(
            f"http://127.0.0.1:8000/api/v1/sessions/{session_id}/user-choice",
            json=choice_data,
            timeout=10
        )
        
        print(f"User Choice Status: {response.status_code}")
        if response.status_code != 200:
            print(f"User choice failed: {response.text}")
    except Exception as e:
        print(f"User choice error: {e}")
    
    time.sleep(1)  # Small delay between requests

def main():
    print("🔧 Simple Endpoint Test")
    print("=" * 40)
    
    # Test health first
    if not test_health_endpoint():
        print("❌ Health check failed - cannot continue")
        sys.exit(1)
    
    time.sleep(1)
    
    # Test session creation
    session_id = test_session_creation()
    
    time.sleep(1)
    
    # Test enhanced endpoints
    test_enhanced_endpoints(session_id)

if __name__ == "__main__":
    main()