#!/usr/bin/env python3
"""
Quick Server Connectivity Test
Checks if backend and frontend servers are accessible
"""

import requests
import time

def test_server_connectivity():
    """Test both backend and frontend connectivity"""
    
    print("🔍 Testing Server Connectivity...")
    print("=" * 50)
    
    # Test backend server
    backend_url = "http://localhost:8000"
    backend_endpoints = [
        "/",
        "/api/v1/health",
        "/health"
    ]
    
    print(f"\n🎯 Testing Backend Server: {backend_url}")
    print("-" * 30)
    
    backend_accessible = False
    for endpoint in backend_endpoints:
        try:
            response = requests.get(f"{backend_url}{endpoint}", timeout=3)
            print(f"✅ {endpoint}: {response.status_code} - OK")
            backend_accessible = True
            break
        except requests.exceptions.ConnectionError:
            print(f"❌ {endpoint}: Connection refused")
        except requests.exceptions.Timeout:
            print(f"⏱️ {endpoint}: Timeout")
        except Exception as e:
            print(f"❓ {endpoint}: {str(e)}")
    
    # Test frontend server  
    frontend_url = "http://localhost:3000"
    print(f"\n🎯 Testing Frontend Server: {frontend_url}")
    print("-" * 30)
    
    frontend_accessible = False
    try:
        response = requests.get(frontend_url, timeout=3)
        print(f"✅ /: {response.status_code} - OK")
        frontend_accessible = True
    except requests.exceptions.ConnectionError:
        print(f"❌ /: Connection refused")
    except requests.exceptions.Timeout:
        print(f"⏱️ /: Timeout")
    except Exception as e:
        print(f"❓ /: {str(e)}")
    
    # Alternative ports check
    print(f"\n🔍 Checking Alternative Ports...")
    print("-" * 30)
    
    alternative_ports = [8080, 5000, 8001, 3001]
    for port in alternative_ports:
        try:
            response = requests.get(f"http://localhost:{port}", timeout=1)
            print(f"✅ Port {port}: Server found! Status {response.status_code}")
        except:
            print(f"❌ Port {port}: No server")
    
    # Summary
    print(f"\n📊 Summary:")
    print("-" * 30)
    print(f"Backend (8000): {'✅ Accessible' if backend_accessible else '❌ Not accessible'}")
    print(f"Frontend (3000): {'✅ Accessible' if frontend_accessible else '❌ Not accessible'}")
    
    if backend_accessible and frontend_accessible:
        print(f"\n🎉 Both servers are running! Ready for user flow testing.")
        return True
    elif backend_accessible:
        print(f"\n⚠️ Backend only - Can test API endpoints but not full user flow.")
        return False
    else:
        print(f"\n🚫 No servers accessible - Please start servers before testing.")
        return False

if __name__ == "__main__":
    test_server_connectivity()