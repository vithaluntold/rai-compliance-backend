#!/usr/bin/env python3
"""
FastAPI Backend Setup and Test Script
"""

import subprocess
import sys
import os
import time
import requests
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is adequate"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is supported")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is too old. Need Python 3.8+")
        return False

def install_dependencies():
    """Install Python dependencies"""
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        return False
    return True

def check_postgres_connection():
    """Check PostgreSQL connection"""
    print("\n🗄️ Checking PostgreSQL connection...")
    try:
        import psycopg2
        from dotenv import load_dotenv
        load_dotenv()
        
        conn_params = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'root'),
            'database': os.getenv('POSTGRES_DB', 'shipment_erp')
        }
        
        conn = psycopg2.connect(**conn_params)
        conn.close()
        print("✅ PostgreSQL connection successful")
        return True
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
        print("💡 Make sure PostgreSQL is running and credentials are correct")
        return False

def create_database_tables():
    """Create database tables"""
    print("\n🗄️ Creating database tables...")
    try:
        from database import engine, Base
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to create database tables: {e}")
        return False

def run_tests():
    """Run API tests"""
    print("\n🧪 Running API tests...")
    return run_command("python -m pytest test_api.py -v", "Running tests")

def start_server_background():
    """Start FastAPI server in background"""
    print("\n🚀 Starting FastAPI server...")
    try:
        import subprocess
        process = subprocess.Popen(
            ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Check if server is running
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("✅ FastAPI server started successfully on http://localhost:8000")
                print("📖 API Documentation: http://localhost:8000/docs")
                print("📋 Alternative Docs: http://localhost:8000/redoc")
                return process
            else:
                print(f"❌ Server health check failed with status {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to connect to server: {e}")
            return None
            
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return None

def test_api_endpoints():
    """Test API endpoints"""
    print("\n🔧 Testing API endpoints...")
    
    base_url = "http://localhost:8000"
    
    # Test health check
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health check endpoint working")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check request failed: {e}")
        return False
    
    # Test shipments endpoint
    try:
        response = requests.get(f"{base_url}/api/shipments")
        if response.status_code == 200:
            print("✅ Shipments endpoint working")
        else:
            print(f"❌ Shipments endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Shipments request failed: {e}")
    
    # Test creating a shipment
    try:
        shipment_data = {
            "sender_name": "Test Sender",
            "sender_address": "123 Test St",
            "receiver_name": "Test Receiver",
            "receiver_address": "456 Test Ave",
            "package_details": "Test Package",
            "weight": 2.5,
            "dimensions": {"length": 30, "width": 20, "height": 10},
            "service_type": "standard",
            "cost": 15.99
        }
        
        response = requests.post(f"{base_url}/api/shipments", json=shipment_data)
        if response.status_code == 201:
            print("✅ Shipment creation working")
            shipment = response.json()
            print(f"📦 Created shipment: {shipment['tracking_number']}")
        else:
            print(f"❌ Shipment creation failed: {response.status_code}")
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"❌ Shipment creation request failed: {e}")
    
    return True

def main():
    """Main setup function"""
    print("🚀 BlueCart ERP FastAPI Backend Setup")
    print("=" * 50)
    
    # Change to backend directory
    backend_dir = Path(__file__).parent
    os.chdir(backend_dir)
    
    steps = [
        ("Check Python version", check_python_version),
        ("Install dependencies", install_dependencies),
        ("Check PostgreSQL connection", check_postgres_connection),
        ("Create database tables", create_database_tables),
        ("Run tests", run_tests),
    ]
    
    failed_steps = []
    
    for step_name, step_function in steps:
        if not step_function():
            failed_steps.append(step_name)
            if step_name in ["Check Python version", "Install dependencies"]:
                print(f"❌ Critical step '{step_name}' failed. Stopping setup.")
                return
    
    if failed_steps:
        print(f"\n⚠️ Setup completed with issues. Failed steps: {', '.join(failed_steps)}")
    else:
        print("\n✅ All setup steps completed successfully!")
    
    # Start server for testing
    print("\n" + "=" * 50)
    print("🌐 Starting server for testing...")
    
    server_process = start_server_background()
    
    if server_process:
        try:
            test_api_endpoints()
            
            print("\n" + "=" * 50)
            print("🎉 FastAPI Backend Setup Complete!")
            print("\n📋 Next Steps:")
            print("1. Server is running at: http://localhost:8000")
            print("2. API Documentation: http://localhost:8000/docs")
            print("3. Alternative Docs: http://localhost:8000/redoc")
            print("4. Test the API endpoints using the documentation")
            print("5. Integrate with your Next.js frontend")
            print("\n🛑 Press Ctrl+C to stop the server")
            
            # Keep server running
            server_process.wait()
            
        except KeyboardInterrupt:
            print("\n🛑 Stopping server...")
            server_process.terminate()
            server_process.wait()
            print("✅ Server stopped")
    else:
        print("❌ Could not start server for testing")

if __name__ == "__main__":
    main()