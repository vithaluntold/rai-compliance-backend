import pytest
import httpx
import asyncio
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from main import app
from database import get_database, Base
from models import Shipment
import os

# Test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Override database dependency
def override_get_database():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_database] = override_get_database

# Create test client
client = TestClient(app)

@pytest.fixture(scope="module")
def setup_database():
    """Setup test database"""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    if os.path.exists("./test.db"):
        os.remove("./test.db")

def test_health_check(setup_database):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "message" in data
    assert "timestamp" in data

def test_create_shipment(setup_database):
    """Test creating a new shipment"""
    shipment_data = {
        "sender_name": "John Doe",
        "sender_phone": "+1234567890",
        "sender_address": "123 Main St, City, State 12345",
        "receiver_name": "Jane Smith",
        "receiver_phone": "+0987654321",
        "receiver_address": "456 Oak Ave, City, State 67890",
        "package_details": "Electronics - Laptop",
        "weight": 2.5,
        "dimensions": {
            "length": 40.0,
            "width": 30.0,
            "height": 5.0
        },
        "service_type": "express",
        "cost": 25.99
    }
    
    response = client.post("/api/shipments", json=shipment_data)
    assert response.status_code == 201
    
    data = response.json()
    assert data["sender_name"] == "John Doe"
    assert data["receiver_name"] == "Jane Smith"
    assert data["status"] == "pending"
    assert data["cost"] == 25.99
    assert "id" in data
    assert "tracking_number" in data

def test_get_shipments(setup_database):
    """Test getting all shipments"""
    response = client.get("/api/shipments")
    assert response.status_code == 200
    
    data = response.json()
    assert "shipments" in data
    assert "total" in data
    assert isinstance(data["shipments"], list)

def test_get_shipment_by_id(setup_database):
    """Test getting a specific shipment"""
    # First create a shipment
    shipment_data = {
        "sender_name": "Test Sender",
        "sender_address": "Test Address",
        "receiver_name": "Test Receiver",
        "receiver_address": "Test Receiver Address",
        "package_details": "Test Package",
        "weight": 1.0,
        "dimensions": {"length": 10.0, "width": 10.0, "height": 10.0},
        "cost": 15.00
    }
    
    create_response = client.post("/api/shipments", json=shipment_data)
    assert create_response.status_code == 201
    created_shipment = create_response.json()
    
    # Then get it by ID
    response = client.get(f"/api/shipments/{created_shipment['id']}")
    assert response.status_code == 200
    
    data = response.json()
    assert data["id"] == created_shipment["id"]
    assert data["sender_name"] == "Test Sender"

def test_update_shipment(setup_database):
    """Test updating a shipment"""
    # Create a shipment
    shipment_data = {
        "sender_name": "Original Sender",
        "sender_address": "Original Address",
        "receiver_name": "Original Receiver",
        "receiver_address": "Original Receiver Address",
        "package_details": "Original Package",
        "weight": 1.0,
        "dimensions": {"length": 10.0, "width": 10.0, "height": 10.0},
        "cost": 20.00
    }
    
    create_response = client.post("/api/shipments", json=shipment_data)
    created_shipment = create_response.json()
    
    # Update the shipment
    update_data = {
        "sender_name": "Updated Sender",
        "status": "in_transit",
        "cost": 25.00
    }
    
    response = client.put(f"/api/shipments/{created_shipment['id']}", json=update_data)
    assert response.status_code == 200
    
    data = response.json()
    assert data["sender_name"] == "Updated Sender"
    assert data["status"] == "in_transit"
    assert data["cost"] == 25.00

def test_delete_shipment(setup_database):
    """Test deleting a shipment"""
    # Create a shipment
    shipment_data = {
        "sender_name": "To Delete",
        "sender_address": "Delete Address",
        "receiver_name": "Delete Receiver",
        "receiver_address": "Delete Receiver Address",
        "package_details": "Delete Package",
        "weight": 1.0,
        "dimensions": {"length": 10.0, "width": 10.0, "height": 10.0},
        "cost": 10.00
    }
    
    create_response = client.post("/api/shipments", json=shipment_data)
    created_shipment = create_response.json()
    
    # Delete the shipment
    response = client.delete(f"/api/shipments/{created_shipment['id']}")
    assert response.status_code == 204
    
    # Verify it's deleted
    get_response = client.get(f"/api/shipments/{created_shipment['id']}")
    assert get_response.status_code == 404

def test_add_shipment_event(setup_database):
    """Test adding an event to a shipment"""
    # Create a shipment
    shipment_data = {
        "sender_name": "Event Test",
        "sender_address": "Event Address",
        "receiver_name": "Event Receiver",
        "receiver_address": "Event Receiver Address",
        "package_details": "Event Package",
        "weight": 1.0,
        "dimensions": {"length": 10.0, "width": 10.0, "height": 10.0},
        "cost": 12.00
    }
    
    create_response = client.post("/api/shipments", json=shipment_data)
    created_shipment = create_response.json()
    
    # Add an event
    event_data = {
        "status": "picked_up",
        "location": "Origin Hub",
        "description": "Package picked up from sender"
    }
    
    response = client.post(f"/api/shipments/{created_shipment['id']}/events", json=event_data)
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "picked_up"
    assert len(data["events"]) >= 2  # Initial event + new event

def test_get_analytics(setup_database):
    """Test getting dashboard analytics"""
    response = client.get("/api/analytics/dashboard")
    assert response.status_code == 200
    
    data = response.json()
    assert "total_shipments" in data
    assert "pending_shipments" in data
    assert "total_revenue" in data
    assert isinstance(data["total_shipments"], int)

def test_shipment_validation(setup_database):
    """Test shipment validation"""
    # Test with invalid data
    invalid_data = {
        "sender_name": "",  # Empty name should fail
        "sender_address": "Valid Address",
        "receiver_name": "Valid Receiver",
        "receiver_address": "Valid Receiver Address",
        "package_details": "Valid Package",
        "weight": -1.0,  # Negative weight should fail
        "dimensions": {"length": 10.0, "width": 10.0, "height": 10.0},
        "cost": 10.00
    }
    
    response = client.post("/api/shipments", json=invalid_data)
    assert response.status_code == 422  # Validation error

if __name__ == "__main__":
    pytest.main([__file__, "-v"])