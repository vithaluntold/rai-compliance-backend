"""
BlueCart ERP Backend - FastAPI Version for Render Deployment
===========================================================
"""

import os
import json
import uuid
import random
import string
from datetime import datetime
from typing import List, Optional
from contextlib import contextmanager
from urllib.parse import urlparse

# FastAPI and related imports
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Database and utilities
import pg8000
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Test mode flag (set to True to run without database)
TEST_MODE = os.getenv("TEST_MODE", "false").lower() == "true"

# FastAPI app initialization
app = FastAPI(
    title=os.getenv("PROJECT_NAME", "BlueCart ERP Backend"),
    description="BlueCart ERP Backend API for logistics and supply chain management",
    version=os.getenv("VERSION", "1.0.0"),
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
origins = [
    "http://localhost:3000",
    "https://localhost:3000",
]

# Add CORS origins from environment
cors_origins = os.getenv("CORS_ORIGINS")
if cors_origins:
    try:
        custom_origins = json.loads(cors_origins)
        origins.extend(custom_origins)
    except json.JSONDecodeError:
        print("Invalid CORS_ORIGINS format, using defaults")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now, will restrict later
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Database utilities
def parse_database_url(url):
    """Parse DATABASE_URL into connection parameters"""
    if not url:
        raise ValueError("DATABASE_URL environment variable is required")
    
    parsed = urlparse(url)
    return {
        'host': parsed.hostname,
        'port': parsed.port or 5432,
        'database': parsed.path[1:],  # Remove leading slash
        'user': parsed.username,
        'password': parsed.password
    }

@contextmanager
def get_database_connection():
    """Database connection context manager"""
    if TEST_MODE:
        # Mock database connection for testing
        yield None
        return
        
    conn = None
    try:
        db_config = parse_database_url(os.getenv("DATABASE_URL"))
        conn = pg8000.connect(**db_config)
        conn.autocommit = True
        yield conn
    except Exception as e:
        print(f"Database connection error: {e}")
        raise
    finally:
        if conn:
            conn.close()

def generate_tracking_number():
    """Generate a unique tracking number"""
    prefix = "BC"
    numbers = ''.join(random.choices(string.digits, k=8))
    return f"{prefix}{numbers}"

def create_tables():
    """Create database tables if they don't exist"""
    if TEST_MODE:
        print("✅ Test mode: Skipping database table creation")
        return
        
    try:
        with get_database_connection() as conn:
            cursor = conn.cursor()
            
            # Create shipments table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS shipments (
                    id SERIAL PRIMARY KEY,
                    tracking_number VARCHAR(100) UNIQUE NOT NULL,
                    sender_name VARCHAR(255) NOT NULL,
                    sender_phone VARCHAR(20),
                    sender_address TEXT NOT NULL,
                    receiver_name VARCHAR(255) NOT NULL,
                    receiver_phone VARCHAR(20),
                    receiver_address TEXT NOT NULL,
                    package_details TEXT NOT NULL,
                    weight DECIMAL(10,2) NOT NULL,
                    dimensions JSONB NOT NULL,
                    service_type VARCHAR(50) DEFAULT 'standard',
                    cost DECIMAL(10,2) NOT NULL,
                    status VARCHAR(50) DEFAULT 'pending',
                    pickup_date TIMESTAMP,
                    estimated_delivery TIMESTAMP,
                    actual_delivery TIMESTAMP,
                    route VARCHAR(255),
                    hub_id VARCHAR(50),
                    events JSONB DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create hubs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS hubs (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    location VARCHAR(200) NOT NULL,
                    capacity INTEGER DEFAULT 1000,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    email VARCHAR(100) UNIQUE NOT NULL,
                    full_name VARCHAR(100),
                    role VARCHAR(20) DEFAULT 'user',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            print("✅ Database tables created successfully")
            
    except Exception as e:
        print(f"❌ Database table creation error: {e}")
        raise

# Pydantic models
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    environment: str

class DatabaseTestResponse(BaseModel):
    database_status: str
    test_query: str
    result: Optional[int]

class Dimensions(BaseModel):
    length: float
    width: float
    height: float

class ShipmentCreate(BaseModel):
    senderName: str
    senderPhone: Optional[str] = None
    senderAddress: str
    receiverName: str
    receiverPhone: Optional[str] = None
    receiverAddress: str
    packageDetails: str
    weight: float
    dimensions: Dimensions
    serviceType: str = "standard"
    cost: float

class ShipmentResponse(BaseModel):
    id: int
    trackingNumber: str
    senderName: str
    senderPhone: Optional[str]
    senderAddress: str
    receiverName: str
    receiverPhone: Optional[str]
    receiverAddress: str
    packageDetails: str
    weight: float
    dimensions: dict
    serviceType: str
    cost: float
    status: str
    createdAt: str
    updatedAt: str

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    try:
        create_tables()
        print("🚀 Backend startup completed successfully")
    except Exception as e:
        print(f"❌ Startup error: {e}")

# API Routes
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "BlueCart ERP Backend API",
        "version": os.getenv("VERSION", "1.0.0"),
        "docs": "/docs",
        "health": "/health",
        "api_v1": "/api/v1/"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version=os.getenv("VERSION", "1.0.0"),
        environment=os.getenv("ENVIRONMENT", "development")
    )

@app.get("/api/v1/health")
async def api_health_check():
    """API versioned health check"""
    return {
        "api_version": "v1",
        "status": "operational",
        "database": "connected",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/v1/test-db", response_model=DatabaseTestResponse)
async def test_database():
    """Test database connectivity"""
    try:
        with get_database_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 as test")
            result = cursor.fetchone()
            
            return DatabaseTestResponse(
                database_status="connected",
                test_query="passed",
                result=result[0] if result else None
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database connection failed: {str(e)}"
        )

@app.get("/api/v1/shipments")
async def get_shipments():
    """Get all shipments"""
    if TEST_MODE:
        # Return mock data for testing
        mock_shipments = [{
            "id": 1,
            "trackingNumber": "BC12345678",
            "senderName": "Test Sender",
            "senderPhone": "+1234567890",
            "senderAddress": "123 Test St, Test City",
            "receiverName": "Test Receiver",
            "receiverPhone": "+0987654321",
            "receiverAddress": "456 Test Ave, Test Town",
            "packageDetails": "Test package",
            "weight": 2.5,
            "dimensions": {"length": 10, "width": 10, "height": 10},
            "serviceType": "standard",
            "cost": 25.0,
            "status": "pending",
            "createdAt": datetime.utcnow().isoformat(),
            "updatedAt": datetime.utcnow().isoformat()
        }]
        return {"shipments": mock_shipments, "count": len(mock_shipments)}
    
    try:
        with get_database_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, tracking_number, sender_name, sender_phone, sender_address,
                       receiver_name, receiver_phone, receiver_address, package_details,
                       weight, dimensions, service_type, cost, status, 
                       created_at, updated_at
                FROM shipments ORDER BY created_at DESC LIMIT 50
            """)
            
            shipments = []
            for row in cursor.fetchall():
                dimensions = row[10] if row[10] else {}
                if isinstance(dimensions, str):
                    try:
                        dimensions = json.loads(dimensions)
                    except:
                        dimensions = {}
                        
                shipments.append({
                    "id": row[0],
                    "trackingNumber": row[1],
                    "senderName": row[2],
                    "senderPhone": row[3],
                    "senderAddress": row[4],
                    "receiverName": row[5],
                    "receiverPhone": row[6],
                    "receiverAddress": row[7],
                    "packageDetails": row[8],
                    "weight": float(row[9]) if row[9] else None,
                    "dimensions": dimensions,
                    "serviceType": row[11],
                    "cost": float(row[12]) if row[12] else None,
                    "status": row[13],
                    "createdAt": row[14].isoformat() if row[14] else None,
                    "updatedAt": row[15].isoformat() if row[15] else None
                })
            
            return {"shipments": shipments, "count": len(shipments)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/shipments")
async def get_shipments_non_versioned():
    """Get all shipments (non-versioned endpoint for frontend compatibility)"""
    return await get_shipments()

@app.get("/api/v1/shipments/{shipment_id}")
async def get_shipment_by_id_v1(shipment_id: int):
    """Get a single shipment by ID (v1 endpoint)"""
    return await get_shipment_by_id_internal(shipment_id)

@app.get("/api/shipments/{shipment_id}")
async def get_shipment_by_id(shipment_id: int):
    """Get a single shipment by ID (non-versioned endpoint)"""
    return await get_shipment_by_id_internal(shipment_id)

async def get_shipment_by_id_internal(shipment_id: int):
    """Internal function to get a shipment by ID"""
    try:
        with get_database_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, tracking_number, sender_name, sender_phone, sender_address,
                       receiver_name, receiver_phone, receiver_address, package_details,
                       weight, dimensions, service_type, cost, status, 
                       created_at, updated_at, events
                FROM shipments WHERE id = %s
            """, (shipment_id,))
            
            row = cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Shipment not found")
            
            dimensions = row[10] if row[10] else {}
            if isinstance(dimensions, str):
                try:
                    dimensions = json.loads(dimensions)
                except:
                    dimensions = {}
            
            events = row[17] if row[17] else []
            if isinstance(events, str):
                try:
                    events = json.loads(events)
                except:
                    events = []
            
            return {
                "id": row[0],
                "trackingNumber": row[1],
                "senderName": row[2],
                "senderPhone": row[3],
                "senderAddress": row[4],
                "receiverName": row[5],
                "receiverPhone": row[6],
                "receiverAddress": row[7],
                "packageDetails": row[8],
                "weight": float(row[9]) if row[9] else None,
                "dimensions": dimensions,
                "serviceType": row[11],
                "cost": float(row[12]) if row[12] else None,
                "status": row[13],
                "createdAt": row[14].isoformat() if row[14] else None,
                "updatedAt": row[15].isoformat() if row[15] else None,
                "events": events
            }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/hubs")
async def get_hubs():
    """Get all hubs"""
    try:
        with get_database_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, location, capacity, created_at FROM hubs")
            
            hubs = []
            for row in cursor.fetchall():
                hubs.append({
                    "id": row[0],
                    "name": row[1],
                    "location": row[2],
                    "capacity": row[3],
                    "created_at": row[4].isoformat() if row[4] else None
                })
            
            return {"hubs": hubs, "count": len(hubs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/shipments", response_model=ShipmentResponse)
async def create_shipment_v1(shipment: ShipmentCreate):
    """Create a new shipment (v1 endpoint)"""
    return await create_shipment_internal(shipment)

@app.post("/api/shipments", response_model=ShipmentResponse)
async def create_shipment(shipment: ShipmentCreate):
    """Create a new shipment (non-versioned endpoint for frontend compatibility)"""
    return await create_shipment_internal(shipment)

async def create_shipment_internal(shipment: ShipmentCreate):
    """Internal function to create a shipment"""
    if TEST_MODE:
        # Return mock response for testing
        tracking_number = generate_tracking_number()
        now = datetime.utcnow()
        
        return ShipmentResponse(
            id=12345,  # Mock ID
            trackingNumber=tracking_number,
            senderName=shipment.senderName,
            senderPhone=shipment.senderPhone,
            senderAddress=shipment.senderAddress,
            receiverName=shipment.receiverName,
            receiverPhone=shipment.receiverPhone,
            receiverAddress=shipment.receiverAddress,
            packageDetails=shipment.packageDetails,
            weight=shipment.weight,
            dimensions=shipment.dimensions.dict(),
            serviceType=shipment.serviceType,
            cost=shipment.cost,
            status="pending",
            createdAt=now.isoformat(),
            updatedAt=now.isoformat()
        )
    
    try:
        with get_database_connection() as conn:
            cursor = conn.cursor()
            
            # Generate tracking number
            tracking_number = generate_tracking_number()
            
            # Ensure tracking number is unique
            while True:
                cursor.execute("SELECT id FROM shipments WHERE tracking_number = %s", (tracking_number,))
                if not cursor.fetchone():
                    break
                tracking_number = generate_tracking_number()
            
            # Insert shipment
            cursor.execute("""
                INSERT INTO shipments (
                    tracking_number, sender_name, sender_phone, sender_address,
                    receiver_name, receiver_phone, receiver_address, package_details,
                    weight, dimensions, service_type, cost, status, events
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id, created_at, updated_at
            """, (
                tracking_number,
                shipment.senderName,
                shipment.senderPhone,
                shipment.senderAddress,
                shipment.receiverName,
                shipment.receiverPhone,
                shipment.receiverAddress,
                shipment.packageDetails,
                shipment.weight,
                json.dumps(shipment.dimensions.dict()),
                shipment.serviceType,
                shipment.cost,
                "pending",
                json.dumps([{
                    "timestamp": datetime.utcnow().isoformat(),
                    "status": "pending",
                    "location": "Origin",
                    "description": "Shipment created"
                }])
            ))
            
            result = cursor.fetchone()
            shipment_id, created_at, updated_at = result
            
            return ShipmentResponse(
                id=shipment_id,
                trackingNumber=tracking_number,
                senderName=shipment.senderName,
                senderPhone=shipment.senderPhone,
                senderAddress=shipment.senderAddress,
                receiverName=shipment.receiverName,
                receiverPhone=shipment.receiverPhone,
                receiverAddress=shipment.receiverAddress,
                packageDetails=shipment.packageDetails,
                weight=shipment.weight,
                dimensions=shipment.dimensions.dict(),
                serviceType=shipment.serviceType,
                cost=shipment.cost,
                status="pending",
                createdAt=created_at.isoformat(),
                updatedAt=updated_at.isoformat()
            )
            
    except Exception as e:
        print(f"Error creating shipment: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create shipment: {str(e)}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found"}
    )

@app.exception_handler(500)
async def server_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Main entry point
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main_fastapi:app", 
        host="0.0.0.0", 
        port=port,
        reload=False if os.getenv("ENVIRONMENT") == "production" else True
    )