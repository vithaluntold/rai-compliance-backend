from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
from datetime import datetime
import uvicorn
from pydantic import BaseModel, Field
import asyncpg
import os
from contextlib import asynccontextmanager
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection settings - will use your Render PostgreSQL
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/bluecart_erp")
print(f"🔗 Database URL configured: {DATABASE_URL[:50]}...") # Don't log full URL for security

# Global database pool
db_pool = None

# Lifespan context manager for database connection
@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool
    # Startup: Create database connection pool
    try:
        print("🔗 Connecting to PostgreSQL database...")
        db_pool = await asyncpg.create_pool(DATABASE_URL)
        print("✅ Database connection established")
        
        # Create tables if they don't exist
        await create_tables()
        print("✅ Database tables ready")
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("⚠️ Falling back to in-memory storage...")
        db_pool = None
    
    yield
    
    # Shutdown: Close database connection pool
    if db_pool:
        await db_pool.close()
        print("🔒 Database connection closed")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="BlueCart ERP API",
    description="Complete ERP system for logistics and shipment management with PostgreSQL",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:3001",
        "https://bluecart-erp.vercel.app",  # Add your Vercel deployment
        "https://*.vercel.app"  # Allow all Vercel preview deployments
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory fallback storage
shipments_memory: Dict[str, Dict] = {}

# Database functions
async def create_tables():
    """Create database tables if they don't exist"""
    if not db_pool:
        return
    
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS shipments (
        id VARCHAR(50) PRIMARY KEY,
        tracking_number VARCHAR(50) UNIQUE NOT NULL,
        sender_name VARCHAR(255) NOT NULL,
        sender_phone VARCHAR(50),
        sender_address TEXT NOT NULL,
        receiver_name VARCHAR(255) NOT NULL,
        receiver_phone VARCHAR(50),
        receiver_address TEXT NOT NULL,
        package_details TEXT NOT NULL,
        weight DECIMAL(10,2) NOT NULL,
        dimensions JSONB NOT NULL,
        service_type VARCHAR(20) NOT NULL DEFAULT 'standard',
        status VARCHAR(30) NOT NULL DEFAULT 'pending',
        pickup_date TIMESTAMP,
        estimated_delivery TIMESTAMP,
        actual_delivery TIMESTAMP,
        route VARCHAR(255),
        hub_id VARCHAR(50),
        events JSONB DEFAULT '[]',
        cost DECIMAL(10,2) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_shipments_tracking ON shipments(tracking_number);
    CREATE INDEX IF NOT EXISTS idx_shipments_status ON shipments(status);
    CREATE INDEX IF NOT EXISTS idx_shipments_created ON shipments(created_at);
    
    -- Hub table already exists, just ensure indexes
    CREATE INDEX IF NOT EXISTS idx_hubs_code_unique ON hubs(code) WHERE code IS NOT NULL;
    CREATE INDEX IF NOT EXISTS idx_hubs_status ON hubs(status);
    CREATE INDEX IF NOT EXISTS idx_hubs_city ON hubs(city);
    """
    
    async with db_pool.acquire() as conn:
        await conn.execute(create_table_sql)

async def get_db_connection():
    """Get database connection from pool"""
    if db_pool:
        return await db_pool.acquire()
    return None

# Pydantic models
class Dimensions(BaseModel):
    length: float = Field(..., gt=0)
    width: float = Field(..., gt=0)
    height: float = Field(..., gt=0)

class ShipmentEvent(BaseModel):
    id: str
    timestamp: datetime
    status: str
    location: str
    description: str

class ShipmentCreate(BaseModel):
    senderName: str = Field(..., min_length=1)
    senderPhone: Optional[str] = None
    senderAddress: str = Field(..., min_length=1)
    receiverName: str = Field(..., min_length=1)
    receiverPhone: Optional[str] = None
    receiverAddress: str = Field(..., min_length=1)
    packageDetails: str = Field(..., min_length=1)
    weight: float = Field(..., gt=0)
    dimensions: Dimensions
    serviceType: str = Field("standard", pattern="^(standard|express|overnight)$")
    cost: float = Field(..., gt=0)

class ShipmentUpdate(BaseModel):
    senderName: Optional[str] = None
    senderPhone: Optional[str] = None
    senderAddress: Optional[str] = None
    receiverName: Optional[str] = None
    receiverPhone: Optional[str] = None
    receiverAddress: Optional[str] = None
    packageDetails: Optional[str] = None
    weight: Optional[float] = None
    dimensions: Optional[Dimensions] = None
    serviceType: Optional[str] = None
    status: Optional[str] = None
    route: Optional[str] = None
    hubId: Optional[str] = None
    cost: Optional[float] = None

class ShipmentResponse(BaseModel):
    id: str
    trackingNumber: str
    senderName: str
    senderPhone: Optional[str]
    senderAddress: str
    receiverName: str
    receiverPhone: Optional[str]
    receiverAddress: str
    packageDetails: str
    weight: float
    dimensions: Dimensions
    serviceType: str
    status: str
    pickupDate: Optional[datetime]
    estimatedDelivery: Optional[datetime]
    actualDelivery: Optional[datetime]
    route: Optional[str]
    hubId: Optional[str]
    events: List[ShipmentEvent]
    cost: float
    createdAt: datetime
    updatedAt: datetime

class ShipmentsListResponse(BaseModel):
    shipments: List[ShipmentResponse]
    total: int
    skip: int
    limit: int

# Hub Pydantic models
class HubCreate(BaseModel):
    name: str = Field(..., min_length=1)
    code: str = Field(..., min_length=1)
    address: str = Field(..., min_length=1)
    city: str = Field(..., min_length=1)
    state: str = Field(..., min_length=1)
    pincode: Optional[str] = None
    phone: Optional[str] = None
    manager: Optional[str] = None
    capacity: int = Field(..., gt=0)
    status: str = Field("active", pattern="^(active|inactive|maintenance)$")

class HubUpdate(BaseModel):
    name: Optional[str] = None
    code: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    pincode: Optional[str] = None
    phone: Optional[str] = None
    manager: Optional[str] = None
    capacity: Optional[int] = None
    status: Optional[str] = None

class HubResponse(BaseModel):
    id: str
    name: str
    code: str
    address: str
    city: str
    state: str
    pincode: Optional[str]
    phone: Optional[str]
    manager: Optional[str]
    capacity: int
    currentLoad: int
    status: str
    createdAt: datetime
    updatedAt: datetime

class HubsListResponse(BaseModel):
    hubs: List[HubResponse]
    total: int
    skip: int
    limit: int

# Utility functions
def generate_id() -> str:
    """Generate unique shipment ID"""
    import time
    return f"SH{int(time.time() * 1000)}"

def generate_tracking_number() -> str:
    """Generate unique tracking number"""
    import time
    return f"TN{int(time.time() * 1000)}"

def calculate_estimated_delivery(service_type: str) -> datetime:
    """Calculate estimated delivery based on service type"""
    from datetime import timedelta
    
    days_map = {
        "standard": 5,
        "express": 3,
        "overnight": 1
    }
    
    days = days_map.get(service_type, 5)
    return datetime.now() + timedelta(days=days)

# Database operations
async def create_shipment_db(shipment_data: dict) -> dict:
    """Create shipment in database"""
    if not db_pool:
        # Fallback to in-memory storage
        shipment_id = shipment_data["id"]
        shipments_memory[shipment_id] = shipment_data
        return shipment_data
    
    async with db_pool.acquire() as conn:
        insert_sql = """
        INSERT INTO shipments (
            id, tracking_number, sender_name, sender_phone, sender_address,
            receiver_name, receiver_phone, receiver_address, package_details,
            weight, dimensions, service_type, status, estimated_delivery,
            route, hub_id, events, cost, created_at, updated_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20)
        RETURNING *
        """
        
        result = await conn.fetchrow(
            insert_sql,
            shipment_data["id"],
            shipment_data["trackingNumber"],
            shipment_data["senderName"],
            shipment_data.get("senderPhone"),
            shipment_data["senderAddress"],
            shipment_data["receiverName"],
            shipment_data.get("receiverPhone"),
            shipment_data["receiverAddress"],
            shipment_data["packageDetails"],
            shipment_data["weight"],
            json.dumps(shipment_data["dimensions"]),
            shipment_data["serviceType"],
            shipment_data["status"],
            shipment_data.get("estimatedDelivery"),
            shipment_data.get("route"),
            shipment_data.get("hubId"),
            json.dumps(shipment_data["events"]),
            shipment_data["cost"],
            shipment_data["createdAt"],
            shipment_data["updatedAt"]
        )
        
        return dict(result)

async def get_shipment_db(shipment_id: str) -> Optional[dict]:
    """Get shipment from database"""
    if not db_pool:
        # Fallback to in-memory storage
        return shipments_memory.get(shipment_id)
    
    async with db_pool.acquire() as conn:
        result = await conn.fetchrow(
            "SELECT * FROM shipments WHERE id = $1 OR tracking_number = $1",
            shipment_id
        )
        
        if result:
            return dict(result)
        return None

async def get_all_shipments_db(skip: int = 0, limit: int = 50) -> List[dict]:
    """Get all shipments from database"""
    if not db_pool:
        # Fallback to in-memory storage
        all_shipments = list(shipments_memory.values())
        return all_shipments[skip:skip + limit]
    
    async with db_pool.acquire() as conn:
        results = await conn.fetch(
            "SELECT * FROM shipments ORDER BY created_at DESC LIMIT $1 OFFSET $2",
            limit, skip
        )
        
        return [dict(row) for row in results]

async def update_shipment_db(shipment_id: str, updates: dict) -> Optional[dict]:
    """Update shipment in database"""
    if not db_pool:
        # Fallback to in-memory storage
        if shipment_id in shipments_memory:
            shipments_memory[shipment_id].update(updates)
            shipments_memory[shipment_id]["updatedAt"] = datetime.now()
            return shipments_memory[shipment_id]
        return None
    
    # Build dynamic update query based on provided fields
    set_clauses = []
    values = []
    param_count = 1
    
    for key, value in updates.items():
        if value is not None:
            # Convert camelCase to snake_case for database columns
            db_key = key
            if key == "senderName": db_key = "sender_name"
            elif key == "senderPhone": db_key = "sender_phone"
            elif key == "senderAddress": db_key = "sender_address"
            elif key == "receiverName": db_key = "receiver_name"
            elif key == "receiverPhone": db_key = "receiver_phone"
            elif key == "receiverAddress": db_key = "receiver_address"
            elif key == "packageDetails": db_key = "package_details"
            elif key == "serviceType": db_key = "service_type"
            elif key == "pickupDate": db_key = "pickup_date"
            elif key == "estimatedDelivery": db_key = "estimated_delivery"
            elif key == "actualDelivery": db_key = "actual_delivery"
            elif key == "hubId": db_key = "hub_id"
            elif key == "dimensions": value = json.dumps(value)
            elif key == "events": value = json.dumps(value)
            
            set_clauses.append(f"{db_key} = ${param_count}")
            values.append(value)
            param_count += 1
    
    if not set_clauses:
        return None
    
    # Add updated_at
    set_clauses.append(f"updated_at = ${param_count}")
    values.append(datetime.now())
    values.append(shipment_id)  # For WHERE clause
    
    update_sql = f"""
    UPDATE shipments 
    SET {', '.join(set_clauses)}
    WHERE id = ${param_count + 1}
    RETURNING *
    """
    
    async with db_pool.acquire() as conn:
        result = await conn.fetchrow(update_sql, *values)
        if result:
            return dict(result)
        return None

async def delete_shipment_db(shipment_id: str) -> bool:
    """Delete shipment from database"""
    if not db_pool:
        # Fallback to in-memory storage
        return shipments_memory.pop(shipment_id, None) is not None
    
    async with db_pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM shipments WHERE id = $1",
            shipment_id
        )
        
        return result == "DELETE 1"

# Hub database operations
async def create_hub_db(hub_data: dict) -> dict:
    """Create hub in database"""
    if not db_pool:
        # For now, just return the data (in-memory fallback could be added)
        return hub_data
    
    async with db_pool.acquire() as conn:
        insert_sql = """
        INSERT INTO hubs (
            id, name, code, address, city, state, pincode, phone, manager,
            capacity, current_load, status, created_at, updated_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
        RETURNING *
        """
        
        result = await conn.fetchrow(
            insert_sql,
            hub_data["id"],
            hub_data["name"],
            hub_data["code"],
            hub_data["address"],
            hub_data["city"],
            hub_data["state"],
            hub_data.get("pincode"),
            hub_data.get("phone"),
            hub_data.get("manager"),
            hub_data["capacity"],
            hub_data.get("currentLoad", 0),
            hub_data["status"],
            hub_data["createdAt"],
            hub_data["updatedAt"]
        )
        
        return dict(result)

async def get_hub_db(hub_id: str) -> Optional[dict]:
    """Get hub from database"""
    if not db_pool:
        return None
    
    async with db_pool.acquire() as conn:
        result = await conn.fetchrow(
            "SELECT * FROM hubs WHERE id = $1 OR code = $1",
            hub_id
        )
        
        if result:
            return dict(result)
        return None

async def get_all_hubs_db(skip: int = 0, limit: int = 50) -> List[dict]:
    """Get all hubs from database"""
    if not db_pool:
        return []
    
    async with db_pool.acquire() as conn:
        results = await conn.fetch(
            "SELECT * FROM hubs ORDER BY created_at DESC LIMIT $1 OFFSET $2",
            limit, skip
        )
        
        return [dict(row) for row in results]

async def update_hub_db(hub_id: str, updates: dict) -> Optional[dict]:
    """Update hub in database"""
    if not db_pool:
        return None
    
    # Build dynamic update query based on provided fields
    set_clauses = []
    values = []
    param_count = 1
    
    for key, value in updates.items():
        if value is not None:
            # Convert camelCase to snake_case for database columns
            db_key = key
            if key == "currentLoad": db_key = "current_load"
            elif key == "createdAt": db_key = "created_at"
            elif key == "updatedAt": db_key = "updated_at"
            
            set_clauses.append(f"{db_key} = ${param_count}")
            values.append(value)
            param_count += 1
    
    if not set_clauses:
        return None
    
    # Add updated_at
    set_clauses.append(f"updated_at = ${param_count}")
    values.append(datetime.now())
    values.append(hub_id)  # For WHERE clause
    
    update_sql = f"""
    UPDATE hubs 
    SET {', '.join(set_clauses)}
    WHERE id = ${param_count + 1}
    RETURNING *
    """
    
    async with db_pool.acquire() as conn:
        result = await conn.fetchrow(update_sql, *values)
        if result:
            return dict(result)
        return None

async def delete_hub_db(hub_id: str) -> bool:
    """Delete hub from database"""
    if not db_pool:
        return False
    
    async with db_pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM hubs WHERE id = $1",
            hub_id
        )
        
        return result == "DELETE 1"

# API Routes
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    db_status = "connected" if db_pool else "in-memory-fallback"
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "database": db_status,
        "version": "2.0.0"
    }

@app.post("/api/shipments", response_model=ShipmentResponse)
async def create_shipment(shipment: ShipmentCreate):
    """Create a new shipment"""
    try:
        # Generate IDs and timestamps
        shipment_id = generate_id()
        tracking_number = generate_tracking_number()
        now = datetime.now()
        
        # Create shipment data
        shipment_data = {
            "id": shipment_id,
            "trackingNumber": tracking_number,
            "senderName": shipment.senderName,
            "senderPhone": shipment.senderPhone,
            "senderAddress": shipment.senderAddress,
            "receiverName": shipment.receiverName,
            "receiverPhone": shipment.receiverPhone,
            "receiverAddress": shipment.receiverAddress,
            "packageDetails": shipment.packageDetails,
            "weight": shipment.weight,
            "dimensions": shipment.dimensions.dict(),
            "serviceType": shipment.serviceType,
            "status": "pending",
            "pickupDate": None,
            "estimatedDelivery": calculate_estimated_delivery(shipment.serviceType),
            "actualDelivery": None,
            "route": None,
            "hubId": None,
            "events": [],
            "cost": shipment.cost,
            "createdAt": now,
            "updatedAt": now
        }
        
        # Save to database
        created_shipment = await create_shipment_db(shipment_data)
        
        print(f"✅ Created shipment: {tracking_number}")
        return created_shipment
        
    except Exception as e:
        print(f"❌ Error creating shipment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create shipment: {str(e)}"
        )

@app.get("/api/shipments", response_model=ShipmentsListResponse)
async def get_shipments(skip: int = 0, limit: int = 50):
    """Get all shipments with pagination"""
    try:
        shipments = await get_all_shipments_db(skip, limit)
        
        # Convert database rows to response format
        formatted_shipments = []
        for shipment in shipments:
            # Handle database column name conversion
            if 'tracking_number' in shipment:
                shipment['trackingNumber'] = shipment.pop('tracking_number')
            if 'sender_name' in shipment:
                shipment['senderName'] = shipment.pop('sender_name')
            if 'sender_phone' in shipment:
                shipment['senderPhone'] = shipment.pop('sender_phone')
            if 'sender_address' in shipment:
                shipment['senderAddress'] = shipment.pop('sender_address')
            if 'receiver_name' in shipment:
                shipment['receiverName'] = shipment.pop('receiver_name')
            if 'receiver_phone' in shipment:
                shipment['receiverPhone'] = shipment.pop('receiver_phone')
            if 'receiver_address' in shipment:
                shipment['receiverAddress'] = shipment.pop('receiver_address')
            if 'package_details' in shipment:
                shipment['packageDetails'] = shipment.pop('package_details')
            if 'service_type' in shipment:
                shipment['serviceType'] = shipment.pop('service_type')
            if 'pickup_date' in shipment:
                shipment['pickupDate'] = shipment.pop('pickup_date')
            if 'estimated_delivery' in shipment:
                shipment['estimatedDelivery'] = shipment.pop('estimated_delivery')
            if 'actual_delivery' in shipment:
                shipment['actualDelivery'] = shipment.pop('actual_delivery')
            if 'hub_id' in shipment:
                shipment['hubId'] = shipment.pop('hub_id')
            if 'created_at' in shipment:
                shipment['createdAt'] = shipment.pop('created_at')
            if 'updated_at' in shipment:
                shipment['updatedAt'] = shipment.pop('updated_at')
            
            # Parse JSON fields
            if isinstance(shipment.get('dimensions'), str):
                shipment['dimensions'] = json.loads(shipment['dimensions'])
            if isinstance(shipment.get('events'), str):
                shipment['events'] = json.loads(shipment['events'])
            
            formatted_shipments.append(shipment)
        
        return {
            "shipments": formatted_shipments,
            "total": len(formatted_shipments),
            "skip": skip,
            "limit": limit
        }
        
    except Exception as e:
        print(f"❌ Error fetching shipments: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch shipments: {str(e)}"
        )

@app.get("/api/shipments/{shipment_id}", response_model=ShipmentResponse)
async def get_shipment(shipment_id: str):
    """Get a specific shipment by ID or tracking number"""
    try:
        shipment = await get_shipment_db(shipment_id)
        
        if not shipment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Shipment not found: {shipment_id}"
            )
        
        # Convert database format to response format (same as in get_shipments)
        if 'tracking_number' in shipment:
            shipment['trackingNumber'] = shipment.pop('tracking_number')
        if 'sender_name' in shipment:
            shipment['senderName'] = shipment.pop('sender_name')
        # ... (repeat other conversions as needed)
        
        return shipment
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error fetching shipment {shipment_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch shipment: {str(e)}"
        )

@app.put("/api/shipments/{shipment_id}", response_model=ShipmentResponse)
async def update_shipment(shipment_id: str, updates: ShipmentUpdate):
    """Update a shipment"""
    try:
        # Convert to dict and remove None values
        update_data = {k: v for k, v in updates.dict().items() if v is not None}
        
        if not update_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No update data provided"
            )
        
        # Add updated timestamp
        update_data["updatedAt"] = datetime.now()
        
        updated_shipment = await update_shipment_db(shipment_id, update_data)
        
        if not updated_shipment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Shipment not found: {shipment_id}"
            )
        
        return updated_shipment
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error updating shipment {shipment_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update shipment: {str(e)}"
        )

@app.delete("/api/shipments/{shipment_id}")
async def delete_shipment(shipment_id: str):
    """Delete a shipment"""
    try:
        deleted = await delete_shipment_db(shipment_id)
        
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Shipment not found: {shipment_id}"
            )
        
        return {"message": "Shipment deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error deleting shipment {shipment_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete shipment: {str(e)}"
        )

# Event management
@app.post("/api/shipments/{shipment_id}/events")
async def add_shipment_event(shipment_id: str, event: dict):
    """Add an event to a shipment"""
    try:
        # Get current shipment
        shipment = await get_shipment_db(shipment_id)
        if not shipment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Shipment not found: {shipment_id}"
            )
        
        # Add new event
        new_event = {
            "id": f"EV{int(datetime.now().timestamp() * 1000)}",
            "timestamp": datetime.now(),
            **event
        }
        
        # Parse existing events
        current_events = shipment.get("events", [])
        if isinstance(current_events, str):
            current_events = json.loads(current_events)
        
        current_events.append(new_event)
        
        # Update shipment with new event
        updated_shipment = await update_shipment_db(shipment_id, {
            "events": current_events,
            "status": event.get("status", shipment.get("status"))
        })
        
        return updated_shipment
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error adding event to shipment {shipment_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add event: {str(e)}"
        )

# Hub API endpoints
@app.post("/api/hubs", response_model=HubResponse)
async def create_hub(hub: HubCreate):
    """Create a new hub"""
    try:
        # Generate hub ID and timestamps
        hub_id = f"HUB{int(datetime.now().timestamp() * 1000)}"
        now = datetime.now()
        
        # Create hub data
        hub_data = {
            "id": hub_id,
            "name": hub.name,
            "code": hub.code.upper(),
            "address": hub.address,
            "city": hub.city,
            "state": hub.state,
            "pincode": hub.pincode,
            "phone": hub.phone,
            "manager": hub.manager,
            "capacity": hub.capacity,
            "currentLoad": 0,
            "status": hub.status,
            "createdAt": now,
            "updatedAt": now
        }
        
        # Save to database
        created_hub = await create_hub_db(hub_data)
        
        # Convert database fields to response format
        if 'current_load' in created_hub:
            created_hub['currentLoad'] = created_hub.pop('current_load')
        if 'created_at' in created_hub:
            created_hub['createdAt'] = created_hub.pop('created_at')
        if 'updated_at' in created_hub:
            created_hub['updatedAt'] = created_hub.pop('updated_at')
        
        print(f"✅ Created hub: {hub.code}")
        return created_hub
        
    except Exception as e:
        print(f"❌ Error creating hub: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create hub: {str(e)}"
        )

@app.get("/api/hubs", response_model=HubsListResponse)
async def get_hubs(skip: int = 0, limit: int = 50):
    """Get all hubs with pagination"""
    try:
        hubs = await get_all_hubs_db(skip, limit)
        
        # Convert database rows to response format
        formatted_hubs = []
        for hub in hubs:
            # Handle database column name conversion
            if 'current_load' in hub:
                hub['currentLoad'] = hub.pop('current_load')
            if 'created_at' in hub:
                hub['createdAt'] = hub.pop('created_at')  
            if 'updated_at' in hub:
                hub['updatedAt'] = hub.pop('updated_at')
            
            formatted_hubs.append(hub)
        
        return {
            "hubs": formatted_hubs,
            "total": len(formatted_hubs),
            "skip": skip,
            "limit": limit
        }
        
    except Exception as e:
        print(f"❌ Error fetching hubs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch hubs: {str(e)}"
        )

@app.get("/api/hubs/{hub_id}", response_model=HubResponse)
async def get_hub(hub_id: str):
    """Get a specific hub by ID"""
    try:
        hub = await get_hub_db(hub_id)
        
        if not hub:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Hub with ID {hub_id} not found"
            )
        
        # Convert database fields to response format
        if 'current_load' in hub:
            hub['currentLoad'] = hub.pop('current_load')
        if 'created_at' in hub:
            hub['createdAt'] = hub.pop('created_at')
        if 'updated_at' in hub:
            hub['updatedAt'] = hub.pop('updated_at')
        
        return hub
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error fetching hub {hub_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch hub: {str(e)}"
        )

@app.put("/api/hubs/{hub_id}", response_model=HubResponse)
async def update_hub(hub_id: str, hub_updates: HubUpdate):
    """Update a hub"""
    try:
        # Check if hub exists
        existing_hub = await get_hub_db(hub_id)
        if not existing_hub:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Hub with ID {hub_id} not found"
            )
        
        # Prepare updates
        updates = {}
        for field, value in hub_updates.dict(exclude_unset=True).items():
            if value is not None:
                updates[field] = value
        
        if not updates:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No updates provided"
            )
        
        # Update in database
        updated_hub = await update_hub_db(hub_id, updates)
        
        if not updated_hub:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update hub"
            )
        
        # Convert database fields to response format
        if 'current_load' in updated_hub:
            updated_hub['currentLoad'] = updated_hub.pop('current_load')
        if 'created_at' in updated_hub:
            updated_hub['createdAt'] = updated_hub.pop('created_at')
        if 'updated_at' in updated_hub:
            updated_hub['updatedAt'] = updated_hub.pop('updated_at')
        
        print(f"✅ Updated hub: {hub_id}")
        return updated_hub
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error updating hub {hub_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update hub: {str(e)}"
        )

@app.delete("/api/hubs/{hub_id}")
async def delete_hub(hub_id: str):
    """Delete a hub"""
    try:
        # Check if hub exists
        existing_hub = await get_hub_db(hub_id)
        if not existing_hub:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Hub with ID {hub_id} not found"
            )
        
        # Delete from database
        deleted = await delete_hub_db(hub_id)
        
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete hub"
            )
        
        print(f"✅ Deleted hub: {hub_id}")
        return {"message": f"Hub {hub_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error deleting hub {hub_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete hub: {str(e)}"
        )

# Analytics endpoints
@app.get("/api/analytics/dashboard")
async def get_dashboard_analytics():
    """Get dashboard analytics"""
    try:
        # This is a simplified version - you can expand based on your needs
        all_shipments = await get_all_shipments_db(0, 1000)  # Get a large sample
        
        total_shipments = len(all_shipments)
        
        # Count by status
        status_counts = {}
        for shipment in all_shipments:
            status = shipment.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "totalShipments": total_shipments,
            "statusBreakdown": status_counts,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        print(f"❌ Error fetching analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch analytics: {str(e)}"
        )

# User Management Models
class UserCreate(BaseModel):
    name: str
    email: str
    role: str = Field(..., description="User role: admin, hub-manager, delivery-personnel, operations, customer")
    phone: Optional[str] = None
    hub_id: Optional[str] = None
    password: str = Field(..., min_length=6)

class UserResponse(BaseModel):
    id: str
    name: str
    email: str
    role: str
    phone: Optional[str]
    hub_id: Optional[str]
    is_active: bool
    created_at: str

class UsersListResponse(BaseModel):
    users: List[UserResponse]
    total: int

# Route Management Models
class RouteCreate(BaseModel):
    name: str
    description: Optional[str] = None
    assigned_to: str  # User ID
    hub_id: str  # Starting hub
    shipment_ids: List[str] = []
    estimated_distance: Optional[float] = None
    estimated_time: Optional[str] = None
    status: str = Field(default="planned", description="Route status: planned, active, completed, cancelled")

class RouteResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    assigned_to: str
    hub_id: str
    shipment_ids: List[str]
    estimated_distance: Optional[float]
    estimated_time: Optional[str]
    status: str
    created_at: str
    updated_at: str

class RoutesListResponse(BaseModel):
    routes: List[RouteResponse]
    total: int

# User Management Endpoints
@app.post("/api/users", response_model=UserResponse)
async def create_user(user_data: UserCreate):
    """Create a new user"""
    try:
        print(f"📝 Creating user: {user_data.name} ({user_data.email})")
        
        user_id = f"USER_{int(datetime.now().timestamp())}"
        
        if db_pool:
            # Store in PostgreSQL
            async with db_pool.acquire() as conn:
                # Check if email already exists
                existing = await conn.fetchrow(
                    "SELECT id FROM users WHERE email = $1", 
                    user_data.email
                )
                if existing:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Email already exists"
                    )
                
                # Create user
                await conn.execute("""
                    INSERT INTO users (id, email, hashed_password, full_name, role, is_active)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """, user_id, user_data.email, f"hashed_{user_data.password}", 
                user_data.name, user_data.role, True)
                
                # Fetch created user
                user_row = await conn.fetchrow(
                    "SELECT * FROM users WHERE id = $1", user_id
                )
                
                return UserResponse(
                    id=user_row["id"],
                    name=user_row["full_name"],
                    email=user_row["email"],
                    role=user_row["role"],
                    phone=None,  # Add phone field to users table if needed
                    hub_id=None,  # Add hub_id field to users table if needed
                    is_active=user_row["is_active"],
                    created_at=user_row["created_at"].isoformat()
                )
        else:
            # In-memory fallback
            user = {
                "id": user_id,
                "name": user_data.name,
                "email": user_data.email,
                "role": user_data.role,
                "phone": user_data.phone,
                "hub_id": user_data.hub_id,
                "is_active": True,
                "created_at": datetime.now().isoformat()
            }
            
            # Store in memory (you'd want to add a users_memory dict)
            return UserResponse(**user)
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error creating user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create user: {str(e)}"
        )

@app.get("/api/users", response_model=UsersListResponse)
async def get_users(skip: int = 0, limit: int = 100):
    """Get all users"""
    try:
        print(f"📥 Fetching users (skip={skip}, limit={limit})")
        
        users = []
        
        if db_pool:
            async with db_pool.acquire() as conn:
                user_rows = await conn.fetch(
                    "SELECT * FROM users ORDER BY created_at DESC LIMIT $1 OFFSET $2",
                    limit, skip
                )
                
                for row in user_rows:
                    users.append(UserResponse(
                        id=row["id"],
                        name=row["full_name"],
                        email=row["email"],
                        role=row["role"],
                        phone=None,
                        hub_id=None,
                        is_active=row["is_active"],
                        created_at=row["created_at"].isoformat()
                    ))
        
        return UsersListResponse(users=users, total=len(users))
        
    except Exception as e:
        print(f"❌ Error fetching users: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch users: {str(e)}"
        )

# Route Management Endpoints  
@app.post("/api/routes", response_model=RouteResponse)
async def create_route(route_data: RouteCreate):
    """Create a new delivery route"""
    try:
        print(f"🚛 Creating route: {route_data.name}")
        
        route_id = f"ROUTE_{int(datetime.now().timestamp())}"
        now = datetime.now().isoformat()
        
        route = {
            "id": route_id,
            "name": route_data.name,
            "description": route_data.description,
            "assigned_to": route_data.assigned_to,
            "hub_id": route_data.hub_id,
            "shipment_ids": route_data.shipment_ids,
            "estimated_distance": route_data.estimated_distance,
            "estimated_time": route_data.estimated_time,
            "status": route_data.status,
            "created_at": now,
            "updated_at": now
        }
        
        # For now, store in memory - add routes table later if needed
        return RouteResponse(**route)
        
    except Exception as e:
        print(f"❌ Error creating route: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create route: {str(e)}"
        )

@app.get("/api/routes", response_model=RoutesListResponse)
async def get_routes(skip: int = 0, limit: int = 100):
    """Get all routes"""
    try:
        print(f"🚛 Fetching routes (skip={skip}, limit={limit})")
        
        # Return empty for now - implement route storage later
        routes = []
        
        return RoutesListResponse(routes=routes, total=len(routes))
        
    except Exception as e:
        print(f"❌ Error fetching routes: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch routes: {str(e)}"
        )

if __name__ == "__main__":
    print("🚀 Starting BlueCart ERP FastAPI Backend with PostgreSQL support...")
    print("📖 API Documentation will be available at: http://localhost:8000/docs")
    print("🔗 Frontend should connect to: http://localhost:8000")
    print("💾 Database: PostgreSQL (with in-memory fallback)")
    print("🌐 CORS enabled for localhost:3000, localhost:3001, and Vercel deployments")
    
    uvicorn.run(
        "main_postgres:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )