from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class ServiceType(str, Enum):
    standard = "standard"
    express = "express"
    overnight = "overnight"


class ShipmentStatus(str, Enum):
    pending = "pending"
    picked_up = "picked_up"
    in_transit = "in_transit"
    out_for_delivery = "out_for_delivery"
    delivered = "delivered"
    failed = "failed"


class Dimensions(BaseModel):
    length: float = Field(..., gt=0, description="Length in cm")
    width: float = Field(..., gt=0, description="Width in cm")
    height: float = Field(..., gt=0, description="Height in cm")


class ShipmentEvent(BaseModel):
    id: Optional[str] = None
    timestamp: datetime
    status: str
    location: str
    description: str


class ShipmentBase(BaseModel):
    sender_name: str = Field(..., min_length=1, max_length=255)
    sender_phone: Optional[str] = Field(None, max_length=20)
    sender_address: str = Field(..., min_length=1)
    receiver_name: str = Field(..., min_length=1, max_length=255)
    receiver_phone: Optional[str] = Field(None, max_length=20)
    receiver_address: str = Field(..., min_length=1)
    package_details: str = Field(..., min_length=1)
    weight: float = Field(..., gt=0, description="Weight in kg")
    dimensions: Dimensions
    service_type: ServiceType = ServiceType.standard
    pickup_date: Optional[datetime] = None
    route: Optional[str] = Field(None, max_length=255)
    hub_id: Optional[str] = Field(None, max_length=50)
    cost: float = Field(..., gt=0, description="Manual cost in USD")

    @validator('sender_phone', 'receiver_phone')
    def validate_phone(cls, v):
        if v and not v.replace('+', '').replace('-', '').replace(' ', '').isdigit():
            raise ValueError('Phone number must contain only digits, +, -, and spaces')
        return v


class ShipmentCreate(ShipmentBase):
    pass


class ShipmentUpdate(BaseModel):
    sender_name: Optional[str] = Field(None, min_length=1, max_length=255)
    sender_phone: Optional[str] = Field(None, max_length=20)
    sender_address: Optional[str] = Field(None, min_length=1)
    receiver_name: Optional[str] = Field(None, min_length=1, max_length=255)
    receiver_phone: Optional[str] = Field(None, max_length=20)
    receiver_address: Optional[str] = Field(None, min_length=1)
    package_details: Optional[str] = Field(None, min_length=1)
    weight: Optional[float] = Field(None, gt=0)
    dimensions: Optional[Dimensions] = None
    service_type: Optional[ServiceType] = None
    status: Optional[ShipmentStatus] = None
    pickup_date: Optional[datetime] = None
    estimated_delivery: Optional[datetime] = None
    actual_delivery: Optional[datetime] = None
    route: Optional[str] = Field(None, max_length=255)
    hub_id: Optional[str] = Field(None, max_length=50)
    cost: Optional[float] = Field(None, gt=0)


class ShipmentResponse(BaseModel):
    id: str
    tracking_number: str
    sender_name: str
    sender_phone: Optional[str]
    sender_address: str
    receiver_name: str
    receiver_phone: Optional[str]
    receiver_address: str
    package_details: str
    weight: float
    dimensions: Dict[str, float]
    service_type: str
    status: str
    pickup_date: Optional[datetime]
    estimated_delivery: Optional[datetime]
    actual_delivery: Optional[datetime]
    route: Optional[str]
    hub_id: Optional[str]
    events: List[Dict[str, Any]]
    cost: float
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ShipmentList(BaseModel):
    shipments: List[ShipmentResponse]
    total: int
    skip: int
    limit: int


class HealthCheck(BaseModel):
    status: str
    message: str
    timestamp: datetime


class UserBase(BaseModel):
    email: str = Field(..., description="User email address")
    full_name: str = Field(..., min_length=1, max_length=255)
    role: str = Field("user", description="User role: admin, manager, user")


class UserCreate(UserBase):
    password: str = Field(..., min_length=6, description="User password")


class UserResponse(UserBase):
    id: str
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class HubBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    address: str = Field(..., min_length=1)
    phone: Optional[str] = Field(None, max_length=20)
    manager_name: Optional[str] = Field(None, max_length=255)
    capacity: int = Field(1000, gt=0)
    coordinates: Optional[Dict[str, float]] = None


class HubCreate(HubBase):
    pass


class HubResponse(HubBase):
    id: str
    current_load: int
    status: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class AnalyticsResponse(BaseModel):
    total_shipments: int
    pending_shipments: int
    in_transit_shipments: int
    delivered_shipments: int
    failed_shipments: int
    total_revenue: float
    average_delivery_time: Optional[float]  # in days
    top_routes: List[Dict[str, Any]]
    daily_shipments: List[Dict[str, Any]]