from sqlalchemy import Column, String, Float, DateTime, JSON, Text, Integer, Boolean
from sqlalchemy.sql import func
from database import Base
import uuid
from datetime import datetime


class Shipment(Base):
    __tablename__ = "shipments"

    # Primary identifiers
    id = Column(String(20), primary_key=True, index=True)
    tracking_number = Column(String(20), unique=True, index=True, nullable=False)
    
    # Sender information
    sender_name = Column(String(255), nullable=False)
    sender_phone = Column(String(20), nullable=True)
    sender_address = Column(Text, nullable=False)
    
    # Receiver information
    receiver_name = Column(String(255), nullable=False)
    receiver_phone = Column(String(20), nullable=True)
    receiver_address = Column(Text, nullable=False)
    
    # Package details
    package_details = Column(Text, nullable=False)
    weight = Column(Float, nullable=False)
    dimensions = Column(JSON, nullable=False)  # {"length": 0, "width": 0, "height": 0}
    
    # Service and status
    service_type = Column(String(20), nullable=False, default="standard")  # standard, express, overnight
    status = Column(String(30), nullable=False, default="pending")
    
    # Dates and scheduling
    pickup_date = Column(DateTime(timezone=True), nullable=True)
    estimated_delivery = Column(DateTime(timezone=True), nullable=True)
    actual_delivery = Column(DateTime(timezone=True), nullable=True)
    
    # Route and hub information
    route = Column(String(255), nullable=True)
    hub_id = Column(String(50), nullable=True)
    
    # Events and tracking
    events = Column(JSON, nullable=False, default=list)  # Array of event objects
    
    # Pricing
    cost = Column(Float, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    def __repr__(self):
        return f"<Shipment(id={self.id}, tracking_number={self.tracking_number}, status={self.status})>"

    def to_dict(self):
        """Convert model to dictionary"""
        return {
            "id": self.id,
            "trackingNumber": self.tracking_number,
            "senderName": self.sender_name,
            "senderPhone": self.sender_phone,
            "senderAddress": self.sender_address,
            "receiverName": self.receiver_name,
            "receiverPhone": self.receiver_phone,
            "receiverAddress": self.receiver_address,
            "packageDetails": self.package_details,
            "weight": self.weight,
            "dimensions": self.dimensions,
            "serviceType": self.service_type,
            "status": self.status,
            "pickupDate": self.pickup_date.isoformat() if self.pickup_date else None,
            "estimatedDelivery": self.estimated_delivery.isoformat() if self.estimated_delivery else None,
            "actualDelivery": self.actual_delivery.isoformat() if self.actual_delivery else None,
            "route": self.route,
            "hubId": self.hub_id,
            "events": self.events,
            "cost": self.cost,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
            "updatedAt": self.updated_at.isoformat() if self.updated_at else None,
        }


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False, default="user")  # admin, manager, user
    is_active = Column(Boolean, default=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<User(id={self.id}, email={self.email}, role={self.role})>"


class Hub(Base):
    __tablename__ = "hubs"

    id = Column(String(50), primary_key=True)
    name = Column(String(255), nullable=False)
    address = Column(Text, nullable=False)
    phone = Column(String(20), nullable=True)
    manager_name = Column(String(255), nullable=True)
    capacity = Column(Integer, nullable=False, default=1000)
    current_load = Column(Integer, nullable=False, default=0)
    status = Column(String(20), nullable=False, default="active")  # active, inactive, maintenance
    
    coordinates = Column(JSON, nullable=True)  # {"lat": 0, "lng": 0}
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<Hub(id={self.id}, name={self.name}, status={self.status})>"