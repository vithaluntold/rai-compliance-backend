from sqlalchemy.orm import Session
from sqlalchemy import or_, func, desc
from models import Shipment, User, Hub
from schemas import ShipmentCreate, ShipmentUpdate
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import random
import string


class ShipmentCRUD:
    def generate_tracking_number(self) -> str:
        """Generate a unique tracking number"""
        prefix = "SHP"
        number = ''.join(random.choices(string.digits, k=6))
        return f"{prefix}{number}"

    def generate_shipment_id(self) -> str:
        """Generate a unique shipment ID"""
        prefix = "SH"
        timestamp = str(int(datetime.now().timestamp()))[-8:]
        return f"{prefix}{timestamp}"

    def calculate_estimated_delivery(self, service_type: str, pickup_date: Optional[datetime] = None) -> datetime:
        """Calculate estimated delivery date based on service type"""
        base_date = pickup_date or datetime.utcnow()
        
        days_map = {
            "standard": 3,
            "express": 2,
            "overnight": 1
        }
        
        days_to_add = days_map.get(service_type, 3)
        return base_date + timedelta(days=days_to_add)

    def create_shipment(self, db: Session, shipment: ShipmentCreate) -> Shipment:
        """Create a new shipment"""
        # Generate IDs
        shipment_id = self.generate_shipment_id()
        tracking_number = self.generate_tracking_number()
        
        # Ensure tracking number is unique
        while db.query(Shipment).filter(Shipment.tracking_number == tracking_number).first():
            tracking_number = self.generate_tracking_number()
        
        # Calculate estimated delivery
        estimated_delivery = self.calculate_estimated_delivery(
            shipment.service_type, 
            shipment.pickup_date
        )
        
        # Create initial event
        initial_event = {
            "id": f"EV{int(datetime.now().timestamp())}",
            "timestamp": datetime.utcnow().isoformat(),
            "status": "pending",
            "location": "Origin Hub",
            "description": "Shipment created and pending pickup"
        }
        
        # Create shipment object
        db_shipment = Shipment(
            id=shipment_id,
            tracking_number=tracking_number,
            sender_name=shipment.sender_name,
            sender_phone=shipment.sender_phone,
            sender_address=shipment.sender_address,
            receiver_name=shipment.receiver_name,
            receiver_phone=shipment.receiver_phone,
            receiver_address=shipment.receiver_address,
            package_details=shipment.package_details,
            weight=shipment.weight,
            dimensions=shipment.dimensions.dict(),
            service_type=shipment.service_type,
            status="pending",
            pickup_date=shipment.pickup_date,
            estimated_delivery=estimated_delivery,
            route=shipment.route,
            hub_id=shipment.hub_id,
            events=[initial_event],
            cost=shipment.cost
        )
        
        db.add(db_shipment)
        db.commit()
        db.refresh(db_shipment)
        
        return db_shipment

    def get_shipment(self, db: Session, shipment_id: str) -> Optional[Shipment]:
        """Get shipment by ID or tracking number"""
        return db.query(Shipment).filter(
            or_(
                Shipment.id == shipment_id,
                Shipment.tracking_number == shipment_id
            )
        ).first()

    def get_shipments(
        self, 
        db: Session, 
        skip: int = 0, 
        limit: int = 100,
        status_filter: Optional[str] = None
    ) -> List[Shipment]:
        """Get shipments with optional filtering"""
        query = db.query(Shipment)
        
        if status_filter:
            query = query.filter(Shipment.status == status_filter)
        
        return query.order_by(desc(Shipment.created_at)).offset(skip).limit(limit).all()

    def get_shipments_count(self, db: Session, status_filter: Optional[str] = None) -> int:
        """Get total count of shipments"""
        query = db.query(Shipment)
        
        if status_filter:
            query = query.filter(Shipment.status == status_filter)
        
        return query.count()

    def update_shipment(
        self, 
        db: Session, 
        shipment_id: str, 
        shipment_update: ShipmentUpdate
    ) -> Optional[Shipment]:
        """Update a shipment"""
        db_shipment = self.get_shipment(db, shipment_id)
        
        if not db_shipment:
            return None
        
        # Update fields
        update_data = shipment_update.dict(exclude_unset=True)
        
        for field, value in update_data.items():
            if field == "dimensions" and value:
                setattr(db_shipment, field, value.dict())
            else:
                setattr(db_shipment, field, value)
        
        # Update estimated delivery if service type changed
        if "service_type" in update_data:
            db_shipment.estimated_delivery = self.calculate_estimated_delivery(
                update_data["service_type"],
                db_shipment.pickup_date
            )
        
        db.commit()
        db.refresh(db_shipment)
        
        return db_shipment

    def delete_shipment(self, db: Session, shipment_id: str) -> bool:
        """Delete a shipment"""
        db_shipment = self.get_shipment(db, shipment_id)
        
        if not db_shipment:
            return False
        
        db.delete(db_shipment)
        db.commit()
        
        return True

    def add_shipment_event(
        self, 
        db: Session, 
        shipment_id: str, 
        event_data: Dict[str, Any]
    ) -> Optional[Shipment]:
        """Add an event to a shipment"""
        db_shipment = self.get_shipment(db, shipment_id)
        
        if not db_shipment:
            return None
        
        # Create new event
        new_event = {
            "id": f"EV{int(datetime.now().timestamp())}",
            "timestamp": datetime.utcnow().isoformat(),
            "status": event_data.get("status", db_shipment.status),
            "location": event_data.get("location", ""),
            "description": event_data.get("description", "")
        }
        
        # Update events list
        events = db_shipment.events or []
        events.append(new_event)
        db_shipment.events = events
        
        # Update shipment status if provided
        if "status" in event_data:
            db_shipment.status = event_data["status"]
            
            # Set actual delivery date if delivered
            if event_data["status"] == "delivered":
                db_shipment.actual_delivery = datetime.utcnow()
        
        db.commit()
        db.refresh(db_shipment)
        
        return db_shipment

    def get_dashboard_analytics(self, db: Session) -> Dict[str, Any]:
        """Get dashboard analytics"""
        # Basic counts
        total_shipments = db.query(Shipment).count()
        pending = db.query(Shipment).filter(Shipment.status == "pending").count()
        in_transit = db.query(Shipment).filter(Shipment.status == "in_transit").count()
        delivered = db.query(Shipment).filter(Shipment.status == "delivered").count()
        failed = db.query(Shipment).filter(Shipment.status == "failed").count()
        
        # Revenue calculation
        total_revenue = db.query(func.sum(Shipment.cost)).scalar() or 0.0
        
        # Average delivery time for delivered shipments
        delivered_shipments = db.query(Shipment).filter(
            Shipment.status == "delivered",
            Shipment.actual_delivery.isnot(None),
            Shipment.pickup_date.isnot(None)
        ).all()
        
        avg_delivery_time = None
        if delivered_shipments:
            total_days = sum([
                (s.actual_delivery - s.pickup_date).days 
                for s in delivered_shipments 
                if s.actual_delivery and s.pickup_date
            ])
            avg_delivery_time = total_days / len(delivered_shipments) if delivered_shipments else 0
        
        # Top routes
        route_stats = db.query(
            Shipment.route,
            func.count(Shipment.id).label('count')
        ).filter(
            Shipment.route.isnot(None)
        ).group_by(Shipment.route).order_by(desc('count')).limit(5).all()
        
        top_routes = [
            {"route": route, "count": count} 
            for route, count in route_stats
        ]
        
        # Daily shipments for the last 7 days
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        daily_stats = db.query(
            func.date(Shipment.created_at).label('date'),
            func.count(Shipment.id).label('count')
        ).filter(
            Shipment.created_at >= seven_days_ago
        ).group_by(func.date(Shipment.created_at)).all()
        
        daily_shipments = [
            {
                "date": date.isoformat() if date else None,
                "count": count
            }
            for date, count in daily_stats
        ]
        
        return {
            "total_shipments": total_shipments,
            "pending_shipments": pending,
            "in_transit_shipments": in_transit,
            "delivered_shipments": delivered,
            "failed_shipments": failed,
            "total_revenue": float(total_revenue),
            "average_delivery_time": avg_delivery_time,
            "top_routes": top_routes,
            "daily_shipments": daily_shipments
        }


# Create singleton instance
shipment_crud = ShipmentCRUD()