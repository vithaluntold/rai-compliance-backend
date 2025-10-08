-- =====================================================
-- BlueCart ERP - Complete Database Schema
-- =====================================================
-- PostgreSQL schema for comprehensive ERP system
-- Compatible with Render PostgreSQL and all cloud providers
-- Supports: Shipments, Hubs, Users, Routes, Analytics, Process Mining
-- Created: October 2025
-- =====================================================

-- Enable UUID extension for unique identifiers
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =====================================================
-- CORE TABLES
-- =====================================================

-- Users table (Authentication and User Management)
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(100) NOT NULL,
    phone VARCHAR(20),
    role VARCHAR(50) DEFAULT 'operator' CHECK (role IN ('admin', 'manager', 'operator', 'viewer')),
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'suspended')),
    avatar_url TEXT,
    last_login TIMESTAMP,
    login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP,
    email_verified BOOLEAN DEFAULT FALSE,
    phone_verified BOOLEAN DEFAULT FALSE,
    two_factor_enabled BOOLEAN DEFAULT FALSE,
    two_factor_secret VARCHAR(32),
    preferences JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Hubs table (Logistics Centers/Warehouses)
CREATE TABLE IF NOT EXISTS hubs (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    name VARCHAR(100) UNIQUE NOT NULL,
    code VARCHAR(10) UNIQUE NOT NULL,
    address TEXT NOT NULL,
    city VARCHAR(50) NOT NULL,
    state VARCHAR(50) NOT NULL,
    country VARCHAR(50) NOT NULL,
    postal_code VARCHAR(20) NOT NULL,
    latitude DECIMAL(10,8),
    longitude DECIMAL(11,8),
    capacity INTEGER DEFAULT 1000 CHECK (capacity > 0),
    current_load INTEGER DEFAULT 0 CHECK (current_load >= 0),
    utilization_percentage DECIMAL(5,2) GENERATED ALWAYS AS (
        CASE 
            WHEN capacity > 0 THEN (current_load::DECIMAL / capacity * 100)
            ELSE 0 
        END
    ) STORED,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'maintenance', 'closed')),
    hub_type VARCHAR(30) DEFAULT 'distribution' CHECK (hub_type IN ('distribution', 'sorting', 'pickup', 'delivery', 'storage')),
    contact_person VARCHAR(100),
    contact_phone VARCHAR(20),
    contact_email VARCHAR(100),
    operating_hours JSONB DEFAULT '{"monday": "09:00-17:00", "tuesday": "09:00-17:00", "wednesday": "09:00-17:00", "thursday": "09:00-17:00", "friday": "09:00-17:00", "saturday": "09:00-13:00", "sunday": "closed"}',
    facilities JSONB DEFAULT '[]', -- Array of facility features
    manager_id INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Routes table (Delivery Routes between Hubs)
CREATE TABLE IF NOT EXISTS routes (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    code VARCHAR(20) UNIQUE NOT NULL,
    origin_hub_id INTEGER NOT NULL REFERENCES hubs(id) ON DELETE CASCADE,
    destination_hub_id INTEGER NOT NULL REFERENCES hubs(id) ON DELETE CASCADE,
    distance_km DECIMAL(10,2) CHECK (distance_km > 0),
    estimated_duration_minutes INTEGER CHECK (estimated_duration_minutes > 0),
    cost_per_kg DECIMAL(10,2) DEFAULT 0.00,
    base_cost DECIMAL(10,2) DEFAULT 0.00,
    route_type VARCHAR(30) DEFAULT 'ground' CHECK (route_type IN ('ground', 'air', 'sea', 'rail')),
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'suspended', 'maintenance')),
    priority INTEGER DEFAULT 1 CHECK (priority BETWEEN 1 AND 10),
    max_weight_kg DECIMAL(10,2) DEFAULT 1000.00,
    max_volume_m3 DECIMAL(10,2) DEFAULT 50.00,
    schedule JSONB DEFAULT '{}', -- Schedule information
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT no_self_route CHECK (origin_hub_id != destination_hub_id)
);

-- Shipments table (Core Shipment Management)
CREATE TABLE IF NOT EXISTS shipments (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    tracking_number VARCHAR(100) UNIQUE NOT NULL,
    reference_number VARCHAR(100),
    
    -- Sender Information
    sender_name VARCHAR(100) NOT NULL,
    sender_email VARCHAR(100),
    sender_phone VARCHAR(20),
    sender_address TEXT NOT NULL,
    sender_city VARCHAR(50) NOT NULL,
    sender_state VARCHAR(50) NOT NULL,
    sender_country VARCHAR(50) NOT NULL,
    sender_postal_code VARCHAR(20) NOT NULL,
    
    -- Receiver Information
    receiver_name VARCHAR(100) NOT NULL,
    receiver_email VARCHAR(100),
    receiver_phone VARCHAR(20),
    receiver_address TEXT NOT NULL,
    receiver_city VARCHAR(50) NOT NULL,
    receiver_state VARCHAR(50) NOT NULL,
    receiver_country VARCHAR(50) NOT NULL,
    receiver_postal_code VARCHAR(20) NOT NULL,
    
    -- Hub Information
    origin_hub_id INTEGER REFERENCES hubs(id),
    destination_hub_id INTEGER REFERENCES hubs(id),
    current_hub_id INTEGER REFERENCES hubs(id),
    
    -- Shipment Details
    package_count INTEGER DEFAULT 1 CHECK (package_count > 0),
    total_weight_kg DECIMAL(10,2) NOT NULL CHECK (total_weight_kg > 0),
    dimensions_length_cm DECIMAL(8,2),
    dimensions_width_cm DECIMAL(8,2),
    dimensions_height_cm DECIMAL(8,2),
    volume_m3 DECIMAL(10,4) GENERATED ALWAYS AS (
        CASE 
            WHEN dimensions_length_cm IS NOT NULL AND dimensions_width_cm IS NOT NULL AND dimensions_height_cm IS NOT NULL 
            THEN (dimensions_length_cm * dimensions_width_cm * dimensions_height_cm) / 1000000
            ELSE NULL 
        END
    ) STORED,
    
    -- Content and Service
    content_description TEXT NOT NULL,
    content_value DECIMAL(12,2),
    content_category VARCHAR(50) DEFAULT 'general',
    service_type VARCHAR(30) DEFAULT 'standard' CHECK (service_type IN ('standard', 'express', 'overnight', 'economy')),
    delivery_instructions TEXT,
    special_handling JSONB DEFAULT '[]',
    
    -- Pricing
    declared_value DECIMAL(12,2),
    insurance_value DECIMAL(12,2),
    base_cost DECIMAL(10,2) NOT NULL DEFAULT 0.00,
    fuel_surcharge DECIMAL(10,2) DEFAULT 0.00,
    insurance_cost DECIMAL(10,2) DEFAULT 0.00,
    additional_fees DECIMAL(10,2) DEFAULT 0.00,
    total_cost DECIMAL(10,2) GENERATED ALWAYS AS (
        base_cost + COALESCE(fuel_surcharge, 0) + COALESCE(insurance_cost, 0) + COALESCE(additional_fees, 0)
    ) STORED,
    currency VARCHAR(3) DEFAULT 'USD',
    
    -- Status and Tracking
    status VARCHAR(30) DEFAULT 'created' CHECK (status IN (
        'created', 'accepted', 'collected', 'in_transit', 'at_hub', 
        'out_for_delivery', 'delivered', 'failed_delivery', 'returned', 
        'cancelled', 'lost', 'damaged'
    )),
    priority VARCHAR(20) DEFAULT 'normal' CHECK (priority IN ('low', 'normal', 'high', 'urgent')),
    
    -- Dates and Times
    pickup_date DATE,
    pickup_time_start TIME,
    pickup_time_end TIME,
    estimated_delivery_date DATE,
    estimated_delivery_time TIME,
    actual_pickup_datetime TIMESTAMP,
    actual_delivery_datetime TIMESTAMP,
    
    -- Assignment and Route
    assigned_route_id INTEGER REFERENCES routes(id),
    assigned_driver_id INTEGER REFERENCES users(id),
    assigned_vehicle VARCHAR(50),
    
    -- Metadata
    created_by_id INTEGER NOT NULL REFERENCES users(id),
    updated_by_id INTEGER REFERENCES users(id),
    notes TEXT,
    tags JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Shipment Events/Tracking History
CREATE TABLE IF NOT EXISTS shipment_events (
    id SERIAL PRIMARY KEY,
    shipment_id INTEGER NOT NULL REFERENCES shipments(id) ON DELETE CASCADE,
    event_type VARCHAR(50) NOT NULL CHECK (event_type IN (
        'created', 'accepted', 'collected', 'departed_hub', 'arrived_hub',
        'in_transit', 'out_for_delivery', 'delivered', 'failed_delivery',
        'returned', 'cancelled', 'damaged', 'lost', 'exception'
    )),
    status VARCHAR(30) NOT NULL,
    location VARCHAR(200),
    hub_id INTEGER REFERENCES hubs(id),
    description TEXT NOT NULL,
    notes TEXT,
    user_id INTEGER REFERENCES users(id),
    event_datetime TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- ANALYTICS AND REPORTING TABLES
-- =====================================================

-- Daily Analytics Summary
CREATE TABLE IF NOT EXISTS daily_analytics (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    hub_id INTEGER REFERENCES hubs(id),
    
    -- Shipment Metrics
    total_shipments INTEGER DEFAULT 0,
    completed_shipments INTEGER DEFAULT 0,
    pending_shipments INTEGER DEFAULT 0,
    failed_shipments INTEGER DEFAULT 0,
    
    -- Financial Metrics
    total_revenue DECIMAL(12,2) DEFAULT 0.00,
    total_costs DECIMAL(12,2) DEFAULT 0.00,
    net_profit DECIMAL(12,2) GENERATED ALWAYS AS (total_revenue - total_costs) STORED,
    
    -- Operational Metrics
    average_delivery_time_hours DECIMAL(8,2),
    on_time_delivery_rate DECIMAL(5,2),
    total_weight_processed_kg DECIMAL(12,2) DEFAULT 0.00,
    total_packages_processed INTEGER DEFAULT 0,
    
    -- Performance Metrics
    hub_utilization_rate DECIMAL(5,2),
    route_efficiency_score DECIMAL(5,2),
    customer_satisfaction_score DECIMAL(3,1),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(date, hub_id)
);

-- System Performance Metrics
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,4) NOT NULL,
    metric_unit VARCHAR(20),
    category VARCHAR(50) NOT NULL,
    subcategory VARCHAR(50),
    hub_id INTEGER REFERENCES hubs(id),
    user_id INTEGER REFERENCES users(id),
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- =====================================================
-- PROCESS MINING AND WORKFLOW TABLES
-- =====================================================

-- Process Definitions
CREATE TABLE IF NOT EXISTS process_definitions (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    version VARCHAR(20) NOT NULL,
    description TEXT,
    process_type VARCHAR(50) NOT NULL,
    definition_data JSONB NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_by_id INTEGER NOT NULL REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(name, version)
);

-- Process Instances
CREATE TABLE IF NOT EXISTS process_instances (
    id SERIAL PRIMARY KEY,
    process_definition_id INTEGER NOT NULL REFERENCES process_definitions(id),
    instance_name VARCHAR(200),
    status VARCHAR(30) DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed', 'suspended')),
    started_by_id INTEGER NOT NULL REFERENCES users(id),
    related_shipment_id INTEGER REFERENCES shipments(id),
    context_data JSONB DEFAULT '{}',
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT
);

-- Process Activities/Steps
CREATE TABLE IF NOT EXISTS process_activities (
    id SERIAL PRIMARY KEY,
    process_instance_id INTEGER NOT NULL REFERENCES process_instances(id) ON DELETE CASCADE,
    activity_name VARCHAR(100) NOT NULL,
    activity_type VARCHAR(50) NOT NULL,
    status VARCHAR(30) DEFAULT 'pending' CHECK (status IN ('pending', 'active', 'completed', 'failed', 'skipped')),
    assigned_to_id INTEGER REFERENCES users(id),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    duration_seconds INTEGER GENERATED ALWAYS AS (
        CASE 
            WHEN started_at IS NOT NULL AND completed_at IS NOT NULL 
            THEN EXTRACT(EPOCH FROM (completed_at - started_at))::INTEGER
            ELSE NULL 
        END
    ) STORED,
    input_data JSONB DEFAULT '{}',
    output_data JSONB DEFAULT '{}',
    error_message TEXT
);

-- =====================================================
-- CONFIGURATION AND SETTINGS TABLES
-- =====================================================

-- System Settings
CREATE TABLE IF NOT EXISTS system_settings (
    id SERIAL PRIMARY KEY,
    setting_key VARCHAR(100) UNIQUE NOT NULL,
    setting_value TEXT NOT NULL,
    setting_type VARCHAR(20) DEFAULT 'string' CHECK (setting_type IN ('string', 'number', 'boolean', 'json')),
    category VARCHAR(50) NOT NULL,
    description TEXT,
    is_editable BOOLEAN DEFAULT TRUE,
    is_sensitive BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User Sessions (for authentication)
CREATE TABLE IF NOT EXISTS user_sessions (
    id SERIAL PRIMARY KEY,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    ip_address INET,
    user_agent TEXT,
    expires_at TIMESTAMP NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Notifications
CREATE TABLE IF NOT EXISTS notifications (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(200) NOT NULL,
    message TEXT NOT NULL,
    type VARCHAR(30) DEFAULT 'info' CHECK (type IN ('info', 'success', 'warning', 'error')),
    category VARCHAR(50) DEFAULT 'general',
    is_read BOOLEAN DEFAULT FALSE,
    related_entity_type VARCHAR(50),
    related_entity_id INTEGER,
    action_url TEXT,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- INDEXES FOR PERFORMANCE OPTIMIZATION
-- =====================================================

-- Users indexes
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_users_status ON users(status);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);

-- Hubs indexes
CREATE INDEX IF NOT EXISTS idx_hubs_name ON hubs(name);
CREATE INDEX IF NOT EXISTS idx_hubs_code ON hubs(code);
CREATE INDEX IF NOT EXISTS idx_hubs_status ON hubs(status);
CREATE INDEX IF NOT EXISTS idx_hubs_city ON hubs(city);
CREATE INDEX IF NOT EXISTS idx_hubs_manager_id ON hubs(manager_id);

-- Routes indexes
CREATE INDEX IF NOT EXISTS idx_routes_origin_hub ON routes(origin_hub_id);
CREATE INDEX IF NOT EXISTS idx_routes_destination_hub ON routes(destination_hub_id);
CREATE INDEX IF NOT EXISTS idx_routes_status ON routes(status);
CREATE INDEX IF NOT EXISTS idx_routes_route_type ON routes(route_type);

-- Shipments indexes
CREATE INDEX IF NOT EXISTS idx_shipments_tracking_number ON shipments(tracking_number);
CREATE INDEX IF NOT EXISTS idx_shipments_status ON shipments(status);
CREATE INDEX IF NOT EXISTS idx_shipments_created_at ON shipments(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_shipments_sender_email ON shipments(sender_email);
CREATE INDEX IF NOT EXISTS idx_shipments_receiver_email ON shipments(receiver_email);
CREATE INDEX IF NOT EXISTS idx_shipments_origin_hub ON shipments(origin_hub_id);
CREATE INDEX IF NOT EXISTS idx_shipments_destination_hub ON shipments(destination_hub_id);
CREATE INDEX IF NOT EXISTS idx_shipments_current_hub ON shipments(current_hub_id);
CREATE INDEX IF NOT EXISTS idx_shipments_assigned_driver ON shipments(assigned_driver_id);
CREATE INDEX IF NOT EXISTS idx_shipments_service_type ON shipments(service_type);
CREATE INDEX IF NOT EXISTS idx_shipments_priority ON shipments(priority);
CREATE INDEX IF NOT EXISTS idx_shipments_estimated_delivery ON shipments(estimated_delivery_date);

-- Shipment Events indexes
CREATE INDEX IF NOT EXISTS idx_shipment_events_shipment_id ON shipment_events(shipment_id);
CREATE INDEX IF NOT EXISTS idx_shipment_events_event_type ON shipment_events(event_type);
CREATE INDEX IF NOT EXISTS idx_shipment_events_datetime ON shipment_events(event_datetime DESC);
CREATE INDEX IF NOT EXISTS idx_shipment_events_hub_id ON shipment_events(hub_id);

-- Analytics indexes
CREATE INDEX IF NOT EXISTS idx_daily_analytics_date ON daily_analytics(date DESC);
CREATE INDEX IF NOT EXISTS idx_daily_analytics_hub_id ON daily_analytics(hub_id);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_category ON performance_metrics(category);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_recorded_at ON performance_metrics(recorded_at DESC);

-- Process Mining indexes
CREATE INDEX IF NOT EXISTS idx_process_instances_definition_id ON process_instances(process_definition_id);
CREATE INDEX IF NOT EXISTS idx_process_instances_status ON process_instances(status);
CREATE INDEX IF NOT EXISTS idx_process_instances_started_at ON process_instances(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_process_activities_instance_id ON process_activities(process_instance_id);
CREATE INDEX IF NOT EXISTS idx_process_activities_status ON process_activities(status);

-- Notifications indexes
CREATE INDEX IF NOT EXISTS idx_notifications_user_id ON notifications(user_id);
CREATE INDEX IF NOT EXISTS idx_notifications_is_read ON notifications(is_read);
CREATE INDEX IF NOT EXISTS idx_notifications_created_at ON notifications(created_at DESC);

-- Sessions indexes
CREATE INDEX IF NOT EXISTS idx_user_sessions_token ON user_sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_expires_at ON user_sessions(expires_at);

-- =====================================================
-- VIEWS FOR COMMON QUERIES
-- =====================================================

-- Shipment Summary View
CREATE OR REPLACE VIEW shipment_summary AS
SELECT 
    s.id,
    s.uuid,
    s.tracking_number,
    s.sender_name,
    s.receiver_name,
    s.total_weight_kg,
    s.total_cost,
    s.status,
    s.service_type,
    s.priority,
    s.created_at,
    s.estimated_delivery_date,
    oh.name AS origin_hub_name,
    dh.name AS destination_hub_name,
    ch.name AS current_hub_name,
    u.full_name AS created_by_name
FROM shipments s
LEFT JOIN hubs oh ON s.origin_hub_id = oh.id
LEFT JOIN hubs dh ON s.destination_hub_id = dh.id
LEFT JOIN hubs ch ON s.current_hub_id = ch.id
LEFT JOIN users u ON s.created_by_id = u.id;

-- Hub Performance View
CREATE OR REPLACE VIEW hub_performance AS
SELECT 
    h.id,
    h.name,
    h.capacity,
    h.current_load,
    h.utilization_percentage,
    h.status,
    COUNT(s.id) AS total_shipments,
    COUNT(CASE WHEN s.status = 'delivered' THEN 1 END) AS delivered_shipments,
    AVG(s.total_cost) AS average_shipment_cost,
    SUM(s.total_cost) AS total_revenue
FROM hubs h
LEFT JOIN shipments s ON (h.id = s.origin_hub_id OR h.id = s.destination_hub_id)
WHERE s.created_at >= CURRENT_DATE - INTERVAL '30 days' OR s.id IS NULL
GROUP BY h.id, h.name, h.capacity, h.current_load, h.utilization_percentage, h.status;

-- Daily Summary View
CREATE OR REPLACE VIEW daily_summary AS
SELECT 
    DATE(s.created_at) as date,
    COUNT(*) as total_shipments,
    COUNT(CASE WHEN s.status = 'delivered' THEN 1 END) as delivered_shipments,
    COUNT(CASE WHEN s.status IN ('created', 'accepted', 'collected', 'in_transit', 'at_hub', 'out_for_delivery') THEN 1 END) as active_shipments,
    SUM(s.total_cost) as total_revenue,
    AVG(s.total_weight_kg) as average_weight,
    AVG(s.total_cost) as average_cost
FROM shipments s
WHERE s.created_at >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY DATE(s.created_at)
ORDER BY date DESC;

-- =====================================================
-- FUNCTIONS AND TRIGGERS
-- =====================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_hubs_updated_at BEFORE UPDATE ON hubs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_routes_updated_at BEFORE UPDATE ON routes
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_shipments_updated_at BEFORE UPDATE ON shipments
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_daily_analytics_updated_at BEFORE UPDATE ON daily_analytics
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to generate tracking numbers
CREATE OR REPLACE FUNCTION generate_tracking_number()
RETURNS TEXT AS $$
DECLARE
    new_tracking_number TEXT;
    counter INTEGER := 0;
BEGIN
    LOOP
        new_tracking_number := 'BC' || TO_CHAR(CURRENT_DATE, 'YYYYMMDD') || 
                              LPAD((EXTRACT(EPOCH FROM CURRENT_TIMESTAMP)::BIGINT % 10000)::TEXT, 4, '0') ||
                              LPAD(counter::TEXT, 3, '0');
        
        -- Check if tracking number already exists
        IF NOT EXISTS (SELECT 1 FROM shipments WHERE tracking_number = new_tracking_number) THEN
            RETURN new_tracking_number;
        END IF;
        
        counter := counter + 1;
        IF counter > 999 THEN
            -- If we've tried 1000 times, add random suffix
            new_tracking_number := new_tracking_number || LPAD((RANDOM() * 999)::INTEGER::TEXT, 3, '0');
            RETURN new_tracking_number;
        END IF;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Function to update hub utilization
CREATE OR REPLACE FUNCTION update_hub_utilization()
RETURNS TRIGGER AS $$
BEGIN
    -- Update current_load for origin hub (decrease)
    IF OLD.origin_hub_id IS NOT NULL THEN
        UPDATE hubs 
        SET current_load = GREATEST(0, current_load - OLD.package_count)
        WHERE id = OLD.origin_hub_id;
    END IF;
    
    -- Update current_load for destination hub (increase if delivered)
    IF NEW.status = 'delivered' AND NEW.destination_hub_id IS NOT NULL THEN
        UPDATE hubs 
        SET current_load = LEAST(capacity, current_load + NEW.package_count)
        WHERE id = NEW.destination_hub_id;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for hub utilization updates
CREATE TRIGGER update_hub_utilization_trigger 
    AFTER UPDATE ON shipments
    FOR EACH ROW 
    WHEN (OLD.status IS DISTINCT FROM NEW.status)
    EXECUTE FUNCTION update_hub_utilization();

-- =====================================================
-- INITIAL DATA SETUP
-- =====================================================

-- Insert default system settings
INSERT INTO system_settings (setting_key, setting_value, setting_type, category, description) VALUES
('app_name', 'BlueCart ERP', 'string', 'general', 'Application name'),
('app_version', '1.0.0', 'string', 'general', 'Application version'),
('currency_default', 'USD', 'string', 'financial', 'Default currency'),
('timezone_default', 'UTC', 'string', 'general', 'Default timezone'),
('weight_unit', 'kg', 'string', 'general', 'Default weight unit'),
('distance_unit', 'km', 'string', 'general', 'Default distance unit'),
('max_file_upload_size', '10485760', 'number', 'system', 'Maximum file upload size in bytes (10MB)'),
('session_timeout_minutes', '480', 'number', 'security', 'Session timeout in minutes (8 hours)'),
('password_min_length', '8', 'number', 'security', 'Minimum password length'),
('tracking_number_prefix', 'BC', 'string', 'shipments', 'Tracking number prefix'),
('notification_email_enabled', 'true', 'boolean', 'notifications', 'Enable email notifications'),
('analytics_retention_days', '365', 'number', 'analytics', 'Analytics data retention period in days')
ON CONFLICT (setting_key) DO NOTHING;

-- Insert sample admin user (password should be hashed in production)
INSERT INTO users (username, email, password_hash, full_name, role, status) VALUES
('admin', 'admin@bluecart.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LwEZgnFG7vDjHJfMe', 'System Administrator', 'admin', 'active')
ON CONFLICT (username) DO NOTHING;

-- Insert sample hubs
INSERT INTO hubs (name, code, address, city, state, country, postal_code, capacity, contact_person, contact_phone, contact_email) VALUES 
('New York Distribution Center', 'NYC01', '123 Main St', 'New York', 'NY', 'USA', '10001', 1500, 'John Smith', '+1-212-555-0101', 'nyc@bluecart.com'),
('Los Angeles Warehouse', 'LA001', '456 West Ave', 'Los Angeles', 'CA', 'USA', '90001', 1200, 'Maria Garcia', '+1-213-555-0102', 'la@bluecart.com'),
('Chicago Logistics Hub', 'CHI01', '789 North Blvd', 'Chicago', 'IL', 'USA', '60601', 1000, 'Robert Johnson', '+1-312-555-0103', 'chicago@bluecart.com'),
('Miami Distribution Point', 'MIA01', '321 South Dr', 'Miami', 'FL', 'USA', '33101', 800, 'Ana Rodriguez', '+1-305-555-0104', 'miami@bluecart.com')
ON CONFLICT (name) DO NOTHING;

-- Insert sample routes
INSERT INTO routes (name, code, origin_hub_id, destination_hub_id, distance_km, estimated_duration_minutes, cost_per_kg, base_cost, route_type) VALUES 
('NYC to LA Express', 'NYCLA1', 1, 2, 2445.5, 2880, 2.50, 150.00, 'ground'),
('LA to Chicago Direct', 'LACHI1', 2, 3, 1745.2, 2160, 2.25, 125.00, 'ground'),
('Chicago to NYC Standard', 'CHINYC', 3, 1, 790.8, 720, 2.00, 100.00, 'ground'),
('Miami to NYC Priority', 'MIANYC', 4, 1, 1280.0, 1440, 3.00, 175.00, 'air'),
('NYC to Miami Overnight', 'NYCMIA', 1, 4, 1280.0, 960, 4.50, 225.00, 'air')
ON CONFLICT (code) DO NOTHING;