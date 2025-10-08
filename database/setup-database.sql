-- Connect to PostgreSQL and create the shipment_erp database
-- Run this script in pgAdmin or psql command line

-- Create database (run this first)
CREATE DATABASE shipment_erp;

-- Connect to the shipment_erp database and run the rest

-- Create shipments table
CREATE TABLE IF NOT EXISTS shipments (
    id VARCHAR(20) PRIMARY KEY,
    tracking_number VARCHAR(20) UNIQUE NOT NULL,
    sender_name VARCHAR(255) NOT NULL,
    sender_phone VARCHAR(20),
    sender_address TEXT NOT NULL,
    receiver_name VARCHAR(255) NOT NULL,
    receiver_phone VARCHAR(20),
    receiver_address TEXT NOT NULL,
    package_details TEXT NOT NULL,
    weight DECIMAL(10,2) NOT NULL,
    dimensions JSONB NOT NULL,
    service_type VARCHAR(20) NOT NULL CHECK (service_type IN ('standard', 'express', 'overnight')),
    status VARCHAR(30) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'picked_up', 'in_transit', 'out_for_delivery', 'delivered', 'failed')),
    pickup_date TIMESTAMP WITH TIME ZONE,
    estimated_delivery TIMESTAMP WITH TIME ZONE,
    actual_delivery TIMESTAMP WITH TIME ZONE,
    route VARCHAR(255),
    hub_id VARCHAR(50),
    events JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    cost DECIMAL(10,2) NOT NULL
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_shipments_tracking_number ON shipments(tracking_number);
CREATE INDEX IF NOT EXISTS idx_shipments_status ON shipments(status);
CREATE INDEX IF NOT EXISTS idx_shipments_created_at ON shipments(created_at);
CREATE INDEX IF NOT EXISTS idx_shipments_sender_name ON shipments(sender_name);
CREATE INDEX IF NOT EXISTS idx_shipments_receiver_name ON shipments(receiver_name);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to automatically update updated_at
DROP TRIGGER IF EXISTS update_shipments_updated_at ON shipments;
CREATE TRIGGER update_shipments_updated_at
    BEFORE UPDATE ON shipments
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Verify table creation
SELECT 'Table created successfully!' as status;
SELECT COUNT(*) as initial_count FROM shipments;