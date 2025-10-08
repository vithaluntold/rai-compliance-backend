import { Pool } from 'pg'
import fs from 'fs'
import path from 'path'

// Connect to default postgres database first to create our database
const defaultPool = new Pool({
  user: 'postgres',
  host: 'localhost',
  database: 'postgres', // Connect to default database first
  password: 'root',
  port: 5432,
})

// Pool for our shipment_erp database
const shipmentPool = new Pool({
  user: 'postgres',
  host: 'localhost',
  database: 'shipment_erp',
  password: 'root',
  port: 5432,
})

async function setupDatabase() {
  console.log('🗄️ Setting up PostgreSQL database...')
  
  try {
    // Step 1: Create database if it doesn't exist
    console.log('1️⃣ Creating database if not exists...')
    
    const checkDbQuery = "SELECT 1 FROM pg_database WHERE datname = 'shipment_erp'"
    const dbExists = await defaultPool.query(checkDbQuery)
    
    if (dbExists.rows.length === 0) {
      await defaultPool.query('CREATE DATABASE shipment_erp')
      console.log('✅ Database "shipment_erp" created successfully!')
    } else {
      console.log('✅ Database "shipment_erp" already exists!')
    }
    
    await defaultPool.end()
    
    // Step 2: Create tables and schema
    console.log('2️⃣ Creating tables and schema...')
    
    const schemaSQL = `
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

      CREATE INDEX IF NOT EXISTS idx_shipments_tracking_number ON shipments(tracking_number);
      CREATE INDEX IF NOT EXISTS idx_shipments_status ON shipments(status);
      CREATE INDEX IF NOT EXISTS idx_shipments_created_at ON shipments(created_at);
      CREATE INDEX IF NOT EXISTS idx_shipments_sender_name ON shipments(sender_name);
      CREATE INDEX IF NOT EXISTS idx_shipments_receiver_name ON shipments(receiver_name);

      CREATE OR REPLACE FUNCTION update_updated_at_column()
      RETURNS TRIGGER AS $$
      BEGIN
          NEW.updated_at = NOW();
          RETURN NEW;
      END;
      $$ language 'plpgsql';

      DROP TRIGGER IF EXISTS update_shipments_updated_at ON shipments;
      CREATE TRIGGER update_shipments_updated_at
          BEFORE UPDATE ON shipments
          FOR EACH ROW
          EXECUTE FUNCTION update_updated_at_column();
    `
    
    await shipmentPool.query(schemaSQL)
    console.log('✅ Tables and schema created successfully!')
    
    // Step 3: Verify setup
    console.log('3️⃣ Verifying database setup...')
    
    const tableCheck = await shipmentPool.query(`
      SELECT table_name, column_name, data_type 
      FROM information_schema.columns 
      WHERE table_name = 'shipments' 
      ORDER BY ordinal_position
    `)
    
    console.log('✅ Table structure verified:')
    tableCheck.rows.forEach(col => {
      console.log(`   - ${col.column_name}: ${col.data_type}`)
    })
    
    // Step 4: Test connection
    const connectionTest = await shipmentPool.query('SELECT NOW() as current_time, COUNT(*) as shipment_count FROM shipments')
    console.log('✅ Database connection test successful!')
    console.log(`🕒 Current time: ${connectionTest.rows[0].current_time}`)
    console.log(`📦 Current shipments: ${connectionTest.rows[0].shipment_count}`)
    
    console.log('\n🎉 Database setup completed successfully!')
    console.log('🚀 You can now start the application with: npm run dev')
    
  } catch (error) {
    console.error('❌ Database setup failed:', error.message)
    
    if (error.code === 'ECONNREFUSED') {
      console.log('\n🔧 Connection refused. Please check:')
      console.log('1. PostgreSQL is running')
      console.log('2. Connection details are correct:')
      console.log('   - Host: localhost')
      console.log('   - Port: 5432')
      console.log('   - User: postgres')
      console.log('   - Password: root')
    }
  } finally {
    await shipmentPool.end()
  }
}

setupDatabase()