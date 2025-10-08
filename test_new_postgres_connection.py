import asyncpg
import asyncio
import os

# Your new PostgreSQL connection details
DATABASE_URL = "postgresql://bluecart_admin:Suftlt22razRiPN9143GEpIt0WJdfKWe@dpg-d3ijvkje5dus73977f1g-a.oregon-postgres.render.com:5432/bluecart_erp"

async def test_database_connection():
    """Test connection to your PostgreSQL database"""
    print("🔗 Testing connection to your PostgreSQL database...")
    print(f"📍 Host: dpg-d3ijvkje5dus73977f1g-a.oregon-postgres.render.com")
    print(f"🗄️  Database: bluecart_erp")
    print(f"👤 User: bluecart_admin")
    print(f"🔐 SSL: Enabled")
    
    try:
        # Connect to database
        print("⏳ Connecting...")
        conn = await asyncpg.connect(DATABASE_URL)
        print("✅ Successfully connected to PostgreSQL!")
        
        # Test basic query
        version = await conn.fetchval('SELECT version()')
        print(f"📊 PostgreSQL Version: {version[:80]}...")
        
        # Check current database
        current_db = await conn.fetchval('SELECT current_database()')
        print(f"🗄️  Current Database: {current_db}")
        
        # List existing tables
        tables = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        
        if tables:
            print(f"📋 Existing tables ({len(tables)}):")
            for table in tables:
                print(f"   - {table['table_name']}")
        else:
            print("📋 No tables found in database")
        
        # Test creating a simple table
        print("🔨 Testing table creation...")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS connection_test (
                id SERIAL PRIMARY KEY,
                test_message VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print("✅ Table creation test successful!")
        
        # Insert test data
        await conn.execute("""
            INSERT INTO connection_test (test_message) 
            VALUES ('Connection test successful!')
        """)
        print("✅ Data insertion test successful!")
        
        # Read test data
        result = await conn.fetchval("""
            SELECT test_message FROM connection_test 
            ORDER BY created_at DESC LIMIT 1
        """)
        print(f"✅ Data retrieval test: {result}")
        
        # Clean up test table
        await conn.execute("DROP TABLE IF EXISTS connection_test")
        print("✅ Cleanup completed!")
        
        # Close connection
        await conn.close()
        print("🔒 Connection closed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_database_connection())
    if success:
        print("\n🎉 PostgreSQL connection is working perfectly!")
        print("🔧 Ready to update your application configuration.")
    else:
        print("\n💥 Connection failed. Please check your credentials.")