import asyncpg
import asyncio

# Your PostgreSQL connection details
DATABASE_URL = "postgresql://bluecart_admin:Suftlt22razRiPN9143GEpIt0WJdfKWe@dpg-d3ijvkje5dus73977f1g-a.oregon-postgres.render.com:5432/bluecart_erp"

async def check_hub_table_structure():
    """Check the current structure of the hubs table"""
    print("🔍 Checking hub table structure...")
    
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        print("✅ Connected to database")
        
        # Check if hubs table exists and its structure
        columns = await conn.fetch("""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns 
            WHERE table_name = 'hubs' AND table_schema = 'public'
            ORDER BY ordinal_position
        """)
        
        if columns:
            print(f"📋 Hubs table structure ({len(columns)} columns):")
            for col in columns:
                nullable = "NULL" if col['is_nullable'] == 'YES' else "NOT NULL"
                default = f" DEFAULT {col['column_default']}" if col['column_default'] else ""
                print(f"   - {col['column_name']}: {col['data_type']} {nullable}{default}")
        else:
            print("❌ Hubs table not found")
        
        # Check existing data
        count = await conn.fetchval("SELECT COUNT(*) FROM hubs")
        print(f"📊 Existing hub records: {count}")
        
        if count > 0:
            sample = await conn.fetch("SELECT * FROM hubs LIMIT 3")
            print("📋 Sample hub data:")
            for hub in sample:
                print(f"   - ID: {hub.get('id', 'N/A')}, Name: {hub.get('name', 'N/A')}")
        
        await conn.close()
        print("🔒 Connection closed")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(check_hub_table_structure())