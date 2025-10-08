import asyncpg
import asyncio

# Your PostgreSQL connection details
DATABASE_URL = "postgresql://bluecart_admin:Suftlt22razRiPN9143GEpIt0WJdfKWe@dpg-d3ijvkje5dus73977f1g-a.oregon-postgres.render.com:5432/bluecart_erp"

async def migrate_hub_table():
    """Migrate the existing hubs table to match the new API structure"""
    print("🔧 Migrating hub table structure...")
    
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        print("✅ Connected to database")
        
        # Add missing columns to the existing hubs table
        migration_queries = [
            # Add code column
            "ALTER TABLE hubs ADD COLUMN IF NOT EXISTS code VARCHAR(50)",
            # Add other missing columns
            "ALTER TABLE hubs ADD COLUMN IF NOT EXISTS city VARCHAR(100)",
            "ALTER TABLE hubs ADD COLUMN IF NOT EXISTS state VARCHAR(100)",
            "ALTER TABLE hubs ADD COLUMN IF NOT EXISTS pincode VARCHAR(20)",
            "ALTER TABLE hubs ADD COLUMN IF NOT EXISTS phone VARCHAR(50)",
            "ALTER TABLE hubs ADD COLUMN IF NOT EXISTS manager VARCHAR(255)",
            # Add updated_at column
            "ALTER TABLE hubs ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            # Update id column to use VARCHAR instead of integer (if needed)
            "ALTER TABLE hubs ALTER COLUMN id TYPE VARCHAR(50)"
        ]
        
        for query in migration_queries:
            try:
                await conn.execute(query)
                print(f"✅ Executed: {query}")
            except Exception as e:
                print(f"⚠️  Warning for query '{query}': {e}")
        
        # Create unique constraint on code column (after adding it)
        try:
            await conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_hubs_code_unique ON hubs(code) WHERE code IS NOT NULL")
            print("✅ Created unique index on code column")
        except Exception as e:
            print(f"⚠️  Warning creating unique index: {e}")
        
        # Check the updated structure
        columns = await conn.fetch("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns 
            WHERE table_name = 'hubs' AND table_schema = 'public'
            ORDER BY ordinal_position
        """)
        
        print(f"📋 Updated hubs table structure ({len(columns)} columns):")
        for col in columns:
            nullable = "NULL" if col['is_nullable'] == 'YES' else "NOT NULL"
            print(f"   - {col['column_name']}: {col['data_type']} {nullable}")
        
        await conn.close()
        print("🔒 Connection closed")
        print("✅ Hub table migration completed!")
        
    except Exception as e:
        print(f"❌ Migration error: {e}")

if __name__ == "__main__":
    asyncio.run(migrate_hub_table())