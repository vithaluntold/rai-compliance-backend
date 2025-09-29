"""
PostgreSQL Configuration Checker for Render Deployment
Run this script to verify your database setup is working correctly.
"""
import os
import sys
from datetime import datetime

def check_environment():
    """Check if all required environment variables are set."""
    print("🔍 Checking Environment Configuration...")
    
    # Check DATABASE_URL
    database_url = os.getenv('DATABASE_URL')
    if database_url:
        if database_url.startswith('postgresql://'):
            print("✅ DATABASE_URL: PostgreSQL connection string found")
            # Mask password for security
            masked_url = database_url.split('@')[0].split(':')[:-1]
            masked_url = ':'.join(masked_url) + ':****@' + database_url.split('@')[1]
            print(f"   URL: {masked_url}")
        else:
            print("⚠️  DATABASE_URL: Found but not PostgreSQL format")
            print(f"   Current: {database_url}")
    else:
        print("❌ DATABASE_URL: Not set - will use SQLite fallback")
    
    # Check environment type
    env = os.getenv('ENVIRONMENT', 'development')
    print(f"🌍 Environment: {env}")
    
    # Check port
    port = os.getenv('PORT', '8000')
    print(f"🔌 Port: {port}")
    
    return database_url is not None and database_url.startswith('postgresql://')

def test_postgresql_connection():
    """Test actual connection to PostgreSQL database."""
    print("\n🐘 Testing PostgreSQL Connection...")
    
    try:
        import psycopg2
        print("✅ psycopg2 driver imported successfully")
    except ImportError:
        print("❌ psycopg2 not installed - run: pip install psycopg2-binary")
        return False
    
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        print("❌ No DATABASE_URL found for testing")
        return False
    
    try:
        # Test connection
        conn = psycopg2.connect(database_url)
        cursor = conn.cursor()
        
        # Test basic query
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        print(f"✅ Connected to PostgreSQL: {version.split(',')[0]}")
        
        # Check if our tables exist
        cursor.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_name IN ('files', 'analysis_results', 'processing_locks');
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        if tables:
            print(f"✅ Found existing tables: {', '.join(tables)}")
        else:
            print("⚠️  No application tables found - will be created on first run")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        return False

def test_persistent_storage():
    """Test the persistent storage system."""
    print("\n💾 Testing Persistent Storage System...")
    
    try:
        # Import our enhanced storage system
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from services.persistent_storage_enhanced import PersistentStorageManager
        
        # Initialize storage
        storage = PersistentStorageManager()
        print(f"✅ Storage system initialized: {storage.db_type}")
        
        # Test table creation
        storage.initialize_tables()
        print("✅ Database tables verified/created")
        
        # Test document storage (mock test)
        test_doc_id = f"TEST-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        test_result = {
            "document_id": test_doc_id,
            "status": "COMPLETED",
            "metadata": {
                "company_name": {"value": "Test Company", "confidence": 0.95}
            },
            "sections": [{"title": "Test Section", "content": "Test content"}]
        }
        
        success = storage.store_analysis_result(test_doc_id, test_result)
        if success:
            print(f"✅ Test document stored: {test_doc_id}")
            
            # Test retrieval
            retrieved = storage.get_analysis_result(test_doc_id)
            if retrieved and retrieved.get('status') == 'COMPLETED':
                print("✅ Test document retrieved successfully")
                return True
            else:
                print("❌ Document retrieval failed")
                return False
        else:
            print("❌ Document storage failed")
            return False
            
    except Exception as e:
        print(f"❌ Storage system error: {str(e)}")
        return False

def main():
    """Run all configuration checks."""
    print("🚀 PostgreSQL Configuration Checker for Render Deployment\n")
    print("=" * 60)
    
    # Run all checks
    env_ok = check_environment()
    db_ok = test_postgresql_connection() if env_ok else False
    storage_ok = test_persistent_storage()
    
    print("\n" + "=" * 60)
    print("📋 SUMMARY:")
    print(f"   Environment Config: {'✅ PASS' if env_ok else '❌ FAIL'}")
    print(f"   PostgreSQL Connection: {'✅ PASS' if db_ok else '❌ FAIL'}")
    print(f"   Persistent Storage: {'✅ PASS' if storage_ok else '❌ FAIL'}")
    
    if all([env_ok, db_ok, storage_ok]):
        print("\n🎉 ALL CHECKS PASSED - Ready for Production Deployment!")
        print("\n📌 Next Steps:")
        print("   1. Push your code to GitHub")
        print("   2. Deploy to Render via dashboard")
        print("   3. Monitor deployment logs for successful startup")
    else:
        print("\n⚠️  ISSUES FOUND - Fix the failed checks above")
        print("\n🔧 Common Solutions:")
        if not env_ok:
            print("   - Set DATABASE_URL environment variable")
        if not db_ok:
            print("   - Check PostgreSQL database is running on Render")
            print("   - Verify connection string is correct")
        if not storage_ok:
            print("   - Check persistent_storage_enhanced.py is present")
            print("   - Ensure all dependencies are installed")

if __name__ == "__main__":
    main()