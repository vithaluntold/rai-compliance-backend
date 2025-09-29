"""
Local Environment Setup for PostgreSQL Testing
This script helps you test PostgreSQL configuration locally before deploying to Render.
"""
import os

def setup_local_postgres_env():
    """Set up local environment variables for PostgreSQL testing."""
    print("🔧 Setting up Local PostgreSQL Test Environment...")
    
    # Example PostgreSQL connection string (replace with your actual details)
    # For local PostgreSQL: postgresql://username:password@localhost:5432/database_name
    # For Render PostgreSQL: Use the connection string from Render dashboard
    
    sample_postgres_url = "postgresql://rai_admin:your_password@localhost:5432/rai_compliance_db"
    
    print("\n📋 To test with PostgreSQL locally, you need:")
    print("1. PostgreSQL installed and running")
    print("2. A database created for testing")
    print("3. Connection string in this format:")
    print(f"   DATABASE_URL={sample_postgres_url}")
    
    print("\n🎯 For Render deployment:")
    print("1. Create PostgreSQL database on Render dashboard")
    print("2. Copy connection string from Render")
    print("3. Add to environment variables in render.yaml")
    
    # Set up for SQLite testing (works immediately)
    os.environ['ENVIRONMENT'] = 'development'
    os.environ['PORT'] = '8000'
    
    print("\n✅ Local environment configured for SQLite testing")
    print("   (PostgreSQL will be used automatically on Render)")

if __name__ == "__main__":
    setup_local_postgres_env()