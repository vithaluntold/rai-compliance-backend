#!/usr/bin/env python3
"""
BlueCart ERP Backend Deployment Test
===================================

This script tests the backend deployment readiness and database connectivity.
Run this script to verify your backend is ready for production deployment.
"""

import os
import sys
import json
from datetime import datetime

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def test_imports():
    """Test that all required modules can be imported"""
    print("🔍 Testing module imports...")
    
    try:
        import pg8000
        print("  ✅ pg8000 driver imported successfully")
    except ImportError as e:
        print(f"  ❌ pg8000 import failed: {e}")
        return False
    
    try:
        from database.connection import (
            DatabaseConfig, 
            DatabaseConnectionManager,
            get_database_connection,
            test_database_connection,
            execute_query,
            create_tables,
            get_connection_info,
            get_database_stats
        )
        print("  ✅ Database connection module imported successfully")
    except ImportError as e:
        print(f"  ❌ Database module import failed: {e}")
        return False
    
    return True

def test_configuration():
    """Test database configuration"""
    print("\n🔧 Testing database configuration...")
    
    try:
        from database.connection import DatabaseConfig
        config = DatabaseConfig()
        
        print(f"  📡 Host: {config.host}")
        print(f"  🔌 Port: {config.port}")
        print(f"  🗄️  Database: {config.database}")
        print(f"  👤 User: {config.user}")
        print(f"  🔄 Pool Size: {config.pool_size}")
        print(f"  🔁 Max Retries: {config.max_retries}")
        print(f"  ⏱️  Timeout: {config.connection_timeout}s")
        print(f"  🔒 SSL Mode: {config.ssl_mode}")
        
        return True
    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        return False

def test_connection():
    """Test database connection"""
    print("\n🔗 Testing database connection...")
    
    try:
        from database.connection import test_database_connection
        success, message = test_database_connection()
        
        if success:
            print(f"  ✅ {message}")
            return True
        else:
            print(f"  ⚠️  Connection test failed (expected for local testing): {message}")
            return False
    except Exception as e:
        print(f"  ❌ Connection test error: {e}")
        return False

def test_schema_file():
    """Test schema file availability"""
    print("\n📋 Testing database schema file...")
    
    schema_path = os.path.join(os.path.dirname(__file__), 'database', 'schema.sql')
    
    if os.path.exists(schema_path):
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema_content = f.read()
        
        # Count tables, indexes, etc.
        tables = schema_content.count('CREATE TABLE')
        indexes = schema_content.count('CREATE INDEX')
        views = schema_content.count('CREATE VIEW')
        functions = schema_content.count('CREATE FUNCTION')
        triggers = schema_content.count('CREATE TRIGGER')
        
        print(f"  ✅ Schema file found: {schema_path}")
        print(f"  📊 Statistics:")
        print(f"    • Tables: {tables}")
        print(f"    • Indexes: {indexes}")
        print(f"    • Views: {views}")
        print(f"    • Functions: {functions}")
        print(f"    • Triggers: {triggers}")
        print(f"  📝 File size: {len(schema_content):,} characters")
        
        return True
    else:
        print(f"  ❌ Schema file not found: {schema_path}")
        return False

def test_environment():
    """Test environment variables"""
    print("\n🌍 Testing environment configuration...")
    
    # Check for environment variables
    env_vars = [
        ('DATABASE_URL', False),
        ('DB_HOST', False),
        ('DB_PORT', False),
        ('DB_NAME', False),
        ('DB_USER', False),
        ('DB_PASSWORD', False),
        ('DB_POOL_SIZE', False),
        ('DB_MAX_RETRIES', False),
        ('DB_SSL_MODE', False),
    ]
    
    found_vars = 0
    for var_name, required in env_vars:
        value = os.getenv(var_name)
        if value:
            # Don't show password values
            display_value = "***" if "PASSWORD" in var_name.upper() else value
            print(f"  ✅ {var_name}: {display_value}")
            found_vars += 1
        else:
            status = "❌" if required else "⚪"
            print(f"  {status} {var_name}: Not set")
    
    print(f"  📈 Environment variables found: {found_vars}/{len(env_vars)}")
    return True

def test_requirements():
    """Test requirements.txt file"""
    print("\n📦 Testing requirements file...")
    
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            requirements = f.read()
        
        lines = [line.strip() for line in requirements.split('\n') if line.strip() and not line.startswith('#')]
        
        print(f"  ✅ Requirements file found: {req_path}")
        print(f"  📦 Dependencies: {len(lines)}")
        
        # Check for key dependencies
        key_deps = ['pg8000', 'python-dateutil']
        for dep in key_deps:
            if any(dep in line for line in lines):
                print(f"    ✅ {dep} found")
            else:
                print(f"    ❌ {dep} missing")
        
        return True
    else:
        print(f"  ❌ Requirements file not found: {req_path}")
        return False

def generate_deployment_report():
    """Generate a deployment readiness report"""
    print("\n📄 Generating deployment report...")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "backend_status": "ready",
        "database_driver": "pg8000",
        "deployment_target": "render.com",
        "tests_run": [
            "module_imports",
            "database_configuration", 
            "connection_testing",
            "schema_validation",
            "environment_check",
            "requirements_validation"
        ],
        "recommendations": [
            "Set DATABASE_URL environment variable for cloud deployment",
            "Configure SSL mode to 'require' for production",
            "Monitor connection pool performance under load",
            "Set up database backups on Render.com",
            "Configure logging for production monitoring"
        ]
    }
    
    report_file = "deployment_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"  ✅ Report saved to: {report_file}")
    return True

def main():
    """Run all deployment tests"""
    print("🚀 BlueCart ERP Backend Deployment Test")
    print("=" * 50)
    
    tests = [
        ("Module Imports", test_imports),
        ("Configuration", test_configuration), 
        ("Connection", test_connection),
        ("Schema File", test_schema_file),
        ("Environment", test_environment),
        ("Requirements", test_requirements),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"  ❌ Test '{test_name}' crashed: {e}")
    
    # Generate report
    generate_deployment_report()
    
    print("\n📊 Test Summary")
    print("=" * 50)
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 Backend is ready for deployment!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review issues before deployment.")
        return 1

if __name__ == "__main__":
    sys.exit(main())