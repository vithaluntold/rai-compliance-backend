#!/usr/bin/env python3
"""
BlueCart ERP Database Connection Module

This module provides robust database connectivity for the BlueCart ERP system.
Features include:
- Connection pooling for improved performance
- Automatic reconnection and retry logic
- Render.com PostgreSQL compatibility
- Environment-based configuration
- Comprehensive error handling and logging
- Health monitoring and statistics

Usage:
    from backend.database.connection import get_database_connection, execute_query
    
    # Using context manager (recommended)
    with get_database_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users")
        results = cursor.fetchall()
    
    # Using helper function
    users = execute_query("SELECT * FROM users WHERE active = %s", (True,))
"""

import os
import sys
import time
import logging
import threading
from contextlib import contextmanager
from typing import Dict, Any, Optional, Tuple, List
from urllib.parse import urlparse
from dataclasses import dataclass

# Import pg8000 for PostgreSQL connectivity
try:
    import pg8000
    import pg8000.native
except ImportError:
    print("❌ pg8000 not found. Install with: pip install pg8000")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration with environment variable support"""
    
    def __init__(self):
        """Initialize database configuration from environment variables"""
        self.database_url = os.getenv("DATABASE_URL")
        
        if self.database_url:
            # Parse DATABASE_URL (used by Render.com and other cloud providers)
            parsed = urlparse(self.database_url)
            self.host = parsed.hostname
            self.port = parsed.port or 5432
            self.database = parsed.path[1:]  # Remove leading slash
            self.user = parsed.username
            self.password = parsed.password
        else:
            # Use individual environment variables
            self.host = os.getenv("DB_HOST", "localhost")
            self.port = int(os.getenv("DB_PORT", "5432"))
            self.database = os.getenv("DB_NAME", "bluecart_erp")
            self.user = os.getenv("DB_USER", "postgres")
            self.password = os.getenv("DB_PASSWORD", "")
        
        # Connection pool settings
        self.pool_size = int(os.getenv("DB_POOL_SIZE", "10"))
        self.max_retries = int(os.getenv("DB_MAX_RETRIES", "3"))
        self.retry_delay = float(os.getenv("DB_RETRY_DELAY", "1.0"))
        self.connection_timeout = int(os.getenv("DB_CONNECTION_TIMEOUT", "30"))
        
        # SSL settings for cloud deployments
        self.ssl_mode = os.getenv("DB_SSL_MODE", "prefer")
        
        # Validate required fields
        if not all([self.host, self.database, self.user]):
            raise ValueError("Missing required database configuration")
    
    def get_connection_params(self) -> Dict[str, Any]:
        """Get connection parameters for pg8000"""
        params = {
            'host': self.host,
            'port': self.port,
            'database': self.database,
            'user': self.user,
            'password': self.password,
            'timeout': self.connection_timeout
        }
        
        # Add SSL configuration for cloud databases
        if self.ssl_mode in ['require', 'prefer']:
            params['ssl_context'] = True
        
        return params
    
    def get_connection_string(self) -> str:
        """Get sanitized connection string for logging"""
        return f"postgresql://{self.user}:***@{self.host}:{self.port}/{self.database}"

class DatabaseConnectionManager:
    """Thread-safe database connection pool manager"""
    
    def __init__(self, config: DatabaseConfig = None):
        """Initialize connection manager with configuration"""
        self.config = config or DatabaseConfig()
        self._connection_pool = []
        self._pool_lock = threading.Lock()
        self._active_connections = 0
        self._max_connections = self.config.pool_size
        
        logger.info(f"Database connection manager initialized: {self.config.get_connection_string()}")
    
    def _create_connection(self) -> pg8000.Connection:
        """Create a new database connection with retry logic"""
        for attempt in range(self.config.max_retries + 1):
            try:
                params = self.config.get_connection_params()
                conn = pg8000.connect(**params)
                conn.autocommit = True
                
                # Test the connection
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                cursor.close()
                
                logger.debug(f"Database connection created successfully (attempt {attempt + 1})")
                return conn
                
            except Exception as e:
                if attempt == self.config.max_retries:
                    logger.error(f"Failed to create database connection after {self.config.max_retries + 1} attempts: {e}")
                    raise
                else:
                    logger.warning(f"Connection attempt {attempt + 1} failed: {e}. Retrying in {self.config.retry_delay}s...")
                    time.sleep(self.config.retry_delay)
    
    def get_connection(self) -> pg8000.Connection:
        """Get a connection from the pool or create a new one"""
        with self._pool_lock:
            # Try to get a connection from the pool
            while self._connection_pool:
                conn = self._connection_pool.pop()
                if self._is_connection_valid(conn):
                    self._active_connections += 1
                    return conn
                else:
                    # Connection is stale, close it
                    try:
                        conn.close()
                    except:
                        pass
            
            # No valid connections in pool, create new one if under limit
            if self._active_connections < self._max_connections:
                conn = self._create_connection()
                self._active_connections += 1
                return conn
            else:
                raise Exception(f"Connection pool exhausted (max: {self._max_connections})")
    
    def return_connection(self, conn: pg8000.Connection):
        """Return a connection to the pool"""
        if not conn:
            return
        
        with self._pool_lock:
            self._active_connections = max(0, self._active_connections - 1)
            
            # Check if connection is still valid and pool has space
            if self._is_connection_valid(conn) and len(self._connection_pool) < self._max_connections:
                self._connection_pool.append(conn)
            else:
                # Close connection if pool is full or connection is invalid
                try:
                    conn.close()
                except:
                    pass
    
    def _is_connection_valid(self, conn: pg8000.Connection) -> bool:
        """Check if a connection is still valid"""
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            return True
        except:
            return False
    
    def close_all_connections(self):
        """Close all connections in the pool"""
        with self._pool_lock:
            while self._connection_pool:
                conn = self._connection_pool.pop()
                try:
                    conn.close()
                except:
                    pass
            self._active_connections = 0
            logger.info("All database connections closed")
    
    def get_pool_stats(self) -> Dict[str, int]:
        """Get connection pool statistics"""
        with self._pool_lock:
            return {
                'active_connections': self._active_connections,
                'pooled_connections': len(self._connection_pool),
                'max_connections': self._max_connections
            }

# Global connection manager instance
_connection_manager = DatabaseConnectionManager()

@contextmanager
def get_database_connection():
    """
    Context manager for database connections with automatic cleanup.
    
    Usage:
        with get_database_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users")
            results = cursor.fetchall()
    """
    conn = None
    try:
        conn = _connection_manager.get_connection()
        yield conn
    except Exception as e:
        logger.error(f"Database operation failed: {e}")
        raise
    finally:
        if conn:
            _connection_manager.return_connection(conn)

def test_database_connection() -> Tuple[bool, str]:
    """
    Test database connection and return status.
    
    Returns:
        Tuple[bool, str]: (success, message)
    """
    try:
        with get_database_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT version()")
            version = cursor.fetchone()[0]
            cursor.close()
            
            message = f"Connection successful. PostgreSQL version: {version}"
            logger.info(message)
            return True, message
    except Exception as e:
        message = f"Connection failed: {str(e)}"
        logger.error(message)
        return False, message

def row_to_dict(cursor, row) -> Optional[Dict[str, Any]]:
    """
    Convert database row to dictionary using cursor description.
    
    Args:
        cursor: Database cursor with description
        row: Database row data
    
    Returns:
        Dictionary representation of the row or None
    """
    if not row:
        return None
    
    try:
        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, row))
    except Exception as e:
        logger.error(f"Failed to convert row to dict: {e}")
        return None

def execute_query(query: str, params: Optional[tuple] = None, fetch_one: bool = False, fetch_all: bool = True) -> Any:
    """
    Execute a database query with error handling.
    
    Args:
        query: SQL query string
        params: Query parameters tuple
        fetch_one: Whether to fetch only one result
        fetch_all: Whether to fetch all results
    
    Returns:
        Query results or None
    """
    try:
        with get_database_connection() as conn:
            cursor = conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            if fetch_one:
                result = cursor.fetchone()
                return row_to_dict(cursor, result) if result else None
            elif fetch_all:
                results = cursor.fetchall()
                return [row_to_dict(cursor, row) for row in results]
            else:
                return cursor.rowcount
            
    except Exception as e:
        logger.error(f"Query execution failed: {e}")
        logger.error(f"Query: {query}")
        logger.error(f"Params: {params}")
        raise

def create_tables():
    """
    Create database tables by executing the schema file.
    """
    try:
        # Get the directory of this script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        schema_file = os.path.join(current_dir, 'schema.sql')
        
        if not os.path.exists(schema_file):
            logger.error(f"Schema file not found: {schema_file}")
            return False
        
        logger.info("Creating database tables...")
        
        with open(schema_file, 'r', encoding='utf-8') as f:
            schema_sql = f.read()
        
        with get_database_connection() as conn:
            cursor = conn.cursor()
            
            # Split and execute SQL statements
            statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
            
            for i, statement in enumerate(statements):
                if statement.lower().startswith(('select', 'insert', 'update', 'delete', 'create', 'drop', 'alter')):
                    try:
                        cursor.execute(statement)
                        logger.debug(f"Executed statement {i+1}/{len(statements)}")
                    except Exception as e:
                        logger.warning(f"Statement {i+1} failed: {e}")
                        # Continue with other statements
            
            cursor.close()
        
        logger.info("Database tables created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        return False

def get_connection_info() -> Dict[str, Any]:
    """
    Get database connection information for debugging.
    
    Returns:
        Dictionary with connection details
    """
    config = DatabaseConfig()
    return {
        'host': config.host,
        'port': config.port,
        'database': config.database,
        'user': config.user,
        'connection_string': config.get_connection_string(),
        'max_retries': config.max_retries,
        'retry_delay': config.retry_delay,
        'connection_timeout': config.connection_timeout,
        'pool_size': config.pool_size
    }

def close_all_connections():
    """Close all database connections"""
    global _connection_manager
    _connection_manager.close_all_connections()

# Health check functions
def get_database_stats() -> Dict[str, Any]:
    """Get database statistics for monitoring"""
    try:
        with get_database_connection() as conn:
            cursor = conn.cursor()
            
            # Get database size
            cursor.execute("SELECT pg_size_pretty(pg_database_size(current_database()))")
            db_size = cursor.fetchone()[0]
            
            # Get connection count
            cursor.execute("SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()")
            connection_count = cursor.fetchone()[0]
            
            # Get table counts
            cursor.execute("""
                SELECT schemaname, tablename, n_tup_ins, n_tup_upd, n_tup_del
                FROM pg_stat_user_tables 
                WHERE schemaname = 'public'
                ORDER BY tablename
            """)
            table_stats = cursor.fetchall()
            
            cursor.close()
            
            return {
                'database_size': db_size,
                'active_connections': connection_count,
                'table_statistics': [row_to_dict(cursor, row) for row in table_stats],
                'pool_stats': _connection_manager.get_pool_stats()
            }
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        return {'error': str(e)}

def execute_script(script_path: str) -> bool:
    """
    Execute a SQL script file.
    
    Args:
        script_path: Path to the SQL script file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if not os.path.exists(script_path):
            logger.error(f"Script file not found: {script_path}")
            return False
        
        with open(script_path, 'r', encoding='utf-8') as f:
            script_sql = f.read()
        
        with get_database_connection() as conn:
            cursor = conn.cursor()
            
            # Execute the entire script
            cursor.execute(script_sql)
            cursor.close()
        
        logger.info(f"Script executed successfully: {script_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to execute script {script_path}: {e}")
        return False

if __name__ == "__main__":
    """Test the database connection when run directly"""
    print("Testing BlueCart ERP Database Connection...")
    print("=" * 50)
    
    # Test connection
    success, message = test_database_connection()
    print(f"Connection Test: {'✅ PASSED' if success else '❌ FAILED'}")
    print(f"Message: {message}")
    
    if success:
        # Show connection info
        print("\nConnection Information:")
        info = get_connection_info()
        for key, value in info.items():
            if key != 'password':  # Don't show password
                print(f"  {key}: {value}")
        
        # Test table creation
        print("\nTesting table creation...")
        if create_tables():
            print("✅ Tables created successfully")
        else:
            print("❌ Table creation failed")
        
        # Show database stats
        print("\nDatabase Statistics:")
        stats = get_database_stats()
        if 'error' not in stats:
            print(f"  Database Size: {stats['database_size']}")
            print(f"  Active Connections: {stats['active_connections']}")
            pool_stats = stats['pool_stats']
            print(f"  Pool Active: {pool_stats['active_connections']}")
            print(f"  Pool Available: {pool_stats['pooled_connections']}")
            print(f"  Pool Max: {pool_stats['max_connections']}")
        else:
            print(f"  Error: {stats['error']}")
    
    # Cleanup
    close_all_connections()
    print("\n✅ Database connection test completed")