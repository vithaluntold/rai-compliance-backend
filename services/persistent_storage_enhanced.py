"""
Enhanced Persistent Storage Service with PostgreSQL Support

This service provides dual database support:
- SQLite for local development
- PostgreSQL for production on Render

Environment Variables:
- DATABASE_URL: Full PostgreSQL connection string (production)
- USE_POSTGRESQL: Set to "true" to use PostgreSQL instead of SQLite
"""

import asyncio
import json
import logging
import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
import base64

# Set up logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import PostgreSQL adapter
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    logger.warning("PostgreSQL support not available. Install psycopg2-binary for production deployment.")
    logger.warning("PostgreSQL support not available. Install psycopg2-binary for production deployment.")

logger = logging.getLogger(__name__)

class PersistentStorageManager:
    """
    Multi-database persistent storage manager.
    
    Supports:
    - SQLite: For local development and testing
    - PostgreSQL: For production deployment on Render
    """
    
    def __init__(self, db_path: str = "persistent_storage.db"):
        self.use_postgresql = os.getenv("USE_POSTGRESQL", "false").lower() == "true"
        self.database_url = os.getenv("DATABASE_URL")
        self.sqlite_path = db_path
        self.lock = threading.RLock()
        
        if self.use_postgresql and POSTGRES_AVAILABLE and self.database_url:
            self.db_type = "PostgreSQL"
            logger.info("🐘 Initializing PostgreSQL storage for production")
            self._init_postgresql()
        else:
            self.db_type = "SQLite"
            logger.info("📁 Initializing SQLite storage for development")
            self._init_sqlite()
    
    def _init_postgresql(self):
        """Initialize PostgreSQL database and tables"""
        try:
            with psycopg2.connect(self.database_url) as conn:
                with conn.cursor() as cursor:
                    # Create files table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS files (
                            document_id VARCHAR(255) PRIMARY KEY,
                            filename VARCHAR(255) NOT NULL,
                            content BYTEA NOT NULL,
                            mime_type VARCHAR(255),
                            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            stored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            file_size INTEGER,
                            metadata TEXT DEFAULT '{}'
                        )
                    """)
                    
                    # Create analysis results table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS analysis_results (
                            document_id VARCHAR(255) PRIMARY KEY,
                            results JSONB NOT NULL,
                            stored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    
                    # Create processing locks table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS processing_locks (
                            document_id VARCHAR(255) PRIMARY KEY,
                            lock_type VARCHAR(100) NOT NULL,
                            lock_data TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    
                    # Migration: Add lock_data column if it doesn't exist
                    try:
                        cursor.execute("ALTER TABLE processing_locks ADD COLUMN lock_data TEXT")
                        logger.info("🔄 Added missing lock_data column to processing_locks table")
                    except Exception:
                        # Column already exists, ignore
                        pass
                    
                    conn.commit()
                    logger.info("✅ PostgreSQL tables initialized successfully")
                    
        except Exception as e:
            logger.error(f"❌ Failed to initialize PostgreSQL: {e}")
            raise
    
    def _init_sqlite(self):
        """Initialize SQLite database and tables (existing implementation)"""
        try:
            db_dir = Path(self.sqlite_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(self.sqlite_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS files (
                        document_id TEXT PRIMARY KEY,
                        filename TEXT NOT NULL,
                        content BLOB NOT NULL,
                        mime_type TEXT,
                        upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        stored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        file_size INTEGER,
                        metadata TEXT DEFAULT '{}'
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS analysis_results (
                        document_id TEXT PRIMARY KEY,
                        results TEXT NOT NULL,
                        stored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS processing_locks (
                        document_id TEXT PRIMARY KEY,
                        lock_type TEXT NOT NULL,
                        lock_data TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Migration: Add lock_data column if it doesn't exist
                try:
                    conn.execute("ALTER TABLE processing_locks ADD COLUMN lock_data TEXT")
                    logger.info("🔄 Added missing lock_data column to processing_locks table")
                except Exception:
                    # Column already exists, ignore
                    pass
                
                conn.commit()
                logger.info("✅ SQLite database initialized successfully")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize SQLite: {e}")
            raise
    
    async def store_file(self, document_id: str, filename: str, content: bytes) -> bool:
        """Store a file in the database"""
        try:
            if self.use_postgresql and POSTGRES_AVAILABLE and self.database_url:
                return await self._store_file_postgresql(document_id, filename, content)
            else:
                return await self._store_file_sqlite(document_id, filename, content)
        except Exception as e:
            logger.error(f"Failed to store file {document_id}: {e}")
            return False
    
    async def _store_file_postgresql(self, document_id: str, filename: str, content: bytes) -> bool:
        """Store file in PostgreSQL"""
        with psycopg2.connect(self.database_url) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO files (document_id, filename, content) VALUES (%s, %s, %s) "
                    "ON CONFLICT (document_id) DO UPDATE SET filename = %s, content = %s, stored_at = CURRENT_TIMESTAMP",
                    (document_id, filename, content, filename, content)
                )
                conn.commit()
                return True
    
    async def _store_file_sqlite(self, document_id: str, filename: str, content: bytes) -> bool:
        """Store file in SQLite"""
        with self.lock:
            with sqlite3.connect(self.sqlite_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO files (document_id, filename, content) VALUES (?, ?, ?)",
                    (document_id, filename, content)
                )
                conn.commit()
                return True
    
    async def get_file(self, document_id: str) -> Optional[Dict[str, Union[str, bytes]]]:
        """Retrieve a file from the database"""
        try:
            if self.use_postgresql and POSTGRES_AVAILABLE and self.database_url:
                return await self._get_file_postgresql(document_id)
            else:
                return await self._get_file_sqlite(document_id)
        except Exception as e:
            logger.error(f"Failed to retrieve file {document_id}: {e}")
            return None
    
    async def _get_file_postgresql(self, document_id: str) -> Optional[Dict[str, Union[str, bytes]]]:
        """Get file from PostgreSQL"""
        with psycopg2.connect(self.database_url) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT filename, content FROM files WHERE document_id = %s", (document_id,))
                result = cursor.fetchone()
                
                if result:
                    return {
                        "filename": result["filename"],
                        "content": bytes(result["content"])
                    }
                return None
    
    async def _get_file_sqlite(self, document_id: str) -> Optional[Dict[str, Union[str, bytes]]]:
        """Get file from SQLite"""
        with self.lock:
            with sqlite3.connect(self.sqlite_path) as conn:
                cursor = conn.execute("SELECT filename, content FROM files WHERE document_id = ?", (document_id,))
                result = cursor.fetchone()
                
                if result:
                    return {
                        "filename": result[0],
                        "content": result[1]
                    }
                return None
    
    async def store_analysis_results(self, document_id: str, results: Dict[str, Any]) -> bool:
        """Store analysis results in the database"""
        try:
            if self.use_postgresql and POSTGRES_AVAILABLE and self.database_url:
                return await self._store_analysis_postgresql(document_id, results)
            else:
                return await self._store_analysis_sqlite(document_id, results)
        except Exception as e:
            logger.error(f"Failed to store analysis results {document_id}: {e}")
            return False
    
    async def _store_analysis_postgresql(self, document_id: str, results: Dict[str, Any]) -> bool:
        """Store analysis in PostgreSQL using JSONB"""
        with psycopg2.connect(self.database_url) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO analysis_results (document_id, results) VALUES (%s, %s) "
                    "ON CONFLICT (document_id) DO UPDATE SET results = %s, stored_at = CURRENT_TIMESTAMP",
                    (document_id, json.dumps(results), json.dumps(results))
                )
                conn.commit()
                return True
    
    async def _store_analysis_sqlite(self, document_id: str, results: Dict[str, Any]) -> bool:
        """Store analysis in SQLite as JSON text"""
        with self.lock:
            with sqlite3.connect(self.sqlite_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO analysis_results (document_id, results) VALUES (?, ?)",
                    (document_id, json.dumps(results))
                )
                conn.commit()
                return True
    
    async def get_analysis_results(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve analysis results from the database"""
        try:
            if self.use_postgresql and POSTGRES_AVAILABLE and self.database_url:
                return await self._get_analysis_postgresql(document_id)
            else:
                return await self._get_analysis_sqlite(document_id)
        except Exception as e:
            logger.error(f"Failed to retrieve analysis results {document_id}: {e}")
            return None
    
    async def _get_analysis_postgresql(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get analysis from PostgreSQL"""
        with psycopg2.connect(self.database_url) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT results FROM analysis_results WHERE document_id = %s", (document_id,))
                result = cursor.fetchone()
                
                if result:
                    return result["results"] if isinstance(result["results"], dict) else json.loads(result["results"])
                return None
    
    async def _get_analysis_sqlite(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get analysis from SQLite"""
        with self.lock:
            with sqlite3.connect(self.sqlite_path) as conn:
                cursor = conn.execute("SELECT results FROM analysis_results WHERE document_id = ?", (document_id,))
                result = cursor.fetchone()
                
                if result:
                    return json.loads(result[0])
                return None
    
    def save_document_audit_log(self, document_id: str, audit_data: Dict[str, Any]) -> None:
        """Save audit log for document operations"""
        if self.use_postgresql and POSTGRES_AVAILABLE:
            self._save_audit_log_postgresql(document_id, audit_data)
        else:
            self._save_audit_log_sqlite(document_id, audit_data)
    
    def _save_audit_log_postgresql(self, document_id: str, audit_data: Dict[str, Any]) -> None:
        """Save audit log to PostgreSQL"""
        try:
            conn = psycopg2.connect(self.database_url)
            with conn:
                with conn.cursor() as cursor:
                    # Create audit table if not exists
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS document_audit_logs (
                            id SERIAL PRIMARY KEY,
                            document_id TEXT NOT NULL,
                            audit_data JSONB NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    
                    # Insert audit log
                    cursor.execute(
                        "INSERT INTO document_audit_logs (document_id, audit_data) VALUES (%s, %s)",
                        (document_id, json.dumps(audit_data))
                    )
        except Exception as e:
            logger.error(f"Failed to save audit log to PostgreSQL: {e}")
    
    def _save_audit_log_sqlite(self, document_id: str, audit_data: Dict[str, Any]) -> None:
        """Save audit log to SQLite"""
        with self.lock:
            with sqlite3.connect(self.sqlite_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS document_audit_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        document_id TEXT NOT NULL,
                        audit_data TEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.execute(
                    "INSERT INTO document_audit_logs (document_id, audit_data) VALUES (?, ?)",
                    (document_id, json.dumps(audit_data))
                )
    
    async def set_processing_lock(self, document_id: str, lock_data: Dict[str, Any], lock_type: str = "compliance_analysis") -> bool:
        """Set a processing lock for a document."""
        try:
            lock_json = json.dumps(lock_data)
            
            if self.use_postgresql and self.postgres_conn:
                with self.postgres_conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO processing_locks (document_id, lock_type, lock_data, created_at)
                        VALUES (%s, %s, %s, NOW())
                        ON CONFLICT (document_id) DO UPDATE SET 
                            lock_type = EXCLUDED.lock_type,
                            lock_data = EXCLUDED.lock_data,
                            created_at = NOW()
                    """, (document_id, lock_type, lock_json))
                    self.postgres_conn.commit()
            else:
                with self.lock:
                    with sqlite3.connect(self.sqlite_path) as conn:
                        conn.execute("""
                            INSERT OR REPLACE INTO processing_locks 
                            (document_id, lock_type, lock_data, created_at) 
                            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                        """, (document_id, lock_type, lock_json))
            
            logger.info(f"🔒 Set processing lock in persistent storage: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to set processing lock {document_id}: {str(e)}")
            return False
    
    async def get_processing_lock(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get processing lock data for a document."""
        try:
            if self.use_postgresql and self.postgres_conn:
                with self.postgres_conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT lock_data, created_at FROM processing_locks 
                        WHERE document_id = %s
                    """, (document_id,))
                    row = cursor.fetchone()
            else:
                with self.lock:
                    with sqlite3.connect(self.sqlite_path) as conn:
                        cursor = conn.execute("""
                            SELECT lock_data, created_at FROM processing_locks 
                            WHERE document_id = ?
                        """, (document_id,))
                        row = cursor.fetchone()
            
            if row:
                lock_json, created_at = row
                lock_data = json.loads(lock_json)
                lock_data['_created_at'] = created_at
                return lock_data
            return None
                
        except Exception as e:
            logger.error(f"❌ Failed to get processing lock {document_id}: {str(e)}")
            return None
    
    async def remove_processing_lock(self, document_id: str) -> bool:
        """Remove a processing lock for a document."""
        try:
            if self.use_postgresql and self.postgres_conn:
                with self.postgres_conn.cursor() as cursor:
                    cursor.execute("DELETE FROM processing_locks WHERE document_id = %s", (document_id,))
                    removed = cursor.rowcount > 0
                    self.postgres_conn.commit()
            else:
                with self.lock:
                    with sqlite3.connect(self.sqlite_path) as conn:
                        cursor = conn.execute("DELETE FROM processing_locks WHERE document_id = ?", (document_id,))
                        conn.commit()
                        removed = cursor.rowcount > 0
            
            if removed:
                logger.info(f"🔓 Removed processing lock from persistent storage: {document_id}")
            return removed
            
        except Exception as e:
            logger.error(f"❌ Failed to remove processing lock {document_id}: {str(e)}")
            return False

# Global instance
_storage_manager = None
_storage_lock = threading.RLock()

def get_persistent_storage_manager() -> PersistentStorageManager:
    """Get the global persistent storage manager instance"""
    global _storage_manager
    
    with _storage_lock:
        if _storage_manager is None:
            _storage_manager = PersistentStorageManager()
        
        return _storage_manager

# Alias for compatibility with old imports
def get_persistent_storage() -> PersistentStorageManager:
    """Alias for get_persistent_storage_manager for backward compatibility"""
    return get_persistent_storage_manager()