"""
Persistent Storage Service for RAI Compliance Backend

This service provides SQLite-based persistent storage for files and analysis results,
ensuring data persists across container restarts in cloud deployments like Render.com.

Key Features:
- File storage and retrieval with binary data support
- Analysis results storage with JSON serialization
- Processing locks for distributed processing coordination
- Automatic database initialization and migration
- Thread-safe operations with connection pooling
"""

import asyncio
import json
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
import base64

# Configure logging
logger = logging.getLogger(__name__)

class PersistentStorageManager:
    """
    SQLite-based persistent storage manager for files and analysis results.
    Provides thread-safe operations and automatic schema management.
    """
    
    def __init__(self, db_path: str = "persistent_storage.db"):
        """Initialize the persistent storage manager."""
        self.db_path = Path(db_path)
        self.local_lock = threading.RLock()
        self._initialized = False
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database with required tables."""
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                conn.execute("PRAGMA foreign_keys = ON")
                conn.execute("PRAGMA journal_mode = WAL") 
                
                # Files table for document storage
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS files (
                        document_id TEXT PRIMARY KEY,
                        filename TEXT NOT NULL,
                        file_data BLOB NOT NULL,
                        mime_type TEXT,
                        upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        file_size INTEGER,
                        metadata TEXT DEFAULT '{}'
                    )
                """)
                
                # Analysis results table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS analysis_results (
                        document_id TEXT PRIMARY KEY,
                        results_json TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        version INTEGER DEFAULT 1
                    )
                """)
                
                # Processing locks table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS processing_locks (
                        document_id TEXT PRIMARY KEY,
                        lock_data TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP
                    )
                """)
                
                # Indexes for performance
                conn.execute("CREATE INDEX IF NOT EXISTS idx_files_upload_date ON files(upload_date)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_results_status ON analysis_results(status)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_locks_expires ON processing_locks(expires_at)")
                
                conn.commit()
                self._initialized = True
                logger.info("✅ Persistent storage database initialized successfully")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize persistent storage database: {str(e)}")
            raise
    
    async def store_file(self, document_id: str, file_path: Union[str, Path], 
                        filename: str, mime_type: str = "application/octet-stream") -> bool:
        """Store a file in persistent storage."""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"File not found for storage: {file_path}")
                return False
            
            # Read file data
            file_data = file_path.read_bytes()
            file_size = len(file_data)
            
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO files 
                    (document_id, filename, file_data, mime_type, file_size, upload_date)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (document_id, filename, file_data, mime_type, file_size, datetime.now().isoformat()))
                conn.commit()
            
            logger.info(f"✅ Stored file in persistent storage: {document_id} ({file_size} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store file {document_id}: {str(e)}")
            return False
    
    async def get_file(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a file from persistent storage."""
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.execute("""
                    SELECT filename, file_data, mime_type, file_size, upload_date, metadata
                    FROM files WHERE document_id = ?
                """, (document_id,))
                row = cursor.fetchone()
                
                if row:
                    filename, file_data, mime_type, file_size, upload_date, metadata_json = row
                    return {
                        'filename': filename,
                        'file_data': file_data,
                        'mime_type': mime_type,
                        'file_size': file_size,
                        'upload_date': upload_date,
                        'metadata': json.loads(metadata_json or '{}')
                    }
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to retrieve file {document_id}: {str(e)}")
            return None
    
    async def restore_file_to_filesystem(self, document_id: str, target_path: Union[str, Path]) -> bool:
        """Restore a file from persistent storage to filesystem."""
        try:
            file_info = await self.get_file(document_id)
            if not file_info:
                logger.warning(f"No file found in persistent storage for {document_id}")
                return False
            
            target_path = Path(target_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file data to filesystem
            target_path.write_bytes(file_info['file_data'])
            
            logger.info(f"✅ Restored file from persistent storage: {document_id} → {target_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to restore file {document_id}: {str(e)}")
            return False
    
    async def store_analysis_results(self, document_id: str, results: Dict[str, Any]) -> bool:
        """Store analysis results in persistent storage."""
        try:
            results_json = json.dumps(results, ensure_ascii=False, indent=2)
            status = results.get('status', 'UNKNOWN')
            
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO analysis_results 
                    (document_id, results_json, status, updated_at)
                    VALUES (?, ?, ?, ?)
                """, (document_id, results_json, status, datetime.now().isoformat()))
                conn.commit()
            
            logger.info(f"✅ Stored analysis results in persistent storage: {document_id} (status: {status})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store analysis results {document_id}: {str(e)}")
            return False
    
    async def get_analysis_results(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve analysis results from persistent storage."""
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.execute("""
                    SELECT results_json, status, updated_at 
                    FROM analysis_results WHERE document_id = ?
                """, (document_id,))
                row = cursor.fetchone()
                
                if row:
                    results_json, status, updated_at = row
                    results = json.loads(results_json)
                    # Add metadata about storage
                    results['_persistent_storage'] = {
                        'status': status,
                        'updated_at': updated_at,
                        'source': 'persistent_database'
                    }
                    return results
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to retrieve analysis results {document_id}: {str(e)}")
            return None
    
    async def set_processing_lock(self, document_id: str, lock_data: Dict[str, Any]) -> bool:
        """Set a processing lock for a document."""
        try:
            lock_json = json.dumps(lock_data, ensure_ascii=False)
            
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO processing_locks 
                    (document_id, lock_data, created_at)
                    VALUES (?, ?, ?)
                """, (document_id, lock_json, datetime.now().isoformat()))
                conn.commit()
            
            logger.info(f"🔐 Set processing lock in persistent storage: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to set processing lock {document_id}: {str(e)}")
            return False
    
    async def get_processing_lock(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get processing lock data for a document."""
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
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
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.execute("DELETE FROM processing_locks WHERE document_id = ?", (document_id,))
                conn.commit()
                removed = cursor.rowcount > 0
            
            if removed:
                logger.info(f"🔓 Removed processing lock from persistent storage: {document_id}")
            return removed
            
        except Exception as e:
            logger.error(f"❌ Failed to remove processing lock {document_id}: {str(e)}")
            return False
    
    def cleanup_expired_locks(self, max_age_hours: int = 24):
        """Clean up expired processing locks."""
        try:
            cutoff_time = datetime.now().replace(hour=datetime.now().hour - max_age_hours)
            
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.execute("""
                    DELETE FROM processing_locks 
                    WHERE created_at < ? OR expires_at < ?
                """, (cutoff_time.isoformat(), datetime.now().isoformat()))
                conn.commit()
                
                if cursor.rowcount > 0:
                    logger.info(f"🧹 Cleaned up {cursor.rowcount} expired processing locks")
                    
        except Exception as e:
            logger.error(f"❌ Failed to cleanup expired locks: {str(e)}")

# Global instance management
_storage_manager_instance = None
_storage_manager_lock = threading.Lock()

def get_persistent_storage_manager() -> PersistentStorageManager:
    """Get or create the global persistent storage manager instance."""
    global _storage_manager_instance
    
    if _storage_manager_instance is None:
        with _storage_manager_lock:
            if _storage_manager_instance is None:
                # Use a database file in the backend directory
                db_path = Path(__file__).parent.parent / "persistent_storage.db"
                _storage_manager_instance = PersistentStorageManager(str(db_path))
    
    return _storage_manager_instance

# Async helper functions for backward compatibility
async def store_file_persistent(document_id: str, file_path: Union[str, Path], 
                               filename: str, mime_type: str = "application/octet-stream") -> bool:
    """Helper function to store a file in persistent storage."""
    manager = get_persistent_storage_manager()
    return await manager.store_file(document_id, file_path, filename, mime_type)

async def get_file_persistent(document_id: str) -> Optional[Dict[str, Any]]:
    """Helper function to get a file from persistent storage."""
    manager = get_persistent_storage_manager()
    return await manager.get_file(document_id)

async def store_analysis_results_persistent(document_id: str, results: Dict[str, Any]) -> bool:
    """Helper function to store analysis results in persistent storage."""
    manager = get_persistent_storage_manager()
    return await manager.store_analysis_results(document_id, results)

async def get_analysis_results_persistent(document_id: str) -> Optional[Dict[str, Any]]:
    """Helper function to get analysis results from persistent storage."""
    manager = get_persistent_storage_manager()
    return await manager.get_analysis_results(document_id)