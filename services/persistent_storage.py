"""
Persistent Storage Service for Render Deployment
Stores files and analysis results in database to survive container restarts
"""
import asyncio
import base64
import json
import logging
import sqlite3
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Setup logging
logger = logging.getLogger(__name__)

class PersistentStorageManager:
    """
    Database-backed storage for uploaded files and analysis results
    Solves the ephemeral filesystem issue on Render.com
    """
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            # Use environment variable or default path
            db_path = os.environ.get('DATABASE_PATH', '/tmp/persistent_storage.db')
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
    
    def _initialize_database(self):
        """Create database tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS uploaded_files (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    content BLOB NOT NULL,
                    upload_date TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    mime_type TEXT,
                    file_hash TEXT
                );
                
                CREATE TABLE IF NOT EXISTS analysis_results (
                    document_id TEXT PRIMARY KEY,
                    results_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    processing_mode TEXT DEFAULT 'smart'
                );
                
                CREATE TABLE IF NOT EXISTS processing_locks (
                    document_id TEXT PRIMARY KEY,
                    lock_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_files_upload_date ON uploaded_files(upload_date);
                CREATE INDEX IF NOT EXISTS idx_results_status ON analysis_results(status);
                CREATE INDEX IF NOT EXISTS idx_locks_expires ON processing_locks(expires_at);
            """)
            logger.info(f"✅ Persistent storage database initialized at {self.db_path}")
    
    # FILE STORAGE METHODS
    async def save_uploaded_file(self, document_id: str, file_path: Path, filename: str, mime_type: str = None) -> bool:
        """Save uploaded file to database"""
        try:
            if not file_path.exists():
                logger.error(f"❌ File not found: {file_path}")
                return False
            
            # Read file content
            with open(file_path, 'rb') as f:
                content = f.read()
            
            file_size = len(content)
            file_hash = str(hash(content))  # Simple hash for deduplication
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO uploaded_files 
                    (document_id, filename, content, upload_date, file_size, mime_type, file_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    document_id, filename, content, datetime.now().isoformat(), 
                    file_size, mime_type, file_hash
                ))
                
            logger.info(f"✅ File saved to database: {document_id} ({filename}, {file_size} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save file {document_id}: {e}")
            return False
    
    async def get_uploaded_file(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve uploaded file from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT document_id, filename, content, upload_date, file_size, mime_type, file_hash
                    FROM uploaded_files WHERE document_id = ?
                """, (document_id,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'document_id': row['document_id'],
                        'filename': row['filename'],
                        'content': row['content'],
                        'upload_date': row['upload_date'],
                        'file_size': row['file_size'],
                        'mime_type': row['mime_type'],
                        'file_hash': row['file_hash']
                    }
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to retrieve file {document_id}: {e}")
            return None
    
    async def write_file_to_temp(self, document_id: str, temp_dir: Path = None) -> Optional[Path]:
        """Write database file to temporary location for processing"""
        try:
            file_data = await self.get_uploaded_file(document_id)
            if not file_data:
                return None
            
            if temp_dir is None:
                temp_dir = Path('/tmp/processing')
            
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine file extension from filename
            filename = file_data['filename']
            file_extension = Path(filename).suffix or '.pdf'
            
            temp_file = temp_dir / f"{document_id}{file_extension}"
            
            with open(temp_file, 'wb') as f:
                f.write(file_data['content'])
                
            logger.info(f"✅ Temporary file written: {temp_file}")
            return temp_file
            
        except Exception as e:
            logger.error(f"❌ Failed to write temp file {document_id}: {e}")
            return None
    
    # ANALYSIS RESULTS METHODS
    async def save_analysis_results(self, document_id: str, results: Dict[str, Any]) -> bool:
        """Save analysis results to database"""
        try:
            results_json = json.dumps(results, ensure_ascii=False, default=str)
            status = results.get('status', 'PROCESSING')
            processing_mode = results.get('processing_mode', 'smart')
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO analysis_results 
                    (document_id, results_json, status, created_at, updated_at, processing_mode)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    document_id, results_json, status,
                    datetime.now().isoformat(), datetime.now().isoformat(),
                    processing_mode
                ))
                
            logger.info(f"✅ Analysis results saved to database: {document_id} (status: {status})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save analysis results {document_id}: {e}")
            return False
    
    async def get_analysis_results(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve analysis results from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT results_json, status, created_at, updated_at, processing_mode
                    FROM analysis_results WHERE document_id = ?
                """, (document_id,))
                
                row = cursor.fetchone()
                if row:
                    results = json.loads(row['results_json'])
                    results.update({
                        'status': row['status'],
                        'created_at': row['created_at'],
                        'updated_at': row['updated_at'],
                        'processing_mode': row['processing_mode']
                    })
                    return results
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to retrieve analysis results {document_id}: {e}")
            return None
    
    # PROCESSING LOCKS METHODS  
    async def create_processing_lock(self, document_id: str, lock_type: str, expires_minutes: int = 60) -> bool:
        """Create processing lock to prevent concurrent processing"""
        try:
            expires_at = datetime.now().timestamp() + (expires_minutes * 60)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO processing_locks 
                    (document_id, lock_type, created_at, expires_at)
                    VALUES (?, ?, ?, ?)
                """, (document_id, lock_type, datetime.now().isoformat(), expires_at))
                
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create processing lock {document_id}: {e}")
            return False
    
    async def check_processing_lock(self, document_id: str, lock_type: str = None) -> bool:
        """Check if processing lock exists and is valid"""
        try:
            current_time = datetime.now().timestamp()
            
            with sqlite3.connect(self.db_path) as conn:
                if lock_type:
                    cursor = conn.execute("""
                        SELECT expires_at FROM processing_locks 
                        WHERE document_id = ? AND lock_type = ?
                    """, (document_id, lock_type))
                else:
                    cursor = conn.execute("""
                        SELECT expires_at FROM processing_locks WHERE document_id = ?
                    """, (document_id,))
                
                row = cursor.fetchone()
                if row and row[0] > current_time:
                    return True  # Lock exists and is valid
                
                # Clean up expired lock
                if row:
                    conn.execute("""
                        DELETE FROM processing_locks 
                        WHERE document_id = ? AND expires_at <= ?
                    """, (document_id, current_time))
                
                return False  # No valid lock
                
        except Exception as e:
            logger.error(f"❌ Failed to check processing lock {document_id}: {e}")
            return False
    
    async def release_processing_lock(self, document_id: str, lock_type: str = None) -> bool:
        """Release processing lock"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if lock_type:
                    conn.execute("""
                        DELETE FROM processing_locks 
                        WHERE document_id = ? AND lock_type = ?
                    """, (document_id, lock_type))
                else:
                    conn.execute("""
                        DELETE FROM processing_locks WHERE document_id = ?
                    """, (document_id,))
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to release processing lock {document_id}: {e}")
            return False

# Global instance
_storage_manager = None

def get_persistent_storage() -> PersistentStorageManager:
    """Get global persistent storage instance"""
    global _storage_manager
    if _storage_manager is None:
        _storage_manager = PersistentStorageManager()
    return _storage_manager

# Convenience functions for backward compatibility
async def save_file(document_id: str, file_path: Path, filename: str, mime_type: str = None) -> bool:
    """Save file using persistent storage"""
    storage = get_persistent_storage()
    return await storage.save_uploaded_file(document_id, file_path, filename, mime_type)

async def get_file_path(document_id: str) -> Optional[Path]:
    """Get temporary file path for processing"""
    storage = get_persistent_storage()
    return await storage.write_file_to_temp(document_id)

async def save_results(document_id: str, results: Dict[str, Any]) -> bool:
    """Save analysis results using persistent storage"""
    storage = get_persistent_storage()
    return await storage.save_analysis_results(document_id, results)

async def get_results(document_id: str) -> Optional[Dict[str, Any]]:
    """Get analysis results using persistent storage"""
    storage = get_persistent_storage()
    return await storage.get_analysis_results(document_id)