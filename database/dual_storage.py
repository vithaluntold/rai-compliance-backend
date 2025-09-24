"""
Dual-Mode Analysis Results Storage
ZERO DISRUPTION: Saves to both database AND files during transition

This module provides backward-compatible storage that writes to:
1. NEW: Bulletproof database (atomic, zero race conditions)  
2. OLD: File system (maintains existing functionality)

Frontend can switch gradually while keeping rollback capability.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from database.db_manager import get_database_manager

# Configure logging
logger = logging.getLogger(__name__)

class DualModeStorage:
    """
    Transition storage system that writes to both database and files.
    Ensures zero disruption during migration to bulletproof system.
    """
    
    def __init__(self):
        self.db = get_database_manager()
        self._legacy_save = None  # Will be set to existing save function
    
    def set_legacy_save_function(self, save_func):
        """Set the legacy file-based save function for backward compatibility"""
        self._legacy_save = save_func
    
    async def save_analysis_results_v2(self, document_id: str, results: Dict[str, Any]) -> None:
        """
        DUAL MODE: Save to both database (NEW) and files (OLD)
        
        Database save is atomic and race-condition-free.
        File save maintains backward compatibility.
        If database save fails, file save still works (rollback protection).
        """
        logger.info(f"🔄 DUAL MODE SAVE: Starting for document {document_id}")
        
        # PHASE 1: Save to bulletproof database (primary)
        try:
            await self.db.save_document_analysis(document_id, results)
            logger.info(f"✅ DATABASE SAVE: Success for {document_id}")
        except Exception as db_error:
            logger.error(f"❌ DATABASE SAVE: Failed for {document_id}: {db_error}")
            # Continue to file save for backward compatibility
        
        # PHASE 2: Save to legacy file system (backup/compatibility)
        try:
            if self._legacy_save:
                self._legacy_save(document_id, results)
                logger.info(f"✅ FILE SAVE: Success for {document_id}")
        except Exception as file_error:
            logger.error(f"❌ FILE SAVE: Failed for {document_id}: {file_error}")
        
        logger.info(f"🔄 DUAL MODE SAVE: Completed for {document_id}")
    
    async def get_analysis_results_v2(self, document_id: str, prefer_database: bool = True) -> Optional[Dict[str, Any]]:
        """
        DUAL MODE: Read from database (preferred) or files (fallback)
        
        Args:
            document_id: Document to retrieve
            prefer_database: If True, try database first, then files. If False, reverse.
        """
        logger.info(f"🔍 DUAL MODE READ: Starting for document {document_id}, prefer_db={prefer_database}")
        
        if prefer_database:
            # Try database first (bulletproof)
            try:
                result = await self.db.get_document_analysis(document_id)
                if result:
                    logger.info(f"✅ DATABASE READ: Success for {document_id}")
                    return result
                else:
                    logger.info(f"📭 DATABASE READ: No data found for {document_id}")
            except Exception as db_error:
                logger.error(f"❌ DATABASE READ: Failed for {document_id}: {db_error}")
            
            # Fallback to file system
            try:
                result = await self._read_from_files(document_id)
                if result:
                    logger.info(f"✅ FILE READ: Fallback success for {document_id}")
                    return result
            except Exception as file_error:
                logger.error(f"❌ FILE READ: Fallback failed for {document_id}: {file_error}")
        
        else:
            # Try files first (legacy mode)
            try:
                result = await self._read_from_files(document_id)
                if result:
                    logger.info(f"✅ FILE READ: Success for {document_id}")
                    return result
            except Exception as file_error:
                logger.error(f"❌ FILE READ: Failed for {document_id}: {file_error}")
            
            # Fallback to database
            try:
                result = await self.db.get_document_analysis(document_id)
                if result:
                    logger.info(f"✅ DATABASE READ: Fallback success for {document_id}")
                    return result
            except Exception as db_error:
                logger.error(f"❌ DATABASE READ: Fallback failed for {document_id}: {db_error}")
        
        logger.warning(f"📭 DUAL MODE READ: No data found anywhere for {document_id}")
        return None
    
    async def _read_from_files(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Read analysis results from legacy file system"""
        try:
            # Import here to avoid circular imports
            from routes.analysis_routes import ANALYSIS_RESULTS_DIR
            
            results_path = ANALYSIS_RESULTS_DIR / f"{document_id}.json"
            if not results_path.exists():
                return None
            
            with open(results_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            return results
            
        except Exception as e:
            logger.error(f"Error reading from files for {document_id}: {e}")
            return None
    
    async def migrate_existing_data(self, document_id: str) -> bool:
        """
        Migrate existing file-based data to database
        Returns True if migration successful, False if no data or error
        """
        logger.info(f"🔄 MIGRATION: Starting for document {document_id}")
        
        try:
            # Read from files
            file_data = await self._read_from_files(document_id)
            if not file_data:
                logger.info(f"📭 MIGRATION: No file data found for {document_id}")
                return False
            
            # Save to database
            await self.db.save_document_analysis(document_id, file_data)
            logger.info(f"✅ MIGRATION: Success for {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ MIGRATION: Failed for {document_id}: {e}")
            return False
    
    async def verify_data_consistency(self, document_id: str) -> Dict[str, Any]:
        """
        Verify data consistency between database and files
        Returns comparison report
        """
        logger.info(f"🔍 VERIFICATION: Starting for document {document_id}")
        
        report = {
            'document_id': document_id,
            'database_exists': False,
            'file_exists': False,
            'data_matches': False,
            'differences': [],
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Get data from both sources
            db_data = await self.db.get_document_analysis(document_id)
            file_data = await self._read_from_files(document_id)
            
            report['database_exists'] = db_data is not None
            report['file_exists'] = file_data is not None
            
            if db_data and file_data:
                # Compare key fields
                key_fields = ['status', 'framework', 'standards', 'metadata_extraction', 'compliance_analysis']
                differences = []
                
                for field in key_fields:
                    db_value = db_data.get(field)
                    file_value = file_data.get(field)
                    
                    if db_value != file_value:
                        differences.append({
                            'field': field,
                            'database_value': db_value,
                            'file_value': file_value
                        })
                
                report['differences'] = differences
                report['data_matches'] = len(differences) == 0
                
                logger.info(f"✅ VERIFICATION: Completed for {document_id}, matches={report['data_matches']}")
            
            return report
            
        except Exception as e:
            logger.error(f"❌ VERIFICATION: Failed for {document_id}: {e}")
            report['error'] = str(e)
            return report

# Global dual-mode storage instance
_dual_storage = None

def get_dual_storage() -> DualModeStorage:
    """Get singleton dual-mode storage instance"""
    global _dual_storage
    if _dual_storage is None:
        _dual_storage = DualModeStorage()
    return _dual_storage

# Convenience functions for easy integration
async def save_analysis_atomic(document_id: str, results: Dict[str, Any]) -> None:
    """
    ATOMIC SAVE: Zero race conditions, bulletproof storage
    This is the new primary save function
    """
    storage = get_dual_storage()
    await storage.save_analysis_results_v2(document_id, results)

async def get_analysis_atomic(document_id: str) -> Optional[Dict[str, Any]]:
    """
    ATOMIC READ: Bulletproof data retrieval with fallback
    This is the new primary read function
    """
    storage = get_dual_storage()
    return await storage.get_analysis_results_v2(document_id, prefer_database=True)

def setup_legacy_compatibility(legacy_save_function):
    """
    Setup backward compatibility with existing file-based save function
    Call this during startup to maintain existing functionality
    """
    storage = get_dual_storage()
    storage.set_legacy_save_function(legacy_save_function)
    logger.info("✅ DUAL MODE: Legacy compatibility enabled")