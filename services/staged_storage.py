"""
Staged Data Storage Manager
Provides separated storage for different processing stages while maintaining backward compatibility.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class StagedStorageManager:
    """
    Manages staged storage for document processing pipeline.
    Maintains backward compatibility while providing stage isolation.
    """
    
    def __init__(self, base_dir: str = "analysis_results"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
    
    def _get_document_dir(self, document_id: str) -> Path:
        """Get or create document-specific directory."""
        doc_dir = self.base_dir / document_id
        doc_dir.mkdir(exist_ok=True)
        return doc_dir
    
    def _get_legacy_path(self, document_id: str) -> Path:
        """Get legacy main results file path."""
        return self.base_dir / f"{document_id}.json"
    
    def save_metadata(self, document_id: str, metadata: Dict[str, Any]) -> None:
        """Save metadata to isolated storage."""
        try:
            doc_dir = self._get_document_dir(document_id)
            metadata_path = doc_dir / "metadata.json"
            
            metadata_data = {
                "document_id": document_id,
                "timestamp": datetime.now().isoformat(),
                "status": "COMPLETED",
                "data": metadata
            }
            
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata_data, f, indent=2, ensure_ascii=False)
            
            # BACKWARD COMPATIBILITY: Also save to legacy metadata file
            legacy_metadata_path = self.base_dir / f"{document_id}_metadata.json"
            with open(legacy_metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ STAGED STORAGE: Saved metadata for {document_id}")
            
        except Exception as e:
            logger.error(f"❌ STAGED STORAGE: Failed to save metadata for {document_id}: {e}")
            raise
    
    def get_metadata(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata from staged storage."""
        try:
            # Try new staged storage first
            doc_dir = self._get_document_dir(document_id)
            metadata_path = doc_dir / "metadata.json"
            
            if metadata_path.exists():
                with open(metadata_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            
            # Fallback to legacy metadata file
            legacy_metadata_path = self.base_dir / f"{document_id}_metadata.json"
            if legacy_metadata_path.exists():
                with open(legacy_metadata_path, "r", encoding="utf-8") as f:
                    return json.load(f)
                    
            logger.warning(f"⚠️ STAGED STORAGE: No metadata found for {document_id}")
            return None
            
        except Exception as e:
            logger.error(f"❌ STAGED STORAGE: Failed to get metadata for {document_id}: {e}")
            return None
    
    def save_processing_status(self, document_id: str, status_data: Dict[str, Any]) -> None:
        """Save processing status to isolated storage."""
        try:
            doc_dir = self._get_document_dir(document_id)
            status_path = doc_dir / "processing_status.json"
            
            status_data.update({
                "document_id": document_id,
                "last_updated": datetime.now().isoformat()
            })
            
            with open(status_path, "w", encoding="utf-8") as f:
                json.dump(status_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"✅ STAGED STORAGE: Saved status for {document_id}")
            
        except Exception as e:
            logger.error(f"❌ STAGED STORAGE: Failed to save status for {document_id}: {e}")
            raise
    
    def get_processing_status(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get processing status from staged storage."""
        try:
            doc_dir = self._get_document_dir(document_id)
            status_path = doc_dir / "processing_status.json"
            
            if status_path.exists():
                with open(status_path, "r", encoding="utf-8") as f:
                    return json.load(f)
                    
            # If no staged status, try to extract from legacy file
            legacy_path = self._get_legacy_path(document_id)
            if legacy_path.exists():
                with open(legacy_path, "r", encoding="utf-8") as f:
                    legacy_data = json.load(f)
                    return {
                        "status": legacy_data.get("status", "UNKNOWN"),
                        "metadata_extraction": legacy_data.get("metadata_extraction", "UNKNOWN"),
                        "compliance_analysis": legacy_data.get("compliance_analysis", "UNKNOWN"),
                        "processing_mode": legacy_data.get("processing_mode", "smart")
                    }
                    
            return None
            
        except Exception as e:
            logger.error(f"❌ STAGED STORAGE: Failed to get status for {document_id}: {e}")
            return None
    
    def save_compliance_results(self, document_id: str, standard: str, results: Dict[str, Any]) -> None:
        """Save compliance analysis results for a specific standard."""
        try:
            doc_dir = self._get_document_dir(document_id)
            compliance_dir = doc_dir / "compliance"
            compliance_dir.mkdir(exist_ok=True)
            
            compliance_path = compliance_dir / f"{standard}.json"
            
            compliance_data = {
                "document_id": document_id,
                "standard": standard,
                "timestamp": datetime.now().isoformat(),
                "results": results
            }
            
            with open(compliance_path, "w", encoding="utf-8") as f:
                json.dump(compliance_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"✅ STAGED STORAGE: Saved compliance results for {document_id}/{standard}")
            
        except Exception as e:
            logger.error(f"❌ STAGED STORAGE: Failed to save compliance for {document_id}/{standard}: {e}")
            raise
    
    def update_legacy_file(self, document_id: str, updates: Dict[str, Any]) -> None:
        """
        Update legacy main results file for backward compatibility.
        SAFE: Only updates specific fields without overwriting metadata.
        """
        try:
            legacy_path = self._get_legacy_path(document_id)
            
            # Read existing data or create new
            if legacy_path.exists():
                with open(legacy_path, "r", encoding="utf-8") as f:
                    legacy_data = json.load(f)
            else:
                legacy_data = {
                    "document_id": document_id,
                    "timestamp": datetime.now().isoformat()
                }
            
            # SAFE UPDATE: Only update specific fields, preserve metadata
            safe_updates = {
                "status": updates.get("status"),
                "metadata_extraction": updates.get("metadata_extraction"), 
                "compliance_analysis": updates.get("compliance_analysis"),
                "processing_mode": updates.get("processing_mode"),
                "message": updates.get("message"),
                "last_updated": datetime.now().isoformat()
            }
            
            # Remove None values
            safe_updates = {k: v for k, v in safe_updates.items() if v is not None}
            
            # Update legacy data
            legacy_data.update(safe_updates)
            
            # Get metadata from staged storage if available
            metadata = self.get_metadata(document_id)
            if metadata and "data" in metadata:
                legacy_data["metadata"] = metadata["data"]
            
            with open(legacy_path, "w", encoding="utf-8") as f:
                json.dump(legacy_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"✅ STAGED STORAGE: Updated legacy file for {document_id}")
            
        except Exception as e:
            logger.error(f"❌ STAGED STORAGE: Failed to update legacy file for {document_id}: {e}")
            raise

# Global instance
staged_storage = StagedStorageManager()