"""
Multi-Process JSON Manager
Prevents race conditions by creating separate JSON files per document/process
Then merges them for compliance analysis - brilliant solution!
"""

import json
import os
import threading
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

@dataclass
class ProcessMetadata:
    """Metadata for each document processing instance"""
    document_id: str
    process_id: str
    started_at: str
    status: str
    file_path: str
    company_metadata: Dict[str, Any]
    extraction_results: Dict[str, Any]
    updated_at: str

class MultiProcessDocumentManager:
    """
    Manages separate JSON files per document process
    Prevents 'flood victims on tamarind rice' scenario by giving each process its own space
    """
    
    def __init__(self, base_dir: str = "analysis_results"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.process_locks: Dict[str, threading.Lock] = {}
        self.active_processes: Dict[str, ProcessMetadata] = {}
        
    def create_process_workspace(self, document_id: str) -> str:
        """
        Create a dedicated workspace for this document process
        No more fighting over shared resources!
        """
        process_id = f"{document_id}_{uuid.uuid4().hex[:8]}"
        process_dir = self.base_dir / process_id
        process_dir.mkdir(exist_ok=True)
        
        # Create process lock
        self.process_locks[process_id] = threading.Lock()
        
        # Initialize process metadata
        metadata = ProcessMetadata(
            document_id=document_id,
            process_id=process_id,
            started_at=datetime.now().isoformat(),
            status="INITIALIZING",
            file_path=str(process_dir / "metadata.json"),
            company_metadata={},
            extraction_results={},
            updated_at=datetime.now().isoformat()
        )
        
        self.active_processes[process_id] = metadata
        
        # Save initial metadata file
        self._save_process_metadata(process_id, metadata)
        
        logger.info(f"Created process workspace: {process_id} for document: {document_id}")
        return process_id
        
    def update_process_status(self, process_id: str, status: str, 
                            company_metadata: Optional[Dict[str, Any]] = None,
                            extraction_results: Optional[Dict[str, Any]] = None):
        """
        Update process status in its own JSON file - no conflicts!
        """
        if process_id not in self.active_processes:
            raise ValueError(f"Process {process_id} not found")
            
        with self.process_locks[process_id]:
            metadata = self.active_processes[process_id]
            metadata.status = status
            metadata.updated_at = datetime.now().isoformat()
            
            if company_metadata:
                metadata.company_metadata.update(company_metadata)
                
            if extraction_results:
                metadata.extraction_results.update(extraction_results)
            
            # Save to separate JSON file
            self._save_process_metadata(process_id, metadata)
            
        logger.info(f"Updated process {process_id} status to {status}")
        
    def _save_process_metadata(self, process_id: str, metadata: ProcessMetadata):
        """Save process metadata to its dedicated JSON file"""
        file_path = Path(metadata.file_path)
        
        # Atomic write to prevent corruption
        temp_path = file_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
        
        # Atomic move
        temp_path.rename(file_path)
        
    def get_process_status(self, process_id: str) -> Optional[ProcessMetadata]:
        """Get current process status from its JSON file"""
        if process_id not in self.active_processes:
            return None
            
        # Read from file to get latest status
        metadata = self.active_processes[process_id]
        try:
            with open(metadata.file_path, 'r') as f:
                data = json.load(f)
            return ProcessMetadata(**data)
        except (FileNotFoundError, json.JSONDecodeError):
            return metadata
            
    def get_document_processes(self, document_id: str) -> List[ProcessMetadata]:
        """Get all processes for a specific document"""
        return [
            metadata for metadata in self.active_processes.values()
            if metadata.document_id == document_id
        ]
        
    def merge_for_compliance_analysis(self, document_id: str) -> Dict[str, Any]:
        """
        BRILLIANT MERGE SOLUTION!
        Collect all separate JSON files for this document and merge them
        No more race conditions - just clean merging!
        """
        processes = self.get_document_processes(document_id)
        
        if not processes:
            return {"error": "No processes found for document"}
            
        merged_result = {
            "document_id": document_id,
            "merged_at": datetime.now().isoformat(),
            "process_count": len(processes),
            "processes": [],
            "combined_metadata": {},
            "extraction_summary": {},
            "ready_for_compliance": False
        }
        
        # Merge all process results
        for process in processes:
            current_metadata = self.get_process_status(process.process_id)
            if current_metadata:
                merged_result["processes"].append({
                    "process_id": process.process_id,
                    "status": current_metadata.status,
                    "started_at": current_metadata.started_at,
                    "updated_at": current_metadata.updated_at
                })
                
                # Merge company metadata (last one wins for conflicts)
                if current_metadata.company_metadata:
                    merged_result["combined_metadata"].update(current_metadata.company_metadata)
                    
                # Merge extraction results
                if current_metadata.extraction_results:
                    for key, value in current_metadata.extraction_results.items():
                        if key not in merged_result["extraction_summary"]:
                            merged_result["extraction_summary"][key] = []
                        merged_result["extraction_summary"][key].append({
                            "process_id": process.process_id,
                            "value": value
                        })
        
        # Check if ready for compliance analysis
        completed_processes = []
        for p in processes:
            process_status = self.get_process_status(p.process_id)
            if process_status and process_status.status == "COMPLETED":
                completed_processes.append(p)
        merged_result["ready_for_compliance"] = len(completed_processes) > 0
        
        # Save merged result
        merged_file = self.base_dir / f"{document_id}_merged.json"
        with open(merged_file, 'w') as f:
            json.dump(merged_result, f, indent=2)
            
        logger.info(f"Merged {len(processes)} processes for document {document_id}")
        return merged_result
        
    def cleanup_process(self, process_id: str):
        """Clean up process workspace after completion"""
        if process_id in self.active_processes:
            metadata = self.active_processes[process_id]
            
            # Archive the process data
            archive_dir = self.base_dir / "archived" / process_id
            archive_dir.mkdir(parents=True, exist_ok=True)
            
            process_dir = Path(metadata.file_path).parent
            if process_dir.exists():
                import shutil
                shutil.move(str(process_dir), str(archive_dir))
            
            # Remove from active processes
            del self.active_processes[process_id]
            if process_id in self.process_locks:
                del self.process_locks[process_id]
                
        logger.info(f"Cleaned up process {process_id}")
        
    def get_all_active_processes(self) -> Dict[str, ProcessMetadata]:
        """Get all currently active processes"""
        active = {}
        for process_id, metadata in self.active_processes.items():
            current = self.get_process_status(process_id)
            if current and current.status not in ["COMPLETED", "FAILED"]:
                active[process_id] = current
        return active

# Global instance
document_manager = MultiProcessDocumentManager()

def get_document_manager() -> MultiProcessDocumentManager:
    """Get the global document manager instance"""
    return document_manager