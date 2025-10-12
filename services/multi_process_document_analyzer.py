"""
Enhanced Document Analyzer with Multi-Process JSON Support
No more "flood victims on tamarind rice" - each process gets its own space!
"""

import json
import asyncio
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from .document_processor import DocumentProcessor
from .multi_process_manager import get_document_manager
from .content_filter import get_content_filter

logger = logging.getLogger(__name__)

class MultiProcessDocumentAnalyzer:
    """
    Document analyzer that uses separate JSON files per process
    Prevents race conditions and resource conflicts
    """
    
    def __init__(self):
        self.processor = DocumentProcessor()
        self.document_manager = get_document_manager()
        self.content_filter = get_content_filter()
        
    async def analyze_document_async(self, document_id: str, file_path: str, 
                                   processing_mode: str = "smart") -> str:
        """
        Analyze document with dedicated process workspace
        Returns process_id for tracking
        """
        # Create dedicated workspace for this process
        process_id = self.document_manager.create_process_workspace(document_id)
        
        try:
            logger.info(f"Starting analysis for document {document_id} in process {process_id}")
            
            # Update status: Starting content filtering
            self.document_manager.update_process_status(
                process_id, 
                "CONTENT_FILTERING",
                extraction_results={"stage": "content_filter"}
            )
            
            # Content filtering with process-specific logging
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # Extract text for content filtering
            if file_path.lower().endswith('.pdf'):
                import PyPDF2
                import io
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
                text_content = ""
                for page in pdf_reader.pages:
                    text_content += page.extract_text()
            else:
                text_content = content.decode('utf-8', errors='ignore')
            
            # Content filter check
            filter_result = self.content_filter.analyze_content_type(text_content, document_id)
            
            if not filter_result["should_process"]:
                self.document_manager.update_process_status(
                    process_id,
                    "REJECTED",
                    extraction_results={
                        "rejection_reason": filter_result.get("rejection_reason"),
                        "content_type": filter_result.get("content_type")
                    }
                )
                return process_id
                
            # Update status: Starting metadata extraction
            self.document_manager.update_process_status(
                process_id,
                "METADATA_EXTRACTION",
                extraction_results={
                    "stage": "metadata_extraction",
                    "content_filter_passed": True
                }
            )
            
            # Process document with enhanced processor
            result = await self.processor.process_document(
                document_id=document_id
            )
            
            # Extract company metadata from result
            company_metadata = {}
            if result.get("metadata"):
                # Map legacy metadata format to new format
                metadata = result["metadata"]
                company_metadata = {
                    "company_name": metadata.get("company_name", ""),
                    "nature_of_business": metadata.get("nature_of_business", ""),
                    "geography_of_operations": metadata.get("operational_demographics", []),
                    "financial_statement_type": metadata.get("financial_statements_type", "Standalone"),
                    "confidence_score": metadata.get("confidence_score", 0)
                }
                
            # Handle direct company_metadata if available
            if result.get("company_metadata"):
                company_metadata.update(result["company_metadata"])
                
            # Update process with extracted metadata
            self.document_manager.update_process_status(
                process_id,
                "COMPLETED",
                company_metadata=company_metadata,
                extraction_results={
                    "stage": "completed",
                    "processing_mode": processing_mode,
                    "sections_found": len(result.get("sections", [])),
                    "metadata_extraction_time": result.get("processing_time", 0)
                }
            )
            
            logger.info(f"Completed analysis for document {document_id} in process {process_id}")
            return process_id
            
        except Exception as e:
            logger.error(f"Error analyzing document {document_id} in process {process_id}: {e}")
            
            # Update process with error status
            self.document_manager.update_process_status(
                process_id,
                "FAILED",
                extraction_results={
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            return process_id
            
    def get_document_status(self, document_id: str) -> Dict[str, Any]:
        """
        Get merged status for document from all its processes
        This is what the frontend will poll!
        """
        try:
            # Get merged result from all processes
            merged_result = self.document_manager.merge_for_compliance_analysis(document_id)
            
            if merged_result.get("error"):
                return {
                    "document_id": document_id,
                    "status": "NOT_FOUND",
                    "message": "No active processes found for this document"
                }
            
            # Determine overall status
            processes = merged_result.get("processes", [])
            if not processes:
                status = "INITIALIZING"
            else:
                # Check if any process completed
                completed = any(p["status"] == "COMPLETED" for p in processes)
                failed = any(p["status"] == "FAILED" for p in processes)
                rejected = any(p["status"] == "REJECTED" for p in processes)
                
                if completed:
                    status = "COMPLETED"
                elif rejected:
                    status = "REJECTED" 
                elif failed:
                    status = "FAILED"
                else:
                    status = "PROCESSING"
            
            # Build response in format expected by frontend
            response = {
                "document_id": document_id,
                "status": status,
                "metadata_extraction": status,
                "compliance_analysis": "PENDING",
                "processing_mode": "smart",
                "metadata": {},  # Legacy format (empty)
                "company_metadata": merged_result.get("combined_metadata", {}),
                "sections": [],
                "progress": {
                    "total_processes": len(processes),
                    "completed_processes": len([p for p in processes if p["status"] == "COMPLETED"])
                },
                "framework": None,
                "standards": [],
                "message": f"Analysis completed with {len(processes)} processes"
            }
            
            # Add rejection reason if rejected
            if status == "REJECTED":
                for process in processes:
                    if process["status"] == "REJECTED":
                        # Get detailed rejection info
                        process_metadata = self.document_manager.get_process_status(process["process_id"])
                        if process_metadata and process_metadata.extraction_results:
                            response["rejection_reason"] = process_metadata.extraction_results.get("rejection_reason")
                        break
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting document status for {document_id}: {e}")
            return {
                "document_id": document_id,
                "status": "ERROR",
                "message": f"Error retrieving status: {str(e)}"
            }
            
    def get_active_processes_summary(self) -> Dict[str, Any]:
        """Get summary of all active processes"""
        active_processes = self.document_manager.get_all_active_processes()
        
        summary = {
            "total_active": len(active_processes),
            "by_status": {},
            "by_document": {},
            "processes": []
        }
        
        for process_id, metadata in active_processes.items():
            # Count by status
            status = metadata.status
            summary["by_status"][status] = summary["by_status"].get(status, 0) + 1
            
            # Count by document
            doc_id = metadata.document_id
            if doc_id not in summary["by_document"]:
                summary["by_document"][doc_id] = 0
            summary["by_document"][doc_id] += 1
            
            # Add process info
            summary["processes"].append({
                "process_id": process_id,
                "document_id": doc_id,
                "status": status,
                "started_at": metadata.started_at,
                "updated_at": metadata.updated_at
            })
            
        return summary

# Global instance
analyzer = MultiProcessDocumentAnalyzer()

def get_document_analyzer() -> MultiProcessDocumentAnalyzer:
    """Get the global document analyzer instance"""
    return analyzer