"""
Multi-Process Compatible Status Endpoint
Replaces the existing get_document_status function with race-condition-free version
"""

from typing import Dict, Any, Union
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

async def get_document_status_multiprocess(document_id: str) -> Union[Dict[str, Any], JSONResponse]:
    """
    Get document status using multi-process system
    No more race conditions - each document has its own JSON files!
    """
    logger.info(f"🔍 Multi-Process GET /documents/{document_id} - Starting request")
    
    try:
        from services.multi_process_document_analyzer import get_document_analyzer
        
        # Use multi-process document analyzer
        analyzer = get_document_analyzer()
        result = analyzer.get_document_status(document_id)
        
        logger.info(f"📊 Multi-process status result: {result.get('status')} for document {document_id}")
        
        # If multi-process system has the document, return its result
        if result.get("status") != "NOT_FOUND":
            logger.info(f"✅ Found multi-process data for {document_id}: {result.get('status')}")
            return result
        
        # Fall back to legacy system for backward compatibility
        logger.info(f"🔄 No multi-process data, falling back to legacy system for {document_id}")
        
        import os
        import json
        from routes.analysis_routes import ANALYSIS_RESULTS_DIR
        
        results_path = os.path.join(ANALYSIS_RESULTS_DIR, f"{document_id}.json")
        
        if os.path.exists(results_path):
            logger.info(f"📁 Found legacy results file: {results_path}")
            
            with open(results_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            
            # Convert legacy format to new multi-process format
            metadata = results.get("metadata", {})
            
            # Extract simple values from nested format with verbose response cleaning
            def extract_value(field_data):
                if isinstance(field_data, dict) and "value" in field_data:
                    value = field_data.get("value", "")
                else:
                    value = str(field_data) if field_data else ""
                
                # Clean up verbose AI responses
                if value:
                    value = value.strip()
                    
                    # For company names, reject generic terms and verbose descriptions
                    lower_val = value.lower()
                    if any(term in lower_val for term in [
                        'the group', 'the company', 'based on the document',
                        'from the financial statements', 'according to the document',
                        'the entity', 'this company', 'the client'
                    ]):
                        return ""
                    
                    # If it's longer than 100 characters, it's probably verbose - reject
                    if len(value) > 100:
                        # Try to extract just the company name from verbose response
                        lines = value.split('\n')
                        for line in lines[:3]:  # Check first 3 lines
                            line = line.strip()
                            if (len(line) < 80 and 
                                any(suffix in line for suffix in ['Ltd', 'PJSC', 'PLC', 'Inc', 'Group', 'Corp', 'LLC']) and
                                not any(word in line.lower() for word in ['based on', 'according to', 'from the'])):
                                return line
                        return ""  # Reject verbose responses
                    
                    # Length validation for company names
                    if len(value) < 2 or len(value) > 80:
                        return ""
                
                return value
            
            company_name = metadata.get('company_name_simple', '') or extract_value(metadata.get('company_name', ''))
            nature_of_business = metadata.get('nature_of_business_simple', '') or extract_value(metadata.get('nature_of_business', ''))
            operational_demo = metadata.get('operational_demographics_simple', '') or extract_value(metadata.get('operational_demographics', ''))
            financial_type = metadata.get('financial_statements_type_simple', '') or extract_value(metadata.get('financial_statements_type', 'Standalone'))
            
            # Convert geography string to array
            geography_list = []
            if operational_demo:
                geography_list = [country.strip() for country in operational_demo.split(',') if country.strip()]
            
            company_metadata = {
                "company_name": company_name,
                "nature_of_business": nature_of_business,
                "geography_of_operations": geography_list,
                "financial_statement_type": financial_type,
                "confidence_score": 90
            }
            
            # Normalize status
            status = results.get("status", "PROCESSING")
            if status.lower() == "completed":
                status = "COMPLETED"
            elif status.lower() in ["failed", "error"]:
                status = "FAILED"
            
            return {
                "document_id": document_id,
                "status": status,
                "metadata_extraction": status,
                "compliance_analysis": results.get("compliance_analysis", "PENDING"),
                "processing_mode": results.get("processing_mode", "smart"),
                "metadata": metadata,
                "company_metadata": company_metadata,
                "sections": results.get("sections", []),
                "progress": results.get("progress", {}),
                "framework": results.get("framework"),
                "standards": results.get("standards", []),
                "message": results.get("message", "Analysis completed")
            }
        
        # Document not found anywhere - provide helpful debugging info
        logger.warning(f"❌ Document {document_id} not found in multi-process or legacy systems")
        
        # Check if document was recently processed but files cleaned up
        import os
        from pathlib import Path
        
        debug_info = {
            "checked_locations": [
                "Multi-process analyzer memory",
                "Legacy results files",
                "Persistent storage database"
            ],
            "possible_causes": [
                "Document never uploaded",
                "Document processed and cleaned up", 
                "Document ID format error",
                "Processing failed and cleaned up"
            ]
        }
        
        # Check for any evidence the document existed
        uploads_dir = Path("uploads")
        if uploads_dir.exists():
            matching_files = list(uploads_dir.glob(f"*{document_id}*"))
            debug_info["upload_files_found"] = [f.name for f in matching_files]
        
        archive_dir = Path("document_archives") / document_id
        if archive_dir.exists():
            debug_info["archive_exists"] = True
            debug_info["archive_files"] = [f.name for f in archive_dir.glob("*")]
        else:
            debug_info["archive_exists"] = False
            
        return JSONResponse(
            status_code=404,
            content={
                "error": "Document not found",
                "message": f"Document {document_id} not found in system - may have been processed and cleaned up, or never uploaded",
                "document_id": document_id,
                "status": "not_found",
                "metadata_extraction": "not_found",
                "debug_info": debug_info
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error getting document status for {document_id}: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Server error",
                "message": f"Failed to get document status: {str(e)}",
                "document_id": document_id
            }
        )