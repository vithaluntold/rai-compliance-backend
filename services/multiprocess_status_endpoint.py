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
            
            # Extract simple values from nested format
            def extract_value(field_data):
                if isinstance(field_data, dict) and "value" in field_data:
                    return field_data.get("value", "")
                return str(field_data) if field_data else ""
            
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
        
        # Document not found anywhere
        logger.warning(f"❌ Document {document_id} not found in multi-process or legacy systems")
        
        return JSONResponse(
            status_code=404,
            content={
                "error": "Document not found",
                "message": f"Document {document_id} not found in system",
                "document_id": document_id
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