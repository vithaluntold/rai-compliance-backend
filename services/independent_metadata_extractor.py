"""
Independent AI-Based Metadata Extractor

This service provides AI-only metadata extraction that is completely independent
from chunking systems and other processing pipelines. It directly processes
document text using AI services with no dependencies on chunk formats or
storage systems.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Union

from services.ai import get_ai_service

logger = logging.getLogger(__name__)


class IndependentMetadataExtractor:
    """AI-only metadata extraction service independent of chunking systems"""
    
    def __init__(self):
        self.ai_service = get_ai_service()
    
    async def extract_metadata_ai_only(
        self, 
        document_text: str, 
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract metadata using AI only - completely independent of chunking
        
        Args:
            document_text: Raw document text (string)
            document_id: Optional document identifier for logging
            
        Returns:
            Metadata dictionary with company info, business nature, etc.
        """
        logger.info(f"🤖 Starting AI-only metadata extraction for document {document_id or 'unknown'}")
        
        try:
            # Validate input
            if not document_text or not isinstance(document_text, str):
                logger.error("❌ Invalid document text provided")
                return self._get_fallback_metadata("Invalid or empty document text")
            
            # Truncate text if too long for AI processing (keep first 15000 chars for context)
            if len(document_text) > 15000:
                truncated_text = document_text[:15000] + "... [truncated]"
                logger.info(f"📄 Truncated document text from {len(document_text)} to 15000 characters")
            else:
                truncated_text = document_text
                logger.info(f"📄 Processing full document text: {len(document_text)} characters")
            
            # Create comprehensive AI prompt for metadata extraction
            extraction_prompt = self._build_ai_extraction_prompt(truncated_text)
            
            # Call AI service for metadata extraction
            logger.info("🧠 Calling AI service for comprehensive metadata extraction...")
            ai_response = await self.ai_service.analyze_compliance(
                document_id=document_id or "unknown",
                text=truncated_text,
                framework="GENERAL",  # Generic framework for metadata extraction
                standard="METADATA_EXTRACTION"
            )
            
            if not ai_response or not ai_response.get('analysis'):
                logger.error("❌ AI service returned empty response")
                return self._get_fallback_metadata("AI service unavailable")
            
            # Parse AI response into structured metadata
            metadata_result = self._parse_ai_response(ai_response['analysis'])
            
            # Add processing information
            metadata_result['extraction_info'] = {
                'method': 'ai_only_independent',
                'ai_tokens_used': ai_response.get('tokens_used', 0),
                'document_length': len(document_text),
                'processed_length': len(truncated_text),
                'document_id': document_id
            }
            
            logger.info(f"✅ AI-only metadata extraction completed for document {document_id}")
            logger.info(f"📊 Extracted metadata: {self._summarize_metadata(metadata_result)}")
            
            return metadata_result
            
        except Exception as e:
            logger.error(f"❌ CRITICAL ERROR in AI-only metadata extraction: {str(e)}")
            return self._get_fallback_metadata(f"Processing error: {str(e)}")
    
    def _build_ai_extraction_prompt(self, document_text: str) -> str:
        """Build comprehensive AI prompt for metadata extraction"""
        
        prompt = f"""
FINANCIAL DOCUMENT METADATA EXTRACTION

Please analyze the following financial document text and extract key metadata information. 
Provide specific, accurate information based on what you can identify in the document.

DOCUMENT TEXT:
{document_text}

EXTRACTION REQUIREMENTS:

1. COMPANY NAME:
   - Extract the exact legal company name
   - Look for headers, title pages, letterheads
   - Include subsidiary or group names if mentioned
   
2. NATURE OF BUSINESS:
   - Identify primary business activities and operations
   - Look for "principal activities", "business description", "operations"
   - Include specific industries, services, or products mentioned
   
3. OPERATIONAL DEMOGRAPHICS:
   - Identify geographic locations, countries, regions
   - Look for operational jurisdictions, registered addresses
   - Include any international operations mentioned
   
4. FINANCIAL STATEMENTS TYPE:
   - Identify specific financial statements present
   - Look for "Statement of Financial Position", "Income Statement", "Cash Flow", etc.
   - Note if consolidated, interim, or annual statements

RESPONSE FORMAT:
Provide your analysis in this exact structure:

COMPANY_NAME: [Exact company name found]
CONFIDENCE_COMPANY: [0.0-1.0 confidence score]

NATURE_OF_BUSINESS: [Detailed business description]
CONFIDENCE_BUSINESS: [0.0-1.0 confidence score]

OPERATIONAL_DEMOGRAPHICS: [Geographic and operational details]
CONFIDENCE_DEMOGRAPHICS: [0.0-1.0 confidence score]

FINANCIAL_STATEMENTS_TYPE: [Types of statements identified]
CONFIDENCE_STATEMENTS: [0.0-1.0 confidence score]

ANALYSIS_NOTES: [Any additional relevant observations]
"""
        return prompt
    
    def _parse_ai_response(self, ai_response: str) -> Dict[str, Any]:
        """Parse structured AI response into metadata dictionary"""
        
        def extract_field_and_confidence(text: str, field_prefix: str, confidence_prefix: str) -> Dict[str, Any]:
            """Extract field value and confidence from AI response"""
            try:
                # Extract field value
                field_pattern = rf"{field_prefix}:\s*(.+?)(?=\n[A-Z_]+:|$)"
                field_match = re.search(field_pattern, text, re.DOTALL | re.IGNORECASE)
                field_value = field_match.group(1).strip() if field_match else ""
                
                # Extract confidence
                conf_pattern = rf"{confidence_prefix}:\s*([0-9.]+)"
                conf_match = re.search(conf_pattern, text, re.IGNORECASE)
                confidence = float(conf_match.group(1)) if conf_match else 0.5
                
                return {
                    "value": field_value,
                    "confidence": min(max(confidence, 0.0), 1.0),  # Clamp to 0-1
                    "extraction_method": "ai_independent",
                    "context": "AI-based extraction from document analysis"
                }
            except Exception as e:
                logger.error(f"❌ Error parsing field {field_prefix}: {e}")
                return {
                    "value": "",
                    "confidence": 0.0,
                    "extraction_method": "ai_independent_error",
                    "context": f"Parse error: {str(e)}"
                }
        
        # Extract all metadata fields
        metadata = {
            "company_name": extract_field_and_confidence(
                ai_response, "COMPANY_NAME", "CONFIDENCE_COMPANY"
            ),
            "nature_of_business": extract_field_and_confidence(
                ai_response, "NATURE_OF_BUSINESS", "CONFIDENCE_BUSINESS"
            ),
            "operational_demographics": extract_field_and_confidence(
                ai_response, "OPERATIONAL_DEMOGRAPHICS", "CONFIDENCE_DEMOGRAPHICS"
            ),
            "financial_statements_type": extract_field_and_confidence(
                ai_response, "FINANCIAL_STATEMENTS_TYPE", "CONFIDENCE_STATEMENTS"
            )
        }
        
        # Extract analysis notes
        notes_pattern = r"ANALYSIS_NOTES:\s*(.+?)$"
        notes_match = re.search(notes_pattern, ai_response, re.DOTALL | re.IGNORECASE)
        analysis_notes = notes_match.group(1).strip() if notes_match else ""
        
        metadata['analysis_notes'] = {'notes': str(analysis_notes) if analysis_notes else ''}
        
        return metadata
    
    def _get_fallback_metadata(self, error_context: str) -> Dict[str, Any]:
        """Return fallback metadata structure when extraction fails"""
        return {
            "company_name": {
                "value": "Company name not extracted", 
                "confidence": 0.0, 
                "extraction_method": "fallback", 
                "context": error_context
            },
            "nature_of_business": {
                "value": "Business nature not determined", 
                "confidence": 0.0, 
                "extraction_method": "fallback", 
                "context": error_context
            },
            "operational_demographics": {
                "value": "Geographic data not available", 
                "confidence": 0.0, 
                "extraction_method": "fallback", 
                "context": error_context
            },
            "financial_statements_type": {
                "value": "Statement type not determined", 
                "confidence": 0.0, 
                "extraction_method": "fallback", 
                "context": error_context
            },
            "analysis_notes": error_context,
            "extraction_info": {
                "method": "fallback",
                "error": error_context
            }
        }
    
    def _summarize_metadata(self, metadata: Dict[str, Any]) -> str:
        """Create a summary string of extracted metadata for logging"""
        try:
            summary_parts = []
            for field in ["company_name", "nature_of_business", "operational_demographics", "financial_statements_type"]:
                if field in metadata:
                    value = metadata[field].get("value", "")
                    confidence = metadata[field].get("confidence", 0.0)
                    if value and len(value) > 50:
                        value = value[:47] + "..."
                    summary_parts.append(f"{field}='{value}' (conf: {confidence:.2f})")
            
            return " | ".join(summary_parts)
        except Exception:
            return "Summary generation failed"


# Convenience function for direct usage
async def extract_metadata_from_text(document_text: str, document_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Direct function to extract metadata from document text using AI
    
    Args:
        document_text: Raw document text
        document_id: Optional document identifier
        
    Returns:
        Extracted metadata dictionary
    """
    extractor = IndependentMetadataExtractor()
    return await extractor.extract_metadata_ai_only(document_text, document_id)
