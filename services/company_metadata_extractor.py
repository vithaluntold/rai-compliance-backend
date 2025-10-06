"""
Independent Company Metadata Extractor
Extracts specific company information from financial documents:
- Company Name
- Nature of Business  
- Geography of Operations
- Type of Financial Statements (Consolidated/Standalone)
"""

import re
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CompanyMetadata:
    """Structured company metadata"""
    company_name: str = ""
    nature_of_business: str = ""
    geography_of_operations: Optional[List[str]] = None
    financial_statement_type: str = ""  # "Consolidated" or "Standalone"
    confidence_score: float = 0.0
    
    def __post_init__(self):
        if self.geography_of_operations is None:
            self.geography_of_operations = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "company_name": self.company_name,
            "nature_of_business": self.nature_of_business,
            "geography_of_operations": self.geography_of_operations,
            "financial_statement_type": self.financial_statement_type,
            "confidence_score": self.confidence_score
        }

class CompanyMetadataExtractor:
    """Independent extractor for company-specific metadata"""
    
    def __init__(self):
        # Company name patterns
        self.company_patterns = [
            r'([A-Z][a-zA-Z\s&\-\.]+(?:Limited|Ltd|LLC|Inc|Corp|Corporation|PLC|Group|Holdings|Company))',
            r'([A-Z][a-zA-Z\s&\-\.]+(?:AG|SA|NV|BV|GmbH|AB|AS|Oy))',
            r'(\b[A-Z][a-zA-Z\s&\-\.]{5,50}(?:\s+(?:Limited|Ltd|PLC|Group|Inc|Corp)))',
        ]
        
        # Business nature keywords
        self.business_nature_keywords = {
            'insurance': ['insurance', 'insurer', 'underwriting', 'policies', 'premiums', 'claims'],
            'banking': ['bank', 'banking', 'financial services', 'loans', 'deposits', 'credit'],
            'technology': ['technology', 'software', 'IT services', 'digital', 'tech', 'platform'],
            'manufacturing': ['manufacturing', 'production', 'factory', 'industrial', 'assembly'],
            'retail': ['retail', 'consumer', 'stores', 'merchandise', 'sales'],
            'real_estate': ['real estate', 'property', 'development', 'construction', 'land'],
            'healthcare': ['healthcare', 'medical', 'pharmaceutical', 'hospital', 'health'],
            'energy': ['energy', 'oil', 'gas', 'power', 'electricity', 'renewable'],
            'telecommunications': ['telecommunications', 'telecom', 'communications', 'network']
        }
        
        # Geography patterns
        self.geography_patterns = [
            r'\b(?:operations?\s+in|based\s+in|located\s+in|operates?\s+in)\s+([A-Z][a-zA-Z\s,]+?)(?:\.|,|\s+and)',
            r'\b([A-Z][a-zA-Z\s]+)(?:\s+operations?|\s+market|\s+subsidiary|\s+office)',
            r'\b(?:country|countries|region|regions?):\s*([A-Z][a-zA-Z\s,]+)',
        ]
        
        # Financial statement type patterns
        self.consolidated_patterns = [
            r'\bconsolidated\s+financial\s+statements?\b',
            r'\bconsolidated\s+(?:balance\s+sheet|income\s+statement|cash\s+flow)\b',
            r'\b(?:group|consolidated)\s+accounts?\b',
            r'\bconsolidated\s+and\s+(?:company|separate)\s+financial\s+statements?\b'
        ]
        
        self.standalone_patterns = [
            r'\bstandalone\s+financial\s+statements?\b',
            r'\bseparate\s+financial\s+statements?\b',
            r'\bindividual\s+financial\s+statements?\b',
            r'\bcompany\s+(?:only\s+)?financial\s+statements?\b'
        ]
    
    def extract_company_metadata(self, document_text: str, document_id: Optional[str] = None) -> CompanyMetadata:
        """Extract comprehensive company metadata from document text"""
        logger.info(f"🏢 Extracting company metadata for document: {document_id or 'Unknown'}")
        
        metadata = CompanyMetadata()
        
        # Extract company name
        metadata.company_name = self._extract_company_name(document_text)
        
        # Extract nature of business
        metadata.nature_of_business = self._extract_business_nature(document_text)
        
        # Extract geography of operations
        metadata.geography_of_operations = self._extract_geography(document_text)
        
        # Extract financial statement type
        metadata.financial_statement_type = self._extract_statement_type(document_text)
        
        # Calculate confidence score
        metadata.confidence_score = self._calculate_confidence(metadata)
        
        logger.info(f"✅ Company metadata extracted - Name: {metadata.company_name}, "
                   f"Business: {metadata.nature_of_business}, Type: {metadata.financial_statement_type}")
        
        return metadata
    
    def _extract_company_name(self, text: str) -> str:
        """Extract company name from text"""
        # Look in first 2000 characters for company name
        header_text = text[:2000]
        
        # First check for patterns at start of document
        lines = header_text.split('\n')[:10]  # First 10 lines
        
        for line in lines:
            line = line.strip()
            if len(line) > 5:
                # Check if line contains company suffixes
                for pattern in self.company_patterns:
                    matches = re.findall(pattern, line, re.IGNORECASE)
                    if matches:
                        company_name = matches[0].strip()
                        if len(company_name) > 5:  # Valid company name
                            logger.debug(f"Found company name: {company_name}")
                            return company_name
        
        # Fallback: Look for title-like patterns in first few lines
        for line in lines[:5]:
            line = line.strip()
            # Simple pattern: starts with capital, contains uppercase words
            if re.match(r'^[A-Z][A-Za-z\s&\-\.]{5,50}$', line) and len(line.split()) <= 6:
                return line
        
        return ""
    
    def _extract_business_nature(self, text: str) -> str:
        """Extract nature of business from document"""
        # Look in first 5000 characters
        business_text = text[:5000].lower()
        
        business_scores = {}
        for business_type, keywords in self.business_nature_keywords.items():
            score = sum(1 for keyword in keywords if keyword in business_text)
            if score > 0:
                business_scores[business_type] = score
        
        if business_scores:
            # Return business type with highest score
            top_business = max(business_scores.items(), key=lambda x: x[1])
            logger.debug(f"Identified business nature: {top_business[0]} (score: {top_business[1]})")
            return top_business[0].replace('_', ' ').title()
        
        # Fallback: Look for explicit business description patterns
        business_patterns = [
            r'(?:nature\s+of\s+business|principal\s+activities?|business\s+activities?)[:,\s]*([^\.]+)',
            r'(?:the\s+(?:company|group))\s+(?:is\s+)?(?:engaged\s+in|operates\s+in|provides?)\s+([^\.]+)',
        ]
        
        for pattern in business_patterns:
            matches = re.findall(pattern, text[:3000], re.IGNORECASE)
            if matches:
                return matches[0].strip()[:100]  # Limit length
        
        return ""
    
    def _extract_geography(self, text: str) -> List[str]:
        """Extract geographical operations from text"""
        geography = set()
        
        # Common country/region names
        countries = [
            'United States', 'United Kingdom', 'Canada', 'Australia', 'Germany', 'France',
            'Japan', 'China', 'India', 'Brazil', 'Mexico', 'Singapore', 'Hong Kong',
            'UAE', 'Dubai', 'Saudi Arabia', 'South Africa', 'Nigeria', 'Kenya',
            'Europe', 'Asia', 'Americas', 'Middle East', 'Africa'
        ]
        
        # Look for geography patterns
        for pattern in self.geography_patterns:
            matches = re.findall(pattern, text[:5000], re.IGNORECASE)
            for match in matches:
                # Check if match contains known countries/regions
                for country in countries:
                    if country.lower() in match.lower():
                        geography.add(country)
        
        # Direct country mentions
        for country in countries:
            if re.search(rf'\b{re.escape(country)}\b', text[:5000], re.IGNORECASE):
                geography.add(country)
        
        return list(geography)[:10]  # Limit to 10 locations
    
    def _extract_statement_type(self, text: str) -> str:
        """Determine if statements are Consolidated or Standalone"""
        # Check first 3000 characters for statement type indicators
        header_text = text[:3000].lower()
        
        consolidated_score = 0
        standalone_score = 0
        
        # Check consolidated patterns
        for pattern in self.consolidated_patterns:
            if re.search(pattern, header_text, re.IGNORECASE):
                consolidated_score += 1
        
        # Check standalone patterns  
        for pattern in self.standalone_patterns:
            if re.search(pattern, header_text, re.IGNORECASE):
                standalone_score += 1
        
        # Additional scoring based on keywords
        if 'group' in header_text or 'consolidated' in header_text:
            consolidated_score += 1
        
        if 'standalone' in header_text or 'separate' in header_text:
            standalone_score += 1
        
        # Return result based on scores
        if consolidated_score > standalone_score:
            return "Consolidated"
        elif standalone_score > consolidated_score:
            return "Standalone"
        else:
            # Default based on common patterns
            if 'group' in header_text or 'holdings' in header_text:
                return "Consolidated"
            return "Standalone"
    
    def _calculate_confidence(self, metadata: CompanyMetadata) -> float:
        """Calculate confidence score for extracted metadata"""
        score = 0.0
        
        # Company name confidence
        if metadata.company_name:
            if len(metadata.company_name) > 10:
                score += 0.3
            else:
                score += 0.15
        
        # Business nature confidence
        if metadata.nature_of_business:
            if len(metadata.nature_of_business) > 5:
                score += 0.25
            else:
                score += 0.1
        
        # Geography confidence
        if metadata.geography_of_operations:
            score += min(0.25, len(metadata.geography_of_operations) * 0.05)
        
        # Statement type confidence
        if metadata.financial_statement_type:
            score += 0.2
        
        return min(1.0, score)  # Cap at 1.0
    
    def extract_metadata_dict(self, document_text: str, document_id: Optional[str] = None) -> Dict[str, Any]:
        """Extract metadata and return as dictionary"""
        metadata = self.extract_company_metadata(document_text, document_id)
        return metadata.to_dict()


# Convenience function for easy import
def extract_company_metadata(document_text: str, document_id: Optional[str] = None) -> Dict[str, Any]:
    """Extract company metadata from document text"""
    extractor = CompanyMetadataExtractor()
    return extractor.extract_metadata_dict(document_text, document_id)


if __name__ == "__main__":
    # Test with sample text
    sample_text = """
    Phoenix Group PLC
    DIRECTORS' AND AUDITOR'S REPORTS
    AND CONSOLIDATED FINANCIAL STATEMENTS
    FOR THE YEAR ENDED 31 DECEMBER 2024
    
    Phoenix Group PLC is a leading insurance company operating in the United Kingdom,
    with subsidiaries across Europe and operations in Ireland, Germany, and France.
    The Group provides life insurance and pension services to customers.
    """
    
    result = extract_company_metadata(sample_text, "test-doc")
    print("Extracted Metadata:")
    for key, value in result.items():
        print(f"  {key}: {value}")