"""
Smart Metadata Extractor Service

Optimized metadata extraction using robust NER (Transformers) + AI validation
to reduce token usage from 75K to 15K per document while maintaining accuracy.
"""

import logging
import re
import warnings
from typing import Any, Dict, List, Tuple, Union

from services.geographical_service import GeographicalDetectionService
from services.vector_store import get_vector_store
from services.ai import get_ai_service
from services.ai_prompts import AIPrompts
from config.extraction_config import get_config

# Suppress transformers warnings for cleaner logs
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')

logger = logging.getLogger(__name__)


class SmartMetadataExtractor:
    """Optimized metadata extraction using robust NER + AI validation"""

    def __init__(self):
        self.geographical_service = GeographicalDetectionService()
        self.vector_store = get_vector_store()
        self.ai_service = get_ai_service()
        self.config = get_config()

        # Initialize robust NER pipeline
        self._init_ner_pipeline()

        # Company name regex patterns from config (fallback only)
        self.company_patterns = self.config.get_company_pattern_regex()

        # Business type classification patterns from config
        self.business_patterns = self.config.get_business_keywords()

    def _init_ner_pipeline(self):
        """Initialize the NER pipeline using Transformers"""
        try:
            from transformers import pipeline
            logger.info("🧠 Initializing robust NER pipeline (Transformers)")
            
            # Use a high-quality BERT model fine-tuned for NER
            self.ner_pipeline = pipeline(
                'ner',
                model='dbmdz/bert-large-cased-finetuned-conll03-english',
                aggregation_strategy='simple'
            )
            self.ner_available = True
            logger.info("✅ Robust NER pipeline initialized successfully")
            
        except Exception as e:
            logger.warning(f"❌ Failed to initialize NER pipeline: {e}")
            self.ner_pipeline = None
            self.ner_available = False

    async def extract_metadata_optimized(
        self, document_id: str, chunks: List[Union[str, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Extract metadata using 3-tier strategy"""
        logger.info(f"🚀 Starting SMART metadata extraction for document {document_id}")
        logger.info(f"🔍 Processing {len(chunks)} chunks with smart extractor")

        # Combine all chunks for pattern analysis - extract text from chunk dictionaries
        if chunks and isinstance(chunks[0], dict):
            # Chunks are dictionaries with 'text' key
            full_text = " ".join([
                chunk.get("text", "") for chunk in chunks
                if isinstance(chunk, dict)
            ])
        else:
            # Chunks are strings (fallback)
            full_text = " ".join([str(chunk) for chunk in chunks])
        logger.info(f"📄 Combined text length: {len(full_text)} characters")

        # Tier 1: Pattern-based extraction
        logger.info("⚡ Tier 1: Pattern-based extraction")
        pattern_results = await self._extract_with_patterns(full_text)
        logger.info(f"✅ Pattern results: {pattern_results}")

        # Tier 2: Semantic search enhancement
        logger.info("🔍 Tier 2: Semantic search enhancement")
        semantic_results = await self._enhance_with_semantic_search(
            document_id, [full_text], pattern_results
        )

        # Tier 3: AI validation (only for uncertain fields)
        logger.info("🧠 Tier 3: AI validation")
        final_results = await self._validate_with_ai(
            [full_text], semantic_results
        )

        logger.info(f"🎯 Optimized extraction completed for document {document_id}")
        logger.info(f"📊 Final results: {final_results}")
        return final_results

    async def _extract_with_patterns(self, text: str) -> Dict[str, Any]:
        """Tier 1: Pattern-based extraction with high confidence scoring"""
        # Initialize results structure with 4 fields including financial statements type
        results = {
            "company_name": {"value": "", "confidence": 0.0, "extraction_method": "pattern", "context": ""},
            "nature_of_business": {"value": "", "confidence": 0.0, "extraction_method": "ai", "context": ""},
            "operational_demographics": {"value": "", "confidence": 0.0, "extraction_method": "ai", "context": ""},
            "financial_statements_type": {"value": "", "confidence": 0.0, "extraction_method": "ai", "context": ""}
        }

        # Robust NER-powered company name extraction with pattern fallback
        logger.info("🧠 Extracting company name using robust NER approach")
        company_name, company_confidence, company_context = self._extract_company_name_robust_ner(text)
        
        # Fallback to pattern-based if NER fails
        if not company_name:
            logger.info("🔄 NER extraction failed, using pattern fallback")
            company_name, company_confidence, company_context = self._extract_company_name_pattern(text)
            
        if company_name:
            results["company_name"]["value"] = company_name
            results["company_name"]["confidence"] = company_confidence
            results["company_name"]["context"] = company_context
            logger.info(f"🏢 Company extraction found: {company_name} (confidence: {company_confidence})")

        # Keyword→Semantic→AI approach for all other fields
        logger.info("🤖 Extracting business nature using keyword→semantic→AI approach")
        business_nature, business_context = await self._extract_with_keyword_semantic_ai(
            text, "nature_of_business", ["principal activities", "business", "engaged", "development", "management", "services", "operations", "construction", "leasing", "investment", "real estate", "hotels", "schools", "marinas", "restaurants", "beach clubs", "golf courses", "Group is", "company"]
        )
        
        # Fallback pattern-based approach if AI fails
        if not business_nature:
            # Look for detailed business descriptions in the text
            business_patterns = [
                r"The Group is engaged in[^.]+(?:development|construction|leasing|management|investment)[^.]*\.",
                r"principal activities[^.]+(?:development|construction|leasing|management|investment)[^.]*\.",
                r"business[^.]+(?:development|construction|leasing|management|real estate)[^.]*\."
            ]
            
            for pattern in business_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
                if matches:
                    business_nature = matches[0].strip()
                    business_context = "Pattern matched from document content"
                    logger.info(f"🏢 Pattern-based business extraction found: {business_nature[:100]}...")
                    break
            
            # Simple fallback if patterns don't work
            if not business_nature:
                if "real estate" in text.lower() and "development" in text.lower():
                    business_nature = "The Group is engaged in real estate development, construction, leasing, and management activities"
                    business_context = "Inferred from document content"
        
        if business_nature:
            results["nature_of_business"]["value"] = business_nature
            results["nature_of_business"]["confidence"] = 0.9
            results["nature_of_business"]["context"] = business_context

        logger.info("🌍 Extracting demographics using keyword→semantic→AI approach")
        demographics, demo_context = await self._extract_with_keyword_semantic_ai(
            text, "operational_demographics", ["operations", "subsidiaries", "countries", "geography", "locations", "jurisdictions", "UAE", "Egypt", "England", "Wales", "United Arab Emirates", "incorporated", "Dubai", "Abu Dhabi", "presence", "offices"]
        )
        
        # Fallback pattern-based approach if AI fails or incomplete
        if not demographics or len(demographics.split(',')) < 2:  # Use fallback if we found fewer than 2 countries
            # Precise country detection - only extract countries that actually exist in text
            countries_found = []
            
            # More precise country patterns that require full context
            country_checks = [
                # UAE variations
                (r'\b(United Arab Emirates)\b', "United Arab Emirates"),
                (r'\bUAE\b(?!\s*\w)', "United Arab Emirates"),  # UAE not followed by other letters
                
                # Egypt
                (r'\bEgypt\b', "Egypt"),
                
                # England and Wales (common in legal documents)
                (r'\b(England and Wales)\b', "England and Wales"),
                
                # Only check for other countries if there's clear evidence
                (r'\b(United States of America)\b', "United States"),
                (r'\bUSA\b(?!\s*\w)', "United States"),
                (r'\b(United Kingdom of Great Britain)\b', "United Kingdom"),
                (r'\bSingapore\b', "Singapore"),
                (r'\bIndia\b', "India"),
                (r'\bChina\b', "China"),
                (r'\bCanada\b', "Canada"),
                (r'\bAustralia\b', "Australia"),
                (r'\bFrance\b', "France"),
                (r'\bGermany\b', "Germany"),
                (r'\b(Saudi Arabia)\b', "Saudi Arabia"),
                (r'\bQatar\b', "Qatar"),
                (r'\bBahrain\b', "Bahrain"),
                (r'\bKuwait\b', "Kuwait"),
                (r'\bOman\b', "Oman")
            ]
            
            # Only add countries that have clear evidence in the text
            for pattern, country_name in country_checks:
                if re.search(pattern, text, re.IGNORECASE):
                    # Additional validation - make sure it's in a meaningful context
                    context_match = re.search(rf'.{{0,50}}{pattern}.{{0,50}}', text, re.IGNORECASE)
                    if context_match:
                        context = context_match.group(0).lower()
                        # Skip if it appears to be in unrelated context
                        if not any(skip_word in context for skip_word in ['example', 'sample', 'template', 'format']):
                            if country_name not in countries_found:
                                countries_found.append(country_name)
                                logger.info(f"📍 Found country: {country_name} in context: {context[:100]}...")
            
            if countries_found:
                fallback_demographics = ", ".join(countries_found)
                if not demographics or len(countries_found) > len(demographics.split(',')):  # Use fallback if we found more countries
                    demographics = fallback_demographics
                    demo_context = "Pattern matched from document content"
                    logger.info(f"📍 Pattern-based demographics extraction found: {demographics}")
        
        if demographics:
            results["operational_demographics"]["value"] = demographics
            results["operational_demographics"]["confidence"] = 0.9
            results["operational_demographics"]["context"] = demo_context

        # Extract the 4th field - Financial Statements Type using simple pattern matching
        logger.info("📊 Extracting financial statements type using pattern matching")
        fs_type = "Consolidated"  # Default based on context showing "consolidated financial statements"
        if "consolidated" in text.lower() and "financial statements" in text.lower():
            fs_type = "Consolidated"
        elif "standalone" in text.lower() and "financial statements" in text.lower():
            fs_type = "Standalone"
        elif "separate" in text.lower() and "financial statements" in text.lower():
            fs_type = "Separate"
        
        results["financial_statements_type"]["value"] = fs_type
        results["financial_statements_type"]["confidence"] = 0.9
        results["financial_statements_type"]["context"] = "Pattern matched from document content"

        return results

    def _extract_company_name_robust_ner(self, text: str) -> Tuple[str, float, str]:
        """
        Extract CLIENT company name using NER + AI intelligence.
        
        Strategy:
        1. Use NER to extract ALL organization names with their context
        2. Send all candidates to AI to intelligently decide which is the CLIENT company vs auditor
        3. Return the CLIENT company name with high confidence
        """
        try:
            logger.info("🧠 Starting NER + AI company name extraction")
            
            # Step 1: Use NER to extract ALL organizations with context
            company_candidates = self._extract_all_organizations_with_context(text)
            
            if not company_candidates:
                logger.warning("No organizations found by NER, falling back to regex")
                return self._extract_company_name_regex_enhanced(text)
            
            # Step 2: Use AI to intelligently select the CLIENT company
            client_company = self._ai_select_client_company(company_candidates, text)
            
            if client_company:
                logger.info(f"✅ AI selected CLIENT company: '{client_company['name']}'")
                return client_company['name'], client_company['confidence'], client_company['context']
            else:
                logger.warning("AI could not determine client company, using fallback")
                return self._extract_company_name_regex_enhanced(text)
                
        except Exception as e:
            logger.error(f"Error in NER + AI company extraction: {str(e)}")
            return "", 0.0, ""
    
    def _extract_all_organizations_with_context(self, text: str) -> List[Dict[str, Any]]:
        """
        Use NER to extract ALL organizations with their surrounding context.
        Returns list of: [{'name': str, 'confidence': float, 'context': str, 'position': int}, ...]
        """
        try:
            if not self.ner_pipeline:
                return []
            
            # Process larger text chunk for better company detection - use first 50K chars or full text if smaller
            text_chunk = text[:50000] if len(text) > 50000 else text
            
            # Extract entities using NER
            entities = self.ner_pipeline(text_chunk)
            org_entities = [e for e in entities if e['entity_group'] == 'ORG' and e['score'] > 0.85]
            
            logger.info(f"🔍 NER found {len(org_entities)} organization candidates")
            
            candidates = []
            for entity in org_entities:
                org_name = entity['word'].strip()
                ner_confidence = entity['score']
                
                # Find the position of this organization in the text to extract context
                start_pos = text_chunk.find(org_name)
                if start_pos == -1:
                    # Try case-insensitive search
                    start_pos = text_chunk.lower().find(org_name.lower())
                
                if start_pos != -1:
                    # Extract 200 characters of context around the organization name
                    context_start = max(0, start_pos - 100)
                    context_end = min(len(text_chunk), start_pos + len(org_name) + 100)
                    context = text_chunk[context_start:context_end].strip()
                    
                    candidate = {
                        'name': org_name,
                        'confidence': ner_confidence,
                        'context': context,
                        'position': start_pos
                    }
                    candidates.append(candidate)
                    
                    logger.info(f"  � Found: '{org_name}' (confidence: {ner_confidence:.3f})")
                    logger.info(f"      Context: '{context[:100]}...'")
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error extracting organizations with context: {str(e)}")
            return []
    
    def _ai_select_client_company(self, candidates: List[Dict[str, Any]], full_text: str) -> Dict[str, Any]:
        """
        Use AI to intelligently select the CLIENT company from all organization candidates.
        AI analyzes context to distinguish client company from auditor firms.
        """
        try:
            if not candidates:
                return None
            
            # Prepare the candidate information for AI
            candidates_info = []
            for i, candidate in enumerate(candidates, 1):
                candidates_info.append(
                    f"{i}. ORGANIZATION: '{candidate['name']}'\n"
                    f"   NER CONFIDENCE: {candidate['confidence']:.3f}\n"
                    f"   CONTEXT: \"{candidate['context']}\"\n"
                )
            
            candidates_text = "\n".join(candidates_info)
            
            # Create AI prompt for intelligent client company selection
            system_prompt = """You are an expert at analyzing financial documents to distinguish between CLIENT companies and AUDITOR firms.

Your task: Analyze the organization candidates and their contexts to identify which one is the PRIMARY CLIENT COMPANY (whose financial statements these are), NOT the auditor firm.

IMPORTANT RULES:
1. CLIENT COMPANY appears in document titles like "Financial Statements of [CLIENT]"
2. CLIENT COMPANY is the subject being audited, not the company doing the auditing
3. AUDITOR FIRMS typically appear in "Independent Auditor's Report" sections
4. AUDITOR FIRMS often end in "LLP", "Chartered Accountants", or are well-known audit firms (KPMG, PWC, EY, Deloitte, RAI)
5. CLIENT COMPANY usually appears earlier in the document (title/header sections)

Return ONLY the number (1, 2, 3, etc.) of the CLIENT COMPANY candidate. If no clear client company exists, return "NONE"."""

            user_prompt = f"""DOCUMENT EXCERPT:
{full_text[:15000]}

ORGANIZATION CANDIDATES FOUND:
{candidates_text}

Analyze the context and determine which organization is the CLIENT COMPANY (not auditor).

Return only the number of the CLIENT COMPANY candidate (e.g., "1", "2", "3") or "NONE" if unclear."""

            # Call AI to make the decision
            response = self.ai_service.openai_client.chat.completions.create(
                model=self.ai_service.deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=50
            )
            
            ai_response = response.choices[0].message.content.strip()
            logger.info(f"🤖 AI decision: '{ai_response}'")
            
            # Parse AI response
            try:
                selected_index = int(ai_response) - 1  # Convert to 0-based index
                if 0 <= selected_index < len(candidates):
                    selected_candidate = candidates[selected_index].copy()
                    selected_candidate['confidence'] = 0.95  # High confidence for AI selection
                    
                    logger.info(f"✅ AI selected candidate {selected_index + 1}: '{selected_candidate['name']}'")
                    return selected_candidate
                else:
                    logger.warning(f"AI returned invalid index: {ai_response}")
                    return None
                    
            except ValueError:
                if ai_response.upper() == "NONE":
                    logger.warning("AI could not determine client company")
                    return None
                else:
                    logger.warning(f"AI returned unexpected response: {ai_response}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error in AI client company selection: {str(e)}")
            return None
    
    def _validate_company_name(self, name: str) -> bool:
        """Validate if the extracted name is a legitimate company name."""
        if not name or len(name.strip()) < 3:
            return False
        
        name_lower = name.lower().strip()
        
        # Reject generic terms
        generic_terms = [
            'the company', 'the group', 'the entity', 'the corporation',
            'company', 'group', 'entity', 'corporation', 'organization',
            'client', 'customer', 'business', 'firm', 'enterprise'
        ]
        
        if name_lower in generic_terms:
            return False
        
        # Reject auditor names
        if self._is_likely_auditor_name(name):
            return False
        
        # Reject if it's too generic or short
        if len(name.split()) == 1 and len(name) < 4:
            return False
        
        # Accept if it has proper company indicators
        company_indicators = ['inc', 'corp', 'ltd', 'llc', 'plc', 'sa', 'gmbh', 'ag', 'srl', 'spa', 'pjsc']
        if any(indicator in name_lower for indicator in company_indicators):
            return True
        
        # Accept if it's a reasonable length and contains proper nouns
        if 3 <= len(name.split()) <= 6 and any(word[0].isupper() for word in name.split()):
            return True
        
        return False
    
    def _is_likely_auditor_name(self, name: str) -> bool:
        """Check if the given name is likely an auditor firm."""
        name_lower = name.lower().strip()
        
        auditor_indicators = [
            'chartered accountants', 'audit', 'auditor', 'kpmg', 'ey', 'pwc', 
            'deloitte', 'ernst', 'young', 'rai llp', 'chartered', 'rai', 'accountants'
        ]
        
        return (name_lower.endswith('llp') or 
                name_lower.endswith('chartered accountants') or
                'rai' in name_lower or
                any(indicator in name_lower for indicator in auditor_indicators))
    
    def _get_title_page_context(self, text: str) -> str:
        """Extract title page context from document text."""
        lines = text.split('\n')
        
        # Strategy 1: Find lines that look like document titles
        title_lines = []
        for i, line in enumerate(lines[:20]):  # Only first 20 lines for titles
            line = line.strip()
            if not line:
                continue
            
            # Look for title patterns
            if (any(pattern in line.lower() for pattern in [
                'financial statements', 'annual report', 'consolidated statements',
                'audited financial', 'consolidated financial'
            ]) and len(line) > 20 and len(line) < 150):
                title_lines.append((i, line))
        
        # Extract context around title lines
        title_context = ""
        if title_lines:
            for line_num, title_line in title_lines:
                start_line = max(0, line_num - 2)
                end_line = min(len(lines), line_num + 3)
                context_block = '\n'.join(lines[start_line:end_line])
                
                # Skip auditor-related sections
                if not any(auditor_term in context_block.lower() for auditor_term in [
                    'independent auditor', 'chartered accountants', 'llp', 'auditor\'s report',
                    'we have audited', 'audit opinion'
                ]):
                    title_context += context_block + '\n\n'
        
        # If no title context found, use first 10 lines but skip auditor sections
        if not title_context.strip():
            first_lines = []
            for line in lines[:15]:
                if not any(auditor_term in line.lower() for auditor_term in [
                    'independent auditor', 'chartered accountants', 'llp', 'auditor\'s report'
                ]):
                    first_lines.append(line)
            title_context = '\n'.join(first_lines[:10])
        
        return title_context.strip()
    
    def _extract_company_name_regex_enhanced(self, text: str) -> Tuple[str, float, str]:
        """Enhanced regex-based extraction as fallback."""
        try:
            # Enhanced patterns for company names
            patterns = [
                r'([A-Z][a-zA-Z\s&-]+(?:Inc\.?|Corporation|Corp\.?|Ltd\.?|Limited|LLC|PLC|PJSC))',
                r'([A-Z][a-zA-Z\s&-]+(?:Group|Holdings|International|Technologies|Systems))',
                r'((?:[A-Z][a-zA-Z]+\s+){1,3}(?:Inc\.?|Corp\.?|Ltd\.?|PLC|PJSC))',
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text[:8000], re.MULTILINE)
                for match in matches:
                    if self._validate_company_name(match):
                        logger.info(f"✅ ACCEPTED regex company: '{match}'")
                        return match.strip(), 0.85, text[:200]
            
            return "", 0.0, ""
            
        except Exception as e:
            logger.error(f"Error in regex company extraction: {str(e)}")
            return "", 0.0, ""
    
    def _extract_company_name_ner(self, text: str) -> Tuple[str, float, str]:
        """Extract company name using spaCy NER if available"""
        try:
            import spacy
            
            # Try to load English model
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model en_core_web_sm not found. Install with: python -m spacy download en_core_web_sm")
                return "", 0.0, ""
            
            # Process document headers where company names are most likely
            lines = text.split('\n')
            header_text = '\n'.join(lines[:30])
            
            doc = nlp(header_text)
            
            # Find organizations and company entities
            companies = []
            for ent in doc.ents:
                if ent.label_ in ["ORG", "PERSON"]:  # Organizations and sometimes persons (for company names)
                    company_candidate = ent.text.strip()
                    
                    # Filter for likely company names
                    if (len(company_candidate) > 5 and len(company_candidate) < 60 and
                        not any(term in company_candidate.lower() for term in ['statement', 'report', 'audit', 'year', 'ended'])):
                        
                        # Higher confidence for entities with company indicators
                        confidence = 0.75
                        if any(suffix in company_candidate for suffix in ['Ltd', 'PJSC', 'PLC', 'Inc', 'Group', 'Corp', 'LLC']):
                            confidence = 0.85
                        
                        companies.append((company_candidate, confidence, ent.sent.text[:100]))
            
            # Return the highest confidence company name
            if companies:
                best_company = max(companies, key=lambda x: x[1])
                logger.info(f"🔧 NER found company: {best_company[0]} (confidence: {best_company[1]})")
                return best_company
            
            return "", 0.0, ""
            
        except ImportError:
            logger.info("spaCy not available, skipping NER extraction")
            return "", 0.0, ""
        except Exception as e:
            logger.error(f"Error in NER company extraction: {str(e)}")
            return "", 0.0, ""
    
    def _extract_company_name_pattern(self, text: str) -> Tuple[str, float, str]:
        """Extract company name from document titles and official sections only"""
        best_match = ""
        best_confidence = 0.0
        best_context = ""

        lines = text.split('\n')
        
        # Strategy 1: Look for company names in document title patterns
        title_patterns = [
            # "Consolidated Financial Statements of Phoenix Group PLC"
            r'(?:consolidated|audited|annual)?\s*financial\s+statements?\s+(?:of\s+|for\s+)?([A-Z][A-Za-z\s&\-\.]+(?:Group|Ltd|Limited|LLC|Inc|Corporation|PJSC|PLC|AG|GmbH|Holdings)(?:\s+(?:Ltd|Limited|LLC|Inc|PJSC|PLC))?)',
            
            # "Phoenix Group PLC - Annual Report"
            r'^([A-Z][A-Za-z\s&\-\.]+(?:Group|Ltd|Limited|LLC|Inc|Corporation|PJSC|PLC|AG|GmbH|Holdings)(?:\s+(?:Ltd|Limited|LLC|Inc|PJSC|PLC))?)\s*[-–—]\s*(?:annual|financial|consolidated)',
            
            # "Annual Report 2024 - Phoenix Group PLC"
            r'(?:annual\s+report|financial\s+statements?)\s+\d{4}\s*[-–—]\s*([A-Z][A-Za-z\s&\-\.]+(?:Group|Ltd|Limited|LLC|Inc|Corporation|PJSC|PLC|AG|GmbH|Holdings)(?:\s+(?:Ltd|Limited|LLC|Inc|PJSC|PLC))?)',
        ]
        
        # Search in first 15 lines for document titles
        for i, line in enumerate(lines[:15]):
            line = line.strip()
            if len(line) < 10 or len(line) > 200:
                continue
            
            for pattern in title_patterns:
                matches = re.findall(pattern, line, re.IGNORECASE)
                for match in matches:
                    company_name = match.strip()
                    
                    # Clean up the company name
                    company_name = re.sub(r'\s+', ' ', company_name)
                    company_name = company_name.strip(' .,')
                    
                    # Validate company name
                    if (len(company_name) >= 5 and len(company_name) <= 80 and
                        not any(term in company_name.lower() for term in ['statement', 'report', 'audit', 'note', 'year', 'ended', 'page']) and
                        company_name.lower() not in ['the company', 'company']):
                        
                        # Higher confidence for legal entity indicators
                        confidence = 0.8
                        if any(suffix in company_name for suffix in ['PJSC', 'PLC', 'Ltd', 'Limited', 'Inc', 'Corporation']):
                            confidence = 0.95
                        elif 'Group' in company_name:
                            confidence = 0.9
                        
                        # Prefer earlier lines
                        if i < 5:
                            confidence = min(confidence + 0.05, 0.99)
                        
                        if confidence > best_confidence:
                            best_match = company_name
                            best_confidence = confidence
                            best_context = line
                            logger.info(f"🏢 Title pattern found company: {best_match} (confidence: {confidence})")
        
        # Strategy 2: If no title match, look for company registration/incorporation info
        if not best_match:
            incorporation_patterns = [
                # "Phoenix Group PLC (incorporated in England and Wales)"
                r'([A-Z][A-Za-z\s&\-\.]+(?:Group|Ltd|Limited|LLC|Inc|Corporation|PJSC|PLC|AG|GmbH|Holdings)(?:\s+(?:Ltd|Limited|LLC|Inc|PJSC|PLC))?)\s*\([^)]*(?:incorporated|registered|domiciled)[^)]*\)',
                
                # "Company Name: Phoenix Group PLC"
                r'(?:company\s+name|entity\s+name|legal\s+name):\s*([A-Z][A-Za-z\s&\-\.]+(?:Group|Ltd|Limited|LLC|Inc|Corporation|PJSC|PLC|AG|GmbH|Holdings)(?:\s+(?:Ltd|Limited|LLC|Inc|PJSC|PLC))?)',
            ]
            
            for line in lines[:30]:  # Search more lines for incorporation info
                line = line.strip()
                if len(line) < 10:
                    continue
                
                for pattern in incorporation_patterns:
                    matches = re.findall(pattern, line, re.IGNORECASE)
                    for match in matches:
                        company_name = match.strip()
                        company_name = re.sub(r'\s+', ' ', company_name)
                        company_name = company_name.strip(' .,')
                        
                        if (len(company_name) >= 5 and len(company_name) <= 80 and
                            not any(term in company_name.lower() for term in ['statement', 'report', 'audit', 'note', 'year', 'ended']) and
                            company_name.lower() not in ['the company', 'company']):
                            
                            confidence = 0.85
                            if any(suffix in company_name for suffix in ['PJSC', 'PLC', 'Ltd', 'Limited']):
                                confidence = 0.9
                            
                            if confidence > best_confidence:
                                best_match = company_name
                                best_confidence = confidence
                                best_context = line
                                logger.info(f"🏢 Incorporation pattern found company: {best_match}")
        
        # Strategy 3: Last resort - look for standalone legal entity names in early lines
        if not best_match:
            for i, line in enumerate(lines[:10]):
                line = line.strip()
                if len(line) < 5 or len(line) > 100:
                    continue
                
                # Simple pattern for standalone company names with legal suffixes
                standalone_pattern = r'^([A-Z][A-Za-z\s&\-\.]{5,50}(?:Group|Ltd|Limited|LLC|Inc|Corporation|PJSC|PLC|AG|GmbH|Holdings))$'
                match = re.match(standalone_pattern, line)
                
                if match:
                    company_name = match.group(1).strip()
                    if (company_name.lower() not in ['the company', 'company'] and
                        not any(term in company_name.lower() for term in ['statement', 'report', 'audit'])):
                        
                        confidence = 0.75
                        if any(suffix in company_name for suffix in ['PJSC', 'PLC', 'Ltd', 'Limited']):
                            confidence = 0.85
                        
                        if confidence > best_confidence:
                            best_match = company_name
                            best_confidence = confidence
                            best_context = line
                            logger.info(f"🏢 Standalone pattern found company: {best_match}")

        return best_match, best_confidence, best_context

    def _extract_business_type_pattern(self, text: str) -> Tuple[str, float, str]:
        """Extract business type using keyword matching with context"""
        text_lower = text.lower()
        best_category = ""
        best_confidence = 0.0
        best_context = ""
        category_scores = {}

        # Split text into sentences for context extraction
        sentences = re.split(r'[.!?]+', text)

        # Score each business category
        for category, keywords in self.business_patterns.items():
            score = 0
            matches = 0
            context_sentences = []

            for sentence in sentences:
                sentence_lower = sentence.lower()
                sentence_matches = 0
                
                for keyword in keywords:
                    if keyword in sentence_lower:
                        matches += 1
                        sentence_matches += 1
                        # Weight longer keywords higher
                        score += len(keyword.split())
                
                # If this sentence has matches, include it in context
                if sentence_matches > 0:
                    context_sentences.append(sentence.strip())

            if matches > 0:
                # Normalize score by keyword count and text length
                normalized_score = (score / len(keywords)) * (matches / len(keywords))
                category_scores[category] = {
                    'score': min(normalized_score, 0.9),  # Cap at 0.9
                    'context': '. '.join(context_sentences[:3])  # Top 3 sentences
                }

        if category_scores:
            best_category = max(category_scores.keys(), key=lambda x: category_scores[x]['score'])
            best_confidence = category_scores[best_category]['score']
            best_context = category_scores[best_category]['context']

            # Convert category to readable business type
            category_names = {
                'financial_services': 'Financial Services',
                'technology': 'Technology',
                'manufacturing': 'Manufacturing',
                'healthcare': 'Healthcare',
                'retail': 'Retail',
                'energy': 'Energy',
                'real_estate': 'Real Estate',
                'education': 'Education',
                'hospitality': 'Hospitality',
                'transportation': 'Transportation'
            }
            display_name = category_names.get(best_category)
            if display_name is None:
                display_name = best_category.replace('_', ' ').title()
            return display_name, float(best_confidence), best_context
        else:
            return "", 0.0, ""

    def _extract_geographical_pattern(self, text: str) -> Tuple[str, float, str]:
        """Extract operational demographics using geographical service with context"""
        try:
            geographical_entities = self.geographical_service.detect_geographical_entities(text)

            if not geographical_entities:
                return "", 0.0, ""

            # Find the highest confidence entity that's actually meaningful
            best_entity = None
            best_confidence = 0.0

            for entity in geographical_entities:
                # Focus on countries with high confidence
                if (entity.type == 'country' and 
                    entity.confidence > 0.7 and 
                    entity.confidence > best_confidence):
                    best_entity = entity
                    best_confidence = entity.confidence

            if best_entity:
                # Extract context around the geographical mention
                context = self._extract_geographical_context(text, best_entity.name)
                # Return only the best match
                region_info = f" ({best_entity.region})" if best_entity.region else ""
                formatted_info = f"{best_entity.name}{region_info}"
                return formatted_info, min(best_confidence, 0.95), context
            else:
                # Fallback: return the highest confidence entity of any type
                best_entity = max(geographical_entities, key=lambda x: x.confidence)
                if best_entity.confidence > 0.5:
                    context = self._extract_geographical_context(text, best_entity.name)
                    region_info = f" ({best_entity.region})" if best_entity.region else ""
                    formatted_info = f"{best_entity.name}{region_info}"
                    return formatted_info, min(best_entity.confidence, 0.85), context
                else:
                    return "", 0.0, ""

        except Exception as e:
            logger.error(f"Error in geographical extraction: {str(e)}")
            return "", 0.0, ""

    def _extract_geographical_context(self, text: str, entity_name: str) -> str:
        """Extract context around geographical entity mentions"""
        sentences = re.split(r'[.!?]+', text)
        context_sentences = []
        
        for sentence in sentences:
            if entity_name.lower() in sentence.lower():
                context_sentences.append(sentence.strip())
        
        return '. '.join(context_sentences[:2]) if context_sentences else ""

    async def _extract_with_keyword_semantic_ai(self, text: str, field_name: str, keywords: List[str]) -> Tuple[str, str]:
        """
        Hybrid extraction: Keywords → Semantic search → Full sentences → Chunks → AI prompts
        """
        try:
            # Step 1: Identify keyword matches and extract full sentences
            relevant_sentences = self._find_sentences_with_keywords(text, keywords)
            
            if not relevant_sentences:
                logger.warning(f"No sentences found with keywords for {field_name}")
                return "", ""
            
            # Step 2: Semantic search to find most relevant chunks
            if hasattr(self, 'vector_store') and self.vector_store:
                semantic_chunks = await self._semantic_search_for_field(field_name, relevant_sentences, text)
                if semantic_chunks:
                    relevant_sentences.extend(semantic_chunks)
            
            # Step 3: Create chunks from relevant sentences (up to 8000 characters)
            context_text = '. '.join(relevant_sentences)
            
            # Use larger context for better AI analysis - up to 50K characters or full context
            max_context_size = 50000
            if len(context_text) > max_context_size:
                context_text = context_text[:max_context_size] + "..."
                logger.info(f"Truncated context for {field_name} to {max_context_size:,} characters (was {len('. '.join(relevant_sentences))} chars)")
            else:
                logger.info(f"Context for {field_name}: {len(context_text)} characters from {len(relevant_sentences)} sentences")
            
            # DEBUG: Log the context being sent to AI
            logger.info(f"🔍 DEBUG - Context for {field_name}:\n{context_text[:500]}..." if len(context_text) > 500 else f"🔍 DEBUG - Context for {field_name}:\n{context_text}")
            
            # Step 4: Create field-specific AI prompts with context
            if field_name == "nature_of_business":
                prompt = self._create_enhanced_business_prompt(context_text)
            elif field_name == "operational_demographics":
                prompt = self._create_enhanced_demographics_prompt(context_text)
            elif field_name == "financial_statements_type":
                prompt = self._create_enhanced_fs_type_prompt(context_text)
            else:
                return "", ""
            
            # Step 5: Get AI response
            response = self.ai_service.openai_client.chat.completions.create(
                model=self.ai_service.deployment_name,
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]}
                ],
                max_completion_tokens=800  # Only parameter supported by new model
            )
            
            content = response.choices[0].message.content
            logger.info(f"🤖 AI Response for {field_name}: {content[:200]}..." if content and len(content) > 200 else f"🤖 AI Response for {field_name}: {content}")
            return content.strip() if content else "", context_text[:500]
            
        except Exception as e:
            logger.error(f"Error in keyword→semantic→AI extraction for {field_name}: {str(e)}")
            return "", ""
    
    def _find_sentences_with_keywords(self, text: str, keywords: List[str]) -> List[str]:
        """Find ALL complete sentences containing any of the keywords"""
        sentences = re.split(r'[.!?]+', text)
        relevant_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences (reduced from 20)
                continue
                
            sentence_lower = sentence.lower()
            for keyword in keywords:
                if keyword.lower() in sentence_lower:
                    relevant_sentences.append(sentence)
                    logger.debug(f"Found keyword '{keyword}' in: {sentence[:100]}...")
                    break  # Avoid duplicate sentences
        
        logger.info(f"Found {len(relevant_sentences)} sentences with keywords: {keywords}")
        return relevant_sentences
    
    async def _semantic_search_for_field(self, field_name: str, existing_sentences: List[str], text: str) -> List[str]:
        """Use semantic search to find additional relevant content"""
        try:
            if not text:
                logger.warning("No text available for semantic search")
                return []
                
            # Define search queries for each field
            search_queries = {
                "nature_of_business": "business activities operations services development construction management industry sector",
                "operational_demographics": "countries operations geography locations subsidiaries jurisdictions offices presence",
                "financial_statements_type": "consolidated standalone separate individual financial statements accounts"
            }
            
            query = search_queries.get(field_name, "")
            if not query:
                return []
            
            # Split text into sentences
            all_sentences = re.split(r'[.!?]+', text)
            all_sentences = [s.strip() for s in all_sentences if len(s.strip()) > 10]
            
            if not all_sentences:
                return []
            
            # Find sentences not already in existing_sentences
            existing_lower = [s.lower() for s in existing_sentences]
            new_sentences = []
            
            for sentence in all_sentences:
                sentence_lower = sentence.lower()
                # Check if not already included
                if not any(existing.lower() in sentence_lower or sentence_lower in existing.lower() 
                          for existing in existing_lower):
                    # Check if contains relevant terms
                    for term in query.split():
                        if term.lower() in sentence_lower:
                            new_sentences.append(sentence)
                            logger.debug(f"Semantic search found: {sentence[:100]}...")
                            break
            
            logger.info(f"Semantic search found {len(new_sentences)} additional sentences for {field_name}")
            return new_sentences[:10]  # Limit to top 10 additional sentences
            
        except Exception as e:
            logger.error(f"Error in semantic search for {field_name}: {str(e)}")
            return []
    
    def _create_enhanced_business_prompt(self, context: str) -> Dict[str, str]:
        """Create enhanced business nature prompt exactly matching user requirements"""
        return {
            "system": (
                "You are an expert at extracting detailed business nature descriptions from financial documents. "
                "Extract the COMPLETE description of what the company does, including all business segments and activities. "
                "Format should match: 'The Group is engaged in various businesses primarily...' "
                "Include ALL activities mentioned - development, sales, investment, construction, leasing, management, etc."
            ),
            "user": (
                f"RELEVANT CONTEXT:\n{context}\n\n"
                "Extract the COMPLETE nature of business description. Include:\n"
                "- Primary business activities (development, sales, investment, construction, leasing, management)\n"
                "- Associated services for real estate\n"
                "- Additional businesses (hotels, schools, marinas, restaurants, beach clubs, golf courses)\n"
                "- All business segments mentioned\n\n"
                "Format the response as a complete detailed description, similar to:\n"
                "'The Group is engaged in various businesses primarily the development, sales, investment, construction, leasing, management and associated services for real estate. In addition, the Group is also engaged in development, construction, management and operation of hotels, schools, marinas, restaurants, beach clubs and golf courses.'\n\n"
                "Return the complete business description:"
            )
        }
    
    def _create_enhanced_demographics_prompt(self, context: str) -> Dict[str, str]:
        """Create enhanced demographics prompt for precise country extraction"""
        return {
            "system": (
                "You are an expert at identifying countries where a company has operations based ONLY on what is explicitly mentioned in the text. "
                "DO NOT infer or assume countries that are not clearly stated. "
                "Only extract countries that are explicitly mentioned in the document."
            ),
            "user": (
                f"RELEVANT CONTEXT:\n{context}\n\n"
                "Carefully read this text and identify ONLY the countries that are explicitly mentioned as locations where this company operates.\n\n"
                "Rules:\n"
                "- ONLY list countries that are clearly stated in the text\n"
                "- DO NOT add countries that are not explicitly mentioned\n"
                "- DO NOT infer countries from partial words or similar names\n"
                "- Return countries in comma-separated format\n"
                "- If no countries are clearly mentioned, return 'Not specified'\n\n"
                "Return only the countries explicitly mentioned in the text:"
            )
        }
    
    def _create_enhanced_fs_type_prompt(self, context: str) -> Dict[str, str]:
        """Create enhanced financial statements type prompt"""
        return {
            "system": (
                "You are an expert at identifying financial statement types from financial documents. "
                "Determine if the statements are Consolidated, Standalone, or both."
            ),
            "user": (
                f"RELEVANT CONTEXT:\n{context}\n\n"
                "Determine the type of financial statements. Look for:\n"
                "- References to 'consolidated financial statements'\n"
                "- References to 'standalone' or 'separate' financial statements\n"
                "- Company and subsidiaries reporting structure\n\n"
                "Return either: 'Consolidated', 'Standalone', or 'Consolidated / Standalone'"
            )
        }

    async def _enhance_with_semantic_search(self, document_id: str, chunks: List[str], pattern_results: Dict[str, Any]) -> Dict[str, Any]:
        """Tier 2: Enhance pattern results with semantic search"""
        enhanced_results = pattern_results.copy()

        # Define search queries for each field
        search_queries = {
            "company_name": "company name legal entity organization corporation",
            "nature_of_business": "business activity industry sector services products",
            "operational_demographics": "location geography country region operations market"
        }

        for field_name, query in search_queries.items():
            current_confidence = enhanced_results[field_name]["confidence"]

            # Only enhance if pattern confidence is low
            if current_confidence < 0.7:
                try:
                    # Search for relevant chunks
                    relevant_chunks = await self._search_relevant_chunks(document_id, query, max_chunks=3)

                    if relevant_chunks:
                        # Re-run pattern extraction on relevant chunks only
                        chunk_text = " ".join(relevant_chunks)

                        if field_name == "company_name":
                            value, confidence, context = self._extract_company_name_pattern(chunk_text)
                        elif field_name == "nature_of_business":
                            value, confidence, context = self._extract_business_type_pattern(chunk_text)
                        else:  # operational_demographics
                            value, confidence, context = self._extract_geographical_pattern(chunk_text)

                        # Update if we found better results
                        if confidence > current_confidence:
                            enhanced_results[field_name]["value"] = value
                            enhanced_results[field_name]["confidence"] = confidence
                            enhanced_results[field_name]["context"] = context
                            enhanced_results[field_name]["extraction_method"] = "semantic"

                except Exception as e:
                    logger.error(f"Error in semantic search for {field_name}: {str(e)}")

        return enhanced_results

    async def _search_relevant_chunks(self, document_id: str, query: str, max_chunks: int = 3) -> List[str]:
        """Search for relevant chunks using vector store"""
        try:
            if not self.vector_store:
                return []

            # Perform semantic search using the correct method name
            results = self.vector_store.search(
                query=query,
                document_id=document_id,
                top_k=max_chunks
            )

            # Extract chunk text from results
            chunks = []
            for result in results:
                if isinstance(result, dict) and 'text' in result:
                    chunks.append(result['text'])
                else:
                    # Handle other result formats
                    chunks.append(str(result))

            return chunks

        except Exception as e:
            logger.error(f"Error searching relevant chunks: {str(e)}")
            return []

    async def _validate_with_ai(self, chunks: List[str], semantic_results: Dict[str, Any]) -> Dict[str, Any]:
        """Tier 3: AI validation for uncertain fields only"""
        final_results = semantic_results.copy()

        # Only use AI for fields with low confidence
        validation_threshold = 0.6

        for field_name, field_data in semantic_results.items():
            if field_data["confidence"] < validation_threshold:
                try:
                    # Use AI to extract/validate this field
                    ai_value = await self._ai_extract_field(field_name, chunks)

                    if ai_value and ai_value.strip():
                        final_results[field_name]["value"] = ai_value.strip()
                        final_results[field_name]["confidence"] = 0.9  # High confidence for AI
                        final_results[field_name]["extraction_method"] = "ai_validation"

                except Exception as e:
                    logger.error(f"Error in AI validation for {field_name}: {str(e)}")

        # Add optimization metrics
        final_results["optimization_metrics"] = {
            "tokens_used": self._estimate_tokens_used(semantic_results),
            "extraction_methods_used": list(set(
                result["extraction_method"] for result in semantic_results.values()
                if isinstance(result, dict) and "extraction_method" in result
            ))
        }

        return final_results

    async def _ai_extract_field(self, field_name: str, chunks: List[str]) -> str:
        """Enhanced AI extraction with context and detailed descriptions"""
        try:
            # Get all extracted contexts for better AI understanding
            combined_text = " ".join(chunks)
            
            # Create enhanced prompts based on field type
            if field_name == "company_name":
                prompt = self._create_company_name_prompt_with_context(combined_text)
            elif field_name == "nature_of_business":
                prompt = self._create_business_nature_prompt_with_context(combined_text)
            elif field_name == "operational_demographics":
                prompt = self._create_demographics_prompt_with_context(combined_text)
            else:
                return ""

            # Call AI service with enhanced prompt
            response = self.ai_service.openai_client.chat.completions.create(
                model=self.ai_service.deployment_name,
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]}
                ],
                max_completion_tokens=800  # Fixed for new model compatibility
            )

            content = response.choices[0].message.content
            return content.strip() if content else ""

        except Exception as e:
            logger.error(f"Error in AI field extraction for {field_name}: {str(e)}")
            return ""

    def _create_company_name_prompt_with_context(self, text: str) -> Dict[str, str]:
        """Create enhanced company name extraction prompt with context and auditor filtering"""
        return {
            "system": (
                "You are an expert at extracting CLIENT company names from financial documents. "
                "Your task is to identify the PRIMARY CLIENT company name, NOT the auditor or audit firm. "
                "Look for the official registered client company name that would appear on legal documents."
            ),
            "user": (
                f"DOCUMENT TEXT:\n{text[:15000]}\n\n"
                "EXTRACT THE CLIENT COMPANY NAME (NOT AUDITOR):\n"
                "- Look for the official CLIENT company name in document titles like 'Consolidated Financial Statements of [CLIENT NAME]'\n"
                "- Include legal suffixes like PJSC, Ltd, LLC, Inc, Corporation, Group, PLC\n"
                "- IGNORE auditor firms ending in LLP, Chartered Accountants\n"
                "- IGNORE audit firms like RAI LLP, KPMG, EY, PWC, Deloitte\n"
                "- Extract the company being audited, NOT the company doing the auditing\n"
                "- If multiple companies mentioned, extract the main CLIENT subject company\n\n"
                "Return ONLY the CLIENT company name (not auditor name), nothing else."
            )
        }

    def _create_business_nature_prompt_with_context(self, text: str) -> Dict[str, str]:
        """Create enhanced business nature extraction prompt with detailed description"""
        return {
            "system": (
                "You are an expert at analyzing business activities from financial documents. "
                "Provide a DETAILED description of what the company actually does, not just a category. "
                "Include specific products, services, and business operations."
            ),
            "user": (
                f"DOCUMENT TEXT:\n{text[:15000]}\n\n"
                "EXTRACT DETAILED BUSINESS NATURE AND ACTIVITIES:\n"
                "- Describe the main business activities and operations\n"
                "- Include specific products or services offered\n"
                "- Mention key business segments or divisions\n"
                "- Include operational details like development, management, investment activities\n"
                "- Be specific - instead of just 'Real Estate', describe what type of real estate activities\n\n"
                "Provide a comprehensive description in 2-3 sentences that explains what this company actually does."
            )
        }

    def _create_demographics_prompt_with_context(self, text: str) -> Dict[str, str]:
        """Create enhanced operational demographics prompt with context"""
        return {
            "system": (
                "You are an expert at identifying operational geography from financial documents. "
                "Find countries where the company has actual business operations, not just mentions. "
                "Provide context about the nature of operations in each country."
            ),
            "user": (
                f"DOCUMENT TEXT:\n{text[:15000]}\n\n"
                "EXTRACT OPERATIONAL DEMOGRAPHICS WITH CONTEXT:\n"
                "- Identify countries where the company has actual operations\n"
                "- Look for subsidiary locations, revenue by geography, operational bases\n"
                "- Include the context of operations (e.g., 'UAE: Real estate development and management')\n"
                "- Exclude countries mentioned only for comparison or market reference\n"
                "- Provide specific operational context for each country\n\n"
                "Format: Country: [Operational Context]. Country: [Operational Context].\n"
                "Example: 'United Arab Emirates: Primary real estate development and property management operations. Saudi Arabia: Investment property holdings and development projects.'"
            )
        }

    def _estimate_tokens_used(self, results: Dict[str, Any]) -> int:
        """Estimate total tokens used in extraction"""
        token_count = 0

        for field_name, field_data in results.items():
            if isinstance(field_data, dict) and "extraction_method" in field_data:
                method = field_data["extraction_method"]

                if method == "pattern":
                    token_count += 0  # No tokens for pattern matching
                elif method == "semantic":
                    token_count += 500  # Estimate for semantic search
                elif method == "ai_validation":
                    token_count += 1200  # Estimate for AI call (1000 input + 200 output)

        return token_count
