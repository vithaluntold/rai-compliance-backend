#!/usr/bin/env python3
"""
AI Content Classification Engine (Tool 3)
Extends the 5D tagging system to classify document content sections

This module applies the same master dictionary and 5D classification framework
used for questions to automatically classify document content into Accounting
Standards (IAS 2, IFRS 7, etc.) and generate identical tag structures.
"""

import os
import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables  
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ContentSegment:
    """A classified content segment with 5D tags"""
    content_text: str
    segment_type: str  # from enhanced_structure_parser
    accounting_standard: Optional[str] = None
    paragraph_hint: Optional[str] = None
    classification_tags: Optional[Dict[str, Any]] = None
    confidence_score: float = 0.0
    page_number: int = 0
    source_document: str = ""
    
@dataclass
class ClassificationResult:
    """Result of content classification operation"""
    success: bool
    segments: Optional[List[ContentSegment]] = None
    total_segments: int = 0
    error: Optional[str] = None

class AIContentClassificationEngine:
    """AI-powered content classifier using 5D tagging system"""
    
    def __init__(self):
        """Initialize the AI content classification engine"""
        self.master_dictionary = self._load_master_dictionary()
        self.accounting_standards_map = self._load_accounting_standards_map()
        self.content_classification_prompt = self._create_content_prompt()
        
        # Initialize OpenAI client (optional)
        self.client = None
        try:
            from openai import OpenAI
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.client = OpenAI(api_key=api_key)
            else:
                logger.info("OpenAI API key not set - using pattern-based classification")
        except ImportError:
            logger.info("OpenAI package not available - using pattern-based classification")
            
    def _load_master_dictionary(self) -> Dict[str, Any]:
        """Load the master dictionary for controlled vocabularies"""
        try:
            dictionary_path = os.path.join(os.path.dirname(__file__), '..', 'ai_parser', 'master_dictionary.json')
            with open(dictionary_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load master dictionary: {e}")
            return {}
            
    def _load_accounting_standards_map(self) -> Dict[str, Dict[str, Any]]:
        """Load mapping of content patterns to accounting standards"""
        return {
            # IAS Standards
            "IAS 1": {
                "keywords": ["presentation", "financial statements", "statement of financial position", 
                           "statement of profit or loss", "statement of comprehensive income"],
                "patterns": [r"presentation.*financial\s+statements", r"statement.*financial\s+position",
                           r"profit.*loss", r"comprehensive\s+income"],
                "content_types": ["primary_statements", "presentation_policies"]
            },
            "IAS 2": {
                "keywords": ["inventories", "cost of inventories", "net realizable value", "inventory valuation"],
                "patterns": [r"inventor(y|ies)", r"cost.*inventor", r"net\s+realizable\s+value", 
                           r"inventory.*valuation", r"cost.*goods\s+sold"],
                "content_types": ["measurement_basis", "accounting_policies_note"]
            },
            "IAS 7": {
                "keywords": ["cash flows", "cash flow statement", "operating activities", "investing activities", 
                           "financing activities"],
                "patterns": [r"cash\s+flow", r"operating.*activities", r"investing.*activities", 
                           r"financing.*activities"],
                "content_types": ["primary_statements", "cash_flow_analysis"]
            },
            "IAS 8": {
                "keywords": ["accounting policies", "changes in estimates", "errors", "prior period errors"],
                "patterns": [r"accounting\s+policies", r"changes.*estimates", r"prior\s+period.*errors", 
                           r"retrospective.*application"],
                "content_types": ["accounting_policies_note", "estimates_judgements"]
            },
            "IAS 10": {
                "keywords": ["events after reporting period", "subsequent events", "adjusting events", 
                           "non-adjusting events"],
                "patterns": [r"events.*after.*reporting", r"subsequent\s+events", r"adjusting\s+events", 
                           r"non-adjusting\s+events"],
                "content_types": ["event_based", "subsequent_events"]
            },
            "IAS 12": {
                "keywords": ["income taxes", "deferred tax", "current tax", "tax assets", "tax liabilities"],
                "patterns": [r"income\s+tax", r"deferred\s+tax", r"current\s+tax", r"tax.*assets", 
                           r"tax.*liabilities"],
                "content_types": ["measurement_basis", "reconciliation_table"]
            },
            "IAS 16": {
                "keywords": ["property, plant and equipment", "PPE", "depreciation", "revaluation", "impairment"],
                "patterns": [r"property.*plant.*equipment", r"depreciation", r"revaluation", 
                           r"carrying\s+amount", r"useful\s+life"],
                "content_types": ["measurement_basis", "roll_forward_table"]
            },
            "IAS 24": {
                "keywords": ["related party", "related party disclosures", "key management personnel"],
                "patterns": [r"related\s+part(y|ies)", r"key\s+management", r"ultimate.*controlling"],
                "content_types": ["related_party", "disclosure_narrative"]
            },
            "IAS 38": {
                "keywords": ["intangible assets", "development costs", "amortisation", "indefinite useful life"],
                "patterns": [r"intangible\s+assets", r"development\s+costs", r"amortisation", 
                           r"indefinite.*useful\s+life"],
                "content_types": ["measurement_basis", "roll_forward_table"]
            },
            # IFRS Standards
            "IFRS 7": {
                "keywords": ["financial instruments", "fair value", "credit risk", "liquidity risk", 
                           "market risk", "hedge accounting"],
                "patterns": [r"financial\s+instruments", r"fair\s+value", r"credit\s+risk", 
                           r"liquidity\s+risk", r"market\s+risk", r"hedge\s+accounting"],
                "content_types": ["risk_exposure", "sensitivity_table"]
            },
            "IFRS 9": {
                "keywords": ["financial instruments", "expected credit losses", "ECL", "impairment"],
                "patterns": [r"expected\s+credit\s+losses", r"ECL", r"impairment.*financial", 
                           r"lifetime.*expected"],
                "content_types": ["measurement_basis", "estimates_judgements"]
            },
            "IFRS 15": {
                "keywords": ["revenue from contracts", "performance obligations", "transaction price", 
                           "contract assets", "contract liabilities"],
                "patterns": [r"revenue.*contracts", r"performance\s+obligations", r"transaction\s+price", 
                           r"contract\s+assets", r"contract\s+liabilities"],
                "content_types": ["accounting_policies_note", "disclosure_narrative"]
            },
            "IFRS 16": {
                "keywords": ["leases", "right-of-use assets", "lease liabilities", "lessee", "lessor"],
                "patterns": [r"lease", r"right-of-use\s+assets", r"lease\s+liabilities", 
                           r"lessee", r"lessor"],
                "content_types": ["measurement_basis", "roll_forward_table"]
            }
        }
        
    def _create_content_prompt(self) -> str:
        """Create prompt template for content classification"""
        return """
You are an AI assistant specialized in classifying financial statement content sections using the 5D framework.
Analyze the document content and identify which Accounting Standard it relates to, then apply appropriate 5D tags.

CONTENT TO ANALYZE:
{content_text}

SEGMENT TYPE: {segment_type}
ACCOUNTING STANDARD DETECTED: {accounting_standard}

Apply the same 5D classification framework used for questions, but adapted for content:

MASTER DICTIONARY (same as questions):
- narrative_categories: accounting_policies_note, event_based, measurement_basis, risk_exposure, estimates_judgements, related_party, going_concern, disclosure_narrative
- table_archetypes: reconciliation_table, roll_forward_table, sensitivity_table, segmental_analysis, carrying_amounts_by_category, maturity_analysis  
- quantitative_expectations: absolute_amounts, class_by_class_totals, estimate_of_financial_effect, comparative_amounts, qualitative_only
- temporal_scope: current_period, comparative_period, multiple_periods, subsequent_events, opening_balance
- cross_reference_anchors: notes_main, policies, primary_statement, segment_note, risk_note, related_standards

CLASSIFICATION RULES FOR CONTENT:
- If content contains accounting policies → accounting_policies_note + policies
- If content shows tables with movements → roll_forward_table or reconciliation_table
- If content discusses risks → risk_exposure + risk_note  
- If content about estimates/judgments → estimates_judgements
- If content includes numerical data → appropriate quantitative_expectations
- If content is narrative only → qualitative_only

Return EXACTLY this JSON format:

{{
  "accounting_standard": "{accounting_standard}",
  "paragraph_hint": "extract_relevant_paragraph_or_topic",
  "facet_focus": {{
    "narrative_categories": ["select_from_master_dictionary"],
    "table_archetypes": ["select_if_applicable"],
    "quantitative_expectations": ["select_from_master_dictionary"], 
    "temporal_scope": ["select_from_master_dictionary"],
    "cross_reference_anchors": ["select_from_master_dictionary"]
  }},
  "conditionality": {{
    "trigger_conditions": ["content_based_requirement"],
    "dependency_chain": ["identify_content_requirement"],
    "exception_scenarios": ["not_applicable_if_no_content"]
  }},
  "evidence_expectations": {{
    "required_documents": ["financial_statements"],
    "data_sources": ["accounting_records"],
    "validation_methods": ["document_review"],
    "quality_indicators": ["completeness_check"]
  }},
  "retrieval_support": {{
    "search_keywords": ["extract_key_terms_from_content"],
    "section_indicators": ["determine_section_type"],
    "pattern_matching": ["standard_format"],
    "context_clues": ["financial_context"]
  }},
  "citation_controls": {{
    "required_disclosures": ["accounting_policy"],
    "cross_references": ["related_standards"],
    "compliance_markers": ["regulatory_compliance"]
  }}
}}
"""

    def classify_content_segment(self, content_text: str, segment_type: str, 
                               page_number: int = 0, source_document: str = "") -> ContentSegment:
        """Classify a single content segment and generate 5D tags"""
        
        # Step 1: Detect accounting standard
        accounting_standard, confidence = self._detect_accounting_standard(content_text)
        
        # Step 2: Apply AI classification using same engine as questions
        classification_tags = self._classify_content_with_ai(
            content_text, segment_type, accounting_standard
        )
        
        # Step 3: Create classified content segment
        segment = ContentSegment(
            content_text=content_text,
            segment_type=segment_type,
            accounting_standard=accounting_standard,
            paragraph_hint=self._extract_paragraph_hint(content_text, accounting_standard),
            classification_tags=classification_tags,
            confidence_score=confidence,
            page_number=page_number,
            source_document=source_document
        )
        
        return segment
        
    def _detect_accounting_standard(self, content_text: str) -> Tuple[str, float]:
        """Detect which accounting standard the content relates to"""
        
        content_lower = content_text.lower()
        best_match = None
        best_score = 0.0
        
        for standard, config in self.accounting_standards_map.items():
            score = 0.0
            
            # Check keyword matches
            keyword_matches = sum(1 for keyword in config["keywords"] 
                                if keyword.lower() in content_lower)
            score += keyword_matches * 2.0
            
            # Check pattern matches  
            pattern_matches = sum(1 for pattern in config["patterns"]
                                if re.search(pattern, content_lower))
            score += pattern_matches * 3.0
            
            # Normalize score by number of total indicators
            total_indicators = len(config["keywords"]) + len(config["patterns"])
            if total_indicators > 0:
                normalized_score = score / total_indicators
                
                if normalized_score > best_score:
                    best_score = normalized_score
                    best_match = standard
                    
        # Set confidence threshold
        confidence = min(best_score, 1.0) if best_match else 0.0
        
        return best_match or "General", confidence
        
    def _extract_paragraph_hint(self, content_text: str, accounting_standard: str) -> str:
        """Extract a relevant paragraph hint from the content"""
        
        # Extract first meaningful sentence or key topic
        sentences = content_text.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and len(sentence) < 150:
                # Clean up and return first good sentence
                return sentence.replace('\n', ' ').replace('\t', ' ')
                
        # Fallback: create hint from accounting standard and content type
        if accounting_standard != "General":
            return f"{accounting_standard} related disclosure"
            
        return "General financial statement disclosure"
        
    def _classify_content_with_ai(self, content_text: str, segment_type: str, 
                                 accounting_standard: str) -> Dict[str, Any]:
        """Use AI to classify content with 5D tags"""
        
        if not self.client:
            return self._classify_content_with_patterns(content_text, segment_type, accounting_standard)
            
        try:
            prompt = self.content_classification_prompt.format(
                content_text=content_text[:2000],  # Limit content for API
                segment_type=segment_type,
                accounting_standard=accounting_standard
            )
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert in IFRS/IAS content classification. Apply 5D tags consistently with the question classification system."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            # Parse response
            content = response.choices[0].message.content
            if content:
                content = content.strip()
            else:
                content = "{}"
            
            # Clean up JSON response
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
                
            classification = json.loads(content)
            return classification
            
        except Exception as e:
            logger.error(f"AI content classification failed: {e}")
            return self._classify_content_with_patterns(content_text, segment_type, accounting_standard)
            
    def _classify_content_with_patterns(self, content_text: str, segment_type: str, 
                                      accounting_standard: str) -> Dict[str, Any]:
        """Classify content using pattern matching when AI is not available"""
        
        content_lower = content_text.lower()
        
        # Determine narrative categories based on content patterns
        narrative_categories = []
        
        if any(term in content_lower for term in ['accounting policy', 'accounting policies', 'measurement basis']):
            narrative_categories.append('accounting_policies_note')
        elif any(term in content_lower for term in ['risk', 'credit risk', 'market risk', 'liquidity risk']):
            narrative_categories.append('risk_exposure')
        elif any(term in content_lower for term in ['estimate', 'judgment', 'assumption']):
            narrative_categories.append('estimates_judgements')
        elif any(term in content_lower for term in ['related party', 'key management']):
            narrative_categories.append('related_party')
        elif any(term in content_lower for term in ['subsequent event', 'after reporting']):
            narrative_categories.append('event_based')
        elif any(term in content_lower for term in ['fair value', 'amortised cost', 'measurement']):
            narrative_categories.append('measurement_basis')
        else:
            narrative_categories.append('disclosure_narrative')
            
        # Determine table archetypes
        table_archetypes = []
        if any(term in content_lower for term in ['movement', 'opening', 'closing', 'balance']):
            table_archetypes.append('reconciliation_table')
        elif any(term in content_lower for term in ['carrying amount', 'by class']):
            table_archetypes.append('roll_forward_table')
        elif any(term in content_lower for term in ['sensitivity', 'stress test']):
            table_archetypes.append('sensitivity_table')
        elif any(term in content_lower for term in ['segment', 'geographic', 'business']):
            table_archetypes.append('segmental_analysis')
        elif any(term in content_lower for term in ['maturity', 'profile']):
            table_archetypes.append('maturity_analysis')
            
        # Determine quantitative expectations
        if re.search(r'\$\d+|\d+,\d+|\d+\.\d+', content_text):
            if any(term in content_lower for term in ['by class', 'category', 'breakdown']):
                quantitative_expectations = ['class_by_class_totals']
            else:
                quantitative_expectations = ['absolute_amounts']
        else:
            quantitative_expectations = ['qualitative_only']
            
        # Determine temporal scope
        if any(term in content_lower for term in ['prior year', 'comparative', 'previous period']):
            temporal_scope = ['comparative_period']
        elif any(term in content_lower for term in ['subsequent', 'after reporting']):
            temporal_scope = ['subsequent_events']
        elif any(term in content_lower for term in ['opening', 'beginning']):
            temporal_scope = ['opening_balance']
        else:
            temporal_scope = ['current_period']
            
        # Determine cross-reference anchors based on segment type
        cross_ref_map = {
            'statement_of_financial_position': ['primary_statement'],
            'statement_of_profit_loss': ['primary_statement'], 
            'statement_of_cash_flows': ['primary_statement'],
            'notes': ['notes_main'],
            'accounting_policies': ['policies'],
            'auditor_report': ['related_standards']
        }
        cross_reference_anchors = cross_ref_map.get(segment_type, ['notes_main'])
        
        # Extract search keywords
        search_keywords = self._extract_search_keywords(content_text)
        
        return {
            "accounting_standard": accounting_standard,
            "paragraph_hint": self._extract_paragraph_hint(content_text, accounting_standard),
            "facet_focus": {
                "narrative_categories": narrative_categories,
                "table_archetypes": table_archetypes,
                "quantitative_expectations": quantitative_expectations,
                "temporal_scope": temporal_scope,
                "cross_reference_anchors": cross_reference_anchors
            },
            "conditionality": {
                "trigger_conditions": ["content_based_requirement"],
                "dependency_chain": ["identify_content_requirement"],
                "exception_scenarios": ["not_applicable"]
            },
            "evidence_expectations": {
                "required_documents": ["financial_statements"],
                "data_sources": ["accounting_records"],
                "validation_methods": ["document_review"],
                "quality_indicators": ["completeness_check"]
            },
            "retrieval_support": {
                "search_keywords": search_keywords,
                "section_indicators": [segment_type],
                "pattern_matching": ["standard_format"],
                "context_clues": ["financial_position"]
            },
            "citation_controls": {
                "required_disclosures": ["accounting_policy"],
                "cross_references": ["related_standards"],
                "compliance_markers": ["regulatory_compliance"]
            }
        }
            
    def _get_default_content_classification(self, segment_type: str, 
                                          accounting_standard: str) -> Dict[str, Any]:
        """Return default classification for content when AI fails"""
        
        # Map segment types to appropriate 5D tags
        segment_mapping = {
            "statement_of_financial_position": {
                "narrative_categories": ["disclosure_narrative"],
                "cross_reference_anchors": ["primary_statement"]
            },
            "statement_of_profit_loss": {
                "narrative_categories": ["disclosure_narrative"],
                "cross_reference_anchors": ["primary_statement"]
            },
            "statement_of_cash_flows": {
                "narrative_categories": ["disclosure_narrative"],
                "cross_reference_anchors": ["primary_statement"]
            },
            "notes": {
                "narrative_categories": ["disclosure_narrative"],
                "cross_reference_anchors": ["notes_main"]
            },
            "accounting_policies": {
                "narrative_categories": ["accounting_policies_note"],
                "cross_reference_anchors": ["policies"]
            },
            "auditor_report": {
                "narrative_categories": ["going_concern"],
                "cross_reference_anchors": ["related_standards"]
            }
        }
        
        segment_config = segment_mapping.get(segment_type, {
            "narrative_categories": ["disclosure_narrative"],
            "cross_reference_anchors": ["notes_main"]
        })
        
        return {
            "accounting_standard": accounting_standard,
            "paragraph_hint": f"{segment_type} content",
            "facet_focus": {
                "narrative_categories": segment_config["narrative_categories"],
                "table_archetypes": [],
                "quantitative_expectations": ["qualitative_only"],
                "temporal_scope": ["current_period"],
                "cross_reference_anchors": segment_config["cross_reference_anchors"]
            },
            "conditionality": {
                "trigger_conditions": ["content_based_requirement"],
                "dependency_chain": ["identify_content_requirement"],
                "exception_scenarios": ["not_applicable"]
            },
            "evidence_expectations": {
                "required_documents": ["financial_statements"],
                "data_sources": ["accounting_records"],
                "validation_methods": ["document_review"],
                "quality_indicators": ["completeness_check"]
            },
            "retrieval_support": {
                "search_keywords": [],
                "section_indicators": [segment_type],
                "pattern_matching": ["standard_format"],
                "context_clues": ["financial_position"]
            },
            "citation_controls": {
                "required_disclosures": ["accounting_policy"],
                "cross_references": ["related_standards"],
                "compliance_markers": ["regulatory_compliance"]
            }
        }
        
    def _extract_search_keywords(self, content_text: str) -> List[str]:
        """Extract relevant search keywords from content"""
        
        # Common financial keywords
        financial_keywords = [
            "financial", "statement", "disclosure", "accounting", "policy",
            "measurement", "recognition", "assets", "liabilities", "equity",
            "revenue", "expenses", "cash", "flows", "comprehensive", "income"
        ]
        
        content_lower = content_text.lower()
        found_keywords = []
        
        for keyword in financial_keywords:
            if keyword in content_lower:
                found_keywords.append(keyword)
                
        return found_keywords[:5]  # Limit to top 5 keywords
        
    def classify_document_segments(self, segments: List[Dict[str, Any]], 
                                 source_document: str = "") -> ClassificationResult:
        """Classify multiple document segments from enhanced structure parser"""
        
        try:
            classified_segments = []
            
            for segment_data in segments:
                # Extract segment information
                content_text = segment_data.get('content', '')
                segment_type = segment_data.get('segment_type', 'unknown')
                page_number = segment_data.get('page_num', 0)
                
                # Skip empty segments
                if not content_text.strip():
                    continue
                    
                # Classify the segment
                classified_segment = self.classify_content_segment(
                    content_text=content_text,
                    segment_type=segment_type,
                    page_number=page_number,
                    source_document=source_document
                )
                
                classified_segments.append(classified_segment)
                
            return ClassificationResult(
                success=True,
                segments=classified_segments,
                total_segments=len(classified_segments)
            )
            
        except Exception as e:
            logger.error(f"Document segment classification failed: {e}")
            return ClassificationResult(
                success=False,
                error=f"Classification failed: {str(e)}"
            )
            
    def create_mega_chunk_by_standard(self, segments: List[ContentSegment]) -> Dict[str, Dict[str, Any]]:
        """Group classified segments by accounting standard for mega-chunk creation"""
        
        mega_chunks = {}
        
        for segment in segments:
            standard = segment.accounting_standard or "General"
            
            if standard not in mega_chunks:
                mega_chunks[standard] = {
                    "accounting_standard": standard,
                    "sub_chunks": [],
                    "total_segments": 0,
                    "combined_tags": self._merge_classification_tags([t for t in [segment.classification_tags] if t is not None]),
                    "confidence_score": segment.confidence_score
                }
                
            # Add segment as sub-chunk
            sub_chunk = {
                "content_text": segment.content_text,
                "segment_type": segment.segment_type,
                "paragraph_hint": segment.paragraph_hint,
                "classification_tags": segment.classification_tags,
                "page_number": segment.page_number,
                "confidence_score": segment.confidence_score
            }
            
            mega_chunks[standard]["sub_chunks"].append(sub_chunk)
            mega_chunks[standard]["total_segments"] += 1
            
            # Update combined confidence (average)
            total_confidence = sum(sc.get("confidence_score", 0) for sc in mega_chunks[standard]["sub_chunks"])
            mega_chunks[standard]["confidence_score"] = total_confidence / mega_chunks[standard]["total_segments"]
            
        return mega_chunks
        
    def _merge_classification_tags(self, tag_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple 5D tag sets into a combined set for mega-chunks"""
        
        if not tag_list or not tag_list[0]:
            return self._get_default_content_classification("general", "General")
            
        # Take first non-null classification as base
        merged_tags = {}
        for tags in tag_list:
            if tags:
                merged_tags = tags.copy()
                break
                
        return merged_tags