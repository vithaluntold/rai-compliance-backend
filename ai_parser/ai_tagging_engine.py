#!/usr/bin/env python3
"""
AI Checklist Parser - Intelligent Tagging Engine
Comprehensive IFRS/IAS Classification System with Controlled Vocabularies
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class TaggingResult:
    """Result of AI tagging operation"""
    success: bool
    classification: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
class IntelligentAITaggingEngine:
    """Master dictionary-based IFRS/IAS classification engine"""
    
    def __init__(self):
        self.master_dictionary = self._load_master_dictionary()
        self.prompt_template = self._create_intelligent_prompt()
    
    def _load_master_dictionary(self) -> Dict[str, Any]:
        """Load the master dictionary for controlled vocabularies"""
        try:
            dictionary_path = os.path.join(os.path.dirname(__file__), 'master_dictionary.json')
            with open(dictionary_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load master dictionary: {e}")
            return {}
        
    def _create_intelligent_prompt(self) -> str:
        """Create master dictionary-based classification prompt"""
        return """
You are an AI assistant responsible for classifying IFRS/IAS disclosure checklist questions into structured tags.
Your output must always be valid JSON with the same schema.
You must use the controlled vocabulary (master dictionary) provided below.
Do not invent new values. Do not leave fields empty unless truly not applicable.

MASTER DICTIONARY

narrative_categories:
• accounting_policies_note → questions on accounting policies, methods, measurement bases
• event_based → questions on specific events (subsequent events, acquisitions, restructuring)
• measurement_basis → fair value, amortised cost, impairment
• risk_exposure → credit risk, liquidity risk, market risk, sensitivity analysis
• estimates_judgements → estimates, assumptions, sources of estimation uncertainty
• related_party → transactions or balances with related parties
• going_concern → disclosures on going concern assumption
• disclosure_narrative → general narrative disclosures not fitting other categories

table_archetypes:
• reconciliation_table → movement schedules (opening/closing balances)
• roll_forward_table → carrying amounts by class across periods
• sensitivity_table → sensitivity or stress-test tables
• segmental_analysis → by business/geographic segment
• carrying_amounts_by_category → totals by category (inventories, PPE, instruments)
• maturity_analysis → maturity profile tables (risk, cash flows)

quantitative_expectations:
• absolute_amounts → single amounts or totals
• class_by_class_totals → breakdown by class or category
• estimate_of_financial_effect → estimates, ranges, financial effects of events
• comparative_amounts → disclosures requiring prior year comparatives
• qualitative_only → narrative without amounts

temporal_scope:
• current_period → applies to current reporting period
• comparative_period → requires prior period comparison
• multiple_periods → spans several years
• subsequent_events → events after the reporting date
• opening_balance → beginning of period disclosures

cross_reference_anchors:
• notes_main → general notes to financial statements
• policies → accounting policies section
• primary_statement → SoFP, P&L, OCI, etc.
• segment_note → segment reporting
• risk_note → risk disclosures
• related_standards → cross-references to other IFRS/IAS

CLASSIFICATION RULES:
• If about policies/methods → use accounting_policies_note
• If about specific events after reporting period → use event_based + subsequent_events
• If about measurement basis → use measurement_basis
• If about risk → use risk_exposure + sensitivity_table if numbers expected
• If about estimates/judgements → use estimates_judgements
• If table implied (movements, balances, maturity) → choose correct table_archetype
• If numbers required → pick absolute_amounts or class_by_class_totals
• If only narrative needed → set quantitative_expectations = qualitative_only

MINIMUM TAGGING MUST INCLUDE:
• 1 narrative_categories
• 1 temporal_scope
• 1 citation_controls.compliance_markers

QUESTION: {question_text}
CONTEXT: {context}

Return EXACTLY this JSON format with proper classifications from the master dictionary:

{{
  "facet_focus": {{
    "narrative_categories": ["select_from_master_dictionary"],
    "table_archetypes": ["select_if_applicable"],
    "quantitative_expectations": ["select_from_master_dictionary"],
    "temporal_scope": ["select_from_master_dictionary"],
    "cross_reference_anchors": ["select_from_master_dictionary"]
  }},
  "conditionality": {{
    "trigger_conditions": ["standard_requirement"],
    "dependency_chain": ["identify_requirement"],
    "exception_scenarios": ["not_applicable"]
  }},
  "evidence_expectations": {{
    "required_documents": ["financial_statements"],
    "data_sources": ["accounting_records"],
    "validation_methods": ["document_review"],
    "quality_indicators": ["completeness_check"]
  }},
  "retrieval_support": {{
    "search_keywords": ["extract_relevant_keywords"],
    "section_indicators": ["notes"],
    "pattern_matching": ["standard_format"],
    "context_clues": ["financial_position"]
  }},
  "citation_controls": {{
    "required_disclosures": ["select_if_applicable"],
    "cross_references": ["related_standards"],
    "compliance_markers": ["regulatory_compliance"]
  }}
}}

CRITICAL: Always follow master dictionary values. Do not invent new categories. Be consistent across all questions.
"""

    def classify_question(self, question_text: str, context: str = "") -> TaggingResult:
        """
        Classify a single question using intelligent pattern recognition
        
        Args:
            question_text: The checklist question to classify
            context: Additional context from the compliance document
            
        Returns:
            TaggingResult with classification or error
        """
        try:
            prompt = self.prompt_template.format(
                question_text=question_text,
                context=context
            )
            
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert accounting standards analyst specializing in intelligent pattern-based classification. Never use default values - only classify based on actual question content patterns."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            # Parse the JSON response
            content = response.choices[0].message.content
            if content:
                content = content.strip()
            else:
                content = "{}"
            
            # Clean up the response if it contains markdown code blocks
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
                
            classification = json.loads(content)
            
            return TaggingResult(success=True, classification=classification)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            return TaggingResult(
                success=False, 
                classification=self._get_default_classification(),
                error=f"JSON parsing failed: {str(e)}"
            )
        except Exception as e:
            logger.error(f"AI classification error: {e}")
            return TaggingResult(
                success=False,
                classification=self._get_default_classification(),
                error=f"AI processing failed: {str(e)}"
            )
    
    def _get_default_classification(self) -> Dict[str, Any]:
        """Return default classification using master dictionary values"""
        return {
            "facet_focus": {
                "narrative_categories": ["disclosure_narrative"],
                "table_archetypes": [],
                "quantitative_expectations": ["qualitative_only"],
                "temporal_scope": ["current_period"],
                "cross_reference_anchors": ["notes_main"]
            },
            "conditionality": {
                "trigger_conditions": ["standard_requirement"],
                "dependency_chain": ["identify_requirement"],
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
                "section_indicators": ["notes"],
                "pattern_matching": ["standard_format"],
                "context_clues": ["financial_position"]
            },
            "citation_controls": {
                "required_disclosures": [],
                "cross_references": ["related_standards"],
                "compliance_markers": ["regulatory_compliance"]
            }
        }

    def enhance_question(self, question: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance a single question with AI-generated 5D tags
        
        Args:
            question: Original question dictionary
            
        Returns:
            Enhanced question with 5D classification tags
        """
        enhanced_question = question.copy()
        
        # Extract question text and context
        question_text = question.get('question', '')
        context = f"Section: {question.get('section', '')}, Reference: {question.get('reference', '')}"
        
        # Get AI classification
        result = self.classify_question(question_text, context)
        
        if result.success and result.classification:
            # Merge the comprehensive classification into the question
            enhanced_question.update(result.classification)
            logger.info(f"Successfully enhanced question {question.get('id', 'unknown')}")
        else:
            # Use default classification on failure
            enhanced_question.update(result.classification or self._get_default_classification())
            logger.warning(f"Used default classification for question {question.get('id', 'unknown')}: {result.error}")
        
        return enhanced_question