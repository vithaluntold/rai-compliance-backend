#!/usr/bin/env python3
"""
AI Checklist Parser - Intelligent Tagging Engine
Pattern-Based 5D Classification System for Accounting Standards

No hardcoded rules - pure pattern recognition and intelligent inference.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# OpenAI configuration
openai.api_key = os.getenv('OPENAI_API_KEY')

@dataclass
class TaggingResult:
    """Result of AI tagging operation"""
    success: bool
    classification: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
class IntelligentAITaggingEngine:
    """Pattern-based 5D classification tagging engine with zero hardcoding"""
    
    def __init__(self):
        self.prompt_template = self._create_intelligent_prompt()
        
    def _create_intelligent_prompt(self) -> str:
        """Create intelligent pattern-based prompt with no hardcoded rules"""
        return """
You are an expert accounting standards analyst. Analyze the question and intelligently classify based on CONTENT PATTERNS.

QUESTION: {question_text}
CONTEXT: {context}

INSTRUCTIONS:
Analyze the question's actual content and infer appropriate classifications. Use pattern recognition, not default values.

PATTERN RECOGNITION GUIDE:

1. NARRATIVE CATEGORIES - What type of disclosure is this?
   - policy_basis: Contains "policy", "basis", "method", "criteria", "recognition"
   - methodology: Contains "measurement", "valuation", "calculated", "approach", "determined" 
   - judgement_estimate: Contains "estimate", "assumption", "judgement", "uncertain", "key assumption"
   - risk_strategy: Contains "risk", "management", "objective", "strategy"
   - derecognition_explainer: Contains "derecognit", "transfer", "extinguish", "removal"
   - hedge_strategy: Contains "hedge", "designat", "effective", "hedge accounting"
   - transition_adoption: Contains "first-time", "adoption", "transition", "new standard"
   - changes_during_period: Contains "change", "reclassif", "restat", "revised"
   - contingent_events: Contains "contingent", "commitment", "subsequent", "event", "after reporting"
   - industry_specific_policy: Contains industry-specific terms

2. TABLE ARCHETYPES - What format would the disclosure take?
   - carrying_amounts_by_category: Asks for "amounts by class/category/type"
   - movement_reconciliation: Asks for "movements", "reconciliation", "opening to closing"
   - maturity_analysis: Asks for "maturity", "timing", "cash flows", "undiscounted"
   - fair_value_hierarchy: Asks for "fair value", "Level 1/2/3", "valuation techniques"
   - sensitivity_table: Asks for "sensitivity", "impact of changes", "assumptions"
   - impairment_rollforward: Asks for "impairment", "loss allowance", "credit losses"
   - collateral_summary: Asks for "collateral", "security", "pledged assets"
   - exposure_summary: Asks for "exposure", "concentration", "counterparty"
   - segment_analysis: Asks for "segment", "geographical", "business line"
   - provision_rollforward: Asks for "provision movements"
   - tax_reconciliation: Asks for "tax rate", "effective vs statutory"
   - eps_calculation: Asks for "earnings per share", "EPS"
   - cash_flow_breakdown: Asks for "cash flow analysis"

3. QUANTITATIVE EXPECTATIONS - Does it require numbers?
   - class_by_class_totals: Asks for specific amounts by category
   - tie_to_primary_statement: Mentions reconciliation to statements
   - opening_to_closing_balances: Asks for period movements
   - comparatives_presented: Mentions "prior year", "comparative"
   - valuation_inputs_quantified: Asks for assumption values
   - risk_concentration_amounts: Asks for concentration amounts
   - maximum_exposure_to_loss: Asks for "maximum exposure"
   - undiscounted_cash_flows: Asks for "contractual cash flows"

4. TEMPORAL SCOPE - What time period?
   - point_in_time: "at reporting date", "year-end", "balance sheet date", "authorization date"
   - period_flow: "during period", "movements", "changes"
   - current_with_comparative: "current and prior year", "comparative"
   - multi_period_trend: "trend", "multiple years"
   - current_only: Default for simple current year questions

5. CROSS-REFERENCE ANCHORS - Where would you find this?
   - primary_statement: Balance/amount questions
   - policies_section: Policy/method questions  
   - notes_main: Main disclosure notes
   - linked_note: References other notes
   - management_commentary: Narrative explanations
   - segment_note: Segment-related disclosures

6. DATA SOURCES - What evidence is needed?
   - accounting_records: Balance/transaction questions
   - board_resolutions: Authorization/approval questions
   - authorization_documents: Policy decisions
   - event_notifications: Subsequent events
   - legal_documentation: Legal/compliance matters
   - valuation_reports: Fair value/estimates
   - management_reports: Management assessments

CRITICAL LOGIC:
- If question is purely narrative (no amounts/balances) → EMPTY table_archetypes and quantitative_expectations
- If question asks for amounts/numbers → Include appropriate table + quantitative tags
- If question is about authorization/approval → point_in_time + board_resolutions
- If question is about policy/method → policy_basis + policies_section
- Always match patterns to actual question content - NO DEFAULT FILLING

Return ONLY this JSON structure:

{{
  "facet_focus": {{
    "narrative_categories": [],
    "table_archetypes": [],
    "quantitative_expectations": [],
    "temporal_scope": [],
    "cross_reference_anchors": []
  }},
  "conditionality": {{
    "trigger_conditions": [],
    "dependency_chain": [],
    "exception_scenarios": []
  }},
  "evidence_expectations": {{
    "required_documents": ["financial_statements"],
    "data_sources": [],
    "validation_methods": ["document_review"],
    "quality_indicators": ["completeness_check"]
  }},
  "retrieval_support": {{
    "search_keywords": [],
    "section_indicators": [],
    "pattern_matching": [],
    "context_clues": []
  }},
  "citation_controls": {{
    "required_disclosures": [],
    "cross_references": [],
    "compliance_markers": ["regulatory_compliance"]
  }}
}}
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
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert accounting standards analyst specializing in intelligent pattern-based classification. Never use default values - only classify based on actual question content patterns."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            # Parse the JSON response
            content = response.choices[0].message.content.strip()
            
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
                classification=self._get_minimal_classification(),
                error=f"JSON parsing failed: {str(e)}"
            )
        except Exception as e:
            logger.error(f"AI classification error: {e}")
            return TaggingResult(
                success=False,
                classification=self._get_minimal_classification(),
                error=f"AI processing failed: {str(e)}"
            )
    
    def _get_minimal_classification(self) -> Dict[str, Any]:
        """Return minimal classification when AI fails - no defaults"""
        return {
            "facet_focus": {
                "narrative_categories": [],
                "table_archetypes": [],
                "quantitative_expectations": [],
                "temporal_scope": [],
                "cross_reference_anchors": []
            },
            "conditionality": {
                "trigger_conditions": [],
                "dependency_chain": [],
                "exception_scenarios": []
            },
            "evidence_expectations": {
                "required_documents": ["financial_statements"],
                "data_sources": [],
                "validation_methods": ["document_review"],
                "quality_indicators": ["completeness_check"]
            },
            "retrieval_support": {
                "search_keywords": [],
                "section_indicators": [],
                "pattern_matching": [],
                "context_clues": []
            },
            "citation_controls": {
                "required_disclosures": [],
                "cross_references": [],
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
            # Merge the 5D classification into the question
            enhanced_question.update(result.classification)
            logger.info(f"Successfully enhanced question {question.get('id', 'unknown')}")
        else:
            # Use minimal classification on failure
            enhanced_question.update(result.classification)
            logger.warning(f"Used minimal classification for question {question.get('id', 'unknown')}: {result.error}")
        
        return enhanced_question