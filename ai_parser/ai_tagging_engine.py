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
    """Pattern-based 5D classification tagging engine with zero hardcoding"""
    
    def __init__(self):
        self.prompt_template = self._create_intelligent_prompt()
        
    def _create_intelligent_prompt(self) -> str:
        """Create comprehensive classification prompt with controlled vocabularies"""
        return """
You are given JSON objects containing disclosure checklist items from IFRS/IAS standards. 
Each item has fields such as:
- section (e.g., IAS 2, IAS 10, IFRS 7)
- reference (paragraph number)
- question (the disclosure requirement in plain text)

Your task: Categorize each checklist item into structured tags under:
- facet_focus
- conditionality
- evidence_expectations
- retrieval_support
- citation_controls

---

### Controlled Vocabularies (and when to use them)

**facet_focus → classification**
- narrative_categories:
  - policy_basis → Use when requirement asks about accounting policies or bases (e.g., cost formulas, recognition criteria).
  - disclosure_narrative → Use when requirement asks for descriptive disclosure (e.g., authorisation date, nature of event).
  - rights_and_amendments → Use for questions about rights to amend FS or approvals.
  - adjusting_events → Use when condition existed at reporting date.
  - non_adjusting_events → Use when event occurs after reporting date, does not affect reporting date conditions.
  - risks → Use when risk exposures are the subject.
  - measurement_basis → Use when valuation methods or fair value are required.
  - recognition_criteria → Use when disclosure is about when/if to recognise.

- table_archetypes:
  - carrying_amounts_by_category → Use when requirement wants carrying values by class (e.g., inventories, loans).
  - reconciliation_table → Use when rollforward or reconciliation is expected.
  - maturity_analysis → Use for liquidity analysis.
  - sensitivity_analysis → Use for market/credit risk sensitivity.
  - rollforward → Use when movements across periods must be shown.
  - impact_summary_if_available → Use when requirement asks for "effect/impact" of something.

- quantitative_expectations:
  - absolute_amounts → Use for single numbers (e.g., write-downs, expense recognised).
  - class_by_class_totals → Use for breakdowns by class/category.
  - adjustment_amounts → Use for reversals, write-downs, adjustments.
  - estimate_of_financial_effect → Use for estimated impacts of non-adjusting events.
  - ranges → Use for disclosures expressed in bands/intervals.
  - qualitative_only → Use if requirement is narrative with no numbers.

- temporal_scope:
  - current_period → Use when requirement is this year only.
  - current_with_comparative → Use when comparative disclosure required.
  - event_based → Use when requirement is tied to a specific event (e.g., subsequent event).
  - multi_period → Use when multiple reporting periods must be shown.
  - forecast_or_forwardlooking → Use when future effects must be estimated.

- cross_reference_anchors:
  - notes_main → Use when disclosure is expected in main notes.
  - subsequent_events_note → Use for IAS 10 non-adjusting disclosures.
  - front_matter → Use for authorisation, signatures, approval blocks.
  - signing_approval_note → Use when identifying board/management approval.
  - SoFP, PnL, cash_flows, equity_statement → Use when tied to statements.
  - related_line_items → Use when disclosure links to line items in statements.
  - other_standards → Use when cross-referenced to another IFRS/IAS.

---

**conditionality → logic**
- trigger_conditions: standard_requirement, event_occurred, conditional_requirement, voluntary_disclosure
- dependency_chain: identify_requirement, determine_adjusting, determine_non_adjusting, identify_authorisation_date, identify_authorising_body_or_individuals, estimate_financial_effect
- exception_scenarios: not_applicable, cannot_estimate_effect, jurisdictional_override

---

**evidence_expectations → evidence types**
- required_documents: financial_statements, accounting_policies, board_minutes_if_referenced, management_reports, press_releases_if_referenced
- data_sources: accounting_records, corporate_secretariat_records, management_estimates, external_valuations_if_any
- validation_methods: document_review, cross_check, recalculation
- quality_indicators: completeness_check, explicit_date_statement, specificity_of_event_description, clear_adjusting_vs_non_adjusting_classification

---

**retrieval_support → AI assistance**
- search_keywords: Extract keywords from question (e.g., "inventory write-down", "subsequent events").
- section_indicators: Likely disclosure location (e.g., notes, subsequent events, approval block).
- pattern_matching: Regex fragments (e.g., "authorised for issue", "write-down reversal").
- context_clues: Additional hints (e.g., "signature block", "impairment triggers").

---

**citation_controls → compliance checks**
- required_disclosures: accounting_policy, authorisation_date, amendment_rights_statement_if_applicable, nature, financial_effect
- cross_references: related_standards, IAS 1, IAS 37, IFRS 3, IAS 33
- compliance_markers: regulatory_compliance, jurisdictional_requirement, voluntary_best_practice

---

QUESTION: {question_text}
CONTEXT: {context}

### Final Rule
If unsure, **prefer broader categories** (e.g., disclosure_narrative + qualitative_only) over leaving fields empty. Only leave arrays blank if the category is truly irrelevant.

Return the classification in exact JSON format with all fields filled appropriately.
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
        """Return default classification structure when AI fails"""
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
                "pattern_matching": [],
                "context_clues": []
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
            enhanced_question.update(result.classification)
            logger.warning(f"Used default classification for question {question.get('id', 'unknown')}: {result.error}")
        
        return enhanced_question