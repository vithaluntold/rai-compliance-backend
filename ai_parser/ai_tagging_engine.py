#!/usr/bin/env python3
"""
AI Checklist Parser - AI Tagging Engine
5D Classification System for Accounting Standards Compliance

This module provides the AI-powered tagging engine that analyzes
compliance questions and applies 5-dimensional classification tags.
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
    
class AITaggingEngine:
    """AI-powered 5D classification tagging engine"""
    
    def __init__(self):
        self.prompt_template = self._load_prompt_template()
        
    def _load_prompt_template(self) -> str:
        """Load the optimized 5D classification prompt template"""
        return """
You are an expert in accounting standards compliance and document analysis. 
Analyze the following compliance checklist question and classify it using our 5-dimensional framework.

CONTEXTUAL CLASSIFICATION RULES:

NARRATIVE CATEGORIES - When to Use:
- policy_basis: Questions about "accounting policies", "bases of preparation", "recognition criteria"
- methodology: Questions about "measurement", "valuation method", "how calculated"
- judgement_estimate: Questions about "assumptions", "estimates", "judgements", "uncertainties"
- risk_strategy: Questions about "risk management", "objectives and policies"
- derecognition_explainer: Questions about "derecognition", "transfers", "extinguishment"
- hedge_strategy: Questions about "hedge accounting", "hedge designation", "hedge effectiveness"
- transition_adoption: Questions about "first-time adoption", "new standards", "transition"
- changes_during_period: Questions about "changes in policy", "reclassifications", "restatements"
- contingent_events: Questions about "contingencies", "commitments", "subsequent events"
- industry_specific_policy: Questions specific to banks, insurers, extractives, etc.

TABLE ARCHETYPES - When to Use:
- carrying_amounts_by_category: Questions asking for "balances by class/category"
- movement_reconciliation: Questions about "movements", "reconciliation from opening to closing"
- maturity_analysis: Questions about "maturity", "timing of cash flows", "undiscounted amounts"
- fair_value_hierarchy: Questions about "fair value levels", "Level 1/2/3", "valuation techniques"
- sensitivity_table: Questions about "sensitivity to changes", "impact of assumptions"
- impairment_rollforward: Questions about "impairment movements", "loss allowances"
- collateral_summary: Questions about "collateral", "security", "assets pledged"
- exposure_summary: Questions about "exposures", "concentrations", "counterparty analysis"
- segment_analysis: Questions about "segment reporting", "geographical/business segments"
- provision_rollforward: Questions about "provision movements" (IAS 37)
- tax_reconciliation: Questions about "tax rate reconciliation", "statutory vs effective"
- eps_calculation: Questions about "earnings per share", "profit attributable", "share counts"
- cash_flow_breakdown: Questions about "cash flow analysis", "operating/financing/investing"
- share_capital_movement: Questions about "share capital changes", "equity movements"
- pension_asset_liability_movement: Questions about "defined benefit movements"
- lease_commitment_schedule: Questions about "lease maturity", "future lease payments"
- industry_specific_table: Industry-specific disclosures (insurance, biological assets)

QUANTITATIVE EXPECTATIONS - When to Use:
- class_by_class_totals: Questions asking for "amounts by class/type/category"
- tie_to_primary_statement: Questions requiring "reconciliation to primary statements"
- opening_to_closing_balances: Questions about "movements from start to end of period"
- comparatives_presented: Questions requiring "prior period comparison"
- valuation_inputs_quantified: Questions about "key assumptions", "valuation inputs"
- risk_concentration_amounts: Questions about "concentration by geography/sector"
- maximum_exposure_to_loss: Questions about "maximum exposure", "potential loss"
- undiscounted_cash_flows: Questions about "contractual cash flows", "undiscounted amounts"
- OCI_reclassification_amounts: Questions about "recycling between OCI and P&L"
- sensitivity_to_key_assumptions: Questions about "quantitative sensitivity analysis"

TEMPORAL SCOPE - When to Use:
- current_only: Questions asking for "current year only"
- current_with_comparative: Questions requiring "current and prior year"
- multi_period_trend: Questions asking for "trend analysis", "multiple years"
- point_in_time: Questions about "balances at reporting date", "year-end position"
- period_flow: Questions about "movements during the period", "activity in year"

SPECIFIC STANDARD RULES:
- IAS 10: Authorization = policy_basis + point_in_time + NO tables + board_resolutions
- IAS 16: PPE movements = movement_reconciliation + opening_to_closing_balances
- IFRS 9: Financial instruments = carrying_amounts_by_category + fair_value_hierarchy
- IAS 1: Presentation = current_with_comparative + tie_to_primary_statement

QUESTION TO ANALYZE:
{question_text}

CONTEXT (if provided):
{context}

Please provide classification in the following JSON structure:

{{
  "facet_focus": {{
    "narrative_categories": ["category1", "category2"],
    "table_archetypes": ["archetype1"],
    "quantitative_expectations": ["expectation1", "expectation2"],
    "temporal_scope": ["scope1"],
    "cross_reference_anchors": ["anchor1"]
  }},
  "conditionality": {{
    "trigger_conditions": ["condition1", "condition2"],
    "dependency_chain": ["step1", "step2"],
    "exception_scenarios": ["exception1"]
  }},
  "evidence_expectations": {{
    "required_documents": ["doc1", "doc2"],
    "data_sources": ["source1", "source2"],
    "validation_methods": ["method1", "method2"],
    "quality_indicators": ["indicator1", "indicator2"]
  }},
  "retrieval_support": {{
    "search_keywords": ["keyword1", "keyword2"],
    "section_indicators": ["section1", "section2"],
    "pattern_matching": ["pattern1", "pattern2"],
    "context_clues": ["clue1", "clue2"]
  }},
  "citation_controls": {{
    "required_disclosures": ["disclosure1", "disclosure2"],
    "cross_references": ["ref1", "ref2"],
    "compliance_markers": ["marker1", "marker2"]
  }}
}}

CLASSIFICATION OPTIONS:

Narrative Categories: [policy_basis, methodology, judgement_estimate, risk_strategy, derecognition_explainer, hedge_strategy, transition_adoption, changes_during_period, contingent_events, industry_specific_policy]

Table Archetypes: [carrying_amounts_by_category, movement_reconciliation, maturity_analysis, fair_value_hierarchy, sensitivity_table, impairment_rollforward, collateral_summary, exposure_summary, segment_analysis, provision_rollforward, tax_reconciliation, eps_calculation, cash_flow_breakdown, share_capital_movement, pension_asset_liability_movement, lease_commitment_schedule, industry_specific_table]

Quantitative Expectations: [class_by_class_totals, tie_to_primary_statement, opening_to_closing_balances, comparatives_presented, valuation_inputs_quantified, risk_concentration_amounts, maximum_exposure_to_loss, undiscounted_cash_flows, OCI_reclassification_amounts, sensitivity_to_key_assumptions, tax_rate_reconciliation_items, EPS_numerators_denominators, segment_profit_loss_assets]

Temporal Scope: [current_only, current_with_comparative, multi_period_trend, point_in_time, period_flow]

Cross-Reference Anchors: [primary_statement, notes_main, policies_section, linked_note, management_commentary, segment_note]

Data Sources: [accounting_records, board_resolutions, authorization_documents, event_notifications, legal_documentation, management_reports, external_confirmations, actuarial_reports, valuation_reports, audit_evidence, regulatory_filings]

CRITICAL RULES:
1. If question is about "authorization date/who authorized" → Use policy_basis + point_in_time + NO table_archetypes + NO quantitative_expectations
2. If question asks for "amounts/balances" → Use appropriate table_archetype + quantitative_expectations
3. If question is purely narrative → Use relevant narrative_category + NO table_archetypes
4. NEVER default to reconciliation_table - match the actual disclosure requirement!
5. Leave arrays EMPTY [] if not applicable - don't fill with defaults!

Respond ONLY with the JSON structure. No additional text or explanation.
"""

    def classify_question(self, question_text: str, context: str = "") -> TaggingResult:
        """
        Classify a single question using AI and return 5D tags
        
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
                    {"role": "system", "content": "You are an expert accounting standards analyst specializing in 5-dimensional content classification."},
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
        """Return a default classification structure when AI fails"""
        return {
            "facet_focus": {
                "narrative_categories": ["disclosure_narrative"],
                "table_archetypes": [],
                "quantitative_expectations": [],
                "temporal_scope": ["current_period"],
                "cross_reference_anchors": []
            },
            "conditionality": {
                "trigger_conditions": ["standard_requirement"],
                "dependency_chain": ["identify_requirement"],
                "exception_scenarios": ["not_applicable"]
            },
            "evidence_expectations": {
                "required_documents": ["financial_statements"],
                "data_sources": [],
                "validation_methods": ["document_review"],
                "quality_indicators": ["completeness_check"]
            },
            "retrieval_support": {
                "search_keywords": ["accounting", "disclosure"],
                "section_indicators": ["notes"],
                "pattern_matching": ["standard_format"],
                "context_clues": ["financial_position"]
            },
            "citation_controls": {
                "required_disclosures": ["accounting_policy"],
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
        
        # Get question text
        question_text = question.get('question', '')
        context = question.get('guidance', '') or question.get('context', '')
        
        # Apply AI classification
        result = self.classify_question(question_text, context)
        
        # Add classification to question
        if result.success and result.classification:
            enhanced_question.update(result.classification)
        else:
            logger.warning(f"Failed to classify question: {result.error}")
            enhanced_question.update(self._get_default_classification())
        
        return enhanced_question