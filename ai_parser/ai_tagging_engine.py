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

Narrative Categories: [disclosure_narrative, risk_assessment, accounting_policy, measurement_basis, recognition_criteria, presentation_format, comparative_analysis, materiality_assessment, judgement_documentation, transition_provisions]

Table Archetypes: [reconciliation_table, aging_analysis, sensitivity_matrix, classification_breakdown, movement_schedule, fair_value_hierarchy, maturity_analysis, geographic_segmentation, product_segmentation, comparative_periods, consolidation_details, currency_translation, tax_analysis, cash_flow_components, related_party_transactions, subsequent_events, contingency_analysis]

Quantitative Expectations: [percentage_calculations, absolute_amounts, ratio_analysis, variance_calculations, trend_analysis, benchmark_comparisons, threshold_assessments, range_validations, statistical_measures, growth_rates, efficiency_metrics, liquidity_measures, profitability_indicators]

Temporal Scope: [current_period, comparative_period, historical_trend, forward_looking, event_driven]

Cross-Reference Anchors: [other_standards, regulatory_requirements, industry_practice, prior_year_comparison, related_disclosures, supporting_schedules]

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
                "table_archetypes": ["reconciliation_table"],
                "quantitative_expectations": ["absolute_amounts"],
                "temporal_scope": ["current_period"],
                "cross_reference_anchors": ["other_standards"]
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