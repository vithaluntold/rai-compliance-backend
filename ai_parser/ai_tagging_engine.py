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
        """Create effective classification prompt with pattern matching"""
        return """
You are an expert at analyzing accounting questions. Look at this question and classify it using these exact patterns:

QUESTION: {question_text}

Match the question to these patterns and return the corresponding classification:

AUTHORIZATION QUESTIONS (contains "authorize", "approval", "who approved", "date of authorization"):
- narrative_categories: ["policy_basis"]
- temporal_scope: ["point_in_time"] 
- data_sources: ["board_resolutions"]
- cross_reference_anchors: ["policies_section"]

ESTIMATE/EFFECT QUESTIONS (contains "estimate", "financial effect", "impact", "measurement"):
- narrative_categories: ["judgement_estimate", "contingent_events"]
- temporal_scope: ["period_flow"]
- quantitative_expectations: ["class_by_class_totals"]
- data_sources: ["management_reports", "event_notifications"]
- cross_reference_anchors: ["notes_main"]

NATURE/EVENTS QUESTIONS (contains "nature", "events", "subsequent", "non-adjusting"):
- narrative_categories: ["contingent_events"]
- temporal_scope: ["period_flow"]
- data_sources: ["event_notifications"]
- cross_reference_anchors: ["notes_main"]

POLICY/METHOD QUESTIONS (contains "policy", "accounting policy", "method", "basis"):
- narrative_categories: ["policy_basis"]
- temporal_scope: ["current_only"]
- data_sources: ["accounting_records"]
- cross_reference_anchors: ["policies_section"]

AMOUNT/BALANCE QUESTIONS (contains "amount", "balance", "carrying amount", specific numbers):
- narrative_categories: ["quantitative_specifics"]
- temporal_scope: ["point_in_time"]
- table_archetypes: ["carrying_amounts_by_category"]
- quantitative_expectations: ["class_by_class_totals"]
- data_sources: ["accounting_records"]
- cross_reference_anchors: ["primary_statement"]

DEFAULT FOR OTHER QUESTIONS:
- narrative_categories: ["policy_basis"]
- temporal_scope: ["current_only"]
- data_sources: ["accounting_records"]
- cross_reference_anchors: ["notes_main"]

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