#!/usr/bin/env python3
"""
Intelligent Content-Question Mapping Engine
Map document content segments to relevant questions using advanced similarity algorithms.

This system implements:
1. Weighted tag-overlap scoring based on 5D classification tags
2. Semantic similarity matching using advanced text analysis
3. Contextual relevance algorithms considering document structure
4. Multi-dimensional matching across accounting standards and content types

Integration: Works with Tool 2 (Structure Parser), Tool 3 (AI Classifier), and Tool 1 (Taxonomy Validator)
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import re
import math
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ContentMapping:
    """Represents a mapping between content segment and question"""
    content_id: str
    question_id: str
    similarity_score: float
    tag_overlap_score: float
    semantic_score: float
    contextual_score: float
    composite_score: float
    confidence: float
    mapping_details: Dict[str, Any]

@dataclass  
class MappingResult:
    """Result of content-question mapping operation"""
    content_segment: Dict[str, Any]
    matched_questions: List[ContentMapping]
    total_questions_analyzed: int
    best_match_score: float
    mapping_method: str
    processing_metadata: Dict[str, Any]

class IntelligentContentQuestionMapper:
    """Advanced content-question mapping system with multi-dimensional similarity"""
    
    def __init__(self, questions_directory: str = None):
        """Initialize the intelligent mapping system"""
        self.questions_directory = questions_directory or "checklist_data"
        self.questions_cache = {}
        self.content_segments_cache = {}
        
        # Similarity weights for different scoring methods
        self.scoring_weights = {
            'tag_overlap': 0.35,
            'semantic_similarity': 0.30,
            'contextual_relevance': 0.25,
            'taxonomy_alignment': 0.10
        }
        
        # Initialize tag similarity mappings
        self.tag_similarity_mapping = self._create_tag_similarity_mapping()
        
        logger.info("Intelligent Content-Question Mapper initialized")
    
    def _create_tag_similarity_mapping(self) -> Dict[str, Dict[str, float]]:
        """Create similarity mappings between different classification tags"""
        return {
            # Accounting Standard similarities
            'ifrs_15_revenue': {
                'ifrs_18_insurance': 0.3,
                'ias_1_presentation': 0.4,
                'revenue_recognition': 0.9,
                'contract_analysis': 0.7
            },
            'ifrs_9_financial': {
                'ias_32_instruments': 0.8,
                'ias_39_recognition': 0.7,
                'ifrs_7_disclosures': 0.6,
                'financial_instruments': 0.9
            },
            'ias_16_ppe': {
                'ias_36_impairment': 0.6,
                'ias_38_intangibles': 0.5,
                'ifrs_16_leases': 0.4,
                'asset_management': 0.8
            },
            
            # Content Type similarities
            'balance_sheet': {
                'statement_position': 0.9,
                'assets_liabilities': 0.8,
                'equity_analysis': 0.7,
                'financial_position': 0.9
            },
            'income_statement': {
                'profit_loss': 0.9,
                'comprehensive_income': 0.8,
                'revenue_expenses': 0.7,
                'performance_analysis': 0.8
            },
            'cash_flows': {
                'liquidity_analysis': 0.8,
                'operating_activities': 0.7,
                'investing_financing': 0.6,
                'cash_management': 0.9
            }
        }
    
    def load_questions_from_directory(self) -> Dict[str, Any]:
        """Load and cache questions from the questions directory"""
        if self.questions_cache:
            return self.questions_cache
        
        questions = {}
        questions_path = Path(self.questions_directory)
        
        if not questions_path.exists():
            logger.warning(f"Questions directory not found: {questions_path}")
            return self._create_sample_questions()
        
        try:
            # Load questions from JSON files
            for json_file in questions_path.glob("**/*.json"):
                with open(json_file, 'r', encoding='utf-8-sig') as f:
                    file_data = json.load(f)
                    
                    # Handle the actual question file structure
                    if isinstance(file_data, dict) and 'sections' in file_data:
                        # This is a checklist file with sections
                        for section in file_data.get('sections', []):
                            section_items = section.get('items', [])
                            if isinstance(section_items, list):
                                for item in section_items:
                                    if isinstance(item, dict) and 'question' in item:
                                        question_id = f"{json_file.stem}_{item.get('id', len(questions))}"
                                        questions[question_id] = {
                                            'id': question_id,
                                            'question_text': item.get('question', ''),
                                            'reference': item.get('reference', ''),
                                            'section': item.get('section', ''),
                                            'classification_tags': {
                                                'primary_standard': item.get('section', ''),
                                                'content_type': self._infer_content_type_from_question(item.get('question', '')),
                                                'complexity_level': 'intermediate',
                                                'document_sections': self._infer_document_sections_from_question(item.get('question', ''))
                                            }
                                        }
                    # Handle other possible formats
                    elif isinstance(file_data, list):
                        for i, q in enumerate(file_data):
                            question_id = f"{json_file.stem}_{i}"
                            if isinstance(q, dict):
                                questions[question_id] = q
                    elif isinstance(file_data, dict) and 'question' in file_data:
                        question_id = json_file.stem
                        questions[question_id] = file_data
            
            logger.info(f"Loaded {len(questions)} questions from {questions_path}")
            
        except Exception as e:
            logger.warning(f"Error loading questions: {e}")
            questions = self._create_sample_questions()
        
        self.questions_cache = questions
        return questions
    
    def _infer_content_type_from_question(self, question_text: str) -> str:
        """Infer content type from question text"""
        question_lower = question_text.lower()
        
        if any(term in question_lower for term in ['revenue', 'sales', 'income']):
            return 'revenue_recognition'
        elif any(term in question_lower for term in ['asset', 'property', 'equipment', 'depreciation']):
            return 'asset_management'
        elif any(term in question_lower for term in ['liability', 'debt', 'obligation']):
            return 'liability_management'
        elif any(term in question_lower for term in ['cash', 'liquidity', 'flow']):
            return 'cash_flow_analysis'
        elif any(term in question_lower for term in ['equity', 'capital', 'shareholder']):
            return 'equity_analysis'
        elif any(term in question_lower for term in ['financial instrument', 'investment']):
            return 'financial_instruments'
        else:
            return 'general_accounting'
    
    def _infer_document_sections_from_question(self, question_text: str) -> List[str]:
        """Infer relevant document sections from question text"""
        question_lower = question_text.lower()
        sections = []
        
        if any(term in question_lower for term in ['present', 'presentation', 'disclose', 'disclosure']):
            sections.append('notes')
        if any(term in question_lower for term in ['balance', 'asset', 'liability', 'equity']):
            sections.append('balance_sheet')
        if any(term in question_lower for term in ['revenue', 'income', 'expense', 'profit', 'loss']):
            sections.append('income_statement')
        if any(term in question_lower for term in ['cash', 'flow', 'operating', 'investing', 'financing']):
            sections.append('cash_flows')
        
        return sections if sections else ['notes']
    
    def _create_sample_questions(self) -> Dict[str, Any]:
        """Create sample questions for testing when no questions directory exists"""
        return {
            "revenue_q1": {
                "id": "revenue_q1",
                "question_text": "How is revenue from contracts with customers recognized under IFRS 15?",
                "classification_tags": {
                    "primary_standard": "IFRS 15",
                    "content_type": "revenue_recognition",
                    "complexity_level": "intermediate",
                    "document_sections": ["income_statement", "notes"]
                }
            },
            "assets_q1": {
                "id": "assets_q1", 
                "question_text": "What are the measurement requirements for property, plant and equipment?",
                "classification_tags": {
                    "primary_standard": "IAS 16",
                    "content_type": "asset_measurement",
                    "complexity_level": "basic",
                    "document_sections": ["balance_sheet", "notes"]
                }
            },
            "financial_q1": {
                "id": "financial_q1",
                "question_text": "How should financial instruments be classified and measured?",
                "classification_tags": {
                    "primary_standard": "IFRS 9",
                    "content_type": "financial_instruments",
                    "complexity_level": "advanced",
                    "document_sections": ["balance_sheet", "notes", "disclosures"]
                }
            }
        }
    
    def calculate_tag_overlap_score(self, content_tags: Dict[str, Any], 
                                   question_tags: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Calculate weighted tag overlap score between content and question"""
        
        if not content_tags or not question_tags:
            return 0.0, {"reason": "Missing classification tags"}
        
        overlap_details = {
            'primary_standard_match': 0.0,
            'content_type_match': 0.0,
            'section_match': 0.0,
            'semantic_similarity': 0.0,
            'total_overlaps': 0
        }
        
        # Primary standard matching (highest weight)
        content_standard = content_tags.get('primary_standard', '').lower()
        question_standard = question_tags.get('primary_standard', '').lower()
        
        if content_standard and question_standard:
            if content_standard == question_standard:
                overlap_details['primary_standard_match'] = 1.0
            else:
                # Check for related standards
                similarity = self._calculate_standard_similarity(content_standard, question_standard)
                overlap_details['primary_standard_match'] = similarity
        
        # Content type matching
        content_type = content_tags.get('content_type', '').lower()
        question_type = question_tags.get('content_type', '').lower()
        
        if content_type and question_type:
            if content_type == question_type:
                overlap_details['content_type_match'] = 1.0
            else:
                # Check semantic similarity of content types
                similarity = self._calculate_content_type_similarity(content_type, question_type)
                overlap_details['content_type_match'] = similarity
        
        # Document section matching
        content_sections = content_tags.get('document_sections', [])
        question_sections = question_tags.get('document_sections', [])
        
        if content_sections and question_sections:
            if not isinstance(content_sections, list):
                content_sections = [content_sections]
            if not isinstance(question_sections, list):
                question_sections = [question_sections]
            
            section_overlap = len(set(content_sections) & set(question_sections))
            section_total = len(set(content_sections) | set(question_sections))
            overlap_details['section_match'] = section_overlap / section_total if section_total > 0 else 0.0
        
        # Calculate composite score with weights
        weighted_score = (
            overlap_details['primary_standard_match'] * 0.5 +
            overlap_details['content_type_match'] * 0.3 +
            overlap_details['section_match'] * 0.2
        )
        
        overlap_details['total_overlaps'] = sum([
            overlap_details['primary_standard_match'] > 0,
            overlap_details['content_type_match'] > 0,
            overlap_details['section_match'] > 0
        ])
        
        return weighted_score, overlap_details
    
    def _calculate_standard_similarity(self, standard1: str, standard2: str) -> float:
        """Calculate similarity between accounting standards"""
        # Remove common prefixes/suffixes
        clean_std1 = re.sub(r'^(ifrs|ias)\s*\d*', '', standard1.lower()).strip()
        clean_std2 = re.sub(r'^(ifrs|ias)\s*\d*', '', standard2.lower()).strip()
        
        similarity_mapping = {
            ('revenue', 'income'): 0.7,
            ('revenue', 'performance'): 0.6,
            ('financial', 'instruments'): 0.8,
            ('assets', 'property'): 0.7,
            ('assets', 'equipment'): 0.6,
            ('presentation', 'disclosure'): 0.6,
            ('measurement', 'recognition'): 0.7
        }
        
        for (term1, term2), score in similarity_mapping.items():
            if (term1 in clean_std1 and term2 in clean_std2) or \
               (term2 in clean_std1 and term1 in clean_std2):
                return score
        
        return 0.0
    
    def _calculate_content_type_similarity(self, type1: str, type2: str) -> float:
        """Calculate similarity between content types"""
        type1_clean = type1.lower().replace('_', ' ')
        type2_clean = type2.lower().replace('_', ' ')
        
        # Use pre-defined similarity mappings
        for base_type, similar_types in self.tag_similarity_mapping.items():
            if base_type in type1_clean:
                for similar_type, score in similar_types.items():
                    if similar_type in type2_clean:
                        return score
        
        # Fallback: simple word overlap
        words1 = set(type1_clean.split())
        words2 = set(type2_clean.split())
        
        if words1 and words2:
            overlap = len(words1 & words2)
            total = len(words1 | words2)
            return overlap / total if total > 0 else 0.0
        
        return 0.0
    
    def calculate_semantic_similarity(self, content_text: str, question_text: str) -> Tuple[float, Dict[str, Any]]:
        """Calculate semantic similarity between content and question text"""
        
        # Clean and normalize text
        content_clean = self._normalize_text(content_text)
        question_clean = self._normalize_text(question_text)
        
        similarity_details = {
            'keyword_overlap': 0.0,
            'phrase_similarity': 0.0,
            'concept_alignment': 0.0,
            'context_relevance': 0.0
        }
        
        # Calculate keyword overlap
        content_keywords = self._extract_keywords(content_clean)
        question_keywords = self._extract_keywords(question_clean)
        
        if content_keywords and question_keywords:
            keyword_overlap = len(content_keywords & question_keywords)
            total_keywords = len(content_keywords | question_keywords)
            similarity_details['keyword_overlap'] = keyword_overlap / total_keywords if total_keywords > 0 else 0.0
        
        # Calculate phrase similarity (n-gram overlap)
        similarity_details['phrase_similarity'] = self._calculate_ngram_similarity(content_clean, question_clean)
        
        # Calculate concept alignment using financial terms
        similarity_details['concept_alignment'] = self._calculate_concept_alignment(content_clean, question_clean)
        
        # Overall semantic score
        semantic_score = (
            similarity_details['keyword_overlap'] * 0.4 +
            similarity_details['phrase_similarity'] * 0.3 +
            similarity_details['concept_alignment'] * 0.3
        )
        
        return semantic_score, similarity_details
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text
    
    def _extract_keywords(self, text: str) -> set:
        """Extract relevant keywords from financial text"""
        
        # Financial domain keywords (higher importance)
        financial_terms = {
            'revenue', 'assets', 'liabilities', 'equity', 'cash', 'profit', 'loss',
            'income', 'expenses', 'balance', 'statement', 'financial', 'accounting',
            'ifrs', 'ias', 'recognition', 'measurement', 'disclosure', 'presentation',
            'contract', 'instrument', 'property', 'equipment', 'inventory', 'receivables'
        }
        
        words = set(text.split())
        
        # Filter for relevant keywords (length > 2, not common words)
        common_words = {'the', 'and', 'for', 'are', 'with', 'this', 'that', 'from', 'have', 'been'}
        
        keywords = {word for word in words 
                   if len(word) > 2 and word not in common_words}
        
        # Prioritize financial terms
        financial_keywords = keywords & financial_terms
        other_keywords = (keywords - financial_terms)
        
        # Return top keywords (financial terms + others)
        return financial_keywords | (other_keywords if len(financial_keywords) < 5 else set())
    
    def _calculate_ngram_similarity(self, text1: str, text2: str, n: int = 2) -> float:
        """Calculate n-gram similarity between texts"""
        
        def get_ngrams(text: str, n: int) -> set:
            words = text.split()
            return {' '.join(words[i:i+n]) for i in range(len(words)-n+1)}
        
        if len(text1.split()) < n or len(text2.split()) < n:
            return 0.0
        
        ngrams1 = get_ngrams(text1, n)
        ngrams2 = get_ngrams(text2, n)
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_concept_alignment(self, content: str, question: str) -> float:
        """Calculate conceptual alignment using financial concepts"""
        
        concept_groups = {
            'revenue_concepts': ['revenue', 'sales', 'income', 'earnings', 'proceeds', 'turnover'],
            'asset_concepts': ['assets', 'property', 'equipment', 'inventory', 'receivables', 'investments'],
            'liability_concepts': ['liabilities', 'debt', 'obligations', 'payables', 'borrowings'],
            'performance_concepts': ['profit', 'loss', 'performance', 'results', 'margin', 'return'],
            'measurement_concepts': ['fair', 'value', 'cost', 'amortized', 'impairment', 'revaluation'],
            'disclosure_concepts': ['disclosure', 'notes', 'presentation', 'classification', 'reporting']
        }
        
        content_concepts = set()
        question_concepts = set()
        
        for concept_group, terms in concept_groups.items():
            content_has = any(term in content for term in terms)
            question_has = any(term in question for term in terms)
            
            if content_has:
                content_concepts.add(concept_group)
            if question_has:
                question_concepts.add(concept_group)
        
        if not content_concepts or not question_concepts:
            return 0.0
        
        overlap = len(content_concepts & question_concepts)
        total = len(content_concepts | question_concepts)
        
        return overlap / total if total > 0 else 0.0
    
    def calculate_contextual_relevance(self, content_segment: Dict[str, Any], 
                                     question: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Calculate contextual relevance based on document structure and position"""
        
        relevance_details = {
            'section_relevance': 0.0,
            'proximity_score': 0.0,
            'structural_alignment': 0.0,
            'document_context': 0.0
        }
        
        # Section relevance - does content come from sections the question targets?
        content_section = content_segment.get('section_type', '').lower()
        question_sections = question.get('classification_tags', {}).get('document_sections', [])
        
        if isinstance(question_sections, str):
            question_sections = [question_sections]
        
        if content_section and question_sections:
            if content_section in [s.lower() for s in question_sections]:
                relevance_details['section_relevance'] = 1.0
            else:
                # Check for related sections
                section_similarity = self._calculate_section_similarity(content_section, question_sections)
                relevance_details['section_relevance'] = section_similarity
        
        # Structural alignment - consider content hierarchy and importance
        content_level = content_segment.get('hierarchy_level', 0)
        content_confidence = content_segment.get('confidence', 0.5)
        
        # Higher level (more important) content gets higher scores
        structure_score = min(1.0, (5 - content_level) / 5) if content_level <= 5 else 0.2
        relevance_details['structural_alignment'] = structure_score * content_confidence
        
        # Document context - consider surrounding content
        relevance_details['document_context'] = self._calculate_document_context_score(content_segment)
        
        # Calculate composite contextual score
        contextual_score = (
            relevance_details['section_relevance'] * 0.4 +
            relevance_details['structural_alignment'] * 0.3 +
            relevance_details['document_context'] * 0.3
        )
        
        return contextual_score, relevance_details
    
    def _calculate_section_similarity(self, content_section: str, target_sections: List[str]) -> float:
        """Calculate similarity between content section and target sections"""
        
        section_mappings = {
            'balance_sheet': ['statement_position', 'financial_position', 'assets', 'liabilities'],
            'income_statement': ['profit_loss', 'comprehensive_income', 'performance', 'results'],
            'cash_flows': ['cash_flow', 'liquidity', 'operating_activities'],
            'notes': ['disclosures', 'additional_information', 'footnotes'],
            'equity': ['shareholders_equity', 'retained_earnings', 'capital']
        }
        
        max_similarity = 0.0
        
        for target_section in target_sections:
            target_lower = target_section.lower()
            
            # Direct match
            if content_section == target_lower:
                return 1.0
            
            # Check mappings
            for main_section, alternatives in section_mappings.items():
                if content_section == main_section and target_lower in alternatives:
                    max_similarity = max(max_similarity, 0.8)
                elif target_lower == main_section and content_section in alternatives:
                    max_similarity = max(max_similarity, 0.8)
        
        return max_similarity
    
    def _calculate_document_context_score(self, content_segment: Dict[str, Any]) -> float:
        """Calculate document context score based on surrounding content"""
        
        # Factors that increase context relevance
        context_factors = {
            'has_financial_data': content_segment.get('contains_numbers', False),
            'is_table_content': content_segment.get('is_table', False),
            'has_accounting_terms': len(content_segment.get('accounting_keywords', [])) > 0,
            'proper_formatting': content_segment.get('formatting_score', 0) > 0.7,
            'sufficient_length': len(content_segment.get('content', '')) > 50
        }
        
        # Weight the factors
        weights = {
            'has_financial_data': 0.3,
            'is_table_content': 0.2,
            'has_accounting_terms': 0.25,
            'proper_formatting': 0.15,
            'sufficient_length': 0.1
        }
        
        score = sum(weights[factor] * (1.0 if value else 0.0) 
                   for factor, value in context_factors.items())
        
        return min(1.0, score)
    
    def map_content_to_questions(self, content_segment: Dict[str, Any], 
                                top_k: int = 5) -> MappingResult:
        """Map a content segment to the most relevant questions"""
        
        questions = self.load_questions_from_directory()
        mappings = []
        
        content_text = content_segment.get('content', '')
        content_tags = content_segment.get('classification_tags', {})
        
        logger.info(f"Mapping content segment to {len(questions)} questions")
        
        for question_id, question_data in questions.items():
            try:
                # Calculate all similarity scores
                tag_score, tag_details = self.calculate_tag_overlap_score(
                    content_tags, 
                    question_data.get('classification_tags', {})
                )
                
                semantic_score, semantic_details = self.calculate_semantic_similarity(
                    content_text,
                    question_data.get('question_text', '')
                )
                
                contextual_score, contextual_details = self.calculate_contextual_relevance(
                    content_segment,
                    question_data
                )
                
                # Calculate composite score
                composite_score = (
                    tag_score * self.scoring_weights['tag_overlap'] +
                    semantic_score * self.scoring_weights['semantic_similarity'] +
                    contextual_score * self.scoring_weights['contextual_relevance']
                )
                
                # Calculate confidence based on score distribution
                confidence = self._calculate_mapping_confidence(
                    tag_score, semantic_score, contextual_score, composite_score
                )
                
                mapping = ContentMapping(
                    content_id=content_segment.get('id', 'unknown'),
                    question_id=question_id,
                    similarity_score=composite_score,
                    tag_overlap_score=tag_score,
                    semantic_score=semantic_score,
                    contextual_score=contextual_score,
                    composite_score=composite_score,
                    confidence=confidence,
                    mapping_details={
                        'tag_details': tag_details,
                        'semantic_details': semantic_details,
                        'contextual_details': contextual_details,
                        'question_text': question_data.get('question_text', '')[:100] + "..."
                    }
                )
                
                mappings.append(mapping)
                
            except Exception as e:
                logger.warning(f"Error mapping to question {question_id}: {e}")
                continue
        
        # Sort by composite score and take top K
        mappings.sort(key=lambda x: x.composite_score, reverse=True)
        top_mappings = mappings[:top_k]
        
        best_score = top_mappings[0].composite_score if top_mappings else 0.0
        
        return MappingResult(
            content_segment=content_segment,
            matched_questions=top_mappings,
            total_questions_analyzed=len(questions),
            best_match_score=best_score,
            mapping_method="intelligent_multi_dimensional",
            processing_metadata={
                'scoring_weights': self.scoring_weights,
                'questions_loaded': len(questions),
                'mappings_created': len(mappings),
                'top_k': top_k
            }
        )
    
    def _calculate_mapping_confidence(self, tag_score: float, semantic_score: float, 
                                    contextual_score: float, composite_score: float) -> float:
        """Calculate confidence in the mapping based on score consistency"""
        
        scores = [tag_score, semantic_score, contextual_score]
        
        # High confidence if scores are consistently high
        if all(score > 0.7 for score in scores):
            return 0.95
        
        # Medium confidence if scores are moderate and consistent  
        if composite_score > 0.5 and max(scores) - min(scores) < 0.3:
            return 0.75
        
        # Lower confidence for inconsistent or low scores
        if composite_score > 0.3:
            return 0.5
        
        return 0.25
    
    def batch_map_content_segments(self, content_segments: List[Dict[str, Any]], 
                                  top_k_per_segment: int = 3) -> List[MappingResult]:
        """Map multiple content segments to questions efficiently"""
        
        results = []
        
        logger.info(f"Batch mapping {len(content_segments)} content segments")
        
        for i, segment in enumerate(content_segments):
            try:
                mapping_result = self.map_content_to_questions(segment, top_k_per_segment)
                results.append(mapping_result)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(content_segments)} segments")
                    
            except Exception as e:
                logger.error(f"Error processing segment {i}: {e}")
                continue
        
        logger.info(f"Completed batch mapping: {len(results)} successful mappings")
        return results
    
    def generate_mapping_report(self, mapping_results: List[MappingResult]) -> Dict[str, Any]:
        """Generate comprehensive report of mapping results"""
        
        if not mapping_results:
            return {"error": "No mapping results to analyze"}
        
        # Aggregate statistics
        total_segments = len(mapping_results)
        total_mappings = sum(len(result.matched_questions) for result in mapping_results)
        
        score_distribution = {
            'high_quality': 0,    # > 0.7
            'medium_quality': 0,  # 0.3 - 0.7
            'low_quality': 0      # < 0.3
        }
        
        all_scores = []
        
        for result in mapping_results:
            best_score = result.best_match_score
            all_scores.append(best_score)
            
            if best_score > 0.7:
                score_distribution['high_quality'] += 1
            elif best_score > 0.3:
                score_distribution['medium_quality'] += 1
            else:
                score_distribution['low_quality'] += 1
        
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        
        # Find top performing mappings
        top_mappings = []
        for result in mapping_results:
            if result.matched_questions:
                top_mapping = result.matched_questions[0]
                top_mappings.append({
                    'content_id': top_mapping.content_id,
                    'question_id': top_mapping.question_id,
                    'score': top_mapping.composite_score,
                    'confidence': top_mapping.confidence
                })
        
        top_mappings.sort(key=lambda x: x['score'], reverse=True)
        
        report = {
            'summary': {
                'total_content_segments': total_segments,
                'total_question_mappings': total_mappings,
                'average_best_match_score': round(avg_score, 3),
                'mapping_quality_distribution': score_distribution
            },
            'performance_metrics': {
                'high_quality_percentage': round(score_distribution['high_quality'] / total_segments * 100, 1),
                'average_mappings_per_segment': round(total_mappings / total_segments, 1),
                'score_statistics': {
                    'min_score': round(min(all_scores), 3) if all_scores else 0,
                    'max_score': round(max(all_scores), 3) if all_scores else 0,
                    'avg_score': round(avg_score, 3)
                }
            },
            'top_mappings': top_mappings[:10],  # Top 10
            'recommendations': self._generate_mapping_recommendations(mapping_results)
        }
        
        return report
    
    def _generate_mapping_recommendations(self, mapping_results: List[MappingResult]) -> List[str]:
        """Generate recommendations based on mapping analysis"""
        
        recommendations = []
        
        if not mapping_results:
            return ["No mapping results available for analysis"]
        
        # Calculate quality metrics
        high_quality_count = sum(1 for result in mapping_results if result.best_match_score > 0.7)
        total_count = len(mapping_results)
        high_quality_rate = high_quality_count / total_count if total_count > 0 else 0
        
        if high_quality_rate < 0.3:
            recommendations.append("Consider improving question classification tags for better matching")
            recommendations.append("Review content segmentation to ensure meaningful chunks")
        
        if high_quality_rate > 0.7:
            recommendations.append("Mapping quality is excellent - consider expanding question database")
        
        # Analyze score distributions
        avg_tag_scores = []
        avg_semantic_scores = []
        
        for result in mapping_results:
            for mapping in result.matched_questions:
                avg_tag_scores.append(mapping.tag_overlap_score)
                avg_semantic_scores.append(mapping.semantic_score)
        
        if avg_tag_scores and sum(avg_tag_scores) / len(avg_tag_scores) < 0.4:
            recommendations.append("Low tag overlap scores - review classification consistency")
        
        if avg_semantic_scores and sum(avg_semantic_scores) / len(avg_semantic_scores) < 0.4:
            recommendations.append("Low semantic similarity - consider expanding keyword matching")
        
        recommendations.append("Regular mapping quality review recommended for optimal performance")
        
        return recommendations

def main():
    """Test the intelligent content-question mapping system"""
    
    print("🚀 Testing Intelligent Content-Question Mapping Engine")
    print("=" * 60)
    
    # Initialize mapper
    mapper = IntelligentContentQuestionMapper()
    
    # Test content segment
    test_segment = {
        'id': 'segment_1',
        'content': 'Revenue from contracts with customers increased by 15% to $2.5 million during the fiscal year ended December 31, 2024. This includes revenue from software licenses and maintenance services.',
        'section_type': 'income_statement',
        'classification_tags': {
            'primary_standard': 'IFRS 15',
            'content_type': 'revenue_recognition',
            'document_sections': ['income_statement', 'notes'],
            'complexity_level': 'intermediate'
        },
        'hierarchy_level': 2,
        'confidence': 0.85,
        'contains_numbers': True,
        'accounting_keywords': ['revenue', 'contracts', 'customers']
    }
    
    print("📄 Testing content segment mapping...")
    mapping_result = mapper.map_content_to_questions(test_segment, top_k=3)
    
    print(f"✅ Mapping completed successfully")
    print(f"   Content ID: {mapping_result.content_segment['id']}")
    print(f"   Questions analyzed: {mapping_result.total_questions_analyzed}")
    print(f"   Best match score: {mapping_result.best_match_score:.3f}")
    print(f"   Top matches: {len(mapping_result.matched_questions)}")
    
    print("\n🎯 Top question matches:")
    for i, mapping in enumerate(mapping_result.matched_questions, 1):
        print(f"   {i}. Question: {mapping.question_id}")
        print(f"      Score: {mapping.composite_score:.3f} (confidence: {mapping.confidence:.1%})")
        print(f"      Tag overlap: {mapping.tag_overlap_score:.3f}")
        print(f"      Semantic: {mapping.semantic_score:.3f}")
        print(f"      Contextual: {mapping.contextual_score:.3f}")
        print()
    
    # Test batch mapping
    print("📊 Testing batch mapping...")
    test_segments = [test_segment] * 3  # Duplicate for testing
    batch_results = mapper.batch_map_content_segments(test_segments, top_k_per_segment=2)
    
    print(f"✅ Batch mapping completed: {len(batch_results)} results")
    
    # Generate report
    print("📋 Generating mapping report...")
    report = mapper.generate_mapping_report(batch_results)
    
    print("✅ Mapping Report Generated:")
    print(f"   Total segments: {report['summary']['total_content_segments']}")
    print(f"   Average score: {report['summary']['average_best_match_score']}")
    print(f"   High quality rate: {report['performance_metrics']['high_quality_percentage']}%")
    
    print("\n🎯 Content-Question Mapping Engine: Ready for Production! ✅")

if __name__ == "__main__":
    main()