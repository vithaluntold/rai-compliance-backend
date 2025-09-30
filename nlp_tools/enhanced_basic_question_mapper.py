#!/usr/bin/env python3
"""
Enhanced-to-Basic Question Mapping Engine
Maps enhanced framework questions (5D tagged) to basic framework questions for compliance analysis
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class QuestionMapping:
    """Mapping between enhanced and basic questions"""
    enhanced_id: str
    enhanced_question: str
    basic_question_ids: List[str]
    basic_questions: List[str]
    mapping_confidence: float
    coverage_weight: float
    ifrs_reference: str

@dataclass
class ChunkAssignment:
    """Assignment of document chunks to basic questions"""
    chunk_id: str
    chunk_content: str
    assigned_basic_questions: List[str]
    enhanced_questions_matched: List[str]
    assignment_confidence: float
    reasoning: str

class EnhancedBasicQuestionMapper:
    """Maps enhanced framework questions to basic questions and routes chunks accordingly"""
    
    def __init__(self):
        """Initialize the mapping system"""
        self.enhanced_framework_path = Path(__file__).parent.parent / "checklist_data" / "Enhanced Framework" / "IFRS"
        self.basic_framework_path = Path(__file__).parent.parent / "checklist_data" / "frameworks" / "IFRS"
        
        self.question_mappings = {}  # enhanced_id -> QuestionMapping
        self.reverse_mappings = {}   # basic_id -> List[enhanced_ids]
        
        logger.info("Enhanced-Basic Question Mapper initialized")
    
    def load_question_mappings(self, standard: str) -> Dict[str, QuestionMapping]:
        """Load and create mappings between enhanced and basic questions for a standard"""
        
        # Load enhanced questions
        enhanced_file = self.enhanced_framework_path / f"enhanced_{standard}.json"
        basic_file = self.basic_framework_path / standard / "checklist.json"
        
        if not enhanced_file.exists():
            logger.warning(f"Enhanced framework file not found: {enhanced_file}")
            return {}
        
        if not basic_file.exists():
            logger.warning(f"Basic framework file not found: {basic_file}")
            return {}
        
        with open(enhanced_file, 'r') as f:
            enhanced_data = json.load(f)
        
        with open(basic_file, 'r') as f:
            basic_data = json.load(f)
        
        # Extract questions
        enhanced_questions = self._extract_enhanced_questions(enhanced_data)
        basic_questions = self._extract_basic_questions(basic_data)
        
        # Create mappings using similarity matching
        mappings = self._create_question_mappings(enhanced_questions, basic_questions, standard)
        
        logger.info(f"Created {len(mappings)} question mappings for {standard}")
        return mappings
    
    def _extract_enhanced_questions(self, enhanced_data: Dict) -> List[Dict]:
        """Extract enhanced questions with 5D tags"""
        questions = []
        
        for section in enhanced_data.get("sections", []):
            for item in section.get("items", []):
                questions.append({
                    "id": item.get("id"),
                    "question": item.get("question"),
                    "reference": item.get("reference"),
                    "facet_focus": item.get("facet_focus", {}),
                    "section": item.get("section")
                })
        
        return questions
    
    def _extract_basic_questions(self, basic_data: Dict) -> List[Dict]:
        """Extract basic questions"""
        questions = []
        
        for section in basic_data.get("sections", []):
            for item in section.get("items", []):
                questions.append({
                    "id": item.get("id"),
                    "question": item.get("question"),
                    "reference": item.get("reference", ""),
                    "section": section.get("section", "")
                })
        
        return questions
    
    def _create_question_mappings(self, enhanced_questions: List[Dict], basic_questions: List[Dict], standard: str) -> Dict[str, QuestionMapping]:
        """Create mappings between enhanced and basic questions using semantic similarity"""
        mappings = {}
        
        for enhanced_q in enhanced_questions:
            # Find matching basic questions
            matches = self._find_matching_basic_questions(enhanced_q, basic_questions)
            
            if matches:
                mapping = QuestionMapping(
                    enhanced_id=enhanced_q["id"],
                    enhanced_question=enhanced_q["question"],
                    basic_question_ids=[m["id"] for m in matches],
                    basic_questions=[m["question"] for m in matches],
                    mapping_confidence=self._calculate_mapping_confidence(enhanced_q, matches),
                    coverage_weight=self._calculate_coverage_weight(enhanced_q),
                    ifrs_reference=enhanced_q.get("reference", "")
                )
                
                mappings[enhanced_q["id"]] = mapping
                
                # Update reverse mappings
                for match in matches:
                    if match["id"] not in self.reverse_mappings:
                        self.reverse_mappings[match["id"]] = []
                    self.reverse_mappings[match["id"]].append(enhanced_q["id"])
        
        return mappings
    
    def _find_matching_basic_questions(self, enhanced_q: Dict, basic_questions: List[Dict]) -> List[Dict]:
        """Find basic questions that match an enhanced question"""
        matches = []
        enhanced_text = enhanced_q["question"].lower()
        enhanced_ref = enhanced_q.get("reference", "").lower()
        
        for basic_q in basic_questions:
            basic_text = basic_q["question"].lower()
            basic_ref = basic_q.get("reference", "").lower()
            
            # Similarity scoring
            similarity_score = 0.0
            
            # 1. IFRS reference matching (high weight)
            if enhanced_ref and basic_ref and enhanced_ref == basic_ref:
                similarity_score += 0.4
            
            # 2. Key term overlap (medium weight)
            enhanced_terms = set(enhanced_text.split())
            basic_terms = set(basic_text.split())
            common_terms = enhanced_terms.intersection(basic_terms)
            term_overlap = len(common_terms) / max(len(enhanced_terms), len(basic_terms), 1)
            similarity_score += term_overlap * 0.3
            
            # 3. Concept matching (medium weight)
            concept_match = self._calculate_concept_similarity(enhanced_text, basic_text)
            similarity_score += concept_match * 0.3
            
            # Accept matches above threshold
            if similarity_score > 0.6:
                matches.append({
                    **basic_q,
                    "similarity_score": similarity_score
                })
        
        # Sort by similarity and return top matches
        matches.sort(key=lambda x: x["similarity_score"], reverse=True)
        return matches[:3]  # Max 3 basic questions per enhanced question
    
    def _calculate_concept_similarity(self, enhanced_text: str, basic_text: str) -> float:
        """Calculate conceptual similarity between questions"""
        
        # Key financial concepts mapping
        concept_keywords = {
            "inventory": ["inventory", "inventories", "stock", "goods"],
            "measurement": ["measure", "measurement", "valued", "valuation", "cost"],
            "disclosure": ["disclose", "disclosure", "present", "presentation"],
            "policy": ["policy", "policies", "method", "approach"],
            "fair_value": ["fair value", "market value", "valuation"],
            "depreciation": ["depreciation", "amortization", "useful life"],
            "impairment": ["impairment", "write-down", "recoverable"]
        }
        
        enhanced_concepts = set()
        basic_concepts = set()
        
        for concept, keywords in concept_keywords.items():
            if any(keyword in enhanced_text for keyword in keywords):
                enhanced_concepts.add(concept)
            if any(keyword in basic_text for keyword in keywords):
                basic_concepts.add(concept)
        
        if not enhanced_concepts and not basic_concepts:
            return 0.0
        
        common_concepts = enhanced_concepts.intersection(basic_concepts)
        return len(common_concepts) / max(len(enhanced_concepts.union(basic_concepts)), 1)
    
    def _calculate_mapping_confidence(self, enhanced_q: Dict, matches: List[Dict]) -> float:
        """Calculate confidence in the mapping"""
        if not matches:
            return 0.0
        
        # Average similarity score of matches
        avg_similarity = sum(m["similarity_score"] for m in matches) / len(matches)
        
        # Adjust for number of matches (1-2 matches preferred)
        match_penalty = 0.0 if len(matches) <= 2 else 0.1 * (len(matches) - 2)
        
        return max(0.0, avg_similarity - match_penalty)
    
    def _calculate_coverage_weight(self, enhanced_q: Dict) -> float:
        """Calculate coverage weight based on enhanced question importance"""
        facet_focus = enhanced_q.get("facet_focus", {})
        
        # Higher weight for mandatory disclosures
        narrative_cats = facet_focus.get("narrative_categories", [])
        if "disclosure_narrative" in narrative_cats:
            return 0.9
        elif "accounting_policies_note" in narrative_cats:
            return 0.8
        elif "measurement_basis" in narrative_cats:
            return 0.7
        else:
            return 0.6
    
    def assign_chunks_to_basic_questions(self, chunk_mappings: List[Dict], standard: str) -> List[ChunkAssignment]:
        """Assign document chunks to basic questions via enhanced question mappings"""
        
        # Load question mappings for this standard
        question_mappings = self.load_question_mappings(standard)
        
        assignments = []
        
        for chunk_mapping in chunk_mappings:
            chunk_id = chunk_mapping.get("content_segment", {}).get("segment_id")
            chunk_content = chunk_mapping.get("content_segment", {}).get("content", "")
            matched_enhanced = chunk_mapping.get("matched_questions", [])
            
            # Collect all basic questions for this chunk
            assigned_basic_questions = []
            enhanced_questions_matched = []
            total_confidence = 0.0
            
            for enhanced_match in matched_enhanced:
                enhanced_id = enhanced_match.get("question_id")
                match_confidence = enhanced_match.get("composite_score", 0.0)
                
                if enhanced_id in question_mappings:
                    mapping = question_mappings[enhanced_id]
                    assigned_basic_questions.extend(mapping.basic_question_ids)
                    enhanced_questions_matched.append(enhanced_id)
                    total_confidence += match_confidence * mapping.coverage_weight
            
            # Remove duplicates
            assigned_basic_questions = list(set(assigned_basic_questions))
            
            if assigned_basic_questions:
                assignment = ChunkAssignment(
                    chunk_id=chunk_id,
                    chunk_content=chunk_content,
                    assigned_basic_questions=assigned_basic_questions,
                    enhanced_questions_matched=enhanced_questions_matched,
                    assignment_confidence=total_confidence / max(len(matched_enhanced), 1),
                    reasoning=f"Mapped via {len(enhanced_questions_matched)} enhanced questions"
                )
                
                assignments.append(assignment)
        
        logger.info(f"Created {len(assignments)} chunk assignments for {standard}")
        return assignments
    
    def get_compliance_analysis_input(self, assignments: List[ChunkAssignment], standard: str) -> Dict[str, Any]:
        """Generate input for compliance analysis using basic questions"""
        
        # Group chunks by basic questions
        question_to_chunks = {}
        
        for assignment in assignments:
            for basic_q_id in assignment.assigned_basic_questions:
                if basic_q_id not in question_to_chunks:
                    question_to_chunks[basic_q_id] = []
                
                question_to_chunks[basic_q_id].append({
                    "chunk_id": assignment.chunk_id,
                    "content": assignment.chunk_content,
                    "confidence": assignment.assignment_confidence
                })
        
        # Load basic questions for compliance analysis
        basic_file = self.basic_framework_path / standard / "checklist.json"
        with open(basic_file, 'r') as f:
            basic_data = json.load(f)
        
        # Structure for compliance analysis
        compliance_input = {
            "framework": "IFRS",
            "standard": standard,
            "sections": [],
            "chunk_assignments": question_to_chunks
        }
        
        # Add sections with assigned chunks
        for section in basic_data.get("sections", []):
            section_items = []
            
            for item in section.get("items", []):
                question_id = item.get("id")
                if question_id in question_to_chunks:
                    # This question has assigned chunks
                    item_with_chunks = {
                        **item,
                        "assigned_chunks": question_to_chunks[question_id],
                        "chunk_count": len(question_to_chunks[question_id])
                    }
                    section_items.append(item_with_chunks)
            
            if section_items:
                compliance_input["sections"].append({
                    "section": section.get("section"),
                    "title": section.get("title"),
                    "items": section_items
                })
        
        return compliance_input

def create_question_mapping_demo():
    """Demo function showing the mapping process"""
    mapper = EnhancedBasicQuestionMapper()
    
    # Example chunk mappings from NLP pipeline
    example_chunk_mappings = [
        {
            "content_segment": {
                "segment_id": "note_002",
                "content": "Inventories are valued at the lower of cost and net realizable value using FIFO method"
            },
            "matched_questions": [
                {
                    "question_id": "423.1",  # Enhanced question ID
                    "composite_score": 0.91,
                    "confidence": 0.89
                }
            ]
        }
    ]
    
    # Create assignments
    assignments = mapper.assign_chunks_to_basic_questions(example_chunk_mappings, "IAS 2")
    
    # Generate compliance analysis input
    compliance_input = mapper.get_compliance_analysis_input(assignments, "IAS 2")
    
    return compliance_input

if __name__ == "__main__":
    demo_result = create_question_mapping_demo()
    print(json.dumps(demo_result, indent=2))