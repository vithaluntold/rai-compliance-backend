#!/usr/bin/env python3
"""
Complete Compliance Analysis Pipeline with Enhanced-Basic Question Mapping
Integrates NLP processing with smart question routing for compliance analysis
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json

# Import existing NLP tools
from nlp_tools.complete_nlp_validation_pipeline import CompleteNLPValidationPipeline
from nlp_tools.intelligent_content_question_mapper import IntelligentContentQuestionMapper
from nlp_tools.enhanced_basic_question_mapper import EnhancedBasicQuestionMapper

# Import existing compliance analysis
from services.ai import get_ai_service

logger = logging.getLogger(__name__)

@dataclass
class ComplianceAnalysisResult:
    """Complete compliance analysis result"""
    document_id: str
    standard: str
    
    # NLP Processing Results
    structure_analysis: Dict[str, Any]
    content_classification: Dict[str, Any] 
    taxonomy_validation: Dict[str, Any]
    
    # Enhanced Framework Mapping
    enhanced_question_mappings: List[Dict[str, Any]]
    coverage_analysis: Dict[str, Any]
    
    # Basic Question Assignment
    chunk_assignments: List[Dict[str, Any]]
    
    # Final Compliance Results
    compliance_results: Dict[str, Any]
    overall_score: float
    
    # Processing Metadata
    processing_time: float
    token_usage: int

class CompleteComplianceAnalysisPipeline:
    """Complete pipeline: NLP → Enhanced Mapping → Basic Questions → Compliance Analysis"""
    
    def __init__(self):
        """Initialize the complete pipeline"""
        
        # Core NLP Pipeline
        self.nlp_pipeline = CompleteNLPValidationPipeline()
        
        # Question Mapping Systems
        self.content_question_mapper = IntelligentContentQuestionMapper()
        self.enhanced_basic_mapper = EnhancedBasicQuestionMapper()
        
        # Compliance Analysis Service
        self.ai_service = get_ai_service()
        
        logger.info("Complete Compliance Analysis Pipeline initialized")
    
    def analyze_document_compliance(self, document_path: str, standard: str) -> ComplianceAnalysisResult:
        """
        Complete compliance analysis workflow
        
        Args:
            document_path: Path to financial statement document
            standard: Accounting standard (e.g., "IAS 2", "IFRS 15")
            
        Returns:
            ComplianceAnalysisResult with complete analysis
        """
        
        import time
        start_time = time.time()
        token_usage = 0
        
        logger.info(f"Starting complete compliance analysis for {document_path} under {standard}")
        
        try:
            # STEP 1: NLP Document Processing
            logger.info("Step 1: NLP document processing...")
            nlp_result = self.nlp_pipeline.process_document_with_validation(document_path)
            
            if not nlp_result.success:
                raise ValueError(f"NLP processing failed: {nlp_result.error}")
            
            # STEP 2: Enhanced Framework Question Mapping  
            logger.info("Step 2: Enhanced framework question mapping...")
            enhanced_mappings = self._map_to_enhanced_questions(nlp_result, standard)
            
            # STEP 3: Coverage Analysis
            logger.info("Step 3: Coverage analysis...")
            coverage_analysis = self._analyze_coverage(enhanced_mappings, standard)
            
            # STEP 4: Enhanced → Basic Question Mapping
            logger.info("Step 4: Enhanced to basic question mapping...")
            chunk_assignments = self.enhanced_basic_mapper.assign_chunks_to_basic_questions(
                enhanced_mappings, standard
            )
            
            # STEP 5: Generate Compliance Analysis Input
            logger.info("Step 5: Generating compliance analysis input...")
            compliance_input = self.enhanced_basic_mapper.get_compliance_analysis_input(
                chunk_assignments, standard
            )
            
            # STEP 6: Run Compliance Analysis with Basic Questions
            logger.info("Step 6: Running compliance analysis...")
            compliance_results = self._run_compliance_analysis(compliance_input)
            token_usage = compliance_results.get("metadata", {}).get("token_usage", 0)
            
            # STEP 7: Calculate Overall Score
            overall_score = self._calculate_overall_score(compliance_results, coverage_analysis)
            
            processing_time = time.time() - start_time
            
            result = ComplianceAnalysisResult(
                document_id=document_path,
                standard=standard,
                structure_analysis=nlp_result.structure_parsing,
                content_classification=nlp_result.content_classification,
                taxonomy_validation=nlp_result.taxonomy_validation.__dict__ if nlp_result.taxonomy_validation else {},
                enhanced_question_mappings=enhanced_mappings,
                coverage_analysis=coverage_analysis,
                chunk_assignments=[assignment.__dict__ for assignment in chunk_assignments],
                compliance_results=compliance_results,
                overall_score=overall_score,
                processing_time=processing_time,
                token_usage=token_usage
            )
            
            logger.info(f"Compliance analysis completed in {processing_time:.2f}s - Score: {overall_score:.1%}")
            return result
            
        except Exception as e:
            logger.error(f"Compliance analysis failed: {e}")
            
            # Return error result
            processing_time = time.time() - start_time
            return ComplianceAnalysisResult(
                document_id=document_path,
                standard=standard,
                structure_analysis={"error": str(e)},
                content_classification={"error": str(e)},
                taxonomy_validation={"error": str(e)},
                enhanced_question_mappings=[],
                coverage_analysis={"error": str(e)},
                chunk_assignments=[],
                compliance_results={"error": str(e)},
                overall_score=0.0,
                processing_time=processing_time,
                token_usage=0
            )
    
    def _map_to_enhanced_questions(self, nlp_result, standard: str) -> List[Dict[str, Any]]:
        """Map NLP-processed content to enhanced framework questions"""
        
        # Extract validated segments from NLP result
        validated_segments = []
        if nlp_result.validated_mega_chunks:
            for standard_key, chunks in nlp_result.validated_mega_chunks.items():
                if isinstance(chunks, dict):
                    for chunk_id, chunk_data in chunks.items():
                        validated_segments.append({
                            "segment_id": chunk_id,
                            "content": chunk_data.get("content", ""),
                            "classification_tags": chunk_data.get("classification_tags", {}),
                            "accounting_standard": chunk_data.get("accounting_standard", ""),
                            "confidence_score": chunk_data.get("confidence_score", 0.0)
                        })
        
        # Use content-question mapper to find enhanced question matches
        enhanced_mappings = []
        
        for segment in validated_segments:
            # Only map segments that match the target standard
            if segment["accounting_standard"] == standard or standard in segment["accounting_standard"]:
                
                mapping_result = self.content_question_mapper.map_content_to_questions(
                    content_segment=segment,
                    questions_directory=f"Enhanced Framework/IFRS"
                )
                
                if mapping_result and mapping_result.matched_questions:
                    enhanced_mappings.append({
                        "content_segment": segment,
                        "matched_questions": [
                            {
                                "question_id": match.question_id,
                                "similarity_score": match.similarity_score,
                                "tag_overlap_score": match.tag_overlap_score,
                                "semantic_score": match.semantic_score,
                                "composite_score": match.composite_score,
                                "confidence": match.confidence
                            }
                            for match in mapping_result.matched_questions
                        ],
                        "best_match_score": mapping_result.best_match_score,
                        "mapping_method": mapping_result.mapping_method
                    })
        
        return enhanced_mappings
    
    def _analyze_coverage(self, enhanced_mappings: List[Dict[str, Any]], standard: str) -> Dict[str, Any]:
        """Analyze coverage of enhanced framework questions"""
        
        # Load all enhanced questions for this standard
        enhanced_file = self.enhanced_basic_mapper.enhanced_framework_path / f"enhanced_{standard}.json"
        
        if not enhanced_file.exists():
            return {"error": f"Enhanced framework file not found for {standard}"}
        
        with open(enhanced_file, 'r') as f:
            enhanced_data = json.load(f)
        
        # Extract all question IDs
        all_enhanced_questions = set()
        mandatory_questions = set()
        
        for section in enhanced_data.get("sections", []):
            for item in section.get("items", []):
                question_id = item.get("id")
                all_enhanced_questions.add(question_id)
                
                # Check if mandatory based on conditionality
                if item.get("conditionality", {}).get("requirement_level") == "mandatory":
                    mandatory_questions.add(question_id)
        
        # Find covered questions
        covered_questions = set()
        for mapping in enhanced_mappings:
            for match in mapping["matched_questions"]:
                covered_questions.add(match["question_id"])
        
        # Calculate coverage metrics
        total_questions = len(all_enhanced_questions)
        covered_count = len(covered_questions)
        mandatory_covered = len(mandatory_questions.intersection(covered_questions))
        mandatory_total = len(mandatory_questions)
        
        coverage_analysis = {
            "total_questions": total_questions,
            "covered_questions": covered_count,
            "uncovered_questions": total_questions - covered_count,
            "overall_coverage": covered_count / max(total_questions, 1),
            "mandatory_questions": mandatory_total,
            "mandatory_covered": mandatory_covered,
            "mandatory_coverage": mandatory_covered / max(mandatory_total, 1),
            "covered_question_ids": list(covered_questions),
            "uncovered_question_ids": list(all_enhanced_questions - covered_questions)
        }
        
        return coverage_analysis
    
    def _run_compliance_analysis(self, compliance_input: Dict[str, Any]) -> Dict[str, Any]:
        """Run compliance analysis using basic questions and AI service"""
        
        try:
            # Extract framework and standard
            framework = compliance_input.get("framework", "IFRS")
            standard = compliance_input.get("standard", "")
            
            # Process each section with the AI service
            results = {
                "framework": framework,
                "standard": standard,
                "sections": [],
                "metadata": {"token_usage": 0}
            }
            
            total_tokens = 0
            
            for section in compliance_input.get("sections", []):
                section_results = {
                    "section": section.get("section"),
                    "title": section.get("title"),
                    "items": []
                }
                
                for item in section.get("items", []):
                    question = item.get("question", "")
                    assigned_chunks = item.get("assigned_chunks", [])
                    
                    if assigned_chunks:
                        # Combine chunk content for analysis
                        combined_content = "\n\n".join([
                            chunk["content"] for chunk in assigned_chunks
                        ])
                        
                        # Use existing AI service for analysis
                        try:
                            analysis_result = self.ai_service.analyze_chunk(
                                chunk=combined_content,
                                question=question,
                                standard_id=standard
                            )
                            
                            item_result = {
                                "id": item.get("id"),
                                "question": question,
                                "analysis": analysis_result,
                                "chunk_count": len(assigned_chunks),
                                "chunks_used": [chunk["chunk_id"] for chunk in assigned_chunks]
                            }
                            
                            # Track token usage if available
                            if isinstance(analysis_result, dict):
                                total_tokens += analysis_result.get("token_usage", 0)
                            
                        except Exception as e:
                            logger.warning(f"Analysis failed for question {item.get('id')}: {e}")
                            item_result = {
                                "id": item.get("id"), 
                                "question": question,
                                "analysis": {"error": str(e)},
                                "chunk_count": len(assigned_chunks),
                                "chunks_used": [chunk["chunk_id"] for chunk in assigned_chunks]
                            }
                        
                        section_results["items"].append(item_result)
                
                results["sections"].append(section_results)
            
            results["metadata"]["token_usage"] = total_tokens
            return results
            
        except Exception as e:
            logger.error(f"Compliance analysis failed: {e}")
            return {"error": str(e), "metadata": {"token_usage": 0}}
    
    def _calculate_overall_score(self, compliance_results: Dict[str, Any], coverage_analysis: Dict[str, Any]) -> float:
        """Calculate overall compliance score"""
        
        if "error" in compliance_results:
            return 0.0
        
        # Extract compliance scores from results
        compliance_scores = []
        
        for section in compliance_results.get("sections", []):
            for item in section.get("items", []):
                analysis = item.get("analysis", {})
                if isinstance(analysis, dict) and "compliance_score" in analysis:
                    compliance_scores.append(analysis["compliance_score"])
        
        # Calculate weighted average
        if compliance_scores:
            avg_compliance = sum(compliance_scores) / len(compliance_scores)
        else:
            avg_compliance = 0.0
        
        # Factor in coverage
        coverage_score = coverage_analysis.get("overall_coverage", 0.0)
        
        # Combined score (70% compliance quality, 30% coverage)
        overall_score = (avg_compliance * 0.7) + (coverage_score * 0.3)
        
        return overall_score

# Demo function
def run_compliance_analysis_demo():
    """Demo the complete compliance analysis pipeline"""
    
    pipeline = CompleteComplianceAnalysisPipeline()
    
    # Example: Analyze a document against IAS 2 standard
    result = pipeline.analyze_document_compliance(
        document_path="test-complete-financial-statements.txt",
        standard="IAS 2"
    )
    
    print(f"Compliance Analysis Results:")
    print(f"Standard: {result.standard}")
    print(f"Overall Score: {result.overall_score:.1%}")
    print(f"Processing Time: {result.processing_time:.2f}s")
    print(f"Token Usage: {result.token_usage}")
    print(f"Enhanced Questions Mapped: {len(result.enhanced_question_mappings)}")
    print(f"Chunk Assignments: {len(result.chunk_assignments)}")
    
    return result

if __name__ == "__main__":
    demo_result = run_compliance_analysis_demo()