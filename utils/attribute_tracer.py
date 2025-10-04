"""
Result Page Attribute Tracer

This module provides utilities to trace how each attribute on the result page
was generated, including the specific AI models, data sources, and processing
stages involved.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AttributeTrace:
    """Represents the complete trace of how a result page attribute was generated"""
    
    attribute_name: str
    current_value: Any
    data_source_model: str
    data_source_field: str
    processing_stages: List[Dict[str, Any]]
    ai_components_used: List[str]
    input_data: Dict[str, Any]
    processing_metadata: Dict[str, Any]
    timestamp: datetime


class ResultPageAttributeTracer:
    """
    Provides tracing capabilities for result page attributes
    """
    
    def __init__(self):
        self.traces: Dict[str, AttributeTrace] = {}
        
    def trace_question_section(self, checklist_item: Dict[str, Any], 
                              framework_data: Dict[str, Any]) -> AttributeTrace:
        """Trace how the question section was generated"""
        
        return AttributeTrace(
            attribute_name="question_section",
            current_value={
                "question": checklist_item.get("question", ""),
                "reference": checklist_item.get("reference", ""),
                "section": checklist_item.get("section", "")
            },
            data_source_model="ChecklistItem",
            data_source_field="question, reference, section",
            processing_stages=[
                {
                    "stage": "framework_selection",
                    "input": framework_data.get("selected_framework"),
                    "processor": "checklist_utils.load_checklist()",
                    "output": "checklist_items"
                },
                {
                    "stage": "standards_selection", 
                    "input": framework_data.get("selected_standards"),
                    "processor": "JSON file loading",
                    "output": "compliance_questions"
                }
            ],
            ai_components_used=["Azure OpenAI for standards recommendation"],
            input_data={
                "framework": framework_data.get("framework_id"),
                "standards": framework_data.get("selected_standards"),
                "checklist_file": f"checklist_data/frameworks/{framework_data.get('framework_id')}/{checklist_item.get('section', '')}.json"
            },
            processing_metadata={
                "checklist_loaded_at": datetime.now().isoformat(),
                "total_questions_in_standard": framework_data.get("total_questions", 0)
            },
            timestamp=datetime.now()
        )
    
    def trace_confidence_score(self, checklist_item: Dict[str, Any],
                              ai_analysis_log: Dict[str, Any]) -> AttributeTrace:
        """Trace how the confidence score was calculated"""
        
        return AttributeTrace(
            attribute_name="confidence_score",
            current_value=checklist_item.get("confidence", 0.0),
            data_source_model="ChecklistItem", 
            data_source_field="confidence",
            processing_stages=[
                {
                    "stage": "document_vectorization",
                    "input": "document_chunks",
                    "processor": "text-embedding-3-large",
                    "output": "vector_embeddings"
                },
                {
                    "stage": "similarity_search",
                    "input": "question_embedding + document_vectors",
                    "processor": "FAISS similarity search",
                    "output": "relevant_chunks"
                },
                {
                    "stage": "ai_analysis",
                    "input": "question + relevant_chunks",
                    "processor": "Azure OpenAI o3-mini",
                    "output": "confidence_score"
                }
            ],
            ai_components_used=[
                "text-embedding-3-large (embeddings)",
                "o3-mini (analysis)"
            ],
            input_data={
                "question_text": checklist_item.get("question"),
                "document_chunks": ai_analysis_log.get("chunks_analyzed", []),
                "vector_similarity_scores": ai_analysis_log.get("similarity_scores", [])
            },
            processing_metadata={
                "token_usage": ai_analysis_log.get("token_usage", {}),
                "processing_time_seconds": ai_analysis_log.get("processing_time"),
                "chunks_analyzed_count": len(ai_analysis_log.get("chunks_analyzed", [])),
                "top_similarity_score": max(ai_analysis_log.get("similarity_scores", [0]))
            },
            timestamp=datetime.now()
        )
    
    def trace_explanation_text(self, checklist_item: Dict[str, Any],
                              ai_response_log: Dict[str, Any]) -> AttributeTrace:
        """Trace how the explanation text was generated"""
        
        return AttributeTrace(
            attribute_name="explanation_text",
            current_value=checklist_item.get("explanation", ""),
            data_source_model="ChecklistItem",
            data_source_field="explanation", 
            processing_stages=[
                {
                    "stage": "context_assembly",
                    "input": "question + top_matching_chunks",
                    "processor": "context_builder",
                    "output": "analysis_context"
                },
                {
                    "stage": "prompt_generation",
                    "input": "analysis_context + system_prompt",
                    "processor": "ai_prompts.get_compliance_analysis_system_prompt()",
                    "output": "full_prompt"
                },
                {
                    "stage": "ai_explanation_generation",
                    "input": "full_prompt",
                    "processor": "Azure OpenAI o3-mini",
                    "output": "explanation_text"
                }
            ],
            ai_components_used=["Azure OpenAI o3-mini"],
            input_data={
                "system_prompt": ai_response_log.get("system_prompt"),
                "user_prompt": ai_response_log.get("user_prompt"),
                "context_chunks": ai_response_log.get("context_chunks", []),
                "question_reference": checklist_item.get("reference")
            },
            processing_metadata={
                "prompt_tokens": ai_response_log.get("prompt_tokens"),
                "completion_tokens": ai_response_log.get("completion_tokens"),
                "total_tokens": ai_response_log.get("total_tokens"),
                "model_response_time": ai_response_log.get("response_time"),
                "context_chunks_count": len(ai_response_log.get("context_chunks", []))
            },
            timestamp=datetime.now()
        )
    
    def trace_evidence_table(self, checklist_item: Dict[str, Any],
                            vector_search_results: List[Dict[str, Any]]) -> AttributeTrace:
        """Trace how the evidence table was populated"""
        
        evidence_rows = []
        for result in vector_search_results:
            evidence_rows.append({
                "reference": checklist_item.get("reference"),
                "requirement": "N/A",  # Usually not populated from vector search
                "description": "N/A",   # Usually not populated from vector search  
                "page_number": result.get("metadata", {}).get("page", "N/A"),
                "extract": result.get("content", ""),
                "content_analysis": result.get("analysis", "")
            })
        
        return AttributeTrace(
            attribute_name="evidence_table",
            current_value=evidence_rows,
            data_source_model="VectorStore + ChecklistItem",
            data_source_field="vector_search_results + reference",
            processing_stages=[
                {
                    "stage": "document_chunking",
                    "input": "uploaded_document",
                    "processor": "document_chunker.py",
                    "output": "text_chunks_with_metadata"
                },
                {
                    "stage": "vector_indexing", 
                    "input": "text_chunks",
                    "processor": "FAISS + text-embedding-3-large",
                    "output": "vector_index"
                },
                {
                    "stage": "similarity_search",
                    "input": "question_embedding + vector_index",
                    "processor": "vector_store.similarity_search()",
                    "output": "relevant_chunks_with_scores"
                },
                {
                    "stage": "evidence_formatting",
                    "input": "relevant_chunks + question_reference",
                    "processor": "evidence_table_builder",
                    "output": "formatted_evidence_rows"
                }
            ],
            ai_components_used=[
                "text-embedding-3-large (document embeddings)",
                "text-embedding-3-large (question embedding)"
            ],
            input_data={
                "question_text": checklist_item.get("question"),
                "search_results_count": len(vector_search_results),
                "similarity_threshold": 0.7,  # Usually configurable
                "max_results": 10  # Usually configurable
            },
            processing_metadata={
                "total_chunks_in_index": len(vector_search_results),
                "chunks_above_threshold": len([r for r in vector_search_results if r.get("score", 0) > 0.7]),
                "avg_similarity_score": sum(r.get("score", 0) for r in vector_search_results) / len(vector_search_results) if vector_search_results else 0,
                "search_execution_time": 0.15  # Usually tracked
            },
            timestamp=datetime.now()
        )
    
    def trace_suggested_disclosure(self, checklist_item: Dict[str, Any],
                                  company_metadata: Dict[str, Any],
                                  ai_suggestion_log: Dict[str, Any]) -> AttributeTrace:
        """Trace how the suggested disclosure was generated"""
        
        return AttributeTrace(
            attribute_name="suggested_disclosure",
            current_value=checklist_item.get("suggestion", ""),
            data_source_model="ChecklistItem",
            data_source_field="suggestion",
            processing_stages=[
                {
                    "stage": "context_preparation",
                    "input": "compliance_analysis_result + company_metadata",
                    "processor": "suggestion_context_builder",
                    "output": "suggestion_context"
                },
                {
                    "stage": "prompt_customization",
                    "input": "suggestion_context + company_profile",
                    "processor": "ai_prompts.get_disclosure_recommendation_prompt()",
                    "output": "customized_prompt"
                },
                {
                    "stage": "ai_recommendation_generation",
                    "input": "customized_prompt",
                    "processor": "Azure OpenAI o3-mini",
                    "output": "suggested_disclosure"
                }
            ],
            ai_components_used=["Azure OpenAI o3-mini"],
            input_data={
                "company_name": company_metadata.get("company_name"),
                "business_nature": company_metadata.get("nature_of_business"),
                "operational_regions": company_metadata.get("operational_demographics"),
                "compliance_status": checklist_item.get("status"),
                "confidence_score": checklist_item.get("confidence"),
                "current_explanation": checklist_item.get("explanation")
            },
            processing_metadata={
                "company_context_tokens": ai_suggestion_log.get("company_context_tokens"),
                "suggestion_tokens": ai_suggestion_log.get("suggestion_completion_tokens"),
                "customization_level": "high",  # Based on company metadata usage
                "suggestion_quality_score": ai_suggestion_log.get("quality_score", 0.8)
            },
            timestamp=datetime.now()
        )
    
    def generate_complete_trace_report(self, document_id: str,
                                     checklist_item_id: str) -> Dict[str, Any]:
        """Generate a complete trace report for a specific result page item"""
        
        # This would fetch actual data from your database/storage
        # For now, providing the structure
        
        return {
            "document_id": document_id,
            "checklist_item_id": checklist_item_id,
            "generated_at": datetime.now().isoformat(),
            "traces": {
                trace_name: {
                    "attribute_name": trace.attribute_name,
                    "current_value": trace.current_value,
                    "data_source": {
                        "model": trace.data_source_model,
                        "field": trace.data_source_field
                    },
                    "processing_pipeline": trace.processing_stages,
                    "ai_components": trace.ai_components_used,
                    "input_data": trace.input_data,
                    "metadata": trace.processing_metadata,
                    "timestamp": trace.timestamp.isoformat()
                }
                for trace_name, trace in self.traces.items()
            },
            "summary": {
                "total_attributes_traced": len(self.traces),
                "ai_models_involved": list(set(
                    model for trace in self.traces.values() 
                    for model in trace.ai_components_used
                )),
                "processing_stages_count": sum(
                    len(trace.processing_stages) for trace in self.traces.values()
                ),
                "total_processing_time": sum(
                    trace.processing_metadata.get("processing_time_seconds", 0)
                    for trace in self.traces.values()
                )
            }
        }
    
    def export_trace_for_debugging(self, output_file: str):
        """Export trace information for debugging purposes"""
        
        trace_data = {
            "export_timestamp": datetime.now().isoformat(),
            "traces": self.traces,
            "debugging_info": {
                "models_used": list(set(
                    trace.data_source_model for trace in self.traces.values()
                )),
                "ai_components": list(set(
                    component for trace in self.traces.values()
                    for component in trace.ai_components_used
                )),
                "processing_stages": list(set(
                    stage["stage"] for trace in self.traces.values()
                    for stage in trace.processing_stages
                ))
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(trace_data, f, indent=2, default=str)
        
        logger.info(f"Trace data exported to {output_file}")


# Example usage function
def trace_result_page_attributes(document_id: str, checklist_item_id: str) -> Dict[str, Any]:
    """
    Example function showing how to trace all attributes for a result page item
    """
    
    tracer = ResultPageAttributeTracer()
    
    # Mock data - in real implementation, fetch from database
    checklist_item = {
        "id": checklist_item_id,
        "question": "If an entity reclassifies property at the date of initial application...",
        "reference": "IAS 40.84E(b)",
        "section": "IAS 40", 
        "confidence": 0.65,
        "explanation": "The document provides numerical disclosures on transfers...",
        "suggestion": "The entity should include a detailed reconciliation...",
        "status": "PARTIAL"
    }
    
    framework_data = {
        "framework_id": "IFRS",
        "selected_standards": ["IAS 40"],
        "total_questions": 150
    }
    
    company_metadata = {
        "company_name": "Aldar Properties PJSC",
        "nature_of_business": "Real estate development",
        "operational_demographics": "United Arab Emirates, Egypt"
    }
    
    # Generate traces for each attribute
    tracer.traces["question"] = tracer.trace_question_section(checklist_item, framework_data)
    tracer.traces["confidence"] = tracer.trace_confidence_score(checklist_item, {})
    tracer.traces["explanation"] = tracer.trace_explanation_text(checklist_item, {})
    tracer.traces["evidence"] = tracer.trace_evidence_table(checklist_item, [])
    tracer.traces["suggestion"] = tracer.trace_suggested_disclosure(checklist_item, company_metadata, {})
    
    return tracer.generate_complete_trace_report(document_id, checklist_item_id)


if __name__ == "__main__":
    # Example usage
    report = trace_result_page_attributes("doc_123", "item_456")
    print(json.dumps(report, indent=2, default=str))