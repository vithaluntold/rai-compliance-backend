#!/usr/bin/env python3
"""
NLP Tools Integration Module
Combines Tool 2 (Enhanced Structure Parser) with Tool 3 (AI Content Classifier)
for complete document processing pipeline
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# Import our NLP tools
from nlp_tools.enhanced_structure_parser import EnhancedFinancialStatementParser, DocumentSegment
from nlp_tools.ai_content_classifier import AIContentClassificationEngine, ContentSegment, ClassificationResult

logger = logging.getLogger(__name__)

@dataclass
class ProcessedDocument:
    """Complete processed document with structure and classification"""
    document_path: str
    total_pages: int
    structured_segments: List[DocumentSegment]
    classified_segments: List[ContentSegment]
    mega_chunks_by_standard: Dict[str, Dict[str, Any]]
    processing_metadata: Dict[str, Any]
    success: bool = True
    error: Optional[str] = None

class NLPDocumentProcessor:
    """Complete NLP processing pipeline combining structure parsing and AI classification"""
    
    def __init__(self):
        """Initialize the complete NLP document processing pipeline"""
        self.structure_parser = EnhancedFinancialStatementParser()
        self.content_classifier = AIContentClassificationEngine()
        
        logger.info("NLP Document Processor initialized with Tool 2 + Tool 3")
        
    def process_document(self, pdf_path: str) -> ProcessedDocument:
        """
        Process a complete financial statement document through the full NLP pipeline
        
        Args:
            pdf_path: Path to the PDF financial statement document
            
        Returns:
            ProcessedDocument with complete structure and classification results
        """
        
        try:
            # Step 1: Enhanced Structure Parsing (Tool 2)
            logger.info(f"Starting document structure parsing for: {pdf_path}")
            
            structured_result = self.structure_parser.parse_document_structure(pdf_path)
            
            if not structured_result.get('success', False):
                return ProcessedDocument(
                    document_path=pdf_path,
                    total_pages=0,
                    structured_segments=[],
                    classified_segments=[],
                    mega_chunks_by_standard={},
                    processing_metadata={},
                    success=False,
                    error=f"Structure parsing failed: {structured_result.get('error', 'Unknown error')}"
                )
                
            # Step 2: Prepare segments for classification
            segments_for_classification = self._prepare_segments_for_classification(
                structured_result.get('segments', [])
            )
            
            # Step 3: AI Content Classification (Tool 3)
            logger.info("Starting AI content classification")
            
            classification_result = self.content_classifier.classify_document_segments(
                segments=segments_for_classification,
                source_document=os.path.basename(pdf_path)
            )
            
            if not classification_result.success:
                return ProcessedDocument(
                    document_path=pdf_path,
                    total_pages=structured_result.get('total_pages', 0),
                    structured_segments=structured_result.get('segments', []),
                    classified_segments=[],
                    mega_chunks_by_standard={},
                    processing_metadata=structured_result.get('metadata', {}),
                    success=False,
                    error=f"Content classification failed: {classification_result.error}"
                )
                
            # Step 4: Create mega-chunks by accounting standard
            logger.info("Creating mega-chunks by accounting standard")
            
            mega_chunks = self.content_classifier.create_mega_chunk_by_standard(
                classification_result.segments
            )
            
            # Step 5: Prepare processing metadata
            processing_metadata = {
                **structured_result.get('metadata', {}),
                'classification_stats': {
                    'total_classified_segments': classification_result.total_segments,
                    'standards_identified': list(mega_chunks.keys()),
                    'avg_confidence_score': self._calculate_average_confidence(classification_result.segments)
                },
                'pipeline_version': 'Tool2+Tool3',
                'processing_timestamp': self._get_current_timestamp()
            }
            
            return ProcessedDocument(
                document_path=pdf_path,
                total_pages=structured_result.get('total_pages', 0),
                structured_segments=structured_result.get('segments', []),
                classified_segments=classification_result.segments,
                mega_chunks_by_standard=mega_chunks,
                processing_metadata=processing_metadata,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            return ProcessedDocument(
                document_path=pdf_path,
                total_pages=0,
                structured_segments=[],
                classified_segments=[],
                mega_chunks_by_standard={},
                processing_metadata={},
                success=False,
                error=f"Processing pipeline failed: {str(e)}"
            )
            
    def _prepare_segments_for_classification(self, structured_segments: List[DocumentSegment]) -> List[Dict[str, Any]]:
        """Convert structured segments to format expected by content classifier"""
        
        classification_segments = []
        
        for segment in structured_segments:
            # Handle both DocumentSegment objects and dict formats
            if isinstance(segment, dict):
                classification_segment = {
                    'content': segment.get('content', ''),
                    'segment_type': segment.get('segment_type', 'unknown'),
                    'page_num': segment.get('page_num', 0)
                }
            else:
                # DocumentSegment object
                page_num = segment.page_range[0] if hasattr(segment, 'page_range') and segment.page_range else 0
                classification_segment = {
                    'content': segment.content,
                    'segment_type': segment.segment_type,
                    'page_num': page_num
                }
            classification_segments.append(classification_segment)
            
        return classification_segments
        
    def _calculate_average_confidence(self, segments: List[ContentSegment]) -> float:
        """Calculate average confidence score across all segments"""
        
        if not segments:
            return 0.0
            
        total_confidence = sum(segment.confidence_score for segment in segments)
        return total_confidence / len(segments)
        
    def _get_current_timestamp(self) -> str:
        """Get current timestamp for metadata"""
        from datetime import datetime
        return datetime.now().isoformat()
        
    def generate_processing_report(self, processed_doc: ProcessedDocument) -> Dict[str, Any]:
        """Generate a comprehensive processing report"""
        
        if not processed_doc.success:
            return {
                'success': False,
                'error': processed_doc.error,
                'document_path': processed_doc.document_path
            }
            
        # Analyze structure parsing results
        structure_stats = {
            'total_pages': processed_doc.total_pages,
            'total_segments': len(processed_doc.structured_segments),
            'segment_types': self._count_segment_types(processed_doc.structured_segments)
        }
        
        # Analyze classification results
        classification_stats = {
            'total_classified': len(processed_doc.classified_segments),
            'standards_identified': len(processed_doc.mega_chunks_by_standard),
            'standards_breakdown': {
                standard: data['total_segments'] 
                for standard, data in processed_doc.mega_chunks_by_standard.items()
            },
            'average_confidence': processed_doc.processing_metadata.get('classification_stats', {}).get('avg_confidence_score', 0.0)
        }
        
        # Create comprehensive report
        report = {
            'success': True,
            'document_path': processed_doc.document_path,
            'processing_pipeline': 'Enhanced Structure Parser (Tool 2) + AI Content Classifier (Tool 3)',
            'structure_analysis': structure_stats,
            'content_classification': classification_stats,
            'mega_chunks_summary': {
                standard: {
                    'sub_chunks': data['total_segments'],
                    'confidence': data.get('confidence_score', 0.0)
                }
                for standard, data in processed_doc.mega_chunks_by_standard.items()
            },
            'processing_metadata': processed_doc.processing_metadata
        }
        
        return report
        
    def _count_segment_types(self, segments: List[DocumentSegment]) -> Dict[str, int]:
        """Count segments by type"""
        
        type_counts = {}
        for segment in segments:
            segment_type = segment.segment_type
            type_counts[segment_type] = type_counts.get(segment_type, 0) + 1
            
        return type_counts
        
    def export_mega_chunks_for_retrieval(self, processed_doc: ProcessedDocument, 
                                       output_path: str) -> bool:
        """Export mega-chunks in format suitable for content-question matching"""
        
        try:
            # Prepare mega-chunks for export
            export_data = {
                'document_info': {
                    'source_document': processed_doc.document_path,
                    'total_pages': processed_doc.total_pages,
                    'processing_date': processed_doc.processing_metadata.get('processing_timestamp', ''),
                    'pipeline_version': 'Tool2+Tool3'
                },
                'mega_chunks': {},
                'metadata': {
                    'total_standards': len(processed_doc.mega_chunks_by_standard),
                    'total_sub_chunks': sum(
                        data['total_segments'] 
                        for data in processed_doc.mega_chunks_by_standard.values()
                    )
                }
            }
            
            # Process each accounting standard mega-chunk
            for standard, mega_chunk_data in processed_doc.mega_chunks_by_standard.items():
                
                export_mega_chunk = {
                    'accounting_standard': standard,
                    'total_sub_chunks': mega_chunk_data['total_segments'],
                    'confidence_score': mega_chunk_data.get('confidence_score', 0.0),
                    'combined_5d_tags': mega_chunk_data.get('combined_tags', {}),
                    'sub_chunks': []
                }
                
                # Add each sub-chunk with full 5D tags
                for sub_chunk in mega_chunk_data.get('sub_chunks', []):
                    export_sub_chunk = {
                        'content_text': sub_chunk['content_text'],
                        'segment_type': sub_chunk['segment_type'],
                        'paragraph_hint': sub_chunk.get('paragraph_hint', ''),
                        'page_number': sub_chunk.get('page_number', 0),
                        'confidence_score': sub_chunk.get('confidence_score', 0.0),
                        '5d_classification_tags': sub_chunk.get('classification_tags', {})
                    }
                    export_mega_chunk['sub_chunks'].append(export_sub_chunk)
                    
                export_data['mega_chunks'][standard] = export_mega_chunk
                
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Mega-chunks exported to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export mega-chunks: {e}")
            return False
            
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get status of the complete NLP pipeline"""
        
        return {
            'pipeline_name': 'NLP Document Processor (Tool 2 + Tool 3)',
            'components': {
                'enhanced_structure_parser': {
                    'status': 'ready',
                    'description': 'Enhanced NLP Document Structure Recognition (Tool 2)'
                },
                'ai_content_classifier': {
                    'status': 'ready', 
                    'description': 'AI Content Classification Engine (Tool 3)'
                }
            },
            'capabilities': [
                'PDF structure parsing and segmentation',
                'Accounting standard detection',
                '5D tag classification for content',
                'Mega-chunk creation by standard',
                'Cross-reference mapping',
                'Table and narrative extraction'
            ],
            'output_formats': [
                'Structured document segments',
                'Classified content with 5D tags',
                'Mega-chunks by accounting standard',
                'Processing reports and metadata'
            ]
        }