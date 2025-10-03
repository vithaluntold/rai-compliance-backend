#!/usr/bin/env python3
"""
NLP Tool 2 Integration Module
Integrates enhanced document structure recognition with existing chunking system
"""

import logging
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent))

from nlp_tools.nlp_document_recognizer import NLPDocumentStructureRecognizer
# Other imports - DocumentChunker has been replaced by enhanced structure parser
from services.ai import AIService

logger = logging.getLogger(__name__)

class NLPEnhancedDocumentProcessor:
    """
    Integration wrapper for NLP Tool 2 with existing document processing
    Provides backward compatibility while enabling enhanced structure recognition
    """
    
    def __init__(self, enable_enhanced_nlp: bool = True):
        """
        Initialize enhanced document processor
        
        Args:
            enable_enhanced_nlp: Enable enhanced NLP processing (Tool 2)
        """
        self.enable_enhanced_nlp = enable_enhanced_nlp
        
        # Document chunking is now handled by enhanced structure parser
        # DocumentChunker has been deprecated and replaced
        logger.info("✅ Using enhanced structure parser for document processing")
        
        # Initialize enhanced NLP Tool 2 (optional)
        if enable_enhanced_nlp:
            try:
                self.nlp_tool = NLPDocumentStructureRecognizer()
                logger.info("✅ NLP Tool 2 initialized - Enhanced processing available")
            except Exception as e:
                logger.warning(f"⚠️ NLP Tool 2 initialization failed: {e}. Using standard processing only.")
                self.nlp_tool = None
                self.enable_enhanced_nlp = False
        else:
            self.nlp_tool = None
            logger.info("📄 Enhanced NLP disabled - Using standard processing only")

    def process_document_enhanced(self, pdf_path: str, document_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Enhanced document processing with NLP Tool 2
        
        Args:
            pdf_path: Path to PDF document
            document_id: Optional document identifier
            
        Returns:
            Enhanced processing results with structure analysis
        """
        logger.info(f"🚀 Enhanced Document Processing: {pdf_path}")
        
        if document_id is None:
            document_id = Path(pdf_path).stem
        
        processing_result = {
            'document_id': document_id,
            'pdf_path': pdf_path,
            'processing_mode': 'enhanced' if self.enable_enhanced_nlp else 'standard',
            'enhanced_analysis': {},
            'standard_chunks': [],
            'unified_output': [],
            'success': False
        }
        
        try:
            # Step 1: Enhanced NLP Processing (if available)
            if self.enable_enhanced_nlp and self.nlp_tool:
                logger.info("🔍 Running NLP Tool 2 - Enhanced Structure Recognition")
                
                try:
                    enhanced_result = self.nlp_tool.process_document(pdf_path, document_id)
                    
                    if enhanced_result['success']:
                        processing_result['enhanced_analysis'] = enhanced_result
                        logger.info(f"✅ Enhanced NLP SUCCESS: {len(enhanced_result['segments'])} segments found")
                    else:
                        logger.warning("⚠️ Enhanced NLP failed, falling back to standard processing")
                        self.enable_enhanced_nlp = False
                        
                except Exception as e:
                    logger.error(f"❌ Enhanced NLP error: {e}, falling back to standard processing")
                    self.enable_enhanced_nlp = False
            
            # Step 2: Standard Processing (deprecated - skip if enhanced processing available)
            standard_chunks = []
            if not processing_result['enhanced_analysis']:
                logger.warning("⚠️ No enhanced analysis available and standard chunker deprecated")
                standard_chunks = []
            processing_result['standard_chunks'] = standard_chunks
            
            # Step 3: Unified Output Generation
            logger.info("🔄 Generating unified output")
            
            if processing_result['enhanced_analysis']:
                # Combine enhanced analysis with standard chunks
                unified_output = self._create_unified_output(
                    processing_result['enhanced_analysis'], 
                    standard_chunks
                )
            else:
                # Use standard chunks only
                unified_output = self._convert_standard_to_unified(standard_chunks)
            
            processing_result['unified_output'] = unified_output
            processing_result['success'] = True
            
            logger.info(f"🎉 Document Processing Complete: {len(unified_output)} unified segments")
            
        except Exception as e:
            logger.error(f"❌ Document processing failed: {e}")
            processing_result['error'] = str(e)
            
        return processing_result

    def _create_unified_output(self, enhanced_analysis: Dict[str, Any], 
                             standard_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create unified output combining enhanced analysis with standard chunks
        """
        unified_segments = []
        
        # Process enhanced segments
        for segment in enhanced_analysis.get('segments', []):
            unified_segment = {
                'segment_id': segment['segment_id'],
                'type': 'enhanced_segment',
                'segment_type': segment['segment_type'],
                'title': segment['title'],
                'content': segment['content'],
                'page_range': segment['page_range'],
                'confidence': segment['confidence'],
                'processing_mode': segment.get('processing_mode', 'enhanced'),
                'priority': segment.get('priority', 5),
                'metadata': {
                    **segment.get('metadata', {}),
                    'source': 'nlp_tool_2',
                    'categorization': segment.get('categorization', {}),
                    'enhanced_features': self._extract_enhanced_features(segment)
                },
                # Standard chunk compatibility
                'chunk_index': len(unified_segments),
                'text': segment['content'],
                'length': len(segment['content'])
            }
            
            unified_segments.append(unified_segment)
        
        # Add enhanced table analysis
        for table in enhanced_analysis.get('tables', []):
            table_segment = {
                'segment_id': f"table_{table['table_id']}",
                'type': 'enhanced_table',
                'segment_type': 'table',
                'title': table['title'],
                'content': self._table_to_text(table),
                'page_range': (table['page_num'], table['page_num']),
                'confidence': table['confidence'],
                'processing_mode': 'table_analysis',
                'priority': 2,
                'metadata': {
                    'source': 'nlp_tool_2',
                    'table_type': table['table_type'],
                    'row_count': table['row_count'],
                    'column_count': table['column_count'],
                    'data_quality': table.get('data_quality', {}),
                    'financial_indicators': table.get('financial_indicators', {})
                },
                'chunk_index': len(unified_segments),
                'text': self._table_to_text(table),
                'length': len(self._table_to_text(table))
            }
            
            unified_segments.append(table_segment)
        
        # Merge with standard chunks for missing content
        unified_segments.extend(self._merge_missing_content(unified_segments, standard_chunks))
        
        # Sort by priority and page order
        unified_segments.sort(key=lambda x: (x['priority'], x['page_range'][0]))
        
        return unified_segments

    def _extract_enhanced_features(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        """Extract enhanced features from processed segment"""
        
        features = {}
        
        if segment['segment_type'] == 'primary_statement':
            features = {
                'statement_subtype': segment.get('statement_subtype'),
                'financial_figures': segment.get('financial_figures', []),
                'key_line_items': segment.get('key_line_items', [])
            }
        elif segment['segment_type'] == 'notes':
            features = {
                'note_topics': segment.get('note_topics', []),
                'standard_references': segment.get('standard_references', []),
                'note_type': segment.get('note_type')
            }
        elif segment['segment_type'] == 'policies':
            features = {
                'policy_topics': segment.get('policy_topics', []),
                'measurement_bases': segment.get('measurement_bases', [])
            }
        elif segment['segment_type'] == 'auditor_report':
            features = {
                'opinion_type': segment.get('opinion_type'),
                'key_audit_matters': segment.get('key_audit_matters', [])
            }
        
        return features

    def _table_to_text(self, table: Dict[str, Any]) -> str:
        """Convert table data to text representation"""
        
        text_parts = []
        
        # Add title
        if table.get('title'):
            text_parts.append(table['title'])
            text_parts.append('')  # Empty line
        
        # Add headers
        if table.get('headers'):
            text_parts.append(' | '.join(str(h) for h in table['headers']))
            text_parts.append('-' * 50)  # Separator
        
        # Add sample rows (limited)
        if table.get('row_count', 0) > 0:
            text_parts.append(f"[Table with {table['row_count']} rows and {table['column_count']} columns]")
            text_parts.append(f"Table Type: {table.get('table_type', 'unknown')}")
        
        return '\n'.join(text_parts)

    def _convert_standard_to_unified(self, standard_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert standard chunks to unified format"""
        
        unified_segments = []
        
        for chunk in standard_chunks:
            unified_segment = {
                'segment_id': f"standard_chunk_{chunk.get('chunk_index', 0)}",
                'type': 'standard_chunk',
                'segment_type': chunk.get('chunk_type', 'content'),
                'title': f"Document Section {chunk.get('page', 0) + 1}",
                'content': chunk.get('text', ''),
                'page_range': (chunk.get('page', 0), chunk.get('page', 0)),
                'confidence': 0.8,  # Default confidence for standard chunks
                'processing_mode': 'standard',
                'priority': 6,  # Lower priority than enhanced segments
                'metadata': {
                    'source': 'standard_chunker',
                    'chunk_type': chunk.get('chunk_type', 'content'),
                    'page': chunk.get('page', 0)
                },
                'chunk_index': chunk.get('chunk_index', 0),
                'text': chunk.get('text', ''),
                'length': chunk.get('length', 0)
            }
            
            unified_segments.append(unified_segment)
        
        return unified_segments

    def _merge_missing_content(self, enhanced_segments: List[Dict[str, Any]], 
                             standard_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge any missing content from standard chunks"""
        
        # Get page ranges covered by enhanced segments
        covered_pages = set()
        for segment in enhanced_segments:
            start_page, end_page = segment['page_range']
            covered_pages.update(range(start_page, end_page + 1))
        
        # Find uncovered content in standard chunks
        missing_segments = []
        for chunk in standard_chunks:
            chunk_page = chunk.get('page', 0)
            
            if chunk_page not in covered_pages:
                # This content is missing in enhanced analysis
                missing_segment = {
                    'segment_id': f"missing_content_{chunk.get('chunk_index', 0)}",
                    'type': 'missing_content',
                    'segment_type': 'supplementary',
                    'title': f"Additional Content (Page {chunk_page + 1})",
                    'content': chunk.get('text', ''),
                    'page_range': (chunk_page, chunk_page),
                    'confidence': 0.6,
                    'processing_mode': 'supplementary',
                    'priority': 7,  # Lowest priority
                    'metadata': {
                        'source': 'missing_content_recovery',
                        'original_chunk_index': chunk.get('chunk_index', 0)
                    },
                    'chunk_index': chunk.get('chunk_index', 0),
                    'text': chunk.get('text', ''),
                    'length': chunk.get('length', 0)
                }
                
                missing_segments.append(missing_segment)
        
        return missing_segments

    def get_processing_statistics(self, processing_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate processing statistics"""
        
        stats = {
            'processing_mode': processing_result.get('processing_mode', 'unknown'),
            'success': processing_result.get('success', False),
            'total_unified_segments': len(processing_result.get('unified_output', [])),
            'enhanced_segments': 0,
            'standard_chunks': len(processing_result.get('standard_chunks', [])),
            'tables_found': 0,
            'segment_types': {}
        }
        
        # Analyze unified output
        for segment in processing_result.get('unified_output', []):
            segment_type = segment.get('segment_type', 'unknown')
            stats['segment_types'][segment_type] = stats['segment_types'].get(segment_type, 0) + 1
            
            if segment.get('type') == 'enhanced_segment':
                stats['enhanced_segments'] += 1
            elif segment.get('type') == 'enhanced_table':
                stats['tables_found'] += 1
        
        # Enhanced analysis statistics
        enhanced_analysis = processing_result.get('enhanced_analysis', {})
        if enhanced_analysis:
            stats['enhanced_confidence'] = enhanced_analysis.get('structure_analysis', {}).get('parsing_confidence', 0.0)
            stats['cross_references'] = len(enhanced_analysis.get('structure_analysis', {}).get('cross_references', {}))
        
        return stats

# Convenience function for backward compatibility
def process_document_with_nlp_tool2(pdf_path: str, document_id: Optional[str] = None, 
                                   enable_enhanced: bool = True) -> Dict[str, Any]:
    """
    Convenience function to process document with NLP Tool 2
    
    Args:
        pdf_path: Path to PDF document
        document_id: Optional document identifier  
        enable_enhanced: Enable enhanced NLP processing
        
    Returns:
        Processing results with unified output format
    """
    processor = NLPEnhancedDocumentProcessor(enable_enhanced_nlp=enable_enhanced)
    return processor.process_document_enhanced(pdf_path, document_id)

def main():
    """Test the NLP Tool 2 integration"""
    
    # Test with enhanced processing
    processor = NLPEnhancedDocumentProcessor(enable_enhanced_nlp=True)
    
    test_pdf = "sample_financial_report.pdf"
    if Path(test_pdf).exists():
        result = processor.process_document_enhanced(test_pdf)
        
        # Print results
        print("NLP Tool 2 Integration Results:")
        print(f"- Success: {result['success']}")
        print(f"- Processing Mode: {result['processing_mode']}")
        
        # Get statistics
        stats = processor.get_processing_statistics(result)
        print(f"- Enhanced Segments: {stats['enhanced_segments']}")
        print(f"- Standard Chunks: {stats['standard_chunks']}")
        print(f"- Tables Found: {stats['tables_found']}")
        print(f"- Total Unified Segments: {stats['total_unified_segments']}")
        
        # Save results
        output_file = f"nlp_tool2_integration_results_{Path(test_pdf).stem}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str, ensure_ascii=False)
        
        print(f"Results saved to: {output_file}")
        
    else:
        print(f"Test file not found: {test_pdf}")

if __name__ == "__main__":
    main()