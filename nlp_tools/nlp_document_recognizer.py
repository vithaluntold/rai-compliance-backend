#!/usr/bin/env python3
"""
NLP Tool 2: Document Structure Recognition System
Integrates enhanced financial statement parsing with existing infrastructure
"""

import logging
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys
import os

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from nlp_tools.enhanced_structure_parser import EnhancedFinancialStatementParser, DocumentSegment, TableStructure
from services.contextual_content_categorizer import ContextualContentCategorizer

logger = logging.getLogger(__name__)

class NLPDocumentStructureRecognizer:
    """
    Main NLP Tool 2 - Document Structure Recognition
    Combines enhanced parsing with existing categorization systems
    """
    
    def __init__(self):
        """Initialize NLP Tool 2 components"""
        logger.info("Initializing NLP Tool 2: Document Structure Recognition")
        
        try:
            # Initialize enhanced parser
            self.enhanced_parser = EnhancedFinancialStatementParser()
            logger.info("✅ Enhanced Financial Statement Parser initialized")
            
            # Initialize existing categorizer (fallback)
            self.categorizer = ContextualContentCategorizer()
            logger.info("✅ Contextual Content Categorizer initialized")
            
            # Initialize segment type mapping
            self._init_segment_type_mapping()
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize NLP Tool 2: {e}")
            raise

    def _init_segment_type_mapping(self):
        """Initialize mapping between segment types and processing categories"""
        
        self.segment_type_mapping = {
            'primary_statement': {
                'priority': 1,
                'processing_mode': 'structured_table_analysis',
                'content_categories': [
                    'financial_position', 'financial_performance', 
                    'cash_flows', 'equity_changes'
                ]
            },
            'notes': {
                'priority': 2,
                'processing_mode': 'narrative_analysis',
                'content_categories': [
                    'accounting_policies', 'significant_estimates',
                    'risk_disclosures', 'segment_reporting'
                ]
            },
            'policies': {
                'priority': 3,
                'processing_mode': 'policy_analysis',
                'content_categories': [
                    'measurement_basis', 'recognition_criteria',
                    'accounting_standards_applied'
                ]
            },
            'auditor_report': {
                'priority': 4,
                'processing_mode': 'opinion_analysis',
                'content_categories': [
                    'audit_opinion', 'key_audit_matters',
                    'going_concern', 'other_information'
                ]
            }
        }

    def process_document(self, pdf_path: str, document_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Main processing function for NLP Tool 2
        
        Args:
            pdf_path: Path to PDF financial document
            document_id: Optional document identifier
            
        Returns:
            Structured analysis with segments, tables, and metadata
        """
        logger.info(f"🚀 NLP Tool 2 Processing: {pdf_path}")
        
        if document_id is None:
            document_id = Path(pdf_path).stem
            
        processing_result = {
            'document_id': document_id,
            'pdf_path': pdf_path,
            'segments': [],
            'tables': [],
            'structure_analysis': {},
            'processing_metadata': {},
            'success': False
        }
        
        try:
            # Step 1: Enhanced Structure Parsing
            logger.info("📖 Step 1: Enhanced structure parsing")
            structure_data = self.enhanced_parser.parse_document_structure(pdf_path)
            
            if not structure_data.get('segments'):
                logger.warning("⚠️  No segments found, falling back to basic processing")
                return self._fallback_processing(pdf_path, document_id)
                
            logger.info(f"✅ Step 1 SUCCESS: Found {len(structure_data['segments'])} segments")
            
            # Step 2: Process Each Segment
            logger.info("🔍 Step 2: Processing individual segments")
            processed_segments = []
            
            for segment in structure_data['segments']:
                processed_segment = self._process_segment(segment, pdf_path)
                processed_segments.append(processed_segment)
                
            logger.info(f"✅ Step 2 SUCCESS: Processed {len(processed_segments)} segments")
            
            # Step 3: Enhance Table Analysis
            logger.info("📊 Step 3: Enhanced table analysis")
            enhanced_tables = self._enhance_table_analysis(structure_data['tables'])
            logger.info(f"✅ Step 3 SUCCESS: Analyzed {len(enhanced_tables)} tables")
            
            # Step 4: Cross-Reference Mapping
            logger.info("🔗 Step 4: Cross-reference mapping")
            cross_ref_analysis = self._analyze_cross_references(
                processed_segments, structure_data['cross_references']
            )
            logger.info("✅ Step 4 SUCCESS: Cross-references mapped")
            
            # Step 5: Generate Final Analysis
            logger.info("📋 Step 5: Generating final analysis")
            processing_result.update({
                'segments': processed_segments,
                'tables': enhanced_tables,
                'structure_analysis': {
                    'parsing_confidence': structure_data['parsing_confidence'],
                    'structure_metadata': structure_data['structure_metadata'],
                    'cross_references': cross_ref_analysis
                },
                'processing_metadata': self._generate_processing_metadata(
                    processed_segments, enhanced_tables, structure_data
                ),
                'success': True
            })
            
            logger.info(f"🎉 NLP Tool 2 SUCCESS: Document processed successfully")
            return processing_result
            
        except Exception as e:
            logger.error(f"❌ NLP Tool 2 ERROR: {e}")
            processing_result['error'] = str(e)
            return processing_result

    def _process_segment(self, segment: DocumentSegment, pdf_path: str) -> Dict[str, Any]:
        """Process individual document segment"""
        
        segment_type = segment.segment_type
        type_config = self.segment_type_mapping.get(segment_type, {})
        
        # Base segment information
        processed_segment = {
            'segment_id': segment.segment_id,
            'segment_type': segment_type,
            'title': segment.title,
            'content': segment.content,
            'page_range': segment.page_range,
            'confidence': segment.confidence,
            'metadata': segment.metadata,
            'processing_mode': type_config.get('processing_mode', 'default'),
            'priority': type_config.get('priority', 5)
        }
        
        # Enhanced processing based on segment type
        if segment_type == 'primary_statement':
            processed_segment.update(self._process_primary_statement(segment))
        elif segment_type == 'notes':
            processed_segment.update(self._process_notes_section(segment))
        elif segment_type == 'policies':
            processed_segment.update(self._process_policies_section(segment))
        elif segment_type == 'auditor_report':
            processed_segment.update(self._process_auditor_report(segment))
        
        # Use existing categorizer for additional insights
        try:
            categorization = self.categorizer.categorize_content(segment.content)
            processed_segment['categorization'] = categorization
        except Exception as e:
            logger.warning(f"Categorization failed for segment {segment.segment_id}: {e}")
            processed_segment['categorization'] = {'error': str(e)}
        
        return processed_segment

    def _process_primary_statement(self, segment: DocumentSegment) -> Dict[str, Any]:
        """Enhanced processing for primary financial statements"""
        
        content = segment.content.lower()
        
        # Detect statement subtype
        statement_subtype = 'unknown'
        if 'financial position' in content or 'balance sheet' in content:
            statement_subtype = 'statement_of_financial_position'
        elif 'profit' in content or 'income' in content or 'comprehensive income' in content:
            statement_subtype = 'statement_of_comprehensive_income'
        elif 'cash flow' in content:
            statement_subtype = 'statement_of_cash_flows'
        elif 'equity' in content or 'shareholders' in content:
            statement_subtype = 'statement_of_changes_in_equity'
        
        # Extract key financial figures (simplified)
        financial_figures = self._extract_financial_figures(segment.content)
        
        return {
            'statement_subtype': statement_subtype,
            'financial_figures': financial_figures,
            'table_count': len(segment.tables),
            'key_line_items': self._identify_key_line_items(segment.content, statement_subtype)
        }

    def _process_notes_section(self, segment: DocumentSegment) -> Dict[str, Any]:
        """Enhanced processing for notes sections"""
        
        content = segment.content
        
        # Identify note topics
        note_topics = self._identify_note_topics(content)
        
        # Extract accounting standard references
        standard_references = self._extract_standard_references(content)
        
        # Identify cross-references
        cross_refs = self._extract_note_cross_references(content)
        
        return {
            'note_topics': note_topics,
            'standard_references': standard_references,
            'cross_references': cross_refs,
            'note_type': self._classify_note_type(content)
        }

    def _process_policies_section(self, segment: DocumentSegment) -> Dict[str, Any]:
        """Enhanced processing for accounting policies"""
        
        content = segment.content
        
        # Extract policy topics
        policy_topics = self._extract_policy_topics(content)
        
        # Identify measurement bases
        measurement_bases = self._identify_measurement_bases(content)
        
        return {
            'policy_topics': policy_topics,
            'measurement_bases': measurement_bases,
            'policy_complexity': len(policy_topics),
            'standards_referenced': self._extract_standard_references(content)
        }

    def _process_auditor_report(self, segment: DocumentSegment) -> Dict[str, Any]:
        """Enhanced processing for auditor's report"""
        
        content = segment.content.lower()
        
        # Determine opinion type
        opinion_type = 'unqualified'
        if 'qualified opinion' in content:
            opinion_type = 'qualified'
        elif 'adverse opinion' in content:
            opinion_type = 'adverse'
        elif 'disclaimer' in content:
            opinion_type = 'disclaimer'
        
        # Extract key audit matters
        key_matters = self._extract_key_audit_matters(segment.content)
        
        return {
            'opinion_type': opinion_type,
            'key_audit_matters': key_matters,
            'going_concern_mentioned': 'going concern' in content,
            'material_uncertainty': 'material uncertainty' in content
        }

    def _extract_financial_figures(self, content: str) -> List[Dict[str, Any]]:
        """Extract financial figures from content"""
        import re
        
        # Pattern for monetary amounts
        money_pattern = r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:thousand|million|billion)?'
        
        figures = []
        lines = content.split('\n')
        
        for line in lines:
            if any(term in line.lower() for term in ['total', 'revenue', 'profit', 'loss', 'assets', 'liabilities']):
                amounts = re.findall(money_pattern, line)
                if amounts:
                    figures.append({
                        'line': line.strip(),
                        'amounts': amounts
                    })
        
        return figures[:10]  # Limit to first 10 figures

    def _identify_key_line_items(self, content: str, statement_type: str) -> List[str]:
        """Identify key line items based on statement type"""
        
        line_items = []
        content_lower = content.lower()
        
        if statement_type == 'statement_of_financial_position':
            key_terms = [
                'total assets', 'current assets', 'non-current assets',
                'total liabilities', 'current liabilities', 'non-current liabilities',
                'total equity', 'retained earnings', 'share capital'
            ]
        elif statement_type == 'statement_of_comprehensive_income':
            key_terms = [
                'revenue', 'cost of sales', 'gross profit',
                'operating expenses', 'operating profit', 'finance costs',
                'profit before tax', 'tax expense', 'profit after tax'
            ]
        elif statement_type == 'statement_of_cash_flows':
            key_terms = [
                'operating activities', 'investing activities', 'financing activities',
                'net cash flow', 'cash at beginning', 'cash at end'
            ]
        else:
            key_terms = []
        
        for term in key_terms:
            if term in content_lower:
                line_items.append(term)
        
        return line_items

    def _identify_note_topics(self, content: str) -> List[str]:
        """Identify topics covered in notes"""
        
        topics = []
        content_lower = content.lower()
        
        topic_keywords = {
            'accounting_policies': ['accounting policies', 'basis of preparation'],
            'revenue_recognition': ['revenue recognition', 'revenue from contracts'],
            'financial_instruments': ['financial instruments', 'fair value'],
            'property_plant_equipment': ['property plant equipment', 'depreciation'],
            'intangible_assets': ['intangible assets', 'goodwill', 'amortisation'],
            'inventory': ['inventory', 'cost of inventory'],
            'provisions': ['provisions', 'contingent liabilities'],
            'related_parties': ['related party', 'related parties'],
            'subsequent_events': ['subsequent events', 'events after'],
            'segment_reporting': ['operating segments', 'segment reporting']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                topics.append(topic)
        
        return topics

    def _extract_standard_references(self, content: str) -> List[str]:
        """Extract accounting standard references"""
        import re
        
        # Patterns for IAS/IFRS references
        patterns = [
            r'IAS\s+\d+',
            r'IFRS\s+\d+',
            r'IFRIC\s+\d+',
            r'SIC\s+\d+'
        ]
        
        standards = []
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            standards.extend(matches)
        
        return list(set(standards))  # Remove duplicates

    def _extract_note_cross_references(self, content: str) -> List[str]:
        """Extract cross-references to other notes"""
        import re
        
        # Pattern for note references
        note_pattern = r'note\s+(\d+)'
        matches = re.findall(note_pattern, content, re.IGNORECASE)
        
        return [f"note_{num}" for num in set(matches)]

    def _classify_note_type(self, content: str) -> str:
        """Classify the type of note"""
        
        content_lower = content.lower()
        
        if any(term in content_lower for term in ['accounting policies', 'basis of preparation']):
            return 'accounting_policies'
        elif any(term in content_lower for term in ['property', 'plant', 'equipment']):
            return 'asset_disclosure'
        elif any(term in content_lower for term in ['revenue', 'income']):
            return 'revenue_disclosure'
        elif any(term in content_lower for term in ['financial instruments', 'fair value']):
            return 'financial_instruments'
        else:
            return 'general_disclosure'

    def _extract_policy_topics(self, content: str) -> List[str]:
        """Extract accounting policy topics"""
        
        topics = []
        content_lower = content.lower()
        
        policy_topics = [
            'revenue recognition', 'inventory valuation', 'depreciation',
            'impairment', 'foreign currency', 'financial instruments',
            'leases', 'provisions', 'employee benefits', 'taxation'
        ]
        
        for topic in policy_topics:
            if topic in content_lower:
                topics.append(topic)
        
        return topics

    def _identify_measurement_bases(self, content: str) -> List[str]:
        """Identify measurement bases mentioned"""
        
        bases = []
        content_lower = content.lower()
        
        measurement_terms = [
            'historical cost', 'fair value', 'amortised cost',
            'present value', 'current cost', 'realisable value'
        ]
        
        for basis in measurement_terms:
            if basis in content_lower:
                bases.append(basis)
        
        return bases

    def _extract_key_audit_matters(self, content: str) -> List[str]:
        """Extract key audit matters from auditor's report"""
        
        matters = []
        lines = content.split('\n')
        
        in_kam_section = False
        for line in lines:
            line_lower = line.lower()
            
            if 'key audit matter' in line_lower:
                in_kam_section = True
                continue
                
            if in_kam_section:
                if line.strip() and len(line.strip()) > 20:
                    matters.append(line.strip())
                    
                # Stop at next major section
                if any(term in line_lower for term in ['opinion', 'responsibility', 'other information']):
                    break
        
        return matters[:5]  # Limit to first 5 matters

    def _enhance_table_analysis(self, tables: List[TableStructure]) -> List[Dict[str, Any]]:
        """Enhance table analysis with additional metadata"""
        
        enhanced_tables = []
        
        for table in tables:
            enhanced_table = {
                'table_id': table.table_id,
                'title': table.title,
                'headers': table.headers,
                'row_count': len(table.rows),
                'column_count': len(table.headers) if table.headers else 0,
                'page_num': table.page_num,
                'table_type': table.table_type,
                'confidence': table.confidence,
                'financial_indicators': self._analyze_table_financial_indicators(table),
                'data_quality': self._assess_table_data_quality(table)
            }
            
            enhanced_tables.append(enhanced_table)
        
        return enhanced_tables

    def _analyze_table_financial_indicators(self, table: TableStructure) -> Dict[str, Any]:
        """Analyze financial indicators in table"""
        
        indicators = {
            'has_monetary_values': False,
            'has_percentages': False,
            'has_dates': False,
            'row_patterns': []
        }
        
        # Check headers for financial indicators
        if table.headers:
            header_text = ' '.join(table.headers).lower()
            indicators['has_monetary_values'] = any(term in header_text for term in ['$', '€', '£', 'million', 'thousand'])
            indicators['has_percentages'] = '%' in header_text
            indicators['has_dates'] = any(term in header_text for term in ['2023', '2024', '2025', 'year', 'period'])
        
        return indicators

    def _assess_table_data_quality(self, table: TableStructure) -> Dict[str, Any]:
        """Assess data quality of extracted table"""
        
        quality = {
            'completeness': 0.0,
            'consistency': 0.0,
            'structure_score': 0.0
        }
        
        if table.rows:
            # Completeness: ratio of non-empty cells
            total_cells = len(table.rows) * len(table.headers) if table.headers else 0
            if total_cells > 0:
                filled_cells = sum(1 for row in table.rows for cell in row if cell and cell.strip())
                quality['completeness'] = filled_cells / total_cells
            
            # Structure score based on consistent column count
            if table.headers:
                expected_cols = len(table.headers)
                consistent_rows = sum(1 for row in table.rows if len(row) == expected_cols)
                quality['structure_score'] = consistent_rows / len(table.rows)
        
        return quality

    def _analyze_cross_references(self, processed_segments: List[Dict[str, Any]], 
                                cross_refs: Dict[str, List[str]]) -> Dict[str, Any]:
        """Analyze cross-references between segments"""
        
        analysis = {
            'total_references': len(cross_refs),
            'reference_types': {},
            'interconnected_sections': []
        }
        
        # Analyze reference types
        for source, targets in cross_refs.items():
            for target in targets:
                if target.startswith('note_'):
                    analysis['reference_types']['note_references'] = analysis['reference_types'].get('note_references', 0) + 1
                else:
                    analysis['reference_types']['other_references'] = analysis['reference_types'].get('other_references', 0) + 1
        
        return analysis

    def _generate_processing_metadata(self, segments: List[Dict[str, Any]], 
                                    tables: List[Dict[str, Any]], 
                                    structure_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive processing metadata"""
        
        return {
            'processing_timestamp': str(Path(__file__).stat().st_mtime),
            'nlp_tool_version': '2.0.0',
            'total_segments': len(segments),
            'total_tables': len(tables),
            'segment_types': {seg['segment_type']: len([s for s in segments if s['segment_type'] == seg['segment_type']]) for seg in segments},
            'average_confidence': sum(seg['confidence'] for seg in segments) / len(segments) if segments else 0.0,
            'structure_confidence': structure_data.get('parsing_confidence', 0.0),
            'processing_features': [
                'enhanced_structure_parsing',
                'table_analysis',
                'cross_reference_mapping',
                'content_categorization',
                'financial_indicators_extraction'
            ]
        }

    def _fallback_processing(self, pdf_path: str, document_id: str) -> Dict[str, Any]:
        """Fallback processing when enhanced parsing fails"""
        
        logger.info("📄 Fallback processing: Using basic document analysis")
        
        try:
            # Use existing categorizer as fallback
            fallback_result = self.categorizer.categorize_document(pdf_path)
            
            return {
                'document_id': document_id,
                'pdf_path': pdf_path,
                'segments': fallback_result.get('segments', []),
                'tables': [],
                'structure_analysis': fallback_result.get('analysis', {}),
                'processing_metadata': {
                    'processing_mode': 'fallback',
                    'fallback_reason': 'enhanced_parsing_failed'
                },
                'success': True
            }
            
        except Exception as e:
            logger.error(f"❌ Fallback processing also failed: {e}")
            return {
                'document_id': document_id,
                'pdf_path': pdf_path,
                'segments': [],
                'tables': [],
                'structure_analysis': {},
                'processing_metadata': {'error': str(e)},
                'success': False
            }

def main():
    """Test NLP Tool 2"""
    
    # Initialize NLP Tool 2
    nlp_tool = NLPDocumentStructureRecognizer()
    
    # Test with sample document
    test_pdf = "sample_financial_report.pdf"
    if Path(test_pdf).exists():
        result = nlp_tool.process_document(test_pdf)
        
        print(f"NLP Tool 2 Processing Results:")
        print(f"- Success: {result['success']}")
        print(f"- Segments found: {len(result['segments'])}")
        print(f"- Tables found: {len(result['tables'])}")
        
        if result['success']:
            print("\nSegment Types:")
            for segment in result['segments']:
                print(f"  - {segment['segment_type']}: {segment['title']} (confidence: {segment['confidence']:.2f})")
        
        # Save results
        output_path = f"nlp_tool_2_results_{Path(test_pdf).stem}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_path}")
    else:
        print(f"Test file not found: {test_pdf}")

if __name__ == "__main__":
    main()