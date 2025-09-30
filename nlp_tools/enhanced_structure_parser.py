#!/usr/bin/env python3
"""
Enhanced NLP Document Structure Recognition (Tool 2)
Advanced PDF parser for financial statement segmentation and structural analysis
"""

import logging
import re
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from pathlib import Path
import fitz  # PyMuPDF
from collections import defaultdict

# Optional import for advanced table extraction
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    pdfplumber = None

logger = logging.getLogger(__name__)

@dataclass
class DocumentSegment:
    """Represents a structured segment of financial document"""
    segment_id: str
    segment_type: str  # primary_statement, notes, policies, auditor_report, other
    title: str
    content: str
    page_range: Tuple[int, int]
    tables: List[Dict[str, Any]]
    cross_references: List[str]
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class TableStructure:
    """Represents extracted table structure"""
    table_id: str
    title: str
    headers: List[str]
    rows: List[List[str]]
    page_num: int
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    table_type: str  # financial_position, profit_loss, cash_flows, notes_table
    confidence: float

class EnhancedFinancialStatementParser:
    """Enhanced parser for financial statement structure recognition"""
    
    def __init__(self):
        self.primary_statement_patterns = {
            'statement_of_financial_position': [
                r'statement\s+of\s+financial\s+position',
                r'balance\s+sheet',
                r'consolidated\s+statement\s+of\s+financial\s+position',
                r'statement\s+of\s+assets\s+and\s+liabilities'
            ],
            'statement_of_profit_loss': [
                r'statement\s+of\s+profit\s+or\s+loss',
                r'income\s+statement',
                r'statement\s+of\s+comprehensive\s+income',
                r'profit\s+and\s+loss\s+statement',
                r'statement\s+of\s+operations'
            ],
            'statement_of_cash_flows': [
                r'statement\s+of\s+cash\s+flows',
                r'cash\s+flow\s+statement',
                r'consolidated\s+statement\s+of\s+cash\s+flows'
            ],
            'statement_of_changes_in_equity': [
                r'statement\s+of\s+changes\s+in\s+equity',
                r'statement\s+of\s+stockholders[\'\"]?s?\s+equity',
                r'changes\s+in\s+equity',
                r'statement\s+of\s+shareholders[\'\"]?s?\s+equity'
            ]
        }
        
        self.notes_patterns = [
            r'notes?\s+to\s+(?:the\s+)?(?:consolidated\s+)?financial\s+statements?',
            r'notes?\s+to\s+(?:the\s+)?accounts?',
            r'explanatory\s+notes?',
            r'note\s+\d+[:.]\s*',
            r'^\d+\.\s+[A-Z]'  # Numbered sections like "1. Accounting Policies"
        ]
        
        self.accounting_policies_patterns = [
            r'significant\s+accounting\s+policies',
            r'accounting\s+policies',
            r'summary\s+of\s+significant\s+accounting\s+policies',
            r'basis\s+of\s+preparation',
            r'accounting\s+standards'
        ]
        
        self.auditor_report_patterns = [
            r'independent\s+auditor[\'"]?s?\s+report',
            r'auditor[\'"]?s?\s+report',
            r'report\s+of\s+(?:the\s+)?independent\s+auditors?',
            r'opinion\s+of\s+(?:the\s+)?auditors?'
        ]
        
        # Load enhanced detection rules
        self._initialize_detection_rules()
        
    def _initialize_detection_rules(self):
        """Initialize enhanced detection rules for financial statements"""
        
        # Financial position indicators
        self.balance_sheet_indicators = {
            'assets': [
                'current assets', 'non-current assets', 'total assets',
                'property plant equipment', 'intangible assets', 'investments',
                'cash and cash equivalents', 'trade receivables', 'inventory'
            ],
            'liabilities': [
                'current liabilities', 'non-current liabilities', 'total liabilities',
                'trade payables', 'borrowings', 'provisions', 'deferred tax'
            ],
            'equity': [
                'shareholders equity', 'share capital', 'retained earnings',
                'total equity', 'reserves', 'accumulated losses'
            ]
        }
        
        # P&L indicators
        self.income_statement_indicators = {
            'revenue': ['revenue', 'sales', 'turnover', 'income from operations'],
            'expenses': [
                'cost of sales', 'operating expenses', 'administrative expenses',
                'finance costs', 'depreciation', 'amortisation'
            ],
            'profit_measures': [
                'gross profit', 'operating profit', 'profit before tax',
                'profit after tax', 'net profit', 'earnings per share'
            ]
        }
        
        # Cash flow indicators
        self.cash_flow_indicators = {
            'operating': [
                'cash flows from operating activities', 'operating cash flow',
                'cash generated from operations'
            ],
            'investing': [
                'cash flows from investing activities', 'investing cash flow',
                'purchase of property plant equipment', 'acquisitions'
            ],
            'financing': [
                'cash flows from financing activities', 'financing cash flow',
                'proceeds from borrowings', 'dividend payments', 'share issues'
            ]
        }

    def parse_document_structure(self, pdf_path: str) -> Dict[str, Any]:
        """
        Parse financial document and extract structured segments
        
        Returns:
            Dictionary containing parsed document structure with segments
        """
        logger.info(f"Starting enhanced structure parsing for: {pdf_path}")
        
        document_structure = {
            'document_id': Path(pdf_path).stem,
            'segments': [],
            'tables': [],
            'cross_references': {},
            'structure_metadata': {},
            'parsing_confidence': 0.0
        }
        
        try:
            # Step 1: Extract pages and basic structure
            pages_data = self._extract_pages_with_structure(pdf_path)
            
            # Step 2: Detect primary financial statements
            primary_statements = self._detect_primary_statements(pages_data)
            
            # Step 3: Identify notes sections
            notes_sections = self._identify_notes_sections(pages_data)
            
            # Step 4: Extract accounting policies
            policy_sections = self._extract_accounting_policies(pages_data)
            
            # Step 5: Find auditor's report
            auditor_sections = self._find_auditor_report(pages_data)
            
            # Step 6: Extract tables with structure analysis
            all_tables = self._extract_structured_tables(pdf_path, pages_data)
            
            # Step 7: Map cross-references between sections
            cross_refs = self._map_cross_references(pages_data)
            
            # Step 8: Assemble final structure
            document_structure = self._assemble_document_structure(
                primary_statements, notes_sections, policy_sections, 
                auditor_sections, all_tables, cross_refs, pages_data
            )
            
            logger.info(f"Successfully parsed document structure: {len(document_structure['segments'])} segments")
            document_structure['success'] = True
            return document_structure
            
        except Exception as e:
            logger.error(f"Error parsing document structure: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            document_structure['success'] = False
            document_structure['error'] = str(e)
            return document_structure

    def _extract_pages_with_structure(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract pages with structural analysis"""
        pages_data = []
        
        try:
            # Use PyMuPDF for PDF extraction
            pdf_document = fitz.open(pdf_path)
            
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                page_text = page.get_text() or ""
                
                # Extract tables using basic text analysis
                page_tables = self._extract_basic_tables(page_text, page_num)
                
                # Analyze text structure
                text_analysis = self._analyze_text_structure(page_text)
                
                page_data = {
                    'page_num': page_num + 1,  # 1-based page numbering
                    'text': page_text,
                    'tables': page_tables,
                    'headings': text_analysis['headings'],
                    'line_count': len(page_text.split('\n')),
                    'table_count': len(page_tables),
                    'structure_indicators': text_analysis['indicators']
                }
                
                pages_data.append(page_data)
            
            pdf_document.close()
                    
        except Exception as e:
            logger.error(f"Error extracting pages from {pdf_path}: {e}")
            
        return pages_data

    def _analyze_text_structure(self, text: str) -> Dict[str, Any]:
        """Analyze text structure to identify headings and indicators"""
        
        lines = text.split('\n')
        headings = []
        indicators = {
            'balance_sheet': 0,
            'income_statement': 0,
            'cash_flows': 0,
            'notes': 0,
            'policies': 0,
            'auditor': 0
        }
        
        for line_num, line in enumerate(lines):
            line_clean = line.strip().lower()
            if not line_clean:
                continue
                
            # Check for heading patterns (lines that are likely headings)
            if self._is_heading_line(line):
                headings.append({
                    'line_num': line_num,
                    'text': line.strip(),
                    'level': self._determine_heading_level(line)
                })
            
            # Count indicators for different statement types
            for category, terms in self.balance_sheet_indicators.items():
                if any(term.lower() in line_clean for term in terms):
                    indicators['balance_sheet'] += 1
                    
            for category, terms in self.income_statement_indicators.items():
                if any(term.lower() in line_clean for term in terms):
                    indicators['income_statement'] += 1
                    
            for category, terms in self.cash_flow_indicators.items():
                if any(term.lower() in line_clean for term in terms):
                    indicators['cash_flows'] += 1
                    
            # Check notes patterns
            if any(re.search(pattern, line_clean, re.IGNORECASE) for pattern in self.notes_patterns):
                indicators['notes'] += 1
                
            # Check policies patterns
            if any(re.search(pattern, line_clean, re.IGNORECASE) for pattern in self.accounting_policies_patterns):
                indicators['policies'] += 1
                
            # Check auditor patterns
            if any(re.search(pattern, line_clean, re.IGNORECASE) for pattern in self.auditor_report_patterns):
                indicators['auditor'] += 1
        
        return {
            'headings': headings,
            'indicators': indicators
        }

    def _extract_basic_tables(self, page_text: str, page_num: int) -> List[Dict[str, Any]]:
        """Extract basic table structures from text using pattern matching"""
        tables = []
        
        # Simple table detection based on alignment patterns
        lines = page_text.split('\n')
        
        # Look for lines with multiple aligned columns (tabs or multiple spaces)
        table_lines = []
        for i, line in enumerate(lines):
            # Check if line has multiple columns (multiple tabs or aligned spaces)
            if '\t' in line or re.search(r'\s{3,}', line):
                # Check if it has numeric data or common table patterns
                if re.search(r'[\d,]+\.?\d*', line) or any(keyword in line.lower() for keyword in 
                    ['total', 'subtotal', 'balance', 'amount', 'year', 'period']):
                    table_lines.append({
                        'line_num': i,
                        'text': line.strip(),
                        'columns': self._split_table_columns(line)
                    })
        
        # Group consecutive table lines into tables
        if table_lines:
            current_table = []
            for table_line in table_lines:
                if not current_table or table_line['line_num'] - current_table[-1]['line_num'] <= 3:
                    current_table.append(table_line)
                else:
                    # Save current table and start new one
                    if len(current_table) >= 2:  # At least 2 rows to be a table
                        tables.append(self._create_table_structure(current_table, page_num))
                    current_table = [table_line]
            
            # Don't forget the last table
            if len(current_table) >= 2:
                tables.append(self._create_table_structure(current_table, page_num))
        
        return tables

    def _split_table_columns(self, line: str) -> List[str]:
        """Split a line into table columns based on spacing/tabs"""
        # First try tab separation
        if '\t' in line:
            return [col.strip() for col in line.split('\t') if col.strip()]
        
        # Then try multiple space separation
        columns = re.split(r'\s{3,}', line)
        return [col.strip() for col in columns if col.strip()]

    def _create_table_structure(self, table_lines: List[Dict[str, Any]], page_num: int) -> Dict[str, Any]:
        """Create table structure from grouped table lines"""
        
        # Extract headers (first row)
        headers = table_lines[0]['columns'] if table_lines else []
        
        # Extract data rows
        data_rows = []
        for line_data in table_lines[1:]:
            data_rows.append(line_data['columns'])
        
        # Determine table type based on content
        table_text = ' '.join(line_data['text'] for line_data in table_lines)
        table_type = self._classify_table_type(table_text, "")
        
        return {
            'table_id': f"table_{page_num}_{table_lines[0]['line_num']}",
            'headers': headers,
            'rows': data_rows,
            'page_num': page_num,
            'table_type': table_type,
            'raw_text': table_text,
            'confidence': 0.7  # Basic confidence for text-extracted tables
        }

    def _is_heading_line(self, line: str) -> bool:
        """Determine if a line is likely a heading"""
        line_clean = line.strip()
        
        # Empty lines are not headings
        if not line_clean:
            return False
            
        # Lines that are all caps and short are likely headings
        if line_clean.isupper() and len(line_clean) < 100:
            return True
            
        # Lines that start with numbers followed by a period or colon
        if re.match(r'^\d+[\.\:]\s+[A-Z]', line_clean):
            return True
            
        # Lines that contain statement keywords and are relatively short
        statement_keywords = [
            'statement', 'report', 'note', 'policy', 'basis', 'summary'
        ]
        if any(keyword in line_clean.lower() for keyword in statement_keywords) and len(line_clean) < 150:
            return True
            
        return False

    def _determine_heading_level(self, line: str) -> int:
        """Determine the hierarchical level of a heading (1-4)"""
        line_clean = line.strip()
        
        # Main statements (level 1)
        for statement_type, patterns in self.primary_statement_patterns.items():
            if any(re.search(pattern, line_clean, re.IGNORECASE) for pattern in patterns):
                return 1
                
        # Major sections like "Notes to Financial Statements" (level 2)
        if any(re.search(pattern, line_clean, re.IGNORECASE) for pattern in self.notes_patterns):
            return 2
            
        # Numbered notes or policies (level 3)
        if re.match(r'^\d+[\.\:]\s+', line_clean):
            return 3
            
        # Sub-sections (level 4)
        return 4

    def _detect_primary_statements(self, pages_data: List[Dict[str, Any]]) -> List[DocumentSegment]:
        """Detect primary financial statements"""
        primary_statements = []
        
        for statement_type, patterns in self.primary_statement_patterns.items():
            # Find pages that contain this statement type
            candidate_pages = []
            
            for page_data in pages_data:
                text = page_data['text'].lower()
                
                # Check if page contains statement title patterns
                title_matches = sum(1 for pattern in patterns if re.search(pattern, text, re.IGNORECASE))
                
                # Check if page contains relevant financial indicators
                if statement_type == 'statement_of_financial_position':
                    indicator_score = page_data['structure_indicators']['balance_sheet']
                elif statement_type == 'statement_of_profit_loss':
                    indicator_score = page_data['structure_indicators']['income_statement']
                elif statement_type == 'statement_of_cash_flows':
                    indicator_score = page_data['structure_indicators']['cash_flows']
                else:
                    indicator_score = 0
                
                # Calculate confidence score
                confidence = (title_matches * 0.4 + min(indicator_score / 10, 1.0) * 0.6)
                
                if confidence > 0.3:  # Threshold for inclusion
                    candidate_pages.append({
                        'page_num': page_data['page_num'],
                        'confidence': confidence,
                        'text': page_data['text'],
                        'tables': page_data['tables']
                    })
            
            if candidate_pages:
                # Sort by confidence and take the best candidates
                candidate_pages.sort(key=lambda x: x['confidence'], reverse=True)
                
                # Group consecutive pages
                statement_pages = self._group_consecutive_pages(candidate_pages)
                
                for page_group in statement_pages:
                    if page_group:  # Ensure group is not empty
                        start_page = min(p['page_num'] for p in page_group)
                        end_page = max(p['page_num'] for p in page_group)
                        
                        combined_content = '\n'.join(p['text'] for p in page_group)
                        combined_tables = []
                        for p in page_group:
                            combined_tables.extend(p['tables'])
                        
                        avg_confidence = sum(p['confidence'] for p in page_group) / len(page_group)
                        
                        segment = DocumentSegment(
                            segment_id=f"{statement_type}_{start_page}_{end_page}",
                            segment_type="primary_statement",
                            title=statement_type.replace('_', ' ').title(),
                            content=combined_content,
                            page_range=(start_page, end_page),
                            tables=combined_tables,
                            cross_references=[],
                            confidence=avg_confidence,
                            metadata={
                                'statement_type': statement_type,
                                'page_count': len(page_group),
                                'table_count': len(combined_tables)
                            }
                        )
                        
                        primary_statements.append(segment)
        
        return primary_statements

    def _group_consecutive_pages(self, candidate_pages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group candidate pages into consecutive ranges"""
        if not candidate_pages:
            return []
            
        # Sort pages by page number
        sorted_pages = sorted(candidate_pages, key=lambda x: x['page_num'])
        
        groups = []
        current_group = [sorted_pages[0]]
        
        for i in range(1, len(sorted_pages)):
            current_page = sorted_pages[i]
            prev_page = sorted_pages[i-1]
            
            # If pages are consecutive and both have reasonable confidence
            if (current_page['page_num'] == prev_page['page_num'] + 1 and 
                current_page['confidence'] > 0.2):
                current_group.append(current_page)
            else:
                # Start new group
                if current_group:
                    groups.append(current_group)
                current_group = [current_page]
        
        # Don't forget the last group
        if current_group:
            groups.append(current_group)
            
        return groups

    def _identify_notes_sections(self, pages_data: List[Dict[str, Any]]) -> List[DocumentSegment]:
        """Identify notes to financial statements sections"""
        notes_sections = []
        
        # Find pages that contain notes patterns
        notes_pages = []
        
        for page_data in pages_data:
            text = page_data['text']
            notes_indicators = page_data['structure_indicators']['notes']
            
            # Check for notes title patterns
            title_matches = 0
            for pattern in self.notes_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    title_matches += 1
            
            # Calculate confidence
            confidence = min((title_matches * 0.3 + notes_indicators * 0.1), 1.0)
            
            if confidence > 0.2:
                notes_pages.append({
                    'page_num': page_data['page_num'],
                    'confidence': confidence,
                    'text': text,
                    'headings': page_data['headings']
                })
        
        if notes_pages:
            # Group notes pages and create segments
            notes_groups = self._group_notes_by_section(notes_pages)
            
            for i, group in enumerate(notes_groups):
                start_page = min(p['page_num'] for p in group)
                end_page = max(p['page_num'] for p in group)
                
                combined_content = '\n'.join(p['text'] for p in group)
                avg_confidence = sum(p['confidence'] for p in group) / len(group)
                
                segment = DocumentSegment(
                    segment_id=f"notes_section_{i}_{start_page}_{end_page}",
                    segment_type="notes",
                    title=f"Notes to Financial Statements (Section {i+1})",
                    content=combined_content,
                    page_range=(start_page, end_page),
                    tables=[],
                    cross_references=[],
                    confidence=avg_confidence,
                    metadata={
                        'section_number': i+1,
                        'page_count': len(group)
                    }
                )
                
                notes_sections.append(segment)
        
        return notes_sections

    def _group_notes_by_section(self, notes_pages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group notes pages by logical sections"""
        if not notes_pages:
            return []
            
        # Sort by page number
        sorted_pages = sorted(notes_pages, key=lambda x: x['page_num'])
        
        # Simple grouping: consecutive pages with similar confidence
        groups = []
        current_group = [sorted_pages[0]]
        
        for i in range(1, len(sorted_pages)):
            current_page = sorted_pages[i]
            prev_page = sorted_pages[i-1]
            
            # Group consecutive pages
            if current_page['page_num'] == prev_page['page_num'] + 1:
                current_group.append(current_page)
            else:
                groups.append(current_group)
                current_group = [current_page]
        
        if current_group:
            groups.append(current_group)
            
        return groups

    def _extract_accounting_policies(self, pages_data: List[Dict[str, Any]]) -> List[DocumentSegment]:
        """Extract accounting policies sections"""
        policy_sections = []
        
        for page_data in pages_data:
            text = page_data['text']
            policy_indicators = page_data['structure_indicators']['policies']
            
            # Check for accounting policies patterns
            title_matches = 0
            for pattern in self.accounting_policies_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    title_matches += 1
            
            confidence = min((title_matches * 0.4 + policy_indicators * 0.2), 1.0)
            
            if confidence > 0.3:
                segment = DocumentSegment(
                    segment_id=f"accounting_policies_{page_data['page_num']}",
                    segment_type="policies",
                    title="Accounting Policies",
                    content=text,
                    page_range=(page_data['page_num'], page_data['page_num']),
                    tables=page_data['tables'],
                    cross_references=[],
                    confidence=confidence,
                    metadata={
                        'page_num': page_data['page_num'],
                        'policy_indicators': policy_indicators
                    }
                )
                
                policy_sections.append(segment)
        
        return policy_sections

    def _find_auditor_report(self, pages_data: List[Dict[str, Any]]) -> List[DocumentSegment]:
        """Find auditor's report sections"""
        auditor_sections = []
        
        for page_data in pages_data:
            text = page_data['text']
            auditor_indicators = page_data['structure_indicators']['auditor']
            
            # Check for auditor report patterns
            title_matches = 0
            for pattern in self.auditor_report_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    title_matches += 1
            
            confidence = min((title_matches * 0.5 + auditor_indicators * 0.3), 1.0)
            
            if confidence > 0.4:
                segment = DocumentSegment(
                    segment_id=f"auditor_report_{page_data['page_num']}",
                    segment_type="auditor_report",
                    title="Independent Auditor's Report",
                    content=text,
                    page_range=(page_data['page_num'], page_data['page_num']),
                    tables=page_data['tables'],
                    cross_references=[],
                    confidence=confidence,
                    metadata={
                        'page_num': page_data['page_num'],
                        'auditor_indicators': auditor_indicators
                    }
                )
                
                auditor_sections.append(segment)
        
        return auditor_sections

    def _extract_structured_tables(self, pdf_path: str, pages_data: List[Dict[str, Any]]) -> List[TableStructure]:
        """Extract tables with structural analysis"""
        all_tables = []
        table_counter = 0
        
        try:
            for page_data in pages_data:
                page_num = page_data['page_num']
                page_tables = page_data['tables']
                
                if page_tables:
                    for table_data in page_tables:
                        if table_data and isinstance(table_data, dict):
                            # Convert our table format to TableStructure
                            headers = table_data.get('headers', [])
                            rows = table_data.get('rows', [])
                            
                            if len(rows) > 0:  # At least 1 row
                                # Determine table type based on content
                                table_type = table_data.get('table_type', 'other')
                                
                                # Calculate confidence based on structure
                                confidence = table_data.get('confidence', 0.7)
                                
                                # Extract table title from raw text
                                table_title = self._extract_table_title(
                                    table_data.get('raw_text', ''), headers + (rows[0] if rows else [])
                                )
                                
                                table_structure = TableStructure(
                                    table_id=table_data.get('table_id', f"table_{table_counter}"),
                                    title=table_title,
                                    headers=headers,
                                    rows=rows,
                                    page_num=page_num,
                                    bbox=(0, 0, 0, 0),  # Simplified bbox for text-extracted tables
                                    table_type=table_type,
                                    confidence=confidence
                                )
                                
                                all_tables.append(table_structure)
                                table_counter += 1
                                
        except Exception as e:
            logger.error(f"Error extracting tables: {e}")
            
        return all_tables

    def _classify_table_type(self, table_data: List[List[str]], page_text: str) -> str:
        """Classify table type based on content and context"""
        
        if not table_data:
            return "unknown"
            
        # Convert table to text for analysis
        table_text = ' '.join(' '.join(row) for row in table_data).lower()
        page_text_lower = page_text.lower()
        
        # Check for financial position indicators
        balance_sheet_terms = ['assets', 'liabilities', 'equity', 'current', 'non-current']
        if sum(1 for term in balance_sheet_terms if term in table_text) >= 2:
            return "financial_position"
            
        # Check for P&L indicators
        income_terms = ['revenue', 'expenses', 'profit', 'loss', 'income']
        if sum(1 for term in income_terms if term in table_text) >= 2:
            return "profit_loss"
            
        # Check for cash flow indicators
        cashflow_terms = ['cash flow', 'operating', 'investing', 'financing']
        if sum(1 for term in cashflow_terms if term in table_text) >= 2:
            return "cash_flows"
            
        # Check if it's a notes table
        if 'note' in page_text_lower:
            return "notes_table"
            
        return "other"

    def _calculate_table_confidence(self, table_data: List[List[str]], table_type: str) -> float:
        """Calculate confidence score for table classification"""
        
        if not table_data or len(table_data) < 2:
            return 0.0
            
        confidence = 0.5  # Base confidence
        
        # Add confidence based on structure
        if len(table_data) >= 3:  # Has header + multiple rows
            confidence += 0.2
            
        if len(table_data[0]) >= 3:  # Has multiple columns
            confidence += 0.2
            
        # Add confidence based on type classification
        if table_type != "unknown":
            confidence += 0.1
            
        return min(confidence, 1.0)

    def _extract_table_title(self, page_text: str, table_data: List[List[str]]) -> str:
        """Extract table title from surrounding text"""
        
        # Simple heuristic: look for lines that might be table titles
        lines = page_text.split('\n')
        
        # Look for patterns that indicate table titles
        title_patterns = [
            r'consolidated\s+statement',
            r'statement\s+of',
            r'notes?\s+to',
            r'table\s+\d+'
        ]
        
        for line in lines:
            line_clean = line.strip()
            if any(re.search(pattern, line_clean, re.IGNORECASE) for pattern in title_patterns):
                if len(line_clean) < 200:  # Reasonable title length
                    return line_clean
                    
        return "Extracted Table"

    def _map_cross_references(self, pages_data: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Map cross-references between document sections"""
        cross_refs = defaultdict(list)
        
        # Pattern for note references
        note_ref_pattern = r'note\s+(\d+)'
        
        for page_data in pages_data:
            text = page_data['text']
            page_num = page_data['page_num']
            
            # Find note references
            note_matches = re.findall(note_ref_pattern, text, re.IGNORECASE)
            for note_num in note_matches:
                cross_refs[f"page_{page_num}"].append(f"note_{note_num}")
        
        return dict(cross_refs)

    def _assemble_document_structure(self, primary_statements: List[DocumentSegment],
                                   notes_sections: List[DocumentSegment],
                                   policy_sections: List[DocumentSegment],
                                   auditor_sections: List[DocumentSegment],
                                   tables: List[TableStructure],
                                   cross_refs: Dict[str, List[str]],
                                   pages_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assemble final document structure"""
        
        all_segments = []
        all_segments.extend(primary_statements)
        all_segments.extend(notes_sections)
        all_segments.extend(policy_sections)
        all_segments.extend(auditor_sections)
        
        # Calculate overall parsing confidence
        if all_segments:
            avg_confidence = sum(seg.confidence for seg in all_segments) / len(all_segments)
        else:
            avg_confidence = 0.0
        
        # Create structure metadata
        structure_metadata = {
            'total_pages': len(pages_data),
            'primary_statements_count': len(primary_statements),
            'notes_sections_count': len(notes_sections),
            'policy_sections_count': len(policy_sections),
            'auditor_sections_count': len(auditor_sections),
            'tables_count': len(tables),
            'cross_references_count': len(cross_refs)
        }
        
        return {
            'success': True,
            'segments': all_segments,
            'tables': tables,
            'cross_references': cross_refs,
            'structure_metadata': structure_metadata,
            'parsing_confidence': avg_confidence
        }

    def _map_cross_references(self, pages_data: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Map cross-references between document sections"""
        cross_refs = {}
        
        # Look for note references in the format "Note X" or "see note X"
        note_pattern = r'(?:see\s+)?note\s+(\d+)'
        page_pattern = r'page\s+(\d+)'
        
        for page_data in pages_data:
            page_num = page_data['page_num']
            text = page_data['text'].lower()
            
            # Find note references
            note_matches = re.findall(note_pattern, text, re.IGNORECASE)
            page_matches = re.findall(page_pattern, text, re.IGNORECASE)
            
            if note_matches or page_matches:
                page_key = f"page_{page_num}"
                cross_refs[page_key] = []
                
                for note_num in note_matches:
                    cross_refs[page_key].append(f"note_{note_num}")
                    
                for page_ref in page_matches:
                    cross_refs[page_key].append(f"page_{page_ref}")
        
        return cross_refs

    def _assemble_document_structure(self, primary_statements: List[DocumentSegment], 
                                   notes_sections: List[DocumentSegment], 
                                   policy_sections: List[DocumentSegment], 
                                   auditor_sections: List[DocumentSegment],
                                   tables: List[TableStructure], 
                                   cross_refs: Dict[str, List[str]], 
                                   pages_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assemble final document structure"""
        
        # Combine all segments
        all_segments = []
        all_segments.extend(primary_statements)
        all_segments.extend(notes_sections)
        all_segments.extend(policy_sections)
        all_segments.extend(auditor_sections)
        
        # Sort segments by page order
        all_segments.sort(key=lambda x: x.page_range[0])
        
        # Calculate overall parsing confidence
        if all_segments:
            avg_confidence = sum(seg.confidence for seg in all_segments) / len(all_segments)
        else:
            avg_confidence = 0.0
        
        # Create structure metadata
        structure_metadata = {
            'total_pages': len(pages_data),
            'primary_statements_count': len(primary_statements),
            'notes_sections_count': len(notes_sections),
            'policy_sections_count': len(policy_sections),
            'auditor_sections_count': len(auditor_sections),
            'tables_count': len(tables),
            'cross_references_count': len(cross_refs)
        }
        
        return {
            'success': True,
            'segments': all_segments,
            'tables': tables,
            'cross_references': cross_refs,
            'structure_metadata': structure_metadata,
            'parsing_confidence': avg_confidence
        }
        
    def _match_pattern(self, pattern: str, text: str) -> bool:
        """
        Helper method for pattern matching used in tests
        
        Args:
            pattern: Regular expression pattern to match
            text: Text to search in
            
        Returns:
            bool: True if pattern matches, False otherwise
        """
        try:
            return bool(re.search(pattern, text, re.IGNORECASE | re.MULTILINE))
        except re.error as e:
            logger.warning(f"Invalid regex pattern '{pattern}': {e}")
            return False
            
    def _segment_content(self, content: str) -> List[Dict[str, Any]]:
        """
        Helper method for content segmentation used in tests
        
        Args:
            content: Text content to segment
            
        Returns:
            List of content segments
        """
        # Simple text segmentation for testing
        segments = []
        lines = content.split('\n')
        current_segment = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Start new segment on headings or statements
            if (self._is_heading_line(line) or 
                any(pattern in line.lower() for pattern in ['statement', 'note', 'policy'])):
                if current_segment:
                    segments.append({
                        'content': '\n'.join(current_segment),
                        'type': 'segment'
                    })
                    current_segment = []
            
            current_segment.append(line)
        
        # Add final segment
        if current_segment:
            segments.append({
                'content': '\n'.join(current_segment),
                'type': 'segment'
            })
            
        return segments
        
    def parse_text_content(self, content: str) -> List[Dict[str, Any]]:
        """
        Parse text content for testing purposes
        
        Args:
            content: Text content to parse
            
        Returns:
            List of parsed segments
        """
        return self._segment_content(content)

def main():
    """Test the enhanced parser"""
    parser = EnhancedFinancialStatementParser()
    
    # Example usage
    test_pdf = "sample_financial_report.pdf"  # Replace with actual path
    if Path(test_pdf).exists():
        structure = parser.parse_document_structure(test_pdf)
        
        print(f"Document Structure Analysis:")
        print(f"- Total segments: {len(structure['segments'])}")
        print(f"- Total tables: {len(structure['tables'])}")
        print(f"- Parsing confidence: {structure['parsing_confidence']:.2f}")
        
        for segment in structure['segments']:
            print(f"  - {segment.segment_type}: {segment.title} (pages {segment.page_range[0]}-{segment.page_range[1]})")

if __name__ == "__main__":
    main()