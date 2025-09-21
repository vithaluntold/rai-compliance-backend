"""
Financial Statement Recognition System
Identifies Balance Sheet, P&L, Cash Flow, Statement of Equity Changes, and Notes in PDFs
"""

import logging
import re
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class FinancialStatementRecognizer:
    """
    Recognizes and categorizes financial statement sections in PDF documents
    """
    
    def __init__(self):
        self.statement_patterns = {
            'BALANCE_SHEET': {
                'keywords': [
                    'balance sheet', 'statement of financial position', 'position statement',
                    'assets', 'liabilities', 'equity', 'shareholders equity',
                    'current assets', 'non-current assets', 'current liabilities'
                ],
                'headers': [
                    r'(?i)balance\s+sheet',
                    r'(?i)statement\s+of\s+financial\s+position',
                    r'(?i)consolidated\s+balance\s+sheet',
                    r'(?i)statement\s+of\s+position'
                ]
            },
            'PROFIT_LOSS': {
                'keywords': [
                    'profit and loss', 'income statement', 'comprehensive income',
                    'statement of profit', 'revenue', 'expenses', 'profit or loss',
                    'total comprehensive income', 'operating income', 'net income'
                ],
                'headers': [
                    r'(?i)profit\s+and\s+loss',
                    r'(?i)statement\s+of\s+profit',
                    r'(?i)income\s+statement',
                    r'(?i)comprehensive\s+income',
                    r'(?i)statement\s+of\s+comprehensive\s+income'
                ]
            },
            'CASH_FLOW': {
                'keywords': [
                    'cash flow', 'cash flows', 'cashflow', 'statement of cash flows',
                    'operating activities', 'investing activities', 'financing activities',
                    'net cash flow', 'cash and cash equivalents'
                ],
                'headers': [
                    r'(?i)cash\s+flow',
                    r'(?i)statement\s+of\s+cash\s+flows?',
                    r'(?i)consolidated\s+cash\s+flow'
                ]
            },
            'EQUITY_CHANGES': {
                'keywords': [
                    'changes in equity', 'statement of changes', 'equity movement',
                    'retained earnings', 'reserves', 'share capital', 'equity reconciliation'
                ],
                'headers': [
                    r'(?i)changes\s+in\s+equity',
                    r'(?i)statement\s+of\s+changes\s+in\s+equity',
                    r'(?i)equity\s+reconciliation'
                ]
            },
            'NOTES': {
                'keywords': [
                    'notes to', 'note ', 'accounting policies', 'significant accounting',
                    'basis of preparation', 'critical accounting', 'note 1', 'note 2'
                ],
                'headers': [
                    r'(?i)notes?\s+to.*financial\s+statements?',
                    r'(?i)note\s+\d+',
                    r'(?i)accounting\s+policies',
                    r'(?i)significant\s+accounting'
                ]
            }
        }
        
    def recognize_statements(self, pdf_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Recognize financial statement sections in PDF
        
        Returns:
            Dict with statement types as keys and list of recognized sections as values
            Each section contains: page_num, start_line, end_line, content_preview, confidence
        """
        logger.info(f"Recognizing financial statements in: {pdf_path}")
        
        recognized_statements = {
            'BALANCE_SHEET': [],
            'PROFIT_LOSS': [],
            'CASH_FLOW': [],
            'EQUITY_CHANGES': [],
            'NOTES': []
        }
        
        try:
            with fitz.open(pdf_path) as doc:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text()
                    
                    # Split into lines for line-by-line analysis
                    lines = text.split('\n')
                    
                    # Analyze each section of the page
                    sections = self._identify_sections(lines, page_num)
                    
                    # Categorize sections by statement type
                    for section in sections:
                        statement_type = self._classify_section(section)
                        if statement_type and statement_type != 'UNKNOWN':
                            recognized_statements[statement_type].append(section)
                            
        except Exception as e:
            logger.error(f"Error recognizing statements: {e}")
            
        return recognized_statements
    
    def _identify_sections(self, lines: List[str], page_num: int) -> List[Dict[str, Any]]:
        """
        Identify logical sections within a page based on content patterns
        """
        sections = []
        current_section = {
            'page_num': page_num,
            'start_line': 0,
            'lines': [],
            'content_preview': ''
        }
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Check if this line starts a new section (header patterns)
            if self._is_section_header(line):
                # Save current section if it has content
                if current_section['lines']:
                    current_section['end_line'] = line_num - 1
                    current_section['content_preview'] = ' '.join(current_section['lines'][:3])
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    'page_num': page_num,
                    'start_line': line_num,
                    'lines': [line],
                    'content_preview': line
                }
            else:
                current_section['lines'].append(line)
        
        # Add final section
        if current_section['lines']:
            current_section['end_line'] = len(lines) - 1
            current_section['content_preview'] = ' '.join(current_section['lines'][:3])
            sections.append(current_section)
            
        return sections
    
    def _is_section_header(self, line: str) -> bool:
        """
        Check if a line appears to be a section header
        """
        # Look for header patterns
        header_indicators = [
            r'(?i)^[A-Z\s]{10,}$',  # All caps lines (likely headers)
            r'(?i)statement\s+of',   # Statement of...
            r'(?i)note\s+\d+',       # Note 1, Note 2, etc.
            r'(?i)^\d+\.\s+[A-Z]',   # Numbered sections
        ]
        
        for pattern in header_indicators:
            if re.search(pattern, line):
                return True
                
        return False
    
    def _classify_section(self, section: Dict[str, Any]) -> Optional[str]:
        """
        Classify a section as one of the financial statement types
        """
        content = ' '.join(section['lines']).lower()
        
        best_match = 'UNKNOWN'
        best_score = 0
        
        for statement_type, patterns in self.statement_patterns.items():
            score = 0
            
            # Check keyword matches
            for keyword in patterns['keywords']:
                if keyword.lower() in content:
                    score += 1
            
            # Check header pattern matches (weighted more heavily)
            for header_pattern in patterns['headers']:
                if re.search(header_pattern, content):
                    score += 3
            
            # Calculate confidence based on content length and matches
            confidence = min(score / (len(patterns['keywords']) + len(patterns['headers'])), 1.0)
            
            if confidence > best_score and confidence > 0.1:  # Minimum confidence threshold
                best_score = confidence
                best_match = statement_type
        
        # Add confidence to section
        section['confidence'] = best_score
        section['classified_as'] = best_match
        
        return best_match if best_score > 0.1 else None
    
    def get_statement_summary(self, recognized_statements: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Generate a summary of recognized financial statements
        """
        summary = {
            'total_sections': 0,
            'statements_found': [],
            'page_distribution': {},
            'confidence_scores': {}
        }
        
        for statement_type, sections in recognized_statements.items():
            if sections:
                summary['statements_found'].append(statement_type)
                summary['total_sections'] += len(sections)
                
                # Calculate average confidence
                confidences = [s.get('confidence', 0) for s in sections]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                summary['confidence_scores'][statement_type] = round(avg_confidence, 2)
                
                # Page distribution
                pages = [s['page_num'] for s in sections]
                summary['page_distribution'][statement_type] = {
                    'pages': sorted(set(pages)),
                    'page_count': len(set(pages))
                }
        
        return summary


def test_recognition():
    """Test function for financial statement recognition"""
    recognizer = FinancialStatementRecognizer()
    
    # Test with a sample PDF if available
    test_files = [
        "c:/Users/saivi/OneDrive/Documents/Audricc all/uploads/RAI-1757795217-3ADC237A.pdf"
    ]
    
    for pdf_path in test_files:
        try:
            print(f"\n=== Testing Recognition on: {pdf_path} ===")
            
            # Recognize statements
            statements = recognizer.recognize_statements(pdf_path)
            
            # Generate summary
            summary = recognizer.get_statement_summary(statements)
            
            print(f"Total sections found: {summary['total_sections']}")
            print(f"Statement types identified: {summary['statements_found']}")
            
            for stmt_type, confidence in summary['confidence_scores'].items():
                section_count = len(statements[stmt_type])
                pages = summary['page_distribution'][stmt_type]['pages']
                print(f"  {stmt_type}: {section_count} sections, confidence {confidence}, pages {pages}")
                
            # Show sample sections
            for stmt_type, sections in statements.items():
                if sections:
                    print(f"\n--- {stmt_type} Sample ---")
                    for i, section in enumerate(sections[:2]):  # Show first 2 sections
                        print(f"  Section {i+1}: Page {section['page_num']}, Lines {section['start_line']}-{section.get('end_line', 'N/A')}")
                        print(f"  Preview: {section['content_preview'][:100]}...")
                        print(f"  Confidence: {section.get('confidence', 0):.2f}")
            
        except Exception as e:
            print(f"Error testing {pdf_path}: {e}")


if __name__ == "__main__":
    test_recognition()