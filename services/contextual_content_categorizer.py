"""
Contextual Content Categorization System
Maps financial document content to Category → Topic → Requirement Type using extended context
"""

import logging
import re
from typing import Dict, List, Tuple, Optional, Any
import json
import spacy
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.financial_statement_recognizer import FinancialStatementRecognizer

logger = logging.getLogger(__name__)


class ContextualContentCategorizer:
    """
    Categorizes financial document content using extended context and smart tagging
    """
    
    def __init__(self):
        # Load our existing categorization data
        self.load_categorization_system()
        # Load spaCy model (with fallback)
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✅ spaCy model loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ spaCy model failed to load: {e}. Using fallback categorization.")
            self.nlp = None
        
        self.statement_recognizer = FinancialStatementRecognizer()
        
    def load_categorization_system(self):
        """Load the existing question categorization system"""
        try:
            # Load category lookup for smart matching
            with open('categorized_questions/category_lookup.json', 'r', encoding='utf-8') as f:
                self.category_lookup = json.load(f)
            
            # Load smart search config
            with open('categorized_questions/smart_search_config.json', 'r', encoding='utf-8') as f:
                self.search_config = json.load(f)
                
            logger.info("Loaded categorization system successfully")
            
        except FileNotFoundError as e:
            logger.error(f"Could not load categorization files: {e}")
            # Fallback patterns if files not found
            self.category_lookup = {}
            self.search_config = {}
    
    def categorize_document_content(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Process entire document and categorize each content piece with extended context
        
        Returns list of categorized content chunks with metadata:
        - category, topic, requirement_type
        - page_num, paragraph_num, line_range
        - cross_references (note numbers, table refs)
        - extended_context (preceding + current + succeeding)
        - statement_type (BALANCE_SHEET, PROFIT_LOSS, etc.)
        """
        logger.info(f"Starting contextual categorization of: {pdf_path}")
        
        # First, recognize financial statement sections
        recognized_statements = self.statement_recognizer.recognize_statements(pdf_path)
        
        categorized_content = []
        
        try:
            import fitz
            with fitz.open(pdf_path) as doc:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text()
                    lines = text.split('\n')
                    
                    # Process each line with extended context
                    page_content = self._process_page_with_context(
                        lines, page_num, recognized_statements
                    )
                    
                    categorized_content.extend(page_content)
                    
        except Exception as e:
            logger.error(f"Error categorizing document: {e}")
            
        logger.info(f"Categorized {len(categorized_content)} content pieces")
        return categorized_content
    
    def categorize_page_texts(self, page_texts: List[Dict[str, Any]], pdf_path: str = None) -> List[Dict[str, Any]]:
        """
        Categorize pre-parsed page texts from document chunker
        
        Args:
            page_texts: List of dicts with 'page_num', 'text', 'length' keys
            pdf_path: Optional PDF path for statement recognition
            
        Returns:
            List of categorized content chunks
        """
        logger.info(f"Starting contextual categorization of {len(page_texts)} pre-parsed pages")
        
        # Try to recognize financial statements if PDF path provided
        recognized_statements = {}
        if pdf_path:
            try:
                recognized_statements = self.statement_recognizer.recognize_statements(pdf_path)
            except Exception as e:
                logger.warning(f"Could not recognize statements from PDF path {pdf_path}: {e}")
        
        categorized_content = []
        
        for page_data in page_texts:
            page_num = page_data.get('page_num', 0)
            text = page_data.get('text', '')
            lines = text.split('\n')
            
            # Process each line with extended context
            page_content = self._process_page_with_context(
                lines, page_num, recognized_statements
            )
            
            categorized_content.extend(page_content)
                    
        logger.info(f"Categorized {len(categorized_content)} content pieces from pre-parsed pages")
        return categorized_content
    
    def _process_page_with_context(
        self, 
        lines: List[str], 
        page_num: int, 
        recognized_statements: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Process each line on a page with extended context (preceding + current + succeeding)
        """
        page_content = []
        
        # Determine what statement type(s) this page contains
        page_statement_types = self._get_page_statement_types(page_num, recognized_statements)
        
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line or len(line) < 10:  # Skip very short lines
                continue
                
            # Get extended context (3 lines before, current, 3 lines after)
            context_lines = self._get_extended_context(lines, line_idx, context_window=3)
            extended_context = ' '.join(context_lines)
            
            # Extract cross-references and metadata
            cross_refs = self._extract_cross_references(extended_context)
            
            # Categorize using NLP and pattern matching
            category_info = self._categorize_content_piece(extended_context, page_statement_types)
            
            # Create content piece with full metadata
            content_piece = {
                # Core content
                'content': line,
                'extended_context': extended_context,
                
                # Location metadata (crucial for citations)
                'page_num': page_num + 1,  # 1-indexed for user display
                'paragraph_num': self._estimate_paragraph_number(lines, line_idx),
                'line_range': [line_idx, line_idx],
                'char_position': sum(len(l) + 1 for l in lines[:line_idx]),
                
                # Cross-references (crucial for citations)
                'cross_references': cross_refs,
                'note_numbers': self._extract_note_numbers(extended_context),
                'table_references': self._extract_table_references(extended_context),
                
                # Statement type context
                'statement_types': page_statement_types,
                'primary_statement': page_statement_types[0] if page_statement_types else 'UNKNOWN',
                
                # Category tagging (Category → Topic → Requirement Type)
                'category': category_info.get('category', 'UNKNOWN'),
                'topic': category_info.get('topic', 'GENERAL'),
                'requirement_type': category_info.get('requirement_type', 'UNKNOWN'),
                'confidence': category_info.get('confidence', 0.0),
                
                # Search optimization
                'search_terms': category_info.get('search_terms', []),
                'semantic_tags': category_info.get('semantic_tags', []),
                
                # Content classification
                'content_type': self._classify_content_type(line, extended_context),
                'is_numerical': self._contains_numbers(line),
                'is_header': self._is_likely_header(line),
                'is_table_data': self._is_table_data(line)
            }
            
            page_content.append(content_piece)
            
        return page_content
    
    def _get_extended_context(self, lines: List[str], center_idx: int, context_window: int = 3) -> List[str]:
        """Get surrounding lines for extended context"""
        start_idx = max(0, center_idx - context_window)
        end_idx = min(len(lines), center_idx + context_window + 1)
        return [line.strip() for line in lines[start_idx:end_idx] if line.strip()]
    
    def _get_page_statement_types(self, page_num: int, recognized_statements: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """Determine which statement types appear on this page"""
        page_types = []
        for statement_type, sections in recognized_statements.items():
            for section in sections:
                if section['page_num'] == page_num:
                    if statement_type not in page_types:
                        page_types.append(statement_type)
        return page_types
    
    def _categorize_content_piece(self, content: str, statement_types: List[str]) -> Dict[str, Any]:
        """
        Categorize content using NLP analysis and pattern matching
        Based on our existing Category → Topic → Requirement Type system
        """
        # Process with spaCy for entity extraction
        doc = self.nlp(content.lower())
        entities = [ent.text for ent in doc.ents]
        
        # Extract key financial terms
        search_terms = self._extract_financial_terms(content)
        
        # Determine category based on content patterns
        category = self._determine_category(content, statement_types)
        topic = self._determine_topic(content, entities, statement_types)
        requirement_type = self._determine_requirement_type(content)
        
        # Calculate confidence based on pattern matches
        confidence = self._calculate_categorization_confidence(content, category, topic)
        
        return {
            'category': category,
            'topic': topic,
            'requirement_type': requirement_type,
            'confidence': confidence,
            'search_terms': search_terms,
            'semantic_tags': entities
        }
    
    def _determine_category(self, content: str, statement_types: List[str]) -> str:
        """Determine Category: DISCLOSURE, PRESENTATION, MEASUREMENT, RECOGNITION"""
        content_lower = content.lower()
        
        # DISCLOSURE patterns
        disclosure_patterns = [
            'disclose', 'disclosure', 'note', 'explain', 'detail', 'describe',
            'information', 'significant', 'accounting policy', 'basis'
        ]
        
        # PRESENTATION patterns  
        presentation_patterns = [
            'present', 'presentation', 'show', 'display', 'statement',
            'line item', 'separate', 'classify', 'format'
        ]
        
        # MEASUREMENT patterns
        measurement_patterns = [
            'fair value', 'measure', 'measurement', 'value', 'amount',
            'calculate', 'valuation', 'impairment', 'cost'
        ]
        
        # RECOGNITION patterns
        recognition_patterns = [
            'recognise', 'recognize', 'recognition', 'record', 'when',
            'initial', 'subsequently', 'derecognise'
        ]
        
        # Score each category
        scores = {
            'DISCLOSURE': sum(1 for pattern in disclosure_patterns if pattern in content_lower),
            'PRESENTATION': sum(1 for pattern in presentation_patterns if pattern in content_lower),
            'MEASUREMENT': sum(1 for pattern in measurement_patterns if pattern in content_lower),
            'RECOGNITION': sum(1 for pattern in recognition_patterns if pattern in content_lower)
        }
        
        # Return highest scoring category
        if max(scores.values()) > 0:
            return max(scores.keys(), key=lambda k: scores[k])
        else:
            return 'DISCLOSURE'  # Default fallback
    
    def _determine_topic(self, content: str, entities: List[str], statement_types: List[str]) -> str:
        """Determine Topic: FINANCIAL_INSTRUMENTS, PROPERTY_ASSETS, etc."""
        content_lower = content.lower()
        
        topic_patterns = {
            'FINANCIAL_INSTRUMENTS': ['financial instrument', 'derivative', 'security', 'investment', 'fair value'],
            'PROPERTY_ASSETS': ['property', 'plant', 'equipment', 'asset', 'depreciation', 'impairment'],
            'REVENUE_PERFORMANCE': ['revenue', 'income', 'performance', 'sale', 'contract'],
            'TAX': ['tax', 'deferred tax', 'income tax', 'current tax'],
            'LEASES': ['lease', 'lessee', 'lessor', 'rental'],
            'FOREIGN_CURRENCY': ['foreign', 'currency', 'exchange', 'translation'],
            'IMPAIRMENT': ['impairment', 'impair', 'recoverable'],
            'FINANCIAL_POSITION': ['balance', 'position', 'asset', 'liability', 'equity']
        }
        
        # Score topics
        topic_scores = {}
        for topic, patterns in topic_patterns.items():
            score = sum(1 for pattern in patterns if pattern in content_lower)
            if score > 0:
                topic_scores[topic] = score
        
        # Consider statement type context
        if statement_types:
            if 'BALANCE_SHEET' in statement_types:
                topic_scores['FINANCIAL_POSITION'] = topic_scores.get('FINANCIAL_POSITION', 0) + 1
            elif 'PROFIT_LOSS' in statement_types:
                topic_scores['REVENUE_PERFORMANCE'] = topic_scores.get('REVENUE_PERFORMANCE', 0) + 1
        
        return max(topic_scores.keys(), key=lambda k: topic_scores[k]) if topic_scores else 'GENERAL'
    
    def _determine_requirement_type(self, content: str) -> str:
        """Determine Requirement Type: MANDATORY, CONDITIONAL, OPTIONAL"""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['shall', 'must', 'require', 'mandatory']):
            return 'MANDATORY'
        elif any(word in content_lower for word in ['if', 'when', 'where', 'conditional']):
            return 'CONDITIONAL'  
        elif any(word in content_lower for word in ['may', 'optional', 'elect']):
            return 'OPTIONAL'
        else:
            return 'MANDATORY'  # Default assumption
    
    def _extract_cross_references(self, content: str) -> List[str]:
        """Extract cross-references like 'see note 15', 'refer to table 3'"""
        patterns = [
            r'(?i)note\s+\d+(?:\.\d+)?',
            r'(?i)see\s+note\s+\d+',
            r'(?i)refer\s+to\s+note\s+\d+',
            r'(?i)table\s+\d+',
            r'(?i)appendix\s+\w+',
            r'(?i)schedule\s+\d+'
        ]
        
        refs = []
        for pattern in patterns:
            matches = re.findall(pattern, content)
            refs.extend(matches)
        
        return list(set(refs))  # Remove duplicates
    
    def _extract_note_numbers(self, content: str) -> List[str]:
        """Extract note numbers for citation tracking"""
        pattern = r'(?i)note\s+(\d+(?:\.\d+)?)'
        matches = re.findall(pattern, content)
        return list(set(matches))
    
    def _extract_table_references(self, content: str) -> List[str]:
        """Extract table references for citation tracking"""
        pattern = r'(?i)table\s+(\d+(?:\.\d+)?)'
        matches = re.findall(pattern, content)
        return list(set(matches))
    
    def _extract_financial_terms(self, content: str) -> List[str]:
        """Extract key financial terms for search optimization"""
        # Common financial terms that are useful for search
        financial_terms = [
            'fair value', 'carrying amount', 'depreciation', 'amortisation',
            'impairment', 'revenue', 'liability', 'asset', 'equity',
            'cash flow', 'comprehensive income', 'profit', 'loss'
        ]
        
        content_lower = content.lower()
        found_terms = [term for term in financial_terms if term in content_lower]
        return found_terms
    
    def _estimate_paragraph_number(self, lines: List[str], line_idx: int) -> int:
        """Estimate paragraph number by counting empty line breaks"""
        paragraph_count = 1
        for i in range(line_idx):
            if not lines[i].strip():  # Empty line indicates paragraph break
                if i > 0 and lines[i-1].strip():  # Only count if previous line had content
                    paragraph_count += 1
        return paragraph_count
    
    def _calculate_categorization_confidence(self, content: str, category: str, topic: str) -> float:
        """Calculate confidence score for categorization"""
        # Base confidence from content length and keyword matches
        base_confidence = min(len(content) / 100, 1.0)  # Longer content = higher confidence
        
        # Boost confidence if we found specific patterns
        if category != 'UNKNOWN':
            base_confidence += 0.2
        if topic != 'GENERAL':
            base_confidence += 0.2
            
        return min(base_confidence, 1.0)
    
    def _classify_content_type(self, line: str, context: str) -> str:
        """Classify the type of content (header, data, narrative, etc.)"""
        if self._is_likely_header(line):
            return 'HEADER'
        elif self._is_table_data(line):
            return 'TABLE_DATA'
        elif self._contains_numbers(line):
            return 'NUMERICAL_DATA'
        else:
            return 'NARRATIVE'
    
    def _is_likely_header(self, line: str) -> bool:
        """Check if line is likely a header"""
        return (
            line.isupper() or
            re.match(r'^\d+\.?\s+[A-Z]', line) or
            len(line.split()) <= 6 and any(c.isupper() for c in line)
        )
    
    def _is_table_data(self, line: str) -> bool:
        """Check if line contains tabular data"""
        # Look for multiple numbers or currency amounts
        number_pattern = r'\d+(?:,\d{3})*(?:\.\d+)?'
        numbers = re.findall(number_pattern, line)
        return len(numbers) >= 2
    
    def _contains_numbers(self, line: str) -> bool:
        """Check if line contains numerical data"""
        return bool(re.search(r'\d', line))


def test_categorization():
    """Test the contextual categorization system"""
    categorizer = ContextualContentCategorizer()
    
    # Test with sample PDF
    pdf_path = "c:/Users/saivi/OneDrive/Documents/Audricc all/uploads/RAI-1757795217-3ADC237A.pdf"
    
    try:
        print("=== Testing Contextual Content Categorization ===")
        categorized_content = categorizer.categorize_document_content(pdf_path)
        
        print(f"Total content pieces categorized: {len(categorized_content)}")
        
        # Show category distribution
        categories = {}
        topics = {}
        for piece in categorized_content:
            cat = piece['category']
            topic = piece['topic']
            categories[cat] = categories.get(cat, 0) + 1
            topics[topic] = topics.get(topic, 0) + 1
        
        print(f"\nCategory Distribution:")
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count}")
        
        print(f"\nTop Topics:")
        for topic, count in sorted(topics.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {topic}: {count}")
        
        # Show sample categorized pieces
        print(f"\n=== Sample Categorized Content ===")
        for i, piece in enumerate(categorized_content[:5]):
            print(f"\nPiece {i+1}:")
            print(f"  Page: {piece['page_num']}, Para: {piece['paragraph_num']}")
            print(f"  Content: {piece['content'][:80]}...")
            print(f"  Category: {piece['category']}, Topic: {piece['topic']}")
            print(f"  Statement Type: {piece['primary_statement']}")
            print(f"  Cross-refs: {piece['cross_references']}")
            print(f"  Confidence: {piece['confidence']:.2f}")
        
    except Exception as e:
        print(f"Error testing categorization: {e}")


if __name__ == "__main__":
    test_categorization()