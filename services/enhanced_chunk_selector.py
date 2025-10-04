"""
Enhanced Financial Statement Chunk Selector

Fixes the chunk quality issue by prioritizing financial statement headers and titles
over notes and detailed content when questions ask about financial statement identification.
"""

import logging
import re
from typing import Dict, List, Any, Tuple
from services.intelligent_chunk_accumulator import CategoryAwareContentStorage, IntelligentChunkAccumulator

logger = logging.getLogger(__name__)


class EnhancedChunkSelector(IntelligentChunkAccumulator):
    """Enhanced chunk selector with financial statement priority"""
    
    def __init__(self, storage: CategoryAwareContentStorage):
        super().__init__(storage)
        
        # Financial statement title patterns (high priority)
        self.financial_statement_patterns = [
            r'consolidated\s+statement\s+of\s+financial\s+position',
            r'statement\s+of\s+financial\s+position',
            r'consolidated\s+balance\s+sheet',
            r'balance\s+sheet',
            r'consolidated\s+statement\s+of\s+profit\s+or\s+loss',
            r'statement\s+of\s+profit\s+or\s+loss',
            r'consolidated\s+statement\s+of\s+comprehensive\s+income',
            r'statement\s+of\s+comprehensive\s+income',
            r'consolidated\s+statement\s+of\s+changes\s+in\s+equity',
            r'statement\s+of\s+changes\s+in\s+equity',
            r'consolidated\s+statement\s+of\s+cash\s+flows',
            r'statement\s+of\s+cash\s+flows',
            r'cash\s+flow\s+statement'
        ]
        
        # Question patterns that need financial statement headers
        self.header_priority_questions = [
            r'financial\s+statements?\s+identified\s+clearly',
            r'using\s+an?\s+unambiguous\s+title',
            r'distinguished\s+from\s+other\s+information',
            r'complete\s+set\s+of\s+financial\s+statements',
            r'present\s+.*\s+statement\s+of\s+financial\s+position',
            r'present\s+.*\s+statement\s+of\s+profit',
            r'present\s+.*\s+statement\s+of\s+comprehensive',
            r'present\s+.*\s+statement\s+of\s+changes',
            r'present\s+.*\s+statement\s+of\s+cash\s+flows'
        ]

    def enhanced_accumulate_relevant_content(self, question: str, document_id: str, 
                                           max_content_length: int = 4500) -> Dict[str, Any]:
        """Enhanced content accumulation with financial statement priority"""
        
        logger.info(f"🎯 ENHANCED CHUNK SELECTION for question: {question[:50]}...")
        
        # Check if this question needs financial statement headers
        needs_headers = self._question_needs_statement_headers(question)
        logger.info(f"🏦 Question needs FS headers: {needs_headers}")
        
        if needs_headers:
            # First try to find chunks with financial statement titles
            header_chunks = self._find_financial_statement_header_chunks(document_id)
            logger.info(f"📋 Found {len(header_chunks)} header chunks")
            
            if header_chunks:
                # Use header chunks with high priority
                return self._create_header_focused_response(header_chunks, question, document_id, max_content_length)
        
        # Fallback to original method
        logger.info(f"🔄 Using standard chunk selection")
        return self.accumulate_relevant_content(question, document_id, max_content_length)
    
    def _question_needs_statement_headers(self, question: str) -> bool:
        """Check if question is asking about financial statement identification"""
        question_lower = question.lower()
        
        for pattern in self.header_priority_questions:
            if re.search(pattern, question_lower):
                logger.info(f"✅ HEADER PRIORITY: Question matches pattern '{pattern}'")
                return True
        
        return False
    
    def _find_financial_statement_header_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Find chunks containing financial statement headers/titles"""
        try:
            # Get all chunks for this document
            import sqlite3
            with sqlite3.connect(self.storage.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT content_chunk, category, subcategory, confidence_score, keywords
                    FROM categorized_content 
                    WHERE document_id = ?
                    ORDER BY confidence_score DESC
                ''', (document_id,))
                
                all_chunks = []
                for row in cursor.fetchall():
                    chunk, cat, subcat, confidence, keywords_json = row
                    try:
                        import json
                        keywords = json.loads(keywords_json or '[]')
                    except:
                        keywords = []
                    
                    all_chunks.append({
                        'content': chunk,
                        'category': cat,
                        'subcategory': subcat,
                        'confidence': confidence,
                        'keywords': keywords
                    })
            
            # Score chunks based on financial statement header content
            header_chunks = []
            
            for chunk in all_chunks:
                content_lower = chunk['content'].lower()
                header_score = 0
                found_patterns = []
                
                # Check for financial statement title patterns
                for pattern in self.financial_statement_patterns:
                    if re.search(pattern, content_lower):
                        header_score += 10  # High score for FS titles
                        found_patterns.append(pattern)
                
                # Bonus for contents/index pages
                if any(word in content_lower for word in ['contents', 'pages', 'directors\' report', 'auditor\'s report']):
                    header_score += 5
                
                # Penalty for notes (we don't want note content for header questions)
                if re.search(r'note\s+\d+', content_lower) or 'notes to the' in content_lower:
                    header_score -= 8
                
                # Penalty for detailed financial data (numbers, tables)
                number_density = len(re.findall(r'\b\d{1,3}(?:,\d{3})*\b', chunk['content'])) / max(len(chunk['content']), 1)
                if number_density > 0.01:  # More than 1% numbers
                    header_score -= 3
                
                if header_score > 0:
                    chunk_with_score = chunk.copy()
                    chunk_with_score['header_score'] = header_score
                    chunk_with_score['found_patterns'] = found_patterns
                    header_chunks.append(chunk_with_score)
                    
                    logger.info(f"📋 HEADER CHUNK FOUND:")
                    logger.info(f"   Score: {header_score}")
                    logger.info(f"   Patterns: {found_patterns}")
                    logger.info(f"   Category: {chunk['category']}")
                    logger.info(f"   Preview: {chunk['content'][:150]}...")
            
            # Sort by header score (descending)
            header_chunks.sort(key=lambda x: x['header_score'], reverse=True)
            
            return header_chunks
            
        except Exception as e:
            logger.error(f"❌ Failed to find header chunks: {e}")
            return []
    
    def _create_header_focused_response(self, header_chunks: List[Dict[str, Any]], 
                                      question: str, document_id: str, 
                                      max_content_length: int) -> Dict[str, Any]:
        """Create response focused on financial statement headers"""
        
        combined_content = ""
        used_chunks = 0
        categories = set()
        total_confidence = 0.0
        total_header_score = 0.0
        all_patterns = []
        
        logger.info(f"🏦 HEADER FOCUSED SELECTION: Processing {len(header_chunks)} header chunks")
        
        for i, chunk in enumerate(header_chunks):
            chunk_content = chunk['content']
            
            # Check if adding this chunk would exceed limit
            if len(combined_content + chunk_content) > max_content_length:
                # Try to add partial content
                remaining_space = max_content_length - len(combined_content)
                if remaining_space > 200:  # Only add if meaningful space left
                    truncated_content = chunk_content[:remaining_space] + "..."
                    combined_content += truncated_content + "\n\n"
                    logger.info(f"✂️  HEADER CHUNK {i+1}: TRUNCATED to fit limit")
                else:
                    logger.info(f"🚫 HEADER CHUNK {i+1}: SKIPPED (would exceed limit)")
                break
            
            combined_content += chunk_content + "\n\n"
            used_chunks += 1
            categories.add(chunk['category'])
            total_confidence += chunk['confidence']
            total_header_score += chunk.get('header_score', 0)
            all_patterns.extend(chunk.get('found_patterns', []))
            
            logger.info(f"✅ HEADER CHUNK {i+1}: SELECTED")
            logger.info(f"   Header score: {chunk.get('header_score', 0)}")
            logger.info(f"   Confidence: {chunk['confidence']:.3f}")
            logger.info(f"   Category: {chunk['category']}")
            logger.info(f"   Patterns found: {chunk.get('found_patterns', [])}")
            logger.info(f"   Content: {chunk_content[:200]}...")
        
        avg_confidence = total_confidence / used_chunks if used_chunks > 0 else 0.0
        avg_header_score = total_header_score / used_chunks if used_chunks > 0 else 0.0
        
        logger.info(f"🎯 HEADER SELECTION FINAL SUMMARY:")
        logger.info(f"   📊 Header chunks selected: {used_chunks}/{len(header_chunks)}")
        logger.info(f"   📏 Final content length: {len(combined_content)} chars")
        logger.info(f"   🎯 Average confidence: {avg_confidence:.3f}")
        logger.info(f"   🏦 Average header score: {avg_header_score:.1f}")
        logger.info(f"   🏷️  Categories covered: {list(categories)}")
        logger.info(f"   📋 FS patterns found: {list(set(all_patterns))}")
        
        return {
            'content': combined_content.strip(),
            'total_chunks': used_chunks,
            'confidence': avg_confidence,
            'categories': list(categories),
            'keywords_used': self.extract_question_keywords(question),
            'enhancement_used': 'financial_statement_headers',
            'header_score': avg_header_score,
            'financial_statement_patterns_found': list(set(all_patterns))
        }


# Create enhanced global instance
def get_enhanced_chunk_selector(storage: CategoryAwareContentStorage) -> EnhancedChunkSelector:
    """Get enhanced chunk selector instance"""
    return EnhancedChunkSelector(storage)