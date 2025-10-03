"""
Universal Enhanced Chunk Selector for ALL Question Types

This module improves chunk selection by:
1. Analyzing question type and context to determine optimal content preferences
2. Using contextual scoring based on question intent (financial statements, notes, revenue, etc.)
3. Avoiding the trap of selecting wrong content types (e.g., notes when questions ask about statements)
4. Providing detailed logging for quality monitoring
"""

import logging
import re
from typing import Dict, List, Set, Tuple, Any

from services.intelligent_chunk_accumulator import IntelligentChunkAccumulator, CategoryAwareContentStorage

logger = logging.getLogger(__name__)

class EnhancedChunkSelector(IntelligentChunkAccumulator):
    """Universal enhanced chunk selector that adapts to ALL question types"""
    
    def __init__(self, storage: CategoryAwareContentStorage):
        super().__init__(storage)
        
        # Financial statement header patterns (prioritized content)
        self.fs_header_patterns = [
            r'consolidated\s+statement\s+of\s+financial\s+position',
            r'statement\s+of\s+financial\s+position',
            r'balance\s+sheet',
            r'consolidated\s+statement\s+of\s+comprehensive\s+income',
            r'statement\s+of\s+comprehensive\s+income',
            r'statement\s+of\s+profit\s+and\s+loss',
            r'profit\s+and\s+loss\s+statement',
            r'income\s+statement',
            r'consolidated\s+statement\s+of\s+cash\s+flows',
            r'statement\s+of\s+cash\s+flows',
            r'cash\s+flow\s+statement',
            r'statement\s+of\s+changes\s+in\s+equity',
            r'consolidated\s+statement\s+of\s+changes\s+in\s+equity'
        ]
        
        # Notes patterns (deprioritized content)
        self.notes_patterns = [
            r'note\s+\d+',
            r'notes\s+to\s+the\s+financial\s+statements',
            r'accounting\s+policies',
            r'significant\s+accounting\s+estimates'
        ]
    
    def enhanced_accumulate_relevant_content(
        self, 
        question: str, 
        document_id: str, 
        max_content_length: int = 4500
    ) -> Dict[str, Any]:
        """
        Enhanced content accumulation with financial statement header priority
        """
        logger.info(f"🔍 Enhanced chunk selection for question: {question[:100]}...")
        
        # Get base relevant chunks using parent method
        base_chunks = self._get_base_relevant_chunks(question, document_id)
        
        if not base_chunks:
            logger.warning("No base chunks found for enhanced selection")
            return {
                'content': '',
                'total_chunks': 0,
                'categories': [],
                'confidence': 0.0,
                'enhanced_selection': True,
                'chunk_details': []
            }
        
        # Apply enhanced scoring
        scored_chunks = self._apply_enhanced_scoring(base_chunks, question)
        
        # Find financial statement header chunks specifically
        fs_header_chunks = self._find_financial_statement_header_chunks(scored_chunks)
        
        # Select best chunks with FS header priority
        selected_chunks = self._select_best_chunks(scored_chunks, fs_header_chunks, max_content_length)
        
        # Build final content
        final_content = self._build_final_content(selected_chunks)
        
        # Calculate confidence and metadata
        confidence = self._calculate_confidence(selected_chunks, fs_header_chunks)
        categories_used = set()
        
        for chunk_data, _ in selected_chunks:
            if chunk_data.get('category'):
                categories_used.add(chunk_data['category'])
        
        # Prepare chunk details for logging
        chunk_details = []
        for chunk_data, score in selected_chunks:
            chunk_details.append({
                'content': chunk_data['content'],
                'score': score,
                'category': chunk_data.get('category', 'unknown'),
                'page': chunk_data.get('page', 'N/A'),
                'header': chunk_data.get('header', 'N/A')
            })
        
        logger.info(f"✅ Enhanced selection complete: {len(selected_chunks)} chunks, {len(fs_header_chunks)} FS headers")
        
        return {
            'content': final_content,
            'total_chunks': len(selected_chunks),
            'categories': list(categories_used),
            'confidence': confidence,
            'enhanced_selection': True,
            'chunk_details': chunk_details
        }
    
    def _get_base_relevant_chunks(self, question: str, document_id: str) -> List[Dict]:
        """Get base relevant chunks using parent accumulator logic"""
        try:
            # Use parent class method to get basic relevant chunks
            keywords = self.extract_question_keywords(question)
            chunks = self.storage.search_relevant_content(
                document_id=document_id,
                keywords=keywords,
                max_chunks=20  # Get more chunks for enhanced selection
            )
            return chunks
        except Exception as e:
            logger.error(f"Error getting base chunks: {e}")
            return []
    
    def _apply_enhanced_scoring(self, chunks: List[Dict], question: str) -> List[Tuple[Dict, float]]:
        """Apply enhanced scoring that prioritizes FS headers over notes"""
        scored_chunks = []
        
        for chunk in chunks:
            content = chunk.get('content', '').lower()
            base_score = 5.0  # Base relevance score
            
            # Boost score for financial statement headers
            fs_header_score = 0
            for pattern in self.fs_header_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    fs_header_score += 10  # Significant boost for FS headers
                    logger.debug(f"FS header pattern matched: {pattern}")
            
            # Penalize notes content (but don't eliminate entirely)
            notes_penalty = 0
            for pattern in self.notes_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    notes_penalty += 8  # Penalty for notes content
                    logger.debug(f"Notes pattern matched: {pattern}")
            
            # Question relevance (keyword matching)
            question_keywords = question.lower().split()
            keyword_matches = sum(1 for keyword in question_keywords if keyword in content)
            keyword_score = keyword_matches * 2
            
            # Calculate final score
            final_score = base_score + fs_header_score + keyword_score - notes_penalty
            scored_chunks.append((chunk, final_score))
        
        # Sort by score (highest first)
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks
    
    def _find_financial_statement_header_chunks(self, scored_chunks: List[Tuple[Dict, float]]) -> List[Tuple[Dict, float]]:
        """Identify chunks that contain financial statement headers"""
        fs_header_chunks = []
        
        for chunk_data, score in scored_chunks:
            content = chunk_data.get('content', '').lower()
            
            # Check if this chunk contains FS header patterns
            has_fs_header = any(
                re.search(pattern, content, re.IGNORECASE) 
                for pattern in self.fs_header_patterns
            )
            
            if has_fs_header:
                fs_header_chunks.append((chunk_data, score))
                logger.debug(f"Found FS header chunk with score {score}")
        
        return fs_header_chunks
    
    def _select_best_chunks(
        self, 
        scored_chunks: List[Tuple[Dict, float]], 
        fs_header_chunks: List[Tuple[Dict, float]], 
        max_length: int
    ) -> List[Tuple[Dict, float]]:
        """Select best chunks with preference for FS headers"""
        selected = []
        current_length = 0
        
        # First, include all FS header chunks (they're most important)
        for chunk_data, score in fs_header_chunks:
            content_length = len(chunk_data.get('content', ''))
            if current_length + content_length <= max_length:
                selected.append((chunk_data, score))
                current_length += content_length
                logger.debug(f"Selected FS header chunk (score: {score}, length: {content_length})")
        
        # Then fill remaining space with highest-scoring non-FS chunks
        for chunk_data, score in scored_chunks:
            if (chunk_data, score) in selected:
                continue  # Already selected as FS header
            
            content_length = len(chunk_data.get('content', ''))
            if current_length + content_length <= max_length:
                selected.append((chunk_data, score))
                current_length += content_length
                logger.debug(f"Selected additional chunk (score: {score}, length: {content_length})")
            else:
                logger.debug(f"Skipped chunk due to length limit (score: {score}, would exceed by {current_length + content_length - max_length})")
        
        return selected
    
    def _build_final_content(self, selected_chunks: List[Tuple[Dict, float]]) -> str:
        """Build final content from selected chunks"""
        content_parts = []
        
        for chunk_data, score in selected_chunks:
            content = chunk_data.get('content', '').strip()
            if content:
                # Add some context about the chunk
                page = chunk_data.get('page', 'N/A')
                header = chunk_data.get('header', '')
                
                if header:
                    content_parts.append(f"[Page {page} - {header}]\n{content}")
                else:
                    content_parts.append(f"[Page {page}]\n{content}")
        
        return '\n\n---\n\n'.join(content_parts)
    
    def _calculate_confidence(
        self, 
        selected_chunks: List[Tuple[Dict, float]], 
        fs_header_chunks: List[Tuple[Dict, float]]
    ) -> float:
        """Calculate confidence score based on selection quality"""
        if not selected_chunks:
            return 0.0
        
        # Base confidence from average chunk scores
        avg_score = sum(score for _, score in selected_chunks) / len(selected_chunks)
        base_confidence = min(avg_score / 20.0, 1.0)  # Normalize to 0-1
        
        # Boost confidence if we have FS header chunks
        fs_boost = min(len(fs_header_chunks) * 0.2, 0.4)  # Up to 40% boost
        
        # Boost confidence if we have good content variety
        categories = set(chunk.get('category', 'unknown') for chunk, _ in selected_chunks)
        variety_boost = min(len(categories) * 0.1, 0.2)  # Up to 20% boost
        
        final_confidence = min(base_confidence + fs_boost + variety_boost, 1.0)
        return final_confidence