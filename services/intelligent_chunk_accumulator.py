"""
Intelligent Chunk Accumulator Module
Provides smart content retrieval and categorization for compliance analysis
"""

import logging
import sqlite3
import os
import json
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class CategoryAwareContentStorage:
    """SQLite-based storage for categorized content chunks"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            # Use backend directory for database
            backend_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            db_path = str(backend_dir / "categorized_content.db")
        
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the categorized content database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS categorized_content (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        document_id TEXT NOT NULL,
                        content_chunk TEXT NOT NULL,
                        category TEXT NOT NULL,
                        subcategory TEXT,
                        confidence_score REAL DEFAULT 0.0,
                        keywords TEXT,  -- JSON array of extracted keywords
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX(document_id),
                        INDEX(category),
                        INDEX(confidence_score)
                    )
                ''')
                conn.commit()
                logger.info(f"✅ CategoryAwareContentStorage database initialized: {self.db_path}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize CategoryAwareContentStorage database: {e}")
            raise
    
    def store_categorized_chunk(self, document_id: str, chunk: str, category: str, 
                               subcategory: str = None, confidence: float = 0.0, 
                               keywords: List[str] = None) -> bool:
        """Store a categorized content chunk"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                keywords_json = json.dumps(keywords or [])
                cursor.execute('''
                    INSERT INTO categorized_content 
                    (document_id, content_chunk, category, subcategory, confidence_score, keywords)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (document_id, chunk, category, subcategory, confidence, keywords_json))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"❌ Failed to store categorized chunk: {e}")
            return False
    
    def get_content_by_category(self, document_id: str, category: str, 
                               max_chunks: int = 10) -> List[Dict[str, Any]]:
        """Retrieve content chunks by category"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT content_chunk, category, subcategory, confidence_score, keywords
                    FROM categorized_content 
                    WHERE document_id = ? AND category = ?
                    ORDER BY confidence_score DESC, id DESC
                    LIMIT ?
                ''', (document_id, category, max_chunks))
                
                results = []
                for row in cursor.fetchall():
                    chunk, cat, subcat, confidence, keywords_json = row
                    try:
                        keywords = json.loads(keywords_json or '[]')
                    except:
                        keywords = []
                    
                    results.append({
                        'content': chunk,
                        'category': cat,
                        'subcategory': subcat,
                        'confidence': confidence,
                        'keywords': keywords
                    })
                
                return results
        except Exception as e:
            logger.error(f"❌ Failed to retrieve content by category: {e}")
            return []
    
    def search_relevant_content(self, document_id: str, keywords: List[str], 
                               max_chunks: int = 5) -> List[Dict[str, Any]]:
        """Search for content chunks containing relevant keywords"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build search query for keywords
                keyword_conditions = []
                params = [document_id]
                
                for keyword in keywords:
                    keyword_conditions.append("(content_chunk LIKE ? OR keywords LIKE ?)")
                    params.extend([f'%{keyword}%', f'%{keyword}%'])
                
                if not keyword_conditions:
                    # Return high-confidence chunks if no keywords
                    cursor.execute('''
                        SELECT content_chunk, category, subcategory, confidence_score, keywords
                        FROM categorized_content 
                        WHERE document_id = ?
                        ORDER BY confidence_score DESC, id DESC
                        LIMIT ?
                    ''', (document_id, max_chunks))
                else:
                    keyword_query = " OR ".join(keyword_conditions)
                    params.append(max_chunks)
                    
                    cursor.execute(f'''
                        SELECT content_chunk, category, subcategory, confidence_score, keywords
                        FROM categorized_content 
                        WHERE document_id = ? AND ({keyword_query})
                        ORDER BY confidence_score DESC, id DESC
                        LIMIT ?
                    ''', params)
                
                results = []
                for row in cursor.fetchall():
                    chunk, cat, subcat, confidence, keywords_json = row
                    try:
                        keywords = json.loads(keywords_json or '[]')
                    except:
                        keywords = []
                    
                    results.append({
                        'content': chunk,
                        'category': cat,
                        'subcategory': subcat,
                        'confidence': confidence,
                        'keywords': keywords
                    })
                
                return results
        except Exception as e:
            logger.error(f"❌ Failed to search relevant content: {e}")
            return []
    
    def get_document_summary(self, document_id: str) -> Dict[str, Any]:
        """Get summary statistics for a document's categorized content"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get total chunks and categories
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_chunks,
                        COUNT(DISTINCT category) as total_categories,
                        AVG(confidence_score) as avg_confidence
                    FROM categorized_content 
                    WHERE document_id = ?
                ''', (document_id,))
                
                stats = cursor.fetchone()
                total_chunks, total_categories, avg_confidence = stats or (0, 0, 0.0)
                
                # Get category distribution
                cursor.execute('''
                    SELECT category, COUNT(*) as count
                    FROM categorized_content 
                    WHERE document_id = ?
                    GROUP BY category
                    ORDER BY count DESC
                ''', (document_id,))
                
                category_dist = dict(cursor.fetchall())
                
                return {
                    'total_chunks': total_chunks,
                    'total_categories': total_categories,
                    'avg_confidence': avg_confidence or 0.0,
                    'category_distribution': category_dist
                }
        except Exception as e:
            logger.error(f"❌ Failed to get document summary: {e}")
            return {'total_chunks': 0, 'total_categories': 0, 'avg_confidence': 0.0, 'category_distribution': {}}


class IntelligentChunkAccumulator:
    """Intelligent content accumulator for smart compliance analysis"""
    
    def __init__(self, storage: CategoryAwareContentStorage):
        self.storage = storage
        self.question_keywords = {}  # Cache for extracted keywords
    
    def extract_question_keywords(self, question: str) -> List[str]:
        """Extract relevant keywords from a compliance question"""
        if question in self.question_keywords:
            return self.question_keywords[question]
        
        # Simple keyword extraction (can be enhanced with NLP)
        import re
        
        # Remove common compliance question words
        stop_words = {
            'does', 'the', 'entity', 'disclose', 'present', 'report', 'include', 'provide',
            'are', 'is', 'has', 'have', 'been', 'being', 'will', 'shall', 'should', 'must',
            'and', 'or', 'but', 'if', 'when', 'where', 'how', 'what', 'which', 'that',
            'for', 'in', 'on', 'at', 'to', 'from', 'by', 'with', 'as', 'an', 'a'
        }
        
        # Extract words (3+ chars, not stop words)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', question.lower())
        keywords = [w for w in words if w not in stop_words]
        
        # Take most relevant keywords (limit to avoid noise)
        keywords = list(set(keywords))[:10]
        
        self.question_keywords[question] = keywords
        logger.debug(f"🔍 Extracted keywords from question: {keywords}")
        return keywords
    
    def accumulate_relevant_content(self, question: str, document_id: str, 
                                  max_content_length: int = 3000) -> Dict[str, Any]:
        """Accumulate relevant content for a compliance question"""
        try:
            logger.info(f"🎯 Accumulating content for question: {question[:50]}...")
            
            # Extract keywords from question
            keywords = self.extract_question_keywords(question)
            logger.info(f"🔍 CHUNK QUALITY: Question keywords extracted: {keywords}")
            
            # Search for relevant content
            relevant_chunks = self.storage.search_relevant_content(
                document_id=document_id,
                keywords=keywords,
                max_chunks=10
            )
            
            # DETAILED CHUNK QUALITY LOGGING
            logger.info(f"📊 CHUNK QUALITY: Found {len(relevant_chunks)} relevant chunks for question")
            for i, chunk in enumerate(relevant_chunks):
                logger.info(f"📄 CHUNK {i+1}: Category='{chunk['category']}', "
                           f"Subcategory='{chunk.get('subcategory', 'N/A')}', "
                           f"Confidence={chunk['confidence']:.3f}, "
                           f"Length={len(chunk['content'])} chars, "
                           f"Keywords={chunk.get('keywords', [])}")
                logger.info(f"📄 FULL CANDIDATE CHUNK #{i+1} CONTENT:\n{chunk['content']}\n" + "="*100)
            
            if not relevant_chunks:
                logger.warning(f"⚠️  No relevant content found for document {document_id}")
                return {
                    'content': '',
                    'total_chunks': 0,
                    'confidence': 0.0,
                    'categories': [],
                    'keywords_used': keywords
                }
            
            # Combine chunks up to max_content_length
            combined_content = ""
            used_chunks = 0
            categories = set()
            total_confidence = 0.0
            
            logger.info(f"🔄 CHUNK SELECTION: Starting combination of {len(relevant_chunks)} chunks (max_length: {max_content_length})")
            
            for i, chunk in enumerate(relevant_chunks):
                chunk_content = chunk['content']
                
                # Check if adding this chunk would exceed limit
                if len(combined_content + chunk_content) > max_content_length:
                    # Try to add partial content
                    remaining_space = max_content_length - len(combined_content)
                    if remaining_space > 100:  # Only add if meaningful space left
                        truncated_content = chunk_content[:remaining_space] + "..."
                        combined_content += truncated_content
                        logger.info(f"✂️  CHUNK {i+1}: TRUNCATED to fit limit (added {remaining_space} chars)")
                        logger.info(f"📄 TRUNCATED CHUNK CONTENT #{i+1}:\n{truncated_content}\n" + "="*100)
                    else:
                        logger.info(f"🚫 CHUNK {i+1}: SKIPPED (would exceed limit)")
                    break
                
                combined_content += chunk_content + "\n\n"
                used_chunks += 1
                categories.add(chunk['category'])
                total_confidence += chunk['confidence']
                logger.info(f"✅ CHUNK {i+1}: SELECTED (confidence: {chunk['confidence']:.3f}, category: {chunk['category']})")
                logger.info(f"📄 FULL CHUNK CONTENT #{i+1}:\n{chunk_content}\n" + "="*100)
            
            avg_confidence = total_confidence / used_chunks if used_chunks > 0 else 0.0
            
            logger.info(f"🎯 CHUNK SELECTION FINAL SUMMARY:")
            logger.info(f"   📊 Total chunks selected: {used_chunks}/{len(relevant_chunks)}")
            logger.info(f"   📏 Final content length: {len(combined_content)} chars")
            logger.info(f"   🎯 Average confidence: {avg_confidence:.3f}")
            logger.info(f"   🏷️  Categories covered: {list(categories)}")
            logger.info(f"   🔍 Keywords used: {keywords}")
            
            return {
                'content': combined_content.strip(),
                'total_chunks': used_chunks,
                'confidence': avg_confidence,
                'categories': list(categories),
                'keywords_used': keywords
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to accumulate relevant content: {e}")
            return {
                'content': '',
                'total_chunks': 0,
                'confidence': 0.0,
                'categories': [],
                'keywords_used': []
            }
    
    def get_document_content_summary(self, document_id: str) -> Dict[str, Any]:
        """Get a summary of available content for a document"""
        return self.storage.get_document_summary(document_id)


# Initialize global storage instance (can be used across the application)
_global_storage = None

def get_global_storage() -> CategoryAwareContentStorage:
    """Get or create the global CategoryAwareContentStorage instance"""
    global _global_storage
    if _global_storage is None:
        _global_storage = CategoryAwareContentStorage()
    return _global_storage


def initialize_smart_categorization() -> bool:
    """Initialize the smart categorization system"""
    try:
        storage = get_global_storage()
        logger.info("✅ Smart categorization system initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize smart categorization system: {e}")
        return False