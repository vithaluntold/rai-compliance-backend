"""
Category-Aware Content Storage and Intelligent Chunk Accumulation System
Stores tagged content and retrieves only matching categories for precise question answering
"""

import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import sqlite3
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class TaggedContent:
    """Structure for storing categorized content with citation metadata"""
    content_id: str
    document_id: str
    content: str
    extended_context: str
    
    # Citation metadata (crucial to preserve)
    page_num: int
    paragraph_num: int
    line_range: List[int]
    char_position: int
    
    # Cross-references (crucial for citations)
    cross_references: List[str]
    note_numbers: List[str]
    table_references: List[str]
    
    # Category tagging (Category → Topic → Requirement Type)
    category: str
    topic: str
    requirement_type: str
    confidence: float
    
    # Statement context
    statement_types: List[str]
    primary_statement: str
    
    # Search optimization
    search_terms: List[str]
    semantic_tags: List[str]
    content_type: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'content_id': self.content_id,
            'document_id': self.document_id,
            'content': self.content,
            'extended_context': self.extended_context,
            'page_num': self.page_num,
            'paragraph_num': self.paragraph_num,
            'line_range': self.line_range,
            'char_position': self.char_position,
            'cross_references': self.cross_references,
            'note_numbers': self.note_numbers,
            'table_references': self.table_references,
            'category': self.category,
            'topic': self.topic,
            'requirement_type': self.requirement_type,
            'confidence': self.confidence,
            'statement_types': self.statement_types,
            'primary_statement': self.primary_statement,
            'search_terms': self.search_terms,
            'semantic_tags': self.semantic_tags,
            'content_type': self.content_type
        }


class CategoryAwareContentStorage:
    """
    Storage system for categorized content with fast category-based retrieval
    """
    
    def __init__(self, db_path: str = "categorized_content.db"):
        self.db_path = db_path
        self.setup_database()
        
    def setup_database(self):
        """Create database tables for categorized content"""
        with sqlite3.connect(self.db_path) as conn:
            # Main content table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS categorized_content (
                    content_id TEXT PRIMARY KEY,
                    document_id TEXT,
                    content TEXT,
                    extended_context TEXT,
                    page_num INTEGER,
                    paragraph_num INTEGER,
                    line_range TEXT,
                    char_position INTEGER,
                    cross_references TEXT,
                    note_numbers TEXT,
                    table_references TEXT,
                    category TEXT,
                    topic TEXT,
                    requirement_type TEXT,
                    confidence REAL,
                    statement_types TEXT,
                    primary_statement TEXT,
                    search_terms TEXT,
                    semantic_tags TEXT,
                    content_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for fast category-based lookups
            conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON categorized_content (category)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_topic ON categorized_content (topic)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_category_topic ON categorized_content (category, topic)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_document ON categorized_content (document_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_statement_type ON categorized_content (primary_statement)")
            
            logger.info("Database setup completed")
    
    def store_categorized_content(self, categorized_content: List[Dict[str, Any]], document_id: str):
        """Store categorized content pieces in database"""
        logger.info(f"STORAGE STEP 1: Starting storage of {len(categorized_content)} categorized content pieces for document {document_id}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                logger.info(f"STORAGE STEP 2: Database connection established, processing content pieces")
                
                stored_count = 0
                for piece in categorized_content:
                    # Generate unique content ID
                    content_id = self._generate_content_id(document_id, piece)
                    
                    # Prepare data for storage
                    data = (
                        content_id,
                        document_id,
                        piece['content'],
                        piece['extended_context'],
                        piece['page_num'],
                        piece['paragraph_num'],
                        json.dumps(piece['line_range']),
                        piece['char_position'],
                        json.dumps(piece['cross_references']),
                        json.dumps(piece['note_numbers']),
                        json.dumps(piece['table_references']),
                        piece['category'],
                        piece['topic'],
                        piece['requirement_type'],
                        piece['confidence'],
                        json.dumps(piece['statement_types']),
                        piece['primary_statement'],
                        json.dumps(piece['search_terms']),
                        json.dumps(piece['semantic_tags']),
                        piece['content_type']
                    )
                    
                    # Insert with REPLACE to handle duplicates
                    conn.execute("""
                        REPLACE INTO categorized_content 
                        (content_id, document_id, content, extended_context, page_num, paragraph_num,
                         line_range, char_position, cross_references, note_numbers, table_references,
                         category, topic, requirement_type, confidence, statement_types, primary_statement,
                         search_terms, semantic_tags, content_type)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, data)
                    stored_count += 1
                    
                    # Log progress every 100 pieces
                    if stored_count % 100 == 0:
                        logger.info(f"⏳ STORAGE PROGRESS: Stored {stored_count}/{len(categorized_content)} content pieces")
                
                logger.info(f"STORAGE STEP 3: Committing transaction to database")
                conn.commit()
                logger.info(f"✅ STORAGE COMPLETE: Successfully stored {len(categorized_content)} content pieces for document {document_id}")
                
        except Exception as e:
            logger.error(f"❌ STORAGE FAILED: Error storing categorized content: {e}")
            raise
    
    def _generate_content_id(self, document_id: str, piece: Dict[str, Any]) -> str:
        """Generate unique content ID based on document and position"""
        content_hash = hashlib.md5(
            f"{document_id}_{piece['page_num']}_{piece['paragraph_num']}_{piece['content'][:50]}"
            .encode()
        ).hexdigest()[:12]
        return f"{document_id}_{content_hash}"


class IntelligentChunkAccumulator:
    """
    Intelligent system that accumulates only relevant content based on question categories
    """
    
    def __init__(self, storage: CategoryAwareContentStorage):
        self.storage = storage
        self.load_question_categorizer()
    
    def create_contextual_chunks(
        self, 
        categorized_content: List[Dict[str, Any]], 
        document_id: str
    ) -> List[Dict[str, Any]]:
        """
        Create intelligent chunks from categorized content for document processing
        This is called by DocumentChunker during initial document processing
        """
        logger.info(f"CHUNK CREATION: Creating intelligent chunks from {len(categorized_content)} categorized pieces for document {document_id}")
        
        try:
            # Store categorized content in database for future retrieval
            logger.info(f"CHUNK CREATION STEP 1: Storing categorized content in database")
            self.storage.store_categorized_content(categorized_content, document_id)
            
            # Group content into logical chunks based on categories and topics
            logger.info(f"CHUNK CREATION STEP 2: Grouping content into logical chunks")
            grouped_chunks = self._group_content_into_chunks(categorized_content)
            
            # Convert to final chunk format
            logger.info(f"CHUNK CREATION STEP 3: Converting to final chunk format")
            final_chunks = self._convert_to_chunk_format(grouped_chunks, document_id)
            
            logger.info(f"✅ CHUNK CREATION COMPLETE: Created {len(final_chunks)} intelligent chunks from {len(categorized_content)} content pieces")
            return final_chunks
            
        except Exception as e:
            logger.error(f"❌ CHUNK CREATION FAILED: Error creating contextual chunks: {e}")
            # Return basic chunks as fallback
            return self._create_fallback_chunks(categorized_content, document_id)
    
    def _group_content_into_chunks(self, categorized_content: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Group categorized content into logical chunks based on:
        - Same category + topic
        - Consecutive pages
        - Maximum chunk size limits
        """
        if not categorized_content:
            return []
        
        # Sort by page and paragraph for logical flow
        sorted_content = sorted(categorized_content, key=lambda x: (x['page_num'], x['paragraph_num']))
        
        chunks = []
        current_chunk = []
        current_length = 0
        max_chunk_length = 1500  # Maximum characters per chunk
        
        for piece in sorted_content:
            piece_length = len(piece['content'])
            
            # Start new chunk if:
            # 1. Current chunk would exceed max length
            # 2. Category/topic changes significantly
            # 3. Page gap is too large (more than 2 pages)
            should_start_new = (
                current_length + piece_length > max_chunk_length or
                (current_chunk and self._should_separate_chunks(current_chunk[-1], piece)) or
                (current_chunk and piece['page_num'] - current_chunk[-1]['page_num'] > 2)
            )
            
            if should_start_new and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_length = 0
            
            current_chunk.append(piece)
            current_length += piece_length
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        logger.info(f"Grouped {len(categorized_content)} pieces into {len(chunks)} logical chunks")
        return chunks
    
    def _should_separate_chunks(self, prev_piece: Dict[str, Any], current_piece: Dict[str, Any]) -> bool:
        """Determine if two pieces should be in separate chunks"""
        # Separate if category changes
        if prev_piece['category'] != current_piece['category']:
            return True
        
        # Separate if topic changes and we have enough content
        if (prev_piece['topic'] != current_piece['topic'] and 
            prev_piece['category'] != 'METADATA_EXTRACTION'):
            return True
        
        # Separate if statement type changes significantly
        if (prev_piece['primary_statement'] != current_piece['primary_statement'] and
            prev_piece['primary_statement'] != 'UNKNOWN' and
            current_piece['primary_statement'] != 'UNKNOWN'):
            return True
        
        return False
    
    def _convert_to_chunk_format(self, grouped_chunks: List[List[Dict[str, Any]]], document_id: str) -> List[Dict[str, Any]]:
        """Convert grouped content into final chunk format"""
        final_chunks = []
        
        for chunk_idx, content_group in enumerate(grouped_chunks):
            if not content_group:
                continue
            
            # Combine content from all pieces in group
            combined_content = []
            all_cross_refs = []
            all_note_numbers = []
            all_table_refs = []
            
            # Aggregate metadata
            page_nums = set()
            categories = set()
            topics = set()
            statement_types = set()
            
            for piece in content_group:
                combined_content.append(piece['content'])
                all_cross_refs.extend(piece.get('cross_references', []))
                all_note_numbers.extend(piece.get('note_numbers', []))
                all_table_refs.extend(piece.get('table_references', []))
                
                page_nums.add(piece['page_num'])
                categories.add(piece['category'])
                topics.add(piece['topic'])
                statement_types.add(piece['primary_statement'])
            
            # Determine chunk metadata
            primary_category = max(categories, key=lambda cat: sum(1 for p in content_group if p['category'] == cat))
            primary_topic = max(topics, key=lambda topic: sum(1 for p in content_group if p['topic'] == topic))
            primary_statement = max(statement_types, key=lambda stmt: sum(1 for p in content_group if p['primary_statement'] == stmt))
            
            # Create final chunk
            chunk_text = ' '.join(combined_content)
            final_chunk = {
                'chunk_index': chunk_idx + 1,  # Start from 1 (metadata is 0)
                'page': min(page_nums),
                'page_no': f"{min(page_nums)}-{max(page_nums)}" if len(page_nums) > 1 else str(min(page_nums)),
                'text': chunk_text,
                'length': len(chunk_text),
                'chunk_type': 'intelligent',
                'category': primary_category,
                'topic': primary_topic,
                'statement_type': primary_statement,
                'cross_references': list(set(all_cross_refs)),
                'note_numbers': list(set(all_note_numbers)),
                'table_references': list(set(all_table_refs)),
                'content_pieces': len(content_group),
                'confidence': sum(p.get('confidence', 0) for p in content_group) / len(content_group)
            }
            
            final_chunks.append(final_chunk)
        
        return final_chunks
    
    def _create_fallback_chunks(self, categorized_content: List[Dict[str, Any]], document_id: str) -> List[Dict[str, Any]]:
        """Create basic chunks if intelligent chunking fails"""
        logger.warning(f"Creating fallback chunks for {len(categorized_content)} content pieces")
        
        chunks = []
        for idx, piece in enumerate(categorized_content[:20]):  # Limit to 20 pieces
            chunk = {
                'chunk_index': idx + 1,
                'page': piece.get('page_num', 0),
                'page_no': str(piece.get('page_num', 0)),
                'text': piece.get('content', ''),
                'length': len(piece.get('content', '')),
                'chunk_type': 'fallback',
                'category': piece.get('category', 'UNCATEGORIZED'),
                'topic': piece.get('topic', 'GENERAL'),
                'statement_type': piece.get('primary_statement', 'UNKNOWN')
            }
            chunks.append(chunk)
        
        return chunks
        
    def load_question_categorizer(self):
        """Load the existing question categorization system"""
        try:
            with open('categorized_questions/categorized_questions.json', 'r', encoding='utf-8') as f:
                self.categorized_questions = json.load(f)
            logger.info("Loaded question categorization system")
        except FileNotFoundError:
            logger.error("Question categorization file not found")
            self.categorized_questions = []
    
    def accumulate_relevant_content(
        self, 
        question: str, 
        document_id: str,
        max_content_length: int = 3000  # Increased from 1000 to 3000
    ) -> Dict[str, Any]:
        """
        Accumulate only content that matches the question's categories
        Returns targeted content with preserved citation metadata
        """
        logger.info(f"ACCUMULATOR STEP 1: Starting content accumulation for question: {question[:50]}...")
        
        try:
            # Step 1: Classify the incoming question
            logger.info(f"ACCUMULATOR STEP 2: Classifying incoming question")
            question_categories = self._classify_question(question)
            logger.info(f"✅ ACCUMULATOR STEP 2 COMPLETE: Question classified as Category: {question_categories['category']}, Topic: {question_categories['topic']}")
            
            # Step 2: Retrieve matching content from storage
            logger.info(f"ACCUMULATOR STEP 3: Retrieving matching content from storage")
            matching_content = self._retrieve_matching_content(document_id, question_categories)
            logger.info(f"✅ ACCUMULATOR STEP 3 COMPLETE: Found {len(matching_content)} matching content pieces")
            
            # Step 3: Rank and select best content pieces
            logger.info(f"ACCUMULATOR STEP 4: Ranking and selecting best content pieces (max length: {max_content_length})")
            selected_content = self._rank_and_select_content(
                matching_content, question, max_content_length
            )
            logger.info(f"✅ ACCUMULATOR STEP 4 COMPLETE: Selected {len(selected_content)} top-ranked content pieces")
            
            # Step 4: Accumulate into final chunk with citations
            logger.info(f"ACCUMULATOR STEP 5: Accumulating final chunk with citations")
            accumulated_chunk = self._accumulate_into_chunk(selected_content, question_categories)
            logger.info(f"✅ ACCUMULATOR STEP 5 COMPLETE: Generated chunk with {len(accumulated_chunk['content'])} characters and {len(accumulated_chunk['citations'])} citations")
            
            logger.info(f"✅ ACCUMULATOR COMPLETE: Successfully accumulated relevant content for question")
            return accumulated_chunk
            
        except Exception as e:
            logger.error(f"❌ ACCUMULATOR FAILED: Error accumulating content: {e}")
            return {
                'status': 'error',
                'content': '',
                'citations': [],
                'total_pieces': 0,
                'confidence': 0.0,
                'category_match': {},
                'total_length': 0
            }
    
    def _classify_question(self, question: str) -> Dict[str, Any]:
        """
        Classify incoming question using our existing categorization system
        """
        question_lower = question.lower()
        
        # Use same categorization logic as in contextual_content_categorizer
        category = self._determine_question_category(question_lower)
        topic = self._determine_question_topic(question_lower)
        requirement_type = self._determine_question_requirement_type(question_lower)
        
        return {
            'category': category,
            'topic': topic,
            'requirement_type': requirement_type,
            'search_terms': self._extract_question_search_terms(question_lower)
        }
    
    def _determine_question_category(self, question: str) -> str:
        """Determine question category based on content"""
        if any(word in question for word in ['disclose', 'disclosure', 'information', 'detail']):
            return 'DISCLOSURE'
        elif any(word in question for word in ['present', 'show', 'display', 'statement']):
            return 'PRESENTATION'
        elif any(word in question for word in ['fair value', 'measure', 'value', 'amount']):
            return 'MEASUREMENT'
        elif any(word in question for word in ['recognise', 'recognize', 'record']):
            return 'RECOGNITION'
        else:
            return 'DISCLOSURE'  # Default
    
    def _determine_question_topic(self, question: str) -> str:
        """Determine question topic based on keywords"""
        topic_keywords = {
            'FINANCIAL_INSTRUMENTS': ['financial instrument', 'derivative', 'investment'],
            'PROPERTY_ASSETS': ['property', 'asset', 'equipment', 'depreciation'],
            'REVENUE_PERFORMANCE': ['revenue', 'income', 'performance'],
            'TAX': ['tax', 'deferred tax', 'income tax'],
            'LEASES': ['lease', 'lessee', 'lessor'],
            'FOREIGN_CURRENCY': ['foreign', 'currency', 'exchange'],
            'FINANCIAL_POSITION': ['balance', 'position', 'financial position']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in question for keyword in keywords):
                return topic
        
        return 'GENERAL'
    
    def _determine_question_requirement_type(self, question: str) -> str:
        """Determine if question is about mandatory, conditional, or optional requirements"""
        if any(word in question for word in ['must', 'shall', 'require']):
            return 'MANDATORY'
        elif any(word in question for word in ['if', 'when', 'where']):
            return 'CONDITIONAL'
        elif any(word in question for word in ['may', 'optional']):
            return 'OPTIONAL'
        else:
            return 'MANDATORY'
    
    def _extract_question_search_terms(self, question: str) -> List[str]:
        """Extract key search terms from question"""
        import re
        # Remove common question words and extract meaningful terms
        stop_words = {'does', 'the', 'entity', 'whether', 'if', 'has', 'have', 'been', 'is', 'are', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = re.findall(r'\b\w+\b', question.lower())
        meaningful_words = [word for word in words if word not in stop_words and len(word) > 2]
        return meaningful_words[:10]  # Limit to top 10 terms
    
    def _retrieve_matching_content(
        self, 
        document_id: str, 
        question_categories: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Retrieve content that matches question categories from storage
        """
        with sqlite3.connect(self.storage.db_path) as conn:
            # Enable row factory for dict-like access
            conn.row_factory = sqlite3.Row
            
            # Primary match: exact category and topic
            primary_query = """
                SELECT * FROM categorized_content 
                WHERE document_id = ? AND category = ? AND topic = ?
                ORDER BY confidence DESC, page_num, paragraph_num
                LIMIT 50
            """
            
            cursor = conn.execute(primary_query, (
                document_id, 
                question_categories['category'],
                question_categories['topic']
            ))
            primary_matches = [dict(row) for row in cursor.fetchall()]
            
            # Secondary match: same category, different topic
            if len(primary_matches) < 20:
                secondary_query = """
                    SELECT * FROM categorized_content 
                    WHERE document_id = ? AND category = ? AND topic != ?
                    ORDER BY confidence DESC, page_num, paragraph_num
                    LIMIT 30
                """
                
                cursor = conn.execute(secondary_query, (
                    document_id,
                    question_categories['category'],
                    question_categories['topic']
                ))
                secondary_matches = [dict(row) for row in cursor.fetchall()]
                primary_matches.extend(secondary_matches)
            
            # Convert JSON fields back to Python objects
            for match in primary_matches:
                match['line_range'] = json.loads(match['line_range'])
                match['cross_references'] = json.loads(match['cross_references'])
                match['note_numbers'] = json.loads(match['note_numbers'])
                match['table_references'] = json.loads(match['table_references'])
                match['statement_types'] = json.loads(match['statement_types'])
                match['search_terms'] = json.loads(match['search_terms'])
                match['semantic_tags'] = json.loads(match['semantic_tags'])
            
            return primary_matches
    
    def _rank_and_select_content(
        self, 
        content_pieces: List[Dict[str, Any]], 
        question: str,
        max_length: int
    ) -> List[Dict[str, Any]]:
        """
        Rank content pieces by relevance and select best ones within length limit
        """
        question_terms = set(self._extract_question_search_terms(question))
        
        # Score each content piece
        scored_pieces = []
        for piece in content_pieces:
            score = 0
            
            # Confidence score
            score += piece['confidence'] * 10
            
            # Search term overlap
            piece_terms = set(piece['search_terms'])
            term_overlap = len(question_terms & piece_terms)
            score += term_overlap * 5
            
            # Content type bonus
            if piece['content_type'] in ['NUMERICAL_DATA', 'TABLE_DATA']:
                score += 3
            
            # Statement type relevance
            if piece['primary_statement'] != 'UNKNOWN':
                score += 2
            
            scored_pieces.append((score, piece))
        
        # Sort by score and select within length limit
        scored_pieces.sort(key=lambda x: x[0], reverse=True)
        
        selected = []
        total_length = 0
        
        for score, piece in scored_pieces:
            content_length = len(piece['content'])
            if total_length + content_length <= max_length:
                selected.append(piece)
                total_length += content_length
            else:
                break
        
        # If no content selected due to length constraints, select at least the top-ranked piece
        if not selected and scored_pieces:
            logger.warning(f"No content fit within {max_length} chars, selecting top-ranked piece")
            selected.append(scored_pieces[0][1])
        
        return selected
    
    def _accumulate_into_chunk(
        self, 
        selected_content: List[Dict[str, Any]], 
        question_categories: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Accumulate selected content into final chunk with preserved citations
        """
        if not selected_content:
            return {
                'content': '',
                'citations': [],
                'category_match': question_categories,
                'total_pieces': 0,
                'confidence': 0.0
            }
        
        # Sort by page and paragraph for logical flow
        selected_content.sort(key=lambda x: (x['page_num'], x['paragraph_num']))
        
        # Accumulate content and build citations
        accumulated_text = []
        citations = []
        
        for i, piece in enumerate(selected_content):
            # Add content
            accumulated_text.append(piece['content'])
            
            # Build citation with preserved metadata
            citation = {
                'citation_id': i + 1,
                'page_num': piece['page_num'],
                'paragraph_num': piece['paragraph_num'],
                'cross_references': piece['cross_references'],
                'note_numbers': piece['note_numbers'],
                'table_references': piece['table_references'],
                'statement_type': piece['primary_statement'],
                'content_preview': piece['content'][:100] + '...' if len(piece['content']) > 100 else piece['content'],
                'confidence': piece['confidence']
            }
            citations.append(citation)
        
        # Calculate overall confidence
        avg_confidence = sum(p['confidence'] for p in selected_content) / len(selected_content)
        
        return {
            'content': ' '.join(accumulated_text),
            'citations': citations,
            'category_match': question_categories,
            'total_pieces': len(selected_content),
            'confidence': avg_confidence,
            'total_length': len(' '.join(accumulated_text))
        }


def test_intelligent_accumulation():
    """Test the complete intelligent accumulation system"""
    print("=== Testing Intelligent Content Accumulation ===")
    
    # Initialize storage and accumulator
    storage = CategoryAwareContentStorage("test_categorized_content.db")
    accumulator = IntelligentChunkAccumulator(storage)
    
    # Test questions
    test_questions = [
        "Does the entity disclose the fair value of investment property?",
        "How does the entity present cash flow from operating activities?",
        "What are the entity's accounting policies for revenue recognition?",
        "Does the entity recognize impairment losses on financial instruments?"
    ]
    
    sample_document_id = "sample_doc_001"
    
    for i, question in enumerate(test_questions):
        print(f"\n--- Test Question {i+1} ---")
        print(f"Question: {question}")
        
        try:
            # Accumulate relevant content
            result = accumulator.accumulate_relevant_content(question, sample_document_id, max_content_length=800)
            
            print(f"Category Match: {result['category_match']}")
            print(f"Total Pieces: {result['total_pieces']}")
            print(f"Content Length: {result['total_length']} characters")
            print(f"Average Confidence: {result['confidence']:.2f}")
            print(f"Citations: {len(result['citations'])}")
            
            # Show sample citations
            for j, citation in enumerate(result['citations'][:3]):
                print(f"  Citation {j+1}: Page {citation['page_num']}, Para {citation['paragraph_num']}")
                print(f"    {citation['content_preview']}")
            
        except Exception as e:
            print(f"Error processing question: {e}")


if __name__ == "__main__":
    test_intelligent_accumulation()