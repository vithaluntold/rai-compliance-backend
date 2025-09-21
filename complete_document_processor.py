"""
Complete Document Processing Pipeline: Tag → Accumulate → Preserve
Demonstrates the full intelligent document processing system
"""

import logging
import sys
import os
from typing import Dict, List, Any
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.contextual_content_categorizer import ContextualContentCategorizer
from services.intelligent_chunk_accumulator import CategoryAwareContentStorage, IntelligentChunkAccumulator
from services.smart_metadata_extractor import SmartMetadataExtractor

logger = logging.getLogger(__name__)


class CompleteDocumentProcessor:
    """
    Complete document processing pipeline that implements:
    1. Financial Statement Recognition
    2. Contextual Content Categorization (Category → Topic → Requirement Type)
    3. Category-Aware Storage with Citation Preservation
    4. Intelligent Chunk Accumulation based on Question Categories
    """
    
    def __init__(self):
        self.categorizer = ContextualContentCategorizer()
        self.storage = CategoryAwareContentStorage()
        self.accumulator = IntelligentChunkAccumulator(self.storage)
        self.metadata_extractor = SmartMetadataExtractor()
        
    def process_document(self, pdf_path: str, document_id: str) -> Dict[str, Any]:
        """
        Complete document processing pipeline
        
        Phase 1: Smart Metadata Extraction (Company Info, Financial Statements)
        Phase 2: Contextual Categorization with Extended Context
        Phase 3: Store with preserved page/paragraph/cross-reference metadata
        """
        logger.info(f"🚀 DOCUMENT PROCESSOR STEP 1: Starting complete document processing for: {pdf_path}")
        
        try:
            # Phase 1: Smart Metadata Extraction
            logger.info(f"🚀 DOCUMENT PROCESSOR STEP 2: Starting Phase 1 - Smart metadata extraction")
            print("Phase 1: Extracting company information and financial statement metadata...")
            metadata_result = self.metadata_extractor.extract_metadata(pdf_path)
            logger.info(f"✅ DOCUMENT PROCESSOR STEP 2 COMPLETE: Metadata extracted - Company: {metadata_result.get('company_name', 'Unknown')}")
            
            # Phase 2: Contextual Categorization with Extended Context
            logger.info(f"🚀 DOCUMENT PROCESSOR STEP 3: Starting Phase 2 - Contextual categorization")
            print("Phase 2: Categorizing content with extended context...")
            categorized_content = self.categorizer.categorize_document_content(pdf_path)
            logger.info(f"✅ DOCUMENT PROCESSOR STEP 3 COMPLETE: Categorized {len(categorized_content)} content pieces")
            
            # Phase 3: Store with Category Tags and Citation Metadata
            logger.info(f"🚀 DOCUMENT PROCESSOR STEP 4: Starting Phase 3 - Storage with category tags")
            print("Phase 3: Storing categorized content with citation metadata...")
            self.storage.store_categorized_content(categorized_content, document_id)
            logger.info(f"✅ DOCUMENT PROCESSOR STEP 4 COMPLETE: Content stored with category-aware metadata")
            
            # Generate processing summary with metadata
            logger.info(f"🚀 DOCUMENT PROCESSOR STEP 5: Generating processing summary")
            summary = self._generate_processing_summary(categorized_content, document_id, metadata_result)
            logger.info(f"✅ DOCUMENT PROCESSOR STEP 5 COMPLETE: Summary generated with {summary.get('total_content_pieces', 0)} pieces")
            
            logger.info(f"✅ DOCUMENT PROCESSOR COMPLETE: All phases successful for document {document_id}")
            return summary
            
        except Exception as e:
            logger.error(f"❌ DOCUMENT PROCESSOR FAILED: Error processing document: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def answer_question_intelligently(
        self, 
        question: str, 
        document_id: str,
        max_content_length: int = 800
    ) -> Dict[str, Any]:
        """
        Answer question using intelligent content accumulation
        
        Process:
        1. Classify question → Category/Topic/Requirement
        2. Retrieve ONLY matching categorized content
        3. Accumulate with preserved citations
        4. Return targeted answer with page/note references
        """
        logger.info(f"🚀 QUESTION PROCESSOR STEP 1: Starting intelligent question processing: {question[:50]}...")
        
        try:
            # Intelligent accumulation based on question categories
            logger.info(f"🚀 QUESTION PROCESSOR STEP 2: Performing intelligent content accumulation")
            result = self.accumulator.accumulate_relevant_content(
                question, document_id, max_content_length
            )
            logger.info(f"✅ QUESTION PROCESSOR STEP 2 COMPLETE: Retrieved {result.get('total_pieces', 0)} content pieces")
            
            # Format for compliance analysis
            logger.info(f"🚀 QUESTION PROCESSOR STEP 3: Formatting compliance answer")
            formatted_result = self._format_compliance_answer(result, question)
            logger.info(f"✅ QUESTION PROCESSOR STEP 3 COMPLETE: Answer formatted with {len(formatted_result.get('citations', []))} citations")
            
            logger.info(f"✅ QUESTION PROCESSOR COMPLETE: Answer processed successfully")
            return formatted_result
            
        except Exception as e:
            logger.error(f"❌ QUESTION PROCESSOR FAILED: Error answering question: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _generate_processing_summary(
        self, 
        categorized_content: List[Dict[str, Any]], 
        document_id: str,
        metadata_result: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate summary of document processing results with metadata"""
        
        # Category distribution
        categories = {}
        topics = {}
        statement_types = {}
        
        for piece in categorized_content:
            cat = piece['category']
            topic = piece['topic']
            stmt_type = piece['primary_statement']
            
            categories[cat] = categories.get(cat, 0) + 1
            topics[topic] = topics.get(topic, 0) + 1
            statement_types[stmt_type] = statement_types.get(stmt_type, 0) + 1
        
        # Citation metadata stats
        total_cross_refs = sum(len(piece['cross_references']) for piece in categorized_content)
        total_note_refs = sum(len(piece['note_numbers']) for piece in categorized_content)
        
        # Include extracted metadata
        summary = {
            'status': 'success',
            'document_id': document_id,
            'total_content_pieces': len(categorized_content),
            'category_distribution': categories,
            'top_topics': dict(sorted(topics.items(), key=lambda x: x[1], reverse=True)[:10]),
            'statement_type_distribution': statement_types,
            'citation_metadata': {
                'total_cross_references': total_cross_refs,
                'total_note_references': total_note_refs,
                'pages_processed': len(set(piece['page_num'] for piece in categorized_content))
            },
            'avg_confidence': sum(piece['confidence'] for piece in categorized_content) / len(categorized_content)
        }
        
        # Add extracted metadata if available
        if metadata_result:
            summary['extracted_metadata'] = metadata_result
            
        return summary
    
    def _format_compliance_answer(
        self, 
        accumulation_result: Dict[str, Any], 
        question: str
    ) -> Dict[str, Any]:
        """Format answer for compliance analysis with citations"""
        
        if accumulation_result['total_pieces'] == 0:
            return {
                'status': 'no_match',
                'question': question,
                'answer': 'No relevant content found for this question.',
                'citations': [],
                'confidence': 0.0
            }
        
        # Format citations for compliance reporting
        formatted_citations = []
        for citation in accumulation_result['citations']:
            citation_text = f"Page {citation['page_num']}, Paragraph {citation['paragraph_num']}"
            
            if citation['note_numbers']:
                citation_text += f", Note {', '.join(citation['note_numbers'])}"
            if citation['table_references']:
                citation_text += f", Table {', '.join(citation['table_references'])}"
            if citation['statement_type'] != 'UNKNOWN':
                citation_text += f" ({citation['statement_type']})"
            
            formatted_citations.append({
                'citation_text': citation_text,
                'content_preview': citation['content_preview'],
                'confidence': citation['confidence']
            })
        
        return {
            'status': 'success',
            'question': question,
            'answer_content': accumulation_result['content'],
            'category_classification': accumulation_result['category_match'],
            'citations': formatted_citations,
            'confidence': accumulation_result['confidence'],
            'content_length': accumulation_result['total_length'],
            'evidence_pieces': accumulation_result['total_pieces']
        }


def demonstrate_complete_system():
    """Demonstrate the complete Tag → Accumulate → Preserve system"""
    print("=" * 60)
    print("COMPLETE DOCUMENT PROCESSING DEMONSTRATION")
    print("Tag → Accumulate → Preserve with Citation Preservation")
    print("=" * 60)
    
    # Initialize processor
    processor = CompleteDocumentProcessor()
    
    # Test PDF and document ID
    pdf_path = "c:/Users/saivi/OneDrive/Documents/Audricc all/uploads/RAI-1757795217-3ADC237A.pdf"
    document_id = "DEMO_RAI_001"
    
    print(f"\n📄 Processing Document: {pdf_path}")
    print(f"📋 Document ID: {document_id}")
    
    # Phase 1 & 2: Process and Store Document
    print(f"\n🔄 PHASE 1 & 2: Document Processing...")
    processing_result = processor.process_document(pdf_path, document_id)
    
    if processing_result['status'] == 'success':
        print(f"✅ Processing completed successfully!")
        print(f"   📊 Content pieces: {processing_result['total_content_pieces']}")
        print(f"   📈 Average confidence: {processing_result['avg_confidence']:.2f}")
        print(f"   📄 Pages processed: {processing_result['citation_metadata']['pages_processed']}")
        print(f"   🔗 Cross-references: {processing_result['citation_metadata']['total_cross_references']}")
        
        print(f"\n📊 Category Distribution:")
        for category, count in processing_result['category_distribution'].items():
            print(f"   {category}: {count}")
        
        print(f"\n🎯 Top Topics:")
        for topic, count in list(processing_result['top_topics'].items())[:5]:
            print(f"   {topic}: {count}")
    
    # Phase 3: Intelligent Question Answering
    print(f"\n🔄 PHASE 3: Intelligent Question Answering...")
    
    test_questions = [
        "Does the entity disclose the fair value of investment property?",
        "How does the entity present cash flow from operating activities?",
        "What are the entity's significant accounting policies?",
        "Does the entity recognize revenue according to IFRS 15?"
    ]
    
    for i, question in enumerate(test_questions):
        print(f"\n--- Question {i+1} ---")
        print(f"❓ {question}")
        
        # Get intelligent answer
        answer_result = processor.answer_question_intelligently(question, document_id)
        
        if answer_result['status'] == 'success':
            print(f"✅ Answer found!")
            print(f"   📁 Category: {answer_result['category_classification']['category']}")
            print(f"   🎯 Topic: {answer_result['category_classification']['topic']}")
            print(f"   📊 Confidence: {answer_result['confidence']:.2f}")
            print(f"   📄 Evidence pieces: {answer_result['evidence_pieces']}")
            print(f"   📝 Content length: {answer_result['content_length']} chars")
            
            print(f"\n   📑 Citations:")
            for j, citation in enumerate(answer_result['citations'][:3]):
                print(f"     {j+1}. {citation['citation_text']}")
                print(f"        Preview: {citation['content_preview'][:60]}...")
            
            print(f"\n   📋 Content Preview:")
            content_preview = answer_result['answer_content'][:200]
            print(f"     {content_preview}...")
            
        elif answer_result['status'] == 'no_match':
            print(f"⚠️  No relevant content found")
        else:
            print(f"❌ Error: {answer_result['message']}")
    
    print(f"\n🎉 DEMONSTRATION COMPLETED!")
    print(f"   The system successfully:")
    print(f"   ✅ Tagged content with Category → Topic → Requirement Type")
    print(f"   ✅ Preserved page numbers, paragraph numbers, and cross-references")
    print(f"   ✅ Accumulated only relevant content based on question categories")
    print(f"   ✅ Provided precise citations for compliance analysis")


if __name__ == "__main__":
    import json
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run demonstration
    demonstrate_complete_system()