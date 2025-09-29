import logging
import re
from typing import Any, Dict, List, Optional, Set

import fitz  # PyMuPDF
from docx import Document as DocxDocument

from services.vector_store import generate_document_id
from services.contextual_content_categorizer import ContextualContentCategorizer
from services.intelligent_chunk_accumulator import IntelligentChunkAccumulator, CategoryAwareContentStorage

logger = logging.getLogger(__name__)


class DocumentChunker:
    def __init__(self, min_chunk_length: int = 30):
        self.min_chunk_length = min_chunk_length
        self.dynamic_headers: Set[str] = set()
        
        try:
            logger.info(f"[INIT] Initializing ContextualContentCategorizer...")
            self.categorizer = ContextualContentCategorizer()
            logger.info(f"[INIT] ContextualContentCategorizer initialized successfully")
        except Exception as e:
            logger.error(f"[INIT] Failed to initialize ContextualContentCategorizer: {e}")
            import traceback
            logger.error(f"[INIT] Full traceback: {traceback.format_exc()}")
            raise
        
        try:
            logger.info(f"[INIT] Initializing CategoryAwareContentStorage...")
            self.storage = CategoryAwareContentStorage()
            logger.info(f"[INIT] CategoryAwareContentStorage initialized successfully")
        except Exception as e:
            logger.error(f"[INIT] Failed to initialize CategoryAwareContentStorage: {e}")
            import traceback
            logger.error(f"[INIT] Full traceback: {traceback.format_exc()}")
            raise
        
        try:
            logger.info(f"[INIT] Initializing IntelligentChunkAccumulator...")
            self.accumulator = IntelligentChunkAccumulator(self.storage)
            logger.info(f"[INIT] IntelligentChunkAccumulator initialized successfully")
        except Exception as e:
            logger.error(f"[INIT] Failed to initialize IntelligentChunkAccumulator: {e}")
            import traceback
            logger.error(f"[INIT] Full traceback: {traceback.format_exc()}")
            raise
        
        logger.info(
            f"Initialized DocumentChunker with NLP categorization, min_chunk_length={min_chunk_length}"
        )

    def chunk_pdf(
        self, pdf_path: str, document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        # Generate a document ID if not provided
        if document_id is None:
            document_id = generate_document_id()

        logger.info(f"[NLP] Processing PDF with contextual categorization: {pdf_path}")
        
        try:
            with fitz.open(pdf_path) as doc:
                total_pages = len(doc)
                
                # Step 1: Create dedicated metadata chunk from first 8 pages
                logger.info(f"[NLP] Step 1: Creating metadata chunk from first 8 pages")
                metadata_chunk = self._create_metadata_chunk(doc, document_id)
                logger.info(f"[NLP] Step 1 SUCCESS: Created metadata chunk with {len(metadata_chunk['text'])} characters")
                
                # Step 2: Extract all pages for NLP processing
                logger.info(f"[NLP] Step 2: Extracting text from {total_pages} pages")
                all_page_texts = []
                for page_num in range(total_pages):
                    page = doc[page_num]
                    text = page.get_text().strip()
                    if text:
                        cleaned_text = self._clean_financial_headers(text)
                        all_page_texts.append({
                            'page_num': page_num,
                            'text': cleaned_text,
                            'length': len(cleaned_text)
                        })
                logger.info(f"[NLP] Step 2 SUCCESS: Extracted {len(all_page_texts)} pages with text")

                # Step 3: Use ContextualContentCategorizer for intelligent classification
                logger.info(f"[NLP] Step 3: Starting categorization with ContextualContentCategorizer")
                categorized_content = []
                smart_categorization_success = False
                
                try:
                    categorized_content = self.categorizer.categorize_page_texts(all_page_texts, pdf_path)
                    logger.info(f"[NLP] Step 3 SUCCESS: Categorizer returned {len(categorized_content)} categorized pieces")
                    smart_categorization_success = True
                except Exception as e:
                    logger.error(f"[NLP] Step 3 FAILED: Categorization error: {e}")
                    logger.error(f"[NLP] Step 3 FAILED: Categorizer type: {type(self.categorizer)}")
                    logger.error(f"[NLP] Step 3 FAILED: Input data sample: {all_page_texts[0] if all_page_texts else 'No pages'}")
                    import traceback
                    logger.error(f"[NLP] Step 3 FAILED: Full traceback: {traceback.format_exc()}")
                    logger.warning(f"[NLP] Step 3 FALLBACK: Will create basic chunks without smart categorization")
                    # Don't raise - continue with fallback
                
                # Step 4: Use IntelligentChunkAccumulator for smart chunking (if categorization succeeded)
                intelligent_chunks = []
                if smart_categorization_success and categorized_content:
                    logger.info(f"[NLP] Step 4: Starting intelligent chunking with IntelligentChunkAccumulator")
                    try:
                        intelligent_chunks = self.accumulator.create_contextual_chunks(
                            categorized_content, 
                            document_id
                        )
                        logger.info(f"[NLP] Step 4 SUCCESS: Accumulator created {len(intelligent_chunks)} intelligent chunks")
                    except Exception as e:
                        logger.error(f"[NLP] Step 4 FAILED: Accumulator error: {e}")
                        logger.error(f"[NLP] Step 4 FAILED: Accumulator type: {type(self.accumulator)}")
                        logger.error(f"[NLP] Step 4 FAILED: Input data count: {len(categorized_content)}")
                        import traceback
                        logger.error(f"[NLP] Step 4 FAILED: Full traceback: {traceback.format_exc()}")
                        logger.warning(f"[NLP] Step 4 FALLBACK: Will create basic chunks without intelligent accumulation")
                        intelligent_chunks = []
                else:
                    logger.warning(f"[NLP] Step 4 SKIPPED: No categorized content available for intelligent chunking")

                # Step 5: Combine metadata chunk with intelligent chunks or create basic chunks as fallback
                logger.info(f"[NLP] Step 5: Combining metadata chunk with intelligent chunks")
                
                # Update metadata chunk with smart categorization status
                metadata_chunk['smart_categorization_status'] = {
                    'enabled': True,
                    'categorization_success': smart_categorization_success,
                    'categorized_pieces': len(categorized_content) if categorized_content else 0,
                    'intelligent_chunks_created': len(intelligent_chunks) if intelligent_chunks else 0
                }
                
                if intelligent_chunks:
                    # Use intelligent chunks if available
                    all_chunks = [metadata_chunk] + intelligent_chunks
                    logger.info(f"[NLP] SUCCESS: Created {len(all_chunks)} total chunks: 1 metadata + {len(intelligent_chunks)} intelligent chunks from {total_pages} pages")
                else:
                    # Fallback: Create basic chunks from page texts when smart categorization fails
                    logger.warning(f"[NLP] FALLBACK: Creating basic content chunks since smart categorization failed")
                    basic_chunks = []
                    for page_data in all_page_texts:
                        page_num = page_data.get('page_num', 0)
                        text = page_data.get('text', '')
                        if text.strip():  # Only create chunks for non-empty pages
                            basic_chunk = {
                                'chunk_id': f"{document_id}_page_{page_num}",
                                'content': text,
                                'page_num': page_num,
                                'length': len(text),
                                'chunk_type': 'basic_content',
                                'category': 'GENERAL',
                                'topic': 'DOCUMENT_CONTENT',
                                'relevance_score': 0.5  # Default relevance for basic chunks
                            }
                            basic_chunks.append(basic_chunk)
                    
                    all_chunks = [metadata_chunk] + basic_chunks
                    logger.info(f"[NLP] FALLBACK SUCCESS: Created {len(all_chunks)} total chunks: 1 metadata + {len(basic_chunks)} basic chunks from {total_pages} pages")
                
                # TRIGGER METADATA EXTRACTION: Send to smart_metadata_extractor after categorization complete
                logger.info(f"[METADATA] Triggering smart metadata extraction for document {document_id}")
                self._trigger_metadata_extraction(document_id, metadata_chunk, categorized_content, smart_categorization_success)
                
                return all_chunks

        except Exception as e:
            logger.error(f"[ERROR] NLP chunking failed with exception: {e}")
            logger.error(f"[ERROR] Exception type: {type(e)}")
            logger.error(f"[ERROR] PDF path: {pdf_path}")
            logger.error(f"[ERROR] Document ID: {document_id}")
            import traceback
            logger.error(f"[ERROR] Full traceback: {traceback.format_exc()}")
            # Fallback to basic chunking if NLP fails
            return self._basic_fallback_chunking(pdf_path, document_id)

    def _create_metadata_chunk(self, doc, document_id: str) -> Dict[str, Any]:
        """
        Create dedicated metadata chunk optimized for SmartMetadataExtractor.
        Extracts content from first 8 pages focusing on:
        - Company name (headers, titles, signatures)
        - Nature of business (business description sections)
        - Operational demographics (geographical references)
        - Financial statements type (consolidated/standalone references)
        """
        logger.info("[METADATA] Creating metadata chunk optimized for SmartMetadataExtractor")
        
        # Extract first 8 pages (where metadata typically appears)
        metadata_pages = min(8, len(doc))
        metadata_text_parts = []
        
        for page_num in range(metadata_pages):
            page = doc[page_num]
            page_text = page.get_text().strip()
            
            if page_text:
                # Preserve important metadata elements while cleaning
                preserved_text = self._preserve_metadata_elements(page_text)
                metadata_text_parts.append(preserved_text)
        
        # Combine metadata text with strategic spacing
        full_metadata_text = "\n\n--- PAGE BREAK ---\n\n".join(metadata_text_parts)
        
        # Clean while preserving metadata indicators
        cleaned_metadata = self._clean_metadata_text(full_metadata_text)
        
        metadata_chunk = {
            "chunk_index": 0,
            "page": 0,
            "page_no": "0-7",  # Indicates multi-page metadata
            "text": cleaned_metadata,
            "length": len(cleaned_metadata),
            "chunk_type": "metadata",
            "category": "METADATA_EXTRACTION",
            "optimization_target": "smart_metadata_extractor",
            "content_focus": ["company_name", "nature_of_business", "operational_demographics", "financial_statements_type"]
        }
        
        logger.info(f"[METADATA] Created metadata chunk: {len(cleaned_metadata)} chars from {metadata_pages} pages")
        return metadata_chunk

    def _preserve_metadata_elements(self, text: str) -> str:
        """Preserve elements critical for metadata extraction"""
        # Don't aggressively clean headers in metadata sections
        # Company names often appear in headers/footers
        
        # Preserve company name patterns (PJSC, Ltd, LLC, etc.)
        company_indicators = ['PJSC', 'PLC', 'LLC', 'Ltd', 'Limited', 'Corporation', 'Inc', 'Properties', 'Holdings', 'Group']
        
        # Preserve geographical references
        geographical_indicators = ['UAE', 'Egypt', 'United Arab Emirates', 'England', 'Wales', 'Dubai', 'Abu Dhabi']
        
        # Preserve business activity keywords
        business_indicators = ['engaged in', 'business', 'activities', 'operations', 'development', 'construction', 'leasing', 'management', 'real estate']
        
        # Preserve financial statement type indicators
        fs_indicators = ['consolidated', 'standalone', 'separate', 'financial statements']
        
        # Keep text as-is for metadata extraction - minimal cleaning
        return text.strip()

    def _clean_metadata_text(self, text: str) -> str:
        """Clean metadata text while preserving extraction targets"""
        cleaned = text
        
        # Only remove obvious noise, preserve potential company/business info
        # Remove page numbers but keep everything else
        cleaned = re.sub(r'\bpage\s+\d+\s*(?:of\s+\d+)?\b', ' ', cleaned, flags=re.IGNORECASE)
        
        # Normalize excessive whitespace but preserve line breaks for context
        cleaned = re.sub(r' +', ' ', cleaned)  # Multiple spaces to single
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)  # Multiple newlines to double
        
        return cleaned.strip()

    def _basic_fallback_chunking(self, pdf_path: str, document_id: str) -> List[Dict[str, Any]]:
        """Fallback to basic chunking if NLP processing fails"""
        logger.warning(f"[FALLBACK] Using basic chunking for {pdf_path}")
        chunks = []
        
        try:
            with fitz.open(pdf_path) as doc:
                # Create metadata chunk even in fallback
                metadata_chunk = self._create_metadata_chunk(doc, document_id)
                chunks.append(metadata_chunk)
                
                # Create content chunks from remaining pages
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text().strip()
                    if text and len(text) >= self.min_chunk_length:
                        cleaned_text = self._clean_financial_headers(text)
                        chunks.append({
                            "chunk_index": len(chunks),
                            "page": page_num,
                            "page_no": page_num,
                            "text": cleaned_text,
                            "length": len(cleaned_text),
                            "chunk_type": "content",
                            "category": "UNCATEGORIZED"
                        })
            return chunks
        except Exception as e:
            logger.error(f"[ERROR] Fallback chunking failed: {e}")
            return []

    def chunk_docx(
        self, docx_path: str, document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        # Generate a document ID if not provided
        if document_id is None:
            document_id = generate_document_id()
        logger.info(f"[DEBUG] Processing DOCX: {docx_path} with ID: {document_id}")
        chunks = []
        try:
            doc = DocxDocument(docx_path)
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            
            # Group paragraphs into 'pages' of ~3000 characters
            PAGE_SIZE = 3000
            current_chunk: list[str] = []
            current_len = 0
            all_chunks = []
            for para in paragraphs:
                if current_len + len(para) > PAGE_SIZE and current_chunk:
                    all_chunks.append("\n".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                current_chunk.append(para)
                current_len += len(para)
            if current_chunk:
                all_chunks.append("\n".join(current_chunk))  # Fixed typo: "ff" to ""
            
            # Create metadata chunk from first 2-3 chunks (equivalent to ~8 pages)
            metadata_chunks_count = 0
            if all_chunks:
                metadata_chunks_count = min(3, len(all_chunks))
                metadata_text = "\n\n--- SECTION BREAK ---\n\n".join(all_chunks[:metadata_chunks_count])
                cleaned_metadata = self._clean_metadata_text(metadata_text)
                
                chunks.append({
                    "chunk_index": 0,
                    "page": 0,
                    "page_no": f"0-{metadata_chunks_count-1}",
                    "text": cleaned_metadata,
                    "length": len(cleaned_metadata),
                    "chunk_type": "metadata",
                    "category": "METADATA_EXTRACTION",
                    "optimization_target": "smart_metadata_extractor",
                    "content_focus": ["company_name", "nature_of_business", "operational_demographics", "financial_statements_type"]
                })
                
            # Add content chunks from remaining sections
            for i, chunk_text in enumerate(all_chunks[metadata_chunks_count:], start=metadata_chunks_count):
                cleaned_text = self._clean_financial_headers(chunk_text)
                if len(cleaned_text) >= self.min_chunk_length:
                    chunks.append({
                        "chunk_index": len(chunks),
                        "page": i,
                        "page_no": i,
                        "text": cleaned_text,
                        "length": len(cleaned_text),
                        "chunk_type": "content",
                        "category": "CONTENT"
                    })
            
            logger.info(
                f"[DEBUG] Created {len(chunks)} chunks from DOCX: 1 metadata + {len(chunks)-1} content chunks"
            )
            return chunks
        except Exception as e:
            logger.error(f"[ERROR] Failed to chunk DOCX: {e}")
            return []

    def _clean_financial_headers(self, text: str) -> str:
        """Clean text while preserving important metadata."""
        cleaned_text = text

        # Only remove headers if they appear multiple times
        for header in self.dynamic_headers:
            # Don't remove headers from first chunk (metadata)
            if "metadata" not in str(cleaned_text):
                cleaned_text = cleaned_text.replace(header, " ")

        # Remove common patterns that aren't useful
        months_pattern = (
            r"January|February|March|April|May|June|July|August|"
            r"September|October|November|December"
        )
        cleaned_text = re.sub(
            rf"\d{{1,2}}\s+({months_pattern})\s+\d{{4}}",
            " ",
            cleaned_text,
        )
        cleaned_text = re.sub(
            rf"({months_pattern})\s+\d{{1,2}},\s+\d{{4}}",
            " ",
            cleaned_text,
        )
        cleaned_text = re.sub(
            r"page\s+\d+\s+of\s+\d+|\d+\s+of\s+\d+|page\s+\d+",
            " ",
            cleaned_text,
            flags=re.IGNORECASE,
        )
        cleaned_text = re.sub(
            r"\(refer (?:to )?note \d+\)", " ", cleaned_text, flags=re.IGNORECASE
        )

        # Normalize whitespace while preserving some structure
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)
        cleaned_text = cleaned_text.strip()

        return cleaned_text


# def convert_to_pdf(input_path: str) -> str:
#     """
#     Convert a DOCX file to PDF. Returns the path to the PDF file.
#     Uses docx2pdf for DOCX.
#     """
#     ext = os.path.splitext(input_path)[1].lower()
#     output_pdf = tempfile.mktemp(suffix='.pd')
#     if ext == '.docx':
#         try:
#             import docx2pdf
#             docx2pdf.convert(input_path, output_pdf)
#             return output_pdf
#         except Exception as e:
#             logger.error(f"docx2pdf conversion failed: {e}")
#             raise
#     else:
#         raise ValueError(
#             f"Unsupported file extension for conversion: {ext}. "
#             "Only DOCX is supported."
#         )

    def _trigger_metadata_extraction(self, document_id: str, metadata_chunk: Dict[str, Any], categorized_content=None, smart_categorization_success=False) -> None:
        """
        Trigger smart metadata extraction in background.
        This runs asynchronously after chunking and categorization is complete.
        """
        import threading
        from services.smart_metadata_extractor import SmartMetadataExtractor
        import asyncio
        from datetime import datetime
        
        def run_metadata_extraction(cat_content=None, cat_success=False):
            try:
                logger.info(f"[METADATA] Starting background metadata extraction for {document_id}")
                
                # Create SmartMetadataExtractor instance
                extractor = SmartMetadataExtractor()
                
                # Run extraction in new event loop (for background thread)
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Extract metadata from the metadata chunk
                metadata_result = loop.run_until_complete(
                    extractor.extract_metadata_optimized(document_id, [metadata_chunk])
                )
                
                # Save metadata using staged storage for isolation
                from services.staged_storage import StagedStorageManager
                storage_manager = StagedStorageManager()
                storage_manager.save_metadata(document_id, metadata_result)
                
                # BACKWARD COMPATIBILITY: Also save to legacy location
                from pathlib import Path
                analysis_results_dir = Path("analysis_results")
                analysis_results_dir.mkdir(exist_ok=True)
                
                metadata_file = analysis_results_dir / f"{document_id}_metadata.json"
                import json
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata_result, f, indent=2, ensure_ascii=False)
                
                # Create the metadata completion flag file for frontend polling
                completion_flag_file = analysis_results_dir / f"{document_id}.metadata_completed"
                with open(completion_flag_file, 'w', encoding='utf-8') as f:
                    f.write(datetime.now().isoformat())
                
                # CRITICAL FIX: Update the MAIN results file that frontend polls (not _status.json)
                main_results_file = analysis_results_dir / f"{document_id}.json"
                
                # Read existing results or create new structure
                try:
                    if main_results_file.exists():
                        with open(main_results_file, 'r', encoding='utf-8') as f:
                            results_data = json.load(f)
                    else:
                        results_data = {
                            "document_id": document_id,
                            "status": "PROCESSING"
                        }
                except Exception:
                    results_data = {
                        "document_id": document_id,
                        "status": "PROCESSING"
                    }
                
                # Update with metadata extraction completion and smart categorization status
                results_data.update({
                    "metadata_extraction": "COMPLETED",
                    "metadata_completed_at": datetime.now().isoformat(),
                    "metadata_file": str(metadata_file),
                    "smart_categorization": {
                        "total_categories": len(set(piece.get('category', 'UNKNOWN') for piece in cat_content)) if cat_content else 0,
                        "content_chunks": len(cat_content) if cat_content else 0,
                        "categorization_complete": cat_success,
                        "categories_found": list(set(piece.get('category', 'UNKNOWN') for piece in cat_content)) if cat_content else []
                    }
                })
                
                # Write back to main results file
                with open(main_results_file, 'w', encoding='utf-8') as f:
                    json.dump(results_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"[METADATA] Background metadata extraction completed for {document_id}")
                logger.info(f"[METADATA] Results saved to {metadata_file}")
                logger.info(f"[METADATA] Completion flag created: {completion_flag_file}")
                
                loop.close()
                
            except Exception as e:
                logger.error(f"[METADATA] Background metadata extraction failed for {document_id}: {e}")
                import traceback
                logger.error(f"[METADATA] Full traceback: {traceback.format_exc()}")
        
        # Start background thread for metadata extraction
        metadata_thread = threading.Thread(
            target=run_metadata_extraction, 
            args=(categorized_content, smart_categorization_success),
            daemon=True
        )
        metadata_thread.start()
        logger.info(f"[METADATA] Metadata extraction thread started for {document_id}")

# Global instance
document_chunker = DocumentChunker(min_chunk_length=30)


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Standalone function to extract raw text from PDF for metadata processing.
    Uses the global document_chunker instance with fallback compatibility.
    
    Deployment: 2025-09-22 15:30 UTC - Added fallback for deployment sync issues
    """
    # Try to use the instance method if it exists
    if hasattr(document_chunker, 'extract_text_from_pdf'):
        return document_chunker.extract_text_from_pdf(pdf_path)
    
    # Fallback: use chunk_pdf and extract text from chunks
    logger.info(f"[FALLBACK] Using chunk_pdf method to extract text from {pdf_path}")
    chunks = document_chunker.chunk_pdf(pdf_path)
    full_text = ""
    for chunk in chunks:
        if chunk.get('text'):
            full_text += chunk['text'] + "\n"
    
    logger.info(f"[FALLBACK] Extracted {len(full_text)} characters using chunk_pdf fallback")
    return full_text.strip()
