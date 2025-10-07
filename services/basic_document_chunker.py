"""
Basic Document Chunker - Temporary Implementation

This is a simple chunking system to handle document processing while the 
enhanced chunking system is being rebuilt from scratch.

Features:
- PDF text extraction using PyMuPDF (fitz)
- DOCX text extraction using python-docx
- Basic sentence-based chunking
- Metadata preservation
- Debug chunk saving
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Import document processing libraries with fallbacks
try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    logger.warning("PyMuPDF not installed, PDF processing will be limited")
    fitz = None
    FITZ_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    logger.warning("python-docx not installed, DOCX support disabled")
    DocxDocument = None
    DOCX_AVAILABLE = False


class BasicDocumentChunker:
    """Basic document chunker for temporary use during system rebuild."""
    
    def __init__(self, chunk_size: int = 1500, overlap_size: int = 200):
        """
        Initialize the basic chunker.
        
        Args:
            chunk_size: Maximum characters per chunk
            overlap_size: Characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        logger.info(f"🔧 BasicDocumentChunker initialized (chunk_size={chunk_size}, overlap={overlap_size})")
    
    def process_document(self, file_path: Path, document_id: str) -> List[Dict[str, Any]]:
        """
        Process a document file and return chunks.
        
        Args:
            file_path: Path to the document file
            document_id: Unique identifier for the document
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        logger.info(f"🚀 Processing document: {document_id} ({file_path.name})")
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document file not found: {file_path}")
        
        # Extract text based on file type
        if file_path.suffix.lower() == ".pdf":
            text = self._extract_pdf_text(file_path)
        elif file_path.suffix.lower() == ".docx":
            text = self._extract_docx_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        if not text.strip():
            raise ValueError("No text extracted from document")
        
        # Create chunks
        chunks = self._create_chunks(text, document_id)
        
        # Save debug information
        self._save_debug_chunks(chunks, document_id)
        
        logger.info(f"✅ Successfully processed {document_id}: {len(chunks)} chunks created")
        return chunks
    
    def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF file."""
        if not FITZ_AVAILABLE:
            raise RuntimeError("PyMuPDF not available for PDF processing")
        
        logger.info(f"📄 Extracting text from PDF: {file_path.name}")
        text = ""
        
        try:
            if fitz is not None:
                doc = fitz.open(str(file_path))
                try:
                    for page_num in range(len(doc)):
                        page = doc[page_num]
                        # Use getattr to avoid linter complaints about PyMuPDF methods
                        page_text = getattr(page, 'get_text')()
                        if page_text.strip():
                            # Add page separator
                            text += f"\n=== PAGE {page_num + 1} ===\n{page_text}\n"
                finally:
                    doc.close()
            
            logger.info(f"✅ PDF text extraction complete: {len(text)} characters")
            return text
            
        except Exception as e:
            logger.error(f"Error extracting PDF text: {str(e)}")
            raise RuntimeError(f"Failed to extract PDF text: {str(e)}")
    
    def _extract_docx_text(self, file_path: Path) -> str:
        """Extract text from DOCX file."""
        if not DOCX_AVAILABLE:
            raise RuntimeError("python-docx not available for DOCX processing")
        
        logger.info(f"📄 Extracting text from DOCX: {file_path.name}")
        text = ""
        
        try:
            if DocxDocument is not None:
                doc = DocxDocument(str(file_path))
                paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
                text = "\n".join(paragraphs)
            
            logger.info(f"✅ DOCX text extraction complete: {len(text)} characters")
            return text
            
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {str(e)}")
            raise RuntimeError(f"Failed to extract DOCX text: {str(e)}")
    
    def _create_chunks(self, text: str, document_id: str) -> List[Dict[str, Any]]:
        """
        Create chunks from text using sentence-based splitting.
        
        Args:
            text: Full document text
            document_id: Document identifier
            
        Returns:
            List of chunk dictionaries
        """
        logger.info(f"🔪 Creating chunks from {len(text)} characters")
        
        # Clean and normalize text
        text = self._normalize_text(text)
        
        # Split into sentences (basic approach)
        sentences = self._split_sentences(text)
        logger.info(f"📝 Split into {len(sentences)} sentences")
        
        # Group sentences into chunks
        chunks = []
        current_chunk = ""
        current_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if adding this sentence would exceed chunk size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
                current_sentences.append(sentence)
            else:
                # Save current chunk if it has content
                if current_chunk.strip():
                    chunk = self._create_chunk_dict(
                        text=current_chunk.strip(),
                        chunk_id=len(chunks),
                        document_id=document_id,
                        sentences=current_sentences.copy()
                    )
                    chunks.append(chunk)
                
                # Start new chunk with overlap
                if self.overlap_size > 0 and current_sentences:
                    # Create overlap from last few sentences
                    overlap_text = self._create_overlap(current_sentences)
                    current_chunk = overlap_text + " " + sentence
                    current_sentences = [sentence]  # Keep only new sentence in tracking
                else:
                    current_chunk = sentence
                    current_sentences = [sentence]
        
        # Add the final chunk
        if current_chunk.strip():
            chunk = self._create_chunk_dict(
                text=current_chunk.strip(),
                chunk_id=len(chunks),
                document_id=document_id,
                sentences=current_sentences
            )
            chunks.append(chunk)
        
        logger.info(f"✅ Created {len(chunks)} chunks")
        return chunks
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for better processing."""
        # Replace multiple whitespace with single space
        text = " ".join(text.split())
        
        # Fix common sentence boundary issues
        text = text.replace(" . ", ". ")
        text = text.replace("...", "…")  # Use ellipsis character
        
        return text
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using basic rules."""
        # Simple sentence splitting - can be enhanced later
        import re
        
        # Split on sentence endings followed by space and capital letter
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Filter out very short sentences (likely not real sentences)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        return sentences
    
    def _create_overlap(self, sentences: List[str]) -> str:
        """Create overlap text from the end of previous chunk."""
        if not sentences:
            return ""
        
        # Take last few sentences for overlap
        overlap_sentences = sentences[-2:] if len(sentences) >= 2 else sentences[-1:]
        overlap_text = " ".join(overlap_sentences)
        
        # Trim if too long
        if len(overlap_text) > self.overlap_size:
            overlap_text = overlap_text[-self.overlap_size:]
            # Find word boundary
            space_idx = overlap_text.find(" ")
            if space_idx > 0:
                overlap_text = overlap_text[space_idx+1:]
        
        return overlap_text
    
    def _create_chunk_dict(self, text: str, chunk_id: int, document_id: str, sentences: List[str]) -> Dict[str, Any]:
        """Create a chunk dictionary with metadata."""
        return {
            "text": text,
            "chunk_id": chunk_id,
            "source": f"document_{document_id}",
            "metadata": {
                "document_id": document_id,
                "chunk_index": chunk_id,
                "char_count": len(text),
                "sentence_count": len(sentences),
                "processing_method": "basic_sentence_split",
                "chunker_version": "basic_v1.0"
            }
        }
    
    def _save_debug_chunks(self, chunks: List[Dict[str, Any]], document_id: str) -> None:
        """Save chunks to file for debugging purposes."""
        try:
            debug_dir = Path("vector_indices")
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            debug_file = debug_dir / f"{document_id}_chunks_debug.json"
            
            with open(debug_file, "w", encoding="utf-8") as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📁 Debug chunks saved to: {debug_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save debug chunks: {str(e)}")


def get_basic_chunker() -> BasicDocumentChunker:
    """Get a configured basic document chunker instance."""
    return BasicDocumentChunker(chunk_size=1500, overlap_size=200)


def process_document_chunks(document_id: str, file_path: Path) -> List[Dict[str, Any]]:
    """
    Process document chunks using the basic chunker.
    
    This is the main function that should be called from routes.
    
    Args:
        document_id: Unique document identifier
        file_path: Path to the document file
        
    Returns:
        List of chunk dictionaries
    """
    chunker = get_basic_chunker()
    return chunker.process_document(file_path, document_id)