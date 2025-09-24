import logging
import re
from typing import Any, Dict, List, Optional, Set

import fitz  # PyMuPDF
from docx import Document as DocxDocument

from services.vector_store import generate_document_id

logger = logging.getLogger(__name__)


class DocumentChunker:
    def __init__(self, min_chunk_length: int = 1):
        self.min_chunk_length = min_chunk_length
        self.dynamic_headers: Set[str] = set()
        logger.info(
            f"Initialized DocumentChunker with min_chunk_length={min_chunk_length}"
        )

    def chunk_pdf(
        self, pdf_path: str, document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        # Generate a document ID if not provided
        if document_id is None:
            document_id = generate_document_id()

        logger.info(f"[DEBUG] Processing PDF: {pdf_path} with ID: {document_id}")
        chunks = []

        try:
            with fitz.open(pdf_path) as doc:
                total_pages = len(doc)

                # First pass: Extract metadata from initial pages
                metadata_text = ""
                for page_num in range(
                    min(5, total_pages)
                ):  # Check first 5 pages for metadata
                    page = doc[page_num]
                    text = page.get_text().strip()  # type: ignore[attr-defined]
                    if text:
                        metadata_text += text + "\n"

                # Add metadata chunk first
                if metadata_text:
                    chunks.append(
                        {
                            "chunk_index": 0,
                            "page": 0,
                            "page_no": 0,
                            "text": self._clean_financial_headers(metadata_text),
                            "length": len(metadata_text),
                            "chunk_type": "metadata",
                        }
                    )

                # Second pass: Process all pages
                for page_num in range(total_pages):
                    page = doc[page_num]
                    text = page.get_text().strip()  # type: ignore[attr-defined]
                    if not text:
                        continue

                    cleaned_text = self._clean_financial_headers(text)
                    if len(cleaned_text) >= self.min_chunk_length:
                        chunks.append(
                            {
                                "chunk_index": len(chunks),
                                "page": page_num,
                                "page_no": page_num,
                                "text": cleaned_text,
                                "length": len(cleaned_text),
                                "chunk_type": "content",
                            }
                        )

            logger.info(
                f"[DEBUG] Created {len(chunks)} chunks from {total_pages} pages"
            )
            return chunks

        except Exception as e:
            logger.error(f"[ERROR] Failed to chunk PDF: {e}")
            return []

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract raw text from PDF for metadata processing."""
        logger.info(f"[DEBUG] Extracting text from PDF: {pdf_path}")
        full_text = ""
        
        try:
            with fitz.open(pdf_path) as doc:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text().strip()  # type: ignore[attr-defined]
                    if text:
                        cleaned_text = self._clean_financial_headers(text)
                        full_text += cleaned_text + "\n"
            
            logger.info(f"[DEBUG] Extracted {len(full_text)} characters from PDF")
            return full_text.strip()
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to extract text from PDF: {e}")
            return ""

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
                all_chunks.append("\n".join(current_chunk))
            # Add metadata chunk (first chunk)
            if all_chunks:
                chunks.append(
                    {
                        "chunk_index": 0,
                        "page": 0,
                        "page_no": 0,
                        "text": self._clean_financial_headers(all_chunks[0]),
                        "length": len(all_chunks[0]),
                        "chunk_type": "metadata",
                    }
                )
            # Add content chunks (rest)
            for i, chunk_text in enumerate(all_chunks[1:], start=1):
                cleaned_text = self._clean_financial_headers(chunk_text)
                if len(cleaned_text) >= self.min_chunk_length:
                    chunks.append(
                        {
                            "chunk_index": len(chunks),
                            "page": i,
                            "page_no": i,
                            "text": cleaned_text,
                            "length": len(cleaned_text),
                            "chunk_type": "content",
                        }
                    )
            logger.info(
                f"[DEBUG] Created {len(chunks)} chunks from DOCX "
                f"(simulated pages, 3000 chars each)"
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
#     output_pdf = tempfile.mktemp(suffix='.pdf')
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
#             f"Only DOCX is supported."
#         )

# Global instance
document_chunker = DocumentChunker(min_chunk_length=30)


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Standalone function to extract raw text from PDF for metadata processing.
    Uses the global document_chunker instance.
    """
    return document_chunker.extract_text_from_pdf(pdf_path)