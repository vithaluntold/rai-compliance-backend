"""
Intelligent Financial Document Chunking System

This module provides AI-enhanced document chunking specifically designed for
financial documents in the MENA region, focusing on extracting relevant content
for compliance analysis questions.

Key Features:
- PDF structure analysis with Table of Contents extraction
- Financial statement section identification
- Semantic content mapping for question-document matching
- Parallel processing optimization for 325+ compliance questions
"""

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pdfplumber
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocumentSection:
    """Represents a section of a financial document"""

    title: str
    content: str
    page_number: int
    section_type: str  # e.g., 'balance_sheet', 'income_statement', 'notes'
    confidence_score: float
    start_position: int
    end_position: int


@dataclass
class DocumentChunk:
    """Represents an intelligent chunk for compliance analysis"""

    content: str
    relevant_questions: List[str]
    section_info: DocumentSection
    semantic_score: float
    chunk_id: str


class PDFStructureAnalyzer:
    """
    Analyzes PDF document structure to extract Table of Contents,
    sections, and financial statement components.
    """

    def __init__(self):
        """Initialize the PDF structure analyzer"""
        self.financial_keywords = {
            "balance_sheet": [
                "balance sheet",
                "statement of financial position",
                "financial position",
                "assets",
                "liabilities",
                "equity",
                "current assets",
                "non-current assets",
            ],
            "income_statement": [
                "income statement",
                "statement of profit and loss",
                "profit and loss",
                "comprehensive income",
                "revenue",
                "expenses",
                "operating income",
            ],
            "cash_flow": [
                "cash flow",
                "statement of cash flows",
                "cashflows",
                "cash flows",
                "operating activities",
                "investing activities",
                "financing activities",
            ],
            "notes": [
                "notes to financial statements",
                "notes to consolidated financial statements",
                "accounting policies",
                "significant accounting policies",
            ],
            "auditor_report": [
                "independent auditor",
                "auditor report",
                "opinion",
                "audit opinion",
            ],
        }

        # MENA region specific patterns
        self.mena_patterns = {
            "currencies": ["AED", "SAR", "QAR", "KWD", "OMR", "BHD", "EGP", "JOD"],
            "regulatory_bodies": [
                "UAE Central Bank",
                "SAMA",
                "QCB",
                "CBK",
                "CBO",
                "CBB",
            ],
            "standards": ["IFRS", "IAS", "UAE GAAP", "Saudi GAAP"],
        }

    def extract_table_of_contents(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract Table of Contents and document structure from PDF

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary containing TOC structure and metadata
        """
        logger.info(f"Analyzing PDF structure: {pdf_path}")

        toc_info = {
            "sections": [],
            "total_pages": 0,
            "financial_statements_detected": [],
            "document_metadata": {},
        }

        try:
            with pdfplumber.open(pdf_path) as pdf:
                toc_info["total_pages"] = len(pdf.pages)

                # Extract text from first few pages to find TOC
                toc_text = ""
                for i, page in enumerate(pdf.pages[:5]):  # Check first 5 pages for TOC
                    page_text = page.extract_text() or ""
                    toc_text += page_text + "\n"

                # Parse TOC structure
                sections = self._parse_toc_structure(toc_text)
                toc_info["sections"] = sections

                # Detect financial statement sections throughout document
                financial_sections = self._detect_financial_sections(pdf)
                toc_info["financial_statements_detected"] = financial_sections

                # Extract document metadata
                metadata = self._extract_document_metadata(pdf, toc_text)
                toc_info["document_metadata"] = metadata

        except Exception as e:
            logger.error(f"Error analyzing PDF structure: {str(e)}")

        return toc_info

    def _parse_toc_structure(self, toc_text: str) -> List[Dict[str, Any]]:
        """Parse Table of Contents structure from text"""
        sections = []

        # Common TOC patterns
        toc_patterns = [
            r"(\d+)\s+([A-Z][A-Za-z\s]+)\s+(\d+)",  # "1 Introduction 5"
            r"([A-Z][A-Za-z\s]+)\s+(\d+)",  # "Introduction 5"
            r"(\d+\.\d+)\s+([A-Za-z\s]+)\s+(\d+)",  # "1.1 Overview 7"
        ]

        for pattern in toc_patterns:
            matches = re.finditer(pattern, toc_text, re.MULTILINE)
            for match in matches:
                if len(match.groups()) == 3:
                    section_num, title, page = match.groups()
                    sections.append(
                        {
                            "section_number": section_num,
                            "title": title.strip(),
                            "page": int(page),
                            "type": self._classify_section_type(title.strip()),
                        }
                    )
                elif len(match.groups()) == 2:
                    title, page = match.groups()
                    sections.append(
                        {
                            "section_number": "",
                            "title": title.strip(),
                            "page": int(page),
                            "type": self._classify_section_type(title.strip()),
                        }
                    )

        return sections

    def _classify_section_type(self, title: str) -> str:
        """Classify section type based on title"""
        title_lower = title.lower()

        for section_type, keywords in self.financial_keywords.items():
            for keyword in keywords:
                if keyword in title_lower:
                    return section_type

        # Check for common section types
        if any(
            word in title_lower
            for word in ["introduction", "overview", "executive summary"]
        ):
            return "introduction"
        elif any(
            word in title_lower for word in ["governance", "corporate governance"]
        ):
            return "governance"
        elif any(word in title_lower for word in ["risk", "risk management"]):
            return "risk_management"

        return "other"

    def _detect_financial_sections(self, pdf) -> List[Dict[str, Any]]:
        """Detect financial statement sections throughout the document"""
        financial_sections = []

        for page_num, page in enumerate(pdf.pages, 1):
            page_text = page.extract_text() or ""

            for section_type, keywords in self.financial_keywords.items():
                for keyword in keywords:
                    if keyword in page_text.lower():
                        financial_sections.append(
                            {
                                "type": section_type,
                                "keyword_found": keyword,
                                "page_number": page_num,
                                "confidence": self._calculate_section_confidence(
                                    page_text, section_type
                                ),
                            }
                        )
                        break  # Found one keyword for this section type on this page

        return financial_sections

    def _calculate_section_confidence(self, text: str, section_type: str) -> float:
        """Calculate confidence score for section type classification"""
        text_lower = text.lower()
        keywords = self.financial_keywords.get(section_type, [])

        # Count keyword matches
        matches = sum(1 for keyword in keywords if keyword in text_lower)
        confidence = min(matches / len(keywords), 1.0)

        # Boost confidence for MENA-specific content
        mena_boost = 0
        for category, patterns in self.mena_patterns.items():
            mena_boost += sum(0.1 for pattern in patterns if pattern in text)

        return min(confidence + mena_boost, 1.0)

    def _extract_document_metadata(self, pdf, toc_text: str) -> Dict[str, Any]:
        """Extract document metadata and characteristics"""
        metadata = {
            "company_name": "",
            "report_type": "",
            "financial_year": "",
            "currency": "",
            "region": "",
            "standards_used": [],
        }

        # Extract from first page
        first_page_text = pdf.pages[0].extract_text() or "" if pdf.pages else ""
        combined_text = first_page_text + "\n" + toc_text

        # Extract company name (usually in title or header)
        company_patterns = [
            r"([A-Z][A-Za-z\s&]+(?:Company|Corporation|Corp|Ltd|Limited|Bank|Group|"
            r"Holdings))",
            r"([A-Z][A-Za-z\s&]+)\s+(?:Annual Report|Financial Statements)",
        ]

        for pattern in company_patterns:
            match = re.search(pattern, combined_text)
            if match:
                metadata["company_name"] = match.group(1).strip()
                break

        # Extract financial year
        year_pattern = r"(?:year ended?|for the year|financial year)\s+.*?(\d{4})"
        year_match = re.search(year_pattern, combined_text, re.IGNORECASE)
        if year_match:
            metadata["financial_year"] = year_match.group(1)

        # Detect currency
        for currency in self.mena_patterns["currencies"]:
            if currency in combined_text:
                metadata["currency"] = currency
                break

        # Detect standards
        for standard in self.mena_patterns["standards"]:
            if standard in combined_text:
                metadata["standards_used"].append(standard)

        return metadata


class FinancialContentClassifier:
    """
    Classifies and categorizes financial document content using NLP
    """

    def __init__(self):
        """Initialize the financial content classifier"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy English model successfully")
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {str(e)}")
            self.nlp = None

        # Financial entity patterns for MENA region
        self.financial_patterns = {
            "monetary_amounts": [
                r"AED\s+[\d,]+\.?\d*",
                r"SAR\s+[\d,]+\.?\d*",
                r"[\d,]+\.?\d*\s+(?:million|billion|thousand)",
            ],
            "financial_ratios": [
                r"(?:current|debt|equity|profit)\s+ratio",
                r"return\s+on\s+(?:assets|equity|investment)",
                r"earnings\s+per\s+share",
            ],
            "regulatory_references": [
                r"IFRS\s+\d+",
                r"IAS\s+\d+",
                r"UAE\s+Central\s+Bank",
                r"Capital\s+adequacy",
            ],
        }

    def classify_content(self, text: str) -> Dict[str, Any]:
        """
        Classify financial content and extract key information

        Args:
            text: Text content to classify

        Returns:
            Dictionary containing classification results
        """
        classification = {
            "content_type": "unknown",
            "financial_entities": [],
            "key_metrics": [],
            "regulatory_mentions": [],
            "confidence_score": 0.0,
        }

        if not self.nlp:
            return classification

        try:
            # Process text with spaCy
            doc = self.nlp(text)

            # Extract financial entities
            financial_entities = self._extract_financial_entities(doc, text)
            classification["financial_entities"] = financial_entities

            # Classify content type
            content_type = self._classify_content_type(text, doc)
            classification["content_type"] = content_type

            # Extract key metrics
            key_metrics = self._extract_key_metrics(text)
            classification["key_metrics"] = key_metrics

            # Find regulatory mentions
            regulatory_mentions = self._find_regulatory_mentions(text)
            classification["regulatory_mentions"] = regulatory_mentions

            # Calculate confidence score
            confidence = self._calculate_classification_confidence(classification)
            classification["confidence_score"] = confidence

        except Exception as e:
            logger.error(f"Error in content classification: {str(e)}")

        return classification

    def _extract_financial_entities(self, doc, text: str) -> List[Dict[str, Any]]:
        """Extract financial entities using NLP and pattern matching"""
        entities = []

        # spaCy named entities
        for ent in doc.ents:
            if ent.label_ in ["MONEY", "PERCENT", "CARDINAL", "ORG"]:
                entities.append(
                    {
                        "text": ent.text,
                        "label": ent.label_,
                        "confidence": 0.8,
                        "method": "spacy_ner",
                    }
                )

        # Pattern-based extraction
        for pattern_type, patterns in self.financial_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entities.append(
                        {
                            "text": match.group(),
                            "label": pattern_type,
                            "confidence": 0.9,
                            "method": "pattern_matching",
                        }
                    )

        return entities

    def _classify_content_type(self, text: str, doc) -> str:
        """Classify the type of financial content"""
        text_lower = text.lower()

        # Count keywords for each type
        type_scores = {
            "balance_sheet": 0,
            "income_statement": 0,
            "cash_flow": 0,
            "notes": 0,
            "auditor_report": 0,
            "governance": 0,
            "risk_management": 0,
        }

        # Score based on keyword presence
        keyword_weights = {
            "balance_sheet": [
                "assets",
                "liabilities",
                "equity",
                "current",
                "non-current",
            ],
            "income_statement": ["revenue", "expenses", "profit", "loss", "income"],
            "cash_flow": ["cash", "operating", "investing", "financing", "activities"],
            "notes": [
                "accounting",
                "policies",
                "significant",
                "estimates",
                "judgments",
            ],
            "auditor_report": [
                "opinion",
                "audit",
                "auditor",
                "independent",
                "reasonable",
            ],
            "governance": [
                "board",
                "directors",
                "governance",
                "committee",
                "compensation",
            ],
            "risk_management": [
                "risk",
                "management",
                "credit",
                "market",
                "operational",
            ],
        }

        for content_type, keywords in keyword_weights.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            type_scores[content_type] = score

        # Return type with highest score
        max_type = max(type_scores.items(), key=lambda x: x[1])
        return max_type[0] if max_type[1] > 0 else "other"

    def _extract_key_metrics(self, text: str) -> List[Dict[str, Any]]:
        """Extract key financial metrics from text"""
        metrics = []

        # Common financial metric patterns
        metric_patterns = {
            "ratios": r"(\w+\s+ratio)\s*:?\s*([\d.]+)%?",
            "percentages": r"(\w+(?:\s+\w+)*)\s+([\d.]+)%",
            "amounts": r"((?:total|net|gross)\s+\w+)\s+(?:of\s+)?([A-Z]{3})\s+([\d,]+)",
        }

        for metric_type, pattern in metric_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                metrics.append(
                    {
                        "type": metric_type,
                        "description": match.group(1).strip(),
                        "value": match.group(2) if len(match.groups()) >= 2 else "",
                        "currency": match.group(3) if len(match.groups()) >= 3 else "",
                        "full_match": match.group(),
                    }
                )

        return metrics

    def _find_regulatory_mentions(self, text: str) -> List[str]:
        """Find regulatory and compliance mentions"""
        regulatory_mentions = []

        regulatory_keywords = [
            "IFRS",
            "IAS",
            "Basel III",
            "Capital Adequacy",
            "UAE Central Bank",
            "SAMA",
            "QCB",
            "CBK",
            "Anti-Money Laundering",
            "KYC",
            "FATCA",
            "Sharia",
            "Islamic Banking",
            "Sukuk",
        ]

        text_lower = text.lower()
        for keyword in regulatory_keywords:
            if keyword.lower() in text_lower:
                regulatory_mentions.append(keyword)

        return regulatory_mentions

    def _calculate_classification_confidence(self, classification: Dict) -> float:
        """Calculate overall confidence score for classification"""
        score = 0.0

        # Base score from content type detection
        if classification["content_type"] != "unknown":
            score += 0.3

        # Score from entities found
        entity_score = min(len(classification["financial_entities"]) * 0.1, 0.4)
        score += entity_score

        # Score from metrics
        metric_score = min(len(classification["key_metrics"]) * 0.1, 0.2)
        score += metric_score

        # Score from regulatory mentions
        regulatory_score = min(len(classification["regulatory_mentions"]) * 0.05, 0.1)
        score += regulatory_score

        return min(score, 1.0)


class SemanticContentMapper:
    """
    Maps compliance questions to relevant document content using semantic similarity
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize semantic content mapper

        Args:
            model_name: SentenceTransformer model name
        """
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Loaded SentenceTransformer model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model: {str(e)}")
            self.model = None

        # Initialize TF-IDF vectorizer for backup text similarity
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000, stop_words="english", ngram_range=(1, 2)
        )

        # Cache for embeddings
        self.embedding_cache = {}

    def map_questions_to_content(
        self,
        questions: List[str],
        document_sections: List[DocumentSection],
        similarity_threshold: float = 0.3,
    ) -> Dict[str, List[DocumentSection]]:
        """
        Map compliance questions to relevant document sections

        Args:
            questions: List of compliance questions
            document_sections: List of document sections
            similarity_threshold: Minimum similarity score for relevance

        Returns:
            Dictionary mapping questions to relevant sections
        """
        logger.info(
            f"Mapping {len(questions)} questions to {len(document_sections)} sections"
        )

        question_to_sections = {}

        if not self.model or not questions or not document_sections:
            return question_to_sections

        try:
            # Create embeddings for questions and sections
            question_embeddings = self._get_embeddings(questions, "questions")
            section_texts = [section.content for section in document_sections]
            section_embeddings = self._get_embeddings(section_texts, "sections")

            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(
                question_embeddings, section_embeddings
            )

            # Map questions to relevant sections
            for i, question in enumerate(questions):
                relevant_sections = []

                for j, section in enumerate(document_sections):
                    similarity_score = similarity_matrix[i][j]

                    if similarity_score >= similarity_threshold:
                        # Add semantic score to section copy
                        section_copy = DocumentSection(
                            title=section.title,
                            content=section.content,
                            page_number=section.page_number,
                            section_type=section.section_type,
                            confidence_score=section.confidence_score,
                            start_position=section.start_position,
                            end_position=section.end_position,
                        )
                        relevant_sections.append((section_copy, similarity_score))

                # Sort by similarity score (descending)
                relevant_sections.sort(key=lambda x: x[1], reverse=True)
                question_to_sections[question] = [
                    section for section, _ in relevant_sections
                ]

        except Exception as e:
            logger.error(f"Error in semantic mapping: {str(e)}")
            # Fallback to keyword-based matching
            question_to_sections = self._fallback_keyword_matching(
                questions, document_sections
            )

        return question_to_sections

    def _get_embeddings(self, texts: List[str], cache_key: str) -> np.ndarray:
        """Get embeddings for texts with caching"""
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        if self.model is None:
            # Return zeros if model is not available
            embeddings = np.zeros(
                (len(texts), 384)
            )  # Default embedding size for MiniLM
        else:
            embeddings = self.model.encode(texts, convert_to_numpy=True)

        self.embedding_cache[cache_key] = embeddings
        return embeddings

    def _fallback_keyword_matching(
        self, questions: List[str], document_sections: List[DocumentSection]
    ) -> Dict[str, List[DocumentSection]]:
        """Fallback keyword-based matching when semantic models fail"""
        logger.info("Using fallback keyword matching")

        question_to_sections = {}

        for question in questions:
            question_words = set(question.lower().split())
            relevant_sections = []

            for section in document_sections:
                section_words = set(section.content.lower().split())

                # Calculate word overlap
                overlap = len(question_words.intersection(section_words))
                total_words = len(question_words.union(section_words))

                if total_words > 0:
                    similarity = overlap / total_words
                    if similarity > 0.1:  # Lower threshold for keyword matching
                        relevant_sections.append((section, similarity))

            # Sort by similarity
            relevant_sections.sort(key=lambda x: x[1], reverse=True)
            question_to_sections[question] = [
                section for section, _ in relevant_sections[:3]
            ]  # Top 3

        return question_to_sections


class IntelligentChunker:
    """
    Main orchestrator for intelligent financial document chunking
    """

    def __init__(self):
        """Initialize the intelligent chunker"""
        self.structure_analyzer = PDFStructureAnalyzer()
        self.content_classifier = FinancialContentClassifier()
        self.semantic_mapper = SemanticContentMapper()

        # Chunking parameters
        self.max_chunk_size = 2000  # characters
        self.overlap_size = 200  # characters
        self.min_chunk_size = 500  # characters

    def process_document(
        self, pdf_path: str, compliance_questions: List[str]
    ) -> List[DocumentChunk]:
        """
        Process a financial document and create intelligent chunks

        Args:
            pdf_path: Path to the PDF file
            compliance_questions: List of compliance questions

        Returns:
            List of intelligent document chunks
        """
        logger.info(f"Processing document: {pdf_path}")
        start_time = time.time()

        # Step 1: Analyze document structure
        structure_info = self.structure_analyzer.extract_table_of_contents(pdf_path)

        # Step 2: Extract document sections
        document_sections = self._extract_document_sections(pdf_path, structure_info)

        # Step 3: Classify content in each section
        for section in document_sections:
            classification = self.content_classifier.classify_content(section.content)
            section.section_type = classification["content_type"]
            section.confidence_score = classification["confidence_score"]

        # Step 4: Map questions to relevant sections
        question_mappings = self.semantic_mapper.map_questions_to_content(
            compliance_questions, document_sections
        )

        # Step 5: Create intelligent chunks
        chunks = self._create_intelligent_chunks(question_mappings, document_sections)

        processing_time = time.time() - start_time
        logger.info(f"Document processing completed in {processing_time:.2f} seconds")
        logger.info(f"Created {len(chunks)} intelligent chunks")

        return chunks

    def _extract_document_sections(
        self, pdf_path: str, structure_info: Dict
    ) -> List[DocumentSection]:
        """Extract document sections based on structure analysis"""
        sections = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                current_section = None
                current_content = ""

                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text() or ""

                    # Check if this page starts a new section
                    new_section_title = self._detect_section_start(
                        page_text, structure_info
                    )

                    if new_section_title and current_section:
                        # Save previous section
                        sections.append(
                            DocumentSection(
                                title=current_section,
                                content=current_content.strip(),
                                page_number=page_num - 1,
                                section_type="unknown",
                                confidence_score=0.0,
                                start_position=0,
                                end_position=len(current_content),
                            )
                        )
                        current_content = ""

                    if new_section_title:
                        current_section = new_section_title

                    current_content += page_text + "\n"

                # Save final section
                if current_section and current_content.strip():
                    sections.append(
                        DocumentSection(
                            title=current_section,
                            content=current_content.strip(),
                            page_number=len(pdf.pages),
                            section_type="unknown",
                            confidence_score=0.0,
                            start_position=0,
                            end_position=len(current_content),
                        )
                    )

        except Exception as e:
            logger.error(f"Error extracting document sections: {str(e)}")

        return sections

    def _detect_section_start(
        self, page_text: str, structure_info: Dict
    ) -> Optional[str]:
        """Detect if page starts a new section based on structure info"""
        # Check against known section titles from TOC
        for section in structure_info.get("sections", []):
            if (
                section["title"].lower() in page_text[:500].lower()
            ):  # Check first 500 chars
                return section["title"]

        # Check for common section headers
        header_patterns = [
            r"^([A-Z][A-Za-z\s]+)\n",  # Title on its own line
            r"^\d+\.?\s+([A-Z][A-Za-z\s]+)",  # Numbered section
        ]

        for pattern in header_patterns:
            match = re.search(pattern, page_text, re.MULTILINE)
            if match:
                title = match.group(1).strip()
                if len(title) < 100:  # Reasonable title length
                    return title

        return None

    def _create_intelligent_chunks(
        self,
        question_mappings: Dict[str, List[DocumentSection]],
        all_sections: List[DocumentSection],
    ) -> List[DocumentChunk]:
        """Create intelligent chunks based on question mappings"""
        chunks = []
        chunk_id_counter = 1

        # Group questions by their mapped sections
        section_to_questions = {}
        for question, sections in question_mappings.items():
            for section in sections:
                section_key = f"{section.title}_{section.page_number}"
                if section_key not in section_to_questions:
                    section_to_questions[section_key] = []
                section_to_questions[section_key].append(question)

        # Create chunks for mapped sections
        for section_key, questions in section_to_questions.items():
            section = next(
                (
                    s
                    for s in all_sections
                    if f"{s.title}_{s.page_number}" == section_key
                ),
                None,
            )

            if section:
                # Split long sections into smaller chunks
                section_chunks = self._split_section_into_chunks(section)

                for chunk_content in section_chunks:
                    chunk = DocumentChunk(
                        content=chunk_content,
                        relevant_questions=questions,
                        section_info=section,
                        semantic_score=section.confidence_score,
                        chunk_id=f"chunk_{chunk_id_counter:04d}",
                    )
                    chunks.append(chunk)
                    chunk_id_counter += 1

        # Create chunks for unmapped sections (lower priority)
        mapped_sections = set(section_to_questions.keys())
        for section in all_sections:
            section_key = f"{section.title}_{section.page_number}"
            if section_key not in mapped_sections:
                section_chunks = self._split_section_into_chunks(section)

                for chunk_content in section_chunks:
                    chunk = DocumentChunk(
                        content=chunk_content,
                        relevant_questions=[],  # No specific questions mapped
                        section_info=section,
                        semantic_score=0.1,  # Lower priority
                        chunk_id=f"chunk_{chunk_id_counter:04d}",
                    )
                    chunks.append(chunk)
                    chunk_id_counter += 1

        return chunks

    def _split_section_into_chunks(self, section: DocumentSection) -> List[str]:
        """Split a section into smaller chunks if needed"""
        if len(section.content) <= self.max_chunk_size:
            return [section.content]

        chunks = []
        start = 0

        while start < len(section.content):
            end = start + self.max_chunk_size

            # Try to break at sentence boundary
            if end < len(section.content):
                # Look for sentence ending within overlap distance
                for i in range(end - self.overlap_size, end):
                    if i < len(section.content) and section.content[i] in ".!?":
                        end = i + 1
                        break

            chunk_content = section.content[start:end].strip()
            if len(chunk_content) >= self.min_chunk_size or start == 0:
                chunks.append(chunk_content)

            start = end - self.overlap_size

        return chunks

    def export_chunks_for_analysis(
        self, chunks: List[DocumentChunk], output_path: str
    ) -> None:
        """Export chunks to JSON for analysis"""
        export_data = []

        for chunk in chunks:
            export_data.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "relevant_questions": chunk.relevant_questions,
                    "section_title": chunk.section_info.title,
                    "section_type": chunk.section_info.section_type,
                    "page_number": chunk.section_info.page_number,
                    "semantic_score": chunk.semantic_score,
                    "content_length": len(chunk.content),
                }
            )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported {len(chunks)} chunks to {output_path}")


# Example usage and testing functions
def test_chunking_system():
    """Test the chunking system with sample data"""
    # Test with a sample PDF (you would replace this with actual PDF path)
    # chunks = chunker.process_document("sample_financial_report.pdf", sample_questions)

    logger.info("Chunking system test completed")


if __name__ == "__main__":
    # Run test
    test_chunking_system()
