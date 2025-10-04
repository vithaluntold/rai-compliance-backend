"""
Document Analysis Module for Intelligent Metadata Extraction
Provides ZAP vs Comprehensive analysis strategies for PDF documents
"""

import time
from typing import Any, Dict, List, Tuple


class KeywordExtractor:
    """Handles keyword-based content targeting for metadata extraction"""

    def __init__(self):
        self.target_keywords = {
            "company_identifiers": [
                "company name",
                "corporation",
                "incorporated",
                "inc.",
                "ltd.",
                "limited",
                "plc",
                "pjsc",
                "llc",
                "company",
                "entity name",
            ],
            "business_nature": [
                "nature of business",
                "principal activities",
                "business activities",
                "operations",
                "primary business",
                "main activities",
                "engaged in",
                "business description",
                "activities of the company",
            ],
            "geographic_info": [
                "registered office",
                "headquarters",
                "principal place of business",
                "country of incorporation",
                "domicile",
                "registered address",
                "location",
                "operations in",
                "based in",
            ],
        }

        self.section_headers = [
            "company information",
            "corporate profile",
            "about the company",
            "directors report",
            "corporate governance",
            "company overview",
            "business overview",
            "executive summary",
        ]

    def find_keyword_sections(self, text: str) -> Dict[str, List[Tuple[int, str]]]:
        """Find text sections containing target keywords with their positions"""
        keyword_sections: Dict[str, List[Tuple[int, str]]] = {
            "company_identifiers": [],
            "business_nature": [],
            "geographic_info": [],
        }

        text_lower = text.lower()

        for category, keywords in self.target_keywords.items():
            for keyword in keywords:
                matches = []
                start = 0
                while True:
                    pos = text_lower.find(keyword.lower(), start)
                    if pos == -1:
                        break

                    # Extract context around the keyword (500 chars before and after)
                    context_start = max(0, pos - 500)
                    context_end = min(len(text), pos + 500)
                    context = text[context_start:context_end]

                    matches.append((pos, context))
                    start = pos + 1

                keyword_sections[category].extend(matches)

        return keyword_sections

    def identify_document_sections(self, text: str) -> List[Tuple[str, int, int]]:
        """Identify major document sections based on headers"""
        sections = []
        text_lower = text.lower()

        for header in self.section_headers:
            start = 0
            while True:
                pos = text_lower.find(header.lower(), start)
                if pos == -1:
                    break

                # Find the end of this section (next header or 2000 chars)
                section_end = pos + 2000
                for other_header in self.section_headers:
                    if other_header != header:
                        next_pos = text_lower.find(other_header.lower(), pos + 100)
                        if next_pos != -1 and next_pos < section_end:
                            section_end = next_pos

                sections.append((header, pos, min(section_end, len(text))))
                start = pos + 1

        return sections


class DocumentAnalyzer:
    """Main document analysis class with ZAP and Comprehensive strategies"""

    def __init__(self, analysis_mode: str = "zap"):
        self.mode = analysis_mode
        self.keyword_extractor = KeywordExtractor()
        self.extraction_log: List[str] = []

    def extract_content(self, pdf_text: str) -> Dict[str, Any]:
        """Main extraction method that routes to appropriate strategy"""
        start_time = time.time()

        if self.mode == "zap":
            result = self.zap_extraction(pdf_text)
        else:
            result = self.comprehensive_extraction(pdf_text)

        processing_time = time.time() - start_time

        return {
            "extracted_content": result["content"],
            "extraction_metadata": {
                "analysis_mode": self.mode,
                "processing_time_seconds": round(processing_time, 2),
                "content_length": len(result["content"]),
                "sections_analyzed": result["sections_analyzed"],
                "keywords_discovered": result["keywords_discovered"],
                "extraction_method": result["method"],
            },
        }

    def zap_extraction(self, text: str) -> Dict[str, Any]:
        """ZAP analysis: First 3000 chars + targeted keyword sections"""
        self.extraction_log.append("Starting ZAP Analysis...")

        # Strategy 1: First 3000 characters (cover page + executive summary)
        primary_content = text[:3000]
        self.extraction_log.append("Extracted first 3000 characters")

        # Strategy 2: Find high-value keyword sections
        keyword_sections = self.keyword_extractor.find_keyword_sections(text)

        additional_content = ""
        keywords_found = []

        # Extract best keyword matches (up to 2000 additional chars)
        for category, matches in keyword_sections.items():
            if matches and len(additional_content) < 2000:
                # Take the best match (first occurrence usually most relevant)
                best_match = matches[0][1]
                additional_content += f"\n\n{best_match}"
                keywords_found.append(f"{category}: {len(matches)} matches")
                self.extraction_log.append(f"Found {len(matches)} {category} keywords")

        final_content = primary_content + additional_content

        # Ensure we don't exceed 5000 chars for zap mode
        if len(final_content) > 5000:
            final_content = final_content[:5000]

        return {
            "content": final_content,
            "method": "ZAP: First 3K chars + keyword targeting",
            "sections_analyzed": [
                "cover_page",
                "executive_summary",
                "keyword_sections",
            ],
            "keywords_discovered": keywords_found,
        }

    def comprehensive_extraction(self, text: str) -> Dict[str, Any]:
        """Comprehensive analysis: Multi-section intelligent sampling"""
        self.extraction_log.append("Starting Comprehensive Analysis...")

        content_parts = []
        sections_analyzed = []
        keywords_found = []

        # Strategy 1: Document structure analysis
        document_sections = self.keyword_extractor.identify_document_sections(text)
        self.extraction_log.append(
            f"Identified {len(document_sections)} document sections"
        )

        # Strategy 2: Strategic sampling from different parts

        # Part 1: First 2000 chars (cover + summary)
        content_parts.append(text[:2000])
        sections_analyzed.append("document_header")
        self.extraction_log.append("Extracted document header (2000 chars)")

        # Part 2: Target specific sections if found
        priority_sections = [
            "company information",
            "corporate profile",
            "directors report",
        ]
        for section_name, start_pos, end_pos in document_sections:
            if any(priority in section_name.lower() for priority in priority_sections):
                section_content = text[start_pos:end_pos][
                    :1500
                ]  # Max 1500 chars per section
                content_parts.append(
                    f"\n\n--- {section_name.upper()} SECTION ---\n{section_content}"
                )
                sections_analyzed.append(section_name)
                self.extraction_log.append(f"Extracted {section_name} section")

        # Part 3: Keyword-targeted extraction
        keyword_sections = self.keyword_extractor.find_keyword_sections(text)

        remaining_space = 8000 - sum(len(part) for part in content_parts)

        for category, matches in keyword_sections.items():
            if matches and remaining_space > 500:
                # Take top 2 matches for each category
                for i, (pos, context) in enumerate(matches[:2]):
                    if remaining_space > 300:
                        content_parts.append(
                            f"\n\n--- {category.upper()} CONTEXT ---\n{context[:300]}"
                        )
                        remaining_space -= 300
                        keywords_found.append(f"{category}: match {i + 1}")
                        self.extraction_log.append(f"Added {category} context")

        # Part 4: End of document sampling (sometimes contains summary info)
        if remaining_space > 500 and len(text) > 5000:
            end_content = text[-1000:]  # Last 1000 chars
            content_parts.append(f"\n\n--- DOCUMENT END ---\n{end_content}")
            sections_analyzed.append("document_end")
            self.extraction_log.append("Added document end section")

        final_content = "".join(content_parts)

        # Ensure we don't exceed 8000 chars for comprehensive mode
        if len(final_content) > 8000:
            final_content = final_content[:8000]

        return {
            "content": final_content,
            "method": "Comprehensive: Multi-section intelligent sampling",
            "sections_analyzed": sections_analyzed,
            "keywords_discovered": keywords_found,
        }

    def get_extraction_log(self) -> List[str]:
        """Return the extraction process log for debugging/feedback"""
        return self.extraction_log

    def estimate_processing_time(self) -> int:
        """Return estimated processing time in seconds"""
        return 8 if self.mode == "zap" else 20


# Utility function for backward compatibility
def extract_metadata_content(pdf_text: str, analysis_mode: str = "zap") -> str:
    """
    Utility function that maintains compatibility with existing code
    while providing intelligent extraction
    """
    analyzer = DocumentAnalyzer(analysis_mode)
    result = analyzer.extract_content(pdf_text)
    return result["extracted_content"]
