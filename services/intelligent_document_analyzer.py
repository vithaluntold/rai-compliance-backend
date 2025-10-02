"""
Intelligent Document Analyzer for Enhanced Evidence Detection
Addresses the issue where AI returns generic policy content instead of substantive disclosures
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DocumentSection:
    note_id: str
    title: str
    content: str
    section_type: str
    relevance_score: float
    page_numbers: List[int]
    start_position: int
    end_position: int


@dataclass
class IntelligentEvidence:
    text: str
    source_section: str
    quality_score: int
    evidence_type: str
    confidence_level: float
    page_reference: str
    reasoning: str


class FinancialDocumentIntelligence:
    def __init__(self):
        """Initialize with standard-to-section mapping and quality indicators"""
        self.standard_section_mapping = {
            "IAS 1": [
                "presentation",
                "financial statements",
                "statement preparation",
                "accounting policies",
            ],
            "IAS 2": ["inventories", "inventory", "cost of goods", "stock valuation"],
            "IAS 7": [
                "cash flows",
                "cash flow",
                "operating activities",
                "investing activities",
                "financing activities",
            ],
            "IAS 8": [
                "accounting policies",
                "changes in estimates",
                "errors",
                "prior period adjustments",
            ],
            "IAS 10": [
                "events after reporting",
                "subsequent events",
                "post balance sheet",
            ],
            "IAS 12": ["income taxes", "tax expense", "deferred tax", "current tax"],
            "IAS 16": [
                "property plant equipment",
                "ppe",
                "fixed assets",
                "depreciation",
                "asset impairment",
            ],
            "IAS 19": [
                "employee benefits",
                "pension",
                "retirement benefits",
                "post employment",
                "defined benefit",
            ],
            "IAS 20": ["government grants", "government assistance", "subsidies"],
            "IAS 21": [
                "foreign exchange",
                "foreign currency",
                "translation differences",
            ],
            "IAS 23": ["borrowing costs", "interest costs", "capitalization"],
            "IAS 24": ["related party", "related parties", "key management personnel"],
            "IAS 26": ["retirement benefit plans", "pension plans"],
            "IAS 27": ["separate financial statements", "parent company"],
            "IAS 28": ["associates", "joint ventures", "equity method"],
            "IAS 29": ["hyperinflationary", "inflation accounting"],
            "IAS 32": ["financial instruments presentation", "equity classification"],
            "IAS 33": ["earnings per share", "eps", "diluted earnings"],
            "IAS 34": ["interim reporting", "quarterly reports"],
            "IAS 36": [
                "impairment",
                "recoverable amount",
                "value in use",
                "cash generating units",
            ],
            "IAS 37": ["provisions", "contingent liabilities", "contingent assets"],
            "IAS 38": ["intangible assets", "goodwill", "development costs"],
            "IAS 39": ["financial instruments recognition", "hedge accounting"],
            "IAS 40": ["investment property", "fair value model"],
            "IAS 41": ["agriculture", "biological assets", "agricultural produce"],
            "IFRS 1": ["first time adoption", "transition to ifrs"],
            "IFRS 2": ["share based payment", "stock options", "equity compensation"],
            "IFRS 3": ["business combinations", "acquisitions", "goodwill"],
            "IFRS 5": ["discontinued operations", "held for sale"],
            "IFRS 6": ["exploration assets", "mineral resources"],
            "IFRS 7": [
                "financial instruments disclosures",
                "risk disclosures",
                "fair value",
            ],
            "IFRS 8": ["operating segments", "reportable segments"],
            "IFRS 9": [
                "financial instruments",
                "expected credit losses",
                "classification measurement",
            ],
            "IFRS 10": ["consolidated financial statements", "control assessment"],
            "IFRS 11": ["joint arrangements", "joint operations", "joint ventures"],
            "IFRS 12": ["interests in other entities", "subsidiaries disclosure"],
            "IFRS 13": ["fair value measurement", "valuation techniques"],
            "IFRS 14": ["regulatory deferral accounts"],
            "IFRS 15": [
                "revenue contracts",
                "performance obligations",
                "contract assets",
            ],
            "IFRS 16": ["leases", "right of use assets", "lease liabilities"],
            "IFRS 17": ["insurance contracts", "insurance liabilities"],
            "IFRS 18": ["presentation disclosure", "financial statement presentation"],
        }

        self.policy_keywords = [
            "accounting policy",
            "significant accounting policies",
            "basis of preparation",
            "critical accounting estimates",
            "summary of significant",
            "accounting principles",
        ]

        self.substantive_keywords = [
            "reconciliation",
            "movement",
            "analysis",
            "breakdown",
            "maturity analysis",
            "aging",
            "fair value hierarchy",
            "sensitivity analysis",
            "risk analysis",
        ]

        self.evidence_quality_indicators = {
            "high": [
                "reconciliation",
                "detailed breakdown",
                "movement analysis",
                "fair value hierarchy",
            ],
            "medium": ["disclosure", "amounts", "balances", "commitments"],
            "low": ["policy", "method", "approach", "basis"],
        }

    def parse_document_structure(self, document_text: str) -> List[DocumentSection]:
        """Parse the document to identify notes and sections with their boundaries"""
        sections = []

        # Multiple patterns to catch different note formats
        note_patterns = [
            r"(?i)note\s+(\d+|[a-z]+)\s*[:\-\.]?\s*([^\n]+?)(?=\n)",
            r"(?i)(\d+)\.\s+([^\n]+?)(?=\n)",
            r"(?i)(appendix\s+[a-z])\s*[:\-\.]?\s*([^\n]+?)(?=\n)",
        ]

        for pattern in note_patterns:
            matches = list(re.finditer(pattern, document_text))
            for i, match in enumerate(matches):
                note_id = match.group(1)
                title = match.group(2).strip()
                start_pos = match.start()

                # Determine section boundaries
                if i + 1 < len(matches):
                    end_pos = matches[i + 1].start()
                else:
                    end_pos = len(document_text)

                content = document_text[start_pos:end_pos]
                section_type = self.classify_section_type(title, content)
                page_numbers = self.extract_page_numbers(content)

                sections.append(
                    DocumentSection(
                        note_id=note_id,
                        title=title,
                        content=content,
                        section_type=section_type,
                        relevance_score=0.0,
                        page_numbers=page_numbers,
                        start_position=start_pos,
                        end_position=end_pos,
                    )
                )

        logger.info(f"Parsed {len(sections)} document sections")
        return sections

    def classify_section_type(self, title: str, content: str) -> str:
        """Classify section as policy, substantive, or general based on content"""
        title_lower = title.lower()
        content_sample = content[:1000].lower()

        # Score based on keyword presence
        policy_score = sum(
            1
            for keyword in self.policy_keywords
            if keyword in title_lower or keyword in content_sample
        )
        substantive_score = sum(
            1 for keyword in self.substantive_keywords if keyword in content_sample
        )

        if policy_score > substantive_score and policy_score > 0:
            return "policy"
        elif substantive_score > 0:
            return "substantive"
        else:
            return "general"

    def extract_page_numbers(self, content: str) -> List[int]:
        """Extract page number references from content"""
        page_pattern = r"(?i)page\s+(\d+)"
        matches = re.findall(page_pattern, content)
        return [int(match) for match in matches]

    def calculate_section_relevance(
        self, section: DocumentSection, standard_id: str
    ) -> float:
        """Calculate how relevant a section is to a specific accounting standard"""
        if standard_id not in self.standard_section_mapping:
            return 0.0

        target_keywords = self.standard_section_mapping[standard_id]
        title_content = f"{section.title} {section.content[:500]}".lower()

        relevance_score = 0.0
        total_keywords = len(target_keywords)

        for keyword in target_keywords:
            if keyword in title_content:
                # Higher weight for title matches
                if keyword in section.title.lower():
                    relevance_score += 2.0
                else:
                    relevance_score += 1.0

        return min(relevance_score / total_keywords, 1.0) if total_keywords > 0 else 0.0

    def find_cross_references(
        self, content: str, all_sections: List[DocumentSection]
    ) -> List[DocumentSection]:
        """Find sections referenced by cross-references in the content"""
        cross_ref_patterns = [
            r"(?i)see\s+note\s+(\d+|[a-z]+)",
            r"(?i)refer\s+to\s+note\s+(\d+|[a-z]+)",
            r"(?i)detailed\s+in\s+note\s+(\d+|[a-z]+)",
            r"(?i)note\s+(\d+|[a-z]+)\s+provides",
        ]

        referenced_sections = []
        for pattern in cross_ref_patterns:
            matches = re.findall(pattern, content)
            for note_ref in matches:
                for section in all_sections:
                    if section.note_id.lower() == note_ref.lower():
                        referenced_sections.append(section)
                        break

        return referenced_sections

    def extract_evidence_quality_score(self, text: str, section_type: str) -> int:
        """Calculate evidence quality score based on content characteristics"""
        base_score = 50

        # Adjust based on section type
        if section_type == "policy":
            base_score -= 30
        elif section_type == "substantive":
            base_score += 25

        text_lower = text.lower()

        # Adjust based on quality indicators
        for quality_level, indicators in self.evidence_quality_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    if quality_level == "high":
                        base_score += 20
                    elif quality_level == "medium":
                        base_score += 10
                    elif quality_level == "low":
                        base_score -= 15

        # Bonus for numerical data
        if re.search(r"\$[\d,]+|\d+\.\d+%|\d{1,3}(,\d{3})*", text):
            base_score += 15

        # Penalty for very short content
        if len(text.strip()) < 200:
            base_score -= 10

        return max(0, min(100, base_score))

    def intelligent_evidence_search(
        self, compliance_question: str, document_text: str, standard_id: str
    ) -> List[IntelligentEvidence]:
        """Main method for intelligent evidence search"""
        logger.info(f"Starting intelligent evidence search for {standard_id}")

        # Parse document structure
        sections = self.parse_document_structure(document_text)
        evidence_results = []

        # Calculate relevance scores for all sections
        for section in sections:
            section.relevance_score = self.calculate_section_relevance(
                section, standard_id
            )

        # Get sections most relevant to the standard
        relevant_sections = [s for s in sections if s.relevance_score > 0.3]
        relevant_sections.sort(key=lambda x: x.relevance_score, reverse=True)

        logger.info(
            f"Found {len(relevant_sections)} relevant sections for {standard_id}"
        )

        # Process relevant sections first
        for section in relevant_sections[:5]:
            quality_score = self.extract_evidence_quality_score(
                section.content, section.section_type
            )

            if quality_score > 30:
                # Find cross-referenced content
                cross_referenced = self.find_cross_references(section.content, sections)

                # Prepare evidence text
                evidence_text = section.content[:2000]
                if cross_referenced:
                    evidence_text += "\n\nCross-referenced content:\n"
                    for ref_section in cross_referenced[:2]:
                        evidence_text += (
                            f"{ref_section.title}: {ref_section.content[:500]}\n"
                        )

                # Calculate confidence
                confidence = min(section.relevance_score * quality_score / 100, 0.95)

                # Prepare page reference
                page_ref = (
                    f"Pages {'-'.join(map(str, section.page_numbers))}"
                    if section.page_numbers
                    else "Page reference not found"
                )

                # Generate reasoning
                reasoning = self.generate_evidence_reasoning(
                    section, quality_score, standard_id
                )

                evidence_results.append(
                    IntelligentEvidence(
                        text=evidence_text,
                        source_section=f"Note {section.note_id}: {section.title}",
                        quality_score=quality_score,
                        evidence_type=section.section_type,
                        confidence_level=confidence,
                        page_reference=page_ref,
                        reasoning=reasoning,
                    )
                )

        # Fallback to non-relevant but high-quality sections if needed
        if len(evidence_results) < 2:
            non_relevant_sections = [
                s
                for s in sections
                if s.relevance_score <= 0.3 and s.section_type != "policy"
            ]
            for section in non_relevant_sections[:2]:
                quality_score = self.extract_evidence_quality_score(
                    section.content, section.section_type
                )
                if quality_score > 50:
                    evidence_results.append(
                        IntelligentEvidence(
                            text=section.content[:1500],
                            source_section=f"Note {section.note_id}: {section.title}",
                            quality_score=quality_score
                            - 20,  # Penalty for non-relevance
                            evidence_type="fallback",
                            confidence_level=0.4,
                            page_reference="Fallback evidence",
                            reasoning="Fallback evidence from non-standard-specific section",
                        )
                    )

        # Sort by quality and confidence
        evidence_results.sort(
            key=lambda x: (x.quality_score, x.confidence_level), reverse=True
        )

        logger.info(f"Generated {len(evidence_results)} intelligent evidence items")
        return evidence_results

    def generate_evidence_reasoning(
        self, section: DocumentSection, quality_score: int, standard_id: str
    ) -> str:
        """Generate human-readable reasoning for evidence selection"""
        reasoning_parts = []

        if section.relevance_score > 0.7:
            reasoning_parts.append(f"High relevance to {standard_id}")
        elif section.relevance_score > 0.4:
            reasoning_parts.append(f"Medium relevance to {standard_id}")

        if section.section_type == "substantive":
            reasoning_parts.append("Contains substantive disclosure content")
        elif section.section_type == "policy":
            reasoning_parts.append("Contains policy information (lower quality)")

        if quality_score > 70:
            reasoning_parts.append("High-quality evidence with detailed disclosures")
        elif quality_score > 50:
            reasoning_parts.append("Medium-quality evidence")
        else:
            reasoning_parts.append("Lower-quality evidence")

        return "; ".join(reasoning_parts)


def enhance_compliance_analysis(
    compliance_question: str,
    document_text: str,
    standard_id: str,
    existing_chunks: Optional[List[str]] = None,
) -> Dict:
    """Main function to enhance compliance analysis with intelligent evidence search"""
    try:
        analyzer = FinancialDocumentIntelligence()
        intelligent_evidence = analyzer.intelligent_evidence_search(
            compliance_question, document_text, standard_id
        )

        best_evidence = intelligent_evidence[0] if intelligent_evidence else None

        return {
            "intelligent_evidence": [
                {
                    "text": evidence.text,
                    "source_section": evidence.source_section,
                    "quality_score": evidence.quality_score,
                    "evidence_type": evidence.evidence_type,
                    "confidence_level": evidence.confidence_level,
                    "page_reference": evidence.page_reference,
                    "reasoning": evidence.reasoning,
                }
                for evidence in intelligent_evidence[:3]
            ],
            "primary_evidence": (
                best_evidence.text
                if best_evidence
                else (existing_chunks[0] if existing_chunks else "")
            ),
            "evidence_quality_assessment": {
                "overall_quality": best_evidence.quality_score if best_evidence else 0,
                "confidence_level": (
                    best_evidence.confidence_level if best_evidence else 0.0
                ),
                "source_type": (
                    best_evidence.evidence_type if best_evidence else "unknown"
                ),
                "is_policy_based": (
                    best_evidence.evidence_type == "policy" if best_evidence else True
                ),
                "evidence_source": (
                    best_evidence.source_section if best_evidence else "Unknown section"
                ),
            },
            "analysis_summary": {
                "total_evidence_sources": len(intelligent_evidence),
                "substantive_evidence_count": len(
                    [
                        e
                        for e in intelligent_evidence
                        if e.evidence_type == "substantive"
                    ]
                ),
                "policy_evidence_count": len(
                    [e for e in intelligent_evidence if e.evidence_type == "policy"]
                ),
                "recommendation": (
                    "High confidence"
                    if (best_evidence and best_evidence.confidence_level > 0.7)
                    else "Manual review recommended"
                ),
            },
        }

    except Exception as e:
        logger.error(f"Error in intelligent evidence analysis: {str(e)}")
        # Fallback to existing chunks if enhancement fails
        return {
            "intelligent_evidence": [],
            "primary_evidence": existing_chunks[0] if existing_chunks else "",
            "evidence_quality_assessment": {
                "overall_quality": 0,
                "confidence_level": 0.0,
                "source_type": "fallback",
                "is_policy_based": True,
                "evidence_source": "Fallback to original system",
            },
            "analysis_summary": {
                "total_evidence_sources": 0,
                "substantive_evidence_count": 0,
                "policy_evidence_count": 0,
                "recommendation": "Enhancement failed - manual review required",
            },
        }
