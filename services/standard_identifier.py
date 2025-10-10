"""
Standard Identifier Service - Taxonomy-Based Accounting Standards Tagging

This service processes Notes to Accounts and tags every sentence with relevant 
IAS/IFRS accounting standards using the IFRSAT-2025 taxonomy library.

Key Features:
1. Extracts Notes to Accounts (excluding financial statements and audit reports)
2. Tags every sentence with applicable accounting standards
3. Handles multiple standards per sentence
4. Provides detailed metadata (page, note, subsection)
5. Uses taxonomy keyword matching for accurate standard identification
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Import taxonomy parser
try:
    from taxonomy.xml_taxonomy_parser import XBRLTaxonomyParser
    TAXONOMY_AVAILABLE = True
except ImportError:
    XBRLTaxonomyParser = None
    TAXONOMY_AVAILABLE = False
    logger.warning("⚠️ Taxonomy parser not available")


@dataclass
class StandardTag:
    """Represents an accounting standard tag for a sentence"""
    standard_code: str          # e.g., "IAS 16", "IFRS 9"
    standard_name: str          # e.g., "Property, Plant and Equipment"
    confidence_score: float     # 0.0 to 1.0
    matching_keywords: List[str] # Keywords that triggered this standard
    taxonomy_concepts: List[str] # Relevant taxonomy concepts


@dataclass
class TaggedSentence:
    """Represents a sentence with its accounting standard tags and metadata"""
    sentence_text: str
    sentence_index: int
    standards: List[StandardTag]
    page_number: int
    note_number: Optional[str]
    subsection: Optional[str]
    context: str                # Surrounding context for the sentence


@dataclass
class NotesSection:
    """Represents a Notes to Accounts section"""
    section_title: str
    note_number: str
    subsection: Optional[str]
    content: str
    page_numbers: List[int]
    tagged_sentences: List[TaggedSentence]


class StandardIdentifier:
    """
    Taxonomy-based accounting standards identifier for Notes to Accounts
    
    Tags every sentence in notes with relevant IAS/IFRS standards using
    comprehensive taxonomy keyword matching.
    """
    
    def __init__(self):
        """Initialize with taxonomy integration and standard mapping"""
        
        # Initialize taxonomy parser
        self.taxonomy_data = None
        self.concept_index = {}
        
        if TAXONOMY_AVAILABLE and XBRLTaxonomyParser is not None:
            try:
                taxonomy_path = Path(__file__).parent.parent / "taxonomy" / "IFRSAT-2025" / "IFRSAT-2025"
                if taxonomy_path.exists():
                    self.taxonomy_parser = XBRLTaxonomyParser(str(taxonomy_path))
                    self.taxonomy_data = self.taxonomy_parser.load_ifrs_taxonomy()
                    self._build_concept_index()
                    logger.info("✅ Taxonomy integration successful for standard identification")
                else:
                    logger.warning("⚠️ Taxonomy directory not found")
            except Exception as e:
                logger.error(f"❌ Failed to initialize taxonomy: {e}")
        else:
            self.taxonomy_parser = None
            self.taxonomy_data = None
        
        # Load fallback keyword mapping from our Phoenix analysis
        self._load_standard_keywords()
        
        # Header-to-standard mapping for section-based tagging
        self._load_header_standard_mapping()
        
        # Financial document structure patterns (in proper order)
        self.financial_statement_markers = [
            r"(?:consolidated\s+)?(?:statement|position)\s+of\s+financial\s+position",
            r"(?:consolidated\s+)?balance\s+sheets?",
            r"(?:consolidated\s+)?(?:statement|income)\s+of\s+(?:profit\s+and\s+loss|comprehensive\s+income)",
            r"(?:consolidated\s+)?(?:statement|position)\s+of\s+cash\s+flows?",
            r"(?:consolidated\s+)?(?:statement|position)\s+of\s+changes\s+in\s+equity",
            r"statement\s+of\s+retained\s+earnings",
        ]
        
        self.notes_section_markers = [
            r"notes?\s+to\s+(?:the\s+)?(?:consolidated\s+)?financial\s+statements?",
            r"notes?\s+to\s+(?:the\s+)?(?:consolidated\s+)?accounts?",
            r"accounting\s+policies\s+and\s+notes?",
            r"explanatory\s+notes?",
        ]
        
        self.directors_report_markers = [
            r"directors?'?\s*reports?",
            r"management\s*discussion\s*(?:and\s*analysis)?",
            r"board\s*reports?",
            r"chairman'?s\s*statements?",
            r"ceo'?s\s*messages?",
        ]
        
        # Legacy document section patterns (kept for backward compatibility)
        self.document_section_patterns = [
            # Notes to Accounts
            r"NOTES?\s+TO\s+(?:THE\s+)?(?:CONSOLIDATED\s+)?FINANCIAL\s+STATEMENTS",
            r"NOTES?\s+ON\s+(?:THE\s+)?(?:CONSOLIDATED\s+)?FINANCIAL\s+STATEMENTS", 
            r"NOTES?\s+TO\s+(?:THE\s+)?ACCOUNTS",
            r"NOTES?\s+ON\s+(?:THE\s+)?ACCOUNTS",
            
            # Director's Report and other reports
            r"DIRECTOR'?S\s+REPORT",
            r"DIRECTORS'\s+REPORT",
            r"CHAIRMAN'?S\s+(?:STATEMENT|REPORT)",
            r"CEO'?S?\s+(?:STATEMENT|REPORT|MESSAGE)",
            r"CHIEF\s+EXECUTIVE\s+OFFICER'?S?\s+(?:STATEMENT|REPORT)",
            r"RISK\s+MANAGEMENT\s+(?:REPORT|SECTION)",
            r"CORPORATE\s+GOVERNANCE\s+(?:REPORT|STATEMENT)",
            r"MANAGEMENT\s+DISCUSSION\s+AND\s+ANALYSIS",
            r"BUSINESS\s+REVIEW"
        ]
        
        # Notes extraction patterns (legacy - keeping for compatibility)
        self.notes_patterns = [
            r"NOTES?\s+TO\s+(?:THE\s+)?(?:CONSOLIDATED\s+)?FINANCIAL\s+STATEMENTS",
            r"NOTES?\s+ON\s+(?:THE\s+)?(?:CONSOLIDATED\s+)?FINANCIAL\s+STATEMENTS", 
            r"NOTES?\s+TO\s+(?:THE\s+)?ACCOUNTS",
            r"NOTES?\s+ON\s+(?:THE\s+)?ACCOUNTS"
        ]
        
        # COMPLETELY REWRITTEN note patterns based on actual Phoenix structure
        self.note_number_patterns = [
            # PRIMARY: Match "1 Corporate information", "2 Summary of material accounting policies"
            r"(?:^|\n)\s*(\d+)\s+([A-Z][A-Za-z\s,\(\)&-]{15,80}?)(?=\n\s*(?:\d+\s+[A-Z]|[A-Z]{2,}|\s*$|\Z))",
            
            # SECONDARY: Match numbered sections with continuation patterns
            r"(?:^|\n)\s*(\d+)\s+([A-Z][A-Za-z\s,\(\)&-]+?)(?:\s+\(continued\))?(?=\n)",
            
            # TERTIARY: Match simple "1 Title" format more flexibly
            r"\n\s*(\d+)\s+([A-Z][A-Za-z][^\n]{10,100}?)(?=\n)",
        ]
        
        # COMPREHENSIVE EXCLUSION PATTERNS - Filter out financial statements AND audit reports
        self.exclusion_patterns = [
            # Financial statement patterns (we want notes, not primary statements)
            r"CONSOLIDATED\s+(?:STATEMENT|BALANCE\s+SHEET)",
            r"STATEMENT\s+OF\s+(?:FINANCIAL\s+POSITION|COMPREHENSIVE\s+INCOME|CASH\s+FLOWS)",
            r"BALANCE\s+SHEET",
            r"INCOME\s+STATEMENT",
            r"CASH\s+FLOW\s+STATEMENT",
            
            # COMPREHENSIVE AUDIT REPORT EXCLUSION - SAME AS FINANCIAL DETECTOR
            r"INDEPENDENT\s+AUDITOR'?S\s+(?:REPORT|OPINION)",
            r"AUDITOR'?S\s+(?:REPORT|OPINION)",
            r"REPORT\s+OF\s+INDEPENDENT\s+AUDITORS?",
            r"IN\s+OUR\s+OPINION\s*,?",
            r"WE\s+HAVE\s+AUDITED\s+THE\s+(?:ACCOMPANYING|CONSOLIDATED|FINANCIAL)",
            r"WE\s+(?:BELIEVE|CONSIDER)\s+THAT\s+THE\s+AUDIT\s+EVIDENCE",
            r"OUR\s+AUDIT\s+INVOLVED\s+PERFORMING",
            r"WE\s+CONDUCTED\s+OUR\s+AUDIT\s+IN\s+ACCORDANCE",
            r"BASIS\s+FOR\s+(?:OPINION|QUALIFIED\s+OPINION)",
            r"KEY\s+AUDIT\s+MATTERS",
            r"MATERIAL\s+UNCERTAINTY\s+RELATED\s+TO\s+GOING\s+CONCERN",
            r"OTHER\s+INFORMATION",
            r"RESPONSIBILITIES\s+OF\s+(?:MANAGEMENT|DIRECTORS|THOSE\s+CHARGED)",
            r"AUDITOR'?S\s+RESPONSIBILITIES\s+FOR\s+THE\s+AUDIT",
            r"REASONABLE\s+ASSURANCE\s+(?:ABOUT|THAT)",
            r"AUDIT\s+EVIDENCE\s+(?:WE\s+HAVE\s+)?OBTAINED",
            r"PROFESSIONAL\s+(?:JUDGMENT|SKEPTICISM)",
            r"EMPHASIS\s+OF\s+MATTER",
            r"OTHER\s+MATTER",
            r"REPORT\s+ON\s+OTHER\s+LEGAL",
            r"CHARTERED\s+ACCOUNTANTS?",
            r"PUBLIC\s+ACCOUNTANTS?",
            r"REGISTERED\s+AUDITORS?",
            r"\d{1,2}\s+(?:JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s+\d{4}.*?AUDITORS?",
            r"AUDIT\s+PARTNER",
            r"FOR\s+AND\s+ON\s+BEHALF\s+OF.*?AUDITORS?"
        ]
    
    def _is_valid_standard(self, standard_code: str) -> bool:
        """Validate if a standard code is a real IFRS/IAS standard"""
        
        # Official IFRS/IAS standards as of 2025
        valid_standards = {
            # IAS Standards (International Accounting Standards)
            "IAS 1", "IAS 2", "IAS 7", "IAS 8", "IAS 10", "IAS 12", "IAS 16", "IAS 19", 
            "IAS 20", "IAS 21", "IAS 23", "IAS 24", "IAS 26", "IAS 27", "IAS 28", 
            "IAS 29", "IAS 32", "IAS 33", "IAS 34", "IAS 36", "IAS 37", "IAS 38", 
            "IAS 40", "IAS 41",
            
            # IFRS Standards (International Financial Reporting Standards)
            "IFRS 1", "IFRS 2", "IFRS 3", "IFRS 4", "IFRS 5", "IFRS 6", "IFRS 7", 
            "IFRS 8", "IFRS 9", "IFRS 10", "IFRS 11", "IFRS 12", "IFRS 13", "IFRS 14", 
            "IFRS 15", "IFRS 16", "IFRS 17", "IFRS 18", "IFRS 19"
        }
        
        return standard_code in valid_standards
    
    def identify_standards_in_notes(self, document_text: str, document_id: str) -> Dict[str, Any]:
        """
        Main method to identify accounting standards using restructured three-level hierarchy:
        1. Header-based section tagging (PRIMARY - maintains structure)
        2. Sentence-level taxonomy analysis (SECONDARY - for untagged sentences)
        3. Semantic similarity fallback (TERTIARY - for remaining untagged)
        
        Expanded scope: Notes to Accounts, Director's Report, and other reports
        """
        logger.info(f"🏷️ Starting restructured three-level standard identification for document {document_id}")
        
        # Step 1: Extract all relevant document sections (Notes + Director's Report + other reports)
        document_sections = self._extract_all_document_sections(document_text, document_id)
        
        if not document_sections:
            logger.warning(f"⚠️ No relevant sections found in document {document_id}")
            return {
                "document_id": document_id,
                "tagged_sentences": [],
                "standards_found": [],
                "coverage_method": "restructured-three-level"
            }
        
        logger.info(f"📝 Found {len(document_sections)} document sections for tagging")
        
        # Step 2: Apply restructured three-level tagging hierarchy
        all_tagged_sentences = []
        all_standards = set()
        coverage_stats = {"header": 0, "taxonomy": 0, "semantic": 0, "untagged": 0}
        
        for section in document_sections:
            logger.info(f"🔍 Processing section: '{section.section_title}'")
            
            # LEVEL 1: Enhanced Header-based section tagging (PRIMARY - maintains structure)
            # Uses: Direct mapping → NLP keywords → Taxonomy concepts → Semantic similarity → Partial matching
            section_standard = self._identify_standard_from_header(section.section_title)
            
            if section_standard:
                # Tag ALL sentences in this section with the header-derived standard
                logger.info(f"📋 Header mapping found: '{section.section_title}' → {section_standard}")
                tagged_sentences = self._tag_all_sentences_with_standard(section, section_standard)
                
                # Process tables in the section with same standard
                table_sentences = self._tag_tables_with_standard(section, section_standard)
                tagged_sentences.extend(table_sentences)
                
                # Convert to simplified format and count
                for sentence in tagged_sentences:
                    if sentence.standards:
                        simple_sentence = {
                            "text": sentence.sentence_text,
                            "page": sentence.page_number,
                            "note": sentence.note_number,
                            "standards": [std.standard_code for std in sentence.standards],
                            "method": "header",
                            "confidence": sentence.standards[0].confidence_score if sentence.standards else 0.0
                        }
                        all_tagged_sentences.append(simple_sentence)
                        coverage_stats["header"] += 1
                        
                        # Collect all standards
                        for standard in sentence.standards:
                            all_standards.add(standard.standard_code)
                
                logger.info(f"✅ Header-tagged {len(tagged_sentences)} sentences in '{section.section_title}' with {section_standard}")
            
            else:
                # No header mapping - proceed with sentence-level analysis
                logger.info(f"❌ No header mapping for '{section.section_title}' - proceeding with sentence-level analysis")
                
                sentences = self._split_into_sentences(section.content)
                
                for i, sentence_text in enumerate(sentences):
                    if len(sentence_text.strip()) < 15:  # Skip very short sentences
                        continue
                    
                    tagged_sentence = None
                    method_used = None
                    
                    # LEVEL 2: Sentence-level taxonomy analysis (SECONDARY)
                    taxonomy_standards = self._identify_via_taxonomy(sentence_text)
                    if taxonomy_standards:
                        tagged_sentence = TaggedSentence(
                            sentence_text=sentence_text.strip(),
                            sentence_index=i + 1,
                            standards=taxonomy_standards,
                            page_number=section.page_numbers[0] if section.page_numbers else 1,
                            note_number=section.note_number,
                            subsection=section.section_title,
                            context=f"Taxonomy match in: {section.section_title}"
                        )
                        method_used = "taxonomy"
                        coverage_stats["taxonomy"] += 1
                        logger.debug(f"🎯 Taxonomy tagged: {sentence_text[:50]}... → {[s.standard_code for s in taxonomy_standards]}")
                    
                    # LEVEL 3: Semantic similarity fallback (TERTIARY)
                    else:
                        standard_descriptions = self._get_standard_descriptions_for_similarity()
                        best_match = self._find_best_semantic_match(sentence_text, standard_descriptions)
                        
                        if best_match and best_match['confidence'] > 0.6:
                            standard_tag = StandardTag(
                                standard_code=best_match['standard'],
                                standard_name=f"{best_match['standard']} (Semantic)",
                                confidence_score=best_match['confidence'],
                                matching_keywords=best_match['keywords'],
                                taxonomy_concepts=[]
                            )
                            tagged_sentence = TaggedSentence(
                                sentence_text=sentence_text.strip(),
                                sentence_index=i + 1,
                                standards=[standard_tag],
                                page_number=section.page_numbers[0] if section.page_numbers else 1,
                                note_number=section.note_number,
                                subsection=section.section_title,
                                context=f"Semantic match in: {section.section_title}"
                            )
                            method_used = "semantic"
                            coverage_stats["semantic"] += 1
                            logger.debug(f"🧠 Semantic tagged: {sentence_text[:50]}... → {best_match['standard']}")
                    
                    # Add tagged sentence or count as untagged
                    if tagged_sentence:
                        # Convert to simplified format
                        simple_sentence = {
                            "text": tagged_sentence.sentence_text,
                            "page": tagged_sentence.page_number,
                            "note": tagged_sentence.note_number,
                            "standards": [std.standard_code for std in tagged_sentence.standards],
                            "method": method_used,
                            "confidence": tagged_sentence.standards[0].confidence_score if tagged_sentence.standards else 0.0
                        }
                        all_tagged_sentences.append(simple_sentence)
                        
                        # Collect all standards
                        for standard in tagged_sentence.standards:
                            all_standards.add(standard.standard_code)
                    else:
                        coverage_stats["untagged"] += 1
                        logger.debug(f"❌ Untagged: {sentence_text[:50]}...")
                
                logger.info(f"✅ Section '{section.section_title}': {coverage_stats['taxonomy'] + coverage_stats['semantic']} sentences tagged via taxonomy/semantic")
        
        # Step 3: Return results with restructured three-level coverage statistics
        total_processed = sum(coverage_stats.values())
        results = {
            "document_id": document_id,
            "tagged_sentences": all_tagged_sentences,
            "standards_found": sorted(list(all_standards)),
            "coverage_method": "restructured-three-level-hierarchy",
            "coverage_statistics": {
                "total_sentences": total_processed,
                "header_tagged": coverage_stats["header"],
                "taxonomy_tagged": coverage_stats["taxonomy"], 
                "semantic_tagged": coverage_stats["semantic"],
                "untagged": coverage_stats["untagged"],
                "coverage_rate": ((total_processed - coverage_stats["untagged"]) / total_processed * 100) if total_processed > 0 else 0
            }
        }
        
        logger.info(f"✅ Restructured three-level standard identification complete:")
        logger.info(f"   📊 Total sentences: {total_processed}")
        logger.info(f"   📋 Header tagged: {coverage_stats['header']} ({coverage_stats['header']/total_processed*100:.1f}%)" if total_processed > 0 else "   📋 Header tagged: 0")
        logger.info(f"   🎯 Taxonomy tagged: {coverage_stats['taxonomy']} ({coverage_stats['taxonomy']/total_processed*100:.1f}%)" if total_processed > 0 else "   🎯 Taxonomy tagged: 0")
        logger.info(f"   🧠 Semantic tagged: {coverage_stats['semantic']} ({coverage_stats['semantic']/total_processed*100:.1f}%)" if total_processed > 0 else "   🧠 Semantic tagged: 0")
        logger.info(f"   ❌ Untagged: {coverage_stats['untagged']} ({coverage_stats['untagged']/total_processed*100:.1f}%)" if total_processed > 0 else "   ❌ Untagged: 0")
        
        return results
    
    def _extract_all_document_sections(self, document_text: str, document_id: str) -> List[NotesSection]:
        """Extract all relevant document sections in proper financial document order:
        1) Director's Report (BEFORE Financial Statements)
        2) Financial Statements 
        3) Notes to Financial Statements (AFTER Financial Statements)
        """
        sections = []
        
        # Find document structure boundaries
        structure = self._identify_document_structure(document_text)
        
        # Extract Director's Report sections (before financial statements)
        if structure['directors_start'] is not None and structure['financial_start'] is not None:
            directors_content = document_text[structure['directors_start']:structure['financial_start']]
            directors_sections = self._extract_directors_report_sections(directors_content, document_id)
            sections.extend(directors_sections)
            logger.info(f"📋 Extracted {len(directors_sections)} Director's Report sections")
        
        # Extract Notes sections (after financial statements)
        if structure['notes_start'] is not None:
            # Pass the entire document but with the notes_start position as context
            notes_sections = self._extract_notes_sections(document_text, document_id, notes_start_pos=structure['notes_start'])
            sections.extend(notes_sections)
            logger.info(f"📋 Extracted {len(notes_sections)} Notes sections")
        else:
            # Fallback: try to find notes in entire document
            notes_sections = self._extract_notes_sections(document_text, document_id)
            sections.extend(notes_sections)
            logger.warning(f"⚠️ Notes section not clearly identified - extracted {len(notes_sections)} sections from entire document")
        
        logger.info(f"📋 Total extracted sections: {len(sections)}")
        logger.info(f"🔧 Enhanced standard detection: NLP keywords + Taxonomy concepts + Semantic matching enabled")
        return sections
    
    def _identify_document_structure(self, document_text: str) -> Dict[str, Optional[int]]:
        """Identify the boundaries of major document sections"""
        structure: Dict[str, Optional[int]] = {
            'directors_start': None,
            'financial_start': None, 
            'notes_start': None
        }
        
        # Find Director's Report start
        for pattern in self.directors_report_markers:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                structure['directors_start'] = match.start()
                logger.info(f"📍 Found Director's Report at position {match.start()}")
                break
        
        # Find Financial Statements start
        for pattern in self.financial_statement_markers:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                structure['financial_start'] = match.start()
                logger.info(f"📍 Found Financial Statements at position {match.start()}")
                break
        
        # Find Notes start
        for pattern in self.notes_section_markers:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                structure['notes_start'] = match.start()
                logger.info(f"📍 Found Notes section at position {match.start()}")
                break
        
        return structure
    
    def _extract_directors_report_sections(self, directors_content: str, document_id: str) -> List[NotesSection]:
        """Extract sections from Director's Report content"""
        sections = []
        
        # Create a single Director's Report section for now
        # TODO: Could be enhanced to extract subsections like Risk Management, etc.
        section = NotesSection(
            section_title="Directors' Report",
            note_number="DR",  # Director's Report identifier
            subsection=None,
            content=directors_content.strip(),
            page_numbers=[1],  # Placeholder
            tagged_sentences=[]
        )
        sections.append(section)
        
        return sections
    
    def _extract_other_report_sections(self, document_text: str, document_id: str) -> List[NotesSection]:
        """Extract Director's Report and other non-notes sections"""
        sections = []
        
        # Look for Director's Report and other sections
        for pattern in self.document_section_patterns:
            if "NOTES" in pattern:  # Skip notes patterns (handled separately)
                continue
                
            matches = list(re.finditer(pattern, document_text, re.IGNORECASE | re.MULTILINE))
            
            for i, match in enumerate(matches):
                section_title = match.group(0).strip()
                
                # Extract full content until next major section or end of document
                start_pos = match.start()
                
                # Find the next major section - STOP at notes or other major sections
                next_section_pos = len(document_text)
                
                # Stop at the first numbered note (1 Corporate information, etc.)
                numbered_note_pattern = r'\n\s*1\s+[A-Z][A-Za-z\s,\(\)-]+(?:\n|$)'
                numbered_match = re.search(numbered_note_pattern, document_text[start_pos + 100:])
                if numbered_match:
                    next_section_pos = min(next_section_pos, start_pos + 100 + numbered_match.start())
                
                # Also stop at other major sections
                for next_pattern in self.document_section_patterns:
                    if "DIRECTOR" in next_pattern:  # Don't stop at another director pattern
                        continue
                    next_matches = list(re.finditer(next_pattern, document_text[start_pos + 100:], re.IGNORECASE | re.MULTILINE))
                    if next_matches:
                        next_section_pos = min(next_section_pos, start_pos + 100 + next_matches[0].start())
                
                content_text = document_text[start_pos:next_section_pos]
                
                if len(content_text.strip()) > 100:  # Minimum viable section
                    sections.append(NotesSection(
                        section_title=section_title,
                        note_number=f"Report_{len(sections)+1}",
                        subsection=None,
                        content=content_text,
                        page_numbers=[max(1, start_pos // 3000 + 1)],
                        tagged_sentences=[]
                    ))
        
        return sections
    
    def _identify_standard_from_header(self, section_title: str) -> Optional[str]:
        """Identify accounting standard from section header using intelligent NLP and taxonomy techniques"""
        
        # Normalize the section title for matching
        normalized_title = section_title.lower().strip()
        
        # Step 1: Direct mapping lookup (fastest)
        if normalized_title in self.header_standard_mapping:
            standard = self.header_standard_mapping[normalized_title]
            logger.debug(f"🎯 Direct match: '{section_title}' → {standard}")
            return standard
        
        # Step 2: Enhanced NLP-based keyword matching
        detected_standard = self._detect_standard_from_keywords(section_title)
        if detected_standard:
            logger.debug(f"🧠 NLP keyword match: '{section_title}' → {detected_standard}")
            return detected_standard
        
        # Step 3: Taxonomy concept matching
        taxonomy_standard = self._match_against_taxonomy_concepts(section_title)
        if taxonomy_standard:
            logger.debug(f"📚 Taxonomy match: '{section_title}' → {taxonomy_standard}")
            return taxonomy_standard
        
        # Step 4: Semantic similarity with standard descriptions
        semantic_standard = self._find_semantic_header_match(section_title)
        if semantic_standard:
            logger.debug(f"🧐 Semantic match: '{section_title}' → {semantic_standard}")
            return semantic_standard
        
        # Step 5: Partial matching for complex headers (fallback)
        for header_key, standard in self.header_standard_mapping.items():
            if header_key in normalized_title or any(word in normalized_title for word in header_key.split()):
                logger.debug(f"🎯 Partial match: '{section_title}' → {standard}")
                return standard
        
        # No mapping found
        logger.debug(f"❌ No standard mapping found for: '{section_title}'")
        return None
    
    def _detect_standard_from_keywords(self, section_title: str) -> Optional[str]:
        """Use NLP keyword analysis to detect accounting standards from section headers"""
        
        normalized_title = section_title.lower().strip()
        
        # Enhanced keyword patterns for IFRS/IAS identification
        standard_keywords = {
            "IAS 1": ["presentation", "financial statements", "statement presentation", "disclosure", "current non-current"],
            "IAS 2": ["inventories", "inventory", "stock", "cost of goods", "net realisable value", "fifo", "weighted average"],
            "IAS 7": ["cash flows", "cash flow", "operating activities", "investing activities", "financing activities"],
            "IAS 8": ["accounting policies", "accounting estimates", "prior period errors", "changes in estimates"],
            "IAS 10": ["events after", "subsequent events", "reporting period", "adjusting events"],
            "IAS 12": ["income tax", "deferred tax", "current tax", "tax expense", "temporary differences"],
            "IAS 16": ["property plant equipment", "property and equipment", "ppe", "depreciation", "useful life", "residual value"],
            "IAS 19": ["employee benefits", "post-employment", "pension", "retirement benefits", "short-term benefits"],
            "IAS 21": ["foreign exchange", "foreign currency", "translation", "functional currency", "presentation currency"],
            "IAS 23": ["borrowing costs", "capitalisation", "qualifying asset", "interest costs"],
            "IAS 24": ["related party", "related parties", "key management", "close family", "control"],
            "IAS 27": ["separate financial statements", "parent", "subsidiary", "investments in subsidiaries"],
            "IAS 28": ["associates", "joint ventures", "equity method", "significant influence"],
            "IAS 32": ["financial instruments presentation", "equity instruments", "liability or equity"],
            "IAS 33": ["earnings per share", "eps", "basic earnings", "diluted earnings"],
            "IAS 36": ["impairment", "recoverable amount", "value in use", "cash generating unit"],
            "IAS 37": ["provisions", "contingent liabilities", "contingent assets", "onerous contracts"],
            "IAS 38": ["intangible assets", "intangibles", "goodwill", "development costs", "amortisation"],
            "IAS 40": ["investment property", "investment properties", "fair value model", "cost model"],
            "IAS 41": ["agriculture", "biological assets", "agricultural produce", "fair value less costs"],
            "IFRS 1": ["first-time adoption", "transition", "ifrs adoption"],
            "IFRS 2": ["share-based payment", "equity-settled", "cash-settled", "stock options"],
            "IFRS 3": ["business combinations", "acquisition", "goodwill", "purchase price allocation"],
            "IFRS 5": ["non-current assets held for sale", "discontinued operations"],
            "IFRS 7": ["financial instruments disclosures", "credit risk", "liquidity risk", "market risk"],
            "IFRS 8": ["operating segments", "reportable segments", "chief operating decision maker"],
            "IFRS 9": ["financial instruments", "expected credit losses", "classification and measurement"],
            "IFRS 10": ["consolidated financial statements", "control", "subsidiaries"],
            "IFRS 11": ["joint arrangements", "joint operations", "joint ventures"],
            "IFRS 12": ["disclosure of interests", "unconsolidated structured entities"],
            "IFRS 13": ["fair value measurement", "fair value hierarchy", "level 1 2 3"],
            "IFRS 15": ["revenue from contracts", "performance obligations", "transaction price"],
            "IFRS 16": ["leases", "right-of-use", "lease liability", "lessee", "lessor"],
            "IFRS 17": ["insurance contracts", "insurance", "coverage units"]
        }
        
        # Score each standard based on keyword matches
        standard_scores = {}
        
        for standard, keywords in standard_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in normalized_title:
                    # Weight longer, more specific keywords higher
                    weight = len(keyword.split()) * 2  # Multi-word keywords get higher weight
                    score += weight
            
            if score > 0:
                standard_scores[standard] = score
        
        # Return the highest scoring standard if above threshold
        if standard_scores:
            best_standard = max(standard_scores.items(), key=lambda x: x[1])
            if best_standard[1] >= 2:  # Minimum score threshold
                return best_standard[0]
        
        return None
    
    def _match_against_taxonomy_concepts(self, section_title: str) -> Optional[str]:
        """Match section headers against XBRL taxonomy concepts"""
        
        if not self.taxonomy_data or not hasattr(self, 'concept_index'):
            return None
        
        normalized_title = section_title.lower().strip()
        
        # Extract meaningful words from title
        import re
        words = re.findall(r'\b[a-z]{3,}\b', normalized_title)  # 3+ letter words
        
        concept_matches = {}
        
        # Check against taxonomy concept keywords
        if hasattr(self, 'concept_index') and 'by_keywords' in self.concept_index:
            for word in words:
                if word.capitalize() in self.concept_index['by_keywords']:
                    concept_ids = self.concept_index['by_keywords'][word.capitalize()]
                    
                    # Map concept IDs to standards (simplified)
                    for concept_id in concept_ids:
                        # Extract standard from concept ID (e.g., 'ifrs-full:PropertyPlantAndEquipment' → 'IAS 16')
                        if 'PropertyPlantAndEquipment' in concept_id or 'PPE' in concept_id:
                            concept_matches['IAS 16'] = concept_matches.get('IAS 16', 0) + 1
                        elif 'IntangibleAsset' in concept_id:
                            concept_matches['IAS 38'] = concept_matches.get('IAS 38', 0) + 1
                        elif 'Revenue' in concept_id:
                            concept_matches['IFRS 15'] = concept_matches.get('IFRS 15', 0) + 1
                        elif 'Lease' in concept_id:
                            concept_matches['IFRS 16'] = concept_matches.get('IFRS 16', 0) + 1
                        # Add more mappings as needed
        
        # Return best match if above threshold
        if concept_matches:
            best_match = max(concept_matches.items(), key=lambda x: x[1])
            if best_match[1] >= 1:  # At least one strong concept match
                return best_match[0]
        
        return None
    
    def _find_semantic_header_match(self, section_title: str) -> Optional[str]:
        """Use semantic similarity to match headers with standard descriptions"""
        
        # Standard descriptions for semantic matching
        standard_descriptions = {
            "IAS 1": "presentation of financial statements disclosure requirements current non-current classification",
            "IAS 2": "inventories cost measurement net realisable value first-in-first-out weighted average",
            "IAS 7": "statement of cash flows operating investing financing activities direct indirect method",
            "IAS 8": "accounting policies changes in accounting estimates and errors retrospective prospective",
            "IAS 12": "income taxes current deferred tax assets liabilities temporary differences",
            "IAS 16": "property plant and equipment cost depreciation useful life residual value revaluation",
            "IAS 19": "employee benefits post-employment pension plans short-term long-term benefits",
            "IAS 21": "effects of changes in foreign exchange rates functional presentation currency translation",
            "IAS 24": "related party disclosures key management personnel close family members control",
            "IAS 28": "investments in associates and joint ventures equity method significant influence",
            "IAS 36": "impairment of assets recoverable amount value in use cash-generating units",
            "IAS 37": "provisions contingent liabilities and contingent assets present obligation probable outflow",
            "IAS 38": "intangible assets development costs research costs amortisation useful life indefinite",
            "IAS 40": "investment property fair value model cost model rental income capital appreciation",
            "IFRS 3": "business combinations acquisition method goodwill purchase price allocation",
            "IFRS 7": "financial instruments disclosures credit risk liquidity risk market risk",
            "IFRS 9": "financial instruments classification measurement impairment expected credit losses",
            "IFRS 15": "revenue from contracts with customers performance obligations transaction price",
            "IFRS 16": "leases right-of-use assets lease liabilities lessee lessor accounting"
        }
        
        # Simple semantic matching based on word overlap
        normalized_title = section_title.lower().strip()
        title_words = set(normalized_title.split())
        
        best_match = None
        best_score = 0
        
        for standard, description in standard_descriptions.items():
            description_words = set(description.split())
            
            # Calculate Jaccard similarity (intersection over union)
            intersection = len(title_words.intersection(description_words))
            union = len(title_words.union(description_words))
            
            if union > 0:
                similarity = intersection / union
                if similarity > best_score and similarity > 0.1:  # Minimum 10% similarity
                    best_score = similarity
                    best_match = standard
        
        return best_match if best_score > 0.15 else None  # Minimum threshold for semantic match
        
    def _tag_all_sentences_with_standard(self, section: NotesSection, standard_code: str) -> List[TaggedSentence]:
        """Tag ALL sentences in a section with the given standard (header-based inheritance)"""
        
        # Split section content into sentences
        sentences = self._split_into_sentences(section.content)
        tagged_sentences = []
        
        for i, sentence_text in enumerate(sentences):
            if len(sentence_text.strip()) < 10:  # Skip very short sentences
                continue
                
            # Create standard tag
            standard_tag = StandardTag(
                standard_code=standard_code,
                standard_name=f"{standard_code} (Header-based)",
                confidence_score=0.9,  # High confidence for header-based tagging
                matching_keywords=["header-based"],
                taxonomy_concepts=[]
            )
            
            # Create tagged sentence
            tagged_sentence = TaggedSentence(
                sentence_text=sentence_text.strip(),
                sentence_index=i + 1,
                standards=[standard_tag],
                page_number=section.page_numbers[0] if section.page_numbers else 1,
                note_number=section.note_number,
                subsection=section.section_title,
                context=f"Section: {section.section_title}"
            )
            
            tagged_sentences.append(tagged_sentence)
        
        return tagged_sentences
    
    def _tag_tables_with_standard(self, section: NotesSection, standard_code: str) -> List[TaggedSentence]:
        """Tag tables in a section with the given standard (placeholder for now)"""
        # For now, return empty list - will implement table extraction later
        return []
    
    def _apply_semantic_similarity_tagging(self, section: NotesSection) -> List[TaggedSentence]:
        """Apply semantic similarity tagging as fallback for sections without header mapping"""
        
        # Split section content into sentences
        sentences = self._split_into_sentences(section.content)
        tagged_sentences = []
        
        # Standard descriptions for semantic matching
        standard_descriptions = self._get_standard_descriptions_for_similarity()
        
        for i, sentence_text in enumerate(sentences):
            if len(sentence_text.strip()) < 15:  # Skip very short sentences
                continue
                
            # Find best matching standard using semantic similarity
            best_match = self._find_best_semantic_match(sentence_text, standard_descriptions)
            
            if best_match and best_match['confidence'] > 0.6:  # Threshold for semantic matching
                # Create standard tag
                standard_tag = StandardTag(
                    standard_code=best_match['standard'],
                    standard_name=f"{best_match['standard']} (Semantic)",
                    confidence_score=best_match['confidence'],
                    matching_keywords=best_match['keywords'],
                    taxonomy_concepts=[]
                )
                
                # Create tagged sentence
                tagged_sentence = TaggedSentence(
                    sentence_text=sentence_text.strip(),
                    sentence_index=i + 1,
                    standards=[standard_tag],
                    page_number=section.page_numbers[0] if section.page_numbers else 1,
                    note_number=section.note_number,
                    subsection=section.section_title,
                    context=f"Semantic match in: {section.section_title}"
                )
                
                tagged_sentences.append(tagged_sentence)
        
        return tagged_sentences
    
    def _get_standard_descriptions_for_similarity(self) -> Dict[str, str]:
        """Get standard descriptions for semantic similarity matching"""
        return {
            # Revenue Standards
            "IFRS 15": "revenue recognition contracts customers performance obligations transaction price contract modification sales income",
            "IFRS 18": "revenue recognition sale goods rendering services contract",
            
            # Asset Standards  
            "IAS 16": "property plant equipment depreciation useful life impairment disposal revaluation buildings machinery",
            "IAS 38": "intangible assets goodwill amortisation impairment development costs software licenses patents",
            "IAS 36": "impairment assets cash generating units recoverable amount value in use fair value",
            "IAS 2": "inventories cost measurement net realizable value obsolete slow moving stock",
            "IAS 40": "investment property fair value cost model rental income",
            "IAS 41": "agriculture biological assets fair value harvest produce livestock",
            
            # Financial Instruments
            "IFRS 9": "financial instruments classification measurement impairment credit losses expected loss model",
            "IFRS 7": "financial instruments disclosures risk management credit risk market risk liquidity risk",
            "IAS 32": "financial instruments presentation equity liability compound instruments",
            "IFRS 13": "fair value measurement hierarchy observable inputs unobservable inputs valuation techniques",
            
            # Liabilities and Provisions
            "IAS 37": "provisions contingent liabilities contingent assets probable outflow reliable estimate",
            "IAS 19": "employee benefits defined benefit plans defined contribution plans post employment benefits",
            "IFRS 2": "share based payment equity settled cash settled stock options employee share schemes",
            
            # Business Combinations and Investments
            "IFRS 3": "business combinations acquisition goodwill purchase price allocation fair value",
            "IAS 27": "separate financial statements subsidiaries associates joint ventures",
            "IAS 28": "investments associates joint ventures equity method significant influence",
            "IFRS 10": "consolidated financial statements control subsidiaries non controlling interests",
            "IFRS 11": "joint arrangements joint operations joint ventures",
            
            # Leases
            "IFRS 16": "leases right of use assets lease liabilities incremental borrowing rate lease term",
            
            # Cash and Financial Reporting
            "IAS 7": "cash flows operating activities investing activities financing activities cash equivalents",
            "IAS 1": "presentation financial statements going concern materiality offsetting comparative information",
            "IAS 8": "accounting policies changes estimates prior period errors retrospective application",
            
            # Tax and Regulatory
            "IAS 12": "income taxes current tax deferred tax temporary differences tax losses",
            "IAS 21": "foreign currency translation functional currency presentation currency exchange differences",
            "IAS 29": "hyperinflationary economies monetary items non monetary items general price index",
            
            # Sector Specific
            "IFRS 8": "operating segments reportable segments segment reporting management approach",
            "IAS 24": "related party disclosures key management personnel transactions balances",
            "IAS 33": "earnings per share basic earnings diluted earnings ordinary shares",
            "IAS 34": "interim financial reporting condensed financial statements selected explanatory notes",
            
            # Specialized Standards
            "IFRS 4": "insurance contracts insurance liabilities adequacy test unbundling",
            "IFRS 5": "non current assets held for sale discontinued operations disposal groups",
            "IFRS 6": "exploration evaluation mineral resources exploration assets development assets",
            "IFRS 14": "regulatory deferral accounts rate regulated activities regulatory assets",
            "IFRS 17": "insurance contracts coverage units contractual service margin risk adjustment",
            "IAS 20": "government grants assistance revenue grants capital grants systematic basis",
            "IAS 23": "borrowing costs capitalisation directly attributable costs qualifying assets",
            "IAS 26": "retirement benefit plans defined benefit plans defined contribution plans net assets"
        }
    
    def _find_best_semantic_match(self, sentence: str, standard_descriptions: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Find best matching standard using keyword-based semantic similarity"""
        
        sentence_lower = sentence.lower()
        best_match = None
        best_score = 0.0
        
        for standard, description in standard_descriptions.items():
            # Simple keyword matching approach (can be enhanced with embeddings later)
            keywords = description.split()
            matched_keywords = []
            score = 0.0
            
            for keyword in keywords:
                if keyword in sentence_lower:
                    matched_keywords.append(keyword)
                    # Weight longer keywords more heavily
                    score += len(keyword) / 10.0
            
            # Normalize score by description length
            if len(keywords) > 0:
                normalized_score = score / len(keywords)
                
                if normalized_score > best_score and len(matched_keywords) > 0:
                    best_score = normalized_score
                    best_match = {
                        'standard': standard,
                        'confidence': min(normalized_score, 0.95),  # Cap at 95%
                        'keywords': matched_keywords[:5]  # Top 5 matching keywords
                    }
        
        return best_match
    
    def _extract_notes_sections(self, document_text: str, document_id: str, notes_start_pos: Optional[int] = None) -> List[NotesSection]:
        """Extract Notes to Accounts sections from document"""
        
        # Use provided position or find Notes to Accounts section
        notes_start = notes_start_pos
        if not notes_start:
            for pattern in self.notes_section_markers:  # Use new markers
                match = re.search(pattern, document_text, re.IGNORECASE)
                if match:
                    notes_start = match.start()
                    logger.info(f"📍 Found Notes section at position {notes_start}")
                    break
        
        if not notes_start:
            logger.warning("⚠️ No Notes to Accounts section found")
            return []
        
        # Extract notes content (exclude financial statements and audit reports)
        notes_content = document_text[notes_start:]
        
        # COMPREHENSIVE AUDIT REPORT REMOVAL - Same approach as financial detector
        notes_content = self._remove_audit_report_content(notes_content, document_id)
        
        # Remove any remaining exclusion patterns  
        for exclusion_pattern in self.exclusion_patterns:
            notes_content = re.sub(exclusion_pattern + r".*?(?=\n\n|$)", "", 
                                  notes_content, flags=re.IGNORECASE | re.DOTALL)
        
        # Split into individual note sections using improved logic
        sections = self._split_into_note_sections(notes_content, document_id)
        
        return sections
    
    def _split_into_note_sections(self, notes_content: str, document_id: str) -> List[NotesSection]:
        """Split notes content into individual note sections using simplified patterns"""
        
        sections = []
        
        # Simplified note detection pattern - look for numbered sections
        simple_pattern = r'(?:^|\n)\s*(\d+)\s+([A-Z][^\n]{10,80})(?=\n|$)'
        matches = list(re.finditer(simple_pattern, notes_content, re.MULTILINE))
        
        if matches:
            logger.info(f"🔢 Found {len(matches)} numbered note sections")
                
            for i, match in enumerate(matches):
                note_number = match.group(1)
                section_title = match.group(2).strip()
                
                # Extract content until next section
                start_pos = match.start()
                end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(notes_content)
                
                section_content = notes_content[start_pos:end_pos].strip()
                
                if len(section_content) > 100:  # Minimum viable section size
                    # Estimate page numbers (rough calculation)
                    page_estimate = [max(1, start_pos // 3000 + 1)]
                    
                    sections.append(NotesSection(
                        section_title=section_title,
                        note_number=note_number,
                        subsection=None,
                        content=section_content,
                        page_numbers=page_estimate,
                        tagged_sentences=[]
                    ))
                    
                    logger.info(f"📝 Created note section {note_number}: {section_title}")
        
        # Fallback: treat entire notes as single section if no numbered sections found
        if not sections and len(notes_content.strip()) > 100:
            logger.warning("📄 No numbered sections found - using entire notes as single section")
            sections.append(NotesSection(
                section_title="Notes to Financial Statements",
                note_number="1",
                subsection=None,
                content=notes_content[:50000],  # Limit to 50K chars to avoid token issues
                page_numbers=[1],
                tagged_sentences=[]
            ))
        
        logger.info(f"📋 Created {len(sections)} note sections")
        return sections
    
    def _tag_sentences_in_section(self, section: NotesSection) -> List[TaggedSentence]:
        """Tag every sentence in a notes section with accounting standards"""
        
        logger.info(f"🏷️ Tagging sentences in section: {section.note_number}")
        
        # Split content into sentences
        sentences = self._split_into_sentences(section.content)
        
        tagged_sentences = []
        
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) < 20:  # Skip very short fragments
                continue
            
            # Identify standards for this sentence
            standards = self._identify_standards_for_sentence(sentence, section)
            
            # Create tagged sentence
            tagged_sentence = TaggedSentence(
                sentence_text=sentence.strip(),
                sentence_index=i,
                standards=standards,
                page_number=section.page_numbers[0] if section.page_numbers else 1,
                note_number=section.note_number,
                subsection=section.subsection,
                context=self._get_sentence_context(sentence, sentences, i)
            )
            
            tagged_sentences.append(tagged_sentence)
        
        logger.info(f"🏷️ Tagged {len(tagged_sentences)} sentences in section {section.note_number}")
        return tagged_sentences
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using multiple delimiters"""
        
        # Enhanced sentence splitting that handles financial text
        sentence_delimiters = [
            r'\.',  # Period
            r';',   # Semicolon 
            r'\n\n',  # Double newline
            r'\n(?=[A-Z])',  # Newline followed by capital letter
            r'(?<=\d)\s*\n(?=[A-Z])',  # Newline after number, before capital
        ]
        
        # Split using regex pattern
        pattern = '(' + '|'.join(sentence_delimiters) + ')'
        parts = re.split(pattern, text)
        
        sentences = []
        current_sentence = ""
        
        for part in parts:
            if re.match('|'.join(sentence_delimiters), part):
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""
            else:
                current_sentence += part
        
        # Add final sentence
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        return [s for s in sentences if len(s.strip()) > 20]
    
    def _identify_standards_for_sentence(self, sentence: str, section: NotesSection) -> List[StandardTag]:
        """Identify all applicable accounting standards for a sentence"""
        
        sentence_upper = sentence.upper()
        identified_standards = {}
        
        # Use taxonomy-based identification if available
        if self.taxonomy_data and self.concept_index:
            taxonomy_standards = self._identify_via_taxonomy(sentence_upper)
            for standard in taxonomy_standards:
                identified_standards[standard.standard_code] = standard
        
        # Use keyword-based identification (fallback + enhancement)
        keyword_standards = self._identify_via_keywords(sentence_upper)
        for standard in keyword_standards:
            if standard.standard_code in identified_standards:
                # Enhance existing standard with additional keywords
                existing = identified_standards[standard.standard_code]
                existing.matching_keywords.extend(standard.matching_keywords)
                existing.confidence_score = max(existing.confidence_score, standard.confidence_score)
            else:
                identified_standards[standard.standard_code] = standard
        
        # Convert to list and sort by confidence
        standards_list = list(identified_standards.values())
        standards_list.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return standards_list
    
    def _identify_via_taxonomy(self, sentence: str) -> List[StandardTag]:
        """Identify standards using taxonomy concept matching"""
        
        standards = []
        
        if not self.concept_index:
            return standards
        
        # Check sentence against taxonomy keywords index
        for keyword, concepts in self.concept_index.get("by_keywords", {}).items():
            if keyword.upper() in sentence:
                # Find standards associated with these concepts
                for concept_id in concepts[:5]:  # Limit to avoid spam
                    if self.taxonomy_data and "concepts" in self.taxonomy_data:
                        concept_data = self.taxonomy_data["concepts"].get(concept_id, {})
                        standard_ref = concept_data.get("standard_reference")
                    else:
                        continue
                    
                    # VALIDATE THE STANDARD BEFORE ADDING
                    if (standard_ref and 
                        self._is_valid_standard(standard_ref) and  # NEW VALIDATION
                        standard_ref not in [s.standard_code for s in standards]):
                        
                        standards.append(StandardTag(
                            standard_code=standard_ref,
                            standard_name=self._get_standard_name(standard_ref),
                            confidence_score=0.7,  # Taxonomy-based confidence
                            matching_keywords=[keyword],
                            taxonomy_concepts=[concept_id]
                        ))
        
        return standards
    
    def _identify_via_keywords(self, sentence: str) -> List[StandardTag]:
        """Identify standards using keyword mapping (based on Phoenix analysis)"""
        
        standards = []
        
        # Keyword patterns from our Phoenix Group analysis
        keyword_mappings = {
            "IAS 1": {
                "keywords": [
                    "STATEMENT OF FINANCIAL POSITION", "BALANCE SHEET", 
                    "STATEMENT OF COMPREHENSIVE INCOME", "OTHER COMPREHENSIVE INCOME",
                    "PRESENTATION OF FINANCIAL STATEMENTS", "CURRENT ASSETS", "CURRENT LIABILITIES"
                ],
                "name": "Presentation of Financial Statements"
            },
            "IAS 2": {
                "keywords": [
                    "INVENTORY", "INVENTORIES", "COST OF GOODS SOLD", "COST OF SALES",
                    "COMMODITY-BROKER TRADER", "FAIR VALUE LESS COST TO SELL"
                ],
                "name": "Inventories"
            },
            "IAS 12": {
                "keywords": [
                    "CURRENT TAX", "DEFERRED TAX", "TAX EXPENSE", "INCOME TAX",
                    "TAXABLE INCOME", "TAX RATES", "TAX CREDIT"
                ],
                "name": "Income Taxes"
            },
            "IAS 16": {
                "keywords": [
                    "PROPERTY AND EQUIPMENT", "PROPERTY, PLANT AND EQUIPMENT",
                    "PLANT AND EQUIPMENT", "DEPRECIATION", "ASSET IMPAIRMENT"
                ],
                "name": "Property, Plant and Equipment"  
            },
            "IAS 21": {
                "keywords": [
                    "FOREIGN EXCHANGE", "FOREIGN CURRENCY", "EXCHANGE RATE",
                    "FOREIGN OPERATION", "NET INVESTMENT", "FUNCTIONAL CURRENCY"
                ],
                "name": "The Effects of Changes in Foreign Exchange Rates"
            },
            "IFRS 3": {
                "keywords": [
                    "BUSINESS COMBINATION", "GOODWILL", "ACQUISITION", 
                    "IDENTIFIABLE ASSETS", "FAIR VALUE ADJUSTMENTS"
                ],
                "name": "Business Combinations"
            },
            "IFRS 9": {
                "keywords": [
                    "EXPECTED CREDIT LOSSES", "FINANCIAL INSTRUMENTS", 
                    "SIMPLIFIED APPROACH", "LIFETIME EXPECTED LOSS", "TRADE RECEIVABLES"
                ],
                "name": "Financial Instruments"
            },
            "IFRS 13": {
                "keywords": [
                    "FAIR VALUE", "FAIR VALUE MEASUREMENT", "FAIR VALUE HIERARCHY",
                    "LEVEL 1", "LEVEL 2", "LEVEL 3", "MARKET APPROACH"
                ],
                "name": "Fair Value Measurement"
            },
            "IFRS 16": {
                "keywords": [
                    "RIGHT OF USE ASSETS", "LEASE", "LEASES", "LEASE LIABILITY",
                    "LEASE PAYMENTS", "DISCOUNT RATE"
                ],
                "name": "Leases"
            }
        }
        
        # Check sentence against all keyword mappings
        for standard_code, mapping in keyword_mappings.items():
            matching_keywords = []
            
            for keyword in mapping["keywords"]:
                if keyword in sentence:
                    matching_keywords.append(keyword)
            
            if matching_keywords:
                # Calculate confidence based on number and specificity of matches
                confidence = min(0.9, 0.5 + (len(matching_keywords) * 0.1))
                
                standards.append(StandardTag(
                    standard_code=standard_code,
                    standard_name=mapping["name"],
                    confidence_score=confidence,
                    matching_keywords=matching_keywords,
                    taxonomy_concepts=[]
                ))
        
        return standards
    
    def _get_sentence_context(self, sentence: str, all_sentences: List[str], index: int) -> str:
        """Get surrounding context for a sentence"""
        
        context_sentences = []
        
        # Add previous sentence
        if index > 0:
            context_sentences.append(all_sentences[index - 1][-100:])  # Last 100 chars
        
        # Add next sentence  
        if index < len(all_sentences) - 1:
            context_sentences.append(all_sentences[index + 1][:100])  # First 100 chars
        
        return " ... ".join(context_sentences)
    
    def _extract_and_tag_tables(self, section: NotesSection) -> List[TaggedSentence]:
        """
        Extract tables from section content and tag them with standards
        
        Tables are identified by:
        1. Multiple columns with aligned data
        2. Headers followed by data rows
        3. Numeric patterns in tabular format
        4. Common financial table indicators
        """
        logger.info(f"🔢 Extracting tables from section: {section.note_number}")
        
        table_sentences = []
        content = section.content
        
        # Table detection patterns
        table_patterns = [
            # Table with headers and data separated by spaces/tabs
            r'([A-Z][A-Za-z\s]{10,})\s+(\d{4})\s+(\d{4})',  # "Revenue 2024 2023"
            r'([A-Z][A-Za-z\s&]{5,})\s+([\d,]+)\s+([\d,]+)',  # "Cash & equivalents 1,000 2,000"
            
            # Table with currency symbols
            r'([A-Z][A-Za-z\s]{10,})\s+\$\s*([\d,]+)\s+\$\s*([\d,]+)',  # "Revenue $ 1,000 $ 2,000"
            r'([A-Z][A-Za-z\s]{10,})\s+£\s*([\d,]+)\s+£\s*([\d,]+)',   # "Revenue £ 1,000 £ 2,000"
            
            # Table rows with multiple numeric columns
            r'([A-Z][A-Za-z\s]{10,})\s+([\d,]+(?:\.\d{2})?)\s+([\d,]+(?:\.\d{2})?)\s+([\d,]+(?:\.\d{2})?)',
            
            # Continuation lines in tables (indented items)
            r'^\s{2,}([A-Z][A-Za-z\s]{5,})\s+([\d,]+)\s+([\d,]+)',
        ]
        
        # Split content into lines for table detection
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if len(line) < 10:  # Skip very short lines
                continue
            
            # Check if line matches table patterns
            for pattern in table_patterns:
                match = re.match(pattern, line)
                if match:
                    # Extract table row description
                    table_item = match.group(1).strip()
                    
                    # Create a table sentence
                    table_text = f"Table data: {table_item} with values {' '.join(match.groups()[1:])}"
                    
                    # Identify standards for this table row
                    standards = self._identify_standards_for_sentence(table_text, section)
                    
                    if standards:  # Only include if standards are identified
                        table_sentence = TaggedSentence(
                            sentence_text=f"{table_item} (table data)",
                            sentence_index=1000 + i,  # Use high index for table items
                            standards=standards,
                            page_number=section.page_numbers[0] if section.page_numbers else 1,
                            note_number=section.note_number,
                            subsection=section.subsection,
                            context=f"Table row: {line}"
                        )
                        
                        table_sentences.append(table_sentence)
                    
                    break  # Found a match, no need to check other patterns
        
        logger.info(f"🔢 Extracted {len(table_sentences)} table entries from section {section.note_number}")
        return table_sentences
    
    def _get_standard_name(self, standard_code: str) -> str:
        """Get full name for accounting standard code"""
        
        standard_names = {
            "IAS 1": "Presentation of Financial Statements",
            "IAS 2": "Inventories", 
            "IAS 7": "Statement of Cash Flows",
            "IAS 8": "Accounting Policies, Changes in Accounting Estimates and Errors",
            "IAS 10": "Events after the Reporting Period",
            "IAS 12": "Income Taxes",
            "IAS 16": "Property, Plant and Equipment",
            "IAS 19": "Employee Benefits",
            "IAS 21": "The Effects of Changes in Foreign Exchange Rates",
            "IAS 23": "Borrowing Costs",
            "IAS 24": "Related Party Disclosures",
            "IAS 27": "Separate Financial Statements",
            "IAS 28": "Investments in Associates and Joint Ventures",
            "IAS 32": "Financial Instruments: Presentation",
            "IAS 33": "Earnings per Share",
            "IAS 34": "Interim Financial Reporting",
            "IAS 36": "Impairment of Assets",
            "IAS 37": "Provisions, Contingent Liabilities and Contingent Assets",
            "IAS 38": "Intangible Assets",
            "IAS 40": "Investment Property",
            "IAS 41": "Agriculture",
            "IFRS 1": "First-time Adoption of International Financial Reporting Standards",
            "IFRS 2": "Share-based Payment",
            "IFRS 3": "Business Combinations",
            "IFRS 5": "Non-current Assets Held for Sale and Discontinued Operations",
            "IFRS 6": "Exploration for and Evaluation of Mineral Resources", 
            "IFRS 7": "Financial Instruments: Disclosures",
            "IFRS 8": "Operating Segments",
            "IFRS 9": "Financial Instruments",
            "IFRS 10": "Consolidated Financial Statements",
            "IFRS 11": "Joint Arrangements",
            "IFRS 12": "Disclosure of Interests in Other Entities",
            "IFRS 13": "Fair Value Measurement",
            "IFRS 14": "Regulatory Deferral Accounts",
            "IFRS 15": "Revenue from Contracts with Customers",
            "IFRS 16": "Leases",
            "IFRS 17": "Insurance Contracts"
        }
        
        return standard_names.get(standard_code, standard_code)
    
    def _build_concept_index(self):
        """Build searchable index from taxonomy data"""
        
        if not self.taxonomy_data:
            return
        
        # This would build the concept index similar to taxonomy integration
        # Using the concepts data to create searchable keyword mappings
        self.concept_index = {
            "by_keywords": {},
            "by_standard": {},
            "by_category": {}
        }
        
        # Extract and index taxonomy concepts (simplified version)
        concepts = self.taxonomy_data.get("concepts", {})
        for concept_id, concept_data in concepts.items():
            # Extract keywords from concept name
            name = concept_data.get("name", "")
            keywords = re.findall(r'[A-Z][a-z]+', name)  # Extract camelCase words
            
            for keyword in keywords:
                if len(keyword) > 3:  # Meaningful keywords only
                    if keyword not in self.concept_index["by_keywords"]:
                        self.concept_index["by_keywords"][keyword] = []
                    self.concept_index["by_keywords"][keyword].append(concept_id)
    
    def _load_header_standard_mapping(self):
        """Load mapping from section headers to accounting standards"""
        self.header_standard_mapping = {
            # Revenue and Income
            "revenue": "IFRS 15",
            "revenue recognition": "IFRS 15", 
            "revenue from contracts": "IFRS 15",
            "contract revenue": "IFRS 15",
            "sales": "IFRS 15",
            "income": "IFRS 15",
            
            # Property, Plant and Equipment
            "property plant equipment": "IAS 16",
            "property plant and equipment": "IAS 16",
            "ppe": "IAS 16",
            "fixed assets": "IAS 16",
            "tangible assets": "IAS 16",
            "plant and equipment": "IAS 16",
            "buildings": "IAS 16",
            "machinery": "IAS 16",
            "equipment": "IAS 16",
            "depreciation": "IAS 16",
            
            # Intangible Assets
            "intangible assets": "IAS 38",
            "intangibles": "IAS 38",
            "goodwill": "IAS 38",
            "software": "IAS 38",
            "licenses": "IAS 38",
            "patents": "IAS 38",
            "trademarks": "IAS 38",
            "amortisation": "IAS 38",
            "amortization": "IAS 38",
            
            # Financial Instruments
            "financial instruments": "IFRS 9",
            "financial assets": "IFRS 9",
            "financial liabilities": "IFRS 9",
            "investments": "IFRS 9",
            "derivatives": "IFRS 9",
            "credit losses": "IFRS 9",
            "impairment": "IFRS 9",
            
            # Leases
            "leases": "IFRS 16",
            "lease": "IFRS 16",
            "right of use": "IFRS 16",
            "right-of-use": "IFRS 16",
            "lease liabilities": "IFRS 16",
            "rental": "IFRS 16",
            
            # Inventories
            "inventories": "IAS 2",
            "inventory": "IAS 2",
            "stock": "IAS 2",
            "cost of goods sold": "IAS 2",
            "raw materials": "IAS 2",
            "finished goods": "IAS 2",
            
            # Employee Benefits
            "employee benefits": "IAS 19",
            "staff costs": "IAS 19",
            "pension": "IAS 19",
            "retirement benefits": "IAS 19",
            "post employment": "IAS 19",
            "gratuity": "IAS 19",
            
            # Share-based Payment
            "share based payment": "IFRS 2",
            "share-based payment": "IFRS 2",
            "stock options": "IFRS 2",
            "equity compensation": "IFRS 2",
            
            # Business Combinations
            "business combinations": "IFRS 3",
            "acquisitions": "IFRS 3",
            "mergers": "IFRS 3",
            "purchase price allocation": "IFRS 3",
            
            # Cash Flow Statement
            "cash flows": "IAS 7",
            "cash flow": "IAS 7",
            "statement of cash flows": "IAS 7",
            
            # Accounting Policies
            "accounting policies": "IAS 8",
            "significant accounting policies": "IAS 8",
            "changes in accounting estimates": "IAS 8",
            "prior period errors": "IAS 8",
            
            # Events After Reporting Period
            "events after reporting period": "IAS 10",
            "subsequent events": "IAS 10",
            "post balance sheet events": "IAS 10",
            
            # Provisions and Contingencies
            "provisions": "IAS 37",
            "contingent liabilities": "IAS 37",
            "contingent assets": "IAS 37",
            "contingencies": "IAS 37",
            
            # Related Party Disclosures
            "related parties": "IAS 24",
            "related party": "IAS 24",
            "related party transactions": "IAS 24",
            
            # Income Taxes
            "income taxes": "IAS 12",
            "taxation": "IAS 12",
            "tax": "IAS 12",
            "deferred tax": "IAS 12",
            "current tax": "IAS 12",
            
            # Foreign Exchange
            "foreign exchange": "IAS 21",
            "foreign currency": "IAS 21",
            "exchange differences": "IAS 21",
            "translation": "IAS 21",
            
            # Fair Value
            "fair value": "IFRS 13",
            "fair value measurement": "IFRS 13",
            "valuation": "IFRS 13",
            
            # Operating Segments
            "operating segments": "IFRS 8",
            "segment reporting": "IFRS 8",
            "geographical segments": "IFRS 8",
            
            # General Financial Information (fallback categories)
            "directors report": "General",
            "director's report": "General", 
            "chairman's statement": "General",
            "ceo report": "General",
            "risk management": "Risk Management",
            "corporate governance": "Corporate Governance",
            "going concern": "IAS 1",
            "presentation": "IAS 1",
            "financial position": "IAS 1"
        }
        
        logger.info(f"✅ Loaded {len(self.header_standard_mapping)} header-to-standard mappings")

    def _load_standard_keywords(self):
        """Load standard keyword mappings (fallback when taxonomy unavailable)"""
        # This is already implemented in _identify_via_keywords method
        pass
    
    def _remove_audit_report_content(self, document_text: str, document_id: Optional[str] = None) -> str:
        """
        COMPLETELY REMOVE audit report content before processing notes
        
        Same comprehensive filtering as financial statement detector
        """
        logger.info(f"🧹 Pre-filtering audit report content from notes for {document_id or 'document'}")
        
        original_length = len(document_text)
        cleaned_text = document_text
        
        # Step 1: Remove large audit report sections using section boundaries
        audit_section_patterns = [
            # Match entire audit report sections from start to end
            r"INDEPENDENT\s+AUDITOR'?S\s+(?:REPORT|OPINION).*?(?=\n\s*(?:CONSOLIDATED\s+STATEMENT|STATEMENT\s+OF|NOTES?\s+TO|DIRECTORS?\s+REPORT|\d+\s+[A-Z])|\Z)",
            r"AUDITOR'?S\s+(?:REPORT|OPINION).*?(?=\n\s*(?:CONSOLIDATED\s+STATEMENT|STATEMENT\s+OF|NOTES?\s+TO|DIRECTORS?\s+REPORT|\d+\s+[A-Z])|\Z)",
            r"REPORT\s+OF\s+INDEPENDENT\s+AUDITORS?.*?(?=\n\s*(?:CONSOLIDATED\s+STATEMENT|STATEMENT\s+OF|NOTES?\s+TO|DIRECTORS?\s+REPORT|\d+\s+[A-Z])|\Z)",
        ]
        
        for pattern in audit_section_patterns:
            matches = list(re.finditer(pattern, cleaned_text, re.IGNORECASE | re.DOTALL))
            for match in reversed(matches):  # Remove from end to preserve positions
                audit_content = match.group(0)
                logger.info(f"🗑️ REMOVING audit section from notes: {len(audit_content)} characters")
                cleaned_text = cleaned_text[:match.start()] + cleaned_text[match.end():]
        
        # Step 2: Remove individual audit phrases that might be embedded
        audit_exclusion_patterns = [
            r"IN\\s+OUR\\s+OPINION\\s*,?",
            r"WE\\s+HAVE\\s+AUDITED\\s+THE\\s+(?:ACCOMPANYING|CONSOLIDATED|FINANCIAL)",
            r"WE\\s+(?:BELIEVE|CONSIDER)\\s+THAT\\s+THE\\s+AUDIT\\s+EVIDENCE",
            r"OUR\\s+AUDIT\\s+INVOLVED\\s+PERFORMING", 
            r"WE\\s+CONDUCTED\\s+OUR\\s+AUDIT\\s+IN\\s+ACCORDANCE",
            r"BASIS\\s+FOR\\s+(?:OPINION|QUALIFIED\\s+OPINION)",
            r"KEY\\s+AUDIT\\s+MATTERS",
            r"MATERIAL\\s+UNCERTAINTY\\s+RELATED\\s+TO\\s+GOING\\s+CONCERN",
            r"RESPONSIBILITIES\\s+OF\\s+(?:MANAGEMENT|DIRECTORS|THOSE\\s+CHARGED)",
            r"AUDITOR'?S\\s+RESPONSIBILITIES\\s+FOR\\s+THE\\s+AUDIT",
            r"REASONABLE\\s+ASSURANCE\\s+(?:ABOUT|THAT)",
            r"AUDIT\\s+EVIDENCE\\s+(?:WE\\s+HAVE\\s+)?OBTAINED",
            r"PROFESSIONAL\\s+(?:JUDGMENT|SKEPTICISM)",
            r"EMPHASIS\\s+OF\\s+MATTER",
            r"OTHER\\s+MATTER",
            r"REPORT\\s+ON\\s+OTHER\\s+LEGAL",
            r"CHARTERED\\s+ACCOUNTANTS?",
            r"PUBLIC\\s+ACCOUNTANTS?",
            r"REGISTERED\\s+AUDITORS?",
            r"AUDIT\\s+PARTNER",
            r"FOR\\s+AND\\s+ON\\s+BEHALF\\s+OF.*?AUDITORS?"
        ]
        
        for pattern in audit_exclusion_patterns:
            # Remove entire paragraphs/sentences containing audit language
            paragraph_pattern = r'[^\n]*' + pattern + r'[^\n]*(?:\n[^\n]*)*?(?=\n\s*\n|\Z)'
            cleaned_text = re.sub(paragraph_pattern, '', cleaned_text, flags=re.IGNORECASE | re.DOTALL)
        
        # Step 3: Clean up extra whitespace
        cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)
        cleaned_text = cleaned_text.strip()
        
        removed_chars = original_length - len(cleaned_text)
        if removed_chars > 0:
            removal_percentage = (removed_chars / original_length) * 100
            logger.info(f"✂️ AUDIT CONTENT REMOVED FROM NOTES: {removed_chars} characters ({removal_percentage:.1f}%)")
        else:
            logger.info("✅ No audit content detected in notes for removal")
        
        return cleaned_text
    



# Create global instance
standard_identifier = StandardIdentifier()

# Export for use in other modules
__all__ = ["standard_identifier", "StandardIdentifier", "TaggedSentence", "StandardTag", "NotesSection"]