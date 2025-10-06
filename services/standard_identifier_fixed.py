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
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass
import json

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
        
        # Notes extraction patterns
        self.notes_patterns = [
            r"NOTES?\s+TO\s+(?:THE\s+)?(?:CONSOLIDATED\s+)?FINANCIAL\s+STATEMENTS",
            r"NOTES?\s+ON\s+(?:THE\s+)?(?:CONSOLIDATED\s+)?FINANCIAL\s+STATEMENTS", 
            r"NOTES?\s+TO\s+(?:THE\s+)?ACCOUNTS",
            r"NOTES?\s+ON\s+(?:THE\s+)?ACCOUNTS"
        ]
        
        # Note numbering patterns
        self.note_number_patterns = [
            r"(?:^|\n)\s*(\d+(?:\.\d+)*)\s+([A-Z][^\\n]*?)(?=\\n|$)",  # "1 Revenue", "2.1 Accounting Policies"
            r"(?:^|\n)\s*NOTE\s+(\d+(?:\.\d+)*)\s*[-–]\s*([A-Z][^\\n]*?)(?=\\n|$)",  # "NOTE 1 - Revenue"
            r"(?:^|\n)\s*([A-Z][A-Z\s]{10,}?)(?=\\n)",  # "REVENUE RECOGNITION", "PROPERTY PLANT EQUIPMENT"
        ]
        
        # Exclusion patterns for financial statements and audit reports
        self.exclusion_patterns = [
            r"CONSOLIDATED\s+(?:STATEMENT|BALANCE\s+SHEET)",
            r"INDEPENDENT\s+AUDITOR'?S\s+REPORT",
            r"AUDITOR'?S\s+REPORT", 
            r"BASIS\s+FOR\s+OPINION",
            r"KEY\s+AUDIT\s+MATTERS"
        ]
    
    def identify_standards_in_notes(self, document_text: str, document_id: str) -> Dict[str, Any]:
        """
        Main method to identify accounting standards in Notes to Accounts
        
        Returns simplified tagging of sentences with applicable standards
        """
        logger.info(f"🏷️ Starting standard identification for document {document_id}")
        
        # Step 1: Extract Notes to Accounts sections
        notes_sections = self._extract_notes_sections(document_text, document_id)
        
        if not notes_sections:
            logger.warning(f"⚠️ No Notes sections found in document {document_id}")
            return {
                "document_id": document_id,
                "tagged_sentences": [],
                "standards_found": []
            }
        
        logger.info(f"📝 Found {len(notes_sections)} Notes sections")
        
        # Step 2: Tag every sentence in each notes section (including tables)
        all_tagged_sentences = []
        all_standards = set()
        
        for section in notes_sections:
            # Process regular sentences
            tagged_sentences = self._tag_sentences_in_section(section)
            
            # Process tables in the section
            table_sentences = self._extract_and_tag_tables(section)
            tagged_sentences.extend(table_sentences)
            
            # Simplify each tagged sentence to exactly the 4 fields needed
            for sentence in tagged_sentences:
                if sentence.standards:  # Only include sentences with identified standards
                    simple_sentence = {
                        "text": sentence.sentence_text,
                        "page": sentence.page_number,
                        "note": sentence.note_number,
                        "standards": [std.standard_code for std in sentence.standards]
                    }
                    all_tagged_sentences.append(simple_sentence)
                    
                    # Collect all standards
                    for standard in sentence.standards:
                        all_standards.add(standard.standard_code)
        
        # Step 3: Return simplified results
        results = {
            "document_id": document_id,
            "tagged_sentences": all_tagged_sentences,
            "standards_found": sorted(list(all_standards))
        }
        
        logger.info(f"✅ Standard identification complete: {len(all_tagged_sentences)} sentences tagged with {len(all_standards)} standards")
        
        return results
    
    def _extract_notes_sections(self, document_text: str, document_id: str) -> List[NotesSection]:
        """Extract Notes to Accounts sections from document"""
        
        # Find Notes to Accounts section
        notes_start = None
        for pattern in self.notes_patterns:
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
        
        # Remove audit report content if present
        for exclusion_pattern in self.exclusion_patterns:
            notes_content = re.sub(exclusion_pattern + r".*?(?=\\n\\n|$)", "", 
                                  notes_content, flags=re.IGNORECASE | re.DOTALL)
        
        # Split into individual note sections
        sections = self._split_into_note_sections(notes_content)
        
        return sections
    
    def _split_into_note_sections(self, notes_content: str) -> List[NotesSection]:
        """Split notes content into individual note sections"""
        
        sections = []
        
        # Try different note numbering patterns
        for pattern in self.note_number_patterns:
            matches = list(re.finditer(pattern, notes_content, re.MULTILINE | re.IGNORECASE))
            
            if matches:
                logger.info(f"🔢 Using pattern to split {len(matches)} note sections")
                
                for i, match in enumerate(matches):
                    note_number = match.group(1) if match.lastindex and match.lastindex >= 1 else f"Note_{i+1}"
                    section_title = match.group(2) if match.lastindex and match.lastindex >= 2 else "Untitled Section"
                    
                    # Extract content until next section
                    start_pos = match.start()
                    end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(notes_content)
                    
                    section_content = notes_content[start_pos:end_pos].strip()
                    
                    if len(section_content) > 100:  # Minimum viable section size
                        # Estimate page numbers (rough calculation)
                        page_estimate = [max(1, start_pos // 3000 + 1)]
                        
                        sections.append(NotesSection(
                            section_title=section_title.strip(),
                            note_number=note_number,
                            subsection=None,
                            content=section_content,
                            page_numbers=page_estimate,
                            tagged_sentences=[]
                        ))
                
                break  # Use first successful pattern
        
        # Fallback: treat entire notes as single section
        if not sections:
            logger.info("📄 Using fallback: treating entire notes as single section")
            sections.append(NotesSection(
                section_title="Notes to Accounts",
                note_number="1",
                subsection=None,
                content=notes_content[:10000],  # Limit size for processing
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
            r'\\.',  # Period
            r'\\;',  # Semicolon 
            r'\\n\\n',  # Double newline
            r'\\n(?=[A-Z])',  # Newline followed by capital letter
            r'(?<=\\d)\\s*\\n(?=[A-Z])',  # Newline after number, before capital
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
                    
                    if standard_ref and standard_ref not in [s.standard_code for s in standards]:
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
    
    def _load_standard_keywords(self):
        """Load standard keyword mappings (fallback when taxonomy unavailable)"""
        # This is already implemented in _identify_via_keywords method
        pass
    



# Create global instance
standard_identifier = StandardIdentifier()

# Export for use in other modules
__all__ = ["standard_identifier", "StandardIdentifier", "TaggedSentence", "StandardTag", "NotesSection"]