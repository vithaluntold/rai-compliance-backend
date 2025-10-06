"""
Intelligent Notes Accumulator

Consolidates sentences from Standard Identifier by accounting standard into
simplified JSON structure optimized for AI processing.
"""

import logging
from typing import Dict, List, Any, Set
from collections import defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ConsolidatedStandard:
    """Represents all content consolidated for a single accounting standard"""
    standard_code: str
    standard_name: str
    total_sentences: int
    notes_covered: List[str]
    pages_covered: List[int]
    consolidated_text: str
    sentence_count_by_note: Dict[str, int]


class IntelligentNotesAccumulator:
    """
    Accumulates and consolidates Notes content by accounting standard
    
    Takes tagged sentences from Standard Identifier and creates consolidated
    content blocks organized by accounting standard for efficient AI processing.
    """
    
    def __init__(self):
        """Initialize accumulator with standard names mapping"""
        self.standard_names = {
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
            "IFRS 4": "Insurance Contracts",
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
    
    def consolidate_by_standard(self, standard_identifier_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Consolidate tagged sentences by accounting standard into AI-optimized structure
        
        Args:
            standard_identifier_output: Output from StandardIdentifier.identify_standards_in_notes()
            
        Returns:
            Consolidated structure optimized for AI processing
        """
        logger.info(f"🔄 Starting consolidation of {len(standard_identifier_output.get('tagged_sentences', []))} tagged sentences")
        
        tagged_sentences = standard_identifier_output.get('tagged_sentences', [])
        document_id = standard_identifier_output.get('document_id', 'unknown')
        
        # Group sentences by standard
        standard_groups = defaultdict(list)
        
        for sentence in tagged_sentences:
            standards = sentence.get('standards', [])
            
            # Add sentence to each applicable standard group
            for standard_code in standards:
                standard_groups[standard_code].append(sentence)
        
        # Consolidate each standard group
        consolidated_standards = {}
        
        for standard_code, sentences in standard_groups.items():
            consolidated = self._consolidate_standard_content(standard_code, sentences)
            consolidated_standards[standard_code] = consolidated
        
        # Create final AI-optimized structure
        result = {
            "document_id": document_id,
            "total_standards_found": len(consolidated_standards),
            "consolidation_summary": {
                "original_sentences": len(tagged_sentences),
                "standards_processed": list(consolidated_standards.keys()),
                "total_consolidated_blocks": len(consolidated_standards)
            },
            "consolidated_content": consolidated_standards
        }
        
        logger.info(f"✅ Consolidation complete: {len(consolidated_standards)} standards consolidated")
        return result
    
    def _consolidate_standard_content(self, standard_code: str, sentences: List[Dict]) -> Dict[str, Any]:
        """
        Consolidate all sentences for a single accounting standard with parameter-based aggregation
        
        Args:
            standard_code: The accounting standard code (e.g., "IFRS 9")
            sentences: List of sentences tagged with this standard
            
        Returns:
            Consolidated content block for this standard with aggregated sections
        """
        if not sentences:
            return {}
        
        # Group sentences by shared parameters (note + page combination)
        parameter_groups = defaultdict(list)
        
        for sentence in sentences:
            note = sentence.get('note', 'unknown')
            page = sentence.get('page', 0)
            
            # Create parameter key for grouping
            param_key = f"note_{note}_page_{page}"
            parameter_groups[param_key].append(sentence)
        
        # Create aggregated sections
        aggregated_sections = []
        all_notes = set()
        all_pages = set()
        total_sentences = len(sentences)
        
        for param_key, grouped_sentences in parameter_groups.items():
            section = self._create_aggregated_section(param_key, grouped_sentences)
            if section:
                aggregated_sections.append(section)
                
                # Collect overall metadata using internal fields
                if section['_internal_note'] and section['_internal_note'] != 'unknown':
                    all_notes.add(section['_internal_note'])
                if section['_internal_page'] and section['_internal_page'] > 0:
                    all_pages.add(section['_internal_page'])
        
        # Sort sections by note number (natural sort) using internal fields
        aggregated_sections.sort(key=lambda x: (
            int(x['_internal_note']) if x['_internal_note'].isdigit() else float('inf'), 
            x['_internal_page']
        ))
        
        # Remove internal fields before returning (clean up for AI consumption)
        for section in aggregated_sections:
            section.pop('_internal_note', None)
            section.pop('_internal_page', None)
        
        # Create overall consolidated text from all sections
        overall_content = " ".join([section['content'] for section in aggregated_sections])
        
        # Create simplified structure optimized for AI with aggregated sections
        return {
            "standard_code": standard_code,
            "standard_name": self.standard_names.get(standard_code, "Unknown Standard"),
            "content": overall_content,
            "aggregated_sections": aggregated_sections,
            "metadata": {
                "sentence_count": total_sentences,
                "section_count": len(aggregated_sections),
                "notes_covered": sorted(list(all_notes)),
                "pages_covered": sorted(list(all_pages)),
                "content_length": len(overall_content)
            }
        }
    
    def _create_aggregated_section(self, param_key: str, sentences: List[Dict]) -> Dict[str, Any]:
        """
        Create an aggregated section for sentences with the same parameters
        
        Args:
            param_key: Parameter key (e.g., "note_5_page_15")
            sentences: List of sentences with same parameters
            
        Returns:
            Simplified aggregated section dictionary (note/page info removed as it's in metadata)
        """
        if not sentences:
            return {}
        
        # Extract common parameters from first sentence (for internal use only)
        first_sentence = sentences[0]
        note = first_sentence.get('note', 'unknown')
        page = first_sentence.get('page', 0)
        
        # Aggregate all sentence texts
        sentence_texts = []
        for sentence in sentences:
            text = sentence.get('text', '').strip()
            if text:
                sentence_texts.append(text)
        
        if not sentence_texts:
            return {}
        
        # Join sentences with proper spacing (no headers - that info is in metadata)
        aggregated_content = " ".join(sentence_texts)
        
        # Return simplified structure without redundant note/page info
        return {
            "sentence_count": len(sentences),
            "content": aggregated_content,
            "_internal_note": note,  # Keep for sorting, but don't expose to AI
            "_internal_page": page   # Keep for sorting, but don't expose to AI
        }
    
    def get_ai_optimized_content(self, consolidated_output: Dict[str, Any], max_length: int = 4000) -> Dict[str, str]:
        """
        Extract content in format optimized for AI processing with length limits
        
        Args:
            consolidated_output: Output from consolidate_by_standard()
            max_length: Maximum length per standard content block
            
        Returns:
            Dictionary mapping standard codes to truncated content
        """
        ai_content = {}
        
        consolidated_content = consolidated_output.get('consolidated_content', {})
        
        for standard_code, standard_data in consolidated_content.items():
            content = standard_data.get('content', '')
            
            # Truncate if needed
            if len(content) > max_length:
                # Try to truncate at sentence boundary
                truncated = content[:max_length]
                last_period = truncated.rfind('.')
                if last_period > max_length * 0.8:  # If we can keep 80% of content
                    content = truncated[:last_period + 1] + "..."
                else:
                    content = truncated + "..."
            
            ai_content[standard_code] = content
        
        return ai_content


# Global instance
intelligent_accumulator = IntelligentNotesAccumulator()


def consolidate_notes_by_standard(standard_identifier_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to consolidate Standard Identifier output by accounting standard
    
    Args:
        standard_identifier_output: Output from StandardIdentifier.identify_standards_in_notes()
        
    Returns:
        Consolidated structure with all sentences grouped by accounting standard
    """
    return intelligent_accumulator.consolidate_by_standard(standard_identifier_output)


def get_ai_content_by_standard(standard_identifier_output: Dict[str, Any], max_length: int = 4000) -> Dict[str, str]:
    """
    Get AI-optimized content blocks by standard with length limits
    
    Args:
        standard_identifier_output: Output from StandardIdentifier.identify_standards_in_notes()
        max_length: Maximum content length per standard
        
    Returns:
        Dictionary mapping standard codes to consolidated content strings
    """
    consolidated = intelligent_accumulator.consolidate_by_standard(standard_identifier_output)
    return intelligent_accumulator.get_ai_optimized_content(consolidated, max_length)