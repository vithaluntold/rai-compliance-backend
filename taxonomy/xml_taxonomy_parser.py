#!/usr/bin/env python3
"""
XML Taxonomy Integration System for IFRS/IAS Standards
Integrates XML taxonomies with 5D classification system
"""

import xml.etree.ElementTree as ET
import json
import os
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class XBRLTaxonomyParser:
    """Parser for IFRS/IAS XML taxonomies"""
    
    def __init__(self, taxonomy_dir: str):
        self.taxonomy_dir = Path(taxonomy_dir)
        self.concepts = {}
        self.presentations = {}
        self.references = {}
        self.loaded_taxonomies = set()
        
    def load_ifrs_taxonomy(self) -> Dict[str, Any]:
        """Load IFRS taxonomy files and build concept mapping"""
        try:
            # Load core IFRS taxonomy files
            self._load_concepts_schema()
            self._load_presentation_linkbase()
            self._load_reference_linkbase()
            self._load_standard_linkbases()  # New method to load individual standard linkbases
            
            return {
                "concepts": self.concepts,
                "presentations": self.presentations, 
                "references": self.references
            }
        except Exception as e:
            logger.error(f"Error loading IFRS taxonomy: {e}")
            return {}
    
    def _load_concepts_schema(self):
        """Load concepts from IFRS schema files"""
        # Look for the main IFRS core schema file
        schema_files = list(self.taxonomy_dir.glob("**/*-cor_*.xsd"))
        
        if not schema_files:
            logger.warning(f"No core schema files found in {self.taxonomy_dir}")
            return
            
        main_schema = schema_files[0]  # Use first found core schema
        logger.info(f"Loading IFRS taxonomy from: {main_schema}")
        
        try:
            tree = ET.parse(main_schema)
            root = tree.getroot()
            
            # Define namespaces for parsing
            namespaces = {
                'xsd': 'http://www.w3.org/2001/XMLSchema',
                'xbrli': 'http://www.xbrl.org/2003/instance',
                'ifrs-full': 'https://xbrl.ifrs.org/taxonomy/2025-03-27/ifrs-full'
            }
            
            # Extract all element definitions
            elements = root.findall(".//xsd:element", namespaces)
            logger.info(f"Found {len(elements)} IFRS elements")
            
            # Debug: log first few elements to see structure
            if len(elements) > 0:
                logger.info(f"First element sample: tag={elements[0].tag}, attrib={dict(elements[0].attrib)}")
            else:
                logger.warning("No elements found - checking root structure")
                logger.info(f"Root tag: {root.tag}, Root attrib: {dict(root.attrib)}")
                # Try without namespace
                elements_no_ns = root.findall(".//element")
                logger.info(f"Elements without namespace: {len(elements_no_ns)}")
            
            for element in elements:
                concept_id = element.get("id", "")
                concept_name = element.get("name", "")
                
                # Debug: log why concepts might be skipped
                if not concept_id:
                    logger.debug(f"Element missing id: {element.tag} {dict(element.attrib)}")
                if not concept_name:
                    logger.debug(f"Element missing name: {element.tag} {dict(element.attrib)}")
                
                if concept_id and concept_name:
                    # Parse IFRS-specific attributes
                    element_type = element.get("type", "")
                    period_type = element.get("{http://www.xbrl.org/2003/instance}periodType")
                    balance = element.get("{http://www.xbrl.org/2003/instance}balance")
                    abstract = element.get("abstract") == "true"
                    
                    # Categorize element
                    category = self._categorize_ifrs_element(concept_name, element_type, abstract)
                    
                    # Extract IAS/IFRS standard reference
                    standard_ref = self._extract_standard_reference(concept_name)
                    
                    self.concepts[concept_id] = {
                        "id": concept_id,
                        "name": concept_name,
                        "type": element_type,
                        "period_type": period_type,
                        "balance": balance,
                        "abstract": abstract,
                        "category": category,
                        "standard_reference": standard_ref,
                        "substitution_group": element.get("substitutionGroup")
                    }
                    
            logger.info(f"Loaded {len(self.concepts)} IFRS concepts")
            
            if len(self.concepts) > 0:
                # Log first few concept IDs for debugging
                sample_concepts = list(self.concepts.keys())[:3]
                logger.info(f"Sample concept IDs: {sample_concepts}")
            else:
                logger.warning("No concepts were loaded from schema - check schema parsing logic")
            
        except Exception as e:
            logger.error(f"Could not parse schema file {main_schema}: {e}")
            
    def _categorize_ifrs_element(self, name: str, element_type: str, abstract: bool) -> str:
        """Categorize IFRS elements based on naming patterns and types"""
        if abstract:
            return "abstract"
        elif "Member" in name or "Member" in element_type:
            return "member"  
        elif "Axis" in name or "Domain" in name:
            return "dimension"
        elif "monetaryItemType" in element_type:
            return "monetary"
        elif "booleanItemType" in element_type:
            return "boolean"
        elif "stringItemType" in element_type:
            return "text"
        elif any(keyword in name.lower() for keyword in ["policy", "policies"]):
            return "accounting_policy"
        elif any(keyword in name.lower() for keyword in ["risk", "exposure"]):
            return "risk_disclosure"
        elif any(keyword in name.lower() for keyword in ["estimate", "assumption"]):
            return "estimate_judgement"
        else:
            return "other"
            
    def _extract_standard_reference(self, concept_name: str) -> Optional[str]:
        """Extract IAS/IFRS standard reference from concept name"""
        import re
        
        # Common patterns for standard references in IFRS taxonomy
        patterns = [
            r'IAS(\d+)',
            r'IFRS(\d+)',
            r'SIC(\d+)', 
            r'IFRIC(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, concept_name, re.IGNORECASE)
            if match:
                standard_type = pattern.replace(r'(\d+)', '').replace('\\', '')
                number = match.group(1)
                return f"{standard_type} {number}"
                
        return None
    
    def _load_presentation_linkbase(self):
        """Load presentation hierarchy from linkbase files"""
        linkbase_files = list(self.taxonomy_dir.glob("**/*-pre.xml"))
        
        for linkbase_file in linkbase_files:
            try:
                tree = ET.parse(linkbase_file)
                root = tree.getroot()
                
                # Extract presentation relationships
                for link in root.findall(".//{http://www.xbrl.org/2003/linkbase}presentationLink"):
                    role = link.get("{http://www.w3.org/1999/xlink}role")
                    
                    if role not in self.presentations:
                        self.presentations[role] = []
                    
                    # Process presentation arcs
                    for arc in link.findall(".//{http://www.xbrl.org/2003/linkbase}presentationArc"):
                        from_ref = arc.get("{http://www.w3.org/1999/xlink}from")
                        to_ref = arc.get("{http://www.w3.org/1999/xlink}to")
                        order = arc.get("order", "1.0")
                        
                        self.presentations[role].append({
                            "from": from_ref,
                            "to": to_ref, 
                            "order": float(order)
                        })
                        
            except Exception as e:
                logger.warning(f"Could not parse linkbase file {linkbase_file}: {e}")
    
    def _load_reference_linkbase(self):
        """Load reference information from linkbase files"""
        ref_files = list(self.taxonomy_dir.glob("**/*-ref.xml"))
        
        for ref_file in ref_files:
            try:
                tree = ET.parse(ref_file)
                root = tree.getroot()
                
                # Extract references
                for reference in root.findall(".//{http://www.xbrl.org/2003/linkbase}reference"):
                    ref_id = reference.get("id")
                    if ref_id:
                        self.references[ref_id] = {}
                        
                        for part in reference.findall(".//{http://www.xbrl.org/2005/xbrldt}referencePart"):
                            part_name = part.get("name")
                            part_value = part.text
                            if part_name and part_value:
                                self.references[ref_id][part_name] = part_value
                                
            except Exception as e:
                logger.warning(f"Could not parse reference file {ref_file}: {e}")
                
    def _load_standard_linkbases(self):
        """Load individual IAS/IFRS standard linkbases to map concepts to standards"""
        # Try multiple possible linkbases directories
        possible_dirs = [
            self.taxonomy_dir / "linkbases",
            self.taxonomy_dir / "IFRSAT-2025" / "IFRSAT-2025" / "full_ifrs",
            self.taxonomy_dir / "IFRSAT-2025" / "full_ifrs",
            self.taxonomy_dir / "full_ifrs"
        ]
        
        linkbases_dir = None
        for dir_path in possible_dirs:
            if dir_path.exists():
                linkbases_dir = dir_path
                break
                
        if not linkbases_dir:
            logger.info(f"No linkbases directory found, using pattern-based taxonomy validation")
            return
            
        logger.info(f"Processing linkbases from: {linkbases_dir}")
        processed_standards = 0
        
        # Process each standard directory - only look in linkbases-ea subfolder
        linkbases_ea_dir = linkbases_dir / "linkbases-ea"
        if not linkbases_ea_dir.exists():
            logger.warning(f"linkbases-ea directory not found in {linkbases_dir}")
            return
            
        # Process each standard directory inside linkbases-ea
        for standard_dir in linkbases_ea_dir.iterdir():
            if standard_dir.is_dir():
                standard_name = self._normalize_standard_name(standard_dir.name)
                logger.info(f"Processing standard: {standard_name}")
                
                # Load presentation files to get concepts for this standard
                presentation_files = list(standard_dir.glob("pre_*.xml"))
                reference_files = list(standard_dir.glob("ref_*.xml"))
                
                concepts_found = 0
                
                # Process presentation files (preferred method)
                for pres_file in presentation_files:
                    try:
                        tree = ET.parse(pres_file)
                        root = tree.getroot()
                        
                        # Define namespaces for XBRL parsing
                        namespaces = {
                            'link': 'http://www.xbrl.org/2003/linkbase',
                            'xlink': 'http://www.w3.org/1999/xlink'
                        }
                        
                        # Find all concept references in this standard's presentation
                        for loc in root.findall(".//link:loc", namespaces):
                            href = loc.get("{http://www.w3.org/1999/xlink}href", "")
                            if "#" in href:
                                concept_id = href.split("#")[-1]
                                
                                # Update concept with standard reference
                                if concept_id in self.concepts:
                                    self.concepts[concept_id]["standard_reference"] = standard_name
                                    concepts_found += 1
                                else:
                                    # Debug: log first few missing concepts for troubleshooting
                                    if concepts_found == 0:  # Only log for first standard being processed
                                        logger.debug(f"Concept ID not found in schema: {concept_id}")
                                    
                    except Exception as e:
                        logger.warning(f"Could not process presentation file {pres_file}: {e}")
                
                # If no presentation files, try reference files
                if not presentation_files and reference_files:
                    logger.info(f"  No presentation files for {standard_name}, trying reference files")
                    
                    for ref_file in reference_files:
                        try:
                            tree = ET.parse(ref_file)
                            root = tree.getroot()
                            
                            # Define namespaces for XBRL parsing
                            namespaces = {
                                'link': 'http://www.xbrl.org/2003/linkbase',
                                'xlink': 'http://www.w3.org/1999/xlink'
                            }
                            
                            # Find all concept references in this standard's reference linkbase
                            for loc in root.findall(".//link:loc", namespaces):
                                href = loc.get("{http://www.w3.org/1999/xlink}href", "")
                                if "#" in href:
                                    concept_id = href.split("#")[-1]
                                    
                                    # Update concept with standard reference
                                    if concept_id in self.concepts:
                                        self.concepts[concept_id]["standard_reference"] = standard_name
                                        concepts_found += 1
                                        
                        except Exception as e:
                            logger.warning(f"Could not process reference file {ref_file}: {e}")
                
                # Also check for concepts that match the standard name pattern in the main schema
                if concepts_found == 0:
                    # Fallback: search main schema for concepts containing the standard name
                    standard_pattern = standard_name.replace(" ", "").lower()  # "IAS 32" -> "ias32"
                    
                    for concept_id, concept_data in self.concepts.items():
                        concept_name_lower = concept_data.get("name", "").lower()
                        if standard_pattern in concept_name_lower or standard_name in concept_data.get("id", ""):
                            if not concept_data.get("standard_reference"):  # Don't override existing
                                self.concepts[concept_id]["standard_reference"] = standard_name
                                concepts_found += 1
                
                if concepts_found > 0:
                    logger.info(f"  Found {concepts_found} concepts for {standard_name}")
                else:
                    logger.debug(f"  No concepts found for {standard_name} (using keyword-based fallback)")
                        
                processed_standards += 1
                
        logger.info(f"Processed {processed_standards} standards with linkbases")
                        
    def _normalize_standard_name(self, directory_name: str) -> str:
        """Convert directory name to standard format (e.g., 'ias_34' -> 'IAS 34')"""
        parts = directory_name.split("_")
        if len(parts) >= 2:
            standard_type = parts[0].upper()
            number = parts[1]
            return f"{standard_type} {number}"
        return directory_name.upper()

class TaxonomyValidator:
    """Validates 5D tags against IFRS taxonomy concepts"""
    
    def __init__(self, taxonomy_parser: XBRLTaxonomyParser):
        self.taxonomy = taxonomy_parser
        self.concept_mapping = self._build_concept_mapping()
        
    def _build_concept_mapping(self) -> Dict[str, List[str]]:
        """Build mapping between 5D tags and IFRS concepts"""
        return {
            # Narrative Categories Mapping
            "accounting_policies_note": [
                "AccountingPolicies",
                "SignificantAccountingPolicies", 
                "SummaryOfSignificantAccountingPolicies"
            ],
            "event_based": [
                "EventsAfterReportingPeriodNonAdjusting",
                "SubsequentEvents",
                "EventsAfterBalance Sheet Date"
            ],
            "measurement_basis": [
                "FairValueMeasurement",
                "AmortisedCost",
                "ImpairmentLoss"
            ],
            "risk_exposure": [
                "FinancialRiskManagement",
                "CreditRisk",
                "LiquidityRisk",
                "MarketRisk"
            ],
            "estimates_judgements": [
                "CriticalAccountingEstimates",
                "SourcesOfEstimationUncertainty",
                "KeyAssumptions"
            ],
            # Cross Reference Anchors Mapping  
            "notes_main": ["NotesToFinancialStatements"],
            "policies": ["AccountingPolicies"],
            "primary_statement": [
                "StatementOfFinancialPosition",
                "StatementOfProfitOrLoss",
                "StatementOfCashFlows"
            ]
        }
        
    def validate_content_tags(self, content_tags: Dict[str, List[str]], 
                            detected_concepts: List[str]) -> Dict[str, Any]:
        """Validate content 5D tags against detected IFRS concepts"""
        
        validation_result = {
            "valid": True,
            "conflicts": [],
            "suggestions": [],
            "normalized_tags": content_tags.copy()
        }
        
        # Check narrative categories against concepts
        for category in content_tags.get("narrative_categories", []):
            expected_concepts = self.concept_mapping.get(category, [])
            
            # Check if detected concepts align with category
            concept_match = any(
                concept in detected_concepts 
                for concept in expected_concepts
            )
            
            if not concept_match and detected_concepts:
                validation_result["conflicts"].append({
                    "tag": category,
                    "expected_concepts": expected_concepts,
                    "detected_concepts": detected_concepts
                })
                validation_result["valid"] = False
                
        # Suggest better tags based on detected concepts
        for concept in detected_concepts:
            for tag, concepts in self.concept_mapping.items():
                if concept in concepts and tag not in content_tags.get("narrative_categories", []):
                    validation_result["suggestions"].append({
                        "suggested_tag": tag,
                        "reason": f"Detected concept: {concept}"
                    })
                    
        return validation_result
        
    def normalize_tag_values(self, tags: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Normalize tag values using canonical IFRS taxonomy terms"""
        
        normalized = {}
        
        for tag_category, values in tags.items():
            normalized[tag_category] = []
            
            for value in values:
                # Look up canonical form in taxonomy
                canonical_value = self._get_canonical_value(value)
                normalized[tag_category].append(canonical_value)
                
        return normalized
        
    def _get_canonical_value(self, value: str) -> str:
        """Get canonical form of tag value from taxonomy"""
        # Simple normalization - extend based on taxonomy structure
        canonical_map = {
            "disclosure_narrative": "DisclosureNarrative",
            "accounting_policies_note": "AccountingPolicies", 
            "notes_main": "NotesToFinancialStatements",
            "current_period": "CurrentPeriod",
            "qualitative_only": "QualitativeDisclosure"
        }
        
        return canonical_map.get(value, value)

def setup_taxonomy_system(taxonomy_dir: str) -> tuple[XBRLTaxonomyParser, TaxonomyValidator]:
    """Setup complete taxonomy validation system"""
    
    parser = XBRLTaxonomyParser(taxonomy_dir)
    taxonomy_data = parser.load_ifrs_taxonomy()
    
    if not taxonomy_data:
        logger.warning("Could not load taxonomy data, using fallback validation")
    
    validator = TaxonomyValidator(parser)
    
    return parser, validator

# Integration example
def integrate_with_chunking_system():
    """Example of how to integrate with existing chunking system"""
    
    # Setup taxonomy system
    taxonomy_dir = "path/to/ifrs/taxonomy/files"
    parser, validator = setup_taxonomy_system(taxonomy_dir)
    
    # Example usage in content processing pipeline
    def process_content_with_taxonomy_validation(content: str, ai_tags: Dict[str, List[str]]):
        
        # 1. Detect IFRS concepts in content (using NLP/AI)
        detected_concepts = detect_ifrs_concepts(content)
        
        # 2. Validate AI-generated tags against taxonomy
        validation_result = validator.validate_content_tags(ai_tags, detected_concepts)
        
        # 3. Normalize tags using canonical taxonomy values
        normalized_tags = validator.normalize_tag_values(ai_tags)
        
        # 4. Return enhanced content with validated tags
        return {
            "content": content,
            "tags": normalized_tags,
            "validation": validation_result,
            "detected_concepts": detected_concepts
        }
        
def detect_ifrs_concepts(content: str) -> List[str]:
    """Detect IFRS concepts in content - placeholder for NLP implementation"""
    # This would use NLP libraries to detect IFRS concepts
    # Return example for demonstration
    return ["AccountingPolicies", "FairValueMeasurement"]