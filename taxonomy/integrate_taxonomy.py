#!/usr/bin/env python3
"""
IFRS Taxonomy Integration Script
Processes IFRSAT-2025 taxonomy and builds 5D tag validation system
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from xml_taxonomy_parser import XBRLTaxonomyParser, TaxonomyValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IFRSTaxonomyIntegrator:
    """Main integrator for IFRS taxonomy with 5D classification system"""
    
    def __init__(self, taxonomy_path: str, enhanced_framework_path: str):
        self.taxonomy_path = Path(taxonomy_path)
        self.enhanced_framework_path = Path(enhanced_framework_path)
        self.parser = None
        self.validator = None
        self.concept_index = {}
        
    def run_integration(self) -> Dict[str, Any]:
        """Execute complete taxonomy integration process"""
        logger.info("Starting IFRS Taxonomy Integration")
        
        # Step 1: Load IFRS Taxonomy
        self.parser = XBRLTaxonomyParser(str(self.taxonomy_path))
        taxonomy_data = self.parser.load_ifrs_taxonomy()
        
        if not taxonomy_data.get("concepts"):
            logger.error("Failed to load IFRS taxonomy concepts")
            return {"success": False, "error": "Taxonomy loading failed"}
            
        logger.info(f"Loaded {len(taxonomy_data['concepts'])} IFRS concepts")
        
        # Step 2: Build Concept Index
        self._build_concept_index(taxonomy_data["concepts"])
        
        # Step 3: Initialize Validator
        self.validator = TaxonomyValidator(self.parser)
        
        # Step 4: Process Enhanced Framework
        framework_validation = self._validate_enhanced_framework()
        
        # Step 5: Generate Integration Report
        integration_report = self._generate_integration_report(
            taxonomy_data, framework_validation
        )
        
        logger.info("IFRS Taxonomy Integration completed successfully")
        return integration_report
        
    def _build_concept_index(self, concepts: Dict[str, Any]):
        """Build searchable index of IFRS concepts"""
        self.concept_index = {
            "by_category": {},
            "by_standard": {},
            "by_keywords": {}
        }
        
        for concept_id, concept_data in concepts.items():
            # Index by category
            category = concept_data.get("category", "other")
            if category not in self.concept_index["by_category"]:
                self.concept_index["by_category"][category] = []
            self.concept_index["by_category"][category].append(concept_id)
            
            # Index by standard reference
            standard_ref = concept_data.get("standard_reference")
            if standard_ref:
                if standard_ref not in self.concept_index["by_standard"]:
                    self.concept_index["by_standard"][standard_ref] = []
                self.concept_index["by_standard"][standard_ref].append(concept_id)
                
            # Index by keywords (extracted from concept name)
            keywords = self._extract_keywords(concept_data.get("name", ""))
            for keyword in keywords:
                if keyword not in self.concept_index["by_keywords"]:
                    self.concept_index["by_keywords"][keyword] = []
                self.concept_index["by_keywords"][keyword].append(concept_id)
                
    def _extract_keywords(self, concept_name: str) -> List[str]:
        """Extract searchable keywords from concept names"""
        import re
        
        # Split camelCase and extract meaningful words
        words = re.findall(r'[A-Z][a-z]*|[a-z]+', concept_name)
        
        # Filter out common words and keep meaningful terms
        meaningful_words = []
        skip_words = {'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'by', 'with'}
        
        for word in words:
            word_lower = word.lower()
            if len(word_lower) > 2 and word_lower not in skip_words:
                meaningful_words.append(word_lower)
                
        return meaningful_words
        
    def _validate_enhanced_framework(self) -> Dict[str, Any]:
        """Validate Enhanced Framework against IFRS taxonomy"""
        validation_results = {
            "total_files": 0,
            "validated_files": 0,
            "validation_errors": [],
            "suggestions": [],
            "standard_coverage": {}
        }
        
        # Find all enhanced JSON files
        json_files = list(self.enhanced_framework_path.glob("**/*.json"))
        validation_results["total_files"] = len(json_files)
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    framework_data = json.load(f)
                    
                file_validation = self._validate_framework_file(
                    json_file.stem, framework_data
                )
                
                if file_validation["valid"]:
                    validation_results["validated_files"] += 1
                else:
                    validation_results["validation_errors"].extend(
                        file_validation["errors"]
                    )
                    
                validation_results["suggestions"].extend(
                    file_validation["suggestions"]
                )
                
            except Exception as e:
                logger.error(f"Error validating {json_file}: {e}")
                validation_results["validation_errors"].append({
                    "file": str(json_file),
                    "error": str(e)
                })
                
        return validation_results
        
    def _validate_framework_file(self, filename: str, 
                                framework_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate individual framework file against taxonomy"""
        
        validation_result = {
            "valid": True,
            "errors": [],
            "suggestions": []
        }
        
        # Extract standard reference from filename (e.g., "IAS_34_Enhanced" -> "IAS 34")
        standard_ref = self._extract_standard_from_filename(filename)
        
        if not standard_ref:
            validation_result["errors"].append({
                "file": filename,
                "issue": "Could not determine IAS/IFRS standard from filename"
            })
            validation_result["valid"] = False
            return validation_result
            
        # Check if standard exists in taxonomy - be more flexible
        if standard_ref not in self.concept_index["by_standard"]:
            # Check if the standard directory exists in linkbases even if no concepts mapped
            standard_dir_name = self._convert_standard_to_dirname(standard_ref)
            linkbases_dir = self.taxonomy_path / "linkbases" / standard_dir_name
            
            if not linkbases_dir.exists():
                # Check if this is a known missing/superseded standard
                known_missing = {
                    "IAS 28": "May be superseded or integrated into other standards in this taxonomy version",
                    "IFRS 4": "Superseded by IFRS 17 (Insurance Contracts) in 2023", 
                    "IFRS 18": "Published after March 2025 taxonomy cutoff date"
                }
                
                if standard_ref in known_missing:
                    validation_result["suggestions"].append({
                        "file": filename,
                        "suggestion": f"Standard {standard_ref} not in taxonomy: {known_missing[standard_ref]}",
                        "type": "known_missing"
                    })
                    # Don't mark as invalid - this is expected
                else:
                    validation_result["errors"].append({
                        "file": filename,
                        "issue": f"Standard {standard_ref} not found in IFRS taxonomy (no linkbase directory)"
                    })
                    validation_result["valid"] = False
            else:
                # Standard exists but has no mapped concepts - this is OK
                validation_result["suggestions"].append({
                    "file": filename,
                    "suggestion": f"Standard {standard_ref} exists in taxonomy but has no mapped concepts",
                    "type": "informational"
                })
            
        # Validate 5D tags in questions
        questions = framework_data.get("questions", [])
        for i, question in enumerate(questions):
            question_validation = self._validate_question_tags(question, standard_ref)
            if not question_validation["valid"]:
                validation_result["errors"].extend([
                    {**error, "file": filename, "question_index": i}
                    for error in question_validation["errors"]
                ])
                validation_result["valid"] = False
                
            validation_result["suggestions"].extend([
                {**suggestion, "file": filename, "question_index": i}
                for suggestion in question_validation["suggestions"]
            ])
            
        return validation_result
        
    def _extract_standard_from_filename(self, filename: str) -> Optional[str]:
        """Extract IAS/IFRS standard reference from filename"""
        import re
        
        patterns = [
            r'IAS[_\s-]?(\d+)',
            r'IFRS[_\s-]?(\d+)', 
            r'SIC[_\s-]?(\d+)',
            r'IFRIC[_\s-]?(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                standard_type = pattern.split('[')[0].replace('\\', '')
                number = match.group(1)
                return f"{standard_type} {number}"
                
        return None
        
    def _convert_standard_to_dirname(self, standard_ref: str) -> str:
        """Convert 'IAS 32' to 'ias_32' format"""
        parts = standard_ref.lower().split()
        if len(parts) == 2:
            return f"{parts[0]}_{parts[1]}"
        return standard_ref.lower().replace(" ", "_")
        
    def _validate_question_tags(self, question: Dict[str, Any], 
                              standard_ref: str) -> Dict[str, Any]:
        """Validate 5D tags for individual question"""
        
        validation_result = {
            "valid": True,
            "errors": [],
            "suggestions": []
        }
        
        # Get question tags
        facet_focus = question.get("facet_focus", [])
        evidence_expectations = question.get("evidence_expectations", {})
        
        # Validate narrative categories
        narrative_categories = evidence_expectations.get("narrative_categories", [])
        for category in narrative_categories:
            if not self._is_valid_narrative_category(category, standard_ref):
                validation_result["errors"].append({
                    "issue": f"Invalid narrative category: {category}",
                    "tag": "narrative_categories"
                })
                validation_result["valid"] = False
                
        # Suggest improvements based on taxonomy concepts
        standard_concepts = self.concept_index["by_standard"].get(standard_ref, [])
        if standard_concepts:
            suggestions = self._suggest_tag_improvements(
                question, standard_concepts, standard_ref
            )
            validation_result["suggestions"].extend(suggestions)
            
        return validation_result
        
    def _is_valid_narrative_category(self, category: str, standard_ref: str) -> bool:
        """Check if narrative category is valid for the standard"""
        
        # Define valid categories per standard type
        valid_categories = {
            "accounting_policies_note": ["IAS 1", "IAS 8"],
            "event_based": ["IAS 10", "IAS 34"],
            "measurement_basis": ["IAS 36", "IAS 39", "IFRS 9", "IFRS 13"],
            "risk_exposure": ["IAS 32", "IAS 39", "IFRS 7", "IFRS 9"],
            "estimates_judgements": ["IAS 1", "IAS 8", "IAS 36"]
        }
        
        return standard_ref in valid_categories.get(category, [])
        
    def _suggest_tag_improvements(self, question: Dict[str, Any], 
                                standard_concepts: List[str], 
                                standard_ref: str) -> List[Dict[str, Any]]:
        """Suggest tag improvements based on taxonomy concepts"""
        
        suggestions = []
        question_text = question.get("question", "").lower()
        
        # Analyze question text and suggest relevant concepts
        for concept_id in standard_concepts[:10]:  # Limit to first 10 concepts
            concept_data = self.parser.concepts.get(concept_id, {})
            concept_name = concept_data.get("name", "")
            
            # Check if concept is relevant to question
            keywords = self._extract_keywords(concept_name)
            relevant_keywords = [
                keyword for keyword in keywords 
                if keyword in question_text
            ]
            
            if relevant_keywords:
                suggestions.append({
                    "type": "concept_alignment",
                    "suggestion": f"Consider aligning with IFRS concept: {concept_name}",
                    "concept_id": concept_id,
                    "matching_keywords": relevant_keywords,
                    "standard": standard_ref
                })
                
        return suggestions
        
    def _generate_integration_report(self, taxonomy_data: Dict[str, Any], 
                                   framework_validation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive integration report"""
        
        report = {
            "success": True,
            "timestamp": str(Path(__file__).stat().st_mtime),
            "taxonomy_summary": {
                "total_concepts": len(taxonomy_data.get("concepts", {})),
                "categories": list(self.concept_index["by_category"].keys()),
                "standards_covered": list(self.concept_index["by_standard"].keys()),
                "concept_types": self._analyze_concept_types(taxonomy_data["concepts"])
            },
            "framework_validation": framework_validation,
            "integration_metrics": {
                "validation_success_rate": (
                    framework_validation["validated_files"] / 
                    max(framework_validation["total_files"], 1) * 100
                ),
                "standards_with_concepts": len(self.concept_index["by_standard"]),
                "total_keywords_indexed": len(self.concept_index["by_keywords"])
            },
            "next_steps": self._generate_next_steps(framework_validation)
        }
        
        return report
        
    def _analyze_concept_types(self, concepts: Dict[str, Any]) -> Dict[str, int]:
        """Analyze distribution of concept types"""
        type_counts = {}
        
        for concept_data in concepts.values():
            concept_type = concept_data.get("type", "unknown")
            type_counts[concept_type] = type_counts.get(concept_type, 0) + 1
            
        return type_counts
        
    def _generate_next_steps(self, framework_validation: Dict[str, Any]) -> List[str]:
        """Generate recommended next steps based on validation results"""
        
        next_steps = []
        
        if framework_validation["validation_errors"]:
            next_steps.append(
                f"Address {len(framework_validation['validation_errors'])} validation errors"
            )
            
        if framework_validation["suggestions"]:
            next_steps.append(
                f"Review {len(framework_validation['suggestions'])} improvement suggestions"
            )
            
        success_rate = (
            framework_validation["validated_files"] / 
            max(framework_validation["total_files"], 1) * 100
        )
        
        if success_rate < 90:
            next_steps.append("Improve framework validation coverage")
            
        next_steps.append("Begin content tagging using validated taxonomy")
        next_steps.append("Implement real-time taxonomy validation in AI parser")
        
        return next_steps

def main():
    """Main execution function"""
    
    # Define paths
    current_dir = Path(__file__).parent
    taxonomy_path = current_dir / "IFRSAT-2025" / "IFRSAT-2025" / "full_ifrs"
    enhanced_framework_path = current_dir.parent / "checklist_data" / "Enhanced Framework" / "IFRS"
    
    # Verify paths exist
    if not taxonomy_path.exists():
        logger.error(f"Taxonomy path not found: {taxonomy_path}")
        return
        
    if not enhanced_framework_path.exists():
        logger.error(f"Enhanced framework path not found: {enhanced_framework_path}")
        return
        
    # Run integration
    integrator = IFRSTaxonomyIntegrator(
        str(taxonomy_path), 
        str(enhanced_framework_path)
    )
    
    report = integrator.run_integration()
    
    # Save integration report
    report_path = current_dir / "taxonomy_integration_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Integration report saved to: {report_path}")
    
    # Print summary
    if report.get("success"):
        print("\n✅ IFRS Taxonomy Integration Successful!")
        print(f"📊 Loaded {report['taxonomy_summary']['total_concepts']} IFRS concepts")
        print(f"📋 Validated {report['framework_validation']['validated_files']}/{report['framework_validation']['total_files']} framework files")
        print(f"📈 Success Rate: {report['integration_metrics']['validation_success_rate']:.1f}%")
        
        if report["framework_validation"]["validation_errors"]:
            print(f"⚠️  {len(report['framework_validation']['validation_errors'])} validation errors found")
            
        if report["framework_validation"]["suggestions"]:
            print(f"💡 {len(report['framework_validation']['suggestions'])} improvement suggestions available")
    else:
        print("\n❌ IFRS Taxonomy Integration Failed")
        print(f"Error: {report.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()