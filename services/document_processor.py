"""
Document Processor Service - Extracted from AI.py

Handles document processing, text extraction, and section management
for compliance analysis workflows.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from services.checklist_utils import load_checklist
from services.vector_store import generate_document_id
from services.rate_limiter import clear_processed_questions

logger = logging.getLogger(__name__)

# Global settings for parallel processing
CHUNK_SIZE = 50  # Process questions in batches


class DocumentProcessor:
    """Handles document processing and text extraction operations."""
    
    def __init__(self, ai_service=None):
        """Initialize document processor with reference to AI service."""
        self.ai_service = ai_service
        logger.info("Document processor initialized")
    
    def set_ai_service(self, ai_service):
        """Set the AI service reference (for circular dependency resolution)."""
        self.ai_service = ai_service
    
    async def process_document(
        self,
        document_id: Optional[str] = None,
        text: Optional[str] = None,
        framework: Optional[str] = None,
        standard: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a document for compliance analysis (async version).
        """
        try:
            # Clear processed questions cache for new analysis
            clear_processed_questions()
            logger.info(
                "Starting new document analysis - cleared processed questions cache"
            )

            # Validate required parameters
            if not framework:
                raise ValueError("framework is required")
            if not standard:
                raise ValueError("standard is required")
            if not text:
                raise ValueError("text is required")

            if document_id is None:
                document_id = generate_document_id()
            
            # Set document ID in AI service
            if self.ai_service:
                self.ai_service.current_document_id = document_id
            
            logger.info(
                f"Starting compliance analysis for document {document_id} using framework={framework}, standard={standard}"
            )
            
            checklist = load_checklist(framework, standard)
            if not checklist or not isinstance(checklist, dict):
                raise ValueError("Failed to load checklist or invalid format")
            
            sections = checklist.get("sections")
            if not isinstance(sections, list):
                raise ValueError(
                    "Invalid checklist format: 'sections' key must be a list"
                )
            
            logger.info(
                f"Loaded checklist with {len(sections)} sections for {framework}/{standard}"
            )
            
            results = {
                "document_id": document_id,
                "timestamp": datetime.now().isoformat(),
                "framework": framework,
                "standard": standard,
                "sections": [],
                "status": "processing",
            }
            
            try:
                # Process model choice section first
                model_section = next(
                    (s for s in sections if s.get("section") == "model_choice"), None
                )
                
                if not model_section:
                    logger.info(
                        f"No model choice section found in {framework}/{standard} checklist, processing all sections"
                    )
                    model_used = "unknown"
                else:
                    model_questions = model_section.get("items", [])
                    model_results = []
                    for question in model_questions:
                        # Ensure document ID is set before each call
                        if self.ai_service:
                            self.ai_service.current_document_id = document_id
                        
                        result = self.ai_service.analyze_chunk(
                            text, question["question"], standard
                        )
                        model_results.append(
                            {
                                "id": question["id"],
                                "question": question["question"],
                                "reference": question.get("reference", ""),
                                **result,
                            }
                        )
                    
                    model_used = self._determine_model_from_results(model_results)
                    logger.info(f"Determined model used: {model_used}")
                    
                    results["sections"].append(
                        {
                            "section": "model_choice",
                            "title": model_section.get("title", "Model Choice"),
                            "items": model_results,
                        }
                    )
                
                # Process remaining sections
                section_tasks = []
                for section in sections:
                    if not isinstance(section, dict):
                        logger.warning(f"Skipping invalid section format: {section}")
                        continue
                    
                    section_name = section.get("section")
                    if not section_name or section_name == "model_choice":
                        continue
                    
                    if self._should_process_section(section_name, model_used):
                        section_tasks.append(
                            self._process_section(section, text, document_id, standard)
                        )
                        logger.info(f"Processing section {section_name}")
                    else:
                        logger.info(
                            f"Skipping section {section_name} based on model {model_used}"
                        )
                
                processed_sections = await asyncio.gather(
                    *section_tasks, return_exceptions=True
                )
                
                for section_result in processed_sections:
                    if isinstance(section_result, dict):
                        results["sections"].append(section_result)
                    else:
                        logger.error(
                            f"Section processing failed: {str(section_result)}",
                            exc_info=True,
                        )
                
                results["status"] = "completed"
                results["completed_at"] = datetime.now().isoformat()
                
                logger.info(
                    f"Successfully processed document {document_id} using {framework}/{standard}"
                )
                
                return results
                
            except Exception as e:
                logger.error(
                    f"Error during document processing: {str(e)}", exc_info=True
                )
                results["status"] = "error"
                results["error"] = str(e)
                results["error_timestamp"] = datetime.now().isoformat()
                return results
                
        except Exception as e:
            logger.error(f"Critical error in process_document: {str(e)}", exc_info=True)
            return {
                "document_id": document_id,
                "timestamp": datetime.now().isoformat(),
                "framework": framework,
                "standard": standard,
                "status": "error",
                "error": str(e),
                "error_timestamp": datetime.now().isoformat(),
                "sections": [],
            }

    def _should_process_section(self, section_name: str, model_used: str) -> bool:
        """
        Determine if a section should be processed based on the model used.
        This helps skip sections that are not relevant for certain models.
        """
        # Define model-specific section mappings
        cost_model_sections = {"cost_model_requirements"}
        fair_value_model_sections = {"fair_value_model_requirements"}
        both_model_sections = {"general_requirements", "disclosure_requirements"}

        if model_used == "cost_model":
            return (
                section_name in cost_model_sections
                or section_name in both_model_sections
            )
        elif model_used == "fair_value_model":
            return (
                section_name in fair_value_model_sections
                or section_name in both_model_sections
            )
        else:
            # If model is unknown or mixed, process all sections to be safe
            logger.warning(
                "Could not determine which model is used - defaulting to 'unknown' and processing ALL sections"
            )
            return True

    def _determine_model_from_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Analyze model choice results to determine if cost model or fair value model is used.
        Returns 'cost_model', 'fair_value_model', or 'unknown'.
        """
        cost_model_indicators = 0
        fair_value_model_indicators = 0

        for result in results:
            if result.get("status") == "YES":
                question = result.get("question", "").lower()
                if "cost model" in question:
                    cost_model_indicators += 1
                elif "fair value model" in question or "fair value" in question:
                    fair_value_model_indicators += 1

        if cost_model_indicators > fair_value_model_indicators:
            return "cost_model"
        elif fair_value_model_indicators > cost_model_indicators:
            return "fair_value_model"
        else:
            return "unknown"

    async def _process_section(
        self,
        section: Dict[str, Any],
        text: str,
        document_id: Optional[str] = None,
        standard_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process a single section of the checklist (async)."""
        try:
            if document_id and self.ai_service:
                self.ai_service.current_document_id = document_id
            
            section_name = section.get("section", "unknown")
            original_title = section.get("title", "")
            
            # Compose full title as 'section_name - original_title' if not already present
            if (
                section_name
                and original_title
                and not original_title.startswith(section_name)
            ):
                full_title = f"{section_name} - {original_title}"
            else:
                full_title = original_title or section_name
            
            items = section.get("items", [])
            logger.info(f"Processing section {section_name} with {len(items)} items")
            
            processed_items = []
            for i in range(0, len(items), CHUNK_SIZE):
                batch = items[i : i + CHUNK_SIZE]

                async def process_item_no_limits(item):
                    """Process individual item without rate limiting."""
                    # Mark question as processing in progress tracker
                    if hasattr(self.ai_service, "progress_tracker") and self.ai_service.progress_tracker:
                        self.ai_service.progress_tracker.mark_question_processing(
                            document_id,
                            standard_id or "unknown",
                            item.get("id", "unknown"),
                        )

                    loop = asyncio.get_running_loop()
                    if self.ai_service:
                        self.ai_service.current_document_id = document_id
                    
                    return await loop.run_in_executor(
                        None, self.ai_service.analyze_chunk, text, item["question"], standard_id
                    )

                # Create tasks without rate limiting - PROCESS ALL QUESTIONS
                tasks = [process_item_no_limits(item) for item in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for idx, result in enumerate(batch_results):
                    item = batch[idx]
                    if isinstance(result, Exception):
                        logger.error(
                            f"Error processing item {item.get('id')}: {str(result)}"
                        )
                        # Mark question as failed in progress tracker
                        if hasattr(self.ai_service, "progress_tracker") and self.ai_service.progress_tracker:
                            self.ai_service.progress_tracker.mark_question_failed(
                                document_id,
                                standard_id or "unknown",
                                item.get("id", "unknown"),
                            )
                        continue

                    # Ensure result is a dictionary before unpacking
                    if not isinstance(result, dict):
                        item_id = item.get('id')
                        logger.error(
                            f"Invalid result type for item {item_id}: {type(result)}"
                        )
                        continue

                    # Mark question as completed in progress tracker
                    if hasattr(self.ai_service, "progress_tracker") and self.ai_service.progress_tracker:
                        self.ai_service.progress_tracker.mark_question_completed(
                            document_id,
                            standard_id or "unknown",
                            item.get("id", "unknown"),
                        )

                    processed_items.append(
                        {
                            "id": item["id"],
                            "question": item["question"],
                            "reference": item.get("reference", ""),
                            **result,
                        }
                    )

            return {
                "section": section_name,
                "title": full_title,
                "items": processed_items,
            }
            
        except Exception as e:
            logger.error(
                f"Error processing section {section.get('section', 'unknown')}: {str(e)}"
            )
            return {
                "section": section.get("section", "unknown"),
                "title": section.get("title", "Unknown Section"),
                "items": [],
                "error": str(e),
            }


# Convenience function for backward compatibility
def get_document_processor(ai_service=None):
    """Get a document processor instance."""
    return DocumentProcessor(ai_service=ai_service)