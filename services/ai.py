import asyncio
import json
import logging
import os
import re
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv  # type: ignore
from openai import AzureOpenAI  # type: ignore

from services.ai_prompts import ai_prompts
from services.compliance_analyzer import ComplianceAnalyzer
from services.document_processor import DocumentProcessor

# from services.progress import progress_service, ProgressStatus # Removed unused import
from services.checklist_utils import load_checklist
from services.vector_store import VectorStore, generate_document_id, get_vector_store
from services.hybrid_financial_detector import detect_financial_statements_hybrid
from services.rate_limiter import (
    check_rate_limit_with_backoff,
    check_duplicate_question,
    clear_processed_questions,
    clear_document_questions,
    get_rate_limit_status,
    get_async_rate_semaphore,
    record_failure,
    reset_circuit_breaker,
    RateLimitError,
    CircuitBreakerOpenError,
)
# Removed deleted modules: enhanced_chunk_selector, intelligent_chunk_accumulator, intelligent_document_analyzer, intelligent_notes_accumulator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] %(message)s",
)
logger = logging.getLogger(__name__)
load_dotenv()

# Create global instance (will be initialized with API key)
vector_store = None
ai_service = None

# Global settings for parallel processing
NUM_WORKERS = min(32, (os.cpu_count() or 1) * 4)  # Optimize worker count
CHUNK_SIZE = 50  # Increased from 10 - process more questions per batch for efficiency

# Azure OpenAI configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "model-router")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"
)
AZURE_OPENAI_EMBEDDING_API_VERSION = os.getenv(
    "AZURE_OPENAI_EMBEDDING_API_VERSION", "2023-05-15"
)

# Rate limiting functionality moved to services/rate_limiter.py


class AIService:
    def __init__(
        self,
        api_key: str,
        azure_endpoint: str,
        deployment_name: str,
        api_version: str,
        embedding_deployment: str,
        embedding_api_version: str,
    ):
        if (
            not api_key
            or not azure_endpoint
            or not deployment_name
            or not embedding_deployment
        ):
            raise ValueError("Azure OpenAI configuration missing (chat or embedding)")
        self.api_key = api_key
        self.azure_endpoint = azure_endpoint
        self.deployment_name = deployment_name  # For chat/completions
        self.api_version = api_version
        self.embedding_deployment = embedding_deployment
        self.embedding_api_version = embedding_api_version
        self.openai_client = AzureOpenAI(
            api_key=api_key, azure_endpoint=azure_endpoint, api_version=api_version
        )
        self.current_document_id = None
        self.progress_tracker = None  # Will be set by analysis routes
        self.executor = ThreadPoolExecutor(max_workers=NUM_WORKERS)
        
        # Initialize compliance analyzer
        self.compliance_analyzer = ComplianceAnalyzer(
            openai_client=self.openai_client,
            deployment_name=deployment_name
        )
        
        # Initialize document processor
        self.document_processor = DocumentProcessor(ai_service=self)
        
        # Storage for structured financial data (extracted once per document)
        self.structured_financial_data = None
        logger.info(f"AIService (Azure) initialized with {NUM_WORKERS} workers")
        global vector_store
        if vector_store is None:
            vector_store = VectorStore(
                api_key, azure_endpoint, embedding_deployment, embedding_api_version
            )
            logger.info("Vector store initialized successfully (Azure)")

    def _get_standard_specific_context(self, question: str, standard_id: Optional[str]) -> str:
        """
        Get context using standard-specific tagged sentences from StandardIdentifier.
        Enhanced with dual file pattern support to fix production file mismatch issue.
        """
        try:
            # Get analysis results containing tagged sentences - check both file patterns
            base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "analysis_results")
            
            # Try main results file first
            main_results_path = os.path.join(base_dir, f"{self.current_document_id}.json")
            parallel_results_path = os.path.join(base_dir, f"{self.current_document_id}_parallel_context.json")
            
            results = None
            results_source = None
            
            if os.path.exists(main_results_path):
                with open(main_results_path, "r", encoding="utf-8") as f:
                    results = json.load(f)
                results_source = "main"
                logger.info(f"✅ Found main results file for document {self.current_document_id}")
                
            elif os.path.exists(parallel_results_path):
                with open(parallel_results_path, "r", encoding="utf-8") as f:
                    results = json.load(f)
                results_source = "parallel"
                logger.info(f"✅ Found parallel context file for document {self.current_document_id}")
                
            else:
                logger.warning(f"No analysis results found for document {self.current_document_id}")
                logger.warning(f"Checked: {main_results_path}")
                logger.warning(f"Checked: {parallel_results_path}")
                return ""
            
            # Get standard identification results - handle both file formats
            identification_result = None
            tagged_sentences = []
            
            if results_source == "main":
                # Main results file format - check for direct structure
                if "tagged_sentences" in results:
                    identification_result = results
                    tagged_sentences = results.get("tagged_sentences", [])
                    logger.info("✅ Using tagged sentences from main results file")
                else:
                    # Check for nested structure in main file
                    parallel_context = results.get("parallel_processing_context", {})
                    standards_result = parallel_context.get("accounting_standards")
                    if standards_result and standards_result.get("details", {}).get("identification_result"):
                        identification_result = standards_result["details"]["identification_result"]
                        tagged_sentences = identification_result.get("tagged_sentences", [])
                        logger.info("✅ Using nested tagged sentences from main results file")
                        
            elif results_source == "parallel":
                # Parallel context file format
                if "accounting_standards" in results:
                    standards_result = results.get("accounting_standards", {})
                    if standards_result.get("details", {}).get("identification_result"):
                        identification_result = standards_result["details"]["identification_result"]
                        tagged_sentences = identification_result.get("tagged_sentences", [])
                        logger.info("✅ Using tagged sentences from parallel context file")
                elif "tagged_sentences" in results:
                    # Direct format in parallel file
                    identification_result = results
                    tagged_sentences = results.get("tagged_sentences", [])
                    logger.info("✅ Using direct tagged sentences from parallel context file")
            
            if not tagged_sentences:
                logger.warning("No tagged sentences found in results")
                logger.warning(f"Results keys: {list(results.keys())}")
                return ""
            
            # 📊 DETAILED CONTEXT LOGGING FOR PRODUCTION
            logger.info("=" * 60)
            logger.info("🔍 CONTEXT RETRIEVAL - FINANCIAL CONTENT ANALYSIS")
            logger.info("=" * 60)
            logger.info(f"📄 DOCUMENT_ID: {self.current_document_id}")
            logger.info(f"📋 QUESTION: {question}")
            logger.info(f"🎯 TARGET_STANDARD: {standard_id}")
            logger.info(f"📊 TOTAL_TAGGED_SENTENCES: {len(tagged_sentences)}")
            
            # Extract standard code from question or use provided standard_id
            target_standards = []
            
            if standard_id:
                target_standards.append(standard_id.upper())
            
            # Also try to detect standard from question text
            question_upper = question.upper()
            standard_patterns = [
                r'IAS\s*(\d+)', r'IFRS\s*(\d+)', r'IPSAS\s*(\d+)', 
                r'ASC\s*(\d+)', r'GAAP', r'REVENUE', r'LEASE'
            ]
            
            for pattern in standard_patterns:
                matches = re.findall(pattern, question_upper)
                for match in matches:
                    if match.isdigit():
                        target_standards.extend([f"IAS {match}", f"IFRS {match}"])
                    else:
                        target_standards.append(match)
            
            # Find relevant sentences for target standards
            relevant_sentences = []
            for sentence_data in tagged_sentences:
                if isinstance(sentence_data, dict):
                    sentence_text = sentence_data.get("text", "").lower()
                    standards = sentence_data.get("standards", [])
                    
                    # Check if this sentence relates to target standards
                    is_relevant = False
                    
                    # Check standards tags
                    for std in standards:
                        std_str = str(std).lower()
                        for target in target_standards:
                            if target.lower() in std_str:
                                is_relevant = True
                                break
                        if is_relevant:
                            break
                    
                    # Check text content for standard mentions
                    for target in target_standards:
                        if target.lower() in sentence_text:
                            is_relevant = True
                            break
                    
                    # Add keyword-based matching for common standards
                    if standard_id:
                        if standard_id.upper() == 'IAS 7':
                            if any(keyword in sentence_text for keyword in ['cash flow', 'indirect method', 'operating activities']):
                                is_relevant = True
                        elif standard_id.upper() == 'IFRS 15':
                            if any(keyword in sentence_text for keyword in ['revenue', 'contract', 'performance obligation']):
                                is_relevant = True
                        elif standard_id.upper() == 'IAS 16':
                            if any(keyword in sentence_text for keyword in ['property', 'plant', 'equipment', 'depreciation']):
                                is_relevant = True
                    
                    if is_relevant:
                        relevant_sentences.append(sentence_data)
            
            logger.info(f"🎯 RELEVANT_SENTENCES_FOUND: {len(relevant_sentences)}")
            
            # Build context from relevant sentences
            if not relevant_sentences:
                logger.warning(f"📊 CONTEXT RETRIEVAL: No sentences found for target standards {target_standards}")
                logger.warning(f"📄 Document: {self.current_document_id}, Question: {question[:100]}...")
                return ""
            
            context_parts = []
            for i, sentence_data in enumerate(relevant_sentences[:15]):  # Limit to prevent token overflow
                if isinstance(sentence_data, dict):
                    text = sentence_data.get("text", "")
                    standards = sentence_data.get("standards", [])
                    if text:
                        context_parts.append(f"[{', '.join(map(str, standards))}] {text}")
                        # Log each sentence being used
                        logger.info(f"📝 SENTENCE_{i+1}: [{', '.join(map(str, standards))}] {text[:200]}{'...' if len(text) > 200 else ''}")
                elif isinstance(sentence_data, str):
                    context_parts.append(sentence_data)
                    logger.info(f"📝 SENTENCE_{i+1}: {sentence_data[:200]}{'...' if len(sentence_data) > 200 else ''}")
            
            final_context = "\n\n".join(context_parts)
            logger.info(f"📊 FINAL_CONTEXT_LENGTH: {len(final_context)} characters")
            logger.info("=" * 60)
            
            return final_context
            
        except Exception as e:
            logger.error(f"Error getting standard-specific context: {str(e)}")
            return ""

    async def process_document(
        self,
        document_id: Optional[str] = None,
        text: Optional[str] = None,
        framework: Optional[str] = None,
        standard: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a document for compliance analysis.
        Delegates to the document processor for modular processing.
        """
        # Set current document ID
        self.current_document_id = document_id
        
        # Extract structured financial data once at the start
        if text and not self.structured_financial_data:
            self.extract_structured_financial_data_once(text)
        
        return await self.document_processor.process_document(
            document_id=document_id,
            text=text,
            framework=framework,
            standard=standard
        )

    # Model determination and section processing methods moved to document processor

    def extract_structured_financial_data_once(self, document_text: str) -> None:
        """
        Extract structured financial data once per document and cache it.
        This should be called at the start of compliance analysis.
        """
        try:
            if self.structured_financial_data is not None:
                logger.info("📊 Structured financial data already extracted for this document")
                return
            
            logger.info("📊 Extracting structured financial data for compliance analysis...")
            
            # Use hybrid financial statement detector to detect and extract structured data
            financial_content = detect_financial_statements_hybrid(
                document_text=document_text, 
                document_id=self.current_document_id
            )
            
            if financial_content and financial_content.statements:
                # Extract structured data using the same format as compliance analyzer
                structured_data = self._extract_structured_financial_data_for_ai(financial_content)
                self.structured_financial_data = structured_data
                
                logger.info(f"✅ Structured financial data extracted: {len(structured_data.get('financial_data', []))} statements")
            else:
                logger.info("📊 No financial statements found for structured extraction")
                self.structured_financial_data = {
                    "financial_data": [], 
                    "extraction_status": "no_financial_statements"
                }
                
        except Exception as e:
            logger.error(f"❌ Error extracting structured financial data: {e}")
            self.structured_financial_data = {
                "financial_data": [], 
                "extraction_status": "extraction_error",
                "error": str(e)
            }

    def analyze_chunk(
        self, chunk: str, question: str, standard_id: Optional[str] = None
    ) -> dict:
        """
        Analyze a chunk of annual report content against a compliance checklist question.
        Delegates to the compliance analyzer for modular processing.
        """
        result = self.compliance_analyzer.analyze_chunk(
            chunk=chunk,
            question=question, 
            standard_id=standard_id,
            current_document_id=self.current_document_id,
            standard_specific_context_func=self._get_standard_specific_context,
            enhanced_evidence=None  # Can be passed through if needed
        )
        
        # Include structured financial data in the result if available
        if self.structured_financial_data:
            result["structured_financial_data"] = self.structured_financial_data
            
        return result

    def _calculate_adequacy(
        self, confidence: float, has_evidence: bool, status: str
    ) -> str:
        """Delegates to compliance analyzer for adequacy calculation."""
        return self.compliance_analyzer.calculate_adequacy(confidence, has_evidence, status)

    def query_ai_with_vector_context(
        self, document_id: Optional[str] = None, question: Optional[dict] = None
    ) -> dict:
        try:
            # Generate a document ID if not provided
            if document_id is None:
                document_id = generate_document_id()

            # Get vector store
            vs_svc = get_vector_store()

            # Validate inputs
            if not document_id or not question:
                return {
                    "status": "Error",
                    "explanation": "Missing document ID or question",
                    "evidence": "",
                    "confidence": 0.0,
                    "adequacy": "Inadequate",
                }

            # Extract question text and reference
            question_text = question.get("question", "")
            reference = question.get("reference", "")
            field_type = question.get("field_type", "metadata_field")

            if not question_text:
                return {
                    "status": "Error",
                    "explanation": "No question provided",
                    "evidence": "",
                    "confidence": 0.0,
                    "adequacy": "Inadequate",
                }

            # For metadata fields, use direct AI analysis without vector search
            if field_type == "metadata_field":
                
                # 📊 METADATA EXTRACTION LOGGING
                logger.info("=" * 60)
                logger.info("📋 METADATA EXTRACTION - PRODUCTION REQUEST")
                logger.info("=" * 60)
                logger.info(f"📄 DOCUMENT_ID: {document_id}")
                logger.info(f"📝 QUESTION: {question_text}")
                logger.info(f"📖 REFERENCE: {reference}")
                logger.info(f"🏷️  FIELD_TYPE: {field_type}")
                
                system_prompt = ai_prompts.get_metadata_extraction_system_prompt()

                # Metadata-specific prompt
                user_prompt = ai_prompts.get_metadata_extraction_user_prompt(
                    reference, question_text
                )
                
                logger.info(f"📤 PROMPT_LENGTH: {len(user_prompt)} characters")

                response = self.openai_client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=300,
                )

                content = response.choices[0].message.content
                
                # 📊 LOG METADATA RESPONSE
                logger.info(f"📥 METADATA_RESPONSE: {content if content else 'NONE'}")
                logger.info("=" * 60)

                # Handle potential None content
                if content is None:
                    logger.error(
                        "OpenAI API returned None content for metadata extraction"
                    )
                    return {
                        "status": "Error",
                        "explanation": "OpenAI API returned empty response",
                        "evidence": "",
                        "adequacy": "low",
                    }

                content = content.strip()

                # Format metadata result
                if content and content.lower() not in [
                    "not found",
                    "n/a",
                    "unknown",
                    "not applicable",
                    "not specified",
                    "not mentioned",
                ]:
                    result = {
                        "status": "COMPLETED",
                        "explanation": content,
                        "evidence": "AI extracted from document",
                        "confidence": 0.9,
                        "adequacy": "Complete",
                    }
                else:
                    result = {
                        "status": "Not found",
                        "explanation": content,
                        "evidence": "",
                        "confidence": 0.0,
                        "adequacy": "Inadequate",
                    }

                return result

            # For other types of questions, use existing vector search logic
            # ... rest of the existing code for non-metadata fields ...

            # Validate inputs
            if not document_id or not question:
                return {
                    "status": "Error",
                    "explanation": "Missing document ID or question",
                    "evidence": "",
                    "confidence": 0.0,
                    "adequacy": "Inadequate",
                }

            # Extract question text and reference
            question_text = question.get("question", "")
            reference = question.get("reference", "")
            field_type = question.get("field_type", "metadata_field")

            if not question_text:
                return {
                    "status": "Error",
                    "explanation": "No question provided",
                    "evidence": "",
                    "confidence": 0.0,
                    "adequacy": "Inadequate",
                }

            # Check if vector index exists
            vector_index_exists = vs_svc.index_exists(document_id)
            if not vector_index_exists:
                logger.warning(
                    f"Vector index not found for document {document_id}. Using direct questioning without vector context."
                )

            logger.info(
                f"Directly querying AI about document {document_id} using vector store as context"
            )

            # Choose system prompt based on question type and vector index status
            if field_type == "metadata_field":
                # Metadata extraction prompt
                system_prompt = ai_prompts.get_vector_metadata_system_prompt(
                    vector_index_exists
                )

                # Metadata-specific output format
                user_prompt = ai_prompts.get_vector_metadata_user_prompt(
                    document_id, question_text, reference, vector_index_exists
                )

                response = self.openai_client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=300,
                )

                # Parse metadata response
                content = response.choices[0].message.content

                # Handle potential None content
                if content is None:
                    logger.error(
                        "OpenAI API returned None content for direct questioning"
                    )
                    return {
                        "status": "Error",
                        "explanation": "OpenAI API returned empty response",
                        "evidence": "",
                        "adequacy": "low",
                    }

                content = content.strip()

                # Format metadata result
                if content and content.lower() not in [
                    "not found",
                    "n/a",
                    "unknown",
                    "not applicable",
                    "not specified",
                    "not mentioned",
                ]:
                    result = {
                        "status": "COMPLETED",
                        "explanation": content,
                        "evidence": "AI extracted from document directly",
                        "confidence": vector_index_exists
                        and 0.9
                        or 0.5,  # Lower confidence if no vector index
                        "adequacy": vector_index_exists and "Complete" or "Partial",
                    }
                else:
                    result = {
                        "status": "Not found",
                        "explanation": "Not found",
                        "evidence": "",
                        "confidence": 0.0,
                        "adequacy": "Inadequate",
                    }

            else:
                # Default checklist analysis prompt for compliance items
                system_prompt = ai_prompts.get_vector_compliance_system_prompt(
                    vector_index_exists
                )

                # Prepare user prompt based on vector index availability
                user_prompt = ai_prompts.get_vector_compliance_user_prompt(
                    document_id, question_text, reference, vector_index_exists
                )

                # Use JSON response format for checklist items
                response = self.openai_client.chat.completions.create(
                    model=self.deployment_name,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=800,
                )

                # Parse JSON response
                content = response.choices[0].message.content
                result = json.loads(content) if content else {}

                # Validate and sanitize result
                if not result:
                    raise ValueError("Empty response from AI")

                # Ensure required fields
                result["status"] = result.get("status", "Not found")
                result["explanation"] = result.get("explanation", "")
                result["evidence"] = result.get("evidence", "")
                result["confidence"] = float(result.get("confidence", 0.0))

                # Adjust confidence if no vector index
                if not vector_index_exists and result["confidence"] > 0.5:
                    result["confidence"] = 0.5  # Cap confidence when no vector index

                # Map to adequacy level
                status = result["status"]
                if status == "Yes":
                    result["adequacy"] = "Complete"
                elif status == "Partially":
                    result["adequacy"] = "Mostly complete"
                else:
                    result["adequacy"] = "Inadequate"

            return result

        except Exception as e:
            logger.error(f"Error in direct AI questioning: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "status": "Error",
                "explanation": f"Analysis error: {str(e)}",
                "evidence": "",
                "confidence": 0.0,
                "adequacy": "Inadequate",
            }

    # Section processing methods moved to document processor

    async def _process_section(
        self,
        section: Dict[str, Any],
        text: str,
        document_id: Optional[str] = None,
        standard_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process a single section of the checklist (async)."""
        try:
            if document_id and self:
                self.current_document_id = document_id
            
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
            
            # Use a batch size for processing
            CHUNK_SIZE = 50  # Process questions in batches
            
            processed_items = []
            for i in range(0, len(items), CHUNK_SIZE):
                batch = items[i : i + CHUNK_SIZE]

                async def process_item_no_limits(item):
                    """Process individual item without rate limiting."""
                    # Mark question as processing in progress tracker
                    if hasattr(self, "progress_tracker") and self.progress_tracker:
                        self.progress_tracker.mark_question_processing(
                            document_id,
                            standard_id or "unknown",
                            item.get("id", "unknown"),
                        )

                    loop = asyncio.get_running_loop()
                    if self:
                        self.current_document_id = document_id
                    
                    return await loop.run_in_executor(
                        None, self.analyze_chunk, text, item["question"], standard_id
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
                        if hasattr(self, "progress_tracker") and self.progress_tracker:
                            self.progress_tracker.mark_question_failed(
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
                    if hasattr(self, "progress_tracker") and self.progress_tracker:
                        self.progress_tracker.mark_question_completed(
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

    async def analyze_compliance(
        self, document_id: str, text: str, framework: str, standard: str
    ) -> Dict[str, Any]:
        """
        Analyze a document for compliance with a specified framework and standard (async).
        """
        logger.info(
            f"Starting compliance analysis for document {document_id} with framework {framework} and standard {standard}"
        )
        try:
            results = await self.process_document(
                document_id=document_id,
                text=text,
                framework=framework,
                standard=standard,
            )
            return {
                "compliance_results": results,
                "document_id": document_id,
                "framework": framework,
                "standard": standard,
                "timestamp": datetime.now().isoformat(),
                "status": results.get("status", "error"),
            }
        except Exception as e:
            logger.error(f"Error in analyze_compliance: {str(e)}", exc_info=True)
            return {
                "document_id": document_id,
                "framework": framework,
                "standard": standard,
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": str(e),
                "error_timestamp": datetime.now().isoformat(),
                "compliance_results": {
                    "sections": [],
                    "status": "error",
                    "error": str(e),
                },
            }


# Global AI service instance
ai_service_instance: Optional[AIService] = None


def get_ai_service() -> AIService:
    """Get the global AI service instance."""
    global ai_service_instance
    if ai_service_instance is None:
        try:
            # This can't be awaited here as we're in a non-async context
            # So we need to create an instance synchronously
            api_key = AZURE_OPENAI_API_KEY
            api_base = AZURE_OPENAI_ENDPOINT
            deployment_id = AZURE_OPENAI_DEPLOYMENT_NAME
            api_version = AZURE_OPENAI_API_VERSION
            embedding_deployment = AZURE_OPENAI_EMBEDDING_DEPLOYMENT
            embedding_api_version = AZURE_OPENAI_EMBEDDING_API_VERSION
            if (
                not api_key
                or not api_base
                or not deployment_id
                or not embedding_deployment
            ):
                raise ValueError(
                    "Azure OpenAI configuration missing in environment variables (chat or embedding)"
                )

            ai_service_instance = AIService(
                api_key,
                api_base,
                deployment_id,
                api_version,
                embedding_deployment,
                embedding_api_version,
            )
            logger.info("AI service initialized synchronously")
        except Exception as e:
            logger.error(f"Error getting AI service: {str(e)}")
            raise RuntimeError("AI service not available")
    return ai_service_instance


# Initialize at module level
try:
    api_key = AZURE_OPENAI_API_KEY
    api_base = AZURE_OPENAI_ENDPOINT
    deployment_id = AZURE_OPENAI_DEPLOYMENT_NAME
    api_version = AZURE_OPENAI_API_VERSION
    embedding_deployment = AZURE_OPENAI_EMBEDDING_DEPLOYMENT
    embedding_api_version = AZURE_OPENAI_EMBEDDING_API_VERSION
    if not api_key or not api_base or not deployment_id or not embedding_deployment:
        raise ValueError(
            "Azure OpenAI configuration missing in environment variables (chat or embedding)"        )
    ai_service_instance = AIService(
        api_key,
        api_base,
        deployment_id,
        api_version,
        embedding_deployment,
        embedding_api_version,
    )
    if ai_service_instance is None:
        raise RuntimeError("AI service initialization failed")
    logger.info("AI service is available and ready")
except Exception as e:
    logger.error(f"Critical error initializing AI service: {str(e)}")
    raise
