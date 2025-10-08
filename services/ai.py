import asyncio
import json
import logging
import os
import re
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv  # type: ignore
from openai import AzureOpenAI  # type: ignore

from services.ai_prompts import ai_prompts
from services.checklist_utils import load_checklist
from services.intelligent_document_analyzer import enhance_compliance_analysis
from services.vector_store import VectorStore, generate_document_id, get_vector_store
from services.standard_identifier import StandardIdentifier
from services.intelligent_notes_accumulator import IntelligentNotesAccumulator

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
QUESTION_BATCH_SIZE = 10  # Optimized batch size for contextual processing with metadata

# Azure OpenAI configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "o3-mini")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"
)
AZURE_OPENAI_EMBEDDING_API_VERSION = os.getenv(
    "AZURE_OPENAI_EMBEDDING_API_VERSION", "2023-05-15"
)

# Enhanced rate limiting settings for Azure OpenAI S0 tier
REQUESTS_PER_MINUTE = 30  # Conservative limit for S0 tier
TOKENS_PER_MINUTE = 40_000  # Conservative limit for S0 tier
MAX_RETRIES = 3
EXPONENTIAL_BACKOFF_BASE = 2
CIRCUIT_BREAKER_THRESHOLD = (
    10  # Number of consecutive failures before circuit breaker opens
)

_rate_lock = threading.Lock()
_request_count = 0
_token_count = 0
_window_start = time.time()
_consecutive_failures = 0
_circuit_breaker_open = False
_circuit_breaker_opened_at = 0
_processed_questions = {}  # Track processed questions per document to prevent duplicates

# Async rate limiting for Zap Mode
_async_rate_semaphore = None  # Will be initialized when needed


def get_async_rate_semaphore():
    """Get or create the async rate limiting semaphore for Zap Mode"""
    global _async_rate_semaphore
    if _async_rate_semaphore is None:
        # Limit to 10 concurrent API calls to prevent overwhelming Azure OpenAI
        _async_rate_semaphore = asyncio.Semaphore(10)
    return _async_rate_semaphore


class RateLimitError(Exception):
    """Custom exception for rate limiting issues"""

    pass


class CircuitBreakerOpenError(Exception):
    """Custom exception when circuit breaker is open"""

    pass


def reset_circuit_breaker():
    """Reset the circuit breaker after successful operations"""
    global _consecutive_failures, _circuit_breaker_open, _circuit_breaker_opened_at
    _consecutive_failures = 0
    _circuit_breaker_open = False
    _circuit_breaker_opened_at = 0


def check_circuit_breaker():
    """Check if circuit breaker should be opened or if it can be closed"""
    global _consecutive_failures, _circuit_breaker_open, _circuit_breaker_opened_at

    # If circuit breaker is open, check if enough time has passed to try again
    if _circuit_breaker_open:
        time_since_opened = time.time() - _circuit_breaker_opened_at
        if time_since_opened > 300:  # 5 minutes cooldown
            logger.info("Circuit breaker cooldown period expired, attempting to close")
            _circuit_breaker_open = False
            _consecutive_failures = 0
        else:
            remaining = 300 - time_since_opened
            raise CircuitBreakerOpenError(
                f"Circuit breaker is open. Retry in {remaining:.0f} seconds."
            )

    # Check if we should open the circuit breaker
    if _consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD:
        _circuit_breaker_open = True
        _circuit_breaker_opened_at = time.time()
        logger.error(
            f"Circuit breaker opened after {_consecutive_failures} consecutive failures"
        )
        raise CircuitBreakerOpenError(
            "Circuit breaker opened due to consecutive failures"
        )


def record_failure():
    """Record a failure for circuit breaker logic"""
    global _consecutive_failures
    _consecutive_failures += 1
    logger.warning(f"Recorded failure #{_consecutive_failures}")


def clear_processed_questions():
    """Clear the processed questions cache (call when starting new analysis)"""
    _processed_questions.clear()
    logger.info("Cleared processed questions cache")


def clear_document_questions(document_id: str):
    """Clear processed questions for a specific document"""
    if document_id in _processed_questions:
        _processed_questions[document_id].clear()
        logger.info(f"Cleared processed questions for document {document_id}")


def get_rate_limit_status() -> Dict[str, Any]:
    """Get current rate limiting status for monitoring"""
    now = time.time()
    window_elapsed = now - _window_start

    return {
        "requests_used": _request_count,
        "requests_limit": REQUESTS_PER_MINUTE,
        "tokens_used": _token_count,
        "tokens_limit": TOKENS_PER_MINUTE,
        "window_elapsed": window_elapsed,
        "window_remaining": max(0, 60 - window_elapsed),
        "consecutive_failures": _consecutive_failures,
        "circuit_breaker_open": _circuit_breaker_open,
        "processed_questions_count": sum(
            len(doc_questions) for doc_questions in _processed_questions.values()
        ),
    }


def check_duplicate_question(question: str, document_id: str) -> bool:
    """Check if this question has already been processed for this document"""
    # Initialize document-specific set if it doesn't exist
    if document_id not in _processed_questions:
        _processed_questions[document_id] = set()

    question_hash = hash(question)
    if question_hash in _processed_questions[document_id]:
        logger.warning(
            f"Duplicate question detected for {document_id}: {question[:50]}..."
        )
        return True

    _processed_questions[document_id].add(question_hash)
    return False


def check_rate_limit_with_backoff(tokens: int = 0, retry_count: int = 0) -> None:
    """Enhanced rate limiting with exponential backoff and circuit breaker"""
    global _request_count, _token_count, _window_start

    # Check circuit breaker first
    check_circuit_breaker()

    with _rate_lock:
        now = time.time()
        elapsed = now - _window_start

        # Reset window if a minute has passed
        if elapsed >= 60:
            _window_start = now
            _request_count = 0
            _token_count = 0
            elapsed = 0

        # Check if we would exceed limits
        if (
            _request_count + 1 > REQUESTS_PER_MINUTE
            or _token_count + tokens > TOKENS_PER_MINUTE
        ):
            if retry_count >= MAX_RETRIES:
                record_failure()
                raise RateLimitError(
                    f"Max retries ({MAX_RETRIES}) exceeded for rate limiting"
                )

            # Calculate backoff time
            backoff_time = min(EXPONENTIAL_BACKOFF_BASE**retry_count, 60)
            remaining_window = 60 - elapsed
            sleep_time = max(backoff_time, remaining_window)

            logger.warning(
                f"Rate limit would be exceeded. Backing off for "
                f"{sleep_time:.2f} seconds (retry {retry_count + 1}/{MAX_RETRIES})"
            )
            time.sleep(sleep_time)

            # Reset window and try again
            _window_start = time.time()
            _request_count = 0
            _token_count = 0

            # Recursive call with increased retry count
            return check_rate_limit_with_backoff(tokens, retry_count + 1)

        # Update counters
        _request_count += 1
        _token_count += tokens
        logger.debug(
            f"Rate limit check passed. Requests: {_request_count}/"
            f"{REQUESTS_PER_MINUTE}, Tokens: {_token_count}/{TOKENS_PER_MINUTE}"
        )


# Legacy function for backward compatibility
def check_rate_limit(tokens: int = 0) -> None:
    """Legacy function - redirects to enhanced version"""
    check_rate_limit_with_backoff(tokens)


class AIService:
    def resolve_document_id(self, filename_or_company: str) -> str:
        """
        Map filename/company name to actual document_id from analysis_results.
        """
        import glob
        import json
        import os
        
        # Clean input
        clean_input = filename_or_company.lower().replace(".pdf", "")
        
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "analysis_results")
        
        # Search for matching metadata
        for json_path in glob.glob(os.path.join(results_dir, "*.json")):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                company_name = data.get("metadata", {}).get("company_name", "").lower()
                
                # Match company name (e.g., "phoenix" matches "Phoenix Group PLC")
                if clean_input in company_name or any(word in company_name for word in clean_input.split()):
                    return data.get("document_id", filename_or_company)
                    
            except Exception:
                continue
        
        # Fallback to first available document ID
        try:
            for item in os.listdir(results_dir):
                if item.startswith("RAI-"):
                    return item.replace(".json", "")
        except:
            pass
            
        return filename_or_company
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
        logger.info(f"AIService (Azure) initialized with {NUM_WORKERS} workers")
        global vector_store
        if vector_store is None:
            vector_store = VectorStore(
                api_key, azure_endpoint, embedding_deployment, embedding_api_version
            )
            logger.info("Vector store initialized successfully (Azure)")

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
            self.current_document_id = document_id  # Ensure it's set at the start
            logger.info(
                f"Starting compliance analysis for document {document_id} "
                f"using framework={framework}, standard={standard}"
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
                f"Loaded checklist with {len(sections)} sections "
                f"for {framework}/{standard}"
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
                model_section = next(
                    (s for s in sections if s.get("section") == "model_choice"), None
                )
                if not model_section:
                    logger.info(
                        f"No model choice section found in {framework}/{standard} "
                        f"checklist, processing all sections"
                    )
                    model_used = "unknown"
                else:
                    model_questions = model_section.get("items", [])
                    model_results = []
                    for question in model_questions:
                        self.current_document_id = (
                            document_id  # Ensure it's set before each call
                        )
                        result = self.analyze_chunk(
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
                            f"Skipping section {section_name} based on "
                            f"model {model_used}"
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
                    f"Successfully processed document {document_id} using "
                    f"{framework}/{standard}"
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
        """Determine if a section should be processed based on the model used."""
        # Always process all sections for any model beside IAS 40 hardcoded conditions
        if model_used != "fair_value_model" and model_used != "cost_model":
            return True

        # For IAS 40 specific logic:
        if model_used == "fair_value_model":
            # Skip cost model sections for fair value model
            return section_name != "cost_model"
        elif model_used == "cost_model":
            # Skip fair value model sections for cost model
            return section_name != "fair_value_model"
        else:
            # Process all sections for other models
            return True

    def _determine_model_from_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Determine which model is used based on the results of model choice
        questions.
        """
        for result in results:
            if result["status"] != "N/A":
                # Safely handle evidence that could be string or list
                evidence = (
                    " ".join(result["evidence"])
                    if isinstance(result["evidence"], list)
                    else str(result["evidence"])
                ).lower()

                if "fair value" in evidence:
                    return "fair_value"
                elif "cost model" in evidence:
                    return "cost"

        logger.warning(
            "Could not determine which model is used - defaulting to 'unknown' "
            "and processing ALL sections"
        )
        return "unknown"

    # Removed duplicate analyze_contextual_batch method - using the one at line ~527
    
    def analyze_contextual_batch(
        self, 
        questions: List[Dict[str, Any]], 
        company_metadata: Dict[str, Any],
        financial_statements: Dict[str, Any],
        relevant_content: Dict[str, str],
        standard_id: Optional[str] = None
    ) -> List[dict]:
        """
        Analyze a batch of questions with full contextual information:
        1. Company metadata for context
        2. Financial statements overview  
        3. Relevant content chunks by accounting standard
        4. Section and page number information
        
        Processes 10 questions per batch for optimal token usage and context preservation.
        """
        try:
            if not self.current_document_id:
                logger.error("analyze_contextual_batch called with no current_document_id set")
                return [{"status": "Error", "confidence": 0.0, "explanation": "No document ID provided"} for _ in questions]
            
            # Build comprehensive context prompt
            context_prompt = self._build_contextual_prompt(
                company_metadata, financial_statements, relevant_content, standard_id
            )
            
            # Build questions section
            questions_prompt = self._build_questions_prompt(questions)
            
            # Combine for final prompt
            full_prompt = f"{context_prompt}\n\n{questions_prompt}"
            
            # Log token estimate
            estimated_tokens = len(full_prompt) // 4
            logger.info(f"🔢 Processing batch of {len(questions)} questions - Est. tokens: {estimated_tokens:,}")
            
            if estimated_tokens > 25000:  # Conservative limit
                logger.warning(f"⚠️ Large prompt: {estimated_tokens:,} tokens - may hit limits")
            
            # Process through AI
            response = self._process_contextual_batch(full_prompt)
            
            # Parse and return results
            return self._parse_batch_response(response, questions)
            
        except Exception as e:
            logger.error(f"Error in analyze_contextual_batch: {str(e)}")
            return [{"status": "Error", "confidence": 0.0, "explanation": str(e)} for _ in questions]
    
    def analyze_chunk(
        self, chunk: str, question: str, standard_id: Optional[str] = None
    ) -> dict:
        """
        Analyze a chunk of annual report content against a compliance
        checklist question.
        Enhanced with intelligent document analysis, rate limiting, and
        duplicate prevention.
        Returns a JSON response with:
          - status: "YES", "NO", or "N/A"
          - confidence: float between 0 and 1
          - explanation: str explaining the analysis
          - evidence: str containing relevant text from the document
          - suggestion: str containing suggestion when status is "NO"
        """
        try:
            # Ensure document_id is set
            if not self.current_document_id:
                logger.error(
                    "analyze_chunk called with no current_document_id set. Attempting to resolve dynamically."
                )
                # Try to resolve using filename or company name
                self.current_document_id = self.resolve_document_id(chunk if chunk else "")
                if not self.current_document_id:
                    return {
                        "status": "Error",
                        "confidence": 0.0,
                        "explanation": "No document ID provided for vector search.",
                        "evidence": "",
                        "suggestion": (
                            "Check backend logic to ensure document_id is always set."
                        ),
                    }

            # Check for duplicate questions
            if check_duplicate_question(question, self.current_document_id):
                logger.warning(
                    f"Skipping duplicate question for document "
                    f"{self.current_document_id}"
                )
                return {
                    "status": "N/A",
                    "confidence": 0.0,
                    "explanation": (
                        "This question has already been processed for this document."
                    ),
                    "evidence": "Duplicate question detected.",
                    "suggestion": "Question already analyzed - check previous results.",
                }

            # Check circuit breaker before proceeding
            try:
                check_circuit_breaker()
            except CircuitBreakerOpenError as e:
                logger.error(f"Circuit breaker is open: {str(e)}")
                return {
                    "status": "N/A",
                    "confidence": 0.0,
                    "explanation": f"Analysis temporarily unavailable: {str(e)}",
                    "evidence": "Circuit breaker open due to repeated failures.",
                    "suggestion": "Please wait and retry the analysis later.",
                }

            # Get relevant chunks using vector search (existing method)
            vs_svc = get_vector_store()
            if not vs_svc:
                raise ValueError("Vector store service not initialized")

            # Search for relevant chunks using the question
            relevant_chunks = vs_svc.search(
                query=question, document_id=self.current_document_id, top_k=3
            )

            # Enhanced: Use intelligent document analyzer if we have full document
            # text and standard ID
            enhanced_evidence = None
            if (
                chunk and len(chunk) > 1000 and standard_id
            ):  # Check if we have substantial document content
                try:
                    logger.info(
                        f"Using intelligent document analyzer for {standard_id}"
                    )
                    enhanced_analysis = enhance_compliance_analysis(
                        compliance_question=question,
                        document_text=chunk,  # This should be the full document text
                        standard_id=standard_id,
                        existing_chunks=(
                            [chunk_data["text"] for chunk_data in relevant_chunks]
                            if relevant_chunks
                            else []
                        ),
                    )
                    enhanced_evidence = enhanced_analysis
                    quality_score = enhanced_analysis.get(
                        'evidence_quality_assessment', {}).get('overall_quality', 0)
                    logger.info(
                        f"Intelligent analysis completed with quality score: "
                        f"{quality_score}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Intelligent document analysis failed, falling back to "
                        f"standard method: {str(e)}"
                    )
                    enhanced_evidence = None

            if not relevant_chunks and not enhanced_evidence:
                logger.warning(f"No relevant chunks found for question: {question}")
                return {
                    "status": "N/A",
                    "confidence": 0.0,
                    "explanation": "No relevant content found in the document",
                    "evidence": "",
                    "suggestion": (
                        "Add a clear statement in the financial statement "
                        "disclosures addressing this requirement."
                    ),
                }

            # ENHANCED: Check if this is a cash flow or financial statement question
            cash_flow_keywords = [
                "cash flow", "statement of cash", "operating activities", "investing activities", 
                "financing activities", "cash flows during the period", "classified by operating"
            ]
            is_cash_flow_question = any(keyword.lower() in question.lower() for keyword in cash_flow_keywords)
            
            # Use enhanced evidence if available, otherwise fall back to original chunks
            if enhanced_evidence and enhanced_evidence.get("primary_evidence"):
                context = enhanced_evidence["primary_evidence"]
                evidence_quality = enhanced_evidence.get(
                    "evidence_quality_assessment", {}
                )
                evidence_source = evidence_quality.get('evidence_source', 'Unknown')
                logger.info(
                    f"Using enhanced evidence from: {evidence_source}"
                )
            else:
                # For cash flow questions, try to include detected financial statements
                if is_cash_flow_question:
                    try:
                        # Get financial statement content from the document analysis
                        results_path = os.path.join(
                            os.path.dirname(os.path.dirname(__file__)), 
                            "analysis_results", 
                            f"{self.current_document_id}.json"
                        )
                        if os.path.exists(results_path):
                            with open(results_path, "r", encoding="utf-8") as f:
                                results = json.load(f)
                            
                            # Check for detected financial statements
                            parallel_context = results.get("parallel_processing_context", {})
                            financial_statements = parallel_context.get("financial_statements")
                            
                            if financial_statements and financial_statements.get("financial_statements"):
                                # Look for cash flow statement content
                                cash_flow_content = ""
                                for statement in financial_statements["financial_statements"]:
                                    if "cash" in statement.get("statement_type", "").lower():
                                        cash_flow_content += f"\n=== {statement.get('statement_type', 'Cash Flow Statement').upper()} ===\n"
                                        cash_flow_content += statement.get("content", "")[:2000]  # Limit size
                                        break
                                
                                if cash_flow_content:
                                    # Combine cash flow content with vector search results
                                    vector_context = "\n\n".join([chunk_data["text"] for chunk_data in relevant_chunks])
                                    context = cash_flow_content + "\n\n=== ADDITIONAL CONTEXT ===\n" + vector_context
                                    logger.info(f"✅ Enhanced cash flow question with detected cash flow statement content")
                                else:
                                    context = "\n\n".join([chunk_data["text"] for chunk_data in relevant_chunks])
                                    logger.warning(f"⚠️ No cash flow statement content found for cash flow question")
                            else:
                                context = "\n\n".join([chunk_data["text"] for chunk_data in relevant_chunks])
                                logger.warning(f"⚠️ No financial statements available for cash flow question: {question[:50]}...")
                        else:
                            context = "\n\n".join([chunk_data["text"] for chunk_data in relevant_chunks])
                            logger.warning(f"⚠️ No analysis results available for enhanced cash flow context")
                    except Exception as e:
                        context = "\n\n".join([chunk_data["text"] for chunk_data in relevant_chunks])
                        logger.error(f"❌ Error enhancing cash flow context: {str(e)}")
                else:
                    context = "\n\n".join([chunk_data["text"] for chunk_data in relevant_chunks])
                    logger.info("Using standard vector search evidence")

            # Construct the prompt for the AI using the prompts library
            prompt = ai_prompts.get_full_compliance_analysis_prompt(
                question=question, context=context, enhanced_evidence=enhanced_evidence
            )

            # Enhanced rate limiting with retries and circuit breaker
            max_api_retries = 3
            for api_retry in range(max_api_retries):
                try:
                    # RATE LIMITING REMOVED - Process all 217 questions without limits
                    # check_rate_limit_with_backoff(tokens=estimated_tokens)

                    # Get AI response
                    response = self.openai_client.chat.completions.create(
                        model=self.deployment_name,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    ai_prompts.get_compliance_analysis_system_prompt()
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                    )

                    # If we get here, the API call was successful
                    reset_circuit_breaker()  # Reset circuit breaker on success
                    content = response.choices[0].message.content
                    break

                except Exception as api_error:
                    error_message = str(api_error).lower()

                    if "429" in error_message or "rate limit" in error_message:
                        record_failure()
                        if api_retry < max_api_retries - 1:
                            backoff_time = EXPONENTIAL_BACKOFF_BASE ** (api_retry + 1)
                            logger.warning(
                                f"Rate limit hit, retrying in {backoff_time} seconds "
                                f"(attempt {api_retry + 1}/{max_api_retries})"
                            )
                            time.sleep(backoff_time)
                            continue
                        else:
                            logger.error(
                                "Max API retries exceeded due to rate limiting"
                            )
                            return {
                                "status": "N/A",
                                "confidence": 0.0,
                                "explanation": (
                                    "Analysis could not be completed due to "
                                    "persistent API rate limits."
                                ),
                                "evidence": "Multiple rate limit errors occurred.",
                                "suggestion": (
                                    "Please retry the analysis later when API rate "
                                    "limits reset."
                                ),
                            }

                    elif "timeout" in error_message or "connection" in error_message:
                        record_failure()
                        if api_retry < max_api_retries - 1:
                            backoff_time = EXPONENTIAL_BACKOFF_BASE**api_retry
                            logger.warning(
                                f"Connection error, retrying in {backoff_time} seconds "
                                f"(attempt {api_retry + 1}/{max_api_retries})"
                            )
                            time.sleep(backoff_time)
                            continue
                        else:
                            logger.error(
                                "Max API retries exceeded due to connection issues"
                            )
                            return {
                                "status": "N/A",
                                "confidence": 0.0,
                                "explanation": (
                                    f"Analysis failed due to connection issues: "
                                    f"{str(api_error)}"
                                ),
                                "evidence": "Connection error occurred.",
                                "suggestion": (
                                    "Please check network connectivity and retry the "
                                    "analysis."
                                ),
                            }

                    else:
                        # Other API error - don't retry
                        record_failure()
                        logger.error(f"Non-retryable API error: {str(api_error)}")
                        return {
                            "status": "N/A",
                            "confidence": 0.0,
                            "explanation": (
                                f"Analysis failed due to API error: {str(api_error)}"
                            ),
                            "evidence": "API error occurred.",
                            "suggestion": (
                                "Please retry the analysis or check system logs for "
                                "details."
                            ),
                        }
            else:
                # This should not happen due to the break statement, but just in case
                logger.error("Unexpected exit from retry loop")
                return {
                    "status": "N/A",
                    "confidence": 0.0,
                    "explanation": "Unexpected error during API retry loop",
                    "evidence": "",
                    "suggestion": "Please retry the analysis.",
                }

            # Console log the API response for each question
            logger.info("=" * 80)
            logger.info("📋 CHECKLIST QUESTION API RESPONSE")
            logger.info("=" * 80)
            logger.info(
                f"🔍 Question: {question[:100]}{'...' if len(question) > 100 else ''}"
            )
            logger.info(
                f"📄 Standard ID: {standard_id or 'Not specified'} (from analyze_chunk)"
            )
            logger.info(f"🎯 Document ID: {self.current_document_id}")
            logger.info("-" * 40)
            logger.info("📤 RAW API RESPONSE:")
            logger.info(content)
            logger.info("-" * 40)

            # Handle potential None content
            if content is None:
                logger.error("❌ OpenAI API returned None content")
                logger.info("=" * 80)
                return {
                    "status": "Error",
                    "confidence": 0.0,
                    "explanation": "OpenAI API returned empty response",
                    "evidence": "",
                    "suggestion": "Please retry the analysis.",
                }

            # Try to parse JSON from the response
            try:
                # First, try to parse as JSON directly
                result = json.loads(content)
            except json.JSONDecodeError:
                # If that fails, try to extract JSON from the text
                logger.warning(
                    f"Response is not valid JSON, attempting to extract "
                    f"structured data: {content[:100]}..."
                )

                # Extract using regex-like approach for key fields
                result = {}

                # Extract status
                if "YES" in content or "Yes" in content:
                    result["status"] = "YES"
                elif "NO" in content or "No" in content:
                    result["status"] = "NO"
                else:
                    result["status"] = "N/A"

                # Extract confidence (look for number between 0 and 1)
                confidence_match = re.search(r"[Cc]onfidence:?\s*(\d+\.\d+)", content)
                if confidence_match:
                    result["confidence"] = float(confidence_match.group(1))
                else:
                    result["confidence"] = 0.5

                # Extract explanation
                explanation_match = re.search(
                    r"[Ee]xplanation:?\s*(.+?)(?:\n\n|\n[A-Z]|$)", content, re.DOTALL
                )
                if explanation_match:
                    result["explanation"] = explanation_match.group(1).strip()
                else:
                    result["explanation"] = "No detailed explanation provided."

                # Extract evidence
                if "|" in content:
                    evidence_lines = re.findall(
                        r"IAS \d+\.\d+\(?\w?\)?\s*\|.+", content
                    )
                    if evidence_lines:
                        result["evidence"] = evidence_lines
                    else:
                        result["evidence"] = ["No structured evidence provided."]
                else:
                    result["evidence"] = [
                        "Evidence not provided in the required format."
                    ]

                # Extract suggestion if status is NO
                if result["status"] == "NO":
                    suggestion_match = re.search(
                        r"[Ss]uggestion:?:?\s*(.+?)(?:\n\n|\n[A-Z]|$)",
                        content,
                        re.DOTALL,
                    )
                    if suggestion_match:
                        result["suggestion"] = suggestion_match.group(1).strip()
                    else:
                        result["suggestion"] = (
                            "Provide a concrete, practical sample disclosure that "
                            "would satisfy this IAS 40 requirement. For example: "
                            "'The entity should disclose [required information] in "
                            "accordance with IAS 40.[paragraph]'."
                        )

                # Extract content_analysis
                content_analysis_match = re.search(
                    r"[Cc]ontent_analysis:?\s*[\"']?(.+?)[\"']?(?:\n\n|\n[A-Z]|,\s*\"|\",|$)",
                    content,
                    re.DOTALL,
                )
                if content_analysis_match:
                    result["content_analysis"] = content_analysis_match.group(1).strip()
                else:
                    result["content_analysis"] = (
                        "No detailed content analysis provided."
                    )

                # Extract disclosure_recommendations
                disclosure_recommendations = []
                # Look for array format ["recommendation1", "recommendation2"]
                disclosure_array_match = re.search(
                    r"[Dd]isclosure_recommendations:?\s*\[(.*?)\]",
                    content,
                    re.DOTALL,
                )
                if disclosure_array_match:
                    # Extract individual recommendations from the array
                    array_content = disclosure_array_match.group(1)
                    recommendations = re.findall(r'["\']([^"\']+)["\']', array_content)
                    disclosure_recommendations.extend(recommendations)
                
                if not disclosure_recommendations:
                    # Look for single recommendation format
                    single_disclosure_match = re.search(
                        r"[Dd]isclosure_recommendations?:?\s*[\"']?(.+?)[\"']?(?:\n\n|\n[A-Z]|,\s*\"|\",|$)",
                        content,
                        re.DOTALL,
                    )
                    if single_disclosure_match:
                        disclosure_recommendations.append(
                            single_disclosure_match.group(1).strip()
                        )
                
                default_rec = (
                    "Consider enhancing the disclosure to provide more "
                    "comprehensive information addressing this requirement."
                )
                result["disclosure_recommendations"] = (
                    disclosure_recommendations if disclosure_recommendations 
                    else [default_rec]
                )

            # Validate and clean the response
            if "status" not in result or result["status"] not in ["YES", "NO", "N/A"]:
                result["status"] = "N/A"
            if "confidence" not in result:
                result["confidence"] = 0.5
            if "explanation" not in result:
                result["explanation"] = "No explanation provided"
            if "evidence" not in result:
                result["evidence"] = ""
            
            # Validate new enhanced fields
            if "content_analysis" not in result:
                result["content_analysis"] = "No detailed content analysis provided."
            if "disclosure_recommendations" not in result:
                result["disclosure_recommendations"] = [
                    "Consider enhancing the disclosure to provide more "
                    "comprehensive information addressing this requirement."
                ]
            elif not isinstance(result["disclosure_recommendations"], list):
                # Convert single string to list if needed
                result["disclosure_recommendations"] = [
                    str(result["disclosure_recommendations"])
                ]

            # Ensure evidence is structured and meaningful
            if result["evidence"] == "" or all(
                e == "N/A | N/A | N/A | N/A" for e in result["evidence"]
            ):
                result["evidence"] = ["No relevant evidence found in the document."]

            # Add a suggestion when status is "NO" and no suggestion provided
            if result["status"] == "NO" and "suggestion" not in result:
                result["suggestion"] = (
                    "Provide a concrete, practical sample disclosure that would "
                    "satisfy this IAS 40 requirement. For example: 'The entity "
                    "should disclose [required information] in accordance with "
                    "IAS 40.[paragraph]'."
                )

            # Add enhanced evidence metadata if available
            if enhanced_evidence:
                result["enhanced_analysis"] = {
                    "evidence_quality_score": enhanced_evidence.get(
                        "evidence_quality_assessment", {}
                    ).get("overall_quality", 0),
                    "confidence_level": enhanced_evidence.get(
                        "evidence_quality_assessment", {}
                    ).get("confidence_level", 0.0),
                    "source_type": enhanced_evidence.get(
                        "evidence_quality_assessment", {}
                    ).get("source_type", "unknown"),
                    "is_policy_based": enhanced_evidence.get(
                        "evidence_quality_assessment", {}
                    ).get("is_policy_based", True),
                    "evidence_source": enhanced_evidence.get(
                        "evidence_quality_assessment", {}
                    ).get("evidence_source", "Unknown"),
                    "recommendation": enhanced_evidence.get("analysis_summary", {}).get(
                        "recommendation", "Manual review recommended"
                    ),
                }

            # Add document segment information from vector search results (no page numbers since we're not chunking)
            if relevant_chunks:
                document_extracts = []
                
                for segment_item in relevant_chunks:
                    # Extract segment attributes
                    segment_text = segment_item.get("text")
                    segment_index = segment_item.get("segment_index", 0)
                    relevance_score = segment_item.get("score", 0.0)
                    
                    # Include document extracts without page references
                    if segment_text:
                        extract_text = (
                            segment_text[:500] + "..." if len(segment_text) > 500 
                            else segment_text
                        )
                        document_extracts.append({
                            "text": extract_text,
                            "segment_index": segment_index,
                            "relevance_score": relevance_score
                        })
                
                if document_extracts:
                    result["document_extracts"] = document_extracts

            # Log the processed result
            logger.info("✅ PROCESSED RESULT:")
            logger.info(f"   Status: {result.get('status', 'N/A')}")
            logger.info(f"   Confidence: {result.get('confidence', 0.0):.2f}")
            explanation = result.get('explanation', 'N/A')
            truncated_explanation = (
                explanation[:150] + ('...' if len(str(explanation)) > 150 else '')
            )
            logger.info(f"   Explanation: {truncated_explanation}")
            evidence = result.get('evidence', [])
            evidence_count = len(evidence) if isinstance(evidence, list) else 1
            logger.info(
                f"   Evidence Count: {evidence_count}"
            )
            if result.get("content_analysis"):
                content_analysis = result.get('content_analysis', 'N/A')[:100]
                content_len = len(str(result.get('content_analysis', '')))
                content_suffix = '...' if content_len > 100 else ''
                logger.info(f"   Content Analysis: {content_analysis}{content_suffix}")
            if result.get("disclosure_recommendations"):
                rec_count = len(result.get('disclosure_recommendations', []))
                logger.info(f"   Disclosure Recommendations: {rec_count} suggestions")
            if result.get("document_extracts"):
                extract_count = len(result.get('document_extracts', []))
                extracts = result.get('document_extracts', [])
                total_score = sum(e.get('relevance_score', 0) for e in extracts)
                avg_score = total_score / extract_count if extract_count > 0 else 0
                segments_info = [f"seg{e.get('segment_index', 0)}" for e in extracts]
                logger.info(
                    f"   Document Extracts: {extract_count} segments ({', '.join(segments_info)}), "
                    f"avg relevance: {avg_score:.3f}"
                )
            if result.get("status") == "NO" and result.get("suggestion"):
                suggestion = result.get('suggestion', 'N/A')
                truncated_suggestion = (
                    suggestion[:100] + ('...' if len(str(suggestion)) > 100 else '')
                )
                logger.info(f"   Suggestion: {truncated_suggestion}")
            if enhanced_evidence:
                enhanced = result.get('enhanced_analysis', {})
                quality_score = enhanced.get('evidence_quality_score', 0)
                logger.info(
                    f"   Enhanced Analysis: Quality Score {quality_score}/100"
                )
            logger.info("=" * 80)

            return result

        except Exception as e:
            logger.error(f"Error in analyze_chunk: {str(e)}")
            return {
                "status": "N/A",
                "confidence": 0.0,
                "explanation": f"Error during analysis: {str(e)}",
                "evidence": "",
                "suggestion": (
                    "Consider adding explicit disclosure addressing this "
                    "requirement."
                ),
            }

    def _build_contextual_prompt(
        self, 
        company_metadata: Dict[str, Any],
        financial_statements: Dict[str, Any], 
        relevant_content: Dict[str, str],
        standard_id: Optional[str] = None
    ) -> str:
        """Build comprehensive contextual prompt with company metadata and financial statements"""
        
        prompt_parts = [
            "=== DOCUMENT ANALYSIS CONTEXT ===",
            "",
            "📋 COMPANY INFORMATION:"
        ]
        
        # Add company metadata
        if company_metadata:
            for key, value in company_metadata.items():
                if value and str(value).strip():
                    prompt_parts.append(f"• {key.replace('_', ' ').title()}: {value}")
        
        prompt_parts.extend([
            "",
            "📊 FINANCIAL STATEMENTS OVERVIEW:"
        ])
        
        # Add financial statements ACTUAL CONTENT - not just metadata
        if financial_statements and hasattr(financial_statements, 'statements'):
            # financial_statements is a FinancialContent object with .statements list
            statements_list = getattr(financial_statements, 'statements', [])
            prompt_parts.append(f"Found {len(statements_list)} financial statements:")
            
            for statement in statements_list:
                prompt_parts.extend([
                    "",
                    f"=== {statement.statement_type.upper()} ===",
                    f"Confidence: {statement.confidence_score:.1f}",
                    f"Pages: {', '.join(map(str, statement.page_numbers))}",
                    "",
                    # Include the ACTUAL CONTENT of the financial statement
                    statement.content[:8000] + ("..." if len(statement.content) > 8000 else ""),
                    ""
                ])
        elif financial_statements:
            # Fallback for old format
            for statement_type, details in financial_statements.items():
                if isinstance(details, dict) and details.get('present', False):
                    prompt_parts.append(f"• {statement_type}: Present ({details.get('confidence', 'N/A')} confidence)")
                elif isinstance(details, str):
                    prompt_parts.append(f"• {statement_type}: {details}")
        
        # Add relevant content by standard
        if standard_id:
            prompt_parts.extend([
                "",
                f"🎯 FOCUSED CONTENT - {standard_id}:"
            ])
            
            standard_content = relevant_content.get(standard_id, "")
            if standard_content:
                # Truncate if too long but preserve structure
                if len(standard_content) > 8000:
                    standard_content = standard_content[:8000] + "... [content truncated]"
                prompt_parts.append(standard_content)
            else:
                prompt_parts.append(f"No specific content found for {standard_id}")
        else:
            prompt_parts.extend([
                "",
                "📄 RELEVANT CONTENT SECTIONS:"
            ])
            
            # Add top 3 most relevant standards
            sorted_content = sorted(relevant_content.items(), key=lambda x: len(x[1]), reverse=True)
            for i, (std, content) in enumerate(sorted_content[:3]):
                if content:
                    preview = content[:1000] + "..." if len(content) > 1000 else content
                    prompt_parts.extend([
                        "",
                        f"--- {std} ---",
                        preview
                    ])
        
        return "\n".join(prompt_parts)
    
    def _build_questions_prompt(self, questions: List[Dict[str, Any]]) -> str:
        """Build questions section of the prompt"""
        
        prompt_parts = [
            "=== COMPLIANCE QUESTIONS ===",
            "",
            "Please analyze the following questions against the provided context.",
            "IMPORTANT: Adequacy is subjective - even partial disclosure should often be considered adequate if it provides meaningful information.",
            "Be lenient with YES responses - if there is some relevant disclosure, consider it compliant rather than demanding perfect completeness.",
            "For each question, provide:",
            "- status: YES/NO/N/A", 
            "- confidence: 0.0-1.0",
            "- explanation: Clear reasoning (favor YES for partial but meaningful disclosures)",
            "- evidence: Specific text from the document",
            "- suggestion: Improvement advice if status is NO",
            ""
        ]
        
        for i, question in enumerate(questions, 1):
            question_text = question.get('question', 'Unknown question')
            question_id = question.get('id', f'Q{i}')
            
            prompt_parts.extend([
                f"QUESTION {i} (ID: {question_id}):",
                question_text,
                ""
            ])
        
        prompt_parts.append("Please respond with a JSON array containing analysis for all questions.")
        return "\n".join(prompt_parts)
    
    def _process_contextual_batch(self, prompt: str) -> str:
        """Process the contextual batch through AI service"""
        try:
            if not AZURE_OPENAI_ENDPOINT:
                raise ValueError("AZURE_OPENAI_ENDPOINT is not configured")
            
            client = AzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
            )
            
            response = client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": "You are an expert financial analyst specializing in IFRS/IAS compliance."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=4000
            )
            
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("AI response content is None")
            return content
            
        except Exception as e:
            logger.error(f"Error processing contextual batch: {str(e)}")
            raise e
    
    def _parse_batch_response(self, response: str, questions: List[Dict[str, Any]]) -> List[dict]:
        """Parse AI response into structured results"""
        try:
            # Try to parse as JSON
            if response.strip().startswith('['):
                return json.loads(response)
            
            # Fallback: create basic responses
            results = []
            for question in questions:
                results.append({
                    "status": "N/A",
                    "confidence": 0.5,
                    "explanation": "Response parsing failed - using fallback",
                    "evidence": response[:200] + "..." if len(response) > 200 else response,
                    "suggestion": "Manual review recommended"
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error parsing batch response: {str(e)}")
            # Return error responses
            return [{
                "status": "Error",
                "confidence": 0.0,
                "explanation": f"Failed to parse response: {str(e)}",
                "evidence": "",
                "suggestion": "Please try again"
            } for _ in questions]

    async def process_document_with_enhanced_identification(
        self,
        document_id: Optional[str] = None,
        text: Optional[str] = None,
        framework: Optional[str] = None,
        standard: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Enhanced document processing that integrates with the StandardIdentifier system
        to provide company metadata, financial statements overview, and contextual content
        organized by accounting standards.
        
        Processes questions in optimized 10-question batches with full context.
        """
        try:
            # Initialize
            if document_id is None:
                document_id = generate_document_id()
            self.current_document_id = document_id
            
            logger.info(f"🚀 Starting enhanced analysis for document {document_id}")
            logger.info(f"   Framework: {framework}, Standard: {standard}")
            
            # Step 1: Run enhanced standard identification
            logger.info("📊 Running enhanced standard identification...")
            identifier = StandardIdentifier()
            accumulator = IntelligentNotesAccumulator()
            
            # Process document through standard identification
            identification_results = identifier.identify_standards_in_notes(text or "", document_id)
            
            if not identification_results or 'identified_standards' not in identification_results:
                logger.warning("⚠️ Standard identification failed - proceeding with basic analysis")
                fallback_text = (text or "")[:10000] if text else ""
                identification_results = {
                    'identified_standards': {'General': fallback_text},  # Fallback
                    'company_metadata': {},
                    'financial_statements': {}
                }
            
            # Extract components
            identified_standards = identification_results['identified_standards']
            company_metadata = identification_results.get('company_metadata', {})
            financial_statements = identification_results.get('financial_statements', {})
            
            logger.info(f"✅ Identified {len(identified_standards)} accounting standards")
            logger.info(f"   Standards: {list(identified_standards.keys())}")
            
            # Step 2: Load compliance checklist
            checklist = load_checklist(framework, standard)
            if not checklist:
                raise ValueError(f"Failed to load checklist for {framework}/{standard}")
            
            all_questions = []
            for section in checklist:
                for item in section.get("items", []):
                    all_questions.append({
                        'id': item.get('id', 'unknown'),
                        'question': item.get('question', ''),
                        'section': section.get('section', 'unknown'),
                        'standard_id': item.get('standard_id', standard)
                    })
            
            logger.info(f"📋 Loaded {len(all_questions)} compliance questions")
            
            # Step 3: Process questions in 10-question batches with contextual data
            logger.info("🔄 Processing questions in optimized 10-question batches...")
            
            all_results = []
            total_batches = (len(all_questions) + QUESTION_BATCH_SIZE - 1) // QUESTION_BATCH_SIZE
            
            for batch_idx in range(0, len(all_questions), QUESTION_BATCH_SIZE):
                batch_num = (batch_idx // QUESTION_BATCH_SIZE) + 1
                batch_questions = all_questions[batch_idx:batch_idx + QUESTION_BATCH_SIZE]
                
                logger.info(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch_questions)} questions)")
                
                # Determine most relevant standard for this batch
                batch_standards = set(q.get('standard_id', standard) for q in batch_questions)
                primary_standard = max(batch_standards, key=lambda s: sum(1 for q in batch_questions if q.get('standard_id') == s))
                
                # Process batch with full context
                try:
                    batch_results = self.analyze_contextual_batch(
                        questions=batch_questions,
                        company_metadata=company_metadata,
                        financial_statements=financial_statements,
                        relevant_content=identified_standards,
                        standard_id=primary_standard
                    )
                    
                    # Add batch metadata to each result
                    for i, result in enumerate(batch_results):
                        if isinstance(result, dict):
                            result['question_id'] = batch_questions[i].get('id', f'Q{batch_idx + i + 1}')
                            result['section'] = batch_questions[i].get('section', 'unknown')
                            result['batch_number'] = batch_num
                            result['standard_id'] = batch_questions[i].get('standard_id', standard)
                    
                    all_results.extend(batch_results)
                    logger.info(f"✅ Batch {batch_num} completed successfully")
                    
                except Exception as e:
                    logger.error(f"❌ Error processing batch {batch_num}: {str(e)}")
                    # Add error results for this batch
                    error_results = [{
                        "status": "Error",
                        "confidence": 0.0,
                        "explanation": f"Batch processing error: {str(e)}",
                        "evidence": "",
                        "suggestion": "Manual review required",
                        "question_id": q.get('id', f'Q{batch_idx + i + 1}'),
                        "section": q.get('section', 'unknown'),
                        "batch_number": batch_num,
                        "standard_id": q.get('standard_id', standard)
                    } for i, q in enumerate(batch_questions)]
                    all_results.extend(error_results)
            
            # Step 4: Organize results by section
            logger.info("📊 Organizing results by section...")
            
            results_by_section = {}
            for result in all_results:
                section_name = result.get('section', 'unknown')
                if section_name not in results_by_section:
                    results_by_section[section_name] = []
                results_by_section[section_name].append(result)
            
            # Step 5: Calculate summary statistics
            total_questions = len(all_results)
            yes_count = sum(1 for r in all_results if r.get('status') == 'YES')
            no_count = sum(1 for r in all_results if r.get('status') == 'NO')
            na_count = sum(1 for r in all_results if r.get('status') == 'N/A')
            error_count = sum(1 for r in all_results if r.get('status') == 'Error')
            
            avg_confidence = sum(r.get('confidence', 0) for r in all_results) / total_questions if total_questions > 0 else 0
            
            logger.info("📈 Analysis Summary:")
            logger.info(f"   Total Questions: {total_questions}")
            logger.info(f"   YES: {yes_count} ({yes_count/total_questions*100:.1f}%)")
            logger.info(f"   NO: {no_count} ({no_count/total_questions*100:.1f}%)")
            logger.info(f"   N/A: {na_count} ({na_count/total_questions*100:.1f}%)")
            logger.info(f"   Errors: {error_count} ({error_count/total_questions*100:.1f}%)")
            logger.info(f"   Average Confidence: {avg_confidence:.3f}")
            logger.info(f"   Identified Standards: {len(identified_standards)}")
            
            return {
                "document_id": document_id,
                "framework": framework,
                "standard": standard,
                "results_by_section": results_by_section,
                "all_results": all_results,
                "summary": {
                    "total_questions": total_questions,
                    "yes_count": yes_count,
                    "no_count": no_count,
                    "na_count": na_count,
                    "error_count": error_count,
                    "average_confidence": avg_confidence,
                    "completion_rate": (total_questions - error_count) / total_questions if total_questions > 0 else 0
                },
                "enhanced_context": {
                    "company_metadata": company_metadata,
                    "financial_statements": financial_statements,
                    "identified_standards": list(identified_standards.keys()),
                    "total_content_tokens": sum(len(content) // 4 for content in identified_standards.values())
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error in enhanced document processing: {str(e)}")
            raise e

    def _calculate_adequacy(
        self, confidence: float, has_evidence: bool, status: str
    ) -> str:
        """Calculate adequacy - be lenient with partial disclosures since adequacy is subjective"""
        if status == "N/A":
            return "low"
        elif status == "YES":
            # If it's YES, even partial disclosure is adequate since adequacy is subjective
            if confidence >= 0.5 or has_evidence:
                return "high" 
            else:
                return "medium"
        elif status == "NO":
            return "low"
        else:
            # Default case - be more lenient
            return "medium" if has_evidence else "low"

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
                system_prompt = ai_prompts.get_metadata_extraction_system_prompt()

                # Metadata-specific prompt
                user_prompt = ai_prompts.get_metadata_extraction_user_prompt(
                    reference, question_text
                )

                response = self.openai_client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=300,
                )

                content = response.choices[0].message.content

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
                    f"Vector index not found for document {document_id}. "
                    f"Using direct questioning without vector context."
                )

            logger.info(
                f"Directly querying AI about document {document_id} using "
                f"vector store as context"
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

    async def _process_section(
        self,
        section: Dict[str, Any],
        text: str,
        document_id: Optional[str] = None,
        standard_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process a single section of the checklist (async)."""
        try:
            if document_id:
                self.current_document_id = (
                    document_id  # Ensure it's set for every section
                )
            section_name = section.get("section", "unknown")
            original_title = section.get("title", "")
            # Compose full title as 'section_name - original_title' if not already
            # present
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
            for i in range(0, len(items), QUESTION_BATCH_SIZE):
                batch = items[i : i + QUESTION_BATCH_SIZE]

                # ASYNC RATE LIMITING REMOVED - Process all questions without throttling
                # async_semaphore = get_async_rate_semaphore()

                async def process_item_no_limits(item):
                    # NO SEMAPHORE - Process immediately without rate limiting
                    # Mark question as processing in progress tracker
                    if hasattr(self, "progress_tracker") and self.progress_tracker:
                        self.progress_tracker.mark_question_processing(
                            document_id,
                            standard_id or "unknown",
                            item.get("id", "unknown"),
                        )

                    # REMOVED: Small random delay - process immediately
                    # await asyncio.sleep(random.uniform(0.1, 0.5))

                    loop = asyncio.get_running_loop()
                    self.current_document_id = (
                        document_id  # Ensure it's set before each call
                    )
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
            section_name = section.get('section', 'unknown')
            logger.error(f"Error processing section {section_name}: {str(e)}")
            return {
                "section": section.get("section", "unknown"),
                "title": section.get("title", ""),
                "items": [],
                "error": str(e),
            }

    async def analyze_compliance(
        self, document_id: str, text: str, framework: str, standard: str
    ) -> Dict[str, Any]:
        """
        Analyze a document for compliance with a specified framework and
        standard (async).
        """
        logger.info(
            f"Starting compliance analysis for document {document_id} with "
            f"framework {framework} and standard {standard}"
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
                    "Azure OpenAI configuration missing in environment "
                    "variables (chat or embedding)"
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
            "Azure OpenAI configuration missing in environment variables "
            "(chat or embedding)"
        )
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
