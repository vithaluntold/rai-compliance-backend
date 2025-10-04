import asyncio
import json
import logging
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv  # type: ignore
from openai import AzureOpenAI  # type: ignore

from services.ai_prompts import ai_prompts

# from services.progress import progress_service, ProgressStatus # Removed unused import
from services.checklist_utils import load_checklist
from services.intelligent_document_analyzer import enhance_compliance_analysis
from services.vector_store import VectorStore, generate_document_id, get_vector_store

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
_processed_questions = (
    {}
)  # Track processed questions per document to prevent duplicates

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
                f"Rate limit would be exceeded. Backing off for {sleep_time:.2f} seconds (retry {retry_count + 1}/{MAX_RETRIES})"
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
            f"Rate limit check passed. Requests: {_request_count}/{REQUESTS_PER_MINUTE}, Tokens: {_token_count}/{TOKENS_PER_MINUTE}"
        )


# Legacy function for backward compatibility
def check_rate_limit(tokens: int = 0) -> None:
    """Legacy function - redirects to enhanced version"""
    check_rate_limit_with_backoff(tokens)


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
        """Determine which model is used based on the results of model choice questions."""
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
            "Could not determine which model is used - defaulting to 'unknown' and processing ALL sections"
        )
        return "unknown"

    def analyze_chunk(
        self, chunk: str, question: str, standard_id: Optional[str] = None
    ) -> dict:
        """
        Analyze a chunk of annual report content against a compliance checklist question.
        Enhanced with intelligent document analysis, rate limiting, and duplicate prevention.
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
                    "analyze_chunk called with no current_document_id set. Cannot perform vector search."
                )
                return {
                    "status": "Error",
                    "confidence": 0.0,
                    "explanation": "No document ID provided for vector search.",
                    "evidence": "",
                    "suggestion": "Check backend logic to ensure document_id is always set.",
                }

            # Check for duplicate questions
            if check_duplicate_question(question, self.current_document_id):
                logger.warning(
                    f"Skipping duplicate question for document {self.current_document_id}"
                )
                return {
                    "status": "N/A",
                    "confidence": 0.0,
                    "explanation": "This question has already been processed for this document.",
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
                        f"Intelligent analysis completed with quality score: {quality_score}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Intelligent document analysis failed, falling back to standard method: {str(e)}"
                    )
                    enhanced_evidence = None

            if not relevant_chunks and not enhanced_evidence:
                logger.warning(f"No relevant chunks found for question: {question}")
                return {
                    "status": "N/A",
                    "confidence": 0.0,
                    "explanation": "No relevant content found in the document",
                    "evidence": "",
                    "suggestion": "Add a clear statement in the financial statement disclosures addressing this requirement.",
                }

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
                context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])
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
                                "content": ai_prompts.get_compliance_analysis_system_prompt(),
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
                                f"Rate limit hit, retrying in {backoff_time} seconds (attempt {api_retry + 1}/{max_api_retries})"
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
                                "explanation": "Analysis could not be completed due to persistent API rate limits.",
                                "evidence": "Multiple rate limit errors occurred.",
                                "suggestion": "Please retry the analysis later when API rate limits reset.",
                            }

                    elif "timeout" in error_message or "connection" in error_message:
                        record_failure()
                        if api_retry < max_api_retries - 1:
                            backoff_time = EXPONENTIAL_BACKOFF_BASE**api_retry
                            logger.warning(
                                f"Connection error, retrying in {backoff_time} seconds (attempt {api_retry + 1}/{max_api_retries})"
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
                                "explanation": f"Analysis failed due to connection issues: {str(api_error)}",
                                "evidence": "Connection error occurred.",
                                "suggestion": "Please check network connectivity and retry the analysis.",
                            }

                    else:
                        # Other API error - don't retry
                        record_failure()
                        logger.error(f"Non-retryable API error: {str(api_error)}")
                        return {
                            "status": "N/A",
                            "confidence": 0.0,
                            "explanation": f"Analysis failed due to API error: {str(api_error)}",
                            "evidence": "API error occurred.",
                            "suggestion": "Please retry the analysis or check system logs for details.",
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
                    f"Response is not valid JSON, attempting to extract structured data: {content[:100]}..."
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
                import re

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
                            "Provide a concrete, practical sample disclosure that would satisfy this IAS 40 requirement. For example: 'The entity should disclose [required information] in accordance with IAS 40.[paragraph]'."
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
                    result["content_analysis"] = "No detailed content analysis provided."

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
                        disclosure_recommendations.append(single_disclosure_match.group(1).strip())
                
                result["disclosure_recommendations"] = disclosure_recommendations if disclosure_recommendations else [
                    "Consider enhancing the disclosure to provide more comprehensive information addressing this requirement."
                ]

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
                    "Consider enhancing the disclosure to provide more comprehensive information addressing this requirement."
                ]
            elif not isinstance(result["disclosure_recommendations"], list):
                # Convert single string to list if needed
                result["disclosure_recommendations"] = [str(result["disclosure_recommendations"])]

            # Ensure evidence is structured and meaningful
            if result["evidence"] == "" or all(
                e == "N/A | N/A | N/A | N/A" for e in result["evidence"]
            ):
                result["evidence"] = ["No relevant evidence found in the document."]

            # Add a suggestion when status is "NO" and no suggestion provided
            if result["status"] == "NO" and "suggestion" not in result:
                result["suggestion"] = (
                    "Provide a concrete, practical sample disclosure that would satisfy this IAS 40 requirement. For example: 'The entity should disclose [required information] in accordance with IAS 40.[paragraph]'."
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

            # Add page number information from vector search results
            if relevant_chunks:
                page_numbers = []
                document_sources = []
                document_extracts = []
                for chunk in relevant_chunks:
                    if chunk.get("page_number") and chunk["page_number"] > 0:
                        page_numbers.append(chunk["page_number"])
                    if chunk.get("chunk_index") is not None:
                        document_sources.append({
                            "page": chunk.get("page_number", 0),
                            "chunk_index": chunk.get("chunk_index", 0),
                            "chunk_type": chunk.get("chunk_type", "text")
                        })
                    # Include document extracts with page references
                    if chunk.get("text"):
                        extract_text = chunk["text"][:500] + "..." if len(chunk["text"]) > 500 else chunk["text"]
                        document_extracts.append({
                            "text": extract_text,
                            "page": chunk.get("page_number", 0),
                            "chunk_index": chunk.get("chunk_index", 0),
                            "relevance_score": chunk.get("score", 0.0)
                        })
                
                if page_numbers:
                    result["source_pages"] = sorted(list(set(page_numbers)))
                if document_sources:
                    result["document_sources"] = document_sources
                if document_extracts:
                    result["document_extracts"] = document_extracts

            # Log the processed result
            logger.info("✅ PROCESSED RESULT:")
            logger.info(f"   Status: {result.get('status', 'N/A')}")
            logger.info(f"   Confidence: {result.get('confidence', 0.0):.2f}")
            logger.info(
                f"   Explanation: {result.get('explanation',
                                              'N/A')[:150]}{'...' if len(str(result.get('explanation',
                                                                                        ''))) > 150 else ''}"
            )
            evidence = result.get('evidence', [])
            evidence_count = len(evidence) if isinstance(evidence, list) else 1
            logger.info(
                f"   Evidence Count: {evidence_count}"
            )
            if result.get("content_analysis"):
                logger.info(
                    f"   Content Analysis: {result.get('content_analysis', 'N/A')[:100]}{'...' if len(str(result.get('content_analysis', ''))) > 100 else ''}"
                )
            if result.get("disclosure_recommendations"):
                logger.info(
                    f"   Disclosure Recommendations: {len(result.get('disclosure_recommendations', []))} suggestions"
                )
            if result.get("source_pages"):
                logger.info(f"   Source Pages: {result.get('source_pages')}")
            if result.get("document_sources"):
                source_info = [f"p{s['page']}" for s in result.get('document_sources', [])]
                logger.info(f"   Document Sources: {', '.join(source_info)}")
            if result.get("document_extracts"):
                extract_count = len(result.get('document_extracts', []))
                avg_score = sum(e.get('relevance_score', 0) for e in result.get('document_extracts', [])) / extract_count if extract_count > 0 else 0
                logger.info(f"   Document Extracts: {extract_count} chunks, avg relevance: {avg_score:.3f}")
            if result.get("status") == "NO" and result.get("suggestion"):
                logger.info(
                    f"   Suggestion: {result.get('suggestion',
                                                 'N/A')[:100]}{'...' if len(str(result.get('suggestion',
                                                                                           ''))) > 100 else ''}"
                )
            if enhanced_evidence:
                quality_score = result.get('enhanced_analysis', {}).get('evidence_quality_score', 0)
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
                "suggestion": "Consider adding explicit disclosure addressing this requirement.",
            }

    def _calculate_adequacy(
        self, confidence: float, has_evidence: bool, status: str
    ) -> str:
        if status == "N/A":
            return "low"
        if confidence >= 0.8 and has_evidence:
            return "high"
        elif confidence >= 0.6 or has_evidence:
            return "medium"
        else:
            return "low"

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
            for i in range(0, len(items), CHUNK_SIZE):
                batch = items[i : i + CHUNK_SIZE]

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
            logger.error(
                f"Error processing section {
                    section.get(
                        'section',
                        'unknown')}: {
                    str(e)}"
            )
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
    if ai_service_instance is None:
        raise RuntimeError("AI service initialization failed")
    logger.info("AI service is available and ready")
except Exception as e:
    logger.error(f"Critical error initializing AI service: {str(e)}")
    raise
