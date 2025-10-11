"""
Rate Limiter Service - Extracted from AI.py

Handles rate limiting, circuit breaker logic, and duplicate prevention
for Azure OpenAI API calls with exponential backoff.
"""

import asyncio
import logging
import threading
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Enhanced rate limiting settings for Azure OpenAI S0 tier
REQUESTS_PER_MINUTE = 30  # Conservative limit for S0 tier
TOKENS_PER_MINUTE = 40_000  # Conservative limit for S0 tier
MAX_RETRIES = 3
EXPONENTIAL_BACKOFF_BASE = 2
CIRCUIT_BREAKER_THRESHOLD = 10  # Number of consecutive failures before circuit breaker opens

# Global state variables
_rate_lock = threading.Lock()
_request_count = 0
_token_count = 0
_window_start = time.time()
_consecutive_failures = 0
_circuit_breaker_open = False
_circuit_breaker_opened_at = 0
_processed_questions = {}  # Track processed questions per document to prevent duplicates

# Async rate limiting for concurrent processing
_async_rate_semaphore = None  # Will be initialized when needed


class RateLimitError(Exception):
    """Custom exception for rate limiting issues"""
    pass


class CircuitBreakerOpenError(Exception):
    """Custom exception when circuit breaker is open"""
    pass


def get_async_rate_semaphore():
    """Get or create the async rate limiting semaphore for concurrent processing"""
    global _async_rate_semaphore
    if _async_rate_semaphore is None:
        # Limit to 10 concurrent API calls to prevent overwhelming Azure OpenAI
        _async_rate_semaphore = asyncio.Semaphore(10)
    return _async_rate_semaphore


def reset_circuit_breaker():
    """Reset the circuit breaker after successful operations"""
    global _consecutive_failures, _circuit_breaker_open, _circuit_breaker_opened_at
    _consecutive_failures = 0
    _circuit_breaker_open = False
    _circuit_breaker_opened_at = 0
    logger.info("Circuit breaker has been reset")


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


def check_rate_limit(tokens: int = 0) -> None:
    """Legacy function - redirects to enhanced version"""
    check_rate_limit_with_backoff(tokens)


# Convenience functions for external access
def get_rate_limiter():
    """Get access to rate limiting functions - for backward compatibility"""
    return {
        'check_rate_limit': check_rate_limit,
        'check_rate_limit_with_backoff': check_rate_limit_with_backoff,
        'check_duplicate_question': check_duplicate_question,
        'clear_processed_questions': clear_processed_questions,
        'clear_document_questions': clear_document_questions,
        'get_rate_limit_status': get_rate_limit_status,
        'reset_circuit_breaker': reset_circuit_breaker,
        'record_failure': record_failure,
    }