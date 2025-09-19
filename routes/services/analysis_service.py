"""
Analysis Service Module

This module contains the core analysis functions extracted from analysis_routes.py
to improve maintainability and modularity.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List

from .semantic_processor import (
    create_semantic_segments,
    find_relevant_segments,
    optimize_context_for_ai,
    prioritize_questions_by_content,
)

# Configure logger
logger = logging.getLogger(__name__)


async def process_smart_mode_analysis(
    checklist: Dict[str, Any],
    text: str,
    document_id: str,
    ai_svc: Any,  # AIService type
    standard: str,
) -> List[Dict[str, Any]]:
    """
    Process compliance analysis using Smart Mode with intelligent segmentation
    and optimized AI processing.

    Smart Mode features:
    - Semantic text segmentation for better context understanding
    - Question-based content prioritization
    - Optimized context delivery to reduce AI processing costs
    - Enhanced accuracy through targeted analysis
    """
    logger.info(f"Starting Smart Mode analysis for standard {standard}")
    start_time = time.time()

    try:
        # Step 1: Extract and analyze all questions from the checklist
        all_questions = []
        section_question_map = {}

        for section_idx, section in enumerate(checklist.get("sections", [])):
            section_questions = []
            for item in section.get("items", []):
                question = item.get("question", "")
                if question:
                    all_questions.append(question)
                    section_questions.append(question)
            section_question_map[section_idx] = section_questions

        logger.info(f"Smart Mode: Analyzing {len(all_questions)} questions")

        # Step 2: Intelligent text segmentation for semantic analysis
        text_segments = create_semantic_segments(text)
        logger.info(f"Smart Mode: Created {len(text_segments)} semantic segments")

        # Step 3: Question-content mapping for optimal processing
        question_priorities = prioritize_questions_by_content(
            all_questions, text_segments
        )

        # Step 4: Process sections with smart AI optimization
        processed_sections = []
        total_tokens_used = 0

        for section_idx, section in enumerate(checklist.get("sections", [])):
            section_start = time.time()

            # Get questions for this section
            section_questions = section_question_map.get(section_idx, [])

            # Find most relevant text segments for this section's questions
            relevant_segments = find_relevant_segments(
                section_questions, text_segments, question_priorities
            )

            # Optimize context for AI processing
            optimized_context = optimize_context_for_ai(relevant_segments, text)

            # Process section with enhanced context
            ai_svc.current_document_id = document_id

            # Enhanced section processing with Smart Mode features
            enhanced_section = await _process_section_smart_mode(
                section, optimized_context, ai_svc, document_id
            )

            # Add Smart Mode metadata
            enhanced_section["processing_mode"] = "smart"
            enhanced_section["processing_time"] = time.time() - section_start
            enhanced_section["segments_analyzed"] = len(relevant_segments)

            processed_sections.append(enhanced_section)

            # Track token usage (estimated)
            estimated_tokens = len(optimized_context.split()) * 1.3
            total_tokens_used += estimated_tokens

        processing_time = time.time() - start_time
        logger.info(
            f"Smart Mode completed for {standard}: {processing_time:.2f}s, "
            f"~{total_tokens_used:.0f} tokens"
        )

        return processed_sections

    except Exception as e:
        logger.error(f"Smart Mode analysis failed for {standard}: {str(e)}")
        # Fallback to standard processing
        logger.info("Falling back to standard processing mode")
        try:
            section_tasks = []
            for section in checklist.get("sections", []):
                ai_svc.current_document_id = document_id
                section_tasks.append(
                    ai_svc._process_section(section, text, document_id)
                )
            return await asyncio.gather(*section_tasks)
        except Exception as fallback_error:
            logger.error(f"Fallback processing also failed: {str(fallback_error)}")
            # Return minimal fallback results
            return [
                {
                    "section_name": section.get("name", f"Section {i + 1}"),
                    "analysis_result": "Analysis unavailable due to processing error",
                    "processing_mode": "fallback",
                    "error": True,
                }
                for i, section in enumerate(checklist.get("sections", []))
            ]


async def process_zap_mode_analysis(
    checklist: Dict[str, Any],
    text: str,
    document_id: str,
    ai_svc: Any,  # AIService type
    standard: str,
) -> List[Dict[str, Any]]:
    """
    Process compliance analysis using Zap Mode with 32 concurrent workers
    for maximum speed.

    Zap Mode features:
    - 32 parallel workers for maximum throughput
    - Direct processing without semantic optimization
    - Speed-first approach with acceptable accuracy trade-offs
    - Minimal context processing for fastest results
    """
    logger.info(f"Starting Zap Mode analysis for standard {standard}")
    start_time = time.time()

    try:
        sections = checklist.get("sections", [])
        if not sections:
            logger.warning(f"No sections found in checklist for standard {standard}")
            return []

        # Create semaphore to limit concurrent workers to 32
        semaphore = asyncio.Semaphore(32)

        async def process_section_with_semaphore(section):
            async with semaphore:
                # Set document ID for this worker
                ai_svc.current_document_id = document_id

                # Process section with minimal optimization for speed
                result = await ai_svc._process_section(section, text, document_id)

                # Add Zap Mode metadata
                result["processing_mode"] = "zap"
                current_task = asyncio.current_task()
                result["worker_id"] = (
                    current_task.get_name() if current_task else "unknown"
                )

                return result

        # Create tasks for all sections
        logger.info(f"Zap Mode: Processing {len(sections)} sections with 32 workers")
        section_tasks = [
            process_section_with_semaphore(section) for section in sections
        ]

        # Execute all tasks concurrently with 32 worker limit
        processed_sections = await asyncio.gather(
            *section_tasks, return_exceptions=True
        )

        # Filter out exceptions and log them
        valid_sections = []
        failed_count = 0

        for i, result in enumerate(processed_sections):
            if isinstance(result, Exception):
                logger.error(f"Zap Mode: Section {i} failed: {str(result)}")
                failed_count += 1
            else:
                valid_sections.append(result)

        processing_time = time.time() - start_time

        logger.info(
            f"Zap Mode completed for {standard}: {processing_time:.2f}s, "
            f"{len(valid_sections)}/{len(sections)} sections successful, "
            f"{failed_count} failures"
        )

        # Add performance metadata to each section
        for section in valid_sections:
            section["zap_mode_stats"] = {
                "total_processing_time": processing_time,
                "concurrent_workers": 32,
                "success_rate": len(valid_sections) / len(sections),
                "sections_processed": len(sections),
            }

        return valid_sections

    except Exception as e:
        logger.error(f"Zap Mode analysis failed for {standard}: {str(e)}")
        # Fallback to standard processing
        logger.info("Falling back to standard processing mode")
        section_tasks = []
        for section in checklist.get("sections", []):
            ai_svc.current_document_id = document_id
            section_tasks.append(ai_svc._process_section(section, text, document_id))
        return await asyncio.gather(*section_tasks)


async def _process_section_smart_mode(
    section: Dict[str, Any], optimized_context: str, ai_svc: Any, document_id: str
) -> Dict[str, Any]:
    """Process a section using Smart Mode with optimized context"""
    # Use the existing _process_section but with optimized context
    return await ai_svc._process_section(section, optimized_context, document_id)
