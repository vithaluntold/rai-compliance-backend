"""
Semantic Processing Module

This module handles semantic text analysis and processing for compliance analysis.
Contains functions for text segmentation, question prioritization, and context optimization.
"""

from typing import Dict, List


def create_semantic_segments(text: str) -> List[str]:
    """Create semantic segments from text for intelligent analysis"""
    # Split text into logical segments (paragraphs, sections)
    segments = []

    # Split by double newlines (paragraph breaks)
    paragraphs = text.split("\n\n")

    current_segment = ""
    max_segment_length = 2000  # Optimal size for semantic analysis

    for paragraph in paragraphs:
        if len(current_segment) + len(paragraph) < max_segment_length:
            current_segment += paragraph + "\n\n"
        else:
            if current_segment.strip():
                segments.append(current_segment.strip())
            current_segment = paragraph + "\n\n"

    # Add final segment
    if current_segment.strip():
        segments.append(current_segment.strip())

    return segments


def prioritize_questions_by_content(
    questions: List[str], segments: List[str]
) -> Dict[str, float]:
    """Prioritize questions based on content relevance"""
    priorities: Dict[str, float] = {}

    # Simple keyword-based prioritization
    for question in questions:
        question_lower = question.lower()
        total_relevance: float = 0.0

        for segment in segments:
            segment_lower = segment.lower()

            # Check for direct keyword matches
            question_words = set(question_lower.split())
            segment_words = set(segment_lower.split())
            overlap = len(question_words.intersection(segment_words))

            if overlap > 0:
                total_relevance += overlap / len(question_words)

        priorities[question] = total_relevance

    return priorities


def find_relevant_segments(
    section_questions: List[str], segments: List[str], priorities: Dict[str, float]
) -> List[str]:
    """Find most relevant text segments for section questions"""
    if not section_questions:
        return segments[:3]  # Return first 3 segments if no questions

    relevant_segments = []
    max_segments = min(5, len(segments))  # Limit for efficiency

    # Score segments based on question relevance
    segment_scores = []
    for i, segment in enumerate(segments):
        score: float = 0.0
        for question in section_questions:
            question_priority = priorities.get(question, 0.0)
            # Simple text overlap scoring
            question_words = set(question.lower().split())
            segment_words = set(segment.lower().split())
            overlap = len(question_words.intersection(segment_words))
            score += overlap * question_priority

        segment_scores.append((score, i, segment))

    # Sort by score and take top segments
    segment_scores.sort(reverse=True)
    for score, _, segment in segment_scores[:max_segments]:
        relevant_segments.append(segment)

    return relevant_segments


def optimize_context_for_ai(segments: List[str], full_text: str) -> str:
    """Optimize context for AI processing to reduce token usage"""
    if not segments:
        # Fallback to beginning of document
        return full_text[:4000]

    # Combine relevant segments with smart truncation
    combined_context = "\n\n".join(segments)

    # Limit total context size for cost optimization
    max_context_length = 6000
    if len(combined_context) > max_context_length:
        combined_context = combined_context[:max_context_length] + "..."

    return combined_context
