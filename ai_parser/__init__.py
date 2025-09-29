#!/usr/bin/env python3
"""
AI Parser Module - 5D Tagging System
"""

from .ai_tagging_engine import IntelligentAITaggingEngine, TaggingResult
from .file_processor import FileProcessor, ProcessingJob, QueueManager, get_queue_manager

__all__ = [
    'AITaggingEngine',  # Backward compatibility
    'IntelligentAITaggingEngine',
    'TaggingResult', 
    'FileProcessor',
    'ProcessingJob',
    'QueueManager',
    'get_queue_manager'
]

# Backward compatibility alias for deployment
AITaggingEngine = IntelligentAITaggingEngine