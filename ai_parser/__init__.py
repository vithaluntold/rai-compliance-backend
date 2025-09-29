#!/usr/bin/env python3
"""
AI Parser Module - 5D Tagging System
"""

from .ai_tagging_engine import AITaggingEngine, TaggingResult
from .file_processor import FileProcessor, ProcessingJob, QueueManager, get_queue_manager

__all__ = [
    'AITaggingEngine',
    'TaggingResult', 
    'FileProcessor',
    'ProcessingJob',
    'QueueManager',
    'get_queue_manager'
]