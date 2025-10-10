#!/usr/bin/env python3
"""
AI Checklist Parser - File Processing Engine
Handles batch processing of JSON files through the AI tagging pipeline
"""

import os
import json
import time
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import queue
import threading

from .ai_tagging_engine import IntelligentAITaggingEngine

logger = logging.getLogger(__name__)

@dataclass
class ProcessingJob:
    """Represents a file processing job in the queue"""
    job_id: str
    filename: str
    file_data: Dict[str, Any]
    created_at: datetime
    status: str = "queued"
    total_questions: int = 0
    processed_questions: int = 0
    current_question: Optional[str] = None

class FileProcessor:
    """Processes uploaded JSON files through the AI tagging pipeline"""
    
    def __init__(self, socketio_app=None):
        self.socketio = socketio_app
        self.is_processing = False
        self.ai_engine = IntelligentAITaggingEngine()
        
    def process_file(self, job: ProcessingJob, download_links: Dict) -> bool:
        """
        Process a single file through the AI tagging pipeline
        
        Args:
            job: Processing job containing file data
            download_links: Dictionary to store download link information
            
        Returns:
            bool: True if processing succeeded, False otherwise
        """
        try:
            logger.info(f"Starting processing for job {job.job_id}: {job.filename}")
            
            # Update job status
            job.status = "processing"
            
            # Extract questions from the file
            questions = self._extract_questions(job.file_data)
            total_questions = sum(len(questions_list) for _, questions_list in questions)
            job.total_questions = total_questions
            
            logger.info(f"Total questions to process: {total_questions}")
            
            # Emit initial progress
            self._emit_progress(job)
            
            # Process each question
            enhanced_data = job.file_data.copy()
            processed_count = 0
            
            for i, (section_key, questions_list) in enumerate(questions):
                logger.info(f"Processing section {section_key} with {len(questions_list)} questions")
                
                for j, question in enumerate(questions_list):
                    # Update current question
                    job.current_question = question.get('question', 'Processing...')[:100] + "..."
                    processed_count += 1
                    job.processed_questions = processed_count
                    
                    logger.info(f"Processing question {processed_count}/{total_questions}: {question.get('id', 'unknown')}")
                    
                    # Emit progress update
                    self._emit_progress(job)
                    
                    # Apply AI classification
                    enhanced_question = self.ai_engine.enhance_question(question)
                    
                    # Update the enhanced data in the correct structure
                    if 'sections' in enhanced_data:
                        # For sections->items structure
                        section_index = int(section_key.replace('section_', ''))
                        enhanced_data['sections'][section_index]['items'][j] = enhanced_question
                    else:
                        # For legacy structure
                        enhanced_data[section_key]['questions'][j] = enhanced_question
                    
                    # Small delay to prevent API rate limiting
                    time.sleep(float(os.getenv('RATE_LIMIT_DELAY', '0.5')))
            
            # Mark as completed
            job.status = "completed"
            job.current_question = "Processing complete"
            
            # Generate download link
            download_id = str(uuid.uuid4())
            output_filename = f"enhanced_{job.filename}"
            
            # Save enhanced file
            success = self._save_enhanced_file(download_id, output_filename, enhanced_data)
            
            if success:
                # Store download link
                download_links[job.job_id] = {
                    'download_id': download_id,
                    'filename': output_filename,
                    'questions_processed': job.total_questions,
                    'created_at': datetime.now()
                }
            
            # Final progress update
            self._emit_progress(job)
            
            logger.info(f"Completed processing for job {job.job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing job {job.job_id}: {e}")
            job.status = "failed"
            job.current_question = f"Error: {str(e)[:100]}"
            self._emit_progress(job)
            return False
    
    def _extract_questions(self, file_data: Dict[str, Any]) -> List[Tuple[str, List[Dict]]]:
        """Extract all questions from the JSON structure"""
        questions = []
        
        # Debug: Log the structure of the file
        logger.info(f"File structure keys: {list(file_data.keys())}")
        
        # Handle the actual JSON structure: sections -> items
        if 'sections' in file_data and isinstance(file_data['sections'], list):
            logger.info(f"Found {len(file_data['sections'])} sections in file")
            for i, section in enumerate(file_data['sections']):
                if isinstance(section, dict) and 'items' in section:
                    items_count = len(section['items']) if isinstance(section['items'], list) else 0
                    logger.info(f"Section {i}: Found {items_count} items")
                    section_key = f"section_{i}"
                    questions.append((section_key, section['items']))
                else:
                    logger.info(f"Section {i}: No 'items' key found, keys: {list(section.keys()) if isinstance(section, dict) else 'Not a dict'}")
        else:
            # Fallback: Original logic for different JSON structures
            logger.info("No 'sections' found, trying original structure")
            for section_key, section_data in file_data.items():
                logger.info(f"Processing section: {section_key}, type: {type(section_data)}")
                if isinstance(section_data, dict):
                    logger.info(f"Section {section_key} keys: {list(section_data.keys())}")
                    if 'questions' in section_data:
                        question_count = len(section_data['questions']) if isinstance(section_data['questions'], list) else 0
                        logger.info(f"Found {question_count} questions in section {section_key}")
                        questions.append((section_key, section_data['questions']))
                elif isinstance(section_data, list):
                    # Handle direct list of questions
                    logger.info(f"Section {section_key} is a direct list with {len(section_data)} items")
                    questions.append((section_key, section_data))
        
        logger.info(f"Total question sections found: {len(questions)}")
        total_questions = sum(len(q_list) for _, q_list in questions)
        logger.info(f"Total questions to process: {total_questions}")
        
        return questions
    
    def _save_enhanced_file(self, download_id: str, filename: str, data: Dict[str, Any]) -> bool:
        """Save enhanced JSON file to downloads directory"""
        try:
            downloads_dir = Path("ai_parser_downloads")
            downloads_dir.mkdir(exist_ok=True)
            
            filepath = downloads_dir / f"{download_id}.json"
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved enhanced file: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save enhanced file: {e}")
            return False
    
    def _emit_progress(self, job: ProcessingJob):
        """Emit progress update via WebSocket"""
        if not self.socketio:
            return
            
        progress_data = {
            'job_id': job.job_id,
            'filename': job.filename,
            'status': job.status,
            'total_questions': job.total_questions,
            'processed_questions': job.processed_questions,
            'current_question': job.current_question,
            'progress_percentage': (job.processed_questions / job.total_questions * 100) if job.total_questions > 0 else 0
        }
        
        try:
            self.socketio.emit('progress_update', progress_data)
        except Exception as e:
            logger.error(f"Failed to emit progress: {e}")

class QueueManager:
    """Manages the file processing queue"""
    
    def __init__(self, socketio_app=None):
        self.file_queue = queue.Queue()
        self.processing_status = {}
        self.download_links = {}
        self.processor = FileProcessor(socketio_app)
        self.worker_thread = None
        self.should_stop = False
        
    def start_worker(self):
        """Start the background queue worker thread"""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self._queue_worker, daemon=True)
            self.worker_thread.start()
            logger.info("Queue worker started")
    
    def stop_worker(self):
        """Stop the background queue worker"""
        self.should_stop = True
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
            logger.info("Queue worker stopped")
    
    def add_job(self, filename: str, file_data: Dict[str, Any]) -> str:
        """Add a new processing job to the queue"""
        job_id = str(uuid.uuid4())
        job = ProcessingJob(
            job_id=job_id,
            filename=filename,
            file_data=file_data,
            created_at=datetime.now()
        )
        
        self.file_queue.put(job)
        self.processing_status[job_id] = job
        
        logger.info(f"Added job {job_id} for file {filename}")
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a processing job"""
        job = self.processing_status.get(job_id)
        if not job:
            return None
        
        return {
            'job_id': job.job_id,
            'filename': job.filename,
            'status': job.status,
            'total_questions': job.total_questions,
            'processed_questions': job.processed_questions,
            'current_question': job.current_question,
            'progress_percentage': (job.processed_questions / job.total_questions * 100) if job.total_questions > 0 else 0
        }
    
    def get_download_info(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get download information for a completed job"""
        return self.download_links.get(job_id)
    
    def get_queue_stats(self) -> Dict[str, int]:
        """Get overall queue statistics"""
        total_jobs = len(self.processing_status)
        completed_jobs = len([job for job in self.processing_status.values() if job.status == "completed"])
        processing_jobs = len([job for job in self.processing_status.values() if job.status == "processing"])
        queued_jobs = len([job for job in self.processing_status.values() if job.status == "queued"])
        failed_jobs = len([job for job in self.processing_status.values() if job.status == "failed"])
        
        return {
            'total_jobs': total_jobs,
            'completed_jobs': completed_jobs,
            'processing_jobs': processing_jobs,
            'queued_jobs': queued_jobs,
            'failed_jobs': failed_jobs,
            'queue_size': self.file_queue.qsize()
        }
    
    def _queue_worker(self):
        """Background worker thread that processes files from the queue"""
        logger.info("Queue worker thread started")
        
        while not self.should_stop:
            try:
                if not self.file_queue.empty() and not self.processor.is_processing:
                    job = self.file_queue.get()
                    self.processor.is_processing = True
                    
                    # Process the file
                    success = self.processor.process_file(job, self.download_links)
                    
                    # Update job status in processing_status
                    self.processing_status[job.job_id] = job
                    
                    self.processor.is_processing = False
                    self.file_queue.task_done()
                    
                    if success:
                        logger.info(f"Successfully processed job {job.job_id}")
                    else:
                        logger.error(f"Failed to process job {job.job_id}")
                        
                else:
                    time.sleep(1)  # Wait before checking again
                    
            except Exception as e:
                logger.error(f"Queue worker error: {e}")
                self.processor.is_processing = False
        
        logger.info("Queue worker thread stopped")

# Global queue manager instance
queue_manager = None

def get_queue_manager(socketio_app=None):
    """Get or create the global queue manager instance"""
    global queue_manager
    if queue_manager is None:
        queue_manager = QueueManager(socketio_app)
        queue_manager.start_worker()
    return queue_manager