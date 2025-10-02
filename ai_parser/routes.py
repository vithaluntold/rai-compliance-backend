#!/usr/bin/env python3
"""
AI Parser Routes - Flask endpoints for the AI tagging system
"""

import os
import json
import logging
from pathlib import Path
from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename

from .file_processor import get_queue_manager

logger = logging.getLogger(__name__)

# Create AI Parser blueprint
ai_parser_bp = Blueprint('ai_parser', __name__, url_prefix='/api/ai-parser')

def init_ai_parser_routes(app, socketio):
    """Initialize AI Parser routes with the Flask app and SocketIO"""
    
    # Get queue manager instance
    queue_manager = get_queue_manager(socketio)
    
    @ai_parser_bp.route('/upload', methods=['POST'])
    def upload_files():
        """Handle multiple file uploads and add to processing queue"""
        try:
            if 'files' not in request.files:
                return jsonify({'error': 'No files provided'}), 400
            
            files = request.files.getlist('files')
            job_ids = []
            
            for file in files:
                if file.filename == '':
                    continue
                    
                if not file.filename.lower().endswith('.json'):
                    continue
                
                # Secure the filename
                filename = secure_filename(file.filename)
                
                # Read and parse JSON file
                try:
                    file_content = file.read().decode('utf-8')
                    file_data = json.loads(file_content)
                except (UnicodeDecodeError, json.JSONDecodeError) as e:
                    logger.error(f"Error parsing file {filename}: {e}")
                    continue
                
                # Add to processing queue
                job_id = queue_manager.add_job(filename, file_data)
                job_ids.append(job_id)
                
                logger.info(f"Added file {filename} to queue with job_id {job_id}")
            
            return jsonify({
                'success': True,
                'job_ids': job_ids,
                'message': f'Successfully queued {len(job_ids)} files for processing'
            })
            
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return jsonify({'error': str(e)}), 500

    @ai_parser_bp.route('/status/<job_id>')
    def get_job_status(job_id):
        """Get the current status of a processing job"""
        try:
            status = queue_manager.get_job_status(job_id)
            if not status:
                return jsonify({'error': 'Job not found'}), 404
            
            return jsonify(status)
            
        except Exception as e:
            logger.error(f"Status check error: {e}")
            return jsonify({'error': str(e)}), 500

    @ai_parser_bp.route('/download/<job_id>')
    def download_file(job_id):
        """Download the enhanced JSON file for a completed job"""
        try:
            download_info = queue_manager.get_download_info(job_id)
            if not download_info:
                return jsonify({'error': 'Download not available'}), 404
            
            download_id = download_info['download_id']
            filename = download_info['filename']
            
            filepath = Path("ai_parser_downloads") / f"{download_id}.json"
            
            if not filepath.exists():
                return jsonify({'error': 'File not found'}), 404
            
            return send_file(
                filepath,
                as_attachment=True,
                download_name=filename,
                mimetype='application/json'
            )
            
        except Exception as e:
            logger.error(f"Download error: {e}")
            return jsonify({'error': str(e)}), 500

    @ai_parser_bp.route('/queue/status')
    def queue_status():
        """Get overall queue status and statistics"""
        try:
            stats = queue_manager.get_queue_stats()
            return jsonify(stats)
            
        except Exception as e:
            logger.error(f"Queue status error: {e}")
            return jsonify({'error': str(e)}), 500

    @ai_parser_bp.route('/health')
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'service': 'ai-parser',
            'queue_stats': queue_manager.get_queue_stats()
        })

    # WebSocket events for AI Parser
    @socketio.on('ai_parser_connect')
    def handle_ai_parser_connect():
        """Handle AI Parser client connection"""
        logger.info('AI Parser client connected')
        socketio.emit('ai_parser_connected', {'data': 'Connected to AI Parser'})

    @socketio.on('ai_parser_subscribe_progress')
    def handle_subscribe_progress(data):
        """Subscribe to progress updates for specific jobs"""
        job_ids = data.get('job_ids', [])
        logger.info(f'Client subscribed to AI Parser progress for jobs: {job_ids}')
        
        # Send current status for all requested jobs
        for job_id in job_ids:
            status = queue_manager.get_job_status(job_id)
            if status:
                socketio.emit('progress_update', status)

    # Register blueprint with app
    app.register_blueprint(ai_parser_bp)
    
    # Ensure downloads directory exists
    Path("ai_parser_downloads").mkdir(exist_ok=True)
    
    logger.info("AI Parser routes initialized")

# Helper functions for integration
def get_ai_parser_stats():
    """Get AI Parser statistics for main dashboard"""
    queue_manager = get_queue_manager()
    if queue_manager:
        return queue_manager.get_queue_stats()
    return {'total_jobs': 0, 'completed_jobs': 0, 'processing_jobs': 0, 'queued_jobs': 0, 'failed_jobs': 0}

def cleanup_old_downloads(days_old=7):
    """Cleanup old download files"""
    try:
        downloads_dir = Path("ai_parser_downloads")
        if not downloads_dir.exists():
            return
        
        import time
        current_time = time.time()
        cutoff_time = current_time - (days_old * 24 * 60 * 60)
        
        for file_path in downloads_dir.glob("*.json"):
            if file_path.stat().st_mtime < cutoff_time:
                file_path.unlink()
                logger.info(f"Cleaned up old download file: {file_path}")
                
    except Exception as e:
        logger.error(f"Error cleaning up downloads: {e}")