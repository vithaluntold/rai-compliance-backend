"""
Smart Document Processing Integration
Replaces current chunking workflow with intelligent categorization
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Add path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from complete_document_processor import CompleteDocumentProcessor
from services.checklist_utils import get_available_frameworks, load_checklist

logger = logging.getLogger(__name__)

# Get backend directory
BACKEND_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPLOADS_DIR = BACKEND_DIR / "uploads"
ANALYSIS_RESULTS_DIR = BACKEND_DIR / "analysis_results"


async def process_upload_tasks_smart(
    document_id: str, ai_svc, text: str = "", processing_mode: str = "smart"
) -> None:
    """Smart document processing with categorization - replaces old workflow"""
    
    logger.info(f"STEP 1: Starting smart processing workflow for document {document_id}")
    
    # Check for duplicate processing
    processing_lock_file = ANALYSIS_RESULTS_DIR / f"{document_id}.processing"
    if processing_lock_file.exists():
        logger.info(f"⚠️  Document {document_id} is already being processed - skipping")
        return
    
    # Create processing lock
    processing_lock_file.touch()
    logger.info(f"✅ STEP 1 COMPLETE: Processing lock created for {document_id}")
    
    try:
        logger.info(f"🔍 STEP 2: Starting file discovery for document {document_id}")
        
        # Find uploaded file
        upload_path = None
        for ext in [".pdf", ".docx"]:
            candidate = UPLOADS_DIR / f"{document_id}{ext}"
            if candidate.exists():
                upload_path = candidate
                logger.info(f"📄 Found upload file: {upload_path}")
                break
        
        if not upload_path:
            logger.error(f"❌ STEP 2 FAILED: Upload file not found for {document_id}")
            raise Exception(f"Upload file not found for {document_id}")
        
        logger.info(f"✅ STEP 2 COMPLETE: File discovered at {upload_path}")
        
        # Phase 1: Smart Document Processing (replaces chunking)
        logger.info(f"🧠 STEP 3: Starting smart categorization for {document_id}")
        processor = CompleteDocumentProcessor()
        
        processing_result = await processor.process_document(str(upload_path), document_id)
        
        if processing_result['status'] != 'success':
            logger.error(f"❌ STEP 3 FAILED: Smart processing failed - {processing_result.get('message', 'Unknown error')}")
            raise Exception(f"Smart processing failed: {processing_result.get('message', 'Unknown error')}")
        
        logger.info(f"✅ STEP 3 COMPLETE: Smart categorization finished - {processing_result['total_content_pieces']} pieces categorized")
        
        # Save enhanced metadata with categorization results
        logger.info(f"💾 STEP 4: Saving categorization metadata for {document_id}")
        enhanced_metadata = {
            "document_id": document_id,
            "processing_mode": "smart_categorization",
            "categorization_results": processing_result,
            "timestamp": datetime.now().isoformat(),
            "_overall_status": "METADATA_COMPLETED"
        }
        
        metadata_path = ANALYSIS_RESULTS_DIR / f"{document_id}_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(enhanced_metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ STEP 4 COMPLETE: Metadata saved to {metadata_path}")
        logger.info(f"📊 Categorization Summary - Categories: {len(processing_result.get('category_distribution', {}))}, Topics: {len(processing_result.get('top_topics', {}))}")
        
        # Phase 2: Smart Checklist Analysis (replaces old compliance processing)
        logger.info(f"📋 STEP 5: Starting smart checklist analysis for {document_id}")
        
        checklist_results = await process_checklist_analysis_smart(
            document_id, ai_svc, processor
        )
        
        logger.info(f"✅ STEP 5 COMPLETE: Checklist analysis finished - {len(checklist_results)} frameworks processed")
        
        # Save comprehensive results
        logger.info(f"💾 STEP 6: Saving final comprehensive results for {document_id}")
        final_results = {
            "document_id": document_id,
            "processing_mode": "smart_categorization",
            "categorization_summary": {
                "total_content_pieces": processing_result['total_content_pieces'],
                "category_distribution": processing_result['category_distribution'],
                "citation_metadata": processing_result['citation_metadata'],
                "avg_confidence": processing_result['avg_confidence']
            },
            "checklist_analysis": checklist_results,
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        }
        
        results_path = ANALYSIS_RESULTS_DIR / f"{document_id}.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ STEP 6 COMPLETE: Final results saved to {results_path}")
        
        # Create completion lock files
        logger.info(f"🔒 STEP 7: Creating completion lock files for {document_id}")
        metadata_lock_file = ANALYSIS_RESULTS_DIR / f"{document_id}.metadata_completed"
        metadata_lock_file.touch()
        
        completion_lock_file = ANALYSIS_RESULTS_DIR / f"{document_id}.completed"
        completion_lock_file.touch()
        
        # Remove processing lock
        if processing_lock_file.exists():
            processing_lock_file.unlink()
        
        logger.info(f"✅ STEP 7 COMPLETE: Lock files created and processing lock removed")
        
        # Cleanup uploaded file (optional - keep for debugging)
        # _cleanup_uploaded_file(document_id)
        
        logger.info(f"🎉 ALL STEPS COMPLETE: Smart processing succeeded for {document_id}")
        logger.info(f"📈 FINAL SUMMARY: {len(checklist_results)} frameworks processed, {processing_result['total_content_pieces']} content pieces categorized")
        logger.info(f"⏱️  Processing completed at: {datetime.now().isoformat()}")
        
    except Exception as e:
        logger.error(f"❌ PROCESSING FAILED at document {document_id}: {e}")
        logger.error(f"🚫 SMART CATEGORIZATION FAILED for {document_id}: {e}")
        logger.error("🎯 STRICT MODE: No fallback processing - document processing terminated")
        
        # Save failure status with strict mode indication
        logger.info(f"💾 Saving failure metadata for {document_id}")
        failure_metadata = {
            "document_id": document_id,
            "processing_mode": "smart_categorization_strict",
            "error": f"STRICT MODE FAILURE: {str(e)}",
            "message": "Smart categorization failed - no fallback processing available",
            "timestamp": datetime.now().isoformat(),
            "_overall_status": "FAILED"
        }
        
        metadata_path = ANALYSIS_RESULTS_DIR / f"{document_id}_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(failure_metadata, f, indent=2, ensure_ascii=False)
        
        # Create error lock file
        error_lock_file = ANALYSIS_RESULTS_DIR / f"{document_id}.error"
        error_lock_file.touch()
        
        # Remove processing lock
        if processing_lock_file.exists():
            processing_lock_file.unlink()
        
        # Re-raise to ensure failure propagates
        raise RuntimeError(f"Smart categorization failed in strict mode: {str(e)}")


async def process_checklist_analysis_smart(
    document_id: str, 
    ai_svc, 
    processor: CompleteDocumentProcessor
) -> Dict[str, Any]:
    """Process checklist using smart categorization (replaces old compliance analysis)"""
    
    try:
        logger.info(f"📋 CHECKLIST STEP 1: Loading available frameworks for {document_id}")
        # Load available frameworks
        available_frameworks = get_available_frameworks()
        logger.info(f"✅ CHECKLIST STEP 1 COMPLETE: Found {len(available_frameworks)} frameworks: {available_frameworks}")
        
        all_results = {}
        total_questions = 0
        
        for framework_name in available_frameworks:
            logger.info(f"📝 CHECKLIST STEP 2: Processing framework '{framework_name}' for {document_id}")
            
            # Load checklist
            checklist_data = load_checklist(framework_name)
            if not checklist_data:
                logger.warning(f"⚠️  Could not load checklist for {framework_name}")
                continue
                
            logger.info(f"✅ Checklist loaded for {framework_name} - {len(checklist_data.get('sections', []))} sections found")
            
            framework_results = []
            questions_processed = 0
            framework_start_time = time.time()
            
            logger.info(f"🤖 CHECKLIST STEP 3: Starting smart question processing for {framework_name}")
            
            # Process questions with smart accumulation
            for section in checklist_data.get("sections", []):
                for item in section.get("items", []):
                    question = item.get("question", "")
                    if not question:
                        continue
                    
                    try:
                        # Smart answer using categorized content
                        smart_result = processor.answer_question_intelligently(
                            question, document_id, max_content_length=800
                        )
                        
                        # Format result for checklist
                        checklist_item = {
                            "reference": item.get("reference", ""),
                            "question": question,
                            "answer_status": smart_result['status'],
                            "answer_content": smart_result.get('answer_content', ''),
                            "category_classification": smart_result.get('category_classification', {}),
                            "citations": smart_result.get('citations', []),
                            "confidence": smart_result.get('confidence', 0.0),
                            "evidence_pieces": smart_result.get('evidence_pieces', 0),
                            "content_length": smart_result.get('content_length', 0),
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        framework_results.append(checklist_item)
                        questions_processed += 1
                        total_questions += 1
                        
                        # Log progress every 20 questions
                        if questions_processed % 20 == 0:
                            logger.info(f"⏳ Progress: {questions_processed} questions processed for {framework_name}")
                        
                        # Log progress every 50 questions overall
                        if total_questions % 50 == 0:
                            logger.info(f"Processed {total_questions} questions so far...")
                        
                    except Exception as e:
                        logger.error(f"❌ Error processing question '{question[:50]}...': {e}")
                        # Continue with next question
                        continue
            
            # Calculate framework statistics
            framework_end_time = time.time()
            processing_time = framework_end_time - framework_start_time
            confidences = [item['confidence'] for item in framework_results if item['confidence'] > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            all_results[framework_name] = {
                "questions_processed": questions_processed,
                "avg_confidence": round(avg_confidence, 3),
                "processing_time_seconds": round(processing_time, 2),
                "results": framework_results,
                "processing_timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✅ CHECKLIST STEP 3 COMPLETE for {framework_name}: {questions_processed} questions processed")
            logger.info(f"📊 Framework Stats - Avg Confidence: {avg_confidence:.3f}, Processing Time: {processing_time:.2f}s")
        
        logger.info(f"🎉 ALL CHECKLIST STEPS COMPLETE: {len(all_results)} frameworks processed, {total_questions} total questions")
        logger.info(f"📈 Overall Stats - Frameworks: {list(all_results.keys())}")
        return all_results
        
    except Exception as e:
        logger.error(f"❌ CHECKLIST PROCESSING FAILED: {e}")
        raise


def _cleanup_uploaded_file(document_id: str):
    """Clean up uploaded file after processing"""
    try:
        for ext in [".pdf", ".docx"]:
            file_path = UPLOADS_DIR / f"{document_id}{ext}"
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Cleaned up uploaded file: {file_path}")
                break
    except Exception as e:
        logger.warning(f"Could not cleanup uploaded file for {document_id}: {e}")


# Test function
async def test_smart_processing():
    """Test the smart processing integration"""
    document_id = "test_smart_001"
    
    # Mock AI service
    class MockAIService:
        pass
    
    ai_svc = MockAIService()
    
    try:
        await process_upload_tasks_smart(document_id, ai_svc)
        print("Smart processing test completed successfully!")
    except Exception as e:
        print(f"Smart processing test failed: {e}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_smart_processing())