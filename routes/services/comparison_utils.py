"""
Comparison Utilities Module

This module handles performance comparison between Smart Mode and Zap Mode
analysis, including metrics calculation and result aggregation.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Tuple

# Configure logger
logger = logging.getLogger(__name__)


async def run_smart_mode_comparison(
    document_id: str,
    text: str,
    framework: str,
    standards: list,
    special_instructions: str,
    extensive_search: bool,
    ai_svc: Any,
    process_compliance_analysis_internal: Any,
    ANALYSIS_RESULTS_DIR: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run Smart Mode analysis and return metrics and results."""
    logger.info(f"Running Smart Mode for comparison - document {document_id}")
    start_time = time.time()

    try:
        await process_compliance_analysis_internal(
            document_id + "_smart_comparison",
            text,
            framework,
            standards,
            special_instructions,
            extensive_search,
            ai_svc,
            "smart",
        )
        end_time = time.time()
        processing_time = end_time - start_time

        # Load Smart Mode results
        results_path = ANALYSIS_RESULTS_DIR / f"{document_id}_smart_comparison.json"
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                results = json.load(f)
        except FileNotFoundError:
            results = {"sections": []}

        # Extract Smart Mode metrics
        metrics = {
            "processing_time_seconds": round(processing_time, 2),
            "questions_processed": len(results.get("sections", [])),
            "success": True,
            "sections_analyzed": len(results.get("sections", [])),
            "error": None,
        }

        return metrics, results

    except Exception as e:
        logger.error(f"Smart mode failed in comparison: {str(e)}")
        end_time = time.time()
        processing_time = end_time - start_time

        metrics = {
            "processing_time_seconds": round(processing_time, 2),
            "questions_processed": 0,
            "success": False,
            "sections_analyzed": 0,
            "error": str(e),
        }
        results = {"sections": []}

        return metrics, results


async def run_zap_mode_comparison(
    document_id: str,
    text: str,
    framework: str,
    standards: list,
    special_instructions: str,
    extensive_search: bool,
    ai_svc: Any,
    process_compliance_analysis_internal: Any,
    ANALYSIS_RESULTS_DIR: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run Zap Mode analysis and return metrics and results."""
    logger.info(f"Running Zap Mode for comparison - document {document_id}")
    start_time = time.time()

    try:
        await process_compliance_analysis_internal(
            document_id + "_zap_comparison",
            text,
            framework,
            standards,
            special_instructions,
            extensive_search,
            ai_svc,
            "zap",
        )
        end_time = time.time()
        processing_time = end_time - start_time

        # Load Zap Mode results
        results_path = ANALYSIS_RESULTS_DIR / f"{document_id}_zap_comparison.json"
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                results = json.load(f)
        except FileNotFoundError:
            results = {"sections": []}

        # Extract Zap Mode metrics
        metrics = {
            "processing_time_seconds": round(processing_time, 2),
            "questions_processed": len(results.get("sections", [])),
            "success": True,
            "sections_analyzed": len(results.get("sections", [])),
            "error": None,
        }

        return metrics, results

    except Exception as e:
        logger.error(f"Zap mode failed in comparison: {str(e)}")
        end_time = time.time()
        processing_time = end_time - start_time

        metrics = {
            "processing_time_seconds": round(processing_time, 2),
            "questions_processed": 0,
            "success": False,
            "sections_analyzed": 0,
            "error": str(e),
        }
        results = {"sections": []}

        return metrics, results


def calculate_speed_improvement(smart_metrics: dict, zap_metrics: dict) -> str:
    """Calculate speed improvement between Smart and Zap modes."""
    if zap_metrics["processing_time_seconds"] > 0:
        speed_ratio = (
            smart_metrics["processing_time_seconds"]
            / zap_metrics["processing_time_seconds"]
        )
        if speed_ratio > 1:
            return f"{speed_ratio:.1f}x faster (Zap vs Smart)"
        else:
            return f"{1 / speed_ratio:.1f}x faster (Smart vs Zap)"
    else:
        return "Unable to calculate"


def determine_mode_recommendation(
    smart_metrics: dict, zap_metrics: dict
) -> Tuple[str, str]:
    """Determine which mode to recommend based on performance."""
    smart_time = smart_metrics["processing_time_seconds"]
    zap_time = zap_metrics["processing_time_seconds"]

    if smart_time < zap_time:
        return "smart", "Smart mode was faster"
    elif zap_time < smart_time:
        return "zap", "Zap mode was faster"
    else:
        return "equivalent", "Both modes performed similarly"


def handle_failed_modes(smart_metrics: dict, zap_metrics: dict) -> Tuple[str, str, str]:
    """Handle cases where one or both modes failed."""
    speed_improvement = "Analysis failed"
    if smart_metrics["success"] and not zap_metrics["success"]:
        return speed_improvement, "smart", "Only Smart mode succeeded"
    elif zap_metrics["success"] and not smart_metrics["success"]:
        return speed_improvement, "zap", "Only Zap mode succeeded"
    else:
        return speed_improvement, "neither", "Both modes failed"


def calculate_comparison_metrics(
    smart_metrics: dict, zap_metrics: dict
) -> Tuple[str, str, str]:
    """Calculate performance comparison metrics between Smart and Zap modes."""
    if smart_metrics["success"] and zap_metrics["success"]:
        speed_improvement = calculate_speed_improvement(smart_metrics, zap_metrics)
        recommendation, reason = determine_mode_recommendation(
            smart_metrics, zap_metrics
        )
        return speed_improvement, recommendation, reason
    else:
        return handle_failed_modes(smart_metrics, zap_metrics)


def build_comparison_results(
    document_id: str,
    framework: str,
    standards: list,
    smart_metrics: dict,
    zap_metrics: dict,
    smart_results: dict,
    zap_results: dict,
    speed_improvement: str,
    recommendation: str,
    reason: str,
) -> dict:
    """Build the final comparison results structure."""
    # Merge results (use Smart mode results as primary if available)
    primary_results = smart_results if smart_metrics["success"] else zap_results
    if not smart_metrics["success"] and not zap_metrics["success"]:
        primary_results = {"sections": [], "status": "FAILED"}

    # Build final comparison results
    results = {
        "document_id": document_id,
        "status": (
            "COMPLETED"
            if (smart_metrics["success"] or zap_metrics["success"])
            else "FAILED"
        ),
        "processing_mode": "comparison",
        "framework": framework,
        "standards": standards,
        "sections": primary_results.get("sections", []),
        "message": f"Comparison analysis completed - {recommendation} mode recommended",
        "comparison_results": {
            "enabled": True,
            "modes_compared": ["smart", "zap"],
            "smart_mode": smart_metrics,
            "zap_mode": zap_metrics,
            "performance_metrics": {
                "speed_improvement": speed_improvement,
                "recommendation": recommendation,
                "recommendation_reason": reason,
                "total_analysis_time": round(
                    smart_metrics["processing_time_seconds"]
                    + zap_metrics["processing_time_seconds"],
                    2,
                ),
            },
        },
    }

    return results


async def process_compliance_comparison(
    document_id: str,
    text: str,
    framework: str,
    standards: list,
    special_instructions: str,
    extensive_search: bool,
    ai_svc: Any,
    process_compliance_analysis_internal: Any,
    ANALYSIS_RESULTS_DIR: Path,
) -> None:
    """
    Process compliance analysis using both Smart and Zap modes for comparison.
    Executes both modes sequentially and compares their performance metrics.
    """
    try:
        logger.info(f"Starting comparison mode analysis for document {document_id}")

        # Initialize base results
        initial_results = {
            "document_id": document_id,
            "status": "PROCESSING",
            "processing_mode": "comparison",
            "framework": framework,
            "standards": standards,
            "message": "Running Smart and Zap mode comparison",
            "comparison_results": {
                "enabled": True,
                "modes_compared": ["smart", "zap"],
            },
        }

        # Save initial progress
        results_path = ANALYSIS_RESULTS_DIR / f"{document_id}.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(initial_results, f, indent=2, ensure_ascii=False)

        # Run both modes and gather results
        smart_metrics, smart_results = await run_smart_mode_comparison(
            document_id,
            text,
            framework,
            standards,
            special_instructions,
            extensive_search,
            ai_svc,
            process_compliance_analysis_internal,
            ANALYSIS_RESULTS_DIR,
        )

        zap_metrics, zap_results = await run_zap_mode_comparison(
            document_id,
            text,
            framework,
            standards,
            special_instructions,
            extensive_search,
            ai_svc,
            process_compliance_analysis_internal,
            ANALYSIS_RESULTS_DIR,
        )

        # Calculate comparison metrics
        speed_improvement, recommendation, reason = calculate_comparison_metrics(
            smart_metrics, zap_metrics
        )

        # Build final results
        final_results = build_comparison_results(
            document_id,
            framework,
            standards,
            smart_metrics,
            zap_metrics,
            smart_results,
            zap_results,
            speed_improvement,
            recommendation,
            reason,
        )

        # Save final results
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Comparison analysis completed for document {document_id} - "
            f"{recommendation} mode recommended"
        )

    except Exception as e:
        logger.error(f"Error in comparison analysis: {str(e)}", exc_info=True)
        # Update results with error
        try:
            results_path = ANALYSIS_RESULTS_DIR / f"{document_id}.json"
            error_results = {
                "document_id": document_id,
                "status": "FAILED",
                "error": str(e),
            }
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(error_results, f, indent=2, ensure_ascii=False)
        except Exception as save_error:
            # Failed to save error state - log but continue
            logger.error(
                f"Failed to save error state for {document_id}: {str(save_error)}"
            )
