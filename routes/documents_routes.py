import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the absolute path to the backend directory
BACKEND_DIR = Path(__file__).resolve().parent.parent

# Define directories using Path
UPLOADS_DIR = BACKEND_DIR / "uploads"
ANALYSIS_RESULTS_DIR = BACKEND_DIR / "analysis_results"
UPLOADS_DIR.mkdir(exist_ok=True)
ANALYSIS_RESULTS_DIR.mkdir(exist_ok=True)

router = APIRouter()


@router.get("/", response_model=List[Dict[str, Any]])
async def list_documents():
    """List all uploaded documents by scanning the uploads and results directories."""
    documents = {}
    try:
        # Scan uploads directory for both .pdf and .docx
        for item in UPLOADS_DIR.iterdir():
            if item.is_file() and item.suffix in [".pdf", ".docx"]:
                doc_id = item.stem
                try:
                    stat_result = item.stat()
                    documents[doc_id] = {
                        "id": doc_id,
                        "filename": item.name,
                        "uploaded_at": datetime.fromtimestamp(
                            stat_result.st_ctime
                        ).isoformat(),
                        "status": "PENDING",  # Default status, update below
                        "file_size": stat_result.st_size,
                    }
                except Exception as stat_err:
                    logger.warning(
                        f"Could not get stats for file {item.name}: {stat_err}"
                    )

        # Scan analysis results directory to update status
        for item in ANALYSIS_RESULTS_DIR.iterdir():
            if item.is_file() and item.suffix == ".json":
                doc_id = item.stem.split("_metadata")[
                    0
                ]  # Handle both metadata and final results files
                if doc_id in documents:
                    try:
                        with open(item, "r", encoding="utf-8") as f:
                            results_data = json.load(f)
                        # Determine status based on file content or existence
                        if "_metadata.json" in item.name:
                            meta_status = results_data.get(
                                "_overall_status", "COMPLETED"
                            )
                            if meta_status == "FAILED":
                                documents[doc_id]["status"] = "FAILED"
                            elif meta_status == "PARTIAL":
                                documents[doc_id][
                                    "status"
                                ] = "PROCESSING"  # Or maybe PARTIAL?
                            elif (
                                documents[doc_id]["status"] == "PENDING"
                            ):  # Only update if still pending
                                documents[doc_id][
                                    "status"
                                ] = "PROCESSING"  # Metadata done, likely processing compliance
                        elif (
                            ".json" in item.name and "_metadata" not in item.name
                        ):  # Final results file
                            # Infer status from final results if available
                            final_status = results_data.get(
                                "status", "completed"
                            )  # Default to completed if file exists
                            if final_status == "failed":
                                documents[doc_id]["status"] = "FAILED"
                            else:
                                documents[doc_id]["status"] = "COMPLETED"

                    except Exception as json_err:
                        logger.warning(
                            f"Could not read or parse result file {
                                item.name}: {json_err}"
                        )
                        if (
                            doc_id in documents
                        ):  # Mark as potentially failed if result file is bad
                            documents[doc_id]["status"] = "FAILED"

        # Return documents sorted by upload time (approximated by filename if
        # ctime fails)
        sorted_docs = sorted(
            documents.values(), key=lambda d: d.get("uploaded_at", ""), reverse=True
        )
        return sorted_docs

    except Exception as e:
        logger.error(f"Error listing documents from file system: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error listing documents: {str(e)}"
        )


@router.get("/{document_id}", response_model=Dict[str, Any])
async def get_document(document_id: str) -> Dict[str, Any]:
    """Get document details by checking file system."""
    try:
        # Check for both .pdf and .docx
        file_path = None
        for ext in [".pdf", ".docx"]:
            candidate = UPLOADS_DIR / f"{document_id}{ext}"
            if candidate.exists():
                file_path = candidate
                break
        metadata_path = ANALYSIS_RESULTS_DIR / f"{document_id}_metadata.json"
        results_path = ANALYSIS_RESULTS_DIR / f"{document_id}.json"

        if not file_path:
            logger.error(f"Document file not found: {document_id}")
            raise HTTPException(status_code=404, detail="Document file not found")

        # Basic info from file stats
        stat_result = file_path.stat()
        doc_info = {
            "id": document_id,
            "filename": file_path.name,
            "uploaded_at": datetime.fromtimestamp(stat_result.st_ctime).isoformat(),
            "status": "PENDING",  # Default status
            "file_size": stat_result.st_size,
            "metadata": None,  # Add later if found
            "analysis_results": None,  # Add later if found
        }

        # Check for metadata results
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata_data = json.load(f)
                doc_info["metadata"] = metadata_data  # Embed the metadata
                meta_status = metadata_data.get("_overall_status", "COMPLETED")
                if meta_status == "FAILED":
                    doc_info["status"] = "FAILED"
                elif meta_status == "PARTIAL":
                    doc_info["status"] = "PROCESSING"
                else:
                    doc_info["status"] = "PROCESSING"  # Assume compliance is next
            except Exception as e:
                logger.warning(f"Could not load metadata file {metadata_path}: {e}")
                doc_info["status"] = "FAILED"  # Mark as failed if metadata is corrupt

        # Check for final results (overrides metadata status if present)
        if results_path.exists():
            try:
                with open(results_path, "r", encoding="utf-8") as f:
                    results_data = json.load(f)
                doc_info["analysis_results"] = results_data  # Embed the results
                final_status = results_data.get("status", "completed")
                if final_status == "failed":
                    doc_info["status"] = "FAILED"
                else:
                    doc_info["status"] = "COMPLETED"
            except Exception as e:
                logger.warning(f"Could not load final results file {results_path}: {e}")
                doc_info["status"] = (
                    "FAILED"  # Mark as failed if final results are corrupt
                )

        return doc_info

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error getting document details for {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# Remove or adapt other endpoints that relied on the database
# e.g., update_checklist_item, export_document might need different logic
# For now, let's comment them out to avoid database errors

# @router.post("/{document_id}/checklist/{item_id}")
# async def update_checklist_item(document_id: str, item_id: str, data: Dict[str, Any]):
#     # ... needs rework without database ...
#     pass

# @router.get("/{document_id}/export")
# async def export_document(document_id: str, format: ExportFormat = ExportFormat.JSON):
#     # ... needs rework without database (likely just read JSON file) ...
#     pass
