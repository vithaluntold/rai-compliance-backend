import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from models.checklist import ChecklistStatus as Status
from services.checklist_utils import load_checklist

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()


class ChecklistItemUpdate(BaseModel):
    status: Optional[Status]
    comment: Optional[str]
    resolved: Optional[bool]


class AutoFillRequest(BaseModel):
    section_id: Optional[str] = None


@router.put("/{document_id}/items/{item_ref}")
async def update_checklist_item(
    document_id: str, item_ref: str, update: ChecklistItemUpdate
):
    """Update a checklist item."""
    try:
        result_path = f"checklist_data/{document_id}.json"
        if not os.path.exists(result_path):
            raise HTTPException(status_code=404, detail="Analysis not found")

        with open(result_path, "r") as f:
            data = json.load(f)

        # Find and update the item
        for item in data["items"]:
            if item["ref"] == item_ref:
                if update.status is not None:
                    item["status"] = update.status
                if update.comment is not None:
                    item["comment"] = update.comment
                if update.resolved is not None:
                    item["resolved"] = update.resolved
                break
        else:
            raise HTTPException(status_code=404, detail="Item not found")

        # Save updated data
        with open(result_path, "w") as f:
            json.dump(data, f, indent=2)

        return {"message": "Item updated successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{document_id}/items/{item_ref}")
async def delete_checklist_item(document_id: str, item_ref: str, comment: str):
    """Delete a checklist item with a mandatory comment."""
    if not comment:
        raise HTTPException(status_code=400, detail="Comment is required for deletion")

    try:
        result_path = f"checklist_data/{document_id}.json"
        if not os.path.exists(result_path):
            raise HTTPException(status_code=404, detail="Analysis not found")

        with open(result_path, "r") as f:
            data = json.load(f)

        # Find and mark item as deleted
        for item in data["items"]:
            if item["ref"] == item_ref:
                item["deleted"] = True
                item["deletion_comment"] = comment
                break
        else:
            raise HTTPException(status_code=404, detail="Item not found")

        # Save updated data
        with open(result_path, "w") as f:
            json.dump(data, f, indent=2)

        return {"message": "Item deleted successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{document_id}/export")
async def export_checklist(document_id: str, format: str = "json"):
    """Export checklist in various formats."""
    try:
        result_path = f"checklist_data/{document_id}.json"
        if not os.path.exists(result_path):
            raise HTTPException(status_code=404, detail="Analysis not found")

        with open(result_path, "r") as f:
            data = json.load(f)

        if format == "json":
            return data

        elif format == "excel":
            # Convert to DataFrame
            df = pd.DataFrame(data["items"])

            # Create Excel file
            excel_path = f"checklist_data/{document_id}_export.xlsx"
            df.to_excel(excel_path, index=False)

            return FileResponse(
                excel_path,
                filename=f"ias40_checklist_{document_id}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        else:
            raise HTTPException(status_code=400, detail="Unsupported format")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/{document_id}/checklist/auto-fill")
async def auto_fill_checklist(
    document_id: str, request: Dict[str, Any] = Body(default={})
):
    """Auto-fill checklist items based on document analysis."""
    try:
        logger.info(f"Received auto-fill request for document {document_id}")
        logger.info(f"Request body: {request}")

        # Extract section_id from request body
        section_id = request.get("section_id") if request else None

        # Get analysis results from file system
        results_path = os.path.join("analysis_results", f"{document_id}.json")
        if not os.path.exists(results_path):
            raise HTTPException(
                status_code=404,
                detail=f"No analysis results found for document {document_id}",
            )

        with open(results_path, "r", encoding="utf-8") as f:
            analysis_data = json.load(f)

        # Check if analysis is completed
        if analysis_data.get("status") not in ["COMPLETED", "completed"]:
            raise HTTPException(
                status_code=400, detail="Document analysis not completed"
            )

        # Load base checklist template
        checklist_template = load_checklist()

        # Auto-fill checklist items based on analysis
        checklist_items = []
        total_items = 0
        completed_items = 0

        for section in checklist_template["sections"]:
            # Skip sections that don't match the requested section_id if one is provided
            if section_id and section["title"] != section_id:
                continue

            for item in section["items"]:
                total_items += 1
                auto_filled_item = {
                    "id": item["id"],
                    "section": section["title"],
                    "requirement": item["requirement"],
                    "reference": item["reference"],
                    "status": "PENDING",
                    "evidence": "",
                    "comments": "",
                    "auto_filled": True,
                }

                # Try to find matching analysis result
                for analysis_section in analysis_data.get("sections", []):
                    for analysis_item in analysis_section.get("items", []):
                        if analysis_item["id"] == item["id"]:
                            auto_filled_item.update(
                                {
                                    "status": analysis_item.get("status", "PENDING"),
                                    "evidence": analysis_item.get("evidence", ""),
                                    "comments": analysis_item.get("ai_explanation", ""),
                                    "suggestion": analysis_item.get("suggestion", ""),
                                    "auto_filled": True,
                                }
                            )
                            if analysis_item.get("status") != "PENDING":
                                completed_items += 1
                            break

                checklist_items.append(auto_filled_item)

        # Create response with metadata
        response = {
            "items": checklist_items,
            "metadata": {
                "total_items": total_items,
                "completed_items": completed_items,
                "compliance_score": (
                    (completed_items / total_items) if total_items > 0 else 0
                ),
            },
        }

        # Save auto-filled checklist
        checklist_path = os.path.join("checklist_data", f"{document_id}.json")
        with open(checklist_path, "w", encoding="utf-8") as f:
            json.dump(response, f, indent=2)

        logger.info(f"Successfully auto-filled checklist for document {document_id}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error auto-filling checklist for document {document_id}: {str(e)}"
        )
        raise HTTPException(status_code=500, detail=str(e))
