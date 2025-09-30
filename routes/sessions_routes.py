"""
Session Management Routes for RAi Compliance Engine
Handles creating, saving, loading, and managing analysis sessions
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import json
import os
import uuid
from pathlib import Path

router = APIRouter(tags=["sessions"])

# Pydantic models for session management
class SessionCreate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None

class SessionUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    chat_state: Optional[Dict[str, Any]] = None
    messages: Optional[List[Dict[str, Any]]] = None

class SessionResponse(BaseModel):
    session_id: str
    title: str
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    document_count: int = 0
    last_document_id: Optional[str] = None
    status: str = "active"  # active, completed, archived

class SessionDetail(SessionResponse):
    chat_state: Optional[Dict[str, Any]] = None
    messages: Optional[List[Dict[str, Any]]] = None
    documents: Optional[List[Dict[str, Any]]] = None

# Session storage directory
SESSIONS_DIR = Path("sessions")
SESSIONS_DIR.mkdir(exist_ok=True)

def generate_session_id() -> str:
    """Generate a unique session ID"""
    return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

def get_session_file_path(session_id: str) -> Path:
    """Get the file path for a session"""
    return SESSIONS_DIR / f"{session_id}.json"

def save_session_to_file(session_id: str, session_data: Dict[str, Any]) -> None:
    """Save session data to file"""
    session_file = get_session_file_path(session_id)
    with open(session_file, 'w', encoding='utf-8') as f:
        json.dump(session_data, f, indent=2, default=str)

def load_session_from_file(session_id: str) -> Optional[Dict[str, Any]]:
    """Load session data from file"""
    session_file = get_session_file_path(session_id)
    if not session_file.exists():
        return None
    
    try:
        with open(session_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading session {session_id}: {e}")
        return None

@router.post("/create", response_model=SessionResponse)
async def create_session(session_data: SessionCreate):
    """Create a new analysis session"""
    try:
        session_id = generate_session_id()
        now = datetime.now()
        
        # Default title if not provided
        title = session_data.title or f"Analysis Session {now.strftime('%Y-%m-%d %H:%M')}"
        
        session = {
            "session_id": session_id,
            "title": title,
            "description": session_data.description,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "document_count": 0,
            "last_document_id": None,
            "status": "active",
            "chat_state": None,
            "messages": [],
            "documents": []
        }
        
        # Save to file
        save_session_to_file(session_id, session)
        
        return SessionResponse(
            session_id=session_id,
            title=title,
            description=session_data.description,
            created_at=now,
            updated_at=now,
            document_count=0,
            last_document_id=None,
            status="active"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")

@router.get("/list", response_model=List[SessionResponse])
async def list_sessions(limit: int = 50, offset: int = 0):
    """List all analysis sessions"""
    try:
        sessions = []
        
        # Get all session files
        session_files = list(SESSIONS_DIR.glob("session_*.json"))
        session_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)  # Sort by modification time
        
        # Apply pagination
        paginated_files = session_files[offset:offset + limit]
        
        for session_file in paginated_files:
            session_data = load_session_from_file(session_file.stem)
            if session_data:
                sessions.append(SessionResponse(
                    session_id=session_data["session_id"],
                    title=session_data["title"],
                    description=session_data.get("description"),
                    created_at=datetime.fromisoformat(session_data["created_at"]),
                    updated_at=datetime.fromisoformat(session_data["updated_at"]),
                    document_count=session_data.get("document_count", 0),
                    last_document_id=session_data.get("last_document_id"),
                    status=session_data.get("status", "active")
                ))
        
        return sessions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")

@router.get("/{session_id}", response_model=SessionDetail)
async def get_session(session_id: str):
    """Get a specific session with full details"""
    try:
        session_data = load_session_from_file(session_id)
        
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return SessionDetail(
            session_id=session_data["session_id"],
            title=session_data["title"],
            description=session_data.get("description"),
            created_at=datetime.fromisoformat(session_data["created_at"]),
            updated_at=datetime.fromisoformat(session_data["updated_at"]),
            document_count=session_data.get("document_count", 0),
            last_document_id=session_data.get("last_document_id"),
            status=session_data.get("status", "active"),
            chat_state=session_data.get("chat_state"),
            messages=session_data.get("messages", []),
            documents=session_data.get("documents", [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")

@router.put("/{session_id}", response_model=SessionResponse)
async def update_session(session_id: str, session_update: SessionUpdate):
    """Update a session with new data"""
    try:
        session_data = load_session_from_file(session_id)
        
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Update fields
        if session_update.title is not None:
            session_data["title"] = session_update.title
        
        if session_update.description is not None:
            session_data["description"] = session_update.description
        
        if session_update.chat_state is not None:
            session_data["chat_state"] = session_update.chat_state
            
            # Update document count and last document if present in chat state
            if "documentId" in session_update.chat_state and session_update.chat_state["documentId"]:
                session_data["last_document_id"] = session_update.chat_state["documentId"]
                session_data["document_count"] = max(session_data.get("document_count", 0), 1)
        
        if session_update.messages is not None:
            session_data["messages"] = session_update.messages
        
        # Update timestamp
        session_data["updated_at"] = datetime.now().isoformat()
        
        # Save updated session
        save_session_to_file(session_id, session_data)
        
        return SessionResponse(
            session_id=session_data["session_id"],
            title=session_data["title"],
            description=session_data.get("description"),
            created_at=datetime.fromisoformat(session_data["created_at"]),
            updated_at=datetime.fromisoformat(session_data["updated_at"]),
            document_count=session_data.get("document_count", 0),
            last_document_id=session_data.get("last_document_id"),
            status=session_data.get("status", "active")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update session: {str(e)}")

@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    try:
        session_file = get_session_file_path(session_id)
        
        if not session_file.exists():
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Delete the session file
        session_file.unlink()
        
        return JSONResponse(content={"message": "Session deleted successfully"})
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")

@router.post("/{session_id}/archive")
async def archive_session(session_id: str):
    """Archive a session (mark as completed)"""
    try:
        session_data = load_session_from_file(session_id)
        
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Update status to archived
        session_data["status"] = "archived"
        session_data["updated_at"] = datetime.now().isoformat()
        
        # Save updated session
        save_session_to_file(session_id, session_data)
        
        return JSONResponse(content={"message": "Session archived successfully"})
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to archive session: {str(e)}")