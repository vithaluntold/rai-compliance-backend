"""AUDRIC API Routes - Consolidated"""

# Import the consolidated router that contains all endpoints
from routes.analysis_routes import router as api_router

# Export all routers for compatibility
analysis_router = api_router  # Main consolidated router
documents_router = api_router  # Same router, different alias
sessions_router = api_router   # Same router, different alias
checklist_router = api_router  # Same router, different alias
health_router = api_router     # Same router, different alias

__all__ = [
    "api_router", 
    "analysis_router", 
    "documents_router", 
    "sessions_router",
    "checklist_router",
    "health_router"
]
