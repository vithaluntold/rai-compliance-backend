"""AUDRIC API Routes"""

from routes.analysis_routes import router as analysis_router
from routes.documents_routes import router as documents_router
from routes.sessions_routes import router as sessions_router

# from routes.checklist_routes import router as checklist_router

__all__ = ["analysis_router", "documents_router", "sessions_router"]  # , 'checklist_router']
