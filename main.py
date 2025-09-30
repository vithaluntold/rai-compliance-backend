from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import routers
from routes import analysis_router, documents_router, sessions_router
from routes.ai_parser_router import router as ai_parser_router

# Initialize FastAPI app with taxonomy module deployment fix
app = FastAPI(
    title="RAi Compliance Engine",
    description="Fully functional AI-powered financial compliance analysis platform",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://raicastingengine.vercel.app",
        "https://raicastingengine-*.vercel.app", 
        "https://complianceengine.vercel.app",
        "https://complianceengine-*.vercel.app",
        "https://compliance-engine.vercel.app",
        "https://compliance-engine-*.vercel.app",
        "https://rai-compliance-frontend.onrender.com",
        "http://localhost:3000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers - ONLY ROUTERS, NO HARDCODED ENDPOINTS
app.include_router(analysis_router, prefix="/api/v1/analysis", tags=["analysis"])
app.include_router(sessions_router, prefix="/api/v1/analysis/sessions", tags=["sessions"])
app.include_router(ai_parser_router, prefix="/api/v1/ai-parser", tags=["ai-parser"])

# Note: documents_router is an alias to analysis_router
# All endpoints are available under /api/v1/analysis/
# Sessions endpoints are available under /api/v1/analysis/sessions/

# Initialize AI Parser queue on startup
@app.on_event("startup")
async def startup_event():
    """Initialize services on application startup"""
    try:
        from ai_parser.file_processor import get_queue_manager
        queue_manager = get_queue_manager()
        queue_manager.start_worker()
        print("✅ AI Parser queue worker started - automatic sequential processing enabled")
    except Exception as e:
        print(f"❌ Error starting AI Parser queue worker: {e}")
        import traceback
        print(traceback.format_exc())

@app.on_event("shutdown")
async def shutdown_event():
    """Handle application shutdown"""
    print("🔄 Application shutdown initiated")
    try:
        from ai_parser.file_processor import get_queue_manager
        queue_manager = get_queue_manager()
        queue_manager.stop_worker()
        print("✅ AI Parser queue worker stopped gracefully")
    except Exception as e:
        print(f"❌ Error during shutdown: {e}")

# Global exception handler to prevent application crashes
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions to prevent application shutdown"""
    logger.error(f"❌ GLOBAL EXCEPTION: {str(exc)}")
    logger.error(f"❌ GLOBAL EXCEPTION TRACEBACK: {traceback.format_exc()}")
    logger.error(f"❌ REQUEST PATH: {request.url.path}")
    logger.error(f"❌ REQUEST METHOD: {request.method}")
    
    # Return error response instead of letting the exception crash the app
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. The application continues to run.",
            "path": str(request.url.path),
            "method": request.method
        }
    )

# Root endpoint
@app.get("/")
async def root():
    return {"message": "RAi Compliance Engine API - Fully Functional", "status": "running"}

# Health endpoint  
@app.get("/api/v1/health")
async def health_check():
    """Enhanced health check with detailed system status"""
    try:
        from ai_parser.file_processor import get_queue_manager
        queue_manager = get_queue_manager()
        queue_status = "running" if queue_manager.worker_thread and queue_manager.worker_thread.is_alive() else "stopped"
    except Exception as e:
        queue_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "uptime": "running",
        "queue_worker": queue_status,
        "features": {
            "file_upload": True,
            "ai_analysis": True,
            "compliance_checking": True,
            "real_time_processing": True
        }
    }

# Keep-alive endpoint to monitor application
@app.get("/api/v1/keepalive")
async def keep_alive():
    """Keep-alive endpoint for monitoring"""
    return {
        "status": "alive",
        "timestamp": datetime.now().isoformat(),
        "message": "Application is running and responsive"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)