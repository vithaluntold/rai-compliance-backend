from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

# Import routers
from routes import analysis_router, documents_router, sessions_router

# Initialize FastAPI app
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
        "http://localhost:3000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers - ONLY ROUTERS, NO HARDCODED ENDPOINTS
app.include_router(analysis_router, prefix="/api/v1/analysis", tags=["analysis"])
app.include_router(documents_router, prefix="/api/v1/documents", tags=["documents"])
app.include_router(sessions_router, prefix="/api/v1/sessions", tags=["sessions"])

# Root endpoint
@app.get("/")
async def root():
    return {"message": "RAi Compliance Engine API - Fully Functional", "status": "running"}

# Health endpoint  
@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "features": {
            "file_upload": True,
            "ai_analysis": True,
            "compliance_checking": True,
            "real_time_processing": True
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)