"""
Simple backend startup script for local development
This starts the backend with minimal Azure OpenAI dependencies for testing
"""
import uvicorn
import os

# Set minimal environment variables for local testing
os.environ.setdefault("AZURE_OPENAI_API_KEY", "test-key")
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://test.cognitiveservices.azure.com/")
os.environ.setdefault("AZURE_OPENAI_DEPLOYMENT_NAME", "model-router")

if __name__ == "__main__":
    print("🚀 Starting RAi Compliance Backend Server...")
    print("📍 URL: http://localhost:8000")
    print("📖 API Docs: http://localhost:8000/docs")
    print("❗ Note: Azure OpenAI features require proper API keys in .env file")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )