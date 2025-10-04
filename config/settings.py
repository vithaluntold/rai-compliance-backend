import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = BASE_DIR / "uploads"
CACHE_DIR = BASE_DIR / "cache"
LOG_DIR = BASE_DIR / "logs"

# Create necessary directories
for directory in [UPLOADS_DIR, CACHE_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API Configuration
API_V1_PREFIX = "/api/v1"
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
]

# Database Configuration
DB_PATH = BASE_DIR / "database.duckdb"

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")

# FAISS Configuration
FAISS_INDEX_PATH = CACHE_DIR / "faiss_index"
USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOG_DIR / "app.log"

# Document Processing
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {".pdf"}
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# WebSocket Configuration
WS_HEARTBEAT_INTERVAL = 30  # seconds
WS_PING_INTERVAL = 20  # seconds

# Cache Configuration
CACHE_TTL = 3600  # 1 hour
MAX_CACHE_SIZE = 1024 * 1024 * 1024  # 1GB

# Security
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
