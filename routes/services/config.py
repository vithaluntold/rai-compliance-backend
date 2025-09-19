"""
Configuration module for analysis routes

Contains shared constants and configurations.
"""

from pathlib import Path

# Configured paths
BASE_DIR = Path(__file__).parent.parent
ANALYSIS_RESULTS_DIR = BASE_DIR / "analysis_results"
UPLOADS_DIR = BASE_DIR / "uploads"

# Ensure directories exist
ANALYSIS_RESULTS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)
