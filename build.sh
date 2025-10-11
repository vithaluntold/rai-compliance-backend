#!/bin/bash
set -e

echo "Starting custom build script..."

# Upgrade pip first
python -m pip install --upgrade pip

# Install psutil explicitly first
echo "Installing psutil..."
python -m pip install psutil==5.9.5

# Verify psutil installation
python -c "import psutil; print(f'psutil version: {psutil.__version__}')"

# Install remaining dependencies
echo "Installing remaining dependencies..."
pip install --no-cache-dir -r requirements.txt

# Download spaCy English model for NER
echo "Installing spaCy English model for NER..."
python -m spacy download en_core_web_sm || echo "Warning: spaCy model download failed, NER will be unavailable"

echo "Build script completed successfully!"