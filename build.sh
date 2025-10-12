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

# Download NLTK data for NER (lightweight alternative)
echo "Installing NLTK data for NER..."
python -c "
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True) 
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
    print('✅ NLTK data downloaded successfully')
except Exception as e:
    print(f'⚠️ NLTK data download failed: {e}')
" || echo "Warning: NLTK data download failed, will download on first use"

# Run deployment verification
echo "Running deployment verification..."
python verify_deployment.py || echo "Warning: Verification issues detected, but continuing deployment"

echo "Build script completed successfully!"