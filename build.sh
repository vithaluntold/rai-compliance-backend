#!/bin/bash
set -e

echo "=== FORCE Python Version Check ==="
echo "Current Python version:"
python --version
python3 --version || echo "python3 not available"

# Check if we're getting the wrong Python version
if python --version | grep -q "3.13"; then
    echo "ERROR: Python 3.13 detected, need Python 3.11!"
    echo "Available Python versions:"
    ls -la /usr/bin/python* 2>/dev/null || echo "No python binaries in /usr/bin/"
    
    # Try to find Python 3.11
    if command -v python3.11 &> /dev/null; then
        echo "Found python3.11, using it instead"
        alias python=python3.11
        alias python3=python3.11
    else
        echo "WARNING: Python 3.11 not found, continuing with available version"
    fi
fi

echo "=== Installing Requirements ==="
python -m pip install psutil==5.9.5

# Verify psutil installation
python -c "import psutil; print(f'psutil version: {psutil.__version__}')"

# Install remaining dependencies
echo "Installing remaining dependencies..."
pip install --no-cache-dir -r requirements.txt

echo "Build script completed successfully!"