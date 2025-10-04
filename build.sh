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

echo "Build script completed successfully!"