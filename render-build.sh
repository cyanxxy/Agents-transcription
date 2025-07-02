#!/usr/bin/env bash
# Exit on error
set -o errexit

echo "Starting build process..."

# Print Python version
python --version

# Print ffmpeg version (should be pre-installed in Render's environment)
echo "Checking for ffmpeg..."
which ffmpeg || echo "ffmpeg not found in PATH"
ffmpeg -version || echo "Could not get ffmpeg version"

# Upgrade pip and install dependencies
echo "Upgrading pip..."
pip install --upgrade pip

echo "-----> Installing Python dependencies from pyproject.toml..."
# Add -vvv for verbose output from pip
pip install . --no-cache-dir -vvv
echo "-----> Python dependencies installation attempt complete."

echo "-----> Checking google-generativeai installation..."
pip show google-generativeai

echo "-----> Attempting to import google.generativeai in build script..."
python -c "import google.generativeai as genai; print('SUCCESS: google.generativeai imported correctly in build script')" || echo "FAILURE: google.generativeai import failed in build script"

echo "-----> Build process finished."
