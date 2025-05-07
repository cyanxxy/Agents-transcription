#!/usr/bin/env bash
# Exit on error
set -o errexit

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
