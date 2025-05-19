#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define the directory where the model will be stored on Render
MODEL_DOWNLOAD_DIR="weights"
mkdir -p $MODEL_DOWNLOAD_DIR

# --- Download the model file ---
# Ensure the Google Drive file linked here is shared publicly or with "Anyone with the link".
# This is the direct download URL derived from your latest link:
MODEL_URL="https://drive.google.com/uc?export=download&id=1B8jbPx4_chCpMtbYd65JNNf3Tu6r1Lvo"

echo "Downloading model from $MODEL_URL to $MODEL_DOWNLOAD_DIR/best.pt"
# Use curl with -L to follow redirects, which is often necessary for cloud storage links
curl -L "$MODEL_URL" -o "$MODEL_DOWNLOAD_DIR/best.pt"

# --- Install Python dependencies ---
echo "Installing Python dependencies from requirements.txt"
pip install -r requirements.txt

echo "Build process completed."