#!/bin/bash
set -e

echo "🚀 Starting BiRefNet Hello World Setup on GCP L4..."

# 1. System Dependencies (optional, usually pre-installed on DL images)
echo "📦 Checking system dependencies..."
if ! command -v python3 &> /dev/null; then
    echo "Python3 could not be found, attempting to install..."
    sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip
fi

# 2. Python Environment
if [ ! -d "venv" ]; then
    echo "🐍 Creating virtual environment..."
    python3 -m venv venv
fi

echo "🔌 Activating virtual environment..."
source venv/bin/activate

# 3. Install Python Packages
echo "⬇️ Installing Python dependencies..."
# Install torch with CUDA support (L4 needs CUDA 11.8 or 12.x)
# Using standard pip install which usually grabs the right wheel for linux
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers timm opencv-python scikit-image kornia einops accelerate pillow requests "numpy<2"

# 4. Run the Test
echo "🏃 Running hello_world.py..."
python hello_world.py

echo "✅ Done! Check hello_world_result.png for the output."
