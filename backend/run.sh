#!/bin/bash
# run.sh - Start the NCERT Exam Evaluator API

set -e

echo "========================================="
echo "NCERT Exam Evaluator API Startup"
echo "========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if .env exists
if [ ! -f ".env" ]; then
    echo ".env file not found. Creating from template..."
    cp .env.example .env
    echo "Please edit .env file with your configuration"
fi

# Install/update dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create storage directories
echo "Creating storage directories..."
mkdir -p storage/chromadb
mkdir -p storage/lora_adapters
mkdir -p storage/training_metadata
mkdir -p storage/logs
mkdir -p storage/uploaded_pdfs
mkdir -p storage/model_cache

# Check configuration
echo "Checking configuration..."
python -c "from config.settings import settings; print(f'Base Model: {settings.BASE_MODEL_NAME}')"

# Start the API
echo "========================================="
echo "Starting API server..."
echo "API will be available at: http://localhost:8001"
echo "API docs at: http://localhost:8001/docs"
echo "========================================="

uvicorn main:app --reload --host 0.0.0.0 --port 8001