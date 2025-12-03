#!/bin/bash
# setup_project.sh - Create all necessary directories and empty files

set -e

echo "========================================="
echo "Setting up NCERT Exam Evaluator Project"
echo "========================================="

# Create directory structure
echo "Creating directories..."
mkdir -p config
mkdir -p models
mkdir -p utils
mkdir -p api/routes
mkdir -p tests
mkdir -p examples
mkdir -p storage/{chromadb,lora_adapters,training_metadata,logs,uploaded_pdfs,model_cache}

# Create __init__.py files
echo "Creating __init__.py files..."

cat > config/__init__.py << 'EOF'
# config/__init__.py
from config.settings import settings, get_model_config, MODEL_CONFIGS
__all__ = ["settings", "get_model_config", "MODEL_CONFIGS"]
EOF

cat > models/__init__.py << 'EOF'
# models/__init__.py
from models.lora_trainer import LoRATrainer
from models.chromadb_handler import ChromaDBHandler
from models.inference import ModelInference
__all__ = ["LoRATrainer", "ChromaDBHandler", "ModelInference"]
EOF

cat > utils/__init__.py << 'EOF'
# utils/__init__.py
from utils.model_loader import (
    download_and_load_base_model,
    download_and_load_embedding_model,
    get_lora_target_modules,
    get_generation_config
)
from utils.pdf_processor import PDFProcessor
__all__ = [
    "download_and_load_base_model",
    "download_and_load_embedding_model",
    "get_lora_target_modules",
    "get_generation_config",
    "PDFProcessor"
]
EOF

cat > api/__init__.py << 'EOF'
# api/__init__.py
from api.routes import training_router, evaluation_router, models_router
__all__ = ["training_router", "evaluation_router", "models_router"]
EOF

cat > api/routes/__init__.py << 'EOF'
# api/routes/__init__.py
from api.routes.training import router as training_router
from api.routes.evaluation import router as evaluation_router
from api.routes.models import router as models_router
__all__ = ["training_router", "evaluation_router", "models_router"]
EOF

cat > tests/__init__.py << 'EOF'
# tests/__init__.py
"""Test suite for NCERT Exam Evaluator API"""
EOF

cat > examples/__init__.py << 'EOF'
# examples/__init__.py
"""Example scripts for NCERT Exam Evaluator API"""
EOF

echo "========================================="
echo "✓ Directory structure created"
echo "✓ __init__.py files created"
echo ""
echo "Next steps:"
echo "1. Copy all Python files to their respective directories"
echo "2. Verify with: ls -la utils/ models/ api/routes/"
echo "3. Run: python main.py"
echo "========================================="