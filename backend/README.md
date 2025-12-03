# NCERT Exam Evaluator API

A FastAPI-based system for training LoRA-adapted language models to evaluate exam answers using RAG (Retrieval Augmented Generation).

## Features

- ✅ **Configurable Base Models** - Easy switching between GPT-2, DistilGPT-2, Llama 3.1, etc.
- ✅ **LoRA Fine-tuning** - Efficient parameter-efficient training
- ✅ **RAG with ChromaDB** - Context retrieval from PDF documents
- ✅ **REST API** - Complete FastAPI endpoints for training and evaluation
- ✅ **Automatic Model Download** - Models downloaded from HuggingFace on first use
- ✅ **Persistent Storage** - All models and data saved locally
- ✅ **Batch Evaluation** - Process multiple questions efficiently

---

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Quick Start](#quick-start)
4. [API Endpoints](#api-endpoints)
5. [Changing Base Model](#changing-base-model)
6. [Project Structure](#project-structure)
7. [Usage Examples](#usage-examples)

---

## Installation

### Prerequisites

- Python 3.9+
- pip
- (Optional) CUDA-capable GPU for faster training

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd ncert-exam-evaluator
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Create Environment File

```bash
cp .env.example .env
```

Edit `.env` to customize settings (optional).

### Step 5: Create Storage Directories

Storage directories are created automatically on first run, but you can create them manually:

```bash
mkdir -p storage/chromadb storage/lora_adapters storage/training_metadata storage/logs storage/uploaded_pdfs storage/model_cache
```

---

## Configuration

All configuration is managed in `config/settings.py` and can be overridden via `.env` file.

### Key Configuration Options

**Base Model Selection:**
```python
BASE_MODEL_NAME = "gpt2"  # Change this to switch models
```

**Supported Models:**
- `gpt2` (124M parameters) - Fast, good for testing
- `distilgpt2` (82M parameters) - Faster, smaller
- `meta-llama/Llama-3.1-8B-Instruct` - Better quality, requires more resources

**Training Hyperparameters:**
```python
LORA_R = 16
LORA_ALPHA = 32
TRAINING_EPOCHS = 10
BATCH_SIZE = 2
LEARNING_RATE = 2e-4
```

**RAG Configuration:**
```python
CHUNK_SIZE = 400  # Words per chunk
CHUNK_OVERLAP = 50
TOP_K_RETRIEVAL = 3  # Number of chunks to retrieve
```

---

## Quick Start

### 1. Start the API Server

```bash
python main.py
```

Or with uvicorn directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Access API Documentation

Open your browser and go to:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

### 3. Check Health Status

```bash
curl http://localhost:8000/health
```

---

## API Endpoints

### Health & Info

- `GET /` - Root endpoint with API info
- `GET /health` - Health check with configuration details
- `GET /models/available` - List all available base models

### Training

- `POST /api/training/train` - Train a new model with examples
- `POST /api/training/train-with-pdf` - Train with PDF context
- `POST /api/training/upload-pdf` - Upload PDF for existing model

### Evaluation

- `POST /api/evaluation/evaluate` - Generate answer for single question
- `POST /api/evaluation/evaluate-batch` - Batch evaluation
- `POST /api/evaluation/evaluate-with-comparison` - Compare with ideal answer
- `DELETE /api/evaluation/cache/{model_name}` - Clear model from cache
- `DELETE /api/evaluation/cache` - Clear all cached models

### Model Management

- `GET /api/models/list` - List all trained models
- `GET /api/models/info/{model_name}` - Get model details
- `DELETE /api/models/delete` - Delete a trained model
- `GET /api/models/chromadb/info/{model_id}` - ChromaDB collection info
- `POST /api/models/chromadb/reset/{model_id}` - Reset ChromaDB collection
- `GET /api/models/storage/stats` - Storage usage statistics

---

## Changing Base Model

### Method 1: Edit .env File

```bash
# Open .env file
nano .env

# Change this line:
BASE_MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct

# Save and restart the application
python main.py
```

### Method 2: Edit config/settings.py

```python
# In config/settings.py, change:
BASE_MODEL_NAME: str = "meta-llama/Llama-3.1-8B-Instruct"
```

### First Run with New Model

When you switch to a new model:
1. The model will be automatically downloaded from HuggingFace
2. Download happens only once - model is cached in `storage/model_cache/`
3. Subsequent runs use the cached model (no re-download)

### Model-Specific Settings

Each model has its own configuration in `MODEL_CONFIGS`:
- Context length (max tokens)
- LoRA target modules
- Generation parameters (temperature, top_p, etc.)

These are automatically applied based on `BASE_MODEL_NAME`.

---

## Project Structure

```
ncert-exam-evaluator/
│
├── config/                     # Configuration management
│   ├── __init__.py
│   └── settings.py            # Centralized settings
│
├── models/                     # Core model logic
│   ├── __init__.py
│   ├── lora_trainer.py        # LoRA training
│   ├── chromadb_handler.py    # RAG with ChromaDB
│   └── inference.py           # Model inference
│
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── model_loader.py        # Model download/loading
│   └── pdf_processor.py       # PDF text extraction
│
├── api/                        # FastAPI routes
│   ├── __init__.py
│   ├── schemas.py             # Request/response models
│   └── routes/
│       ├── __init__.py
│       ├── training.py        # Training endpoints
│       ├── evaluation.py      # Evaluation endpoints
│       └── models.py          # Model management
│
├── storage/                    # Persistent data
│   ├── chromadb/              # Vector database
│   ├── lora_adapters/         # Trained LoRA weights
│   ├── training_metadata/     # Training info (JSON)
│   ├── logs/                  # Application logs
│   ├── uploaded_pdfs/         # Original PDFs
│   ├── model_cache/           # HuggingFace model cache
│   └── README.md
│
├── main.py                     # FastAPI application
├── requirements.txt            # Python dependencies
├── .env.example               # Environment template
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

---

## Usage Examples

### Example 1: Train a Model

```bash
curl -X POST "http://localhost:8000/api/training/train" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "biology_class_10",
    "training_examples": [
      {
        "question": "What is photosynthesis?",
        "ideal_answer": "Photosynthesis is the process by which plants convert light energy into chemical energy stored in glucose."
      },
      {
        "question": "What is cellular respiration?",
        "ideal_answer": "Cellular respiration is the process of breaking down glucose to produce ATP energy."
      }
    ]
  }'
```

### Example 2: Train with PDF Context

```bash
curl -X POST "http://localhost:8000/api/training/train-with-pdf" \
  -F "model_name=physics_class_11" \
  -F "training_examples=[{\"question\":\"What is Newton's first law?\",\"ideal_answer\":\"An object remains at rest or in uniform motion unless acted upon by force.\"}]" \
  -F "pdf_file=@textbook.pdf"
```

### Example 3: Evaluate a Question

```bash
curl -X POST "http://localhost:8000/api/evaluation/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "biology_class_10",
    "question": "Explain the process of photosynthesis in detail",
    "use_rag": true
  }'
```

### Example 4: Batch Evaluation

```bash
curl -X POST "http://localhost:8000/api/evaluation/evaluate-batch" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "biology_class_10",
    "questions": [
      "What is photosynthesis?",
      "What is cellular respiration?",
      "What are stomata?"
    ],
    "use_rag": true
  }'
```

### Example 5: List All Models

```bash
curl http://localhost:8000/api/models/list
```

### Example 6: Get Model Info

```bash
curl http://localhost:8000/api/models/info/biology_class_10
```

### Example 7: Delete a Model

```bash
curl -X DELETE "http://localhost:8000/api/models/delete" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "biology_class_10",
    "confirm": true
  }'
```

### Example 8: Check Storage Usage

```bash
curl http://localhost:8000/api/models/storage/stats
```

---

## Python Client Examples

### Training Example

```python
import requests

# Train a model
response = requests.post(
    "http://localhost:8000/api/training/train",
    json={
        "model_name": "chemistry_class_12",
        "training_examples": [
            {
                "question": "What is an ionic bond?",
                "ideal_answer": "An ionic bond is formed by transfer of electrons..."
            }
        ]
    }
)

print(response.json())
```

### Evaluation Example

```python
import requests

# Evaluate a question
response = requests.post(
    "http://localhost:8000/api/evaluation/evaluate",
    json={
        "model_name": "chemistry_class_12",
        "question": "Explain ionic bonding",
        "use_rag": True
    }
)

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Used RAG: {result['used_rag']}")
print(f"Context chunks: {result['num_context_chunks']}")
```

---

## Troubleshooting

### Model Download Fails

**Issue:** Model download from HuggingFace fails

**Solution:**
- Check internet connection
- Verify model name in `BASE_MODEL_NAME`
- Check HuggingFace Hub status
- For Llama models, ensure you have access (some require approval)

### Out of Memory

**Issue:** Training fails with CUDA out of memory

**Solution:**
- Reduce `BATCH_SIZE` in settings
- Use smaller model (gpt2 instead of Llama)
- Set `FP16_TRAINING=False` if on CPU
- Close other GPU applications

### ChromaDB Errors

**Issue:** ChromaDB collection errors

**Solution:**
- Delete and reset collection: `POST /api/models/chromadb/reset/{model_id}`
- Check `storage/chromadb/` permissions
- Ensure embedding model is downloaded

### PDF Text Extraction Fails

**Issue:** PDF upload succeeds but no text extracted

**Solution:**
- Check if PDF is image-based (requires OCR)
- Try different extraction method (pypdf2 vs pdfplumber)
- Verify PDF file is not corrupted
- Check PDF size is under `MAX_PDF_SIZE_MB`

---

## Performance Tips

1. **GPU Usage:** Enable GPU for 10-50x faster training
2. **Batch Size:** Increase for GPU, decrease for CPU
3. **Model Caching:** First load is slow, subsequent loads are fast
4. **RAG Performance:** Adjust `CHUNK_SIZE` and `TOP_K_RETRIEVAL` based on document size
5. **FP16 Training:** Enable for GPU to reduce memory usage

---

## Security Notes

- Do not commit `.env` file to version control
- Keep API keys and sensitive data in `.env`
- Use authentication middleware in production
- Limit file upload sizes
- Sanitize user inputs

---

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request

---

## License

[Add your license here]

---

## Support

For issues and questions:
- Create an issue on GitHub
- Check API documentation at `/docs`
- Review logs in `storage/logs/`

---

## Changelog

### Version 1.0.0
- Initial release
- LoRA fine-tuning support
- ChromaDB RAG integration
- Complete REST API
- Configurable base models