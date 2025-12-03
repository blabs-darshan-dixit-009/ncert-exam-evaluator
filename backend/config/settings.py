# config/settings.py

from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    """
    Centralized configuration for the exam evaluator system.
    All model names and hyperparameters are defined here.
    Change model by updating BASE_MODEL_NAME and restarting the app.
    """
    
    # ============ MODEL CONFIGURATION ============
    # Base model from Hugging Face (change this to switch models)
    BASE_MODEL_NAME: str = "gpt2"  # Options: "gpt2", "distilgpt2", "meta-llama/Llama-3.1-8B-Instruct"
    
    # Embedding model for ChromaDB RAG
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # ============ LoRA HYPERPARAMETERS ============
    LORA_R: int = 16
    LORA_ALPHA: int = 32
    LORA_DROPOUT: float = 0.05
    LORA_TARGET_MODULES: list = ["c_attn"]  # For GPT-2. Change to ["q_proj", "v_proj"] for Llama
    
    # ============ TRAINING CONFIGURATION ============
    TRAINING_EPOCHS: int = 10
    BATCH_SIZE: int = 2
    LEARNING_RATE: float = 2e-4
    MAX_LENGTH: int = 1024  # Token limit for GPT-2. Use 2048+ for Llama
    FP16_TRAINING: bool = True  # Use mixed precision (faster on GPU)
    
    # ============ CHROMADB CONFIGURATION ============
    CHUNK_SIZE: int = 400  # Words per chunk
    CHUNK_OVERLAP: int = 50  # Overlap between chunks
    TOP_K_RETRIEVAL: int = 3  # Number of relevant chunks to retrieve
    
    # ============ STORAGE PATHS ============
    CHROMADB_PATH: str = "./storage/chromadb"
    LORA_ADAPTERS_PATH: str = "./storage/lora_adapters"
    TRAINING_METADATA_PATH: str = "./storage/training_metadata"
    LOGS_PATH: str = "./storage/logs"
    UPLOADED_PDFS_PATH: str = "./storage/uploaded_pdfs"
    
    # ============ API CONFIGURATION ============
    API_TITLE: str = "NCERT Exam Evaluator API"
    API_VERSION: str = "1.0.0"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8001  # Changed from 8000 to avoid conflict with tata-backend
    
    # ============ VALIDATION LIMITS ============
    MAX_PDF_SIZE_MB: int = 50
    MIN_TRAINING_EXAMPLES: int = 5
    MAX_TRAINING_EXAMPLES: int = 500
    MAX_BATCH_EVALUATION_SIZE: int = 1000
    MODEL_NAME_MAX_LENGTH: int = 50
    
    # ============ PERFORMANCE SETTINGS ============
    INFERENCE_TIMEOUT_SECONDS: int = 30  # Per question
    MAX_GENERATION_LENGTH: int = 300  # Max tokens for model output
    
    # ============ LOGGING CONFIGURATION ============
    LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_ROTATION_DAYS: int = 30
    LOG_MAX_SIZE_MB: int = 100
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


# ============ MODEL CONFIGURATION MAPPING ============
# Define model-specific configurations here
MODEL_CONFIGS = {
    "gpt2": {
        "display_name": "GPT-2 (124M)",
        "max_length": 1024,
        "lora_target_modules": ["c_attn"],
        "generation_config": {
            "max_new_tokens": 300,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }
    },
    "distilgpt2": {
        "display_name": "DistilGPT-2 (82M)",
        "max_length": 1024,
        "lora_target_modules": ["c_attn"],
        "generation_config": {
            "max_new_tokens": 300,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "display_name": "Llama 3.1 8B Instruct",
        "max_length": 8192,
        "lora_target_modules": ["q_proj", "v_proj"],
        "generation_config": {
            "max_new_tokens": 500,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }
    }
}


def get_model_config():
    """Get configuration for the currently selected base model"""
    model_name = settings.BASE_MODEL_NAME
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model '{model_name}' not configured. Available: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_name]