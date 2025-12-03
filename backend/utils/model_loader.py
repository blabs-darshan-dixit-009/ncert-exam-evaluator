# utils/model_loader.py

import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from config.settings import settings, get_model_config
import torch

logger = logging.getLogger("exam_evaluator.model_loader")


def download_and_load_base_model():
    """
    Download (if not cached) and load the base model from Hugging Face.
    Model name is read from settings.BASE_MODEL_NAME.
    
    Returns:
        tuple: (model, tokenizer, model_config)
    """
    model_name = settings.BASE_MODEL_NAME
    model_config = get_model_config()
    
    logger.info(f"Loading base model: {model_name}", extra={
        "model_name": model_name,
        "display_name": model_config["display_name"]
    })
    
    try:
        # Download from Hugging Face Hub (caches locally after first download)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir="./storage/model_cache"  # Local cache directory
        )
        
        # Set pad token if not present (required for GPT-2)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir="./storage/model_cache",
            torch_dtype=torch.float16 if settings.FP16_TRAINING and torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        logger.info(f"Model loaded successfully", extra={
            "model_name": model_name,
            "parameters": sum(p.numel() for p in model.parameters()),
            "device": str(model.device) if hasattr(model, 'device') else "N/A"
        })
        
        return model, tokenizer, model_config
        
    except Exception as e:
        logger.error(f"Failed to load model: {model_name}", extra={
            "model_name": model_name,
            "error": str(e)
        }, exc_info=True)
        raise


def download_and_load_embedding_model():
    """
    Download (if not cached) and load the embedding model for ChromaDB.
    Model name is read from settings.EMBEDDING_MODEL_NAME.
    
    Returns:
        SentenceTransformer: Loaded embedding model
    """
    model_name = settings.EMBEDDING_MODEL_NAME
    
    logger.info(f"Loading embedding model: {model_name}", extra={
        "model_name": model_name
    })
    
    try:
        embedding_model = SentenceTransformer(
            model_name,
            cache_folder="./storage/model_cache"
        )
        
        logger.info(f"Embedding model loaded successfully", extra={
            "model_name": model_name,
            "embedding_dim": embedding_model.get_sentence_embedding_dimension()
        })
        
        return embedding_model
        
    except Exception as e:
        logger.error(f"Failed to load embedding model: {model_name}", extra={
            "model_name": model_name,
            "error": str(e)
        }, exc_info=True)
        raise


def get_lora_target_modules():
    """
    Get the correct LoRA target modules for the current base model.
    
    Returns:
        list: Target module names for LoRA
    """
    model_config = get_model_config()
    return model_config["lora_target_modules"]


def get_generation_config():
    """
    Get the generation configuration for the current base model.
    
    Returns:
        dict: Generation parameters
    """
    model_config = get_model_config()
    return model_config["generation_config"]