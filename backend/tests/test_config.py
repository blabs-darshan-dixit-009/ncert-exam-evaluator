# tests/test_config.py

import pytest
from config.settings import settings, get_model_config, MODEL_CONFIGS


def test_settings_loaded():
    """Test that settings are loaded correctly"""
    assert settings.BASE_MODEL_NAME is not None
    assert settings.EMBEDDING_MODEL_NAME is not None
    assert settings.API_TITLE == "NCERT Exam Evaluator API"


def test_model_configs_exist():
    """Test that model configurations are defined"""
    assert "gpt2" in MODEL_CONFIGS
    assert "distilgpt2" in MODEL_CONFIGS
    assert "meta-llama/Llama-3.1-8B-Instruct" in MODEL_CONFIGS


def test_get_model_config():
    """Test getting model configuration"""
    # Should work for default model
    config = get_model_config()
    assert "display_name" in config
    assert "max_length" in config
    assert "lora_target_modules" in config
    assert "generation_config" in config


def test_invalid_model_config():
    """Test that invalid model name raises error"""
    # Temporarily change to invalid model
    original = settings.BASE_MODEL_NAME
    settings.BASE_MODEL_NAME = "invalid_model_name"
    
    with pytest.raises(ValueError):
        get_model_config()
    
    # Restore original
    settings.BASE_MODEL_NAME = original


def test_lora_hyperparameters():
    """Test LoRA hyperparameters are valid"""
    assert settings.LORA_R > 0
    assert settings.LORA_ALPHA > 0
    assert 0 < settings.LORA_DROPOUT < 1


def test_training_hyperparameters():
    """Test training hyperparameters are valid"""
    assert settings.TRAINING_EPOCHS > 0
    assert settings.BATCH_SIZE > 0
    assert settings.LEARNING_RATE > 0
    assert settings.MAX_LENGTH > 0


def test_storage_paths():
    """Test storage paths are defined"""
    assert settings.CHROMADB_PATH is not None
    assert settings.LORA_ADAPTERS_PATH is not None
    assert settings.LOGS_PATH is not None


def test_validation_limits():
    """Test validation limits are reasonable"""
    assert settings.MAX_PDF_SIZE_MB > 0
    assert settings.MIN_TRAINING_EXAMPLES >= 1
    assert settings.MAX_TRAINING_EXAMPLES > settings.MIN_TRAINING_EXAMPLES