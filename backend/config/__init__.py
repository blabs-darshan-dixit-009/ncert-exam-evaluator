# config/__init__.py

"""
Configuration module for NCERT Exam Evaluator.
Provides centralized settings management.
"""

from config.settings import settings, get_model_config, MODEL_CONFIGS

__all__ = ["settings", "get_model_config", "MODEL_CONFIGS"]