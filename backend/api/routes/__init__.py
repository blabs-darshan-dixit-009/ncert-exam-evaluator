# api/routes/__init__.py

"""
API Routes for NCERT Exam Evaluator

This module exports all API routers for easy inclusion in main.py
"""

from api.routes.training import router as training_router
from api.routes.evaluation import router as evaluation_router
from api.routes.models import router as models_router

__all__ = [
    "training_router",
    "evaluation_router",
    "models_router"
]