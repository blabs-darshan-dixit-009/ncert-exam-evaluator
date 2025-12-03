# api/__init__.py

"""
API module for NCERT Exam Evaluator.
Contains FastAPI routes and request/response schemas.
"""

from api.routes import training_router, evaluation_router, models_router

__all__ = ["training_router", "evaluation_router", "models_router"]