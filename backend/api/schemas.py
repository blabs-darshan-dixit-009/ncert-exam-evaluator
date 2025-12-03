# api/schemas.py

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
from datetime import datetime

from config.settings import settings


# ============ TRAINING SCHEMAS ============

class TrainingExample(BaseModel):
    """Single training example with question and ideal answer"""
    question: str = Field(..., min_length=10, max_length=1000)
    ideal_answer: str = Field(..., min_length=10, max_length=2000)
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is photosynthesis?",
                "ideal_answer": "Photosynthesis is the process by which plants convert light energy into chemical energy..."
            }
        }


class TrainingRequest(BaseModel):
    """Request to train a new LoRA model"""
    model_name: str = Field(
        ..., 
        min_length=3, 
        max_length=settings.MODEL_NAME_MAX_LENGTH,
        pattern="^[a-zA-Z0-9_-]+$"
    )
    training_examples: List[TrainingExample] = Field(
        ..., 
        min_items=settings.MIN_TRAINING_EXAMPLES,
        max_items=settings.MAX_TRAINING_EXAMPLES
    )
    
    @validator("model_name")
    def validate_model_name(cls, v):
        """Ensure model name is valid"""
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Model name must contain only letters, numbers, hyphens and underscores")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "biology_class_10",
                "training_examples": [
                    {
                        "question": "What is photosynthesis?",
                        "ideal_answer": "Photosynthesis is the process..."
                    }
                ]
            }
        }


class TrainingResponse(BaseModel):
    """Response after training completion"""
    success: bool
    model_name: str
    training_date: str
    num_examples: int
    training_loss: float
    base_model: str
    hyperparameters: Dict
    message: str


# ============ EVALUATION SCHEMAS ============

class EvaluationRequest(BaseModel):
    """Request to evaluate a question"""
    model_name: str = Field(..., min_length=3)
    question: str = Field(..., min_length=10, max_length=1000)
    use_rag: bool = Field(default=True)
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "biology_class_10",
                "question": "Explain the process of cellular respiration",
                "use_rag": True
            }
        }


class EvaluationResponse(BaseModel):
    """Response with generated answer"""
    question: str
    answer: str
    model_name: str
    used_rag: bool
    num_context_chunks: int
    generation_config: Dict


class BatchEvaluationRequest(BaseModel):
    """Request to evaluate multiple questions"""
    model_name: str = Field(..., min_length=3)
    questions: List[str] = Field(
        ..., 
        min_items=1,
        max_items=settings.MAX_BATCH_EVALUATION_SIZE
    )
    use_rag: bool = Field(default=True)
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "biology_class_10",
                "questions": [
                    "What is photosynthesis?",
                    "Explain cellular respiration"
                ],
                "use_rag": True
            }
        }


class BatchEvaluationResponse(BaseModel):
    """Response with multiple answers"""
    model_name: str
    results: List[Dict]
    total_questions: int
    successful: int
    failed: int


# ============ MODEL MANAGEMENT SCHEMAS ============

class ModelInfo(BaseModel):
    """Information about a trained model"""
    model_name: str
    base_model: str
    base_model_display: str
    training_date: str
    num_examples: int
    hyperparameters: Dict
    adapter_path: str


class ModelListResponse(BaseModel):
    """List of all trained models"""
    models: List[ModelInfo]
    total_models: int


class ModelDeleteRequest(BaseModel):
    """Request to delete a model"""
    model_name: str = Field(..., min_length=3)
    confirm: bool = Field(default=False)


class ModelDeleteResponse(BaseModel):
    """Response after model deletion"""
    success: bool
    model_name: str
    message: str


# ============ PDF UPLOAD SCHEMAS ============

class PDFUploadResponse(BaseModel):
    """Response after PDF upload"""
    success: bool
    filename: str
    file_path: str
    text_length: int
    word_count: int
    message: str


# ============ CHROMADB SCHEMAS ============

class ChromaDBInfoResponse(BaseModel):
    """Information about ChromaDB collection"""
    model_id: str
    collection_name: str
    document_count: int
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k_retrieval: int


# ============ HEALTH CHECK SCHEMAS ============

class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    version: str
    configuration: Dict
    storage: Dict


# ============ ERROR SCHEMAS ============

class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str
    detail: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "detail": "Model name must be at least 3 characters",
                "timestamp": "2024-01-01T12:00:00"
            }
        }