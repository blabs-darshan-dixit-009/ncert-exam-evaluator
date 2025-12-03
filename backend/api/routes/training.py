# api/routes/training.py

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List
import logging
import json
from pathlib import Path

from api.schemas import (
    TrainingRequest,
    TrainingResponse,
    PDFUploadResponse,
    ErrorResponse
)
from models.lora_trainer import LoRATrainer
from models.chromadb_handler import ChromaDBHandler
from utils.pdf_processor import PDFProcessor
from config.settings import settings

router = APIRouter()
logger = logging.getLogger("exam_evaluator.training_routes")


@router.post(
    "/train",
    response_model=TrainingResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
async def train_model(request: TrainingRequest):
    """
    Train a new LoRA model with provided examples.
    
    Steps:
    1. Validates training examples count
    2. Loads base model from HuggingFace (downloads if needed)
    3. Applies LoRA configuration
    4. Trains on provided examples
    5. Saves adapters and metadata
    
    Returns training metadata including loss and hyperparameters.
    """
    logger.info(f"Training request received", extra={
        "model_name": request.model_name,
        "num_examples": len(request.training_examples)
    })
    
    try:
        # Convert Pydantic models to dicts
        training_examples = [
            {
                "question": ex.question,
                "ideal_answer": ex.ideal_answer
            }
            for ex in request.training_examples
        ]
        
        # Initialize trainer
        trainer = LoRATrainer()
        
        # Train model
        metadata = trainer.train(
            training_examples=training_examples,
            model_name=request.model_name
        )
        
        return TrainingResponse(
            success=True,
            model_name=metadata["model_name"],
            training_date=metadata["training_date"],
            num_examples=metadata["num_examples"],
            training_loss=metadata["training_loss"],
            base_model=metadata["base_model"],
            hyperparameters=metadata["hyperparameters"],
            message=f"Model '{request.model_name}' trained successfully"
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.post(
    "/train-with-pdf",
    response_model=TrainingResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
async def train_model_with_pdf(
    model_name: str = Form(...),
    training_examples: str = Form(...),  # JSON string
    pdf_file: UploadFile = File(...)
):
    """
    Train a model with PDF context and training examples.
    
    Steps:
    1. Uploads and validates PDF
    2. Extracts text from PDF
    3. Stores in ChromaDB for RAG
    4. Trains LoRA model with examples
    
    The PDF content will be used for context retrieval during inference.
    """
    logger.info(f"Training with PDF request received", extra={
        "model_name": model_name,
        "pdf_filename": pdf_file.filename
    })
    
    try:
        # Parse training examples from JSON string
        examples_data = json.loads(training_examples)
        training_examples_list = [
            {
                "question": ex["question"],
                "ideal_answer": ex["ideal_answer"]
            }
            for ex in examples_data
        ]
        
        # Validate count
        if len(training_examples_list) < settings.MIN_TRAINING_EXAMPLES:
            raise ValueError(
                f"Minimum {settings.MIN_TRAINING_EXAMPLES} examples required"
            )
        
        # Save PDF file
        pdf_content = await pdf_file.read()
        pdf_path = PDFProcessor.save_pdf(pdf_content, pdf_file.filename)
        
        # Extract text from PDF
        pdf_text = PDFProcessor.extract_text(pdf_path)
        
        # Initialize ChromaDB and add document
        chromadb = ChromaDBHandler()
        chromadb.add_document(pdf_text, model_name)
        
        # Train model
        trainer = LoRATrainer()
        metadata = trainer.train(
            training_examples=training_examples_list,
            model_name=model_name,
            pdf_text=pdf_text
        )
        
        return TrainingResponse(
            success=True,
            model_name=metadata["model_name"],
            training_date=metadata["training_date"],
            num_examples=metadata["num_examples"],
            training_loss=metadata["training_loss"],
            base_model=metadata["base_model"],
            hyperparameters=metadata["hyperparameters"],
            message=f"Model '{model_name}' trained with PDF context"
        )
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in training_examples: {str(e)}")
        raise HTTPException(
            status_code=400, 
            detail="Invalid JSON format in training_examples"
        )
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Training with PDF failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Training with PDF failed: {str(e)}"
        )


@router.post(
    "/upload-pdf",
    response_model=PDFUploadResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
async def upload_pdf(
    model_id: str = Form(...),
    pdf_file: UploadFile = File(...)
):
    """
    Upload PDF and add to ChromaDB for a specific model.
    
    Use this endpoint to add context documents to an existing model
    without retraining.
    """
    logger.info(f"PDF upload request", extra={
        "model_id": model_id,
        "filename": pdf_file.filename
    })
    
    try:
        # Save PDF
        pdf_content = await pdf_file.read()
        pdf_path = PDFProcessor.save_pdf(pdf_content, pdf_file.filename)
        
        # Extract text
        pdf_text = PDFProcessor.extract_text(pdf_path)
        word_count = len(pdf_text.split())
        
        # Add to ChromaDB
        chromadb = ChromaDBHandler()
        chromadb.add_document(pdf_text, model_id)
        
        return PDFUploadResponse(
            success=True,
            filename=pdf_file.filename,
            file_path=str(pdf_path),
            text_length=len(pdf_text),
            word_count=word_count,
            message=f"PDF uploaded and indexed for model '{model_id}'"
        )
        
    except ValueError as e:
        logger.error(f"PDF validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"PDF upload failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PDF upload failed: {str(e)}")