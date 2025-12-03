# api/routes/evaluation.py

from fastapi import APIRouter, HTTPException
import logging
from typing import List

from api.schemas import (
    EvaluationRequest,
    EvaluationResponse,
    BatchEvaluationRequest,
    BatchEvaluationResponse,
    ErrorResponse
)
from models.inference import ModelInference
from config.settings import settings

router = APIRouter()
logger = logging.getLogger("exam_evaluator.evaluation_routes")

# Cache loaded models to avoid reloading
loaded_models = {}


def get_or_load_model(model_name: str) -> ModelInference:
    """
    Get cached model or load it if not in cache.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        ModelInference: Loaded model inference instance
    """
    if model_name not in loaded_models:
        logger.info(f"Loading model into cache: {model_name}")
        inference = ModelInference()
        inference.load_model(model_name)
        loaded_models[model_name] = inference
    
    return loaded_models[model_name]


@router.post(
    "/evaluate",
    response_model=EvaluationResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
async def evaluate_question(request: EvaluationRequest):
    """
    Generate an answer for a single question using a trained model.
    
    Steps:
    1. Loads the specified trained model (cached after first load)
    2. Retrieves relevant context from ChromaDB if RAG enabled
    3. Generates answer using LoRA fine-tuned model
    
    Returns the generated answer with metadata.
    """
    logger.info(f"Evaluation request received", extra={
        "model_name": request.model_name,
        "question_length": len(request.question),
        "use_rag": request.use_rag
    })
    
    try:
        # Get or load model
        inference = get_or_load_model(request.model_name)
        
        # Generate answer
        result = inference.generate_answer(
            question=request.question,
            use_rag=request.use_rag
        )
        
        return EvaluationResponse(
            question=result["question"],
            answer=result["answer"],
            model_name=request.model_name,
            used_rag=result["used_rag"],
            num_context_chunks=result["num_context_chunks"],
            generation_config=result["generation_config"]
        )
        
    except ValueError as e:
        logger.error(f"Model not found: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@router.post(
    "/evaluate-batch",
    response_model=BatchEvaluationResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
async def evaluate_batch(request: BatchEvaluationRequest):
    """
    Generate answers for multiple questions in batch.
    
    More efficient than calling /evaluate multiple times as the model
    stays loaded in memory between questions.
    
    Maximum batch size is controlled by settings.MAX_BATCH_EVALUATION_SIZE.
    """
    logger.info(f"Batch evaluation request received", extra={
        "model_name": request.model_name,
        "num_questions": len(request.questions),
        "use_rag": request.use_rag
    })
    
    try:
        # Validate batch size
        if len(request.questions) > settings.MAX_BATCH_EVALUATION_SIZE:
            raise ValueError(
                f"Batch size {len(request.questions)} exceeds maximum "
                f"{settings.MAX_BATCH_EVALUATION_SIZE}"
            )
        
        # Get or load model
        inference = get_or_load_model(request.model_name)
        
        # Batch evaluate
        results = inference.batch_evaluate(
            questions=request.questions,
            use_rag=request.use_rag
        )
        
        # Count successful and failed
        successful = sum(1 for r in results if "error" not in r)
        failed = len(results) - successful
        
        return BatchEvaluationResponse(
            model_name=request.model_name,
            results=results,
            total_questions=len(results),
            successful=successful,
            failed=failed
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Batch evaluation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Batch evaluation failed: {str(e)}"
        )


@router.post(
    "/evaluate-with-comparison",
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
async def evaluate_with_comparison(
    model_name: str,
    question: str,
    ideal_answer: str,
    use_rag: bool = True
):
    """
    Generate answer and compare with ideal answer.
    
    Useful for testing model quality by comparing generated answers
    with known correct answers.
    
    Returns generated answer along with comparison metrics.
    """
    logger.info(f"Evaluation with comparison request", extra={
        "model_name": model_name,
        "question_length": len(question),
        "use_rag": use_rag
    })
    
    try:
        # Get or load model
        inference = get_or_load_model(model_name)
        
        # Generate and compare
        result = inference.compare_with_ideal(
            question=question,
            ideal_answer=ideal_answer,
            use_rag=use_rag
        )
        
        return result
        
    except ValueError as e:
        logger.error(f"Model not found: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Evaluation with comparison failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Evaluation with comparison failed: {str(e)}"
        )


@router.delete("/cache/{model_name}")
async def clear_model_cache(model_name: str):
    """
    Remove a model from the memory cache.
    
    Use this to free up memory or force reload of a model.
    """
    if model_name in loaded_models:
        del loaded_models[model_name]
        logger.info(f"Model removed from cache: {model_name}")
        return {
            "success": True,
            "message": f"Model '{model_name}' removed from cache"
        }
    else:
        return {
            "success": False,
            "message": f"Model '{model_name}' not in cache"
        }


@router.delete("/cache")
async def clear_all_cache():
    """
    Clear all models from memory cache.
    
    Use this to free up memory when not actively using the API.
    """
    num_models = len(loaded_models)
    loaded_models.clear()
    logger.info(f"All models cleared from cache ({num_models} models)")
    
    return {
        "success": True,
        "message": f"Cleared {num_models} models from cache"
    }


@router.get("/cache/status")
async def get_cache_status():
    """
    Get information about currently cached models.
    """
    return {
        "cached_models": list(loaded_models.keys()),
        "num_cached": len(loaded_models)
    }