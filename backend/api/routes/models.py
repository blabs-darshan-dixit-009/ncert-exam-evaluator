# api/routes/models.py

from fastapi import APIRouter, HTTPException
import logging
from pathlib import Path
import json
import shutil

from api.schemas import (
    ModelInfo,
    ModelListResponse,
    ModelDeleteRequest,
    ModelDeleteResponse,
    ChromaDBInfoResponse,
    ErrorResponse
)
from models.chromadb_handler import ChromaDBHandler
from config.settings import settings

router = APIRouter()
logger = logging.getLogger("exam_evaluator.models_routes")


@router.get(
    "/list",
    response_model=ModelListResponse,
    responses={500: {"model": ErrorResponse}}
)
async def list_models():
    """
    List all trained models with their metadata.
    
    Returns information about each trained model including:
    - Model name
    - Base model used
    - Training date
    - Number of training examples
    - Hyperparameters
    """
    logger.info("Listing all trained models")
    
    try:
        metadata_dir = Path(settings.TRAINING_METADATA_PATH)
        models = []
        
        if metadata_dir.exists():
            for metadata_file in metadata_dir.glob("*.json"):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    models.append(ModelInfo(
                        model_name=metadata["model_name"],
                        base_model=metadata["base_model"],
                        base_model_display=metadata["base_model_display"],
                        training_date=metadata["training_date"],
                        num_examples=metadata["num_examples"],
                        hyperparameters=metadata["hyperparameters"],
                        adapter_path=metadata["adapter_path"]
                    ))
                except Exception as e:
                    logger.warning(f"Failed to load metadata from {metadata_file}: {str(e)}")
                    continue
        
        logger.info(f"Found {len(models)} trained models")
        
        return ModelListResponse(
            models=models,
            total_models=len(models)
        )
        
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@router.get(
    "/info/{model_name}",
    response_model=ModelInfo,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
async def get_model_info(model_name: str):
    """
    Get detailed information about a specific trained model.
    """
    logger.info(f"Getting info for model: {model_name}")
    
    try:
        metadata_path = Path(settings.TRAINING_METADATA_PATH) / f"{model_name}.json"
        
        if not metadata_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found"
            )
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return ModelInfo(
            model_name=metadata["model_name"],
            base_model=metadata["base_model"],
            base_model_display=metadata["base_model_display"],
            training_date=metadata["training_date"],
            num_examples=metadata["num_examples"],
            hyperparameters=metadata["hyperparameters"],
            adapter_path=metadata["adapter_path"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get model info: {str(e)}"
        )


@router.delete(
    "/delete",
    response_model=ModelDeleteResponse,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}}
)
async def delete_model(request: ModelDeleteRequest):
    """
    Delete a trained model and all associated data.
    
    This will delete:
    - LoRA adapter weights
    - Training metadata
    - ChromaDB collection (if exists)
    
    Requires confirmation flag to be set to true.
    """
    logger.info(f"Delete request for model: {request.model_name}", extra={
        "model_name": request.model_name,
        "confirmed": request.confirm
    })
    
    if not request.confirm:
        raise HTTPException(
            status_code=400,
            detail="Deletion must be confirmed by setting 'confirm' to true"
        )
    
    try:
        adapter_path = Path(settings.LORA_ADAPTERS_PATH) / request.model_name
        metadata_path = Path(settings.TRAINING_METADATA_PATH) / f"{request.model_name}.json"
        
        # Check if model exists
        if not adapter_path.exists() and not metadata_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Model '{request.model_name}' not found"
            )
        
        # Delete adapter weights
        if adapter_path.exists():
            shutil.rmtree(adapter_path)
            logger.info(f"Deleted adapter weights: {adapter_path}")
        
        # Delete metadata
        if metadata_path.exists():
            metadata_path.unlink()
            logger.info(f"Deleted metadata: {metadata_path}")
        
        # Delete ChromaDB collection
        try:
            chromadb = ChromaDBHandler()
            chromadb.delete_collection(request.model_name)
            logger.info(f"Deleted ChromaDB collection for: {request.model_name}")
        except Exception as e:
            logger.warning(f"Failed to delete ChromaDB collection: {str(e)}")
        
        return ModelDeleteResponse(
            success=True,
            model_name=request.model_name,
            message=f"Model '{request.model_name}' deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete model: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete model: {str(e)}"
        )


@router.get(
    "/chromadb/info/{model_id}",
    response_model=ChromaDBInfoResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
async def get_chromadb_info(model_id: str):
    """
    Get information about the ChromaDB collection for a model.
    
    Returns document count and configuration details.
    """
    logger.info(f"Getting ChromaDB info for: {model_id}")
    
    try:
        chromadb = ChromaDBHandler()
        info = chromadb.get_collection_info(model_id)
        
        return ChromaDBInfoResponse(**info)
        
    except Exception as e:
        logger.error(f"Failed to get ChromaDB info: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get ChromaDB info: {str(e)}"
        )


@router.post(
    "/chromadb/reset/{model_id}",
    responses={500: {"model": ErrorResponse}}
)
async def reset_chromadb(model_id: str):
    """
    Reset (clear) the ChromaDB collection for a model.
    
    Use this to clear all indexed documents without deleting the model.
    You'll need to re-upload PDFs after resetting.
    """
    logger.info(f"Resetting ChromaDB for: {model_id}")
    
    try:
        chromadb = ChromaDBHandler()
        chromadb.delete_collection(model_id)
        chromadb.initialize(model_id)
        
        return {
            "success": True,
            "model_id": model_id,
            "message": f"ChromaDB collection reset for '{model_id}'"
        }
        
    except Exception as e:
        logger.error(f"Failed to reset ChromaDB: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset ChromaDB: {str(e)}"
        )


@router.get("/storage/stats")
async def get_storage_stats():
    """
    Get storage statistics for all directories.
    
    Shows disk space usage for models, ChromaDB, logs, etc.
    """
    logger.info("Getting storage statistics")
    
    try:
        def get_dir_size(path: Path) -> int:
            """Calculate total size of directory in bytes"""
            if not path.exists():
                return 0
            total = 0
            for item in path.rglob("*"):
                if item.is_file():
                    total += item.stat().st_size
            return total
        
        stats = {
            "lora_adapters": {
                "path": settings.LORA_ADAPTERS_PATH,
                "size_mb": round(get_dir_size(Path(settings.LORA_ADAPTERS_PATH)) / (1024 * 1024), 2)
            },
            "chromadb": {
                "path": settings.CHROMADB_PATH,
                "size_mb": round(get_dir_size(Path(settings.CHROMADB_PATH)) / (1024 * 1024), 2)
            },
            "logs": {
                "path": settings.LOGS_PATH,
                "size_mb": round(get_dir_size(Path(settings.LOGS_PATH)) / (1024 * 1024), 2)
            },
            "uploaded_pdfs": {
                "path": settings.UPLOADED_PDFS_PATH,
                "size_mb": round(get_dir_size(Path(settings.UPLOADED_PDFS_PATH)) / (1024 * 1024), 2)
            },
            "model_cache": {
                "path": "./storage/model_cache",
                "size_mb": round(get_dir_size(Path("./storage/model_cache")) / (1024 * 1024), 2)
            }
        }
        
        total_mb = sum(s["size_mb"] for s in stats.values())
        
        return {
            "directories": stats,
            "total_size_mb": round(total_mb, 2),
            "total_size_gb": round(total_mb / 1024, 2)
        }
        
    except Exception as e:
        logger.error(f"Failed to get storage stats: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get storage stats: {str(e)}"
        )