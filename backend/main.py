# main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from config.settings import settings, get_model_config
import logging
from pathlib import Path

# Configure logging from settings
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(settings.LOGS_PATH) / "app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("exam_evaluator.main")

# Create FastAPI app with configuration
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="NCERT Exam Evaluator with LoRA fine-tuning and RAG"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create storage directories on startup
@app.on_event("startup")
async def create_storage_directories():
    """Create all required storage directories if they don't exist"""
    directories = [
        settings.CHROMADB_PATH,
        settings.LORA_ADAPTERS_PATH,
        settings.TRAINING_METADATA_PATH,
        settings.LOGS_PATH,
        settings.UPLOADED_PDFS_PATH,
        "./storage/model_cache"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info("Storage directories initialized", extra={
        "directories": directories
    })


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": settings.API_TITLE,
        "version": settings.API_VERSION,
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint with current model configuration"""
    try:
        model_config = get_model_config()
        
        return {
            "status": "healthy",
            "service": settings.API_TITLE,
            "version": settings.API_VERSION,
            "configuration": {
                "base_model": settings.BASE_MODEL_NAME,
                "base_model_display": model_config["display_name"],
                "embedding_model": settings.EMBEDDING_MODEL_NAME,
                "max_context_length": model_config["max_length"],
                "lora_config": {
                    "r": settings.LORA_R,
                    "alpha": settings.LORA_ALPHA,
                    "dropout": settings.LORA_DROPOUT
                },
                "training_config": {
                    "epochs": settings.TRAINING_EPOCHS,
                    "batch_size": settings.BATCH_SIZE,
                    "learning_rate": settings.LEARNING_RATE
                }
            },
            "storage": {
                "chromadb": settings.CHROMADB_PATH,
                "lora_adapters": settings.LORA_ADAPTERS_PATH,
                "logs": settings.LOGS_PATH
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.get("/models/available")
async def list_available_models():
    """List all available base models that can be configured"""
    from config.settings import MODEL_CONFIGS
    
    return {
        "current_model": settings.BASE_MODEL_NAME,
        "available_models": [
            {
                "model_id": model_id,
                "display_name": config["display_name"],
                "max_length": config["max_length"],
                "lora_targets": config["lora_target_modules"]
            }
            for model_id, config in MODEL_CONFIGS.items()
        ],
        "note": "To change model, update BASE_MODEL_NAME in .env and restart the application"
    }


# Import and include routers
from api.routes import training_router, evaluation_router, models_router

app.include_router(training_router, prefix="/api/training", tags=["Training"])
app.include_router(evaluation_router, prefix="/api/evaluation", tags=["Evaluation"])
app.include_router(models_router, prefix="/api/models", tags=["Models"])


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting {settings.API_TITLE} v{settings.API_VERSION}", extra={
        "host": settings.API_HOST,
        "port": settings.API_PORT,
        "base_model": settings.BASE_MODEL_NAME
    })
    
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )