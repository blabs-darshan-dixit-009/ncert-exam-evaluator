# utils/training_progress.py

import logging
from datetime import datetime
from typing import Optional, Dict
import json
from pathlib import Path
from config.settings import settings

logger = logging.getLogger("exam_evaluator.training_progress")


class TrainingProgressManager:
    """
    Manages training progress for background training jobs.
    Stores progress in JSON files for real-time monitoring.
    """
    
    def __init__(self):
        self.progress_file = Path(settings.TRAINING_METADATA_PATH) / "current_training.json"
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
    
    def start_training(self, model_name: str, total_examples: int, total_epochs: int):
        """Initialize training progress"""
        progress = {
            "model_name": model_name,
            "status": "initializing",
            "stage": "Preparing data",
            "current_epoch": 0,
            "total_epochs": total_epochs,
            "current_step": 0,
            "total_steps": 0,
            "training_loss": None,
            "progress_percentage": 0,
            "start_time": datetime.now().isoformat(),
            "estimated_time_remaining": None,
            "total_examples": total_examples,
            "message": "Initializing training...",
            "error": None
        }
        self._save_progress(progress)
        logger.info(f"Training started for model: {model_name}", extra={
            "model_name": model_name,
            "total_examples": total_examples,
            "total_epochs": total_epochs
        })
        return progress
    
    def update_stage(self, stage: str, message: str):
        """Update current training stage"""
        progress = self._load_progress()
        if progress:
            progress["stage"] = stage
            progress["message"] = message
            progress["status"] = "running"
            self._save_progress(progress)
            logger.info(f"Training stage: {stage}", extra={
                "stage": stage,
                "message": message
            })
    
    def update_epoch(self, current_epoch: int, total_epochs: int, loss: float = None):
        """Update epoch progress"""
        progress = self._load_progress()
        if progress:
            progress["current_epoch"] = current_epoch
            progress["total_epochs"] = total_epochs
            progress["status"] = "training"
            progress["stage"] = f"Training (Epoch {current_epoch}/{total_epochs})"
            
            if loss is not None:
                progress["training_loss"] = round(loss, 6)
                progress["message"] = f"Epoch {current_epoch}/{total_epochs} - Loss: {loss:.6f}"
            
            # Calculate progress percentage
            progress["progress_percentage"] = int((current_epoch / total_epochs) * 100)
            
            # Estimate time remaining
            if progress["start_time"] and current_epoch > 0:
                elapsed = (datetime.now() - datetime.fromisoformat(progress["start_time"])).total_seconds()
                time_per_epoch = elapsed / current_epoch
                remaining_epochs = total_epochs - current_epoch
                estimated_remaining = time_per_epoch * remaining_epochs
                progress["estimated_time_remaining"] = f"{int(estimated_remaining // 60)}m {int(estimated_remaining % 60)}s"
            
            self._save_progress(progress)
            logger.info(f"Epoch progress: {current_epoch}/{total_epochs}", extra={
                "current_epoch": current_epoch,
                "total_epochs": total_epochs,
                "loss": loss,
                "progress_percentage": progress["progress_percentage"]
            })
    
    def update_step(self, current_step: int, total_steps: int, loss: float = None):
        """Update step progress within epoch"""
        progress = self._load_progress()
        if progress:
            progress["current_step"] = current_step
            progress["total_steps"] = total_steps
            
            if loss is not None:
                progress["training_loss"] = round(loss, 6)
            
            # Update message
            step_pct = int((current_step / total_steps) * 100) if total_steps > 0 else 0
            progress["message"] = f"Epoch {progress['current_epoch']}/{progress['total_epochs']} - Step {current_step}/{total_steps} ({step_pct}%)"
            
            self._save_progress(progress)
            
            # Log every 10% progress
            if current_step % max(1, total_steps // 10) == 0:
                logger.info(f"Step progress: {current_step}/{total_steps}", extra={
                    "current_step": current_step,
                    "total_steps": total_steps,
                    "loss": loss
                })
    
    def complete_training(self, final_loss: float, metadata: Dict):
        """Mark training as complete"""
        progress = self._load_progress()
        if progress:
            progress["status"] = "completed"
            progress["stage"] = "Training completed"
            progress["progress_percentage"] = 100
            progress["training_loss"] = round(final_loss, 6)
            progress["message"] = f"Training completed successfully! Final loss: {final_loss:.6f}"
            progress["end_time"] = datetime.now().isoformat()
            progress["metadata"] = metadata
            
            # Calculate total training time
            if progress["start_time"]:
                elapsed = (datetime.now() - datetime.fromisoformat(progress["start_time"])).total_seconds()
                progress["total_training_time"] = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
            
            self._save_progress(progress)
            logger.info(f"Training completed", extra={
                "model_name": progress["model_name"],
                "final_loss": final_loss,
                "total_time": progress.get("total_training_time")
            })
    
    def fail_training(self, error_message: str):
        """Mark training as failed"""
        progress = self._load_progress()
        if progress:
            progress["status"] = "failed"
            progress["stage"] = "Training failed"
            progress["error"] = error_message
            progress["message"] = f"Training failed: {error_message}"
            progress["end_time"] = datetime.now().isoformat()
            self._save_progress(progress)
            logger.error(f"Training failed", extra={
                "model_name": progress["model_name"],
                "error": error_message
            })
    
    def get_progress(self) -> Optional[Dict]:
        """Get current training progress"""
        return self._load_progress()
    
    def clear_progress(self):
        """Clear training progress file"""
        if self.progress_file.exists():
            self.progress_file.unlink()
    
    def _save_progress(self, progress: Dict):
        """Save progress to file"""
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def _load_progress(self) -> Optional[Dict]:
        """Load progress from file"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load progress: {str(e)}")
                return None
        return None


# Global progress manager instance
progress_manager = TrainingProgressManager()