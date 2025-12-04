# utils/background_training.py

import logging
import threading
from typing import List, Dict, Optional
from models.lora_trainer import LoRATrainer
from utils.training_progress import progress_manager

logger = logging.getLogger("exam_evaluator.background_training")


class BackgroundTrainingManager:
    """
    Manages training in a background thread so API remains responsive.
    """
    
    def __init__(self):
        self.training_thread: Optional[threading.Thread] = None
        self.is_training = False
    
    def start_training(
        self,
        model_name: str,
        training_examples: List[Dict[str, str]],
        pdf_text: str = None
    ) -> bool:
        """
        Start training in a background thread.
        
        Args:
            model_name: Name for the trained model
            training_examples: List of training Q&A pairs
            pdf_text: Optional PDF text content
            
        Returns:
            bool: True if training started, False if already training
        """
        # Check if already training
        if self.is_training:
            logger.warning("Training already in progress")
            return False
        
        # Create and start background thread
        self.training_thread = threading.Thread(
            target=self._train_in_background,
            args=(model_name, training_examples, pdf_text),
            daemon=True,  # Thread will stop when main program exits
            name=f"Training-{model_name}"
        )
        
        self.is_training = True
        self.training_thread.start()
        
        logger.info(f"Background training started for model: {model_name}", extra={
            "model_name": model_name,
            "thread_id": self.training_thread.ident
        })
        
        return True
    
    def _train_in_background(
        self,
        model_name: str,
        training_examples: List[Dict[str, str]],
        pdf_text: str = None
    ):
        """
        Internal method that runs training in background thread.
        """
        try:
            logger.info(f"Starting background training thread for: {model_name}")
            
            # Initialize trainer
            trainer = LoRATrainer()
            
            # Train model
            metadata = trainer.train(
                training_examples=training_examples,
                model_name=model_name,
                pdf_text=pdf_text
            )
            
            logger.info(f"Background training completed successfully: {model_name}", extra={
                "model_name": model_name,
                "final_loss": metadata.get("training_loss")
            })
            
        except Exception as e:
            logger.error(f"Background training failed: {str(e)}", exc_info=True)
            progress_manager.fail_training(str(e))
            
        finally:
            self.is_training = False
            logger.info(f"Background training thread ended for: {model_name}")
    
    def get_status(self) -> Dict:
        """
        Get current training status.
        
        Returns:
            Dict: Status information
        """
        return {
            "is_training": self.is_training,
            "thread_alive": self.training_thread.is_alive() if self.training_thread else False,
            "thread_name": self.training_thread.name if self.training_thread else None
        }
    
    def is_training_active(self) -> bool:
        """Check if training is currently active"""
        return self.is_training and (
            self.training_thread is not None and self.training_thread.is_alive()
        )


# Global background training manager instance
background_trainer = BackgroundTrainingManager()