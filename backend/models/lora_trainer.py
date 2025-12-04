# models/lora_trainer.py

import logging
from pathlib import Path
from typing import List, Dict
import json
from datetime import datetime
import torch
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset

from utils.model_loader import (
    download_and_load_base_model,
    get_lora_target_modules,
    get_generation_config
)
from config.settings import settings
from utils.training_progress import progress_manager
from models.training_callback import ProgressTrackingCallback

logger = logging.getLogger("exam_evaluator.lora_trainer")


class LoRATrainer:
    """
    Handles LoRA fine-tuning of base models.
    Uses configuration from settings.py for all hyperparameters.
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_config = None
        self.training_metadata = {}
    
    
    def prepare_training_data(
        self, 
        training_examples: List[Dict[str, str]]
    ) -> Dataset:
        """
        Prepare training examples into tokenized dataset.
        
        Args:
            training_examples: List of dicts with 'question' and 'ideal_answer' keys
            
        Returns:
            Dataset: Tokenized dataset ready for training
        """
        logger.info(f"Preparing {len(training_examples)} training examples")
        
        # Format examples as instruction-following prompts
        formatted_texts = []
        for example in training_examples:
            prompt = f"""Question: {example['question']}

Answer: {example['ideal_answer']}"""
            formatted_texts.append(prompt)
        
        # Tokenize
        tokenized = self.tokenizer(
            formatted_texts,
            truncation=True,
            max_length=settings.MAX_LENGTH,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Create dataset
        dataset = Dataset.from_dict({
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"]
        })
        
        logger.info(f"Dataset prepared with {len(dataset)} examples")
        return dataset
    
    
    def train(
        self,
        training_examples: List[Dict[str, str]],
        model_name: str,
        pdf_text: str = None
    ) -> Dict:
        """
        Train LoRA adapters on the base model.
        
        Args:
            training_examples: List of training Q&A pairs
            model_name: Name to save the trained model under
            pdf_text: Optional PDF text for metadata
            
        Returns:
            Dict: Training results and metadata
        """
        logger.info(f"Starting LoRA training for model: {model_name}", extra={
            "model_name": model_name,
            "num_examples": len(training_examples),
            "base_model": settings.BASE_MODEL_NAME
        })
        
        # Validate training examples count
        if len(training_examples) < settings.MIN_TRAINING_EXAMPLES:
            raise ValueError(
                f"Minimum {settings.MIN_TRAINING_EXAMPLES} training examples required, "
                f"got {len(training_examples)}"
            )
        
        if len(training_examples) > settings.MAX_TRAINING_EXAMPLES:
            raise ValueError(
                f"Maximum {settings.MAX_TRAINING_EXAMPLES} training examples allowed, "
                f"got {len(training_examples)}"
            )
        
        try:
            # Initialize progress tracking
            print("\n" + "="*80)
            print(f"🎯 INITIALIZING TRAINING FOR: {model_name}")
            print("="*80)
            
            progress_manager.start_training(
                model_name=model_name,
                total_examples=len(training_examples),
                total_epochs=settings.TRAINING_EPOCHS
            )
            
            # Load base model (downloads from HuggingFace if not cached)
            print(f"\n📥 Loading base model: {settings.BASE_MODEL_NAME}...")
            progress_manager.update_stage("Loading base model", f"Loading {settings.BASE_MODEL_NAME} from HuggingFace...")
            self.model, self.tokenizer, self.model_config = download_and_load_base_model()
            print(f"✓ Model loaded: {self.model_config['display_name']}")
            
            logger.info(f"Base model loaded: {self.model_config['display_name']}")
            
            # Configure LoRA (CPU-compatible, no bitsandbytes)
            print(f"\n⚙️  Configuring LoRA adapters...")
            print(f"   • LoRA rank (r): {settings.LORA_R}")
            print(f"   • LoRA alpha: {settings.LORA_ALPHA}")
            print(f"   • Dropout: {settings.LORA_DROPOUT}")
            
            progress_manager.update_stage("Configuring LoRA", "Setting up LoRA adapters...")
            lora_config = LoraConfig(
                r=settings.LORA_R,
                lora_alpha=settings.LORA_ALPHA,
                target_modules=get_lora_target_modules(),  # Model-specific
                lora_dropout=settings.LORA_DROPOUT,
                bias="none",
                task_type="CAUSAL_LM",
                inference_mode=False
            )
            
            # Apply LoRA to base model
            print(f"\n🔧 Applying LoRA to base model...")
            progress_manager.update_stage("Applying LoRA", "Injecting LoRA adapters into base model...")
            self.model = get_peft_model(self.model, lora_config)
            
            print("\n📊 Trainable Parameters:")
            self.model.print_trainable_parameters()
            
            # Prepare training dataset
            print(f"\n📝 Preparing training data ({len(training_examples)} examples)...")
            progress_manager.update_stage("Preparing data", "Tokenizing training examples...")
            train_dataset = self.prepare_training_data(training_examples)
            print(f"✓ Dataset prepared with {len(train_dataset)} tokenized examples")
            
            # Setup output directory
            output_dir = Path(settings.LORA_ADAPTERS_PATH) / model_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Configure training arguments
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                num_train_epochs=settings.TRAINING_EPOCHS,
                per_device_train_batch_size=settings.BATCH_SIZE,
                learning_rate=settings.LEARNING_RATE,
                fp16=settings.FP16_TRAINING and torch.cuda.is_available(),
                logging_steps=1,  # Log every step for real-time feedback
                logging_first_step=True,
                save_strategy="epoch",
                save_total_limit=2,
                report_to="none",
                remove_unused_columns=False,
                disable_tqdm=False,  # Enable progress bars
                logging_dir=str(Path(settings.LOGS_PATH) / "training_logs"),
                log_level="info",  # Set log level to info
                logging_strategy="steps"  # Log at each step
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
            
            # Create trainer
            progress_manager.update_stage("Initializing trainer", "Setting up training pipeline...")
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=data_collator,
                callbacks=[ProgressTrackingCallback()]  # Add progress tracking
            )
            
            # Train
            logger.info("Starting training...")
            progress_manager.update_stage("Training", "Training model with LoRA...")
            train_result = trainer.train()
            
            # Save LoRA adapters
            print(f"\n💾 Saving LoRA adapters...")
            progress_manager.update_stage("Saving model", "Saving LoRA adapters and tokenizer...")
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            print(f"✓ Model saved to: {output_dir}")
            
            # Save training metadata
            print(f"\n📋 Saving training metadata...")
            progress_manager.update_stage("Saving metadata", "Saving training metadata...")
            metadata = {
                "model_name": model_name,
                "base_model": settings.BASE_MODEL_NAME,
                "base_model_display": self.model_config["display_name"],
                "training_date": datetime.now().isoformat(),
                "num_examples": len(training_examples),
                "hyperparameters": {
                    "lora_r": settings.LORA_R,
                    "lora_alpha": settings.LORA_ALPHA,
                    "lora_dropout": settings.LORA_DROPOUT,
                    "epochs": settings.TRAINING_EPOCHS,
                    "batch_size": settings.BATCH_SIZE,
                    "learning_rate": settings.LEARNING_RATE,
                    "max_length": settings.MAX_LENGTH
                },
                "training_loss": train_result.training_loss,
                "adapter_path": str(output_dir)
            }
            
            # Save metadata to JSON
            metadata_path = Path(settings.TRAINING_METADATA_PATH) / f"{model_name}.json"
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"✓ Metadata saved to: {metadata_path}")
            
            print("\n" + "="*80)
            print(f"🎉 TRAINING COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(f"📊 Final Training Loss: {train_result.training_loss:.6f}")
            print(f"📁 Model Location: {output_dir}")
            print(f"📋 Metadata: {metadata_path}")
            print("="*80 + "\n")
            
            logger.info(f"Training completed successfully", extra={
                "model_name": model_name,
                "training_loss": train_result.training_loss,
                "output_dir": str(output_dir)
            })
            
            # Mark training as complete
            progress_manager.complete_training(train_result.training_loss, metadata)
            
            return metadata
            
        except Exception as e:
            print(f"\n❌ TRAINING FAILED!")
            print(f"Error: {str(e)}\n")
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            progress_manager.fail_training(str(e))
            raise
    
    
    def load_trained_model(self, model_name: str):
        """
        Load a previously trained LoRA model.
        
        Args:
            model_name: Name of the trained model to load
        """
        adapter_path = Path(settings.LORA_ADAPTERS_PATH) / model_name
        
        if not adapter_path.exists():
            raise ValueError(f"Trained model not found: {model_name}")
        
        logger.info(f"Loading trained model: {model_name}", extra={
            "model_name": model_name,
            "adapter_path": str(adapter_path)
        })
        
        # Load base model
        base_model, tokenizer, model_config = download_and_load_base_model()
        
        # Load LoRA adapters
        self.model = PeftModel.from_pretrained(base_model, str(adapter_path))
        self.tokenizer = tokenizer
        self.model_config = model_config
        
        logger.info(f"Model loaded successfully: {model_name}")
        
        return self.model, self.tokenizer