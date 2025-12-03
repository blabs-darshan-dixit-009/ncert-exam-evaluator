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
            # Load base model (downloads from HuggingFace if not cached)
            self.model, self.tokenizer, self.model_config = download_and_load_base_model()
            
            logger.info(f"Base model loaded: {self.model_config['display_name']}")
            
            # Configure LoRA
            lora_config = LoraConfig(
                r=settings.LORA_R,
                lora_alpha=settings.LORA_ALPHA,
                target_modules=get_lora_target_modules(),  # Model-specific
                lora_dropout=settings.LORA_DROPOUT,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            # Apply LoRA to base model
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            
            # Prepare training dataset
            train_dataset = self.prepare_training_data(training_examples)
            
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
                logging_steps=10,
                save_strategy="epoch",
                save_total_limit=2,
                report_to="none",
                remove_unused_columns=False
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
            
            # Create trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=data_collator
            )
            
            # Train
            logger.info("Starting training...")
            train_result = trainer.train()
            
            # Save LoRA adapters
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            # Save training metadata
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
            
            logger.info(f"Training completed successfully", extra={
                "model_name": model_name,
                "training_loss": train_result.training_loss,
                "output_dir": str(output_dir)
            })
            
            return metadata
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
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