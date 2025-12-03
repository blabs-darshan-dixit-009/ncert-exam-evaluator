# models/inference.py

import logging
from typing import Dict, List
import torch
from transformers import pipeline

from models.lora_trainer import LoRATrainer
from models.chromadb_handler import ChromaDBHandler
from config.settings import settings
from utils.model_loader import get_generation_config

logger = logging.getLogger("exam_evaluator.inference")


class ModelInference:
    """
    Handles inference with trained LoRA models.
    Uses ChromaDB for RAG to provide context.
    """
    
    def __init__(self):
        self.trainer = LoRATrainer()
        self.chromadb = ChromaDBHandler()
        self.model = None
        self.tokenizer = None
        self.generation_config = None
    
    
    def load_model(self, model_name: str):
        """
        Load a trained LoRA model for inference.
        
        Args:
            model_name: Name of the trained model
        """
        logger.info(f"Loading model for inference: {model_name}", extra={
            "model_name": model_name
        })
        
        try:
            # Load trained model
            self.model, self.tokenizer = self.trainer.load_trained_model(model_name)
            
            # Initialize ChromaDB
            self.chromadb.initialize(model_name)
            
            # Get generation config from settings
            self.generation_config = get_generation_config()
            
            logger.info(f"Model loaded and ready for inference", extra={
                "model_name": model_name,
                "generation_config": self.generation_config
            })
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            raise
    
    
    def generate_answer(
        self, 
        question: str, 
        use_rag: bool = True
    ) -> Dict:
        """
        Generate an answer to a question using the loaded model.
        
        Args:
            question: Question to answer
            use_rag: Whether to use RAG context retrieval
            
        Returns:
            Dict: Generated answer with metadata
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info(f"Generating answer", extra={
            "question_length": len(question),
            "use_rag": use_rag
        })
        
        try:
            # Retrieve context from ChromaDB if RAG enabled
            context = ""
            retrieved_chunks = []
            
            if use_rag:
                retrieved_chunks = self.chromadb.retrieve_context(
                    question, 
                    top_k=settings.TOP_K_RETRIEVAL
                )
                context = "\n\n".join(retrieved_chunks)
                logger.info(f"Retrieved {len(retrieved_chunks)} context chunks")
            
            # Build prompt
            if context:
                prompt = f"""Context: {context}

Question: {question}

Answer:"""
            else:
                prompt = f"""Question: {question}

Answer:"""
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=settings.MAX_LENGTH
            )
            
            # Move to same device as model
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.generation_config["max_new_tokens"],
                    temperature=self.generation_config["temperature"],
                    top_p=self.generation_config["top_p"],
                    do_sample=self.generation_config["do_sample"],
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            # Extract only the answer part (after "Answer:")
            if "Answer:" in generated_text:
                answer = generated_text.split("Answer:")[-1].strip()
            else:
                answer = generated_text.strip()
            
            result = {
                "question": question,
                "answer": answer,
                "used_rag": use_rag,
                "num_context_chunks": len(retrieved_chunks),
                "generation_config": self.generation_config
            }
            
            logger.info(f"Answer generated successfully", extra={
                "answer_length": len(answer),
                "used_rag": use_rag
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {str(e)}", exc_info=True)
            raise
    
    
    def batch_evaluate(
        self, 
        questions: List[str], 
        use_rag: bool = True
    ) -> List[Dict]:
        """
        Generate answers for multiple questions in batch.
        
        Args:
            questions: List of questions
            use_rag: Whether to use RAG context retrieval
            
        Returns:
            List[Dict]: List of results for each question
        """
        if len(questions) > settings.MAX_BATCH_EVALUATION_SIZE:
            raise ValueError(
                f"Batch size exceeds maximum: {settings.MAX_BATCH_EVALUATION_SIZE}"
            )
        
        logger.info(f"Starting batch evaluation", extra={
            "num_questions": len(questions),
            "use_rag": use_rag
        })
        
        results = []
        
        for i, question in enumerate(questions):
            logger.info(f"Processing question {i+1}/{len(questions)}")
            
            try:
                result = self.generate_answer(question, use_rag=use_rag)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process question {i+1}: {str(e)}")
                results.append({
                    "question": question,
                    "answer": None,
                    "error": str(e),
                    "used_rag": use_rag
                })
        
        logger.info(f"Batch evaluation completed", extra={
            "total_questions": len(questions),
            "successful": sum(1 for r in results if "error" not in r)
        })
        
        return results
    
    
    def compare_with_ideal(
        self, 
        question: str, 
        ideal_answer: str, 
        use_rag: bool = True
    ) -> Dict:
        """
        Generate answer and compare with ideal answer.
        
        Args:
            question: Question to answer
            ideal_answer: Expected ideal answer
            use_rag: Whether to use RAG
            
        Returns:
            Dict: Generated answer with comparison metadata
        """
        logger.info("Generating answer with comparison to ideal")
        
        result = self.generate_answer(question, use_rag=use_rag)
        
        # Add comparison metadata
        result["ideal_answer"] = ideal_answer
        result["generated_length"] = len(result["answer"])
        result["ideal_length"] = len(ideal_answer)
        
        # Simple similarity metrics (can be enhanced)
        generated_words = set(result["answer"].lower().split())
        ideal_words = set(ideal_answer.lower().split())
        
        if ideal_words:
            word_overlap = len(generated_words & ideal_words) / len(ideal_words)
        else:
            word_overlap = 0.0
        
        result["word_overlap_score"] = round(word_overlap, 3)
        
        return result