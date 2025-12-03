# models/chromadb_handler.py

import logging
from pathlib import Path
from typing import List, Dict
import chromadb
from chromadb.config import Settings as ChromaSettings

from utils.model_loader import download_and_load_embedding_model
from config.settings import settings

logger = logging.getLogger("exam_evaluator.chromadb_handler")


class ChromaDBHandler:
    """
    Handles ChromaDB operations for RAG (Retrieval Augmented Generation).
    Uses embedding model from settings.py.
    """
    
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedding_model = None
    
    
    def initialize(self, model_id: str):
        """
        Initialize ChromaDB client and collection for a specific model.
        
        Args:
            model_id: Unique identifier for the model (used as collection name)
        """
        logger.info(f"Initializing ChromaDB for model: {model_id}", extra={
            "model_id": model_id,
            "storage_path": settings.CHROMADB_PATH
        })
        
        try:
            # Create persistent ChromaDB client
            self.client = chromadb.PersistentClient(
                path=settings.CHROMADB_PATH,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Load embedding model (downloads from HuggingFace if not cached)
            self.embedding_model = download_and_load_embedding_model()
            
            # Get or create collection for this model
            self.collection = self.client.get_or_create_collection(
                name=f"model_{model_id}",
                metadata={"model_id": model_id}
            )
            
            logger.info(f"ChromaDB initialized", extra={
                "model_id": model_id,
                "collection_name": self.collection.name,
                "document_count": self.collection.count()
            })
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}", exc_info=True)
            raise
    
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks for embedding.
        Uses chunk size and overlap from settings.
        
        Args:
            text: Full text to chunk
            
        Returns:
            List[str]: List of text chunks
        """
        words = text.split()
        chunks = []
        
        chunk_size = settings.CHUNK_SIZE
        overlap = settings.CHUNK_OVERLAP
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        logger.info(f"Text chunked into {len(chunks)} chunks", extra={
            "num_chunks": len(chunks),
            "chunk_size": chunk_size,
            "overlap": overlap
        })
        
        return chunks
    
    
    def add_document(self, pdf_text: str, model_id: str):
        """
        Add PDF text to ChromaDB after chunking and embedding.
        
        Args:
            pdf_text: Full text extracted from PDF
            model_id: Model identifier for collection
        """
        logger.info(f"Adding document to ChromaDB", extra={
            "model_id": model_id,
            "text_length": len(pdf_text)
        })
        
        try:
            # Initialize if not already done
            if self.collection is None:
                self.initialize(model_id)
            
            # Chunk the text
            chunks = self.chunk_text(pdf_text)
            
            if not chunks:
                raise ValueError("No chunks created from PDF text")
            
            # Generate embeddings using the embedding model
            logger.info("Generating embeddings...")
            embeddings = self.embedding_model.encode(
                chunks,
                show_progress_bar=True,
                convert_to_numpy=True
            ).tolist()
            
            # Prepare data for ChromaDB
            ids = [f"chunk_{i}" for i in range(len(chunks))]
            
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=[{"chunk_index": i} for i in range(len(chunks))]
            )
            
            logger.info(f"Document added to ChromaDB", extra={
                "model_id": model_id,
                "num_chunks": len(chunks),
                "total_documents": self.collection.count()
            })
            
        except Exception as e:
            logger.error(f"Failed to add document to ChromaDB: {str(e)}", exc_info=True)
            raise
    
    
    def retrieve_context(self, question: str, top_k: int = None) -> List[str]:
        """
        Retrieve relevant context chunks for a given question.
        
        Args:
            question: Question to find relevant context for
            top_k: Number of chunks to retrieve (uses settings.TOP_K_RETRIEVAL if None)
            
        Returns:
            List[str]: List of relevant text chunks
        """
        if top_k is None:
            top_k = settings.TOP_K_RETRIEVAL
        
        if self.collection is None or self.embedding_model is None:
            raise ValueError("ChromaDB not initialized. Call initialize() first.")
        
        logger.info(f"Retrieving context for question", extra={
            "question_length": len(question),
            "top_k": top_k
        })
        
        try:
            # Generate embedding for the question
            question_embedding = self.embedding_model.encode(
                [question],
                convert_to_numpy=True
            ).tolist()[0]
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[question_embedding],
                n_results=top_k
            )
            
            # Extract documents
            contexts = results["documents"][0] if results["documents"] else []
            
            logger.info(f"Retrieved {len(contexts)} context chunks", extra={
                "num_contexts": len(contexts),
                "top_k": top_k
            })
            
            return contexts
            
        except Exception as e:
            logger.error(f"Failed to retrieve context: {str(e)}", exc_info=True)
            raise
    
    
    def delete_collection(self, model_id: str):
        """
        Delete a collection from ChromaDB.
        
        Args:
            model_id: Model identifier for collection to delete
        """
        logger.info(f"Deleting collection for model: {model_id}", extra={
            "model_id": model_id
        })
        
        try:
            if self.client is None:
                self.initialize(model_id)
            
            collection_name = f"model_{model_id}"
            self.client.delete_collection(name=collection_name)
            
            logger.info(f"Collection deleted", extra={
                "model_id": model_id,
                "collection_name": collection_name
            })
            
        except Exception as e:
            logger.error(f"Failed to delete collection: {str(e)}", exc_info=True)
            raise
    
    
    def get_collection_info(self, model_id: str) -> Dict:
        """
        Get information about a collection.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Dict: Collection information
        """
        if self.collection is None:
            self.initialize(model_id)
        
        return {
            "model_id": model_id,
            "collection_name": self.collection.name,
            "document_count": self.collection.count(),
            "embedding_model": settings.EMBEDDING_MODEL_NAME,
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "top_k_retrieval": settings.TOP_K_RETRIEVAL
        }