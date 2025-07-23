"""
Embedding generation utilities for semantic search.
Supports both local models (sentence-transformers) and OpenAI embeddings.
"""

import os
import logging
from typing import List, Union, Optional
import numpy as np
from abc import ABC, abstractmethod
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator(ABC):
    """Abstract base class for embedding generators."""
    
    @abstractmethod
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        pass


class SentenceTransformerEmbedding(EmbeddingGenerator):
    """
    Embedding generator using sentence-transformers library.
    Supports local models that run without API keys.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the sentence transformer model.
        
        Args:
            model_name (str): Name of the sentence transformer model
        """
        self.model_name = model_name
        self.model = None
        self._dimension = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # Get embedding dimension
            sample_embedding = self.model.encode("sample text")
            self._dimension = len(sample_embedding)
            logger.info(f"Model loaded successfully. Embedding dimension: {self._dimension}")
            
        except ImportError:
            logger.error("sentence-transformers library not found. Please install it with: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text (str): Input text
            
        Returns:
            List[float]: Embedding vector
        """
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        try:
            # Clean and preprocess text
            text = self._preprocess_text(text)
            
            # Generate embedding
            embedding = self.model.encode(text, convert_to_numpy=True)
            
            # Normalize the embedding (optional, but often helpful for cosine similarity)
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch processing).
        
        Args:
            texts (List[str]): List of input texts
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        try:
            # Clean and preprocess texts
            processed_texts = [self._preprocess_text(text) for text in texts]
            
            # Generate embeddings in batch
            embeddings = self.model.encode(processed_texts, convert_to_numpy=True)
            
            # Normalize embeddings
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embeddings for texts: {e}")
            raise
    
    def get_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        return self._dimension
    
    @staticmethod
    def _preprocess_text(text: str) -> str:
        """
        Preprocess text before embedding generation.
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = text.strip()
        text = ' '.join(text.split())  # Remove extra whitespace
        
        return text


class OpenAIEmbedding(EmbeddingGenerator):
    """
    Embedding generator using OpenAI's embedding API.
    Requires OpenAI API key.
    """
    
    def __init__(self, model_name: str = "text-embedding-ada-002", api_key: Optional[str] = None):
        """
        Initialize OpenAI embedding generator.
        
        Args:
            model_name (str): OpenAI embedding model name
            api_key (str, optional): OpenAI API key
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.client = None
        self._dimension = 1536  # Default for text-embedding-ada-002
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client."""
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
            logger.info(f"OpenAI client initialized with model: {self.model_name}")
        except ImportError:
            logger.error("openai library not found. Please install it with: pip install openai")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text using OpenAI API.
        
        Args:
            text (str): Input text
            
        Returns:
            List[float]: Embedding vector
        """
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")
        
        try:
            # Clean and preprocess text
            text = self._preprocess_text(text)
            
            # Generate embedding
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text
            )
            
            embedding = response.data[0].embedding
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate OpenAI embedding: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using OpenAI API.
        
        Args:
            texts (List[str]): List of input texts
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")
        
        try:
            # Clean and preprocess texts
            processed_texts = [self._preprocess_text(text) for text in texts]
            
            # Generate embeddings in batch
            response = self.client.embeddings.create(
                model=self.model_name,
                input=processed_texts
            )
            
            embeddings = [data.embedding for data in response.data]
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate OpenAI embeddings: {e}")
            raise
    
    def get_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        return self._dimension
    
    @staticmethod
    def _preprocess_text(text: str) -> str:
        """
        Preprocess text before embedding generation.
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = text.strip()
        text = ' '.join(text.split())  # Remove extra whitespace
        
        # Truncate if too long (OpenAI has token limits)
        if len(text) > 8000:  # Conservative limit
            text = text[:8000]
        
        return text


class EmbeddingFactory:
    """Factory class to create embedding generators based on configuration."""
    
    @staticmethod
    def create_embedding_generator(model_type: str = "sentence-transformers") -> EmbeddingGenerator:
        """
        Create an embedding generator based on the specified type.
        
        Args:
            model_type (str): Type of embedding model ("sentence-transformers" or "openai")
            
        Returns:
            EmbeddingGenerator: Configured embedding generator
        """
        if model_type.lower() in ["sentence-transformers", "local", "huggingface"]:
            model_name = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
            return SentenceTransformerEmbedding(model_name)
        
        elif model_type.lower() in ["openai", "api"]:
            api_key = os.getenv('OPENAI_API_KEY')
            return OpenAIEmbedding(api_key=api_key)
        
        else:
            raise ValueError(f"Unsupported embedding model type: {model_type}")


def get_embedding_generator(model_type: Optional[str] = None) -> EmbeddingGenerator:
    """
    Get a configured embedding generator.
    
    Args:
        model_type (str, optional): Type of embedding model
        
    Returns:
        EmbeddingGenerator: Configured embedding generator
    """
    if model_type is None:
        # Try to determine from environment
        embedding_model = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        if embedding_model.lower() == 'openai':
            model_type = 'openai'
        else:
            model_type = 'sentence-transformers'
    
    return EmbeddingFactory.create_embedding_generator(model_type)


if __name__ == "__main__":
    # Example usage
    try:
        # Test local sentence transformer
        print("Testing sentence transformer embedding...")
        local_embedder = SentenceTransformerEmbedding()
        
        test_texts = [
            "This is a test document about machine learning.",
            "Python is a great programming language for data science.",
            "PostgreSQL is a powerful relational database system."
        ]
        
        embeddings = local_embedder.generate_embeddings(test_texts)
        print(f"Generated {len(embeddings)} embeddings with dimension {local_embedder.get_dimension()}")
        
        # Test single embedding
        single_embedding = local_embedder.generate_embedding("Single test text")
        print(f"Single embedding dimension: {len(single_embedding)}")
        
        print("Embedding generation test completed successfully!")
        
    except Exception as e:
        print(f"Embedding generation test failed: {e}")
