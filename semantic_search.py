"""
Core semantic search engine that orchestrates document loading, embedding, and similarity search.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import time
from datetime import datetime

from database import DatabaseManager, get_db_manager, initialize_database
from embeddings import EmbeddingGenerator, get_embedding_generator
from document_loader import Document, create_document_loader, preprocess_document_content

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticSearchEngine:
    """
    Main semantic search engine that handles document indexing and similarity search.
    """
    
    def __init__(self, 
                 embedding_model_type: str = "sentence-transformers",
                 db_manager: Optional[DatabaseManager] = None):
        """
        Initialize the semantic search engine.
        
        Args:
            embedding_model_type (str): Type of embedding model to use
            db_manager (DatabaseManager, optional): Database manager instance
        """
        self.embedding_generator = get_embedding_generator(embedding_model_type)
        self.db_manager = db_manager or get_db_manager()
        self.embedding_dimension = self.embedding_generator.get_dimension()
        
        logger.info(f"Semantic search engine initialized with {embedding_model_type} embeddings")
        logger.info(f"Embedding dimension: {self.embedding_dimension}")
    
    def initialize_database(self):
        """Initialize the database with required tables and extensions."""
        initialize_database(self.embedding_dimension)
        logger.info("Database initialized successfully")
    
    def load_and_index_documents(self, 
                                source_type: str, 
                                batch_size: int = 10,
                                **loader_kwargs) -> Dict[str, Any]:
        """
        Load documents from a source and index them in the database.
        
        Args:
            source_type (str): Type of document source ("sample", "json", "text", "web")
            batch_size (int): Number of documents to process in each batch
            **loader_kwargs: Additional arguments for document loader
            
        Returns:
            Dict[str, Any]: Indexing results and statistics
        """
        start_time = time.time()
        logger.info(f"Starting document loading and indexing from {source_type} source...")
        
        try:
            # Load documents
            loader = create_document_loader(source_type, **loader_kwargs)
            documents = loader.load_documents()
            
            if not documents:
                logger.warning("No documents loaded")
                return {"success": False, "message": "No documents to index"}
            
            logger.info(f"Loaded {len(documents)} documents, starting indexing...")
            
            # Process documents in batches
            indexed_count = 0
            failed_count = 0
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_results = self._index_document_batch(batch)
                
                indexed_count += batch_results["indexed"]
                failed_count += batch_results["failed"]
                
                logger.info(f"Processed batch {i//batch_size + 1}: "
                           f"{batch_results['indexed']} indexed, {batch_results['failed']} failed")
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            results = {
                "success": True,
                "total_documents": len(documents),
                "indexed_documents": indexed_count,
                "failed_documents": failed_count,
                "processing_time_seconds": processing_time,
                "embedding_dimension": self.embedding_dimension,
                "source_type": source_type
            }
            
            logger.info(f"Indexing completed: {indexed_count}/{len(documents)} documents indexed "
                       f"in {processing_time:.2f} seconds")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to load and index documents: {e}")
            return {"success": False, "error": str(e)}
    
    def _index_document_batch(self, documents: List[Document]) -> Dict[str, int]:
        """
        Index a batch of documents.
        
        Args:
            documents (List[Document]): Batch of documents to index
            
        Returns:
            Dict[str, int]: Batch processing results
        """
        indexed = 0
        failed = 0
        
        try:
            # Prepare texts for batch embedding generation
            texts = []
            for doc in documents:
                # Combine title and content for embedding
                full_text = f"{doc.title}. {doc.content}"
                processed_text = preprocess_document_content(full_text)
                texts.append(processed_text)
            
            # Generate embeddings for the batch
            embeddings = self.embedding_generator.generate_embeddings(texts)
            
            # Insert documents into database
            for doc, embedding in zip(documents, embeddings):
                try:
                    doc_id = self.db_manager.insert_document(
                        title=doc.title,
                        content=doc.content,
                        embedding=embedding,
                        source=doc.source,
                        metadata=doc.metadata
                    )
                    indexed += 1
                    logger.debug(f"Indexed document ID {doc_id}: {doc.title[:50]}...")
                    
                except Exception as e:
                    logger.warning(f"Failed to index document '{doc.title[:50]}...': {e}")
                    failed += 1
                    
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            failed = len(documents)
        
        return {"indexed": indexed, "failed": failed}
    
    def search(self, 
               query: str, 
               limit: int = 5, 
               similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Perform semantic search for similar documents.
        
        Args:
            query (str): Search query
            limit (int): Maximum number of results to return
            similarity_threshold (float): Minimum similarity score (0-1)
            
        Returns:
            List[Dict[str, Any]]: Search results with similarity scores
        """
        start_time = time.time()
        logger.info(f"Performing semantic search for query: '{query[:100]}...'")
        
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_generator.generate_embedding(query)
            
            # Search for similar documents
            results = self.db_manager.search_similar_documents(
                query_embedding=query_embedding,
                limit=limit,
                threshold=similarity_threshold
            )
            
            search_time = time.time() - start_time
            
            # Enhance results with additional metadata
            enhanced_results = []
            for result in results:
                enhanced_result = {
                    "id": result["id"],
                    "title": result["title"],
                    "content": result["content"],
                    "source": result["source"],
                    "metadata": result["metadata"],
                    "similarity_score": round(result["similarity_score"], 4),
                    "content_preview": result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"]
                }
                enhanced_results.append(enhanced_result)
            
            logger.info(f"Search completed in {search_time:.3f} seconds, found {len(results)} results")
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dict[str, Any]: Database statistics
        """
        try:
            document_count = self.db_manager.get_document_count()
            
            return {
                "total_documents": document_count,
                "embedding_dimension": self.embedding_dimension,
                "database_status": "connected" if self.db_manager.test_connection() else "disconnected"
            }
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {"error": str(e)}
    
    def clear_database(self):
        """Clear all documents from the database."""
        try:
            self.db_manager.delete_all_documents()
            logger.info("Database cleared successfully")
        except Exception as e:
            logger.error(f"Failed to clear database: {e}")
            raise
    
    def add_document(self, title: str, content: str, 
                    source: Optional[str] = None, 
                    metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Add a single document to the search index.
        
        Args:
            title (str): Document title
            content (str): Document content
            source (str, optional): Document source
            metadata (Dict[str, Any], optional): Additional metadata
            
        Returns:
            int: ID of the added document
        """
        try:
            # Generate embedding
            full_text = f"{title}. {content}"
            processed_text = preprocess_document_content(full_text)
            embedding = self.embedding_generator.generate_embedding(processed_text)
            
            # Insert into database
            doc_id = self.db_manager.insert_document(
                title=title,
                content=content,
                embedding=embedding,
                source=source,
                metadata=metadata
            )
            
            logger.info(f"Added document with ID {doc_id}: {title[:50]}...")
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            raise


def create_search_engine(embedding_model_type: str = "sentence-transformers") -> SemanticSearchEngine:
    """
    Create and initialize a semantic search engine.
    
    Args:
        embedding_model_type (str): Type of embedding model
        
    Returns:
        SemanticSearchEngine: Configured search engine
    """
    engine = SemanticSearchEngine(embedding_model_type)
    engine.initialize_database()
    return engine


if __name__ == "__main__":
    # Example usage and testing
    try:
        print("Initializing semantic search engine...")
        engine = create_search_engine()
        
        print("Loading and indexing sample documents...")
        results = engine.load_and_index_documents("sample", num_articles=10)
        print(f"Indexing results: {results}")
        
        print("\nDatabase statistics:")
        stats = engine.get_database_stats()
        print(f"Stats: {stats}")
        
        print("\nTesting search queries...")
        test_queries = [
            "artificial intelligence and machine learning",
            "climate change and environment",
            "quantum computing research",
            "space exploration Mars"
        ]
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            search_results = engine.search(query, limit=3, similarity_threshold=0.5)
            
            for i, result in enumerate(search_results, 1):
                print(f"  {i}. {result['title']} (Score: {result['similarity_score']})")
                print(f"     {result['content_preview']}")
        
        print("\nSemantic search engine test completed successfully!")
        
    except Exception as e:
        print(f"Semantic search engine test failed: {e}")
        import traceback
        traceback.print_exc()
