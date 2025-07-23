"""
Database utilities and connection management for semantic search application.
This module handles PostgreSQL connection, table creation, and pgvector setup.
"""

import os
import logging
from typing import Optional, List, Tuple
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages PostgreSQL database connections and operations for semantic search.
    """
    
    def __init__(self):
        """Initialize database manager with connection parameters."""
        self.host = os.getenv('POSTGRES_HOST', 'localhost')
        self.port = os.getenv('POSTGRES_PORT', '5432')
        self.database = os.getenv('POSTGRES_DB', 'semantic_search_db')
        self.user = os.getenv('POSTGRES_USER', 'postgres')
        self.password = os.getenv('POSTGRES_PASSWORD', 'password')
        self.database_url = os.getenv('DATABASE_URL', 
            f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}")
        
        # SQLAlchemy engine
        self.engine = None
        self.SessionLocal = None
        
    def initialize_engine(self):
        """Initialize SQLAlchemy engine and session factory."""
        try:
            self.engine = create_engine(
                self.database_url,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True
            )
            self.SessionLocal = sessionmaker(
                autocommit=False, 
                autoflush=False, 
                bind=self.engine
            )
            logger.info("Database engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database engine: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for getting a raw psycopg2 connection.
        
        Yields:
            psycopg2.connection: Database connection
        """
        conn = None
        try:
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                cursor_factory=RealDictCursor
            )
            conn.autocommit = False
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    @contextmanager
    def get_session(self):
        """
        Context manager for getting a SQLAlchemy session.
        
        Yields:
            sqlalchemy.orm.Session: Database session
        """
        if not self.SessionLocal:
            self.initialize_engine()
        
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def test_connection(self) -> bool:
        """
        Test database connection.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT version();")
                    version = cur.fetchone()
                    logger.info(f"Connected to PostgreSQL: {version['version']}")
                    return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def setup_pgvector_extension(self):
        """
        Enable pgvector extension in the database.
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Enable pgvector extension
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    conn.commit()
                    logger.info("pgvector extension enabled successfully")
        except Exception as e:
            logger.error(f"Failed to enable pgvector extension: {e}")
            raise
    
    def create_documents_table(self, embedding_dimension: int = 384):
        """
        Create the documents table with vector column for embeddings.
        
        Args:
            embedding_dimension (int): Dimension of the embedding vectors
        """
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            title VARCHAR(500),
            content TEXT NOT NULL,
            source VARCHAR(200),
            metadata JSONB,
            embedding vector({embedding_dimension}),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Create index for vector similarity search
        CREATE INDEX IF NOT EXISTS documents_embedding_idx 
        ON documents USING ivfflat (embedding vector_cosine_ops);
        
        -- Create text search index
        CREATE INDEX IF NOT EXISTS documents_content_idx 
        ON documents USING gin(to_tsvector('english', content));
        
        -- Create trigger for updated_at
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
        
        DROP TRIGGER IF EXISTS update_documents_updated_at ON documents;
        CREATE TRIGGER update_documents_updated_at
            BEFORE UPDATE ON documents
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(create_table_sql)
                    conn.commit()
                    logger.info("Documents table created successfully")
        except Exception as e:
            logger.error(f"Failed to create documents table: {e}")
            raise
    
    def insert_document(self, title: str, content: str, embedding: List[float], 
                       source: Optional[str] = None, metadata: Optional[dict] = None) -> int:
        """
        Insert a document with its embedding into the database.
        
        Args:
            title (str): Document title
            content (str): Document content
            embedding (List[float]): Document embedding vector
            source (str, optional): Document source
            metadata (dict, optional): Additional metadata
            
        Returns:
            int: ID of the inserted document
        """
        insert_sql = """
        INSERT INTO documents (title, content, source, metadata, embedding)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id;
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(insert_sql, (title, content, source, metadata, embedding))
                    doc_id = cur.fetchone()['id']
                    conn.commit()
                    logger.debug(f"Document inserted with ID: {doc_id}")
                    return doc_id
        except Exception as e:
            logger.error(f"Failed to insert document: {e}")
            raise
    
    def search_similar_documents(self, query_embedding: List[float], 
                                limit: int = 5, threshold: float = 0.8) -> List[dict]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query_embedding (List[float]): Query embedding vector
            limit (int): Maximum number of results to return
            threshold (float): Similarity threshold (0-1)
            
        Returns:
            List[dict]: List of similar documents with similarity scores
        """
        search_sql = """
        SELECT 
            id,
            title,
            content,
            source,
            metadata,
            1 - (embedding <=> %s) as similarity_score
        FROM documents
        WHERE 1 - (embedding <=> %s) > %s
        ORDER BY embedding <=> %s
        LIMIT %s;
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(search_sql, (
                        query_embedding, query_embedding, threshold, 
                        query_embedding, limit
                    ))
                    results = cur.fetchall()
                    return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Failed to search similar documents: {e}")
            raise
    
    def get_document_count(self) -> int:
        """
        Get the total number of documents in the database.
        
        Returns:
            int: Number of documents
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) as count FROM documents;")
                    result = cur.fetchone()
                    return result['count']
        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            raise
    
    def delete_all_documents(self):
        """Delete all documents from the database."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM documents;")
                    conn.commit()
                    logger.info("All documents deleted successfully")
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise


# Global database manager instance
db_manager = DatabaseManager()


def get_db_manager() -> DatabaseManager:
    """
    Get the global database manager instance.
    
    Returns:
        DatabaseManager: Database manager instance
    """
    return db_manager


def initialize_database(embedding_dimension: int = 384):
    """
    Initialize the database with pgvector extension and documents table.
    
    Args:
        embedding_dimension (int): Dimension of embedding vectors
    """
    logger.info("Initializing database...")
    
    # Test connection
    if not db_manager.test_connection():
        raise Exception("Failed to connect to database")
    
    # Setup pgvector extension
    db_manager.setup_pgvector_extension()
    
    # Create documents table
    db_manager.create_documents_table(embedding_dimension)
    
    # Initialize SQLAlchemy engine
    db_manager.initialize_engine()
    
    logger.info("Database initialization completed successfully")


if __name__ == "__main__":
    # Example usage
    try:
        initialize_database()
        print("Database setup completed successfully!")
    except Exception as e:
        print(f"Database setup failed: {e}")
