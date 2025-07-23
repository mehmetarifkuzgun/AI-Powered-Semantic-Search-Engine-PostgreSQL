"""
Document loading and processing utilities for semantic search.
Supports loading documents from various sources and formats.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass
from pathlib import Path
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a document with metadata."""
    title: str
    content: str
    source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}


class DocumentLoader:
    """Base class for document loaders."""
    
    def load_documents(self) -> List[Document]:
        """Load documents from source."""
        raise NotImplementedError


class SampleNewsLoader(DocumentLoader):
    """
    Loads sample news articles for demonstration.
    Creates synthetic news data if no external source is available.
    """
    
    def __init__(self, num_articles: int = 50):
        """
        Initialize sample news loader.
        
        Args:
            num_articles (int): Number of sample articles to generate
        """
        self.num_articles = num_articles
    
    def load_documents(self) -> List[Document]:
        """
        Load sample news documents.
        
        Returns:
            List[Document]: List of sample news documents
        """
        logger.info(f"Loading {self.num_articles} sample news articles...")
        
        # Sample news articles data
        sample_articles = [
            {
                "title": "Breakthrough in Quantum Computing Research",
                "content": "Scientists at leading universities have achieved a major breakthrough in quantum computing, developing a new algorithm that could revolutionize cryptography and computational chemistry. The research demonstrates unprecedented quantum coherence times and error correction capabilities.",
                "category": "Technology",
                "tags": ["quantum computing", "research", "breakthrough"]
            },
            {
                "title": "Climate Change Impact on Global Agriculture",
                "content": "A comprehensive study reveals how climate change is affecting agricultural productivity worldwide. Rising temperatures, changing precipitation patterns, and extreme weather events are forcing farmers to adapt their practices and crop selections.",
                "category": "Environment",
                "tags": ["climate change", "agriculture", "environment"]
            },
            {
                "title": "Artificial Intelligence in Healthcare Diagnostics",
                "content": "Machine learning models are showing remarkable accuracy in medical diagnostics, particularly in radiology and pathology. AI systems can now detect certain conditions earlier and more accurately than traditional methods.",
                "category": "Healthcare",
                "tags": ["AI", "healthcare", "diagnostics", "machine learning"]
            },
            {
                "title": "Renewable Energy Storage Solutions",
                "content": "New battery technologies and energy storage systems are making renewable energy more viable for grid-scale applications. Advanced lithium-ion batteries and emerging solid-state technologies promise longer duration storage.",
                "category": "Energy",
                "tags": ["renewable energy", "battery", "storage", "grid"]
            },
            {
                "title": "Space Exploration and Mars Missions",
                "content": "Recent Mars missions have provided unprecedented insights into the red planet's geology, atmosphere, and potential for past life. Advanced rovers and orbiters continue to collect valuable scientific data.",
                "category": "Space",
                "tags": ["Mars", "space exploration", "rovers", "science"]
            },
            {
                "title": "Cybersecurity Threats in Remote Work Era",
                "content": "The shift to remote work has created new cybersecurity challenges. Organizations are implementing zero-trust architectures and advanced threat detection systems to protect distributed workforces.",
                "category": "Technology",
                "tags": ["cybersecurity", "remote work", "threats", "protection"]
            },
            {
                "title": "Gene Therapy Advances in Rare Diseases",
                "content": "Gene therapy treatments are showing promising results for previously untreatable rare genetic diseases. CRISPR and other gene editing technologies are enabling precise therapeutic interventions.",
                "category": "Healthcare",
                "tags": ["gene therapy", "CRISPR", "rare diseases", "genetics"]
            },
            {
                "title": "Sustainable Urban Development Initiatives",
                "content": "Cities worldwide are implementing smart city technologies and sustainable development practices. Green infrastructure, smart transportation, and energy-efficient buildings are becoming standard.",
                "category": "Urban Planning",
                "tags": ["smart cities", "sustainability", "urban development", "green infrastructure"]
            },
            {
                "title": "Blockchain Applications Beyond Cryptocurrency",
                "content": "Blockchain technology is finding applications in supply chain management, digital identity verification, and decentralized finance. The technology promises increased transparency and security.",
                "category": "Technology",
                "tags": ["blockchain", "supply chain", "identity", "DeFi"]
            },
            {
                "title": "Ocean Conservation and Marine Biodiversity",
                "content": "Marine scientists are documenting declining ocean health and implementing conservation strategies. Coral reef restoration, plastic pollution reduction, and marine protected areas are key focus areas.",
                "category": "Environment",
                "tags": ["ocean conservation", "marine biodiversity", "coral reefs", "pollution"]
            }
        ]
        
        documents = []
        
        # Generate documents with variations
        for i in range(self.num_articles):
            base_article = sample_articles[i % len(sample_articles)]
            
            # Create variations for additional articles
            variation_suffix = ""
            if i >= len(sample_articles):
                variation_number = (i // len(sample_articles)) + 1
                variation_suffix = f" - Update {variation_number}"
            
            document = Document(
                title=base_article["title"] + variation_suffix,
                content=base_article["content"],
                source="sample_news",
                metadata={
                    "category": base_article["category"],
                    "tags": base_article["tags"],
                    "article_id": i + 1,
                    "generated": True
                }
            )
            documents.append(document)
        
        logger.info(f"Successfully loaded {len(documents)} sample documents")
        return documents


class JSONFileLoader(DocumentLoader):
    """
    Loads documents from JSON files.
    Expects JSON format with title, content, and optional metadata fields.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize JSON file loader.
        
        Args:
            file_path (str): Path to JSON file
        """
        self.file_path = Path(file_path)
    
    def load_documents(self) -> List[Document]:
        """
        Load documents from JSON file.
        
        Returns:
            List[Document]: List of documents from JSON file
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {self.file_path}")
        
        logger.info(f"Loading documents from JSON file: {self.file_path}")
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = []
            
            # Handle different JSON structures
            if isinstance(data, list):
                # Array of documents
                for item in data:
                    documents.append(self._create_document_from_dict(item))
            elif isinstance(data, dict):
                if 'documents' in data:
                    # Nested structure with documents key
                    for item in data['documents']:
                        documents.append(self._create_document_from_dict(item))
                else:
                    # Single document
                    documents.append(self._create_document_from_dict(data))
            
            logger.info(f"Successfully loaded {len(documents)} documents from JSON")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to load documents from JSON file: {e}")
            raise
    
    def _create_document_from_dict(self, data: Dict[str, Any]) -> Document:
        """
        Create Document object from dictionary.
        
        Args:
            data (Dict[str, Any]): Document data dictionary
            
        Returns:
            Document: Document object
        """
        title = data.get('title', 'Untitled')
        content = data.get('content', data.get('text', ''))
        source = data.get('source', str(self.file_path))
        
        # Extract metadata (everything except title, content, source)
        metadata = {k: v for k, v in data.items() 
                   if k not in ['title', 'content', 'text', 'source']}
        
        return Document(
            title=title,
            content=content,
            source=source,
            metadata=metadata
        )


class TextFileLoader(DocumentLoader):
    """
    Loads documents from text files in a directory.
    Each text file becomes a separate document.
    """
    
    def __init__(self, directory_path: str, file_extension: str = ".txt"):
        """
        Initialize text file loader.
        
        Args:
            directory_path (str): Path to directory containing text files
            file_extension (str): File extension to filter by
        """
        self.directory_path = Path(directory_path)
        self.file_extension = file_extension
    
    def load_documents(self) -> List[Document]:
        """
        Load documents from text files.
        
        Returns:
            List[Document]: List of documents from text files
        """
        if not self.directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {self.directory_path}")
        
        logger.info(f"Loading text files from directory: {self.directory_path}")
        
        documents = []
        text_files = list(self.directory_path.glob(f"*{self.file_extension}"))
        
        for file_path in text_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                if content:  # Skip empty files
                    document = Document(
                        title=file_path.stem,  # Filename without extension
                        content=content,
                        source=str(file_path),
                        metadata={
                            "file_name": file_path.name,
                            "file_size": file_path.stat().st_size,
                            "file_extension": self.file_extension
                        }
                    )
                    documents.append(document)
            
            except Exception as e:
                logger.warning(f"Failed to load file {file_path}: {e}")
                continue
        
        logger.info(f"Successfully loaded {len(documents)} text documents")
        return documents


class WebDataLoader(DocumentLoader):
    """
    Loads sample documents from web APIs (demo purposes).
    This is a simplified example - in practice, you'd want more robust web scraping.
    """
    
    def __init__(self, source_type: str = "news"):
        """
        Initialize web data loader.
        
        Args:
            source_type (str): Type of web data to load
        """
        self.source_type = source_type
    
    def load_documents(self) -> List[Document]:
        """
        Load documents from web sources.
        Note: This is a demo implementation using placeholder data.
        
        Returns:
            List[Document]: List of documents from web sources
        """
        logger.info(f"Loading {self.source_type} data from web sources...")
        
        # For demo purposes, return sample data
        # In a real implementation, you would make API calls or scrape websites
        return SampleNewsLoader(num_articles=20).load_documents()


def create_document_loader(source_type: str, **kwargs) -> DocumentLoader:
    """
    Factory function to create document loaders.
    
    Args:
        source_type (str): Type of document loader
        **kwargs: Additional arguments for the loader
        
    Returns:
        DocumentLoader: Configured document loader
    """
    if source_type.lower() == "sample":
        return SampleNewsLoader(kwargs.get('num_articles', 50))
    elif source_type.lower() == "json":
        return JSONFileLoader(kwargs['file_path'])
    elif source_type.lower() == "text":
        return TextFileLoader(kwargs['directory_path'], kwargs.get('file_extension', '.txt'))
    elif source_type.lower() == "web":
        return WebDataLoader(kwargs.get('source_type', 'news'))
    else:
        raise ValueError(f"Unsupported source type: {source_type}")


def preprocess_document_content(content: str, max_length: int = 5000) -> str:
    """
    Preprocess document content for embedding generation.
    
    Args:
        content (str): Raw document content
        max_length (int): Maximum content length
        
    Returns:
        str: Preprocessed content
    """
    if not content:
        return ""
    
    # Basic cleaning
    content = content.strip()
    content = ' '.join(content.split())  # Normalize whitespace
    
    # Truncate if too long
    if len(content) > max_length:
        content = content[:max_length].rsplit(' ', 1)[0]  # Break at word boundary
    
    return content


if __name__ == "__main__":
    # Example usage
    try:
        # Test sample news loader
        loader = SampleNewsLoader(num_articles=5)
        documents = loader.load_documents()
        
        print(f"Loaded {len(documents)} sample documents:")
        for doc in documents[:3]:  # Show first 3
            print(f"- {doc.title}")
            print(f"  Content: {doc.content[:100]}...")
            print(f"  Metadata: {doc.metadata}")
            print()
        
        print("Document loading test completed successfully!")
        
    except Exception as e:
        print(f"Document loading test failed: {e}")
