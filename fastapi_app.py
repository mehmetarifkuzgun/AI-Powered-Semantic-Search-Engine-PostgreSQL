"""
FastAPI web application for semantic search engine.
Provides REST API endpoints for document indexing and similarity search.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, validator
import uvicorn

from semantic_search import SemanticSearchEngine, create_search_engine
from document_loader import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global search engine instance
search_engine: Optional[SemanticSearchEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global search_engine
    
    # Startup
    logger.info("Initializing semantic search engine...")
    try:
        embedding_model = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        model_type = 'openai' if embedding_model.lower() == 'openai' else 'sentence-transformers'
        search_engine = create_search_engine(model_type)
        logger.info("Semantic search engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize search engine: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down semantic search engine...")


# Create FastAPI app
app = FastAPI(
    title="AI-Powered Semantic Search Engine",
    description="Semantic search engine using PostgreSQL and pgvector for document similarity search",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class SearchRequest(BaseModel):
    """Search request model."""
    query: str
    limit: Optional[int] = 5
    similarity_threshold: Optional[float] = 0.7
    
    @validator('limit')
    def validate_limit(cls, v):
        if v is not None and (v < 1 or v > 50):
            raise ValueError('limit must be between 1 and 50')
        return v
    
    @validator('similarity_threshold')
    def validate_threshold(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError('similarity_threshold must be between 0 and 1')
        return v


class SearchResult(BaseModel):
    """Search result model."""
    id: int
    title: str
    content: str
    source: Optional[str]
    metadata: Optional[Dict[str, Any]]
    similarity_score: float
    content_preview: str


class SearchResponse(BaseModel):
    """Search response model."""
    query: str
    results: List[SearchResult]
    total_results: int
    search_time_ms: float


class AddDocumentRequest(BaseModel):
    """Add document request model."""
    title: str
    content: str
    source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class IndexDocumentsRequest(BaseModel):
    """Index documents request model."""
    source_type: str
    batch_size: Optional[int] = 10
    loader_kwargs: Optional[Dict[str, Any]] = {}


class DatabaseStats(BaseModel):
    """Database statistics model."""
    total_documents: int
    embedding_dimension: int
    database_status: str


def get_search_engine() -> SemanticSearchEngine:
    """Dependency to get the search engine instance."""
    if search_engine is None:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    return search_engine


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve a simple web interface for the search engine."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI-Powered Semantic Search</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { 
                font-family: Arial, sans-serif; 
                max-width: 1200px; 
                margin: 0 auto; 
                padding: 20px; 
                background-color: #f5f5f5;
            }
            .header { 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; 
                padding: 30px; 
                border-radius: 10px; 
                text-align: center; 
                margin-bottom: 30px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .search-container { 
                background: white; 
                padding: 30px; 
                border-radius: 10px; 
                margin-bottom: 30px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            .search-box { 
                width: 100%; 
                padding: 15px; 
                font-size: 16px; 
                border: 2px solid #ddd; 
                border-radius: 5px; 
                margin-bottom: 15px;
            }
            .search-btn { 
                background: #667eea; 
                color: white; 
                padding: 15px 30px; 
                border: none; 
                border-radius: 5px; 
                cursor: pointer; 
                font-size: 16px;
                margin-right: 10px;
            }
            .search-btn:hover { background: #5a6fd8; }
            .controls { 
                display: flex; 
                gap: 15px; 
                align-items: center; 
                flex-wrap: wrap;
                margin-bottom: 15px;
            }
            .control-group { 
                display: flex; 
                align-items: center; 
                gap: 5px;
            }
            .control-input { 
                padding: 8px; 
                border: 1px solid #ddd; 
                border-radius: 3px; 
                width: 80px;
            }
            .results { 
                background: white; 
                padding: 30px; 
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            .result-item { 
                border: 1px solid #ddd; 
                margin: 15px 0; 
                padding: 20px; 
                border-radius: 8px; 
                background: #fafafa;
                transition: transform 0.2s;
            }
            .result-item:hover { 
                transform: translateY(-2px); 
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }
            .result-title { 
                font-weight: bold; 
                color: #333; 
                font-size: 18px; 
                margin-bottom: 10px;
            }
            .result-score { 
                color: #667eea; 
                font-weight: bold; 
                float: right;
                background: #f0f4ff;
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 12px;
            }
            .result-content { 
                color: #666; 
                line-height: 1.5; 
                margin-bottom: 10px;
            }
            .result-meta { 
                font-size: 12px; 
                color: #999; 
                border-top: 1px solid #eee; 
                padding-top: 10px;
            }
            .loading { 
                text-align: center; 
                color: #667eea; 
                font-style: italic; 
            }
            .error { 
                color: #e74c3c; 
                background: #fdf2f2; 
                padding: 15px; 
                border-radius: 5px; 
                margin: 15px 0;
            }
            .stats { 
                background: white; 
                padding: 20px; 
                border-radius: 10px; 
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .stats-item { 
                text-align: center;
            }
            .stats-number { 
                font-size: 24px; 
                font-weight: bold; 
                color: #667eea;
            }
            .stats-label { 
                color: #666; 
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔍 AI-Powered Semantic Search</h1>
            <p>Search documents using advanced semantic similarity powered by PostgreSQL and pgvector</p>
        </div>
        
        <div class="stats" id="stats">
            <div class="stats-item">
                <div class="stats-number" id="total-docs">-</div>
                <div class="stats-label">Total Documents</div>
            </div>
            <div class="stats-item">
                <div class="stats-number" id="embedding-dim">-</div>
                <div class="stats-label">Embedding Dimension</div>
            </div>
            <div class="stats-item">
                <div class="stats-number" id="db-status">-</div>
                <div class="stats-label">Database Status</div>
            </div>
        </div>
        
        <div class="search-container">
            <input type="text" class="search-box" id="searchQuery" 
                   placeholder="Enter your search query... (e.g., 'artificial intelligence', 'climate change', 'quantum computing')"
                   onkeypress="if(event.key==='Enter') performSearch()">
            
            <div class="controls">
                <div class="control-group">
                    <label>Results:</label>
                    <input type="number" class="control-input" id="limitInput" value="5" min="1" max="50">
                </div>
                <div class="control-group">
                    <label>Min Similarity:</label>
                    <input type="number" class="control-input" id="thresholdInput" value="0.7" min="0" max="1" step="0.1">
                </div>
                <button class="search-btn" onclick="performSearch()">🔍 Search</button>
                <button class="search-btn" onclick="loadSampleData()" style="background: #27ae60;">📚 Load Sample Data</button>
            </div>
        </div>
        
        <div class="results" id="results"></div>
        
        <script>
            // Load database stats on page load
            loadStats();
            
            async function loadStats() {
                try {
                    const response = await fetch('/api/stats');
                    const stats = await response.json();
                    
                    document.getElementById('total-docs').textContent = stats.total_documents;
                    document.getElementById('embedding-dim').textContent = stats.embedding_dimension;
                    document.getElementById('db-status').textContent = stats.database_status;
                } catch (error) {
                    console.error('Failed to load stats:', error);
                }
            }
            
            async function performSearch() {
                const query = document.getElementById('searchQuery').value.trim();
                const limit = parseInt(document.getElementById('limitInput').value) || 5;
                const threshold = parseFloat(document.getElementById('thresholdInput').value) || 0.7;
                
                if (!query) {
                    alert('Please enter a search query');
                    return;
                }
                
                const resultsDiv = document.getElementById('results');
                resultsDiv.innerHTML = '<div class="loading">Searching...</div>';
                
                try {
                    const response = await fetch('/api/search', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            query: query, 
                            limit: limit, 
                            similarity_threshold: threshold 
                        })
                    });
                    
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }
                    
                    const data = await response.json();
                    displayResults(data);
                    
                } catch (error) {
                    resultsDiv.innerHTML = `<div class="error">Search failed: ${error.message}</div>`;
                }
            }
            
            function displayResults(data) {
                const resultsDiv = document.getElementById('results');
                
                if (data.results.length === 0) {
                    resultsDiv.innerHTML = `
                        <div style="text-align: center; color: #666; padding: 40px;">
                            <h3>No results found</h3>
                            <p>Try adjusting your query or lowering the similarity threshold</p>
                        </div>
                    `;
                    return;
                }
                
                const resultsHtml = `
                    <h2>Search Results</h2>
                    <p style="color: #666; margin-bottom: 20px;">
                        Found ${data.total_results} results for "${data.query}" in ${data.search_time_ms.toFixed(2)}ms
                    </p>
                    ${data.results.map((result, index) => `
                        <div class="result-item">
                            <div class="result-title">
                                ${index + 1}. ${result.title}
                                <span class="result-score">${(result.similarity_score * 100).toFixed(1)}%</span>
                            </div>
                            <div class="result-content">${result.content_preview}</div>
                            <div class="result-meta">
                                Source: ${result.source || 'Unknown'} | 
                                ID: ${result.id} |
                                ${result.metadata ? Object.entries(result.metadata).map(([k,v]) => `${k}: ${v}`).join(' | ') : ''}
                            </div>
                        </div>
                    `).join('')}
                `;
                
                resultsDiv.innerHTML = resultsHtml;
            }
            
            async function loadSampleData() {
                if (!confirm('This will load sample documents into the database. Continue?')) {
                    return;
                }
                
                const resultsDiv = document.getElementById('results');
                resultsDiv.innerHTML = '<div class="loading">Loading sample documents...</div>';
                
                try {
                    const response = await fetch('/api/index', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            source_type: 'sample',
                            batch_size: 10,
                            loader_kwargs: { num_articles: 20 }
                        })
                    });
                    
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }
                    
                    const data = await response.json();
                    
                    resultsDiv.innerHTML = `
                        <div style="background: #d4edda; color: #155724; padding: 15px; border-radius: 5px;">
                            <h3>✅ Sample Data Loaded Successfully!</h3>
                            <p>Indexed ${data.indexed_documents}/${data.total_documents} documents in ${data.processing_time_seconds.toFixed(2)} seconds</p>
                        </div>
                    `;
                    
                    // Refresh stats
                    loadStats();
                    
                } catch (error) {
                    resultsDiv.innerHTML = `<div class="error">Failed to load sample data: ${error.message}</div>`;
                }
            }
        </script>
    </body>
    </html>
    """
    return html_content


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/stats", response_model=DatabaseStats)
async def get_database_stats(engine: SemanticSearchEngine = Depends(get_search_engine)):
    """Get database statistics."""
    try:
        stats = engine.get_database_stats()
        return DatabaseStats(**stats)
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest, 
                          engine: SemanticSearchEngine = Depends(get_search_engine)):
    """Perform semantic search for documents."""
    import time
    
    start_time = time.time()
    
    try:
        results = engine.search(
            query=request.query,
            limit=request.limit or 5,
            similarity_threshold=request.similarity_threshold or 0.7
        )
        
        search_time_ms = (time.time() - start_time) * 1000
        
        search_results = [SearchResult(**result) for result in results]
        
        return SearchResponse(
            query=request.query,
            results=search_results,
            total_results=len(search_results),
            search_time_ms=search_time_ms
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/api/documents")
async def add_document(request: AddDocumentRequest,
                      engine: SemanticSearchEngine = Depends(get_search_engine)):
    """Add a single document to the search index."""
    try:
        doc_id = engine.add_document(
            title=request.title,
            content=request.content,
            source=request.source,
            metadata=request.metadata
        )
        
        return {
            "success": True,
            "document_id": doc_id,
            "message": f"Document '{request.title}' added successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to add document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add document: {str(e)}")


@app.post("/api/index")
async def index_documents(request: IndexDocumentsRequest,
                         background_tasks: BackgroundTasks,
                         engine: SemanticSearchEngine = Depends(get_search_engine)):
    """Index documents from various sources."""
    try:
        # For demonstration, we'll run this synchronously
        # In production, you might want to use background tasks for large datasets
        results = engine.load_and_index_documents(
            source_type=request.source_type,
            batch_size=request.batch_size or 10,
            **(request.loader_kwargs or {})
        )
        
        if results["success"]:
            return {
                "success": True,
                "total_documents": results["total_documents"],
                "indexed_documents": results["indexed_documents"],
                "failed_documents": results["failed_documents"],
                "processing_time_seconds": results["processing_time_seconds"],
                "message": f"Successfully indexed {results['indexed_documents']} documents"
            }
        else:
            raise HTTPException(status_code=500, detail=results.get("error", "Indexing failed"))
        
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@app.delete("/api/documents")
async def clear_documents(engine: SemanticSearchEngine = Depends(get_search_engine)):
    """Clear all documents from the database."""
    try:
        engine.clear_database()
        return {"success": True, "message": "All documents cleared successfully"}
        
    except Exception as e:
        logger.error(f"Failed to clear documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear documents: {str(e)}")


if __name__ == "__main__":
    # Get configuration from environment
    host = os.getenv('API_HOST', '127.0.0.1')
    port = int(os.getenv('API_PORT', '8000'))
    
    logger.info(f"Starting FastAPI server on {host}:{port}")
    
    uvicorn.run(
        "fastapi_app:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
