# AI-Powered Semantic Search with PostgreSQL and pgvector

A complete semantic search engine implementation using PostgreSQL with the pgvector extension, supporting both local and OpenAI embeddings for document similarity search.

## 🌟 Features

- **Vector Similarity Search**: Uses PostgreSQL pgvector extension for efficient cosine similarity search
- **Multiple Embedding Models**: Support for both local models (sentence-transformers) and OpenAI embeddings
- **Document Management**: Load documents from various sources (JSON, text files, sample data)
- **Web Interfaces**: Both FastAPI (REST API + web UI) and Streamlit applications
- **Batch Processing**: Efficient batch embedding generation and database insertion
- **Real-time Search**: Fast semantic search with configurable similarity thresholds
- **Database Management**: Built-in tools for database initialization and management

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Interface │    │  Embedding       │    │   PostgreSQL    │
│  (FastAPI/      │◄──►│  Generation      │◄──►│   + pgvector    │
│   Streamlit)    │    │  (Local/OpenAI)  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                       │
         │                        │                       │
         ▼                        ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Document       │    │  Vector          │    │  Similarity     │
│  Processing     │    │  Storage         │    │  Search         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

1. **PostgreSQL** (version 12+) with pgvector extension installed
2. **Python** 3.8 or higher
3. **Optional**: OpenAI API key for OpenAI embeddings

### Installing pgvector

**Ubuntu/Debian:**
```bash
sudo apt install postgresql-contrib
sudo apt install build-essential git
git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

**macOS (with Homebrew):**
```bash
brew install pgvector
```

**Windows:**
See [pgvector documentation](https://github.com/pgvector/pgvector#installation) for Windows installation instructions.

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository or copy the files
cd postgresql

# Install Python dependencies
pip install -r requirements.txt

# Copy and configure environment file
cp .env.example .env
# Edit .env with your PostgreSQL credentials
```

### 2. Configure Environment

Edit `.env` file with your settings:

```bash
# PostgreSQL Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/semantic_search_db
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=semantic_search_db
POSTGRES_USER=username
POSTGRES_PASSWORD=password

# OpenAI API (optional)
OPENAI_API_KEY=your_openai_api_key_here

# Model Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
```

### 3. Initialize Database

```bash
python setup.py
```

This script will:
- Check Python version and dependencies
- Test PostgreSQL connection
- Initialize database with pgvector extension
- Create required tables and indexes
- Test embedding generation

### 4. Run the Application

**Option A: FastAPI Web Interface**
```bash
python fastapi_app.py
```
Then open http://127.0.0.1:8000 in your browser.

**Option B: Streamlit Interface**
```bash
streamlit run streamlit_app.py
```

**Option C: Python Script**
```bash
python semantic_search.py
```

## 📁 Project Structure

```
postgresql/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .env.example             # Environment configuration template
├── setup.py                 # Setup and initialization script
├── database.py              # Database connection and management
├── embeddings.py            # Embedding generation (local/OpenAI)
├── document_loader.py       # Document loading utilities
├── semantic_search.py       # Core search engine logic
├── fastapi_app.py          # FastAPI web application
└── streamlit_app.py        # Streamlit web application
```

## 🎯 Usage Examples

### Basic Search Engine Usage

```python
from semantic_search import create_search_engine

# Initialize search engine
engine = create_search_engine()

# Load sample documents
results = engine.load_and_index_documents("sample", num_articles=50)
print(f"Indexed {results['indexed_documents']} documents")

# Perform semantic search
search_results = engine.search(
    query="artificial intelligence and machine learning",
    limit=5,
    similarity_threshold=0.7
)

for result in search_results:
    print(f"Title: {result['title']}")
    print(f"Similarity: {result['similarity_score']:.3f}")
    print(f"Content: {result['content_preview']}")
    print("-" * 50)
```

### Loading Custom Documents

```python
# From JSON file
engine.load_and_index_documents(
    "json", 
    file_path="path/to/documents.json"
)

# From text files directory
engine.load_and_index_documents(
    "text", 
    directory_path="path/to/text/files",
    file_extension=".txt"
)

# Add single document
doc_id = engine.add_document(
    title="Custom Document",
    content="This is a custom document for testing...",
    source="manual",
    metadata={"category": "test", "tags": ["custom"]}
)
```

### Advanced Search Options

```python
# Search with custom parameters
results = engine.search(
    query="climate change environmental impact",
    limit=10,
    similarity_threshold=0.6  # Lower threshold for more results
)

# Get database statistics
stats = engine.get_database_stats()
print(f"Total documents: {stats['total_documents']}")
print(f"Embedding dimension: {stats['embedding_dimension']}")
```

## 🌐 Web Interfaces

### FastAPI Interface

The FastAPI application provides:
- **REST API endpoints** for programmatic access
- **Interactive web UI** for manual testing
- **Automatic API documentation** at `/docs`
- **Real-time search** with visual similarity scores

**Key Endpoints:**
- `GET /`: Web interface
- `POST /api/search`: Perform semantic search
- `POST /api/documents`: Add single document
- `POST /api/index`: Batch index documents
- `GET /api/stats`: Database statistics

### Streamlit Interface

The Streamlit app offers:
- **Interactive sidebar** with database stats
- **Visual similarity charts** using Plotly
- **Document management tools**
- **Real-time embedding visualization**
- **Sample data loading interface**

## 🔧 Configuration Options

### Embedding Models

**Local Models (Sentence Transformers):**
```bash
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2  # Default, 384 dims
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2  # Higher quality, 768 dims
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2  # Multilingual
```

**OpenAI Models:**
```bash
EMBEDDING_MODEL=openai
OPENAI_API_KEY=your_api_key_here
```

### Database Configuration

```bash
# Connection settings
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=semantic_search_db
POSTGRES_USER=username
POSTGRES_PASSWORD=password

# Or use full connection URL
DATABASE_URL=postgresql://user:pass@host:port/db
```

## 📊 Performance Considerations

### Indexing Performance
- **Batch size**: Adjust based on available memory (10-50 documents per batch)
- **Embedding dimension**: Lower dimensions = faster search, higher = better accuracy
- **Hardware**: GPU acceleration available for sentence-transformers models

### Search Performance
- **Vector indexes**: Automatically created for efficient similarity search
- **Similarity threshold**: Higher thresholds = fewer results, faster queries
- **Result limit**: Limit results to improve response time

### Database Optimization

```sql
-- Tune pgvector index parameters
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

-- For better recall with more memory usage
SET ivfflat.probes = 10;
```

## 🐛 Troubleshooting

### Common Issues

**1. pgvector extension not found:**
```bash
ERROR: extension "vector" is not available
```
Solution: Install pgvector extension for PostgreSQL.

**2. Python module import errors:**
```bash
ModuleNotFoundError: No module named 'sentence_transformers'
```
Solution: Install requirements: `pip install -r requirements.txt`

**3. Database connection failed:**
```bash
psycopg2.OperationalError: connection failed
```
Solution: Check PostgreSQL service, credentials, and network connectivity.

**4. Out of memory during embedding generation:**
```bash
RuntimeError: CUDA out of memory
```
Solution: Reduce batch size or use CPU instead of GPU.

### Performance Issues

**Slow search queries:**
- Check if vector indexes are created
- Reduce similarity threshold
- Limit number of results
- Consider using faster embedding models

**High memory usage:**
- Reduce batch size during indexing
- Use smaller embedding models
- Implement connection pooling

## 🔒 Security Considerations

- **Environment Variables**: Keep credentials in `.env` file, never commit to version control
- **Database Access**: Use least-privilege database users
- **API Security**: Add authentication for production deployments
- **Input Validation**: Sanitize user inputs to prevent injection attacks

## 🚀 Deployment

### Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "fastapi_app.py"]
```

### Production Considerations

- Use production ASGI server (gunicorn + uvicorn)
- Set up SSL/TLS certificates
- Configure proper logging
- Implement health checks
- Set up monitoring and alerting

## 📚 Further Reading

- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [PostgreSQL Vector Operations](https://www.postgresql.org/docs/current/functions-array.html)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review PostgreSQL and pgvector logs
3. Test with sample data first
4. Verify environment configuration

---

**Happy searching! 🔍✨**
