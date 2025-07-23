"""
Setup script to initialize the PostgreSQL database and install dependencies.
Run this script first to set up your semantic search environment.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_python_version():
    """Check if Python version is compatible."""
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        logger.error("Python 3.8 or higher is required!")
        return False
    logger.info(f"Python version: {sys.version}")
    return True


def install_requirements():
    """Install Python dependencies from requirements.txt."""
    logger.info("Installing Python dependencies...")
    
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        logger.error("requirements.txt not found!")
        return False
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        logger.info("✅ Python dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install dependencies: {e}")
        return False


def check_postgresql_connection():
    """Check PostgreSQL connection and availability."""
    logger.info("Checking PostgreSQL connection...")
    
    try:
        from database import DatabaseManager
        
        db_manager = DatabaseManager()
        if db_manager.test_connection():
            logger.info("✅ PostgreSQL connection successful!")
            return True
        else:
            logger.error("❌ PostgreSQL connection failed!")
            return False
            
    except ImportError as e:
        logger.error(f"❌ Failed to import database module: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ PostgreSQL connection error: {e}")
        return False


def setup_database():
    """Initialize database with pgvector extension and tables."""
    logger.info("Setting up database...")
    
    try:
        from database import initialize_database
        
        # Initialize with default embedding dimension (384 for all-MiniLM-L6-v2)
        embedding_dim = int(os.getenv('EMBEDDING_DIMENSION', '384'))
        initialize_database(embedding_dim)
        
        logger.info("✅ Database setup completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database setup failed: {e}")
        return False


def test_embedding_generation():
    """Test embedding generation functionality."""
    logger.info("Testing embedding generation...")
    
    try:
        from embeddings import get_embedding_generator
        
        # Test sentence transformer (default)
        embedder = get_embedding_generator("sentence-transformers")
        
        test_text = "This is a test document for semantic search."
        embedding = embedder.generate_embedding(test_text)
        
        logger.info(f"✅ Embedding generation successful! Dimension: {len(embedding)}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Embedding generation test failed: {e}")
        return False


def create_env_file():
    """Create .env file from .env.example if it doesn't exist."""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        logger.info("Creating .env file from .env.example...")
        try:
            env_file.write_text(env_example.read_text())
            logger.info("✅ .env file created! Please update it with your configuration.")
        except Exception as e:
            logger.error(f"❌ Failed to create .env file: {e}")
            return False
    
    return True


def print_setup_instructions():
    """Print setup instructions and next steps."""
    print("\n" + "="*60)
    print("🚀 SEMANTIC SEARCH ENGINE SETUP")
    print("="*60)
    
    print("\n📋 PREREQUISITES:")
    print("1. PostgreSQL server running and accessible")
    print("2. pgvector extension installed in PostgreSQL")
    print("3. Python 3.8 or higher")
    
    print("\n⚙️ CONFIGURATION:")
    print("1. Update the .env file with your PostgreSQL credentials")
    print("2. Set OPENAI_API_KEY if you want to use OpenAI embeddings")
    print("3. Adjust EMBEDDING_MODEL and EMBEDDING_DIMENSION as needed")
    
    print("\n🎯 NEXT STEPS:")
    print("1. Run setup: python setup.py")
    print("2. Test the engine: python semantic_search.py")
    print("3. Start FastAPI: python fastapi_app.py")
    print("4. Or start Streamlit: streamlit run streamlit_app.py")
    
    print("\n📚 SAMPLE USAGE:")
    print("# Load sample documents and search")
    print("from semantic_search import create_search_engine")
    print("engine = create_search_engine()")
    print("engine.load_and_index_documents('sample', num_articles=20)")
    print("results = engine.search('artificial intelligence')")
    
    print("\n" + "="*60)


def main():
    """Main setup function."""
    print_setup_instructions()
    
    print("\n🔧 Starting setup process...\n")
    
    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Step 2: Create .env file if needed
    if not create_env_file():
        logger.warning("⚠️ Failed to create .env file, continuing...")
    
    # Step 3: Install dependencies
    if not install_requirements():
        logger.error("❌ Setup failed during dependency installation!")
        sys.exit(1)
    
    # Step 4: Check PostgreSQL connection
    if not check_postgresql_connection():
        logger.error("❌ Setup failed: PostgreSQL connection issue!")
        logger.error("Please ensure PostgreSQL is running and credentials in .env are correct.")
        sys.exit(1)
    
    # Step 5: Setup database
    if not setup_database():
        logger.error("❌ Setup failed during database initialization!")
        sys.exit(1)
    
    # Step 6: Test embedding generation
    if not test_embedding_generation():
        logger.error("❌ Setup failed during embedding test!")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("✅ SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    print("\n🎉 Your semantic search engine is ready to use!")
    print("\n🚀 Quick start options:")
    print("   • FastAPI web interface: python fastapi_app.py")
    print("   • Streamlit app: streamlit run streamlit_app.py")
    print("   • Python script: python semantic_search.py")
    
    print("\n📖 Next steps:")
    print("   1. Load some documents using the web interface")
    print("   2. Try searching with natural language queries")
    print("   3. Experiment with different similarity thresholds")
    
    print("\n💡 Pro tips:")
    print("   • Use longer, descriptive queries for better results")
    print("   • Adjust similarity threshold based on your needs")
    print("   • Check the database stats to monitor performance")


if __name__ == "__main__":
    main()
