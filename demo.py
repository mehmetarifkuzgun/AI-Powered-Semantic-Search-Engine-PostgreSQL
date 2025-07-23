"""
Demo script to showcase the semantic search engine functionality.
Run this after setting up the database to see the engine in action.
"""

import logging
import time
from typing import List, Dict

from semantic_search import create_search_engine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def print_separator(title: str = ""):
    """Print a nice separator for demo sections."""
    print("\n" + "=" * 80)
    if title:
        print(f" {title} ".center(80, "="))
        print("=" * 80)
    print()


def display_search_results(query: str, results: List[Dict], search_time: float):
    """Display search results in a formatted way."""
    print(f"🔍 Query: '{query}'")
    print(f"⏱️ Search time: {search_time*1000:.2f}ms")
    print(f"📊 Found {len(results)} results")
    print("-" * 60)
    
    for i, result in enumerate(results, 1):
        similarity_percentage = result['similarity_score'] * 100
        print(f"{i}. {result['title']}")
        print(f"   📈 Similarity: {similarity_percentage:.1f}%")
        print(f"   📂 Source: {result.get('source', 'Unknown')}")
        
        # Show content preview
        content_preview = result['content_preview']
        if len(content_preview) > 200:
            content_preview = content_preview[:200] + "..."
        print(f"   📄 Preview: {content_preview}")
        
        # Show metadata if available
        if result.get('metadata'):
            metadata = result['metadata']
            if metadata.get('category'):
                print(f"   🏷️ Category: {metadata['category']}")
            if metadata.get('tags'):
                tags = metadata['tags'] if isinstance(metadata['tags'], list) else [metadata['tags']]
                print(f"   🏷️ Tags: {', '.join(tags)}")
        
        print()


def demo_basic_functionality():
    """Demonstrate basic search engine functionality."""
    print_separator("🚀 BASIC FUNCTIONALITY DEMO")
    
    print("1. Initializing semantic search engine...")
    engine = create_search_engine()
    
    print("2. Getting database statistics...")
    stats = engine.get_database_stats()
    print(f"   📊 Total documents: {stats['total_documents']}")
    print(f"   🧠 Embedding dimension: {stats['embedding_dimension']}")
    print(f"   🔗 Database status: {stats['database_status']}")
    
    # Load sample data if database is empty
    if stats['total_documents'] == 0:
        print("\n3. Loading sample documents...")
        load_results = engine.load_and_index_documents("sample", num_articles=25, batch_size=10)
        
        if load_results['success']:
            print(f"   ✅ Successfully indexed {load_results['indexed_documents']} documents")
            print(f"   ⏱️ Processing time: {load_results['processing_time_seconds']:.2f} seconds")
        else:
            print(f"   ❌ Failed to load documents: {load_results.get('error', 'Unknown error')}")
            return
    else:
        print(f"\n3. Using existing {stats['total_documents']} documents in database")
    
    return engine


def demo_search_queries(engine):
    """Demonstrate various search queries."""
    print_separator("🔍 SEMANTIC SEARCH DEMO")
    
    # Define test queries with different themes
    test_queries = [
        {
            "query": "artificial intelligence and machine learning",
            "description": "Technology-focused query"
        },
        {
            "query": "climate change environmental impact",
            "description": "Environmental science query"
        },
        {
            "query": "quantum computing research breakthrough",
            "description": "Advanced technology query"
        },
        {
            "query": "space exploration Mars missions",
            "description": "Space science query"
        },
        {
            "query": "renewable energy solar power battery storage",
            "description": "Clean energy query"
        }
    ]
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"{i}. {test_case['description']}")
        
        start_time = time.time()
        results = engine.search(
            query=test_case['query'],
            limit=3,
            similarity_threshold=0.5  # Lower threshold for demo
        )
        search_time = time.time() - start_time
        
        display_search_results(test_case['query'], results, search_time)
        
        if i < len(test_queries):
            input("Press Enter to continue to next query...")


def demo_custom_document():
    """Demonstrate adding a custom document and searching for it."""
    print_separator("➕ CUSTOM DOCUMENT DEMO")
    
    engine = create_search_engine()
    
    print("1. Adding a custom document about neural networks...")
    
    custom_doc = {
        "title": "Deep Learning Neural Networks in Computer Vision",
        "content": """
        Deep learning neural networks have revolutionized computer vision tasks. 
        Convolutional Neural Networks (CNNs) are particularly effective for image recognition, 
        object detection, and image segmentation. These networks use multiple layers of 
        convolution and pooling operations to extract hierarchical features from images. 
        Popular architectures include ResNet, VGG, and EfficientNet, which have achieved 
        state-of-the-art results on benchmark datasets like ImageNet. Transfer learning 
        allows these pre-trained models to be adapted for specific computer vision tasks 
        with limited training data.
        """,
        "source": "demo_custom",
        "metadata": {
            "category": "Technology",
            "tags": ["deep learning", "neural networks", "computer vision", "CNN"],
            "custom_added": True
        }
    }
    
    try:
        doc_id = engine.add_document(**custom_doc)
        print(f"   ✅ Custom document added with ID: {doc_id}")
    except Exception as e:
        print(f"   ❌ Failed to add custom document: {e}")
        return
    
    print("\n2. Searching for the custom document...")
    
    # Search queries that should match the custom document
    search_queries = [
        "neural networks computer vision",
        "deep learning image recognition",
        "convolutional networks CNN",
        "transfer learning ResNet"
    ]
    
    for query in search_queries:
        print(f"\n   Query: '{query}'")
        start_time = time.time()
        results = engine.search(query, limit=2, similarity_threshold=0.3)
        search_time = time.time() - start_time
        
        # Check if our custom document appears in results
        custom_doc_found = any(result['id'] == doc_id for result in results)
        
        if custom_doc_found:
            custom_result = next(result for result in results if result['id'] == doc_id)
            similarity = custom_result['similarity_score'] * 100
            print(f"   ✅ Custom document found! Similarity: {similarity:.1f}%")
        else:
            print(f"   ⚠️ Custom document not in top results (searched {len(results)} results)")
        
        print(f"   ⏱️ Search time: {search_time*1000:.2f}ms")


def demo_similarity_comparison():
    """Demonstrate how similarity scores work with different queries."""
    print_separator("📊 SIMILARITY COMPARISON DEMO")
    
    engine = create_search_engine()
    
    base_query = "artificial intelligence machine learning"
    
    # Queries with varying similarity to the base query
    comparison_queries = [
        ("artificial intelligence machine learning", "Identical query"),
        ("AI and ML algorithms", "Synonymous terms"),
        ("neural networks deep learning", "Related concepts"),
        ("computer science programming", "Broader field"),
        ("climate change global warming", "Unrelated topic")
    ]
    
    print(f"Base query: '{base_query}'")
    print("Comparing similarity scores for different queries:\n")
    
    # Get results for base query
    base_results = engine.search(base_query, limit=3, similarity_threshold=0.3)
    
    if not base_results:
        print("❌ No results found for base query. Make sure documents are loaded.")
        return
    
    # Use the top result as reference
    reference_doc = base_results[0]
    reference_title = reference_doc['title']
    
    print(f"Reference document: '{reference_title}'\n")
    
    for query, description in comparison_queries:
        results = engine.search(query, limit=1, similarity_threshold=0.1)
        
        if results:
            # Find the reference document in results
            ref_result = None
            for result in engine.search(query, limit=10, similarity_threshold=0.1):
                if result['id'] == reference_doc['id']:
                    ref_result = result
                    break
            
            if ref_result:
                similarity = ref_result['similarity_score'] * 100
                print(f"'{query}' -> {similarity:.1f}% similarity ({description})")
            else:
                print(f"'{query}' -> <10% similarity ({description})")
        else:
            print(f"'{query}' -> No results found ({description})")


def demo_performance_metrics():
    """Demonstrate performance characteristics."""
    print_separator("⚡ PERFORMANCE METRICS DEMO")
    
    engine = create_search_engine()
    
    print("1. Database statistics:")
    stats = engine.get_database_stats()
    print(f"   📊 Total documents: {stats['total_documents']}")
    print(f"   🧠 Embedding dimension: {stats['embedding_dimension']}")
    
    print("\n2. Search performance test:")
    
    test_queries = [
        "machine learning artificial intelligence",
        "climate change environment",
        "quantum computing physics",
        "space exploration astronomy",
        "renewable energy sustainability"
    ]
    
    total_time = 0
    total_searches = len(test_queries)
    
    for i, query in enumerate(test_queries, 1):
        start_time = time.time()
        results = engine.search(query, limit=5, similarity_threshold=0.5)
        search_time = time.time() - start_time
        total_time += search_time
        
        print(f"   Search {i}: {search_time*1000:.2f}ms ({len(results)} results)")
    
    avg_time = total_time / total_searches
    print(f"\n📈 Performance Summary:")
    print(f"   Average search time: {avg_time*1000:.2f}ms")
    print(f"   Total test time: {total_time:.3f}s")
    print(f"   Searches per second: {total_searches/total_time:.1f}")


def main():
    """Main demo function."""
    print_separator("🎯 AI-POWERED SEMANTIC SEARCH ENGINE DEMO")
    
    print("Welcome to the Semantic Search Engine Demo!")
    print("This demo will showcase various features of the search engine.")
    print("\nMake sure you've run setup.py first to initialize the database.")
    
    try:
        # Basic functionality demo
        engine = demo_basic_functionality()
        if not engine:
            print("❌ Failed to initialize engine. Exiting demo.")
            return
        
        input("\nPress Enter to continue to search demo...")
        
        # Search queries demo
        demo_search_queries(engine)
        
        input("\nPress Enter to continue to custom document demo...")
        
        # Custom document demo
        demo_custom_document()
        
        input("\nPress Enter to continue to similarity comparison demo...")
        
        # Similarity comparison demo
        demo_similarity_comparison()
        
        input("\nPress Enter to continue to performance metrics...")
        
        # Performance metrics demo
        demo_performance_metrics()
        
        print_separator("🎉 DEMO COMPLETED")
        
        print("Demo completed successfully! 🎉")
        print("\nNext steps:")
        print("• Try the web interfaces:")
        print("  - FastAPI: python fastapi_app.py")
        print("  - Streamlit: streamlit run streamlit_app.py")
        print("• Experiment with your own documents and queries")
        print("• Check the README.md for more advanced usage")
        
    except KeyboardInterrupt:
        print("\n\n❌ Demo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
