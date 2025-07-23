"""
Streamlit web application for semantic search engine.
Provides an interactive web interface for document indexing and similarity search.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any
import time
import logging

from semantic_search import SemanticSearchEngine, create_search_engine
from document_loader import create_document_loader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI-Powered Semantic Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .search-result {
        border: 1px solid #ddd;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
        background-color: #f9f9f9;
    }
    .result-title {
        font-weight: bold;
        color: #2e7d32;
        font-size: 1.1em;
    }
    .similarity-score {
        color: #1976d2;
        font-weight: bold;
        float: right;
        background: #e3f2fd;
        padding: 0.2rem 0.5rem;
        border-radius: 1rem;
        font-size: 0.9em;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_search_engine():
    """Initialize and cache the search engine."""
    try:
        with st.spinner("Initializing semantic search engine..."):
            return create_search_engine()
    except Exception as e:
        st.error(f"Failed to initialize search engine: {e}")
        return None


def display_database_stats(engine: SemanticSearchEngine):
    """Display database statistics."""
    try:
        stats = engine.get_database_stats()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="📚 Total Documents",
                value=stats.get("total_documents", 0)
            )
        
        with col2:
            st.metric(
                label="🧠 Embedding Dimension",
                value=stats.get("embedding_dimension", 0)
            )
        
        with col3:
            status = stats.get("database_status", "unknown")
            st.metric(
                label="🔗 Database Status",
                value=status.title(),
                delta="Connected" if status == "connected" else "Disconnected"
            )
            
    except Exception as e:
        st.error(f"Failed to load database statistics: {e}")


def load_sample_documents(engine: SemanticSearchEngine):
    """Load sample documents into the database."""
    st.subheader("📚 Load Sample Documents")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        num_articles = st.slider(
            "Number of sample articles to load:",
            min_value=5,
            max_value=100,
            value=20,
            step=5
        )
    
    with col2:
        batch_size = st.selectbox(
            "Batch size:",
            options=[5, 10, 20, 50],
            index=1
        )
    
    if st.button("🚀 Load Sample Data", type="primary"):
        with st.spinner(f"Loading and indexing {num_articles} sample documents..."):
            progress_bar = st.progress(0)
            
            try:
                # Load and index documents
                results = engine.load_and_index_documents(
                    source_type="sample",
                    batch_size=batch_size,
                    num_articles=num_articles
                )
                
                progress_bar.progress(1.0)
                
                if results["success"]:
                    st.success(
                        f"✅ Successfully indexed {results['indexed_documents']}/{results['total_documents']} "
                        f"documents in {results['processing_time_seconds']:.2f} seconds!"
                    )
                    
                    # Display results metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("📝 Total Documents", results["total_documents"])
                    with col2:
                        st.metric("✅ Successfully Indexed", results["indexed_documents"])
                    with col3:
                        st.metric("⏱️ Processing Time", f"{results['processing_time_seconds']:.2f}s")
                    
                    st.rerun()  # Refresh the app to update stats
                else:
                    st.error(f"❌ Indexing failed: {results.get('error', 'Unknown error')}")
                    
            except Exception as e:
                st.error(f"❌ Failed to load documents: {e}")
            finally:
                progress_bar.empty()


def perform_semantic_search(engine: SemanticSearchEngine):
    """Perform semantic search interface."""
    st.subheader("🔍 Semantic Search")
    
    # Search input section
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., artificial intelligence, climate change, quantum computing...",
            help="Enter natural language queries to find semantically similar documents"
        )
    
    with col2:
        limit = st.number_input(
            "Max Results:",
            min_value=1,
            max_value=20,
            value=5,
            step=1
        )
    
    with col3:
        similarity_threshold = st.slider(
            "Min Similarity:",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Minimum similarity score (0.0 to 1.0)"
        )
    
    # Example queries
    st.write("**💡 Try these example queries:**")
    example_queries = [
        "artificial intelligence and machine learning",
        "climate change environmental impact",
        "quantum computing research breakthrough",
        "space exploration Mars missions",
        "renewable energy solar battery"
    ]
    
    example_cols = st.columns(len(example_queries))
    for i, example in enumerate(example_queries):
        with example_cols[i]:
            if st.button(f"🔸 {example.split()[0].title()}", key=f"example_{i}"):
                st.session_state.example_query = example
    
    # Use example query if selected
    if 'example_query' in st.session_state:
        query = st.session_state.example_query
        del st.session_state.example_query
        st.rerun()
    
    # Perform search
    if query and st.button("🚀 Search", type="primary"):
        with st.spinner("Searching for similar documents..."):
            start_time = time.time()
            
            try:
                results = engine.search(
                    query=query,
                    limit=limit,
                    similarity_threshold=similarity_threshold
                )
                
                search_time = time.time() - start_time
                
                # Display search results
                if results:
                    st.success(f"✅ Found {len(results)} results in {search_time*1000:.2f}ms")
                    
                    # Create visualization of similarity scores
                    if len(results) > 1:
                        scores_df = pd.DataFrame([
                            {"Document": f"{r['title'][:30]}..." if len(r['title']) > 30 else r['title'],
                             "Similarity Score": r['similarity_score']}
                            for r in results
                        ])
                        
                        fig = px.bar(
                            scores_df,
                            x="Similarity Score",
                            y="Document",
                            orientation='h',
                            title="Document Similarity Scores",
                            color="Similarity Score",
                            color_continuous_scale="viridis"
                        )
                        fig.update_layout(height=300, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Display individual results
                    st.subheader("📋 Search Results")
                    
                    for i, result in enumerate(results, 1):
                        with st.container():
                            st.markdown(f"### {i}. {result['title']}")
                            
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.write(f"**Similarity Score:** {result['similarity_score']:.3f} ({result['similarity_score']*100:.1f}%)")
                                st.write(f"**Source:** {result.get('source', 'Unknown')}")
                                
                                if result.get('metadata'):
                                    metadata_str = " | ".join([f"{k}: {v}" for k, v in result['metadata'].items() if k != 'generated'])
                                    if metadata_str:
                                        st.write(f"**Metadata:** {metadata_str}")
                            
                            with col2:
                                # Similarity gauge
                                fig = go.Figure(go.Indicator(
                                    mode="gauge+number",
                                    value=result['similarity_score'],
                                    domain={'x': [0, 1], 'y': [0, 1]},
                                    title={'text': "Similarity"},
                                    gauge={
                                        'axis': {'range': [None, 1]},
                                        'bar': {'color': "darkblue"},
                                        'steps': [
                                            {'range': [0, 0.5], 'color': "lightgray"},
                                            {'range': [0.5, 0.8], 'color': "yellow"},
                                            {'range': [0.8, 1], 'color': "green"}
                                        ],
                                        'threshold': {
                                            'line': {'color': "red", 'width': 4},
                                            'thickness': 0.75,
                                            'value': 0.9
                                        }
                                    }
                                ))
                                fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Content preview
                            with st.expander("📄 View Content", expanded=False):
                                st.text_area(
                                    "Document Content:",
                                    value=result['content'],
                                    height=200,
                                    disabled=True,
                                    key=f"content_{i}"
                                )
                            
                            st.divider()
                else:
                    st.warning("🔍 No results found. Try adjusting your query or lowering the similarity threshold.")
                    
            except Exception as e:
                st.error(f"❌ Search failed: {e}")


def add_custom_document(engine: SemanticSearchEngine):
    """Interface to add custom documents."""
    st.subheader("➕ Add Custom Document")
    
    with st.form("add_document_form"):
        title = st.text_input("Document Title:", placeholder="Enter document title...")
        
        content = st.text_area(
            "Document Content:",
            placeholder="Enter document content...",
            height=200
        )
        
        source = st.text_input("Source (optional):", placeholder="e.g., website, book, article...")
        
        # Metadata input
        st.write("**Metadata (optional):**")
        col1, col2 = st.columns(2)
        
        with col1:
            category = st.text_input("Category:", placeholder="e.g., Technology, Science...")
        
        with col2:
            tags = st.text_input("Tags (comma-separated):", placeholder="e.g., AI, machine learning")
        
        submitted = st.form_submit_button("📝 Add Document", type="primary")
        
        if submitted:
            if not title or not content:
                st.error("❌ Title and content are required!")
            else:
                try:
                    # Prepare metadata
                    metadata = {}
                    if category:
                        metadata["category"] = category
                    if tags:
                        metadata["tags"] = [tag.strip() for tag in tags.split(",")]
                    metadata["custom_added"] = True
                    
                    with st.spinner("Adding document to search index..."):
                        doc_id = engine.add_document(
                            title=title,
                            content=content,
                            source=source or "custom",
                            metadata=metadata
                        )
                    
                    st.success(f"✅ Document added successfully! ID: {doc_id}")
                    st.rerun()  # Refresh to update stats
                    
                except Exception as e:
                    st.error(f"❌ Failed to add document: {e}")


def manage_database(engine: SemanticSearchEngine):
    """Database management interface."""
    st.subheader("🗄️ Database Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Refresh Stats", type="secondary"):
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear All Documents", type="secondary"):
            if st.session_state.get("confirm_clear", False):
                try:
                    with st.spinner("Clearing all documents..."):
                        engine.clear_database()
                    st.success("✅ All documents cleared successfully!")
                    st.session_state.confirm_clear = False
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to clear documents: {e}")
            else:
                st.session_state.confirm_clear = True
                st.warning("⚠️ Click again to confirm deletion of all documents!")


def main():
    """Main Streamlit application."""
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 AI-Powered Semantic Search Engine</h1>
        <p>Search documents using advanced semantic similarity powered by PostgreSQL and pgvector</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize search engine
    engine = initialize_search_engine()
    
    if engine is None:
        st.error("❌ Failed to initialize search engine. Please check your configuration.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # Database statistics
        st.subheader("📊 Database Statistics")
        display_database_stats(engine)
        
        st.divider()
        
        # Navigation
        page = st.selectbox(
            "📖 Navigate to:",
            options=[
                "🔍 Search Documents",
                "📚 Load Sample Data",
                "➕ Add Custom Document",
                "🗄️ Database Management"
            ]
        )
        
        st.divider()
        
        # Configuration info
        st.subheader("⚙️ Configuration")
        st.info(f"**Embedding Model:** {engine.embedding_generator.__class__.__name__}")
        st.info(f"**Embedding Dimension:** {engine.embedding_dimension}")
    
    # Main content based on navigation
    if page == "🔍 Search Documents":
        perform_semantic_search(engine)
    
    elif page == "📚 Load Sample Data":
        load_sample_documents(engine)
    
    elif page == "➕ Add Custom Document":
        add_custom_document(engine)
    
    elif page == "🗄️ Database Management":
        manage_database(engine)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🚀 **Powered by:** PostgreSQL + pgvector + Sentence Transformers + Streamlit",
        help="This semantic search engine uses PostgreSQL with the pgvector extension for efficient vector similarity search."
    )


if __name__ == "__main__":
    main()
