"""Integration tests against a real PostgreSQL + pgvector (skipped when none is reachable)."""
import pytest

QUERIES = {
    "artificial intelligence and machine learning": "Artificial Intelligence in Healthcare Diagnostics",
    "battery storage for renewable energy": "Renewable Energy Storage Solutions",
    "exploring Mars": "Space Exploration and Mars Missions",
    "protecting the oceans and coral reefs": "Ocean Conservation and Marine Biodiversity",
    "gene editing for rare diseases": "Gene Therapy Advances in Rare Diseases",
}


@pytest.mark.parametrize("query,expected_title", QUERIES.items())
def test_top_result_is_the_matching_article(engine, query, expected_title):
    results = engine.search(query, limit=3, similarity_threshold=0.0)
    assert results[0]["title"] == expected_title
    scores = [r["similarity_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_threshold_filters_results(engine):
    q = "gene editing for rare diseases"
    assert len(engine.search(q, limit=10, similarity_threshold=0.0)) > 1
    strict = engine.search(q, limit=10, similarity_threshold=0.3)
    assert [r["title"] for r in strict] == ["Gene Therapy Advances in Rare Diseases"]
    assert engine.search(q, limit=10, similarity_threshold=0.99) == []


def test_metadata_roundtrips_as_json(engine):
    doc_id = engine.add_document("Custom", "A custom note about volcano monitoring",
                                 source="manual", metadata={"tags": ["a", "b"], "n": 3})
    hit = engine.search("volcano monitoring", limit=1, similarity_threshold=0.0)[0]
    assert hit["id"] == doc_id
    assert hit["metadata"] == {"tags": ["a", "b"], "n": 3}
    assert hit["source"] == "manual"


def test_stats_and_clear(engine):
    assert engine.get_database_stats()["total_documents"] == 10
    engine.clear_database()
    assert engine.get_database_stats()["total_documents"] == 0


def test_dimension_mismatch_is_reported_clearly(app_modules, engine):
    with pytest.raises(ValueError, match="vector\\(384\\)"):
        app_modules["database"].get_db_manager().create_documents_table(embedding_dimension=768)


def test_hnsw_index_exists(app_modules, engine):
    db = app_modules["database"].get_db_manager()
    with db.get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT indexdef FROM pg_indexes WHERE indexname = 'documents_embedding_idx'")
        assert "hnsw" in cur.fetchone()["indexdef"]
