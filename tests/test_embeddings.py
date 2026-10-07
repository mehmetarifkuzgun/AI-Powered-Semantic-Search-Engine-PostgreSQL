import numpy as np

from embeddings import HashingEmbedding, get_embedding_generator


def cos(a, b):
    return float(np.dot(a, b))


def test_dimension_normalised_and_deterministic():
    emb = HashingEmbedding(384)
    v1, v2 = emb.generate_embedding("Quantum computing research"), emb.generate_embedding("Quantum computing research")
    assert len(v1) == emb.get_dimension() == 384
    assert v1 == v2
    assert abs(np.linalg.norm(v1) - 1.0) < 1e-9


def test_related_text_scores_higher_than_unrelated():
    emb = HashingEmbedding()
    q = emb.generate_embedding("renewable energy storage batteries")
    related = emb.generate_embedding("New battery technologies make renewable energy storage viable")
    unrelated = emb.generate_embedding("Marine scientists document coral reef biodiversity")
    assert cos(q, related) > cos(q, unrelated) + 0.2


def test_empty_text_is_still_a_valid_unit_vector():
    v = HashingEmbedding().generate_embedding("")
    assert abs(np.linalg.norm(v) - 1.0) < 1e-9


def test_batch_matches_single():
    emb = HashingEmbedding()
    texts = ["alpha beta", "gamma delta"]
    assert emb.generate_embeddings(texts) == [emb.generate_embedding(t) for t in texts]


def test_env_selects_hashing_backend():
    # conftest sets EMBEDDING_MODEL=hashing
    assert isinstance(get_embedding_generator(), HashingEmbedding)
