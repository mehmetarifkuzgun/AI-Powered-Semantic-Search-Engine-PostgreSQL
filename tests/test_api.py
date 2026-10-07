import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client(app_modules, engine):
    with TestClient(app_modules["fastapi_app"].app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "healthy"


def test_stats(client):
    body = client.get("/api/stats").json()
    assert body["total_documents"] == 10 and body["embedding_dimension"] == 384


def test_search_returns_ranked_results(client):
    r = client.post("/api/search", json={"query": "exploring Mars", "limit": 3, "similarity_threshold": 0.0})
    assert r.status_code == 200
    body = r.json()
    assert body["results"][0]["title"] == "Space Exploration and Mars Missions"
    assert body["total_results"] == len(body["results"])


def test_zero_threshold_is_honoured_not_replaced_by_default(client):
    # Regression: `similarity_threshold or 0.7` turned an explicit 0.0 into 0.7
    r = client.post("/api/search", json={"query": "exploring Mars", "limit": 5, "similarity_threshold": 0.0})
    default = client.post("/api/search", json={"query": "exploring Mars", "limit": 5})
    assert r.json()["total_results"] > default.json()["total_results"]


def test_validation(client):
    assert client.post("/api/search", json={"query": "x", "limit": 0}).status_code == 422
    assert client.post("/api/search", json={"query": "x", "similarity_threshold": 1.5}).status_code == 422


def test_add_and_clear_documents(client):
    r = client.post("/api/documents", json={"title": "T", "content": "volcano monitoring network",
                                            "metadata": {"k": "v"}})
    assert r.status_code == 200 and r.json()["success"]
    assert client.get("/api/stats").json()["total_documents"] == 11
    assert client.delete("/api/documents").json()["success"]
    assert client.get("/api/stats").json()["total_documents"] == 0
