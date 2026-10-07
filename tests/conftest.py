"""Shared fixtures. Database tests need a PostgreSQL server with the pgvector extension.

Point them at one with the usual POSTGRES_HOST / POSTGRES_PORT / POSTGRES_USER / POSTGRES_PASSWORD
variables (the CI workflow uses a pgvector service container). Each test session creates and
drops its own throw-away database, so nothing you already have is touched. Without a reachable
server those tests are skipped.
"""
import importlib
import os
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The offline embedding backend: no model download, no API key
os.environ["EMBEDDING_MODEL"] = "hashing"
os.environ["EMBEDDING_DIMENSION"] = "384"


def _admin_conn():
    import psycopg2

    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
        dbname="postgres",
        connect_timeout=3,
    )


@pytest.fixture(scope="session")
def test_database():
    """Create a throw-away database with the pgvector extension; yield its name."""
    try:
        conn = _admin_conn()
    except Exception as exc:  # no server reachable
        pytest.skip(f"PostgreSQL not reachable: {exc}")
    conn.autocommit = True
    name = f"semsearch_test_{uuid.uuid4().hex[:8]}"
    cur = conn.cursor()
    try:
        cur.execute(f'CREATE DATABASE "{name}"')
    except Exception as exc:
        conn.close()
        pytest.skip(f"cannot create a test database: {exc}")
    try:
        yield name
    finally:
        cur.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        conn.close()


@pytest.fixture(scope="session")
def app_modules(test_database):
    """Import the project modules *after* the environment points at the test database."""
    os.environ["POSTGRES_DB"] = test_database
    os.environ.pop("DATABASE_URL", None)
    mods = {}
    for name in ("database", "embeddings", "document_loader", "semantic_search", "fastapi_app"):
        sys.modules.pop(name, None)
        mods[name] = importlib.import_module(name)
    return mods


@pytest.fixture()
def engine(app_modules):
    """A search engine with a freshly emptied index containing the 10 sample articles."""
    eng = app_modules["semantic_search"].create_search_engine()
    eng.clear_database()
    result = eng.load_and_index_documents("sample", num_articles=10)
    assert result["success"] and result["failed_documents"] == 0
    return eng
