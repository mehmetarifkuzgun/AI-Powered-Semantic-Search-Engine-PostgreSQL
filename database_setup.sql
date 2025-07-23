-- SQL script to set up PostgreSQL database for semantic search
-- Run this script as a PostgreSQL superuser

-- Create database (if not exists)
CREATE DATABASE semantic_search_db;

-- Connect to the database
\c semantic_search_db;

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create user for the application (optional)
-- CREATE USER semantic_search_user WITH PASSWORD 'your_password_here';
-- GRANT ALL PRIVILEGES ON DATABASE semantic_search_db TO semantic_search_user;

-- Verify pgvector installation
SELECT * FROM pg_extension WHERE extname = 'vector';

-- Show available vector operators
\do *<->*
\do *<#>*
\do *<=>*

-- The documents table will be created automatically by the Python application
-- But you can create it manually if needed:

/*
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    title VARCHAR(500),
    content TEXT NOT NULL,
    source VARCHAR(200),
    metadata JSONB,
    embedding vector(384),  -- Adjust dimension based on your model
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS documents_embedding_idx 
ON documents USING ivfflat (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS documents_content_idx 
ON documents USING gin(to_tsvector('english', content));

-- Create trigger for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
*/

-- Check that everything is working
SELECT version();
SELECT * FROM pg_available_extensions WHERE name = 'vector';
