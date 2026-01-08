-- Unfold Database Initialization Script
-- Run during first PostgreSQL container startup

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE unfold TO unfold;

-- Create schema
CREATE SCHEMA IF NOT EXISTS unfold;

-- Set default search path
ALTER DATABASE unfold SET search_path TO unfold, public;

-- Log initialization
DO $$
BEGIN
    RAISE NOTICE 'Unfold database initialized successfully at %', NOW();
END $$;
