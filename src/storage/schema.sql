-- Storage for extracted bills and their text chunks. Idempotent: safe to run again.
-- The read-only role used for question answering is created by db.py, because its
-- name and password come from the environment.

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per extracted bill. Field names and meanings match src/schema.py.
CREATE TABLE IF NOT EXISTS bills (
    id                   TEXT PRIMARY KEY,          -- bill id, e.g. 'bill_0003'
    source_filename      TEXT NOT NULL,             -- the PDF the bill was extracted from
    template             TEXT,                      -- generator layout; NULL for real bills
    provider             TEXT NOT NULL,             -- company that issued the bill
    account_number       TEXT NOT NULL,             -- customer's account id, as printed
    billing_period_start DATE NOT NULL,             -- first day of service covered
    billing_period_end   DATE NOT NULL,             -- last day of service covered
    due_date             DATE,                      -- payment deadline; NULL if none printed
    usage_amount         NUMERIC,                   -- metered consumption; NULL for flat-fee services
    usage_unit           TEXT,                      -- 'kWh' (electricity) or 'm³' (gas, water); NULL with usage_amount
    total_amount         NUMERIC(12, 2) NOT NULL,   -- total due; equals the sum of the bill's line items
    currency             CHAR(3) NOT NULL,          -- ISO 4217 code: 'USD' or 'CAD'
    extraction_model     TEXT NOT NULL              -- model that extracted the row
);

-- The charges and credits on each bill, in printed order. Credits are negative.
CREATE TABLE IF NOT EXISTS line_items (
    bill_id     TEXT NOT NULL REFERENCES bills (id) ON DELETE CASCADE,
    position    INTEGER NOT NULL,                   -- 1-based order on the bill
    description TEXT NOT NULL,                      -- label as printed, e.g. 'Delivery charge', 'Previous balance'
    amount      NUMERIC(12, 2) NOT NULL,            -- negative for credits, discounts, adjustments
    PRIMARY KEY (bill_id, position)
);

-- Text chunks of each bill's PDF, for semantic search. 384 dimensions matches the
-- default embedding model (all-MiniLM-L6-v2); db.EMBEDDING_DIM must agree.
CREATE TABLE IF NOT EXISTS chunks (
    bill_id     TEXT NOT NULL REFERENCES bills (id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL CHECK (page_number >= 1),  -- 1-based PDF page, for citations
    chunk_index INTEGER NOT NULL,                   -- order of the chunk within the bill
    heading     TEXT NOT NULL,                      -- bill identity prepended before embedding
    text        TEXT NOT NULL,                      -- the chunk's text as extracted from the PDF
    embedding   vector(384) NOT NULL,               -- normalised embedding of heading + text
    PRIMARY KEY (bill_id, chunk_index)
);

-- Vector index: HNSW with cosine distance.
-- * Cosine (vector_cosine_ops) because retrieval orders by `embedding <=> query`, and
--   sentence-transformer embeddings are compared by angle, not magnitude.
-- * HNSW over IVFFlat because IVFFlat clusters the rows that exist when the index is
--   built: built on an empty or small table, its lists are poor and recall suffers
--   until it is rebuilt. HNSW needs no training step, can be created before any data
--   is loaded, and keeps high recall as rows are added. Its costs (slower builds, more
--   memory) don't matter at this scale.
-- * At a few hundred chunks, Postgres may well choose an exact sequential scan instead;
--   the index is there so the same schema holds when the collection grows.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);
