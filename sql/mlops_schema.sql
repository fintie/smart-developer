-- Smart Developer MLOps Schema
-- PostgreSQL 16+
-- Purpose:
--   - retrieval request logging
--   - retrieval result logging
--   - user feedback logging
--   - report job tracking
--   - model registry / model version tracking

BEGIN;

-- =========================
-- Extensions
-- =========================

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- =========================
-- 1. Retrieval Requests
-- =========================

CREATE TABLE IF NOT EXISTS retrieval_requests (
    request_id TEXT PRIMARY KEY,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    user_id TEXT,
    session_id TEXT,

    strategy TEXT NOT NULL,
    query_text TEXT NOT NULL,

    locality TEXT,
    address_contains TEXT,

    top_k INTEGER NOT NULL,
    recall_k INTEGER NOT NULL,

    retrieval_model TEXT NOT NULL,
    reranking_model TEXT,

    use_dcn_reranker BOOLEAN NOT NULL DEFAULT TRUE,
    with_explanations BOOLEAN NOT NULL DEFAULT FALSE,
    dedupe_by_address BOOLEAN NOT NULL DEFAULT TRUE,

    latency_ms DOUBLE PRECISION,
    result_count INTEGER,

    request_payload JSONB,
    planner_payload JSONB,
    metadata_payload JSONB
);

CREATE INDEX IF NOT EXISTS idx_retrieval_requests_created_at
ON retrieval_requests(created_at);

CREATE INDEX IF NOT EXISTS idx_retrieval_requests_strategy
ON retrieval_requests(strategy);

CREATE INDEX IF NOT EXISTS idx_retrieval_requests_user_id
ON retrieval_requests(user_id);

CREATE INDEX IF NOT EXISTS idx_retrieval_requests_session_id
ON retrieval_requests(session_id);

CREATE INDEX IF NOT EXISTS idx_retrieval_requests_retrieval_model
ON retrieval_requests(retrieval_model);

CREATE INDEX IF NOT EXISTS idx_retrieval_requests_reranking_model
ON retrieval_requests(reranking_model);

CREATE INDEX IF NOT EXISTS idx_retrieval_requests_locality
ON retrieval_requests(locality);


-- =========================
-- 2. Retrieval Results
-- =========================

CREATE TABLE IF NOT EXISTS retrieval_results (
    id BIGSERIAL PRIMARY KEY,

    request_id TEXT NOT NULL
        REFERENCES retrieval_requests(request_id)
        ON DELETE CASCADE,

    rid TEXT NOT NULL,
    rank_position INTEGER NOT NULL,

    address TEXT,
    base_site_address TEXT,

    primary_zoning_code TEXT,
    zoning_band TEXT,
    lot_size_band TEXT,
    constraint_severity_band TEXT,
    station_distance_band TEXT,

    strategy_score DOUBLE PRECISION,
    retrieval_similarity DOUBLE PRECISION,
    fusion_score DOUBLE PRECISION,
    dcn_prob DOUBLE PRECISION,
    final_rank_score DOUBLE PRECISION,

    result_payload JSONB,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_retrieval_result_request_rid_rank
        UNIQUE (request_id, rid, rank_position)
);

CREATE INDEX IF NOT EXISTS idx_retrieval_results_request_id
ON retrieval_results(request_id);

CREATE INDEX IF NOT EXISTS idx_retrieval_results_rid
ON retrieval_results(rid);

CREATE INDEX IF NOT EXISTS idx_retrieval_results_rank_position
ON retrieval_results(rank_position);

CREATE INDEX IF NOT EXISTS idx_retrieval_results_base_site_address
ON retrieval_results(base_site_address);

CREATE INDEX IF NOT EXISTS idx_retrieval_results_strategy_score
ON retrieval_results(strategy_score);

CREATE INDEX IF NOT EXISTS idx_retrieval_results_dcn_prob
ON retrieval_results(dcn_prob);


-- =========================
-- 3. User Feedback
-- =========================

CREATE TABLE IF NOT EXISTS user_feedback (
    feedback_id TEXT PRIMARY KEY,

    request_id TEXT NOT NULL
        REFERENCES retrieval_requests(request_id)
        ON DELETE CASCADE,

    rid TEXT,
    rank_position INTEGER,

    event_type TEXT NOT NULL,
    event_value JSONB,

    user_note TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_feedback_request_id
ON user_feedback(request_id);

CREATE INDEX IF NOT EXISTS idx_user_feedback_rid
ON user_feedback(rid);

CREATE INDEX IF NOT EXISTS idx_user_feedback_event_type
ON user_feedback(event_type);

CREATE INDEX IF NOT EXISTS idx_user_feedback_created_at
ON user_feedback(created_at);

CREATE INDEX IF NOT EXISTS idx_user_feedback_request_rid
ON user_feedback(request_id, rid);


-- =========================
-- 4. Report Jobs
-- =========================

CREATE TABLE IF NOT EXISTS report_jobs (
    report_id TEXT PRIMARY KEY,

    request_id TEXT NOT NULL
        REFERENCES retrieval_requests(request_id)
        ON DELETE CASCADE,

    status TEXT NOT NULL,

    explanation_mode TEXT NOT NULL,

    output_markdown_path TEXT,
    output_pdf_path TEXT,

    latency_ms DOUBLE PRECISION,
    error_message TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,

    report_payload JSONB,

    CONSTRAINT chk_report_jobs_status
        CHECK (status IN ('queued', 'running', 'ready', 'failed', 'cancelled'))
);

CREATE INDEX IF NOT EXISTS idx_report_jobs_request_id
ON report_jobs(request_id);

CREATE INDEX IF NOT EXISTS idx_report_jobs_status
ON report_jobs(status);

CREATE INDEX IF NOT EXISTS idx_report_jobs_created_at
ON report_jobs(created_at);


-- =========================
-- 5. Model Registry
-- =========================

CREATE TABLE IF NOT EXISTS model_registry (
    model_version TEXT PRIMARY KEY,

    model_type TEXT NOT NULL,

    artifact_path TEXT NOT NULL,
    preprocessing_path TEXT,

    status TEXT NOT NULL,

    metrics JSONB,
    model_card JSONB,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    promoted_at TIMESTAMPTZ,

    notes TEXT,

    CONSTRAINT chk_model_registry_status
        CHECK (status IN ('candidate', 'staging', 'production', 'archived'))
);

CREATE INDEX IF NOT EXISTS idx_model_registry_model_type
ON model_registry(model_type);

CREATE INDEX IF NOT EXISTS idx_model_registry_status
ON model_registry(status);

CREATE INDEX IF NOT EXISTS idx_model_registry_created_at
ON model_registry(created_at);


-- =========================
-- 6. Optional Seed Production Models
-- =========================

INSERT INTO model_registry (
    model_version,
    model_type,
    artifact_path,
    preprocessing_path,
    status,
    notes
)
VALUES
(
    'two_tower_v1',
    'two_tower_retrieval',
    'algorithm/artifacts/models/two_tower_v1/model.pt',
    NULL,
    'production',
    'Current default retrieval model for intent-to-site recall.'
),
(
    'dcn_reranker_v1',
    'dcn_reranker',
    'algorithm/artifacts/models/dcn_reranker_v1/model.pt',
    'algorithm/artifacts/models/dcn_reranker_v1/preprocessing.json',
    'production',
    'Current default DCN reranking model for second-stage site ranking.'
)
ON CONFLICT (model_version) DO UPDATE SET
    model_type = EXCLUDED.model_type,
    artifact_path = EXCLUDED.artifact_path,
    preprocessing_path = EXCLUDED.preprocessing_path,
    status = EXCLUDED.status,
    notes = EXCLUDED.notes;


COMMIT;