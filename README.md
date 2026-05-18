cat > README.md <<'MD'
# Smart Developer

Smart Developer is an AI-powered site recommendation and development feasibility platform built for NextGenius. It ranks candidate property sites for different redevelopment strategies, explains why sites are suitable, logs user interactions, and generates report outputs for product/demo workflows.

The system is designed as a lightweight ML product platform rather than a standalone algorithm script.

## Current Features

- Two-stage site recommendation pipeline:
  - two-tower retrieval model
  - DCN-style reranker
- FastAPI algorithm service with warm model loading
- Product-facing backend gateway
- React/Vite demo frontend
- PostgreSQL logging for:
  - retrieval requests
  - retrieval results
  - user feedback
  - report jobs
  - model registry
- Deterministic template explanations for low-latency site-card rationale
- Markdown/PDF report generation from logged request results
- Feedback dataset export for future reranker training

## Architecture

```text
React frontend
    |
    v
Backend gateway FastAPI service
    |
    v
Algorithm FastAPI service
    |
    v
PostgreSQL
```

## Services
|**Service**|**Port**|**Purpose**|
|-----------|--------|-----------|
|Frontend|5173|Demo web UI|
|Backend gateway|8002|Product-facing API wrapper|
|Algorithm service|8001|Retrieval, reranking, explanations, MLOps logging|
|PostgreSQL|55435|Local MLOps database|

## Repository Structure
```text
smart-developer/
├── algorithm/
│   ├── src/
│   │   ├── inference/
│   │   ├── retrieval/
│   │   ├── explanation/
│   │   ├── mlops/
│   │   └── serving/
│   └── artifacts/
│
├── backend/
│   └── app/
│       ├── main.py
│       ├── schemas.py
│       └── services/
│
├── frontend/
│   └── src/
│       ├── App.tsx
│       ├── App.css
│       └── api.ts
│
├── data/
├── scripts/
├── docker-compose.yml
├── .env.example
└── README.md
```

## Required Local Artifacts
The current local demo assumes the following processed data and model artifacts already exist:
```text
data/processed/retrieval/candidate_sites.parquet
data/processed/retrieval/query_intents.jsonl

algorithm/artifacts/models/two_tower_v1/model.pt
algorithm/artifacts/models/dcn_reranker_v1/model.pt
algorithm/artifacts/models/dcn_reranker_v1/preprocessing.json
```

Model cards may also be stored under:
```text
algorithm/artifacts/models/two_tower_v1/model_card.json
algorithm/artifacts/models/dcn_reranker_v1/model_card.json
```

## Environment Setup
Create a `.env` file from the example:
```bash
cp .env.example .env
```

Example database URL:
```env
DATABASE_URL=postgresql+psycopg://smart_dev:smart_dev_password@localhost:55435/smart_developer
ALGORITHM_SERVICE_URL=http://localhost:8001
```

Install Python dependencies in your virtual environment:
```bash
pip install -r requirements.txt
pip install fastapi "uvicorn[standard]" httpx pydantic python-dotenv
```

Install UI dependencies:
```bash
cd frontend
npm install
cd ..
```

## Database Setup
Start PostgreSQL:
```bash
docker compose up -d postgres
```

Initialise database tables:
```bash
python -m algorithm.src.mlops.init_db
```

Seed the model registry:
```bash
python -m algorithm.src.mlops.seed_model_registry
```

Check registered models:
```bash
psql "postgresql://smart_dev:smart_dev_password@localhost:55435/smart_developer" \
  -c "SELECT model_version, model_type, status, artifact_path FROM model_registry ORDER BY created_at DESC;"
```

## Running the Demo Platform
The demo platform requires 4 running components:
1. PostgreSQL
2. Algorithm service
3. Backend gateway
4. Frontend

### Terminal 1: Start PostgreSQL
```bash
docker compose up -d postgres
```

### Terminal 2: Start Algorithm Service
```bash
uvicorn algorithm.src.serving.api:app --host 0.0.0.0 --port 8001
```

Wait until the model is loaded and the service is ready. Cold start can take around 1-2 minutes locally.

Check health:
```bash
curl http://localhost:8001/health | python -m json.tool
```

Expected status is something like:
```json
{
  "status": "ready"
}
```

### Terminal 3: Start Backend Gateway
```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8002
```

Check backend health:
```bash
curl http://localhost:8002/health | python -m json.tool
```

### Terminal 4: Start Frontend
```bash
cd frontend
npm run dev
```

Open the demo UI:
```text
http://localhost:5173
```

### Demo Flow
The frontend supports the following product style flow:
```text
Search sites
    -> view ranked site cards
    -> click/save/dismiss results
    -> generate markdown/PDF report
```

Recommended demo query:
```text
Strategy:
Low-rise apartment

Locality:
Wolli Creek

Query:
I want a site for low-rise apartment redevelopment near a train station, with high development zoning, a large site, and limited planning constraints.
```

Another recommended query:
```text
Strategy:
Single dwelling rebuild

Locality:
Rhodes

Query:
I want a site for detached house redevelopment on standard residential land, with low planning constraints and a suitable lot size.
```

Note that `Locality` is not mandatory.

### Demo API Endpoints
**Algorithm Service**
```text
GET  /health
POST /retrieve-sites
POST /feedback
POST /report-jobs
GET  /report-jobs/{report_id}
```

**Backend Gateway**
```text
GET  /health
POST /api/search
POST /api/feedback
POST /api/reports
GET  /api/reports/{report_id}
```

The frontend calls the backend gateway. The backend gateway calls the internal algorithm service.

### Testing Search from Terminal
Example backend search request:
```bash
curl -X POST http://localhost:8002/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "low_rise_apartment",
    "query_text": "I want a site for low-rise apartment redevelopment near a train station, with high development zoning, a large site, and limited planning constraints.",
    "top_k": 5,
    "recall_k": 1000,
    "locality": "RHODES",
    "with_explanations": false,
    "use_template_explanations": true,
    "log_request": true
  }' | python -m json.tool
```

### Testing Locality Filter Behaviour
Run:
```bash
./scripts/test_location_filter.sh RHODES low_rise_apartment
```

Test a missing locality:
```bash
./scripts/test_location_filter.sh "BREAKFAST POINT" single_dwelling_rebuild
```

Test a fake locality:
```bash
./scripts/test_location_filter.sh "ZZZ_NOT_A_REAL_SUBURB" low_rise_apartment
```

The backend gateway applies a strict locality guard for the demo UI. If the user enters a locality and no returned address matches it exactly, the frontend shows no results instead of unrelated global recommendations.

### Generating Reports
From the UI, click `Generate Report` after a search.

From terminal:
```bash
curl -X POST http://localhost:8002/api/reports \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "req_replace_with_real_request_id",
    "explanation_mode": "template",
    "output_markdown": true,
    "output_pdf": true,
    "audience": "developer",
    "title": "Smart Developer Site Recommendation Report"
  }' | python -m json.tool
```

Generated reports are saved under:
```text
algorithm/artifacts/reports/
```

Open a generated PDF locally:
```bash
open algorithm/artifacts/reports/report_xxx.pdf
```

### Feedback Logging
The frontend can log user actions such as:
```text
click
save
dismiss
select
manual_positive
manual_negative
download_pdf
export_report
```

Example terminal request:
```bash
curl -X POST http://localhost:8002/api/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "req_replace_with_real_request_id",
    "event_type": "save",
    "rid": "5304579",
    "rank_position": 1,
    "event_value": {
      "source": "demo"
    },
    "user_id": "demo_user",
    "session_id": "frontend_demo"
  }' | python -m json.tool
```

### Exporting Feedback Dataset
Build a full feedback dataset:
```bash
python -m algorithm.src.mlops.build_feedback_dataset
```

Build labelled-only dataset:
```bash
python -m algorithm.src.mlops.build_feedback_dataset \
  --labelled-only \
  --output algorithm/artifacts/feedback/feedback_reranker_dataset_labelled.parquet
```

Build weak-negative dataset for pipeline testing:
```bash
python -m algorithm.src.mlops.build_feedback_dataset \
  --weak-negative-unfeedbacked \
  --output algorithm/artifacts/feedback/feedback_reranker_dataset_weak.parquet
```

The weak-negative dataset is mainly for validating the training pipeline. It should not be treated as high-quality production training data until enough real feedback is collected.

### Useful SQL Checks
Latest retrieval requests:
```sql
SELECT 
    request_id, 
    strategy, 
    locality, 
    latency_ms, 
    result_count, 
    created_at
FROM retrieval_requests
ORDER BY created_at DESC
LIMIT 10;
```

Latest feedback:
```sql
SELECT 
    feedback_id, 
    request_id, 
    rid, 
    event_type, 
    rank_position, 
    created_at
FROM user_feedback
ORDER BY created_at DESC
LIMIT 10;
```

Latest report jobs:
```sql
SELECT 
    report_id, 
    request_id, 
    status, 
    explanation_mode, 
    output_pdf_path, 
    latency_ms, 
    created_at
FROM report_jobs
ORDER BY created_at DESC
LIMIT 10;
```

Model registry:
```sql
SELECT 
    model_version, 
    model_type, 
    status, 
    artifact_path, 
    created_at
FROM model_registry
ORDER BY created_at DESC;
```

### Note for Demo
The current product path is:
```text
Synchronous:
retrieval + reranking + template explanation + logging

Asynchronous/future:
richer LLM explanation + report generation worker + retraining pipeline
```

The system intentionally uses deterministic template explanations for site cards so that the main search flow does not block on a local LLM.

Local LLM explanations can still be used later for richer reports, but they should not be required for the main user-facing search experience.

## Current Limitations
* Candidate coverage depends on the current processed dataset.
* Some localitites may not have matching candidate sites.
* DCN outputs should be treated as internal ranking scores, not calibrated probabilities.
* Feedback data is currently small and mostly useful for validating the MLOps loop.
* The current report job endpoint is synchronous for demo simplicity; a background worker can be added later.