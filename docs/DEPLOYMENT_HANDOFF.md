# Smart Developer Handoff

This document is intended for backend collaborators who will continue development on the Smart Developer project.

Smart Developer is a property development intelligence platform. The current demo stack is:

```text
React + Vite frontend
        ↓
FastAPI backend gateway
        ↓
FastAPI algorithm service
        ↓
ML / ranking / policy / economics pipeline
```

The deployed demo is already working with the backend and algorithm service on Render and the frontend on Vercel. The next stage is to make the backend/database layer more production-ready.

---

## 1. Repository Structure

Expected high-level project layout:

```text
smart-developer/
├── algorithm/
│   ├── src/
│   │   ├── inference/
│   │   ├── retrieval/
│   │   ├── ranking/
│   │   ├── economics/
│   │   ├── policy/
│   │   ├── explanation/
│   │   ├── serving/
│   │   └── mlops/
│   └── artifacts/
│       ├── models/
│       ├── economics/
│       └── policy_index/
│
├── backend/
│   └── app/
│       ├── main.py
│       ├── schemas.py
│       └── services/
│           └── algorithm_client.py
│
├── frontend/
│   ├── src/
│   ├── package.json
│   └── vite.config.ts
│
├── data/
│   └── processed/
│
├── requirements.txt
├── docker-compose.yml        # recommended for local PostgreSQL
├── .env                      # local only, do not commit secrets
└── README.md
```

---

## 2. Services Overview

### Frontend

The frontend is a React/Vite app. It should call the backend gateway only.

```text
Frontend API target:
VITE_API_BASE_URL=https://smart-developer-backend.onrender.com
```

For local development, it should usually point to:

```text
VITE_API_BASE_URL=http://localhost:8002
```

The frontend should not call the algorithm service directly.

### Backend Gateway

The backend is a FastAPI gateway responsible for:

- Accepting product-facing API requests from the frontend.
- Forwarding search/report requests to the algorithm service.
- Applying frontend-safe response shaping or location guard logic.
- Later: owning user accounts, saved searches, feedback, permissions, and database-backed business logic.

Current deployed backend URL:

```text
https://smart-developer-backend.onrender.com
```

Important backend endpoints:

```text
GET  /health
POST /api/search
POST /api/feedback
POST /api/export-report
GET  /api/reports/{report_id}
```

The backend calls the algorithm service through:

```text
ALGORITHM_SERVICE_URL=https://smart-developer-algorithm.onrender.com
```

### Algorithm Service

The algorithm service is a FastAPI service that owns the ML/recommendation logic.

Current deployed algorithm URL:

```text
https://smart-developer-algorithm.onrender.com
```

Important algorithm endpoints:

```text
GET  /health
POST /retrieve-sites
POST /feedback
POST /export-report
POST /report-jobs
GET  /report-jobs/{report_id}
```

The algorithm service currently includes:

- Two-tower semantic retrieval.
- Deep & Cross Network reranking.
- Geospatial feature-based strategy scoring.
- Policy-aware scoring and NSW Planning policy evidence retrieval.
- Economics-aware scoring:
  - transaction value model,
  - market trend adjustment,
  - construction cost trend,
  - acquisition/development/soft cost/contingency estimates.
- Markdown/PDF report generation.

---

## 3. Local Development Setup

### 3.1 Prerequisites

Recommended:

```text
Python 3.11
Node.js 20+
Docker Desktop
Git
```

Docker is recommended mainly for PostgreSQL. It avoids everyone installing and configuring local PostgreSQL manually.

---

## 4. Backend + Algorithm Python Environment

From the project root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If `python3.11` is not available, use the local Python 3.11 binary or pyenv.

Check FastAPI import:

```bash
python - <<'PY'
import fastapi
import uvicorn
print("FastAPI environment OK")
PY
```

---

## 5. Frontend Development Setup

From the project root:

```bash
cd frontend
npm install
npm run dev
```

Local frontend should run at something like:

```text
http://localhost:5173
```

For local development, create:

```bash
touch frontend/.env.local
```

Then add:

```env
VITE_API_BASE_URL=http://localhost:8002
```

For deployed Vercel frontend, the production environment variable should be:

```env
VITE_API_BASE_URL=https://smart-developer-backend.onrender.com
```

Build check:

```bash
cd frontend
npm run build
```

---

## 6. Running Services Locally

Use three terminals.

### Terminal 1: Algorithm Service

From project root:

```bash
source .venv/bin/activate

export LAZY_LOAD_PREDICTOR=true
export SKIP_STARTUP_WARMUP=true
export ENABLE_POLICY_RAG=true

uvicorn algorithm.src.serving.api:app --host 0.0.0.0 --port 8001 --reload
```

Health check:

```bash
curl http://localhost:8001/health | python -m json.tool
```

Search test:

```bash
curl -s -X POST http://localhost:8001/retrieve-sites \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "low_rise_apartment",
    "query_text": "cost efficient lower capital requirement lower project cost",
    "top_k": 3,
    "recall_k": 1000,
    "with_explanations": false,
    "use_template_explanations": true,
    "locality": "Zetland",
    "ranking_profile": "budget_sensitive",
    "log_request": false,
    "debug": false
  }' | python -m json.tool
```

### Terminal 2: Backend Gateway

From project root:

```bash
source .venv/bin/activate

export ALGORITHM_SERVICE_URL=http://localhost:8001

uvicorn backend.app.main:app --host 0.0.0.0 --port 8002 --reload
```

Health check:

```bash
curl http://localhost:8002/health | python -m json.tool
```

Backend search test:

```bash
curl -s -X POST http://localhost:8002/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "low_rise_apartment",
    "query_text": "cost efficient lower capital requirement lower project cost",
    "top_k": 3,
    "recall_k": 1000,
    "with_explanations": false,
    "use_template_explanations": true,
    "locality": "Zetland",
    "ranking_profile": "budget_sensitive",
    "log_request": false,
    "debug": false
  }' | python -m json.tool
```

### Terminal 3: Frontend

```bash
cd frontend
npm run dev
```

Open:

```text
http://localhost:5173
```

---

## 7. Database Setup with Docker

For backend/database work, Docker is recommended.

Create or use this `docker-compose.yml` at the project root:

```yaml
services:
  db:
    image: postgres:16
    container_name: smart_developer_db
    restart: unless-stopped
    environment:
      POSTGRES_DB: smart_developer
      POSTGRES_USER: smart_dev
      POSTGRES_PASSWORD: smart_dev_password
    ports:
      - "5432:5432"
    volumes:
      - smart_developer_pgdata:/var/lib/postgresql/data

volumes:
  smart_developer_pgdata:
```

Start database:

```bash
docker compose up -d postgres
```

Check running containers:

```bash
docker ps
```

Connect with psql if available:

```bash
psql postgresql+psycopg://smart_dev:smart_dev_password@localhost:55435/smart_developer
```

Stop database:

```bash
docker compose down
```

Delete database volume if you want a clean reset:

```bash
docker compose down -v
```

---

## 8. Environment Variables

Recommended local `.env` file at project root:

```env
# Backend -> algorithm
ALGORITHM_SERVICE_URL=http://localhost:8001

# Optional local database
DATABASE_URL=postgresql+psycopg://smart_dev:smart_dev_password@localhost:55435/smart_developer

# Algorithm service startup behaviour
LAZY_LOAD_PREDICTOR=true
SKIP_STARTUP_WARMUP=true

# Policy RAG
ENABLE_POLICY_RAG=true
```

Frontend `.env.local`:

```env
VITE_API_BASE_URL=http://localhost:8002
```

Render backend production env:

```env
ALGORITHM_SERVICE_URL=https://smart-developer-algorithm.onrender.com
```

Render algorithm production env:

```env
PYTHON_VERSION=3.11.9
LAZY_LOAD_PREDICTOR=true
SKIP_STARTUP_WARMUP=true
ENABLE_POLICY_RAG=true
```

Vercel frontend production env:

```env
VITE_API_BASE_URL=https://smart-developer-backend.onrender.com
```

Do not commit `.env` files with secrets.

---

## 9. Database Responsibilities for Backend Work

The current demo can run without PostgreSQL. Database work is the next stage.

Recommended backend-owned tables:

### users

For authentication/authorization if needed later.

```text
id
email
name
role
created_at
updated_at
```

### search_requests

Stores user search/query context.

```text
id
user_id
session_id
strategy
query_text
locality
ranking_profile
payload_json
created_at
```

### search_results

Stores returned site-level results.

```text
id
request_id
rank_position
rid
address
base_site_address
latitude
longitude
strategy_score
agent_opportunity_score
policy_upside_score
cost_efficiency_score
estimated_total_project_cost
payload_json
created_at
```

### feedback

Stores user feedback for future optimisation.

```text
id
request_id
result_id
rid
feedback_type
rating
comment
created_at
```

### reports

Stores generated report metadata.

```text
id
request_id
report_type
file_path_or_url
status
created_at
updated_at
```

For the next backend phase, Zoe can implement these with SQLAlchemy + Alembic migrations.

---

## 10. Suggested Backend Architecture

Recommended backend structure:

```text
backend/app/
├── main.py
├── config.py
├── database.py
├── schemas.py
├── models/
│   ├── search.py
│   ├── feedback.py
│   └── report.py
├── routers/
│   ├── search.py
│   ├── feedback.py
│   ├── reports.py
│   └── health.py
├── services/
│   ├── algorithm_client.py
│   ├── search_service.py
│   ├── feedback_service.py
│   └── report_service.py
└── migrations/
```

Current backend is intentionally simple. Refactor only when adding persistent database logic.

---

## 11. Algorithm Artifacts Required for Serving

The algorithm service depends on model/data artifacts. Important examples:

```text
data/processed/retrieval/candidate_sites_geo.parquet
data/processed/economics/trend/construction_cost_indices.parquet
data/processed/economics/trend/suburb_monthly_market.parquet
data/processed/cost/locality_sales_summary.parquet

algorithm/artifacts/models/two_tower_v1/model.pt
algorithm/artifacts/models/two_tower_v1/candidate_embeddings.npy
algorithm/artifacts/models/dcn_reranker_v1/model.pt

algorithm/artifacts/economics/xgb_market_value_v1/
algorithm/artifacts/economics/market_trend_regression_v1/

algorithm/artifacts/policy_index/chroma/
```

Important serving note:

```text
candidate_embeddings.npy should be precomputed and loaded directly.
Do not encode all candidate sites on first request in cloud serving.
```

If this file is missing, Render may appear to hang while encoding many candidate texts.

---

## 12. Current Demo Deployment

### Algorithm on Render

Start command:

```bash
uvicorn algorithm.src.serving.api:app --host 0.0.0.0 --port $PORT
```

Important env:

```env
PYTHON_VERSION=3.11.9
LAZY_LOAD_PREDICTOR=true
SKIP_STARTUP_WARMUP=true
ENABLE_POLICY_RAG=true
```

URL:

```text
https://smart-developer-algorithm.onrender.com
```

### Backend on Render

Start command:

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT
```

Important env:

```env
ALGORITHM_SERVICE_URL=https://smart-developer-algorithm.onrender.com
```

URL:

```text
https://smart-developer-backend.onrender.com
```

### Frontend on Vercel

Build settings:

```text
Framework: Vite
Build Command: npm run build
Output Directory: dist
Root Directory: frontend
```

Important env:

```env
VITE_API_BASE_URL=https://smart-developer-backend.onrender.com
```

---

## 13. Common Issues

### Backend search returns `Failed to call algorithm service`

Check backend env:

```text
ALGORITHM_SERVICE_URL must point to the algorithm service, not backend, localhost, or /retrieve-sites.
```

Correct:

```text
https://smart-developer-algorithm.onrender.com
```

Wrong:

```text
http://localhost:8001
https://smart-developer-backend.onrender.com
https://smart-developer-algorithm.onrender.com/retrieve-sites
```

Also ensure backend HTTP timeout is long enough for cold cloud inference.

### Algorithm returns Chroma / `_type` error

This usually means ChromaDB version mismatch with the persisted policy index.

Recommended fix:

```bash
pip install chromadb==0.6.3
rm -rf algorithm/artifacts/policy_index/chroma
python -m algorithm.src.policy.build_policy_index
git add -f algorithm/artifacts/policy_index/chroma
git commit -m "Rebuild policy index with chromadb 0.6.3"
git push
```

Short-term fallback:

```env
ENABLE_POLICY_RAG=false
```

This disables retrieved policy evidence while preserving policy scoring.

### Frontend cannot call backend

Check browser console for CORS or wrong API URL.

Frontend env must be:

```env
VITE_API_BASE_URL=https://smart-developer-backend.onrender.com
```

Backend CORS should allow the Vercel frontend domain. For demo only, permissive CORS is acceptable:

```python
allow_origins=["*"]
```

### Frontend map TypeScript build error

If `react-leaflet` type definitions complain about `center` or `attribution`, the current workaround is to spread props cast as `any` inside `ResultsMap.tsx`.

### Render cold start / slow first request

Render free tier can be slow after inactivity. Before demo, warm up:

```bash
curl -s -X POST https://smart-developer-backend.onrender.com/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "low_rise_apartment",
    "query_text": "cost efficient lower capital requirement lower project cost",
    "top_k": 5,
    "recall_k": 1000,
    "with_explanations": false,
    "use_template_explanations": true,
    "locality": "Zetland",
    "ranking_profile": "budget_sensitive",
    "log_request": false,
    "debug": false
  }' > /tmp/smart_dev_warmup.json
```

---

## 14. Recommended Next Steps for Zoe

1. Set up local dev:
   - Python environment.
   - Frontend environment.
   - Docker PostgreSQL.
   - Backend connected to local algorithm service.

2. Refactor backend into routers/services:
   - `search`
   - `feedback`
   - `reports`
   - `health`

3. Add PostgreSQL persistence:
   - search requests,
   - search results,
   - feedback,
   - report metadata.

4. Add Alembic migrations.

5. Add backend tests for:
   - `/health`
   - `/api/search`
   - `/api/feedback`
   - `/api/export-report`

6. Add proper CORS config for deployed frontend URL.

7. Add user/session identity later if required.

8. Keep algorithm service separate from backend business logic.

---

## 15. Development Rule of Thumb

For the backend:

```text
Frontend should talk only to backend.
Backend should own product/session/database logic.
Algorithm service should own ML inference, ranking, policy/economics scoring, and report generation.
Database should store user interactions, outputs, feedback, and report metadata.
```

Avoid putting ML model logic directly in the frontend or product backend.
