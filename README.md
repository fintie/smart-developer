# Smart Developer

Smart Developer is an AI-assisted property development intelligence platform for real estate agents and developers. It recommends development opportunities from site, zoning, policy, location, cost, and market signals, then generates agent-facing explanations and PDF reports.

The current demo is deployed as a three-service web application:

```text
React / Vite frontend on Vercel
        ↓
FastAPI backend gateway on Render
        ↓
FastAPI algorithm service on Render
```

## Live Deployment

| Service | Purpose | URL |
|---|---|---|
| Frontend | User-facing web app | https://smart-developer-frontend.vercel.app |
| Backend Gateway | Product API and frontend proxy | https://smart-developer-backend.onrender.com |
| Algorithm Service | Retrieval, ranking, policy, economics, and report generation | https://smart-developer-algorithm.onrender.com |

The frontend should call the backend only. The backend calls the algorithm service internally.

## Product Features

- Development opportunity search for selected strategies such as low-rise apartments, land banking, assembly opportunity, townhouse / multi-dwelling, granny flat, and rebuild opportunities.
- Two-stage recommendation pipeline using two-tower retrieval and DCN-style reranking.
- Policy-aware scoring with NSW housing policy and transport-oriented development signals.
- Economics-aware ranking using acquisition cost, estimated development cost, construction cost trends, value potential, and cost-efficiency signals.
- Ranking profiles such as balanced, policy upside, budget sensitive, and high value.
- Locality filtering and map-based result display using latitude and longitude.
- Agent-facing explanations and pitch summaries for each recommended site.
- PDF report export for shortlisted sites.

## Current Demo Stack

### Frontend

- React
- TypeScript
- Vite
- React Leaflet / OpenStreetMap
- Vercel deployment

### Backend Gateway

- FastAPI
- httpx
- Render deployment
- CORS-enabled product API
- Proxies search and PDF export requests to the algorithm service

### Algorithm Service

- FastAPI
- PyTorch
- sentence-transformers / transformer bi-encoder retrieval
- DCN-style reranking
- pandas / NumPy / scikit-learn / XGBoost
- Chroma / LangChain policy evidence retrieval
- ReportLab / Markdown PDF export
- Render deployment

## Repository Structure

```text
smart-developer/
├── algorithm/
│   ├── artifacts/
│   │   ├── economics/
│   │   │   ├── market_trend_regression_v1/
│   │   │   └── xgb_market_value_v1/
│   │   ├── models/
│   │   │   ├── dcn_reranker_v1/
│   │   │   └── two_tower_v1/
│   │   └── policy_index/
│   │       └── chroma/
│   └── src/
│       ├── economics/
│       ├── explanation/
│       ├── inference/
│       ├── policy/
│       ├── ranking/
│       ├── retrieval/
│       └── serving/
├── backend/
│   └── app/
│       ├── main.py
│       ├── schemas.py
│       └── services/
├── data/
│   └── processed/
│       ├── cost/
│       ├── economics/
│       └── retrieval/
├── frontend/
│   ├── src/
│   ├── package.json
│   └── vite.config.ts
├── requirements.txt
└── README.md
```

## Required Artifacts for the Demo

The deployed algorithm service expects these files to exist in the repository or deployment image:

```text
data/processed/retrieval/candidate_sites_geo.parquet
data/processed/economics/trend/construction_cost_indices.parquet
data/processed/economics/trend/suburb_monthly_market.parquet
data/processed/cost/locality_sales_summary.parquet
algorithm/artifacts/economics/xgb_market_value_v1/
algorithm/artifacts/economics/market_trend_regression_v1/
algorithm/artifacts/policy_index/chroma/
algorithm/artifacts/models/dcn_reranker_v1/model.pt
algorithm/artifacts/models/two_tower_v1/model.pt
algorithm/artifacts/models/two_tower_v1/candidate_embeddings.npy
```

`candidate_embeddings.npy` is important for cloud serving. Without it, the algorithm service may try to encode all candidate site texts at request time, which is too slow on small Render instances.

## Local Setup

Create and activate a virtual environment from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install frontend dependencies:

```bash
cd frontend
npm install
cd ..
```

## Running Locally

Run the algorithm service:

```bash
uvicorn algorithm.src.serving.api:app --host 0.0.0.0 --port 8001
```

Run the backend gateway in another terminal:

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8002
```

Run the frontend in another terminal:

```bash
cd frontend
npm run dev
```

The frontend should use:

```text
VITE_API_BASE_URL=http://localhost:8002
```

## Environment Variables

### Algorithm Service

For Render deployment:

```text
PYTHON_VERSION=3.11.9
LAZY_LOAD_PREDICTOR=true
SKIP_STARTUP_WARMUP=true
ENABLE_POLICY_RAG=true
```

Optional:

```text
DATABASE_URL=postgresql://...
```

If `DATABASE_URL` is not set, MLOps logging is skipped and the demo still works.

### Backend Gateway

```text
ALGORITHM_SERVICE_URL=https://smart-developer-algorithm.onrender.com
```

For local development:

```text
ALGORITHM_SERVICE_URL=http://localhost:8001
```

### Frontend

For Vercel production:

```text
VITE_API_BASE_URL=https://smart-developer-backend.onrender.com
```

For local development:

```text
VITE_API_BASE_URL=http://localhost:8002
```

## API Endpoints

### Backend Gateway

```text
GET  /health
POST /api/search
POST /api/feedback
POST /api/reports
GET  /api/reports/{report_id}
POST /api/export-report
```

### Algorithm Service

```text
GET  /health
POST /retrieve-sites
POST /feedback
POST /report-jobs
GET  /report-jobs/{report_id}
POST /export-report
```

## Example Search Request

Call the backend gateway:

```bash
curl -s -X POST https://smart-developer-backend.onrender.com/api/search \
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

Expected response fields include:

```text
request_id
metadata
result_count
results[].address
results[].agent_opportunity_score
results[].strategy_score
results[].policy_upside_score
results[].policy_evidence
results[].estimated_total_project_cost
results[].cost_efficiency_score
results[].value_potential_score
results[].fast_explanation
results[].agent_pitch
```

## PDF Report Export

The frontend calls:

```text
POST /api/export-report
```

with the current search results. The backend forwards the request to the algorithm service and returns a downloadable PDF.

The report includes:

- Executive summary
- Shortlisted site table
- Site-level rationale
- Policy and planning signals
- Retrieved policy evidence
- Economics and feasibility signals
- Risks and checks
- Agent-facing summaries

## Deployment Notes

### Render Algorithm Service

Start command:

```bash
uvicorn algorithm.src.serving.api:app --host 0.0.0.0 --port $PORT
```

Important settings:

```text
LAZY_LOAD_PREDICTOR=true
SKIP_STARTUP_WARMUP=true
ENABLE_POLICY_RAG=true
```

### Render Backend Gateway

Start command:

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT
```

Environment variable:

```text
ALGORITHM_SERVICE_URL=https://smart-developer-algorithm.onrender.com
```

### Vercel Frontend

If deploying from the `frontend` directory using the Vercel CLI:

```bash
cd frontend
vercel
vercel env add VITE_API_BASE_URL
vercel --prod
```

Production environment variable:

```text
VITE_API_BASE_URL=https://smart-developer-backend.onrender.com
```

If importing from GitHub, set:

```text
Framework Preset: Vite
Root Directory: frontend
Build Command: npm run build
Output Directory: dist
Install Command: npm install
```

## Demo Warm-Up

Render free instances may spin down. Before a demo, warm up the backend and algorithm service:

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
  }' > /tmp/smart_developer_warmup.json
```

## Known Demo Limitations

- Current results are based on available screening data and should not be treated as formal valuation, planning, legal, or investment advice.
- Cost and value estimates are indicative proxies for ranking and explanation.
- Policy matching is an indicative signal and should be verified against official NSW Planning, LEP, SEPP, and professional planning advice.
- Render free-tier services can be slow on first request after inactivity.
- Large ML artifacts are currently packaged for demo deployment; future production deployment should move model and embedding artifacts to managed object storage or a model registry.

## Future Improvements

- Move model artifacts and candidate embeddings to S3, GCS, or a model registry.
- Add persistent Postgres logging for requests, feedback, and report jobs.
- Add proper authentication and organisation-level access control.
- Replace demo artifact packaging with CI/CD-managed deployment assets.
- Improve candidate deduplication across unit-level and base-site-level parcels.
- Add stronger feasibility modelling with professional quantity surveying assumptions.
- Add richer policy retrieval and citation cleaning.
- Add monitoring for latency, error rate, cold starts, and model version usage.
