# Smart Developer

Smart Developer is a prototype site-recommendation tool for real estate agents and property developers. It helps shortlist NSW development opportunities by combining geospatial site features, planning-policy signals, market/economics indicators, and ranking profiles.

The current MVP is designed for local demo and internal testing. It runs as a three-part application:

```text
React frontend        http://localhost:5173
FastAPI backend       http://localhost:8002
Algorithm service     http://localhost:8001
```

The frontend talks to the backend gateway, and the backend forwards search requests to the algorithm service.

---

## What the system does

Given a development strategy, locality, query, and ranking profile, Smart Developer returns a ranked shortlist of candidate sites.

The algorithm stack currently includes:

- **Locality-aware candidate retrieval**  
  Filters candidate sites by suburb/locality before retrieval, so suburb searches do not depend on global recall.

- **Two-tower retrieval model**  
  Retrieves strategy-relevant candidate sites using query and site embeddings.

- **DCN reranker**  
  Reranks retrieved candidates using structured site features.

- **Policy-aware scoring**  
  Scores planning-policy upside using structured policy rules and retrieved NSW Planning evidence snippets.

- **Economics-aware scoring**  
  Adds market value, development cost, cost risk, cost efficiency, local market trend, and construction cost trend signals.

- **Ranking profiles**  
  Supports different recommendation objectives:
  - `balanced`
  - `policy_upside`
  - `budget_sensitive`
  - `high_value`

- **Agent-facing explanations**  
  Generates deterministic explanations and opportunity summaries for site cards and reports.

---

## Local startup flow

Run these from the project root.

### 1. Activate the Python environment

```bash
source .venv/bin/activate
```

If the virtual environment does not exist yet, create it and install dependencies first:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### 2. Start the algorithm service

```bash
uvicorn algorithm.src.serving.api:app --host 0.0.0.0 --port 8001
```

Health check:

```bash
curl http://localhost:8001/health
```

The algorithm service loads the retrieval model, reranker, policy scorer, economics pipeline, and model artifacts.

---

### 3. Start the backend gateway

Open a second terminal:

```bash
source .venv/bin/activate
uvicorn backend.app.main:app --host 0.0.0.0 --port 8002
```

Health check:

```bash
curl http://localhost:8002/api/health
```

The backend gateway exposes the frontend-facing API and forwards search requests to the algorithm service.

---

### 4. Start the frontend

Open a third terminal:

```bash
cd frontend
npm install
npm run dev
```

Then open:

```text
http://localhost:5173
```

---

## Example API smoke test

With the algorithm service running on port `8001`:

```bash
curl -s -X POST http://localhost:8001/retrieve-sites \
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
  }' | python -m json.tool
```

Expected behavior:

- Response contains `results`.
- Metadata includes:
  - `retrieval_top_k`
  - `final_top_k`
  - `location_prefilter_applied`
  - `location_prefilter_candidate_count`
- Budget-sensitive results should prefer relatively lower-cost and higher cost-efficiency sites within the selected strategy.

---

## Required artifacts and data files

The local demo expects the following artifacts/data to exist.

### Candidate site table

```text
data/processed/retrieval/candidate_sites_geo.parquet
```

This is the candidate site table used for retrieval and map display. It should include site features plus coordinates, such as:

```text
address
base_site_address
RID
latitude
longitude
primary_zoning_code
lot_size_band
constraint_severity_band
station_distance_band
strategy score columns
```

The model config should point to this file:

```text
algorithm/configs/model.yaml
```

Expected setting:

```yaml
candidate_sites_path: "data/processed/retrieval/candidate_sites_geo.parquet"
```

---

### Two-tower retrieval model

Expected under the model artifact directory configured in:

```text
algorithm/configs/model.yaml
```

Typical required file:

```text
model.pt
```

The default production retrieval model name is:

```text
two_tower_v1
```

---

### DCN reranker model

Expected under the DCN reranker artifact directory configured in:

```text
algorithm/configs/model.yaml
```

Typical required files:

```text
model.pt
preprocessing.json
```

The default production reranking model name is:

```text
dcn_reranker_v1
```

---

### Policy rules and policy evidence index

Structured policy rules:

```text
algorithm/configs/policies/policy_rules.yaml
```

Policy sources config:

```text
algorithm/configs/policies/policy_sources.yaml
```

Policy evidence vector index:

```text
algorithm/artifacts/policy_index/chroma/
```

The policy evidence index is used to attach NSW Planning evidence snippets to policy-aware explanations.

---

### Market value model

```text
algorithm/artifacts/economics/xgb_market_value_v1/
```

Required files:

```text
model.joblib
feature_columns.json
metrics.json
model_card.json
```

This model provides an indicative transaction-level market value estimate. It is not a formal valuation model.

---

### Market trend model

```text
algorithm/artifacts/economics/market_trend_regression_v1/
```

Required files:

```text
model.joblib
feature_columns.json
metrics.json
model_card.json
```

This rolling Ridge model estimates short-term local market momentum. Its prediction is used as a conservative trend adjustment signal, not as a formal price forecast.

---

### Construction cost trend data

```text
data/processed/economics/trend/construction_cost_indices.parquet
```

This file contains the ABS WPI/PPI-derived construction cost trend signal used by the economics pipeline.

---

### Market trend feature data

```text
data/processed/economics/trend/suburb_monthly_market.parquet
```

This file contains suburb-month market features derived from NSW PSI sales data and is used by the market trend predictor.

---

### Locality sales summary

```text
data/processed/cost/locality_sales_summary.parquet
```

This file is used as a locality-level market proxy and fallback for economics scoring.

---

## Local service ports

| Service | URL | Purpose |
|---|---:|---|
| Frontend | `http://localhost:5173` | Browser UI |
| Backend gateway | `http://localhost:8002` | Frontend API gateway |
| Algorithm service | `http://localhost:8001` | Retrieval, reranking, policy, economics, explanations |

---

## Common troubleshooting

### Port already in use

```bash
lsof -ti:8001 | xargs kill -9
lsof -ti:8002 | xargs kill -9
```

Then restart the services.

---

### Frontend cannot connect to backend

Check that the backend gateway is running:

```bash
curl http://localhost:8002/api/health
```

Also check the frontend API base URL in:

```text
frontend/src/api.ts
```

It should point to:

```text
http://localhost:8002
```

unless overridden by environment variables.

---

### Backend cannot connect to algorithm service

Check that the algorithm service is running:

```bash
curl http://localhost:8001/health
```

Also check backend settings/environment for the algorithm service URL. It should point to:

```text
http://localhost:8001
```

---

### Search returns no results for a suburb

Possible causes:

- The suburb/locality is not present in `candidate_sites_geo.parquet`.
- The locality text does not match the address text.
- `recall_k` is too small.
- Backend strict locality guard filtered results after retrieval.

For local demo, use:

```text
recall_k = 1000 or higher
```

and try known localities such as:

```text
WOLLI CREEK
ZETLAND
WAITARA
EPPING
CHATSWOOD
FAIRFIELD
AUBURN
```

---

## Notes and limitations

This is an MVP screening tool. Its outputs are intended for opportunity discovery and prioritisation, not for formal planning, valuation, legal, financial, or investment advice.

Important limitations:

- The market value model is an indicative transaction-level model, not a licensed valuation.
- Policy evidence snippets should be checked against full official planning documents.
- Development cost estimates are high-level feasibility proxies.
- Market trend and construction cost trend outputs are conservative screening signals.
- Ranking profiles influence prioritisation but do not guarantee perfect alignment with every free-text query.