# Deployment Handoff

Internal notes for backend / full-stack integration of the Smart Developer algorithm service.

## 1. Current Inference Stack

```text
strategy + query_text
-> optional query planner / query rewrite
-> two_tower_v1 retrieval
-> optional locality/address filter
-> dcn_reranker_v1 rerank
-> base-site dedupe
-> optional template or LLM explanation
-> return top-k sites
```

Main Python entrypoint:

```python
from algorithm.src.inference.predictor import retrieve_sites
```

Recommended product serving path:

```text
FastAPI service starts
-> load predictor once
-> run warmup query
-> serve warm requests
```

Do not start a new Python process for every request.

## 2. Required Runtime Artifacts

These should be placed in the same repo-relative paths.

```text
data/processed/retrieval/
├── candidate_sites.parquet
└── query_intents.jsonl

algorithm/artifacts/models/two_tower_v1/
└── model.pt

algorithm/artifacts/models/dcn_reranker_v1/
├── model.pt
├── preprocessing.json
└── model_card.json
```

Required files:

- `data/processed/retrieval/candidate_sites.parquet`
- `data/processed/retrieval/query_intents.jsonl`
- `algorithm/artifacts/models/two_tower_v1/model.pt`
- `algorithm/artifacts/models/dcn_reranker_v1/model.pt`
- `algorithm/artifacts/models/dcn_reranker_v1/preprocessing.json`

Optional:

- Ollama + local model if using rich local LLM explanations
- PostgreSQL if using MLOps logging / feedback / report jobs

## 3. Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If using local LLM explanations:

```bash
ollama serve
ollama pull llama3.1:8b-instruct-q4_K_M
```

For production-like retrieval, prefer:

```python
with_explanations=False
use_template_explanations=True
```

This keeps retrieval fast and avoids blocking on Ollama.

## 4. Direct Python Usage

```python
from algorithm.src.inference.predictor import retrieve_sites

response = retrieve_sites(
    strategy="low_rise_apartment",
    query_text=(
        "I want a site for low-rise apartment redevelopment near a train station, "
        "with high development zoning, a large site, and limited planning constraints."
    ),
    top_k=5,
    recall_k=1000,
    with_explanations=False,
    retrieval_model="two_tower_v1",
    use_dcn_reranker=True,
    reranking_model="dcn_reranker_v1",
    locality="WAITARA",
)
```

Important response fields:

- `request_id`
- `metadata.latency_ms`
- `results[*].RID`
- `results[*].address`
- `results[*].base_site_address`
- `results[*].strategy_score`
- `results[*].dcn_rank_score`
- `results[*].fast_explanation`
- `results[*].explanation`

Frontend should display `base_site_address` first.  
`address` is the raw source address and may be unit-level.

## 5. FastAPI Service

Current service entrypoint:

```bash
uvicorn algorithm.src.serving.api:app --host 0.0.0.0 --port 8001
```

Endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Check service readiness and loaded model names |
| `POST /retrieve-sites` | Retrieve/rerank sites and optionally log request/results |
| `POST /feedback` | Log user feedback events |
| `POST /report-jobs` | Generate markdown/PDF report from a logged request |
| `GET /report-jobs/{report_id}` | Check report job status |

The service uses a predictor singleton. Cold start may be slow, but warm requests should be fast.

## 6. Retrieval API Example

```bash
curl -X POST http://localhost:8001/retrieve-sites \\
  -H "Content-Type: application/json" \\
  -d '{
    "strategy": "single_dwelling_rebuild",
    "query_text": "I want a site for detached house redevelopment on standard residential land, with low planning constraints and a suitable lot size.",
    "top_k": 5,
    "recall_k": 1000,
    "with_explanations": false,
    "use_template_explanations": true,
    "user_id": "demo_user",
    "session_id": "demo_session",
    "log_request": true
  }'
```

Recommended defaults:

```text
retrieval_model = two_tower_v1
reranking_model = dcn_reranker_v1
use_dcn_reranker = true
dedupe_by_address = true
with_explanations = false
use_template_explanations = true
recall_k = 1000
```

## 7. Location Filter

Current location filtering is lightweight and address-text based.

Supported fields:

- `locality`
- `address_contains`

Example:

```json
{
  "locality": "WAITARA"
}
```

Because `candidate_sites.parquet` does not yet include structured `locality`, `postcode`, or `lga_name`, this is not final production-grade filtering.

Future candidate table should include:

- `locality`
- `postcode`
- `lga_name`
- `latitude`
- `longitude`

## 8. Base-Site Dedupe

The retriever dedupes unit-level addresses into base-site addresses.

Example:

```text
623/21-37 WAITARA AVENUE WAITARA
-> 21-37 WAITARA AVENUE WAITARA
```

Frontend/report output should use `base_site_address` and keep raw `address` as source/debug information.

## 9. Explanation Modes

Recommended split:

| Mode | Use case | Latency |
|---|---|---|
| no explanation | fastest retrieval | fastest |
| template explanation | product site cards | fast |
| local LLM explanation | richer report text | slower |

Product path should usually use:

```text
with_explanations = false
use_template_explanations = true
```

LLM explanation should be optional or report-only.

## 10. MLOps Logging

PostgreSQL logging is available for product feedback and future retraining.

Main tables:

| Table | Purpose |
|---|---|
| `retrieval_requests` | query, strategy, filters, model versions, latency |
| `retrieval_results` | one row per returned ranked site |
| `user_feedback` | click/save/dismiss/select/manual feedback |
| `report_jobs` | markdown/PDF report generation status |
| `model_registry` | model versions, artifact paths, metrics, status |

Feedback endpoint example:

```bash
curl -X POST http://localhost:8001/feedback \\
  -H "Content-Type: application/json" \\
  -d '{
    "request_id": "req_xxx",
    "rid": "5304579",
    "rank_position": 1,
    "event_type": "save",
    "event_value": {
      "source": "site_card"
    },
    "user_note": "User saved this site."
  }'
```

## 11. Report Jobs

Reports can be generated from a logged `request_id`.

```bash
curl -X POST http://localhost:8001/report-jobs \\
  -H "Content-Type: application/json" \\
  -d '{
    "request_id": "req_xxx",
    "explanation_mode": "template",
    "output_markdown": true,
    "output_pdf": true,
    "audience": "developer",
    "title": "Smart Developer Site Recommendation Report"
  }'
```

This generates:

```text
algorithm/artifacts/reports/report_xxx.md
algorithm/artifacts/reports/report_xxx.pdf
```

Check status:

```bash
curl http://localhost:8001/report-jobs/report_xxx
```

## 12. Feedback Dataset Export

Feedback logs can be exported into a future reranker training dataset.

```bash
python -m algorithm.src.mlops.build_feedback_dataset
```

Useful variants:

```bash
python -m algorithm.src.mlops.build_feedback_dataset --labelled-only
python -m algorithm.src.mlops.build_feedback_dataset --weak-negative-unfeedbacked
```

Current label mapping:

| Event | Label |
|---|---:|
| `click` | 0.5 |
| `save` / `select` / `manual_positive` | 1.0 |
| `dismiss` / `manual_negative` | 0.0 |
| shown but no feedback | unlabelled by default |

## 13. Demo Scripts

Useful scripts:

```bash
./scripts/run_demo_retrieval.sh
./scripts/run_demo_report.sh
./scripts/export_feedback_dataset.sh
./scripts/demo_full_mlops_flow.sh
```

Full local MLOps demo:

```bash
./scripts/demo_full_mlops_flow.sh
```

This runs:

```text
health check
-> retrieval
-> feedback
-> report job
-> report status
```

## 14. Minimal Smoke Test

If using CLI demo:

```bash
python -m algorithm.demo_retrieval \\
  --experiment two_tower_v1 \\
  --strategy low_rise_apartment \\
  --query-text "I want a site for low-rise apartment redevelopment near a train station, with high development zoning, a large site, and limited planning constraints." \\
  --top-k 5 \\
  --recall-k 1000 \\
  --locality WAITARA
```

If using service demo:

```bash
curl http://localhost:8001/health
./scripts/demo_full_mlops_flow.sh
```

## 15. Current Limitations

- Labels are still weakly supervised from heuristic strategy scores.
- Some strategies overlap heavily, especially house redevelopment and dual occupancy.
- Current location filtering is address-text based.
- DCN probability is useful for ranking but should not be treated as a calibrated product-facing probability.
- Local LLM explanations are slower and should not block the main retrieval request.
- Candidate representation is still property-level, so base-site dedupe remains important.

## 16. Recommended Default

For internal backend integration:

```text
Long-running FastAPI service
PostgreSQL logging enabled
retrieval_model = two_tower_v1
reranking_model = dcn_reranker_v1
with_explanations = false
use_template_explanations = true
dedupe_by_address = true
recall_k = 1000
```

This gives fast retrieval, product-friendly explanations, request/result/feedback logging, and downloadable report generation.