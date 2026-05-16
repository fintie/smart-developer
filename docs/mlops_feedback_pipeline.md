# Smart Developer MLOps and Feedback Pipeline

## Purpose

This document describes the next-stage MLOps design for the Smart Developer algorithm pipeline.

The current algorithm pipeline already supports strategy-aware site retrieval, DCN reranking, base-site dedupe, explanation generation, markdown report generation, and PDF export.

The next goal is to make the system more deployable and more product-ready by adding:

- a fast inference path for product search
- an asynchronous explanation / report generation path
- request and result logging
- user feedback collection
- model version tracking
- retraining and model promotion workflow

This document is intended for internal discussion between algorithm, backend, frontend, and product.

---

## 1. Current System

The current default inference stack is:

```text
strategy + query_text
-> query planner
-> two_tower_v1 recall
-> dcn_reranker_v1 rerank
-> location / address filtering
-> base-site dedupe
-> optional explanation
-> return top-k results
```

The main backend-facing Python entrypoint is:

```python
from algorithm.src.inference.predictor import retrieve_sites
```

Current recommended model names:

```text
retrieval_model = two_tower_v1
reranking_model = dcn_reranker_v1
```

Current runtime artifacts include:

```text
data/processed/retrieval/candidate_sites.parquet
data/processed/retrieval/query_intents.jsonl
algorithm/artifacts/models/two_tower_v1/model.pt
algorithm/artifacts/models/dcn_reranker_v1/model.pt
algorithm/artifacts/models/dcn_reranker_v1/preprocessing.json
```

---

## 2. Main Product Issue: Inference Latency

The current local explanation generation uses Ollama. This is useful for internal testing because it avoids external hosted LLM dependencies, but it is too slow to block the main product search request.

The main reason is that explanation generation may require multiple local LLM calls, especially when each shortlisted site receives its own rationale.

For product use, the retrieval request should not wait for local LLM generation.

Therefore, the system should be split into:

```text
Fast Path: retrieval / reranking / dedupe
Slow Path: explanation / report / PDF generation
```

---

## 3. Recommended Product Architecture

```text
Frontend
  |
  | POST /api/retrieve-sites
  v
Backend API
  |
  | calls predictor / algorithm service
  v
Fast Retrieval Path
  |
  | query planner
  | two-tower retrieval
  | DCN reranker
  | filtering
  | base-site dedupe
  v
Return top-k sites immediately
  |
  | optional async job
  v
Slow Report Path
  |
  | generate explanations
  | build markdown report
  | export PDF
  | cache report artifact
  v
Frontend polls report status / downloads report
```

The user-facing experience should be:

```text
1. User submits query.
2. Backend returns ranked site shortlist quickly.
3. Frontend displays site cards / table immediately.
4. Report or explanation is shown as pending / generating.
5. Backend generates report asynchronously or on demand.
6. Frontend displays or downloads markdown / PDF report once ready.
```

---

## 4. Fast Path: Retrieval Endpoint

### Purpose

The retrieval endpoint should return the ranked shortlist as quickly as possible.

It should not call Ollama or generate long-form explanations synchronously.

### Suggested endpoint

```text
POST /api/retrieve-sites
```

### Suggested request

```json
{
  "strategy": "low_rise_apartment",
  "query_text": "I want a site for low-rise apartment redevelopment near a train station, with high development zoning, a large site, and limited planning constraints.",
  "top_k": 5,
  "recall_k": 1000,
  "locality": "WAITARA",
  "address_contains": null,
  "retrieval_model": "two_tower_v1",
  "reranking_model": "dcn_reranker_v1",
  "use_dcn_reranker": true,
  "dedupe_by_address": true,
  "with_explanations": false
}
```

### Suggested response

```json
{
  "request_id": "req_20260504_000001",
  "status": "retrieval_complete",
  "result_count": 5,
  "latency_ms": 842,
  "request": {
    "strategy": "low_rise_apartment",
    "query_text": "...",
    "retrieval_model": "two_tower_v1",
    "reranking_model": "dcn_reranker_v1"
  },
  "results": [
    {
      "rank_position": 1,
      "RID": 5304579,
      "address": "623/21-37 WAITARA AVENUE WAITARA",
      "base_site_address": "21-37 WAITARA AVENUE WAITARA",
      "primary_zoning_code": "R4",
      "lot_size_band": "xl",
      "station_distance_band": "within_800m",
      "constraint_severity_band": "low",
      "strategy_score": 97.5,
      "retrieval_similarity": 0.81,
      "dcn_prob": 0.99,
      "dcn_rank_score": 1.01
    }
  ],
  "report_status": "not_generated"
}
```

### Notes

- `with_explanations=False` should be the default for product search.
- `dcn_prob` is useful internally for ranking but should not be displayed as a user-facing probability.
- `base_site_address` should be displayed in the frontend instead of unit-level `address`.
- `address` can still be retained as source/debug information.

---

## 5. Slow Path: Report Generation Endpoint

### Purpose

The report generation endpoint should produce markdown and/or PDF reports after retrieval results already exist.

This can be synchronous for internal demo, but should ideally become asynchronous for product use.

### Suggested endpoint

```text
POST /api/reports
```

### Suggested request

```json
{
  "request_id": "req_20260504_000001",
  "formats": ["markdown", "pdf"],
  "explanation_mode": "template",
  "audience": "developer"
}
```

### Suggested response

```json
{
  "report_id": "report_20260504_000001",
  "request_id": "req_20260504_000001",
  "status": "queued"
}
```

### Suggested status endpoint

```text
GET /api/reports/{report_id}
```

### Suggested status response

```json
{
  "report_id": "report_20260504_000001",
  "request_id": "req_20260504_000001",
  "status": "ready",
  "markdown_path": "algorithm/artifacts/reports/report_20260504_000001.md",
  "pdf_path": "algorithm/artifacts/reports/report_20260504_000001.pdf",
  "latency_ms": 3260
}
```

---

## 6. Explanation Modes

To reduce latency, explanation generation should support multiple modes.

### Mode 1: `none`

No explanation is generated.

Use for fastest retrieval.

```text
retrieval only
```

### Mode 2: `template`

Generate deterministic explanation text from structured site features.

This should be fast and stable.

Example:

```text
This site ranks highly for low-rise apartment redevelopment because it combines R4 zoning, an XL lot-size band, close rail/metro access, and low identified planning constraints.
```

Recommended use:

```text
site cards, frontend shortlist, fast report draft
```

### Mode 3: `local_llm_async`

Generate richer explanation text using local Ollama in the background.

Recommended use:

```text
internal reports, developer/investor-style long-form report generation
```

### Mode 4: `cached`

Return an existing report/explanation if the same request has already been generated.

Cache key can include:

```text
strategy
query_text
locality
address_contains
top_rids
retrieval_model
reranking_model
explanation_mode
explanation_model
```

---

## 7. Suggested Database Tables

The system should log enough data to support product analytics, debugging, and future model retraining.

The first version can be implemented in PostgreSQL.

---

## 7.1 `retrieval_requests`

Stores one row per retrieval request.

Suggested columns:

| Column | Type | Description |
|---|---:|---|
| `request_id` | text / uuid | Unique request ID |
| `created_at` | timestamp | Request timestamp |
| `user_id` | text nullable | Optional user ID |
| `session_id` | text nullable | Optional anonymous session ID |
| `raw_query_text` | text | Original user query |
| `selected_strategy` | text | Strategy selected by user or backend |
| `planner_selected_strategy` | text nullable | Strategy after query planner |
| `planner_rewritten_query` | text nullable | Rewritten query used for retrieval |
| `planner_warnings` | jsonb nullable | Planner warnings / alternatives |
| `locality` | text nullable | Locality filter |
| `address_contains` | text nullable | Address text filter |
| `top_k` | integer | Number of returned results |
| `recall_k` | integer | Recall candidate count |
| `retrieval_model` | text | Retrieval model name |
| `reranking_model` | text | Reranking model name |
| `use_dcn_reranker` | boolean | Whether DCN reranker was enabled |
| `dedupe_by_address` | boolean | Whether base-site dedupe was enabled |
| `with_explanations` | boolean | Whether explanation was requested synchronously |
| `latency_ms` | integer nullable | Retrieval latency |
| `status` | text | success / failed |
| `error_message` | text nullable | Error details if failed |

---

## 7.2 `retrieval_results`

Stores one row per result shown to the user.

Suggested columns:

| Column | Type | Description |
|---|---:|---|
| `request_id` | text / uuid | Parent request ID |
| `rid` | text / bigint | Candidate property/site ID |
| `rank_position` | integer | Rank shown to user |
| `address` | text | Original source address |
| `base_site_address` | text | Dedupe-normalised site address |
| `primary_zoning_code` | text nullable | Zoning code |
| `zoning_band` | text nullable | Zoning band |
| `lot_size_proxy_sqm` | double precision nullable | Approx site area |
| `lot_size_band` | text nullable | Lot size band |
| `station_distance_band` | text nullable | Station distance band |
| `constraint_severity_band` | text nullable | Overall constraint band |
| `strategy_score` | double precision nullable | Heuristic strategy score |
| `retrieval_similarity` | double precision nullable | Retrieval similarity |
| `fusion_score` | double precision nullable | Fusion score |
| `serving_boost` | double precision nullable | Serving adjustment |
| `dcn_prob` | double precision nullable | DCN output for ranking |
| `dcn_rank_score` | double precision nullable | Final rerank score |
| `shown_to_user` | boolean | Whether displayed in frontend |
| `created_at` | timestamp | Insert timestamp |

Notes:

- `dcn_prob` should be stored for debugging/training but not treated as a calibrated probability.
- `base_site_address` should be the primary display field.

---

## 7.3 `user_feedback`

Stores explicit and implicit user interactions.

Suggested columns:

| Column | Type | Description |
|---|---:|---|
| `feedback_id` | text / uuid | Unique feedback ID |
| `request_id` | text / uuid | Parent request ID |
| `rid` | text / bigint | Site ID |
| `rank_position` | integer nullable | Position at time of display |
| `event_type` | text | Feedback event |
| `event_value` | jsonb nullable | Optional event payload |
| `user_note` | text nullable | Optional user note |
| `created_at` | timestamp | Feedback timestamp |

Suggested `event_type` values:

```text
view
click
save
dismiss
select
export_report
share_pdf
manual_positive
manual_negative
```

Possible training interpretation:

| Event | Label interpretation |
|---|---|
| `click` | weak positive |
| `save` | positive |
| `select` | strong positive |
| `dismiss` | weak negative |
| `manual_positive` | strong positive |
| `manual_negative` | strong negative |

---

## 7.4 `report_jobs`

Stores report generation jobs and output paths.

Suggested columns:

| Column | Type | Description |
|---|---:|---|
| `report_id` | text / uuid | Unique report ID |
| `request_id` | text / uuid | Linked retrieval request |
| `created_at` | timestamp | Job creation time |
| `started_at` | timestamp nullable | Job start time |
| `finished_at` | timestamp nullable | Job completion time |
| `status` | text | queued / running / ready / failed |
| `explanation_mode` | text | none / template / local_llm_async / cached |
| `explanation_model` | text nullable | Ollama model if used |
| `report_format` | text | markdown / pdf / both |
| `markdown_path` | text nullable | Markdown report path |
| `pdf_path` | text nullable | PDF report path |
| `latency_ms` | integer nullable | Job runtime |
| `error_message` | text nullable | Error details if failed |

---

## 7.5 `model_registry`

Stores model metadata and production status.

Suggested columns:

| Column | Type | Description |
|---|---:|---|
| `model_version` | text | Model version name |
| `model_type` | text | retrieval / reranker / explanation |
| `artifact_path` | text | Main model artifact path |
| `preprocessing_path` | text nullable | Preprocessing artifact path |
| `metrics_path` | text nullable | Metrics JSON path |
| `training_data_path` | text nullable | Training data path |
| `eval_data_path` | text nullable | Evaluation data path |
| `status` | text | candidate / staging / production / archived |
| `created_at` | timestamp | Creation time |
| `promoted_at` | timestamp nullable | Production promotion time |
| `notes` | text nullable | Human-readable notes |

Example statuses:

```text
candidate
staging
production
archived
```

The production backend should ideally load model versions from config or registry instead of hardcoding them.

---

## 8. Feedback-to-Training Pipeline

The long-term goal is to move from purely weakly supervised heuristic labels to product-feedback-informed training data.

Current training labels are derived from strategy score thresholds.

Future training data can combine:

```text
heuristic strategy score labels
+ user clicks
+ saved sites
+ selected sites
+ dismissed sites
+ manual review labels
```

Suggested offline training flow:

```text
retrieval_requests
+ retrieval_results
+ user_feedback
-> build_feedback_training_dataset.py
-> train_dcn_reranker_v2
-> evaluate against production reranker
-> write metrics and model card
-> manual review
-> promote if better
```

---

## 9. Candidate Training Label Design

Possible label rules for reranker training:

```text
selected site        -> label 1.0
saved site           -> label 1.0
clicked site         -> label 0.7
dismissed site       -> label 0.0
shown but ignored    -> weak negative, label 0.2 or sample as negative carefully
manual positive      -> label 1.0
manual negative      -> label 0.0
```

Important caution:

```text
A result not clicked is not always a true negative.
```

Therefore, ignored results should be treated carefully, ideally with position bias awareness.

---

## 10. Model Promotion Workflow

The model lifecycle should be manual at first.

Recommended stages:

```text
train candidate model
-> evaluate candidate
-> compare to production model
-> human review
-> promote to production
```

Example:

```text
dcn_reranker_v1 = production
dcn_reranker_v2 = candidate
```

If v2 improves offline metrics and passes sanity checks:

```text
dcn_reranker_v2 -> staging
staging smoke test
staging -> production
```

Avoid automatic model promotion until more reliable feedback data exists.

---

## 11. Suggested Metrics

### Retrieval / ranking metrics

```text
top10_match_rate
top20_match_rate
top20_mean_strategy_score
top20_high_score_rate
NDCG@k if labels are available
click-through rate after deployment
save rate
selected-site rate
dismiss rate
```

### Product metrics

```text
search latency
report generation latency
PDF export success rate
user report download rate
query-to-save conversion
query-to-selected-site conversion
```

### System metrics

```text
request count
error rate
p50 / p95 / p99 latency
model load time
report job queue time
report job failure rate
cache hit rate
```

---

## 12. Latency Strategy

### Recommended default

```text
retrieve-sites endpoint:
  with_explanations = false

report generation endpoint:
  explanation_mode = template by default
  local_llm_async optional
```

### Recommended optimisation order

1. Do not block retrieval on Ollama.
2. Add deterministic template explanations.
3. Cache reports and explanations.
4. Batch LLM explanation for top-k sites if LLM is still needed.
5. Try smaller local models if quality is acceptable.
6. Consider dedicated model serving infrastructure later.

---

## 13. Template Explanation Strategy

A deterministic explanation layer should generate short rationales from structured features.

Example inputs:

```text
strategy
primary_zoning_code
zoning_band
lot_size_band
lot_size_proxy_sqm
station_distance_band
distance_to_station_m
constraint_severity_band
heritage_flag
flood_flag
bushfire_flag
strategy_score
```

Example output:

```text
This site is a strong fit for low-rise apartment redevelopment because it combines R4 zoning, an XL lot-size band, close access to rail/metro, and low identified planning constraints. These signals support higher-intensity residential redevelopment screening, although formal planning controls and site-specific constraints should still be verified.
```

This can be used for:

```text
frontend site cards
fast markdown report
fallback if Ollama fails
```

---

## 14. Report Generation Strategy

The current report layer supports:

```text
ranked results
-> markdown report
-> PDF export
```

Recommended product behaviour:

```text
Search results return immediately.
User clicks Generate Report.
Backend creates report job.
Frontend shows report status.
Report job produces markdown and PDF.
User views or downloads PDF.
```

The PDF export should be treated as an output artifact, not as part of the retrieval response.

---

## 15. Minimal Implementation Plan

### Phase 1: Documentation and metadata

- Add this MLOps feedback pipeline document.
- Add latency metadata to predictor response.
- Add model version metadata to response.
- Keep `with_explanations=False` as recommended product default.

### Phase 2: Fast explanation fallback

- Add deterministic template explanation generator.
- Use template explanations for fast reports.
- Keep local LLM as optional rich explanation mode.

### Phase 3: Logging schema

- Add SQL schema draft for:
  - `retrieval_requests`
  - `retrieval_results`
  - `user_feedback`
  - `report_jobs`
  - `model_registry`

### Phase 4: Async report jobs

- Add report job abstraction.
- Generate markdown/PDF by request ID.
- Store paths and status.

### Phase 5: Feedback training dataset

- Build offline script:

```text
algorithm/src/training/build_feedback_training_dataset.py
```

- Train a candidate reranker version.
- Evaluate candidate vs production.

### Phase 6: Model registry / promotion

- Add model cards.
- Track active production model.
- Support manual promotion.

---

## 16. Backend Integration Notes

Backend should ideally own:

```text
HTTP endpoints
request IDs
DB inserts
user/session identity
frontend report status API
artifact file serving
```

Algorithm layer should own:

```text
retrieval and reranking
query planning
feature-based explanations
local LLM explanation implementation
markdown report construction
PDF export
model artifacts and metrics
training scripts
```

---

## 17. Suggested API Split

### Search

```text
POST /api/retrieve-sites
```

Fast, synchronous.

### Report

```text
POST /api/reports
GET /api/reports/{report_id}
```

Potentially asynchronous.

### Feedback

```text
POST /api/feedback
```

Logs user behaviour.

### Model status

```text
GET /api/models/active
```

Optional internal endpoint for debugging.

---

## 18. Current Limitations

- Current model labels are weakly supervised from heuristic strategy scores.
- DCN output is useful for ranking but is not calibrated as a probability.
- Local LLM explanations are too slow for synchronous product search.
- Location filtering is currently basic and should later use structured suburb/postcode/LGA/geospatial filters.
- User feedback is not yet captured.
- Model promotion is not yet formalised.

---

## 19. Recommended Near-Term Decision

The next engineering step should not be more model tuning.

The recommended near-term focus is:

```text
1. Keep retrieval fast and synchronous.
2. Move explanation/report generation to an async or on-demand path.
3. Add request/result/feedback logging schema.
4. Add deterministic template explanations as a fast fallback.
5. Add model/version metadata to support future retraining and promotion.
```

This will turn the current Smart Developer algorithm demo into a more realistic product and MLOps pipeline.

## 20. Testing
```bash
curl -X POST http://localhost:8001/retrieve-sites \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "single_dwelling_rebuild",
    "query_text": "I want a site for detached house redevelopment on standard residential land, with low planning constraints and a suitable lot size.",
    "top_k": 5,
    "recall_k": 1000,
    "with_explanations": false,
    "user_id": "demo_user",
    "session_id": "meeting_demo_house",
    "log_request": true
  }' | python -m json.tool
```

```sql
SELECT request_id, strategy, locality, latency_ms, result_count
FROM retrieval_requests
ORDER BY created_at DESC
LIMIT 5;
```