from algorithm.src.inference.predictor import retrieve_sites

# Deployment Handoff (for Backend)

## Purpose

This document is for internal backend / full-stack integration of the current Smart Developer retrieval pipeline.

Current recommended inference chain:

```text
strategy + query_text
-> two_tower_v1 recall
-> dcn_reranker_v1 rerank
-> dedupe
-> optional explanation
-> return top-k
```

Main Python entrypoint for backend use:
```python
from algorithm.src.inference.predictor import retrieve_sites
```

## Environment Setup
### Python

Use Python 3.10+.

Create `env` and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Local explanation model
Explanation currently uses a local Ollama model.

Install Ollama, then pull:
```bash
ollama pull llama3.1:8b-instruct-q4_K_M
```

If explanation is not needed at deployment time, backend can call predictor with:
* `with_explanations=False`

This removes the Ollama dependency from inference.

## Required Artifacts
### Required for inference
These files MUST exist locally on the deployment machine:

**Data**: `data/processed/retrieval/candidate_sites.parquet`

**Retrieval model**: `algorithm/artifacts/models/two_tower_v1/model.pt`

**DCN reranker**: 
* `algorithm/artifacts/models/dcn_reranker_v1/model.pt`
* `algorithm/artifacts/models/dcn_reranker_v1/preprocessing.json`

**Optional**: If explanation is enabled, local Ollama model installed and available.

## How to Produce Artifacts
If processed base data already exists, the practical rebuild flow is:
```bash
# Build training data
python -m algorithm.src.features.build_features
python -m algorithm.src.scoring.scoring
python -m algorithm.src.retrieval.build_candidate_sites
python -m algorithm.src.retrieval.build_training_pairs

# Train retrieval model
python -m algorithm.src.models.train_two_tower_v1 --experiment two_tower_v1
python -m algorithm.src.models.evaluate_two_tower --experiment two_tower_v1

# Train reranking model
python -m algorithm.src.models.train_dcn_reranker --experiment dcn_reranker_v1
```

If artifacts already built, backend does not need to rerun training.

## Backend Calling Interface
### Recommended backend call

Example:

```python
from algorithm.src.inference.predictor import retrieve_sites

response = retrieve_sites(
    strategy="low_rise_apartment",
    query_text="I want a site for low-rise apartment redevelopment near a train station, with high development zoning, a large site, and limited planning constraints.",
    top_k=5,
    with_explanations=True,
    retrieval_model="two_tower_v1",
    use_dcn_reranker=True,
    reranking_model="dcn_reranker_v1",
)
```

### Response shape
Returns a JSON-serialisable dictionary:
```json
{
  "request": {...},
  "result_count": 5,
  "results": [
    {
      "RID": ...,
      "address": ...,
      "primary_zoning_code": ...,
      "lot_size_band": ...,
      "constraint_severity_band": ...,
      "station_distance_band": ...,
      "top_strategy": ...,
      "strategy_score": ...,
      "retrieval_similarity": ...,
      "fusion_score": ...,
      "dcn_prob": ...,
      "explanation": ...
    }
  ]
}
```
* `strategy` is currently required
* `query_text` is free text
* `with_explanations=False` is recommended for lower-latency deployments if explanation is not needed inline.

## Suggested Deployment Shape
### Simplest deployment
Python service wraps `retrieve_sites(...)` and exposes HTTP.

Suggested API shape:

**Request**
```json
{
  "strategy": "low_rise_apartment",
  "query_text": "I want a site for low-rise apartment redevelopment near a train station...",
  "top_k": 5,
  "with_explanations": true
}
```

**Response**
```json
{
  "request": {...},
  "result_count": 5,
  "results": [...]
}
```

### Suggested serving behaviour
* load models once at service start
* keep predictor singleton in memory
* do not reload candidate parquet on every request
* explanations can be optional / async if latency matters

## Recommended DB Design
Current interface does not require a database if parquet artifacts are already local.

For backend integration, recommended DB design is:

### A. Source-of-truth analytical table
One site-level table, one row per candidate site / property entity.

Suggested columns:
* `rid`
* `primary_zoning_code`
* `zoning_band`
* `lot_size_proxy_sqm`
* `lot_size_band`
* `mixed_zoning_flag`
* `heritage_flag`
* `heritage_max_significance`
* `bushfire_risk_level`
* `flood_flag`
* `primary_flood_class`
* `distance_to_station_m`
* `within_800m_catchment`
* `station_distance_band`
* all strategy score columns
* optional explanation cache fields

This table should align with `data/processed/retrieval/candidate_sites.parquet`

### B. Request / retrieval log table

Store user query and returned results.

Suggested fields:
* `request_id`
* `timestamp`
* `strategy`
* `query_text`
* `top_k`
* `with_explanations`
* `retrieval_experiment`
* `dcn_experiment`
* `returned_rids` or child result rows

### C. Feedback table
If frontend later supports user feedback, keep a separate table:

Suggested fields:
* `request_id`
* `rid`
* `rank_position`
* `clicked`
* `saved`
* `dismissed`
* `manual_label`
* `user_note`

This is the cleanest path for future training data.

## Suggested Extension Points
### A. Request logging
Store every retrieval request and returned shortlist.

Useful for:
* debugging
* product analytics
* future training data

### B. User feedback loop
Potential future feedback:
* clicked result
* shortlisted result
* rejected result
* final selected site

This can give us more training data for the model and become better supervision than current weak labels.

### C. Explanation mode split
Two deployment modes:
* synchronous explanation in response
* asynchronous explanation generation after retrieval

### D. Filter-aware retrieval
Backend can later expose structured filters such as:
* zoning include/exclude
* min lot size
* max station distance
* constraint exclusions

These can be applied before or after recall.

### E. Cached result store
For repeated demo / common strategy queries, backend can cache:
* normalised request
* returned shortlist
* explanations

## Current Limitations
* Current labels are weakly supervised from heuristic strategy scores.
* Some strategies overlap heavily.
* Explanation generation is useful but still somewhat repetitive for very similar sites.
* Strategy must currently be provided explicitly.
* Current candidate representation is property-level, so some post-processing / dedupe is still important.

## Recommended Default Deployment Mode
For now, recommended default is:
* retrieval model: `two_tower_v1`
* reranker: `dcn_reranker_v1`
* dedupe: enabled
* explanations: enabled for demo, optional for production-like backend
* predictor entrypoint: `algorithm.src.inference.predictor.retrieve_sites`

## Minimal Smoke Test
After environment setup and artifact placement:
```bash
python -m algorithm.demo_retrieval \
  --experiment two_tower_v1 \
  --strategy low_rise_apartment \
  --query-text "I want a site for low-rise apartment redevelopment near a train station, with high development zoning, a large site, and limited planning constraints." \
  --top-k 5 \
  --with-explanations
```

If this works, backend deployment should also be able to call the same pipeline through `predictor.py`.