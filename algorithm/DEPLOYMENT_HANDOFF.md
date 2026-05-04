# Deployment Handoff (for Backend)

## Purpose

This document is for internal backend / full-stack integration of the current Smart Developer retrieval pipeline.

Current recommended inference chain:

```text
strategy + query_text
-> optional query planner / query rewrite
-> two_tower_v1 recall
-> optional location filter
-> dcn_reranker_v1 rerank
-> base-site dedupe
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

Create environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Local explanation model

Explanation currently uses a local Ollama model.

Install Ollama, then in another terminal run:

```bash
ollama serve
```

Return to the working terminal and pull:

```bash
ollama pull llama3.1:8b-instruct-q4_K_M
```

If explanation is not needed at deployment time, backend can call predictor with:

```python
with_explanations=False
```

This removes the Ollama dependency from inference.

## Required Artifacts

These can be found in the shared Google Drive. They must exist locally on the deployment machine.

### Required for inference

#### Retrieval data

```text
data/processed/retrieval/
├── candidate_sites.parquet
└── query_intents.jsonl
```

Required files:
- `data/processed/retrieval/candidate_sites.parquet`
- `data/processed/retrieval/query_intents.jsonl`

`candidate_sites.parquet` is the candidate table used at inference time.

`query_intents.jsonl` is required by evaluation/demo flows and should be included in the runtime artifact bundle so the same repo setup can run demos, smoke tests, and evaluation scripts consistently.

#### Retrieval model

```text
algorithm/artifacts/models/two_tower_v1/
└── model.pt
```

Required file:
- `algorithm/artifacts/models/two_tower_v1/model.pt`

#### DCN reranker

```text
algorithm/artifacts/models/dcn_reranker_v1/
├── model.pt
└── preprocessing.json
```

Required files:
- `algorithm/artifacts/models/dcn_reranker_v1/model.pt`
- `algorithm/artifacts/models/dcn_reranker_v1/preprocessing.json`

### Optional

If explanation is enabled:
- Ollama must be installed
- the configured local model must be available

## How to Produce Artifacts

If processed base data already exists, the practical rebuild flow is:

```bash
# Build feature and retrieval data
python -m algorithm.src.features.build_features
python -m algorithm.src.scoring.scoring
python -m algorithm.src.retrieval.build_candidate_sites
python -m algorithm.src.retrieval.build_training_pairs

# Train and evaluate retrieval model
python -m algorithm.src.models.train_two_tower_v1 --experiment two_tower_v1
python -m algorithm.src.models.evaluate_two_tower --experiment two_tower_v1

# Build reranker dataset
python -m algorithm.src.retrieval.build_reranker_dataset \
  --experiment dcn_reranker_v1 \
  --recall-experiment two_tower_v1

# Train and evaluate reranking model
python -m algorithm.src.models.train_dcn_reranker --experiment dcn_reranker_v1
python -m algorithm.src.models.evaluate_dcn_reranker \
  --experiment dcn_reranker_v1 \
  --recall-experiment two_tower_v1
```

If artifacts are already built, backend does not need to rerun training.

A full local training pipeline can also be run with:

```bash
chmod +x ./scripts/run_training.sh
./scripts/run_training.sh
```

## Backend Calling Interface

### Recommended backend call

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
    with_explanations=True,
    retrieval_experiment="two_tower_v1",
    use_dcn_reranker=True,
    dcn_experiment="dcn_reranker_v1",
    locality="WAITARA",          # optional
    address_contains=None,       # optional
)
```

### Important parameter names

Use these current names:

- `retrieval_experiment="two_tower_v1"`
- `dcn_experiment="dcn_reranker_v1"`
- `use_dcn_reranker=True`

Do not use older names such as `retrieval_model` or `reranking_model` unless the backend wrapper explicitly maps them.

### Response shape

Returns a JSON-serialisable dictionary:

```json
{
  "request": {...},
  "result_count": 5,
  "results": [
    {
      "RID": 5304579,
      "address": "623/21-37 WAITARA AVENUE WAITARA",
      "base_site_address": "21-37 WAITARA AVENUE WAITARA",
      "primary_zoning_code": "R4",
      "lot_size_band": "xl",
      "constraint_severity_band": "low",
      "station_distance_band": "within_800m",
      "top_strategy": "low_rise_apartment",
      "strategy_score": 97.5,
      "retrieval_similarity": 0.67,
      "fusion_score": 0.91,
      "serving_boost": 0.0,
      "dcn_prob": 0.99,
      "dcn_rank_score": 0.99,
      "explanation": "..."
    }
  ]
}
```

Notes:
- `strategy` is currently required.
- `query_text` is free text.
- `base_site_address` is the preferred display address for frontend/report output.
- `address` is the raw source address example and may be unit-level.
- `with_explanations=False` is recommended for lower-latency deployments if explanation is not needed inline.

## Supported Product-Layer Options

### Location filter

The current lightweight location filter works against address text.

Supported fields:
- `locality`
- `address_contains`

Example:

```python
response = retrieve_sites(
    strategy="low_rise_apartment",
    query_text="I want a site for low-rise apartment redevelopment near a train station.",
    top_k=5,
    recall_k=1000,
    locality="WAITARA",
)
```

Because the current candidate table does not yet include structured `locality`, `postcode`, or `lga_name`, filtering is currently text-based on `address`. A future candidate table should include structured location fields.

### Base-site dedupe

The current retriever dedupes unit-level addresses into base-site addresses.

Example:

```text
623/21-37 WAITARA AVENUE WAITARA
-> 21-37 WAITARA AVENUE WAITARA
```

Frontend should generally display `base_site_address` first.

### Query planner

The lightweight query planner can rewrite contradictory user queries and suggest alternative strategies.

Example problem:

```text
strategy = single_dwelling_rebuild
query = house redevelopment near station with high development zoning and large site
```

The planner can:
- keep the selected strategy
- sanitise conflicting phrases
- warn that the query also matches higher-intensity strategies
- suggest alternatives such as `low_rise_apartment` or `land_bank_hold`

This is useful for product UX but should remain transparent to the user or operator.

## Suggested Deployment Shape

### Simplest deployment

Python service wraps `retrieve_sites(...)` and exposes HTTP.

Suggested API request:

```json
{
  "strategy": "low_rise_apartment",
  "query_text": "I want a site for low-rise apartment redevelopment near a train station...",
  "top_k": 5,
  "recall_k": 1000,
  "with_explanations": true,
  "locality": "WAITARA"
}
```

Suggested API response:

```json
{
  "request": {...},
  "result_count": 5,
  "results": [...]
}
```

### Suggested serving behaviour

- Load models once at service start.
- Keep predictor singleton in memory.
- Do not reload candidate parquet on every request.
- Use `base_site_address` for display.
- Keep `address` as source/debug field.
- Make explanations optional or async if latency matters.
- For locality/address filters, use a larger `recall_k` such as `1000`.

## Recommended DB Design

Current interface does not require a database if parquet artifacts are already local.

For backend integration, recommended DB design is below.

### A. Source-of-truth analytical table

One site-level table, one row per candidate site / property entity.

Suggested columns:
- `rid`
- `address`
- `base_site_address`
- `primary_zoning_code`
- `zoning_band`
- `lot_size_proxy_sqm`
- `lot_size_band`
- `mixed_zoning_flag`
- `heritage_flag`
- `heritage_max_significance`
- `bushfire_risk_level`
- `flood_flag`
- `primary_flood_class`
- `distance_to_station_m`
- `within_800m_catchment`
- `station_distance_band`
- all strategy score columns
- optional explanation cache fields

This table should align with `data/processed/retrieval/candidate_sites.parquet`.

Future preferred location fields:
- `locality`
- `postcode`
- `lga_name`
- `latitude`
- `longitude`

These will make location filtering much cleaner than address-text matching.

### B. Request / retrieval log table

Store user query and returned results.

Suggested fields:
- `request_id`
- `timestamp`
- `strategy`
- `query_text`
- `rewritten_query`
- `top_k`
- `recall_k`
- `with_explanations`
- `retrieval_experiment`
- `dcn_experiment`
- `use_dcn_reranker`
- `locality`
- `address_contains`
- `returned_rids` or child result rows

### C. Result rows table

For cleaner analytics, store one row per returned site.

Suggested fields:
- `request_id`
- `rid`
- `rank_position`
- `address`
- `base_site_address`
- `strategy_score`
- `retrieval_similarity`
- `fusion_score`
- `dcn_prob`
- `dcn_rank_score`

### D. Feedback table

If frontend later supports user feedback, keep a separate table:

Suggested fields:
- `request_id`
- `rid`
- `rank_position`
- `clicked`
- `saved`
- `dismissed`
- `manual_label`
- `user_note`

This is the cleanest path for future training data.

## Suggested Extension Points

### A. Request logging

Store every retrieval request and returned shortlist.

Useful for:
- debugging
- product analytics
- future training data

### B. User feedback loop

Potential future feedback:
- clicked result
- shortlisted result
- rejected result
- final selected site

This can provide better supervision than the current weak labels.

### C. Explanation mode split

Two deployment modes:
- synchronous explanation in response
- asynchronous explanation generation after retrieval

### D. Filter-aware retrieval

Backend can later expose structured filters such as:
- zoning include/exclude
- min lot size
- max station distance
- locality / LGA / postcode
- constraint exclusions

Current filters are lightweight and address-text based. Future filters should ideally use structured DB fields.

### E. Cached result store

For repeated demo / common strategy queries, backend can cache:
- normalised request
- returned shortlist
- explanations

### F. Report generation

Report generation is available through the explanation/report layer and can be exposed as:
- markdown report
- HTML report
- later PDF export

The current report output is suitable for internal developer/investor-style shortlist demos.

## Current Limitations

- Current labels are weakly supervised from heuristic strategy scores.
- Some strategies overlap heavily.
- Explanation generation is useful but still somewhat repetitive for very similar sites.
- Strategy is currently expected as an explicit input, although the query planner can assist.
- Current location filtering is address-text based because structured locality/postcode fields are not yet in `candidate_sites.parquet`.
- Current candidate representation is property-level, so base-site dedupe is important.

## Recommended Default Deployment Mode

For now, recommended default is:
- retrieval experiment: `two_tower_v1`
- reranker experiment: `dcn_reranker_v1`
- query planner: optional, useful for user-facing input
- location filter: optional, use `locality` / `address_contains`
- base-site dedupe: enabled
- explanations: enabled for demo, optional for production-like backend
- predictor entrypoint: `algorithm.src.inference.predictor.retrieve_sites`

## Minimal Smoke Test

After environment setup and artifact placement:

```bash
python -m algorithm.demo_retrieval \
  --experiment two_tower_v1 \
  --strategy low_rise_apartment \
  --query-text "I want a site for low-rise apartment redevelopment near a train station, with high development zoning, a large site, and limited planning constraints." \
  --top-k 5 \
  --recall-k 1000 \
  --locality WAITARA \
  --with-explanations
```

Expected behaviour:
- loads `two_tower_v1`
- uses `dcn_reranker_v1` by default
- filters to Waitara-related addresses
- dedupes to base-site addresses
- returns top-k results with explanations

If this works, backend deployment should also be able to call the same pipeline through `predictor.py`.