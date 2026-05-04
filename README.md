# Smart Developer

Working internal ML pipeline for strategy-aware property site retrieval, reranking, report generation, and explanation.

This repo contains the current algorithm pipeline for Smart Developer. It is designed for internal use by the product lead and engineering team, and is intended to support backend integration, internal demo deployment, and future product iteration.

The current system supports:

- property-level geospatial feature building
- multi-strategy heuristic site scoring
- two-tower retrieval for intent-to-site matching
- DCN-based second-stage reranking
- query planning / query sanitisation
- locality / address text filtering
- base-site address deduplication
- local explanation generation
- markdown report generation
- terminal / notebook demo workflows
- backend-facing inference entrypoints

The current default inference stack is:

```text
strategy + query_text
-> optional query planner
-> two_tower_v1 recall
-> optional location filter
-> dcn_reranker_v1 rerank
-> base-site dedupe
-> optional explanation
-> return top-k
```

It is now a usable internal algorithm pipeline with deployable inference components.

---

## Repo structure

### Main working area

Most of the algorithm / retrieval logic is in:

- `algorithm/src/` — core code files
- `algorithm/configs/` — configuration and hyperparameters
- `algorithm/notebooks/` — EDA, experiments, and demos
- `algorithm/artifacts/` — trained models, evaluation outputs, reports, and model metadata
- `scripts/` — shell scripts for training and demo workflows

### Data

Raw and processed data are expected under:

- `data/raw/`
- `data/processed/`

Most backend / demo usage does **not** require rebuilding raw data.

---

## Environment setup

### 1. Clone the repo

```bash
git clone https://github.com/fintie/smart-developer.git
cd smart-developer
```

### 2. Create and activate a virtual environment

Use Python 3.10 if possible. Python 3.11 or 3.12 should also be fine if dependencies install correctly.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

If a package is missing during local development, install it directly as needed and update `requirements.txt` afterwards.

---

## Required local tooling

### Ollama

Ollama is required only if you want to run the **local explanation generation** workflow.

Install Ollama locally, then pull a local instruction model:

```bash
ollama pull llama3.1:8b-instruct-q4_K_M
```

If you use a different local model, update the explanation model name in the retrieval / report commands.

If explanation is disabled, Ollama is not required.

### Jupyter

Jupyter is required only for notebooks / demo notebooks.

```bash
pip install notebook jupyterlab
```

Run with:

```bash
jupyter lab
```

or:

```bash
jupyter notebook
```

---

## Data setup

### Important

The large processed parquet files are **not expected to be rebuilt from scratch** every time.

A Google Drive folder has been shared in the project WhatsApp group chat. Download the processed parquet files and place them into the matching repo paths.

### Base processed data required for rebuilding the pipeline

Place files under:

```text
data/processed/
├── nsw_addressing/
│   └── addresspoint_all.parquet
├── nsw_bushfire/
│   └── bushfire.parquet
├── nsw_flood/
│   └── flood.parquet
├── nsw_heritage/
│   └── heritage.parquet
├── nsw_property/
│   └── property.parquet
├── nsw_zoning/
│   └── land_zoning.parquet
├── retrieval/
├── site_features/
└── transport/
    └── rail_metro_stations_raw.parquet
```

Once these files are available, start from:

```bash
python -m algorithm.src.features.build_features
```

---

## Minimal runtime artifacts for demo / backend inference

If you do **not** need to retrain models, the following files are required to run the current demo / backend inference pipeline:

### Candidate and query data

- `data/processed/retrieval/candidate_sites.parquet`
- `data/processed/retrieval/query_intents.jsonl`

### Two-tower retrieval model

- `algorithm/artifacts/models/two_tower_v1/model.pt`

### DCN reranker

- `algorithm/artifacts/models/dcn_reranker_v1/model.pt`
- `algorithm/artifacts/models/dcn_reranker_v1/preprocessing.json`

### Optional for explanation

- local Ollama model, e.g. `llama3.1:8b-instruct-q4_K_M`

For product lead / PM / demo-only users, these runtime artifacts are usually enough. They do not need raw geospatial data or intermediate training files.

---

## Core pipeline

### 1. Build site features

```bash
python -m algorithm.src.features.build_features
```

This builds the property-level site feature bundle from processed geospatial layers.

Typical outputs go to:

```text
data/processed/site_features/
```

### 2. Score sites by strategy

```bash
python -m algorithm.src.scoring.scoring
```

This applies the current heuristic multi-strategy scorecard and produces strategy scores such as:

- `single_dwelling_rebuild_score`
- `assembly_opportunity_score`
- `granny_flat_score`
- `land_bank_hold_score`
- `townhouse_multi_dwelling_score`
- `low_rise_apartment_score`
- `dual_occupancy_score`

### 3. Build retrieval candidate table

```bash
python -m algorithm.src.retrieval.build_candidate_sites
```

This creates:

```text
data/processed/retrieval/candidate_sites.parquet
```

This table is used by:

- embedding baseline evaluation
- two-tower retrieval
- hybrid retrieval demo
- backend inference
- report generation

### 4. Build weakly supervised training pairs

```bash
python -m algorithm.src.retrieval.build_training_pairs
```

This creates:

```text
data/processed/retrieval/training_pairs.parquet
```

These pairs are derived from the heuristic strategy scoring layer.

### 5. Train two-tower retrieval model

Model configs and hyperparameters are in:

```text
algorithm/configs/model.yaml
```

Experiments are selected through `--experiment`.

Train the current default retrieval model:

```bash
python -m algorithm.src.models.train_two_tower_v1 --experiment two_tower_v1
```

Artifacts are saved under:

```text
algorithm/artifacts/models/two_tower_v1/
```

### 6. Evaluate two-tower retrieval model

```bash
python -m algorithm.src.models.evaluate_two_tower --experiment two_tower_v1
```

Evaluation outputs are saved under the corresponding model artifact folder.

### 7. Build DCN reranker dataset

The DCN reranker uses first-stage retrieval outputs to build a pair-level reranking dataset.

```bash
python -m algorithm.src.retrieval.build_reranker_dataset \
  --experiment dcn_reranker_v1 \
  --recall-experiment two_tower_v1
```

This creates:

```text
algorithm/artifacts/reranker/reranker_train.parquet
algorithm/artifacts/reranker/reranker_eval.parquet
```

### 8. Train DCN reranker

```bash
python -m algorithm.src.models.train_dcn_reranker --experiment dcn_reranker_v1
```

Artifacts are saved under:

```text
algorithm/artifacts/models/dcn_reranker_v1/
```

### 9. Evaluate DCN reranker

```bash
python -m algorithm.src.models.evaluate_dcn_reranker \
  --experiment dcn_reranker_v1 \
  --recall-experiment two_tower_v1
```

---

## Training shortcut

Run the automation script for the main training pipeline:

```bash
chmod +x ./scripts/run_training.sh
./scripts/run_training.sh
```

To skip expensive rebuild steps, use environment flags. Example: rerun only the DCN section after two-tower artifacts already exist:

```bash
RUN_FEATURES=false \
RUN_SCORING=false \
RUN_CANDIDATES=false \
RUN_TRAINING_PAIRS=false \
RUN_TWO_TOWER_TRAIN=false \
RUN_TWO_TOWER_EVAL=false \
./scripts/run_training.sh
```

---

## Running retrieval demo

### Basic demo

```bash
python -m algorithm.demo_retrieval \
  --experiment two_tower_v1 \
  --strategy single_dwelling_rebuild \
  --query-text "I want a site for detached house redevelopment on standard residential land, with low planning constraints and a suitable lot size." \
  --top-k 5
```

By default, the current demo path uses:

- `two_tower_v1` for candidate recall
- `dcn_reranker_v1` for second-stage reranking
- base-site dedupe

### With explanations

```bash
python -m algorithm.demo_retrieval \
  --experiment two_tower_v1 \
  --strategy low_rise_apartment \
  --query-text "I want a site for low-rise apartment redevelopment near a train station, with high development zoning, a large site, and limited planning constraints." \
  --top-k 5 \
  --with-explanations
```

### With query planner

The query planner detects strategy/query conflicts and rewrites the query into a more strategy-aware form.

```bash
python -m algorithm.demo_retrieval \
  --experiment two_tower_v1 \
  --strategy single_dwelling_rebuild \
  --query-text "I want a site for house redevelopment near a train station, with high development zoning and a large site." \
  --top-k 5 \
  --use-query-planner
```

### With location filter

Current location filtering is address-text based, because the current `candidate_sites.parquet` does not yet contain structured suburb / postcode columns.

```bash
python -m algorithm.demo_retrieval \
  --experiment two_tower_v1 \
  --strategy low_rise_apartment \
  --query-text "I want a site for low-rise apartment redevelopment near a train station, with high development zoning, a large site, and limited planning constraints." \
  --top-k 5 \
  --recall-k 1000 \
  --locality WAITARA
```

### Shell script demo

Edit variables inside:

```text
scripts/run_demo_retrieval.sh
```

Then run:

```bash
chmod +x ./scripts/run_demo_retrieval.sh
./scripts/run_demo_retrieval.sh
```

---

## Report generation demo

The report layer converts retrieval outputs into a markdown-style developer / investor brief.

Example:

```bash
python -m algorithm.demo_report \
  --strategy low_rise_apartment \
  --query-text "I want a site for low-rise apartment redevelopment near a train station, with high development zoning, a large site, and limited planning constraints." \
  --top-k 5 \
  --recall-k 1000 \
  --locality WAITARA \
  --output algorithm/artifacts/reports/waitara_low_rise_apartment.md
```

Or edit variables in `scripts/run_demo_report.sh` and run:
```bash
chmod +x ./scripts/run_demo_report.sh
./scripts/run_demo_report.sh
```

The report includes:

- executive summary
- shortlisted base sites
- site-level rationale
- risks / checks
- suggested next checks
- formal note that this is a screening tool, not planning advice

---

## Notebook demos

The main demo notebook is:

```text
algorithm/notebooks/demo_retrieval.ipynb
```

Other notebooks document:

- scoring sanity checks
- local explanation tests
- candidate table construction
- embedding baseline experiments
- clean baseline evaluation
- two-tower vs baseline comparisons
- hybrid retrieval evaluation

---

## Local explanation generation

Explanation generation uses the local pipeline in:

```text
algorithm/src/explanation/
```

It is designed to avoid reliance on external hosted LLM APIs.

You need a local model available via Ollama:

```bash
ollama pull llama3.1:8b-instruct-q4_K_M
```

If explanations fail, check:

- Ollama is running
- the model name matches the configured / requested model
- local endpoint is available

If you do not need explanations, run demo / backend calls with explanation disabled.

---

## Query planner

The lightweight query planner is in:

```text
algorithm/src/agent/query_planner.py
```

It currently supports:

- strategy inference from query text
- strategy/query conflict warnings
- strategy-aware query rewriting
- sanitised user query generation
- suggested alternative strategies

Example use case:

```text
Input strategy: single_dwelling_rebuild
Input query: house redevelopment near a train station with high development zoning

Planner output:
- keep selected strategy as single_dwelling_rebuild
- sanitise high-intensity terms
- warn that the query also matches low_rise_apartment
- suggest alternatives such as low_rise_apartment and land_bank_hold
```

---

## Location filtering and base-site dedupe

Current location filtering is simple and product-oriented:

- `--locality WAITARA` filters by address text
- `--address-contains` filters by address substring

Because the current candidate table does not yet contain structured `locality`, `postcode`, or `LGA` fields, this is a first-pass filter. Future versions should add structured location fields to `candidate_sites.parquet`.

Base-site dedupe normalises unit-level addresses into site-level addresses. Example:

```text
623/21-37 WAITARA AVENUE WAITARA
-> 21-37 WAITARA AVENUE WAITARA
```

This makes the output more suitable for development site recommendation rather than individual unit recommendation.

---

## Configuration files

### Strategy scoring

```text
algorithm/configs/strategies.yaml
```

Defines first-pass strategy scoring logic. This can be updated as the product / domain logic evolves.

### Model training / retrieval

```text
algorithm/configs/model.yaml
```

Defines:

- defaults
- experiment-specific overrides
- artifact paths
- model hyperparameters
- training settings
- reranker settings

---

## Internal model docs

Useful internal docs:

- `algorithm/MODEL_OVERVIEW.md`
- `algorithm/DEPLOYMENT_HANDOFF.md`
- `algorithm/strategy_feature_mapping.md`
- `algorithm/two_tower_data_design.md`

These explain the retrieval / scoring / deployment design at a higher level.

---

## Frontend / backend integration

If you do not need to retrain models, the main inference entrypoint is:

```text
algorithm/src/inference/predictor.py
```

Recommended function:

```python
from algorithm.src.inference.predictor import retrieve_sites
```

Example:

```python
response = retrieve_sites(
    strategy="low_rise_apartment",
    query_text="I want a site for low-rise apartment redevelopment near a train station, with high development zoning, a large site, and limited planning constraints.",
    top_k=5,
    recall_k=1000,
    with_explanations=True,
    locality="WAITARA",
)
```

The response is a backend-friendly Python dict containing:

- request metadata
- ranked site results
- base-site address
- original source address
- retrieval / reranking scores
- strategy score
- explanation text if enabled

Core serving components:

- `algorithm/src/inference/predictor.py`
- `algorithm/src/retrieval/hybrid_retrieve.py`
- `algorithm/src/explanation/`
- `algorithm/src/agent/query_planner.py`

### Required inference artifacts

- `data/processed/retrieval/candidate_sites.parquet`
- `data/processed/retrieval/query_intents.jsonl`
- `algorithm/artifacts/models/two_tower_v1/model.pt`
- `algorithm/artifacts/models/dcn_reranker_v1/model.pt`
- `algorithm/artifacts/models/dcn_reranker_v1/preprocessing.json`

### Suggested backend behaviour

- load predictor once at service start
- keep model and candidate table in memory
- avoid reloading parquet/model per request
- make explanations optional or async if latency matters
- log request, returned sites, user clicks/saves/dismissals for future reranker supervision

---

## Current default recommendation for internal demo

For internal demo, use:

- retrieval backbone: `two_tower_v1`
- reranker: `dcn_reranker_v1`
- query planner: optional, useful for ambiguous user queries
- location filter: optional, useful for suburb-level demo
- dedupe: enabled
- explanations: enabled if Ollama is available
- report generation: use `algorithm.demo_report`

Recommended demo command:

```bash
python -m algorithm.demo_report \
  --strategy low_rise_apartment \
  --query-text "I want a site for low-rise apartment redevelopment near a train station, with high development zoning, a large site, and limited planning constraints." \
  --top-k 5 \
  --recall-k 1000 \
  --locality WAITARA \
  --output algorithm/artifacts/reports/waitara_low_rise_apartment.md
```

---

## Current limitations

- Supervision is weakly derived from heuristic strategy scores, not human-labelled relevance.
- Some strategies overlap heavily, so top-strategy match is not always a perfect metric.
- Location filtering is currently text-based because structured suburb / postcode fields are not yet in the candidate table.
- Explanation quality is useful for demo, but can be repetitive when sites have similar evidence.
- This is a screening and prioritisation system, not formal planning advice.

---

## Likely next steps

- Add structured location fields to `candidate_sites.parquet`.
- Add backend request/result/feedback logging.
- Use human feedback to improve reranker supervision.
- Add strategy comparison reports.
- Improve report formatting for frontend / PDF export.
- Add more robust product filters such as LGA, postcode, lot-size threshold, and constraint exclusions.