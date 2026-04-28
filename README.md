# Smart Developer

Working internal ML pipeline for strategy-aware property site retrieval, reranking, and explanation.

This repo contains the current algorithm pipeline for Smart Developer.  
It is designed for internal use by the product lead and engineering team, and is intended to support backend integration and internal demo deployment.

The current system supports:
* property-level geospatial feature building
* multi-strategy heuristic site scoring
* two-tower retrieval for intent-to-site matching
* DCN-based reranking
* local explanation generation
* terminal / notebook demo workflows
* backend-facing inference entrypoints

The current default inference stack is:

```text
strategy + query_text
-> two_tower_v1 recall
-> dcn_reranker_v1 rerank
-> dedupe
-> optional explanation
-> return top-k
```

It is now a usable internal algorithm pipeline with deployable inference components.

## Repo structure
### Main working area
Most of the algorithm/retrieval logic are in:
* `algorithm/src/` (core code files)
* `algorithm/configs` (configuration and hyperparameters)
* `algorithm/notebooks/` (EDA and experimentation)
* `algorithm/artifacts/` (model performances)

### Data
Processed and raw data in:
* `data/raw/`
* `data/processed/`

## Environment setup for running algorithm pipeline
### 1. Clone the repo
```bash
git clone https://github.com/fintie/smart-developer.git
cd smart-developer
```

### 2. Create and activate a virtual environment
Make sure installed Python already (ideally 3.10 version, but 3.11 or 3.12 should also be fine).
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```
If some packages are missing during local development, install them directly as needed.

## Required local tooling
### Ollama
Required only if you want to run the **local explanation generation** workflow.

Install Ollama locally, then pull a local instruction model, e.g.
```bash
ollama pull llama3.1:8b-instruct-q4_K_M
```
If you use a different local model, update the explanation model name in the retrieval / explanation commands.

### Jupyter
Required for notebooks / demo notebooks.

If needed:
```bash
pip install notebook jupyterlab
```

## Data setup
### IMPORTANT
The large processed parquet files are **not expected to be rebuilt from scratch** every time.

I've shared a Google Drive folder in the project WhatsApp group chat.

### What to do
Download the parquet files from the **shared Google Drive** and put them into the matching folders accordingly:
* `data/processed/nsw_addressing/`
* `data/processed/nsw_bushfire/`
* `data/processed/nsw_flood/`
* `data/processed/nsw_heritage/`
* `data/processed/nsw_property/`
* `data/processed/nsw_zoning/`
* `data/processed/transport/`

```text
data/processed/
├── nsw_addressing/
│   └── addresspoint_all.parquet
├── nsw_bushfire/
│   └── bushfire.parquet
├── nsw_flood/
│   └── flood.parquet
├── nsw_heritage/
│   └── heritage.parquet
├── nsw_property/
│   └── property.parquet
├── nsw_zoning/
│   └── land_zoning.parquet
├── retrieval/
├── site_features/
└── transport/
    └── rail_metro_stations_raw.parquet
```

Once these are done, start from:
```bash
python -m algorithm.src.features.build_features
```

## Core pipeline
### 1. Build site features
Entry point:
```bash
python -m algorithm.src.features.build_features
```
This builds the property-level site feature bundle from processed geospatial layers.

Typical outputs go to `data/processed/site_features/`.

### 2. Score sites by strategy
Run:
```bash
python -m algorithm.src.scoring.scoring
```
This applies the current heuristic multi-strategy scorecard and produces strategy scores such as:
* `single_dwelling_rebuild_score`
* `granny_flat_score`
* `dual_occupancy_score`
* etc.

### 3. Build retrieval candidate table
Run:
```bash
python -m algorithm.src.retrieval.build_candidate_sites
```
This creates: `data/processed/retrieval/candidate_sites.parquet`

This table is used by:
* embedding baseline evaluation
* two-tower retrieval 
* hybrid retrieval demo

### 4. Build weakly supervised training pairs
Run:
```bash
python -m algorithm.src.retrieval.build_training_pairs
```
This creates: `data/processed/retrieval/training_pairs.parquet`

These pairs are derived from the heuristic strategy scoring layer.

### 5. Train retrieval models
Model configs and hyperparameters are in `algorithm/configs/model.yaml`

Experiments are selected through argparse `--experiment`

Train `two_tower_v1` (best model):
```bash
python -m algorithm.src.models.train_two_tower_v1 --experiment two_tower_v1
```

Artifacts are saved under: `algorithm/artifacts/models/two_tower_v1/`

### 6. Evaluate retrieval models
Evaluate v1
```bash
python -m algorithm.src.models.evaluate_two_tower --experiment two_tower_v1
```

Evaluation outputs are saved under the corresponding model artifact folders.

## Shortcut
Run the automation script for the above pipeline:
```bash
chmod +x ./scripts/run_training.sh
./scripts/run_training.sh
```

## Running the demo
Run directly, e.g.
```bash
python -m algorithm.demo_retrieval \
  --experiment two_tower_v1 \
  --strategy single_dwelling_rebuild \
  --query-text "I want a site for detached house redevelopment on standard residential land, with low planning constraints and a suitable lot size." \
  --top-k 5 \
  --with-explanations
```
OR edit variables inside the shell script `scripts/run_demo_retrieval.sh` and run:
```bash
chmod +x ./scripts/run_demo_retrieval.sh
./scripts/run_demo_retrieval.sh
```

## Notebook demos
The main demo notebook is: `algorithm/notebooks/demo_retrieval.ipynb`

Other notebooks document:
* scoring sanity checks
* embedding baseline experiments
* two-tower vs baseline comparisons
* hybrid retrieval evaluation
* local explanation tests

To run notebooks:
```bash
jupyter lab
```
or
```bash
jupyter notebook
```

## Local explanation generation
Explanation generation uses the local pipeline in: `algorithm/src/explanation/`

This is designed to avoid reliance on external hosted LLM APIs, which avoids complicated accessibility authentications.

You need a local model available via Ollama. E.g.
```bash
ollama pull llama3.1:8b-instruct-q4_K_M
```

If explanations fail, check:
* Ollama is running
* the model name matches the configured / requested model
* local endpoint is available

## Current default recommendation for demo

For internal demo, the current recommended setup is:
* retrieval backbone: `two_tower_v1`
* reranker: `dcn_reranker_v1`
* dedupe: enabled
* explanations: enabled

In practice, the easiest command is:

```bash
python -m algorithm.demo_retrieval --experiment two_tower_v1 --with-explanations
```

By default, the demo path uses:
* `two_tower_v1` or candidate recall
* `dcn_reranker_v1` for second-stage reranking

## Configuration files
### Strategy scoring
`algorithm/configs/strategies.yaml` defines first-pass strategy scoring logic (we can change as time goes). 

### Model training / retrieval
`algorithm/configs/model.yaml` defines:
* defaults
* experiment-specific overrides
* artifact paths

## Internal model docs
Useful internal docs:
* `algorithm/MODEL_OVERVIEW.md`
* `algorithm/strategy_feature_mapping.md`
* `algorithm/two_tower_data_design.md`

These explain the retrieval / scoring design at high level.

## For frontend/backend integration

If you do not need to retrain models, the main inference entrypoint is:

* `algorithm/src/inference/predictor.py`

Recommended function to call:

```python
from algorithm.src.inference.predictor import retrieve_sites
```

This returns a backend-friendly Python dict containing:
* request metadata
* ranked site results
* retrieval / reranking scores
* explanation text if enabled

Core serving components:
* `algorithm/src/inference/predictor.py`
* `algorithm/src/retrieval/hybrid_retrieve.py`
* `algorithm/src/explanation/`

Required inference artifacts:
* `data/processed/retrieval/candidate_sites.parquet`
* `algorithm/artifacts/models/two_tower_v1/model.pt`
* `algorithm/artifacts/models/dcn_reranker_v1/model.pt`
* `algorithm/artifacts/models/dcn_reranker_v1/preprocessing.json`