# Smart Developer Model Lifecycle

This document describes the current lightweight model lifecycle for Smart Developer.

The goal is not to build a heavy MLOps platform yet. The goal is to make model serving, logging, retraining, and promotion understandable for the internal team.

## 1. Current Production Models

Current default inference stack:

```text
query_text + strategy
-> query planner
-> two_tower_v1 retrieval
-> dcn_reranker_v1 reranking
-> dedupe
-> optional explanation/report
```

Current production models:

| Model | Role | Status |
|---|---|---|
| `two_tower_v1` | Retrieval backbone | production |
| `dcn_reranker_v1` | Second-stage reranker | production |

Model metadata is stored in the `model_registry` table.

Each production model should have:
- `model_version`
- `model_type`
- `artifact_path`
- `preprocessing_path` if needed
- `metrics`
- `status`
- `model_card`

## 2. Serving Flow

The algorithm service should run as a long-running FastAPI service.

Do not start a new Python process for every request.

Recommended serving flow:

```text
service startup
-> load predictor
-> warm up retrieval/reranker
-> mark service ready
-> handle user requests with warm predictor
```

The heavy model loading happens once at startup. Warm requests should reuse the loaded predictor instance.

## 3. Logging Flow

Every retrieval request can be logged into Postgres.

The current logging tables are:

| Table | Purpose |
|---|---|
| `retrieval_requests` | Stores user query, strategy, filters, latency, model versions |
| `retrieval_results` | Stores ranked sites returned for each request |
| `user_feedback` | Stores user actions such as click, save, dismiss, select |
| `model_registry` | Stores model versions, artifact paths, metrics, and status |

This gives us a basic feedback loop:

```text
user query
-> ranked results
-> user feedback
-> logged data
-> future training dataset
```

## 4. Feedback Dataset Export

Feedback data can be exported into a reranker training dataset with:

```bash
python -m algorithm.src.mlops.build_feedback_dataset
```

Useful variants:

```bash
# Full dataset: labelled + unlabelled rows
python -m algorithm.src.mlops.build_feedback_dataset

# Labelled-only dataset
python -m algorithm.src.mlops.build_feedback_dataset --labelled-only

# Weak-negative dataset
python -m algorithm.src.mlops.build_feedback_dataset --weak-negative-unfeedbacked
```

Current label mapping:

| Event | Label |
|---|---:|
| `click` | 0.5 |
| `save` | 1.0 |
| `select` | 1.0 |
| `manual_positive` | 1.0 |
| `dismiss` | 0.0 |
| `manual_negative` | 0.0 |
| shown but no feedback | unlabelled by default |

Shown-but-unfeedbacked results should not be treated as hard negatives too early.

## 5. Training Candidate Models

A new model should be trained as a candidate version first.

Example future naming:

```text
two_tower_v2
dcn_reranker_v2
```

Recommended process:

```text
export feedback dataset
-> train candidate model
-> evaluate candidate model
-> compare against production model
-> register candidate in model_registry
```

Candidate models should not automatically replace production models.

## 6. Evaluation

Before promotion, compare candidate model against current production model.

Useful metrics:
- top-k strategy match rate
- mean strategy score in top-k
- high-score rate in top-k
- latency
- qualitative sanity checks on known demo queries

For feedback-trained models, also track:
- click/save/select rate when enough real usage data exists
- performance by strategy
- performance by locality/filter

## 7. Promotion

A model can be promoted only after internal review.

Promotion means:
1. artifact is available locally or in shared storage
2. model card exists
3. metrics are recorded
4. `model_registry.status` is updated
5. service config/default model name is updated
6. service is restarted and warmup passes

Recommended statuses:

```text
candidate
staging
production
archived
```

Only one retrieval model and one reranking model should be treated as production by default.

## 8. Rollback

Rollback should be simple.

If a new model performs poorly:
1. change service config back to previous production model
2. restart service
3. run health check
4. run smoke test query
5. mark failed model as `archived` or `candidate`

Previous production artifacts should not be deleted immediately.

## 9. Current Practical Rule

For now, keep the lifecycle simple:

```text
train manually
evaluate manually
seed model_registry
promote manually
rollback manually
```

Automation can be added later after the product flow and feedback data become stable.
