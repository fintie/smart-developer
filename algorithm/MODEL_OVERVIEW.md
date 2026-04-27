# Model Overview

This document summarises the current retrieval and scoring models used in the Smart Developer prototype.

The system is designed to retrieve property sites that match a given development strategy, then explain why those sites appear relevant.

## 1. Problem Setting

We formulate the task as **strategy-to-site retrieval**.

Given:
- a development strategy query, such as `low_rise_apartment` or `dual_occupancy`
- a large candidate set of property sites

the system retrieves and ranks sites that best match that strategy.

This is not a generic semantic search problem.  
Instead, it is a structured retrieval problem where the query encodes planning intent and the candidate encodes site characteristics.

## 2. Site Representation

The analytical entity is the **property polygon**.

Each site is represented using a feature bundle derived from open geospatial and transport data.

### Core feature groups

**Planning / feasibility**
- primary zoning code
- zoning band
- mixed zoning flag
- lot size proxy

**Constraint / risk**
- heritage flag and significance
- bushfire flag and risk level
- flood flag and flood class

**Accessibility**
- distance to station
- within-800m catchment flag
- station distance band

These structured fields are also converted into compact text representations for retrieval.

## 3. Heuristic Multi-Strategy Scoring

Before retrieval training, each site is scored for multiple strategies using an interpretable rule-based scorecard.

Strategies include:
- single dwelling rebuild
- assembly opportunity
- granny flat
- land bank / hold
- townhouse / multi-dwelling
- low-rise apartment
- dual occupancy

For each strategy, the score combines:

- feasibility signals
- opportunity signals
- constraint penalties

A simplified form is:

$$
\text{score} = \text{feasibility} + \text{opportunity} - \text{constraint penalty}
$$

This scorecard serves three purposes:
1. an interpretable baseline
2. a weak label generator
3. a reranking signal

## 4. Two-Tower Retrieval Model

The learned retrieval model is a **two-tower / bi-encoder architecture**.

One tower encodes the strategy query.  
The other tower encodes the candidate site text.

### Query tower input
A compact strategy-intent text, for example:

- strategy name
- zoning preference
- lot size preference
- access preference
- constraint preference

### Candidate tower input
A compact site text, for example:

- zoning code
- zoning band
- lot size band
- station distance band
- constraint severity
- heritage / flood / bushfire indicators

### Embedding
The model maps both query and candidate into a shared embedding space:

$$
q = f_{\theta_q}(x_q),\quad c = f_{\theta_c}(x_c)
$$

where:
- $x_q$ is the query text
- $x_c$ is the candidate text
- $q$ and $c$ are dense vectors

Similarity is computed using the dot product of normalised embeddings:
$$
s(q, c) = q^\top c
$$
Higher similarity means stronger retrieval relevance.

## 5. Training Variants

### Two-Tower V1
The first version is trained with positive query-candidate pairs and in-batch negatives.

It uses a multiple-negatives style retrieval objective and is designed to be a stable general retriever.

### Two-Tower V2
The second version introduces explicit hard negatives using triplet-style training.

A simplified triplet loss is:

$$
L = \max(0, m - s(q, c^+) + s(q, c^-))
$$

where:
- $c^+$ is a positive candidate
- $c^-$ is a negative candidate
- $m$ is the margin

This version is better at sharpening boundaries between similar strategies, but can be less balanced overall.

## 6. Hybrid Retrieval and Reranking

The current serving stack uses **hybrid retrieval**.

### Step 1: learned retrieval
The tuned two-tower model retrieves a candidate pool using embedding similarity.

### Step 2: fusion reranking
The retrieved candidates are reranked using both:
- retrieval similarity
- heuristic strategy score

A simplified fusion rule is:
$$
\text{final score} = \alpha \cdot \text{sim}_{norm} + \beta \cdot \text{score}_{norm}
$$
where:
- $\text{sim}_{norm}$ is normalized retrieval similarity
- $\text{score}_{norm}$ is normalised heuristic strategy score
- $\alpha$ and $\beta$ control the balance

This helps combine:
- the semantic flexibility of learned retrieval
- the interpretability of the heuristic scoring layer

## 7. Post-Processing

After reranking, the system applies lightweight serving logic such as:
- address-based deduplication
- compact result formatting
- optional explanation attachment

This improves demo quality and avoids repeated near-duplicate results from the same building or site.

## 8. Explanation Layer

The explanation layer is local and does not rely on external LLM APIs for core retrieval.

It works in three stages:
1. extract structured evidence from the site row
2. build a strategy-specific explanation payload
3. generate short natural-language rationale

The explanation typically covers:
- overall fit
- positive drivers
- main constraints

## 9. Current System Role

The current model stack should be viewed as a prototype retrieval system with weak supervision.

It is already useful for:
- internal demo
- qualitative product exploration
- strategy-specific site discovery
- further model iteration

It is not yet a final production ranking system.

## 10. Current Strengths

- interpretable property-level feature bundle
- strategy-specific site scoring
- local two-tower retrieval
- hybrid reranking
- local explanation generation
- modular architecture for future extension

## 11. Current Limitations

- supervision is derived from heuristic scoring rather than human-labelled relevance
- some strategies overlap strongly, making evaluation harder
- top-strategy match is not always the best metric for strategic or ambiguous cases
- explanation diversity is still limited when many sites share similar evidence
- more robust human evaluation is still needed

## 12. Next Steps

Likely next steps include:
- human-labelled retrieval evaluation
- strategy-specific routing or reranking
- improved hard negative mining
- better explanation diversity
- API serving and product integration