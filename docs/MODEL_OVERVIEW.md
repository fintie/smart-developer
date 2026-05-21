# Model Overview

This document summarises the current retrieval, reranking, and explanation models used in the Smart Developer prototype.

The system is designed to retrieve property sites that match a given development strategy, then explain why those sites appear relevant.

## 1. Problem Setting

We formulate the task as **strategy-to-site retrieval and reranking**.

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

These structured fields are also converted into compact text representations for retrieval and pairwise reranking.

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

This scorecard serves four purposes:
1. an interpretable baseline
2. a weak label generator
3. a retrieval supervision source
4. a reranking signal

## 4. Two-Tower Retrieval Model

The first-stage learned retrieval model is a **two-tower / bi-encoder architecture**.

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
\mathbf{q} = f_{\theta_q}(x_q),\quad \mathbf{c} = f_{\theta_c}(x_c)
$$

where:
- $x_q$ is the query text
- $x_c$ is the candidate text
- $\mathbf{q}$ and $\mathbf{c}$ are dense vectors

Similarity is computed using the inner product of normalised embeddings:

$$
s(\mathbf{q}, \mathbf{c}) = \langle\mathbf{q},\mathbf{c}\rangle
$$

Higher similarity means stronger retrieval relevance using the fact that

$$
\langle\mathbf{q},\mathbf{c}\rangle=\lVert\mathbf{q}\rVert_2\lVert\mathbf{c}\rVert_2\cos{\theta},
$$

where $\theta$ is the angle between vector $\mathbf{q}$ and $\mathbf{c}$.

## 5. Two-Tower Training Variants

### Two-Tower V1
The first version is trained with positive query-candidate pairs and in-batch negatives.

It uses a multiple-negatives style retrieval objective and is designed to be a stable general retriever.

### Two-Tower V2
The second version introduces explicit hard negatives using triplet-style training.

A simplified triplet loss is:

$$
L = \max(0, m - s(\mathbf{q}, \mathbf{c}^+) + s(\mathbf{q}, \mathbf{c}^-))
$$

where:
- $\mathbf{c}^+$ is a positive candidate
- $\mathbf{c}^-$ is a negative candidate
- $m$ is the margin

This version is better at sharpening boundaries between similar strategies, but can be less balanced overall.

## 6. DCN Reranker

The second-stage reranker is a **Deep & Cross Network (DCN)**.

Unlike the two-tower model, which is designed for efficient candidate recall, the DCN reranker scores each retrieved **query-candidate pair** directly using structured features.

### Why DCN
This problem contains many important feature interactions, for example:
- zoning × lot size
- zoning × transport access
- strategy × constraint pattern
- strategy × site intensity
- retrieval similarity × heuristic strategy score

These are exactly the kinds of interactions that a cross network is designed to model.

### Input to the reranker
The DCN reranker consumes pair-level features such as:
- retrieval similarity
- strategy-specific heuristic score
- lot size proxy
- zoning code / zoning band
- lot size band
- station distance band
- constraint severity band
- mixed zoning flag
- heritage flag
- flood flag
- bushfire risk level
- within-800m catchment flag
- top strategy score

### Cross Network intuition
A simplified cross layer is:

$$
\mathbf{x}_{l+1} = \mathbf{x}_0 (\mathbf{w}_l^\top \mathbf{x}_l) + \mathbf{b}_l + \mathbf{x}_l
$$

where:
- $\mathbf{x}_0$ is the original input
- $\mathbf{x}_l$ is the current layer representation

This allows the model to explicitly learn useful feature crosses rather than relying only on implicit MLP interactions.

### Output
The reranker outputs a pairwise relevance logit:

$$
r(\mathbf{q}, \mathbf{c}) = g_{\phi}(\mathbf{z}_{q,c})
$$

where:
- $\mathbf{z}_{q,c}$ is the structured feature vector for the query-candidate pair
- $r(\mathbf{q},\mathbf{c})$ is the reranking score

The reranker is trained as a binary relevance model using weak supervision derived from strategy scores.

## 7. Retrieval and Reranking Stack

The current serving stack is:

### Step 1: first-stage recall
The tuned two-tower model retrieves a candidate pool using embedding similarity.

### Step 2: heuristic fusion baseline
A simple fallback / baseline reranking combines:
- retrieval similarity
- heuristic strategy score

A simplified fusion rule is:

$$
\text{fusion score} = \alpha \cdot \text{sim}_{norm} + \beta \cdot \text{score}_{norm}
$$

where:
- $\text{sim}_{norm}$ is normalized retrieval similarity
- $\text{score}_{norm}$ is normalised heuristic strategy score
- $\alpha$ and $\beta$ control the balance

### Step 3: learned second-stage reranking
The current preferred reranking layer is the DCN reranker, which reorders the recalled candidates using structured pair features.

So the recommended inference path is:

```text
strategy + query_text
-> two_tower_v1 recall
-> dcn_reranker_v1 rerank
-> dedupe
-> optional explanation
-> top-k results
```

## 8. Post-Processing

After reranking, the system applies lightweight serving logic such as:
- address-based deduplication
- compact result formatting
- optional explanation attachment

This improves demo quality and avoids repeated near-duplicate results from the same building or site.

## 9. Explanation Layer

The explanation layer is local and does not rely on external LLM APIs for core retrieval.

It works in three stages:
1. extract structured evidence from the site row
2. build a strategy-specific explanation payload
3. generate short natural-language rationale

The explanation typically covers:
- overall fit
- positive drivers
- main constraints

## 10. Current System Role

The current model stack should be viewed as a prototype retrieval system with weak supervision.

It is already useful for:
- internal demo
- qualitative product exploration
- strategy-specific site discovery
- further model iteration

It is not yet a final production ranking system.

## 11. Current Strengths

- interpretable property-level feature bundle
- strategy-specific site scoring
- local two-tower retrieval
- hybrid reranking
- local explanation generation
- modular architecture for future extension

## 12. Current Limitations

- supervision is derived from heuristic scoring rather than human-labelled relevance
- some strategies overlap strongly, making evaluation harder
- top-strategy match is not always the best metric for strategic or ambiguous cases
- explanation diversity is still limited when many sites share similar evidence
- more robust human evaluation is still needed