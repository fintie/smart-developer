# Two-Tower Data Design

This document defines the first version of the two-tower retrieval setup for the Smart Developer project.

The goal is not traditional recommendation.  
Instead, the task is to retrieve property sites that best match a development intent or strategy.

---

## 1. Retrieval Task Definition

We formulate the problem as:

**intent-to-site retrieval**

- **Query side**: development intent, strategy intent, or user goal
- **Candidate side**: property site representation

Examples of query intent:
- low-rise apartment opportunity near station
- granny flat potential on residential land
- dual occupancy candidate with limited constraints
- land bank / hold site with future redevelopment upside
- townhouse / multi-dwelling potential

The two-tower model should learn a shared embedding space where:
- semantically relevant site candidates are close to the query
- irrelevant or weak-fit sites are far away

---

## 2. Analytical Entity

The main analytical entity is:

**property polygon**

Reason:
- it better represents the site as a development unit
- it supports site geometry and lot-size-related features
- planning/risk overlays are naturally attached at property/site level

The address layer remains important, but mainly as:
- input resolution layer
- geocoding layer
- lookup layer

---

## 3. Candidate Representation

Each site candidate should be represented using a compact retrieval-ready view.

### 3.1 Structured features

Core structured features:
- primary_zoning_code
- primary_zoning_class
- mixed_zoning_flag
- lot_size_proxy_sqm
- heritage_flag
- heritage_max_significance
- bushfire_flag
- bushfire_risk_level
- flood_flag
- primary_flood_class
- distance_to_station_m
- within_800m_catchment
- station_access_score

### 3.2 Derived categorical abstractions

For retrieval, some raw fields should be converted into abstract bands:
- zoning band
- lot size band
- station distance band
- constraint severity band

This reduces sparsity and improves semantic matching.

### 3.3 Text view

Each site should also have a compact textual representation.

Example:
- "MU1 zoning, large lot, within 800m of rail, no flood, low bushfire risk, no heritage constraint"
- "R2 zoning, medium lot, no strong transport access, suitable for low-intensity residential redevelopment"

This text view can be used:
- as direct candidate tower input
- as a baseline embedding representation
- as a weakly supervised explanation-aware site description

---

## 4. Query Representation

The query side should support both:

### 4.1 Structured intent queries
Examples:
- strategy=low_rise_apartment; near_station=yes; avoid_constraints=yes
- strategy=granny_flat; low_density=yes
- strategy=land_bank_hold; future_upside=yes

### 4.2 Natural language intent queries
Examples:
- site suited for low-rise apartment redevelopment near station
- residential site with granny flat potential and low planning risk
- strategic land bank opportunity with future uplift potential

The first version can start with templated structured-to-text queries, then gradually expand to freer natural-language forms.

---

## 5. Candidate Schema (v1)

Each candidate row should contain:

### Identity
- RID
- address
- geometry

### Planning / feasibility
- primary_zoning_code
- primary_zoning_class
- mixed_zoning_flag
- zoning_code_count

### Site capacity
- lot_size_proxy_sqm
- lot_size_band

### Constraints
- heritage_flag
- heritage_max_significance
- bushfire_flag
- bushfire_risk_level
- flood_flag
- primary_flood_class

### Accessibility
- distance_to_station_m
- within_800m_catchment
- station_access_score
- station_distance_band

### Scoring outputs
- single_dwelling_rebuild_score
- assembly_opportunity_score
- granny_flat_score
- land_bank_hold_score
- townhouse_multi_dwelling_score
- low_rise_apartment_score
- dual_occupancy_score

### Text fields
- site_summary_text
- optional strategy-specific explanation text

---

## 6. Query Schema (v1)

Each query should contain:

- query_id
- strategy
- text
- optional structured fields:
  - prefer_near_station
  - avoid_constraints
  - prefer_low_density
  - prefer_high_density
  - prefer_large_site
  - long_term_hold

This lets us support both:
- direct retrieval experiments
- strategy-aware retrieval training

---

## 7. Weak Supervision for Training Pairs

At this stage, there is no real user feedback or click data.

So the first version of supervision should come from the scoring layer.

### Positive pairs
A site can be treated as a positive for a strategy query if:
- its strategy score is above a threshold, e.g. >= 70
- or it is in the top percentile for that strategy

### Negative pairs
A site can be treated as a negative if:
- its strategy score is very low, e.g. <= 20
- or it is a hard negative with superficially similar features but poor strategic fit

### Hard negatives
Hard negatives are especially valuable when:
- zoning looks partially compatible but constraints are heavy
- lot size is attractive but accessibility is poor
- accessibility is strong but zoning is weak
- the site fits another strategy much better than the current one

---

## 8. Training Data Format

The preferred first-pass format is either:

### Pair format
- query_text
- candidate_text
- label

### Triplet format
- query_text
- positive_candidate_text
- negative_candidate_text

Triplet format is better for retrieval training and ranking quality.

---

## 9. Tower Input Design

### Query tower
Input:
- strategy intent text
- structured query text
- optional natural language query

### Candidate tower
Input:
- compact site summary text
- selected structured tags converted into text
- optional strategy-aware site summary

The first version should stay compact and semantically clear, instead of feeding every raw field directly.

---

## 10. Relationship to the Current Scoring System

The current scorecard system serves three roles:
- baseline ranking layer
- weak label generator
- interpretability anchor

The two-tower system does not replace scoring immediately.

Instead:
1. scoring defines weak supervision
2. two-tower learns retrieval semantics
3. a later reranker or scoring layer can refine final ordering