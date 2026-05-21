# Policy RAG Layer
## Purpose
The policy layer is one of the key product differentiators. Instead of only saying that a site is suitable because it has good zoning or a large lot, we attempt to explain **why current planning policy may support or weaken a redevelopment opportunity**.

The policy layer has two parts:
```text
Structured policy scoring
+
Policy RAG evidence retrieval
```

The structured scorer gives a numerical policy signal, while the RAG component retrieves supporting snippets from official NSW Planning sources.

## Policy Scoring
The structured policy scorer is rule-based for the current MVP. It checks whether a property matches policy-relevant conditions such as:
- zoning code
- lot size band
- station distance band
- constraint severity
- selected development strategy

Example policy-relevant strategies include:
- dual occupancy
- townhouse / multi-dwelling
- low-rise apartment
- assembly opportunity
- land bank /hold

For each site-strategy pair, the scorer returns:
```text
policy_upside_score
policy_signal_band
policy_matched_rules
policy_matched_policies
policy_matched_policy_names
policy_explanation
```

The policy score is not intended to be a planning approval prediction. It is a screening signal that helps identify which sites deserve closer review.

## Policy RAG Evidence
The RAG layer indexes official planning policy pages and retrieves relevant evidence snippets for each matched policy.

Current indexed policy areas include:
- NSW Low and Mid-Rise Housing policy
- Transport Oriented Development policy
- TOD planning controls
- Housing SEPP
- In-fill affordable housing provisions

The workflow is:
```text
Official policy source pages
        ↓
Text extraction and cleaning
        ↓
Chunking
        ↓
Embedding with sentence-transformer
        ↓
Chroma vector index
        ↓
Policy-specific retrieval at inference time
```

At inference time, once a site matches structured policy rules, the system queries the policy index using the matched policy IDs, selected strategy, and site context. The returned evidence is attached to the result as:
```text
policy_evidence
policy_evidence_count
```

The agent-facing pitch can then say, for example, that a high policy signal is supported by retrieved NSW Planning evidence snippets from official sources.

## Why combine rules and RAG?
Pure RAG is flexible but can be hard to control and explainability can be low. Pure rules are stable but may not provide enough evidence or context. The hybrid design gives us both.

Structured policy rules:
- deterministic
- easy to debug
- safe for scoring
- useful for ranking

Policy RAG:
- gives supporting evidence
- improves explanation quality
- helps agents understand why a policy signal exists
- allows policy documentation to be updated without retraining the ranking model

This also reduces over-reliance on an LLM. The scoring logic remains deterministic, while retrieval provides supporting context.

## Important Limitation
The current policy layer is an opportunity screening system, not a legal or planning compliance engine.

The output should be interpreted as *"This site may have policy-supported redevelopment upside and should be reviewed further."* Instead of *"This site will definitely be approved."*

The report should always include a disclaimer that final decisions require planning, legal, valuation, and feasibility review.