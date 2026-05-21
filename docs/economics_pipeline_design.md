# Economics Pipeline
## Purpose
The economics pipeline estimates whether a property opportunity is economically attractive after considering:
- transaction-level market value
- local market trend
- development cost
- construction cost escalation
- value potential
- cost risk
- cost efficiency

The goal is not to produce a formal valuation. Instead, the economics layer provides an **indicative feasibility screen** that helps agents and developer clients compare opportunities.

## High-Level Structure
The current economics pipeline has four main components:

1. **Market value model:**
   XGBoost transaction-level market value estimator

2. **Market trend model:**
   Rolling-window Ridge local market momentum model

3. **Development cost estimator:**
   Rule/config-based strategy-specific cost model

4. **Construction cost trend model:**
   ABS WPI/PPI-based cost escalation signal

These are combined inside the `EconomicsPipeline`, which returns final fields used by ranking, API serving, and report generation.

## Market Value Model
The market value model estimates a typical transaction-level market value for a property's locality context.

The current model is `xgb_market_value_v1`. It is trained on NSW PSI sales data.

Input features include:
- suburb
- postcode
- property class
- contract year / month / quarter
- recent suburb sales count
- recent suburb median price
- recent suburb mean price
- recent suburb price percentiles
- recent postcode sales count
- recent postcode median price

Output:
```text
ml_estimated_market_value
ml_value_lower_bound
ml_value_upper_bound
ml_value_confidence
ml_value_error_pct
```
The model is used as a **transaction-level value proxy**, not a formal valuation.

### Mathematical View
For a site $i$, define its tabular feature vector as $\mathbf{x}_i\in\mathbb{R}^d$. The XGBoost model estimates log market value:

$$
\hat{y}_y=f_{\text{value}}(\mathbf{x}_i)
$$

where:

$$
\hat{y}_i\approx\log(\text{sale price}_i)
$$

The estimated market value is then:

$$
\widehat{V}_i=e^{\hat{y}_i}
$$

In practice, the model output is interpreted as:

$$
\texttt{ml\_estimated\_market\_value}=\widehat{V}_i
$$

Because the data is noisy and property-level information is limited, this is treated as a locality-level transaction value proxy rather than a formal valuation.

## Market Trend Model
The XGBoost value model gives a current transaction-level value estimate. However, market conditions can change over time. To adjust for short-term local movement, we add a market trend layer.

Few models were experimented with. LSTM on suburb-level monthly sequences showed only marginal improvement over simple baselines and was not suitable for production because the suburb-month panel is sparse and irregular. The production pipeline uses `rolling_ridge_market_trend_v1`, a rolling-window Ridge regression model trained on suburb-level monthly market features.

### Input Features
The market trend model uses recent suburb-level features such as:
- `sales_count`
- `median_sale_price`
- `log_median_sale_price`
- `growth_1m`
- `growth_3m`
- `rolling_median_3m`
- `rolling_median_6m`
- `rolling_sales_count_3m`
- `rolling_sales_count_6m`
- `lagged growth features`
- `lagged log price features`
- `lagged sales count features`
- `rolling recent momentum features`

Target: `target_growth_3m`

This represents future 3-month log price movement.

For stability, the training target is clipped:

$$
\texttt{target\_growth\_3m\_clipped}\in[-0.30,0.30]
$$

At inference time, predictions are also scaled and clipped to avoid overreacting to noisy suburb-level transaction data.

Current inference calibration:
```python
prediction_scale = 0.2
prediction_clip = 0.035
```

So the final market trend adjustment is capped at approximately $\pm3.5$% over 3 months.

### Mathematical View
For a suburb-month observation $t$, define the market trend feature vector $\mathbf{z}_t\in\mathbb{R}^p$.

The Ridge regression model estimates:

$$
\hat{g}_t=\mathbf{w}^\top\mathbf{z}_t+b
$$

where $\hat{g}_t$ is the estimated 3-month log growth.

The Ridge objective is to estimate:

$$
(\mathbf{w}',b')=\underset{\mathbf{w},b}{\text{argmin}}\Bigg[\sum_{t=1}^n\Big(g_t-\mathbf{w}^\top\mathbf{z}_t-b\Big)^2+\lambda\lVert\mathbf{w}\rVert_2^2\Bigg]
$$

where $\lambda>0$ is the regularisation strength.

At inference time, the raw prediction is calibrated:

$$
\tilde{g}_t = \text{clip} \left(s\hat{g}_t,-g_{\max},g_{\max}\right)
$$

where:
- $s$ = prediction_scale = 0.2
- $g_{\max}$ = prediction_clip = 0.035

The market trend multiplier is:

$$
M_t=e^{\tilde{g}_t}
$$

The trend-adjusted market value is:

$$
\widehat{V}^{\text{trend}}_i=\widehat{V}_iM_t
$$

This gives:
```text
trend_adjusted_ml_market_value
```

### Interpretation
The market trend model should be interpreted as *short-term local market momentum*, instead of *precise house price forecasting*. The model is deliberately conservative because suburb-level monthly medians can be noisy due to transaction mix.

Example output:
```text
Recent local transaction data suggests a positive short-term market trend,
with an indicative 3-month movement of 3.0%.
```

## Development Cost Estimator
The development cost estimator approximates the cost of pursuing a selected strategy on a given site.

Strategies include:
- single dwelling rebuild
- granny flat
- dual occupancy
- townhouse / multi-dwelling
- low-rise apartment
- assembly opportunity
- land bank / hold

The setimator uses config-driven assumptions such as:
- floor space ratio proxy
- build cost per square metre
- strategy complexity multiplier
- constraint severity multiplier
- soft cost ratio
- contingency ratio
- labour/material escalation multiplier

### Mathematical View

For site $i$ and strategy $k$, let:
- $A_i$ = estimated land/site area proxy
- $r_k$ = strategy-specific FSR proxy
- $c_k$ = build cost per sqm for strategy k
- $m_k$ = strategy complexity multiplier
- $q_i$ = constraint multiplier

The gross floor area proxy is:

$$
GFA_{i,k}-A_i\cdot r_k
$$

The base construction cost is:

$$
C^{\text{base}}_{i,k} = GFA_{i,k} \cdot c_k \cdot m_k \cdot q_i
$$

Soft cost and contingency are estimated as:

$$
C^{\text{soft}}_{i,k} = \rho_{\text{soft}} C^{\text{base}}_{i,k}
$$

$$
C^{\text{cont}}_{i,k} = \rho_{\text{cont}} C^{\text{base}}_{i,k}
$$

The initial development cost is:

$$
C^{\text{dev}}_{i,k} = C^{\text{base}}_{i,k} + C^{\text{soft}}_{i,k} + C^{\text{cont}}_{i,k}
$$

## Construction Cost Trend Model
Development costs are affected by broader labour and construction cost conditions. To account for this, the pipeline uses Australian Bureau of Statistics (ABS) cost indices.

Current sources:
- ABS Wage Price Index
- ABS Producer Price Index

The current construction cost trend model is `wpi_construction_plus_ppi_output_proxy_v1`.

It combines:
- WPI construction index
- PPI output proxy index

The latest output includes:
```text
predicted_construction_cost_growth_qoq
construction_cost_escalation_multiplier
construction_cost_trend_score
construction_cost_trend_band
```

Example:
```text
Construction cost trend is moderate,
with an indicative next-quarter cost movement of 0.7%.
```

### Mathematical View
Let $W_t$ be the rebased WPI construction index, $P_t$ be the rebased PPI output proxy index.

The combined construction cost index is:

$$
I_t=0.55W_t+0.45P_t
$$

Quarter-on-quarter growth is:

$$
h_t=\frac{I_t - I_{t-1}}{I_{t-1}}
$$

The predicted next-quarter construction cost growth is currently estimated using a smoothed recent growth signal:

$$
\hat{h}_t = \text{clip} \Bigg(\frac{1}{4} \sum_{j=0}^{3} h_{t-j},\;-0.03,\;0.05 \Bigg)
$$

The construction cost escalation multiplier is:

$$
E_t=1+\hat{h}_t
$$

The trend-adjusted development cost is:

$$
C^{\text{dev,trend}}_{i,k}=C^{\text{dev}}_{i,k}\cdot E_t
$$

## Acquisition Cost Proxy

For redevelopment opportunities, transaction-level market value may underestimate the total acquisition requirement for a larger site or assembly candidate.

The pipeline therefore uses an acquisition proxy: `estimated_acquisition_cost`.

Depending on the site and strategy, this may come from:
- ML transaction value model
- locality median sale price
- scaled locality market proxy

For larger redevelopment candidates, the scaled locality proxy prevents the total project cost from being unrealistically low.

Total project cost is then:

$$
C^{\text{total}}_{i,k} = C^{\text{acq}}_{i,k}+C^{\text{dev, trend}}_{i,k}
$$

where:
- $C_{\text{acq}}$ = estimated acquisition cost 
- $C_{\text{dev},\text{trend}}$ = trend-adjusted development cost

## Value Potential and Cost Efficiency
### Value Potential
The value potential score is a bounded screening score based on site and locality attributes.

It considers:
- zoning
- lot size
- station proximity
- policy score
- locality sales count
- locality median price
- constraint severity
- selected strategy

This returns:
```text
value_potential_score
value_potential_band
```
The value potential score is not a predicted profit margin. It is an opportunity signal indicating whether the site has characteristics associated with stronger redevelopment potential.

### Cost Risk
Cost risk is based on total project cost and strategy-specific cost bands.

Higher cost projects receive higher capital intensity risk. Planning constraints can also increase cost risk.

Output:
```text
cost_risk_score
cost_band
```

### Cost Efficiency
Cost efficiency is designed to distinguish between opportunities with similar policy and value potential but different capital requirements.

For example, two sites may both have high policy upside, but one may require a much larger total project cost. The cost efficiency score helps a budget-sensitive ranking profile prefer the more capital-efficient opportunity.

A simplified representation is:

$$
S^{\text{eff}}_{i,k} = 0.65 S^{\text{cost}}_{i,k}+0.35 S^{\text{value}}_{i,k}
$$

where:
- $S_{\text{cost}}$ = score derived from total project cost relative to strategy benchmark
- $S_{\text{value}}$ = value_potential_score

Output: `cost_efficiency_score`

## Final Opportunity Fusion
The economics pipeline feeds into the final opportunity ranking layer.

For each site $i$ and strategy $k$, the final agent opportunity score combines:
- base strategy fit
- policy upside
- value potential
- cost efficiency
- budget fit
- cost risk penalty

A simplified scoring form is:

$$
S_{i,k} = w_b S^{\text{base}}_{i,k} + w_p S^{\text{policy}}_{i,k}+w_v S^{\text{value}}_{i,k}+w_e S^{\text{eff}}_{i,k}+w_{\text{budget}} S^{\text{budget}}_{i,k}-w_c S^{\text{risk}}_{i,k}
$$

where each component is normalised to a 0-100 style score before fusion.

Different ranking profiles use different weights:
```text
balanced
policy_upside
budget_sensitive
high_value
```

For example: `budget_sensitive` places more weight on cost efficiency and penalises high capital intensity more strongly.

This allows the same site pool to be ranked differently depending on the agent or client’s objective.
