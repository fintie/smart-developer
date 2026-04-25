# Strategy Feature Mapping

This document defines the first-pass logic behind each development strategy score.  
The goal is not to encode planning law exactly, but to create a practical and explainable scoring framework using the current site feature bundle.

## 1. Single Dwelling Rebuild

**Idea**  
A conservative, low-intensity redevelopment pathway. Suitable when the site appears usable for standard residential rebuilding, without requiring strong uplift or major assembly potential.

**Logic**  
This strategy should score well on low- to medium-density residential land, moderate lot size, and low constraint burden.  
It should be penalised by strong heritage, flood, and bushfire constraints.  
Very high-intensity zones should not necessarily score highest here, because the site may be better suited to denser redevelopment.

## 2. Assembly Opportunity

**Idea**  
A strategic holding or aggregation opportunity where the site may become more valuable when combined with neighbouring land.

**Logic**  
This strategy should score well when zoning intensity is relatively high, the lot is larger, zoning is mixed or flexible, and transport access is strong.  
It should still be penalised by major constraints such as heritage, flood, and severe bushfire exposure.  
This is less about immediate standalone development and more about strategic redevelopment potential.

## 3. Granny Flat

**Idea**  
A low-intensity secondary-use pathway, typically suitable for lower-density residential contexts with enough usable land.

**Logic**  
This strategy should score well on low-density residential zoning and moderate-to-large lot size.  
It should be less dependent on transport access than apartment-style strategies.  
Heritage and flood should reduce the score materially, since they may complicate even small-scale additional development.

## 4. Land Bank / Hold

**Idea**  
A medium- to long-term hold strategy where the site may not be optimal for immediate redevelopment, but may still have future planning or location upside.

**Logic**  
This strategy should score well where zoning suggests future uplift potential, transport access is good, or the site has broader strategic value.  
Constraints should reduce the score, but not always eliminate it entirely, because holding logic is more tolerant of present-day development friction than immediate build strategies.

## 5. Townhouse / Multi-Dwelling

**Idea**  
A medium-density redevelopment pathway, typically stronger where zoning supports more than detached housing and the lot is sufficiently large.

**Logic**  
This strategy should score well on medium-density or mixed-use style zoning, larger lot size, and better station access.  
It should be penalised by strong constraints, especially heritage, flood, and higher bushfire risk.  
It should also benefit from signs that the site sits in a more flexible or transition-oriented planning context.

## 6. Low-Rise Apartment

**Idea**  
A higher-intensity redevelopment pathway requiring stronger planning support, larger site capacity, and better accessibility.

**Logic**  
This strategy should score highest where zoning is clearly development-oriented, lot size is large, and rail/metro accessibility is strong.  
It should be heavily penalised by heritage, flood, and severe bushfire constraints.  
This is one of the most planning-sensitive and accessibility-sensitive strategies in the framework.

## 7. Dual Occupancy

**Idea**  
A moderate redevelopment pathway between a detached rebuild and a more intensive multi-dwelling form.

**Logic**  
This strategy should score well on low- to medium-density residential zoning with moderate-to-large lot size.  
It should be more lot-sensitive than single dwelling rebuild, but less demanding than townhouse or apartment strategies.  
Heritage, flood, and bushfire should reduce the score, especially where development complexity becomes materially higher.

## Shared Scoring Principles

**Feasibility**  
The strategy should align with the site's zoning intensity and basic physical capacity.

**Constraint Penalty**  
Heritage, flood, and bushfire act as downward pressure on development suitability.

**Opportunity Bonus**  
Transport accessibility and favourable planning context act as upward pressure, especially for denser strategies.

**Interpretability**  
Each strategy score should remain explainable in plain English:
- why the site scored well
- what is holding it back
- what kind of development path it most resembles

## Generic Scoring Formula

For each strategy $s$, we define a site suitability score as

$$S_s(x) = 100 \cdot \text{clip}(B_s + F_s(x) + O_s(x) - C_s(x),0,1)$$

where:

- $S_s(x)$ is the final score for strategy $s$ on site $x$
- $B_s$ is the base prior for strategy $s$
- $F_s(x)$ is the feasibility component
- $O_s(x)$ is the opportunity bonus component
- $C_s(x)$ is the constraint penalty component
- $\text{clip}(\cdot,0,1)$ truncates the score into the interval $[0,1]$

A more explicit weighted version is

$$S_s(x) = 100 \cdot \text{clip}(B_s + w^{(F)}_s F_s(x) + w^{(O)}_s O_s(x) - w^{(C)}_s C_s(x),0,1)$$

where the weights $w^{(F)}_s$, $w^{(O)}_s$, and $w^{(C)}_s$ depend on the strategy.

In the current framework, the components are interpreted as:

- $F_s(x)$: zoning and physical feasibility, such as zoning intensity and lot size
- $O_s(x)$: upside and accessibility, such as proximity to rail/metro stations
- $C_s(x)$: development friction, such as heritage, flood, and bushfire constraints