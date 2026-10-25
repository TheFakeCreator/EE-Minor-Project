# ADR-0007: Critical Clearing Time (CCT) Ground-Truth Bisection Search and ML Regression

## Status
**Accepted**

## Context
Binary classification ($y \in \{0, 1\}$) informs an operator whether a specific clearing time $t_{\text{clear}}$ causes instability, but it does not specify the maximum allowable fault duration the system can endure before collapse. 

In power engineering, the **Critical Clearing Time (CCT)** is the maximum fault duration $t_{\text{cct}}$ for which synchronism is maintained:
$$t_{\text{clear}} \le t_{\text{cct}} \implies \text{Stable}$$
$$t_{\text{clear}} > t_{\text{cct}} \implies \text{Unstable}$$

Traditionally, CCT is computed through iterative bisection search using time-domain simulations. Because each scenario requires 5–8 successive simulation runs of 20 seconds, this process takes 15–30 seconds per operating condition, rendering it unusable for real-time dispatch or online contingency screening.

## Decision
1. **Ground-Truth Computation:** Implement an automated **bisection search algorithm** in ANDES with tolerance $\epsilon = 0.01\,\text{s}$ over $t \in [0.05, 0.45]\,\text{s}$.
2. **ML Regression Formulation:** Train regression models (Ridge Regression, Random Forest Regressor, XGBoost Regressor) to predict $t_{\text{cct}}$ directly from pre-fault operating conditions (fault bus, load scaling, active line outages, electrical distance to generators).
3. **Speedup Benchmark:** Quantify the latency advantage of ML inference ($t_{\text{ML}}$ in milliseconds) versus full numerical bisection ($t_{\text{TDS}}$ in seconds):
$$\text{Speedup} = \frac{t_{\text{TDS}}}{t_{\text{ML}}}$$

## Consequences
- Provides grid operators with continuous stability margins rather than just binary labels.
- Delivers the project's most concrete, quotable operational benchmark ("~2 ms ML inference vs ~15 s simulation = >7,000x speedup").
- Enables integration into real-time digital twins and the interactive Streamlit demo.
