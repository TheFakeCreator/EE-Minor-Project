# ADR-0006: Model Interpretability via TreeSHAP and Physics Validation

## Status
**Accepted**

## Context
While machine learning models achieve high accuracy on transient stability classification, deep learning and gradient boosted trees operate as black boxes. For power grid transmission system operators (TSOs) and academic defense, pure prediction without explanation is unacceptable:
- Grid operators cannot blindly trust an automated trip or islanding signal without understanding the physical causal mechanism.
- We must verify whether the model is learning true electro-mechanical dynamics (e.g. voltage sag severity, generator acceleration) or exploiting dataset artifacts/spurious correlations.

## Decision
We integrate **TreeSHAP (SHapley Additive exPlanations)** grounded in cooperative game theory (`shap.TreeExplainer`):
1. **Global Interpretability:** Generate SHAP beeswarm summary plots to visualize directional feature impact across the entire operating scenario envelope.
2. **Local Interpretability:** Generate sample-specific waterfall plots for both stable and unstable scenarios to quantify exact additive contributions $\phi_i$ to the log-odds prediction $\ln\frac{P(\text{Stable})}{1 - P(\text{Stable})}$.
3. **Physical Cross-Validation:** Verify that top SHAP attributions conform strictly to the **Equal Area Criterion** and **Swing Equation**:
$$M \frac{d^2\delta}{dt^2} = P_m - P_e(V)$$
where deep voltage drops ($v_{\text{mean}}, v_{\min}$) reduce electrical output $P_e$, driving positive accelerating torque.

## Consequences
- Confirmed that system voltage collapse metrics (`v_mean_system`, `v_min_system`) and generator proximity (`fault_bus_39`) dominate model decisions.
- Provides defensible, visual explanations for faculty presentations and publications.
