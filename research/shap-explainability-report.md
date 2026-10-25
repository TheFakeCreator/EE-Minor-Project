# Research Report: SHAP Explainability & Physical Validation in Grid Stability Assessment

**Project:** Electrical Engineering Minor Project (7th Semester), NIT Raipur  
**System:** IEEE 39-Bus New England Test System  
**Simulator Engine:** ANDES  
**Module:** Phase 6 — Explainability via TreeSHAP  
**Date:** October 2026  

---

## 1. Executive Summary & Objective

In modern mission-critical power systems, **Explainable AI (XAI)** is a strict prerequisite for deployment. A Transmission System Operator (TSO) cannot execute emergency control (such as generator tripping or rapid islanding) based solely on a binary 0/1 prediction without knowing **why** the grid is predicted to destabilize.

Using **TreeSHAP (SHapley Additive exPlanations)**, grounded in cooperative game theory, we decompose the model's non-linear predictions into exact additive contributions:
$$f(\mathbf{x}) = \phi_0 + \sum_{i=1}^{M} \phi_i(\mathbf{x})$$
where $\phi_0 = \mathbb{E}[f(\mathbf{x})]$ is the expected baseline value (base rate) and $\phi_i(\mathbf{x})$ is the Shapley attribution of feature $i$.

This study validates that the ML model has learned **authentic power-systems physics** rather than dataset artifacts or spurious correlations.

---

## 2. Global Feature Attributions & Physics Validation

### Quantitative Ranking of Top Predictive Features

From our trained XGBoost classifier evaluated across all test scenarios, the top predictive features by mean absolute SHAP value are:

| Rank | Feature | Mean \|SHAP Value\| | Power System Physical Significance |
| :---: | :--- | :---: | :--- |
| **1** | `v_mean_system` | **3.7526** | **Grid-Wide Transient Voltage Integrity:** System-wide average voltage across all 39 buses during fault clearance. Reflects reactive power balance and network-wide coupling. |
| **2** | `v_min_system` | **0.5192** | **Local Fault Severity & Accelerating Power:** Deepest voltage drop observed. Directly governs electrical power delivery $P_e = \frac{E V}{X} \sin \delta$. |
| **3** | `fault_bus_16` | **0.1728** | **Central Load Transit Hub:** Bus 16 connects multiple transmission lines (Lines 15-16, 16-17, 16-19, 16-21, 16-24). Faults here isolate major load corridors. |
| **4** | `fault_bus_39` | **0.1662** | **Generator 10 Terminal:** Direct connection to synchronous generator 10 ($H = 42.0\,\text{s}$). Three-phase fault causes instantaneous zero power export ($P_e \to 0$). |
| **5** | `v_sag_impulse` | **0.0873** | **Equal Area Criterion Kinetic Impulse Proxy:** $(1 - v_{\min}) \times t_{\text{clear}}$, approximating accelerating kinetic energy $\int (P_m - P_e) dt$. |

---

## 3. Deep Physical Analysis of Findings

### 3.1 Why `v_mean_system` and `v_min_system` Dominate
Under the classical **Swing Equation**:
$$M_i \frac{d^2\delta_i}{dt^2} = P_{m,i} - P_{e,i}$$
During a three-phase fault, the electrical power output $P_{e,i} = \sum_{j=1}^{n} |V_i| |V_j| |Y_{ij}| \cos(\delta_i - \delta_j - \theta_{ij})$ plummets towards zero because bus voltages $V_i, V_j$ collapse. 
- High values of `v_mean_system` (near 1.0 p.u.) indicate rapid voltage support and strong network stiffness, generating large negative SHAP values that **pull the model towards Stability (1)**.
- Deep drops in `v_mean_system` and `v_min_system` mean generators cannot export power, causing kinetic energy accumulation ($E_k = \frac{1}{2} J \omega^2$) that pushes rotor angle deviations beyond the $180^\circ$ COI threshold, **pulling the model towards Instability (0)**.

### 3.2 The Asymmetry of Bus Locations (`fault_bus_39` vs `fault_bus_16`)
- `fault_bus_39` has a large positive SHAP value pushing towards instability whenever a fault occurs at Bus 39. This confirms that the model has learned the physical vulnerability of **direct generator terminal faults** versus remote transmission bus faults.
- `fault_bus_16` represents the central switching hub in the New England system. When a fault occurs at Bus 16, power transfer between the northern and southern areas of the grid is severely choked.

---

## 4. Local Scenario Explanations (Waterfall Case Studies)

### Case Study A: Confidently Stable Scenario
- **Test Case:** Scenario with fast clearing time ($t_{\text{clear}} = 0.05\,\text{s}$) and high post-fault voltage ($v_{\text{mean}} \approx 1.05\,\text{p.u.}$).
- **Waterfall Mechanics:** The base rate starts at $\mathbb{E}[f(x)] \approx +0.77$. The high mean system voltage contributes a positive SHAP push ($+2.35$), counteracting any minor line outage effects and resulting in $f(x) > +3.0$ ($P(\text{Stable}) > 99\%$).
- **Artifact:** Saved to [`reports/figures/shap_waterfall_stable.png`](../reports/figures/shap_waterfall_stable.png).

### Case Study B: Confidently Unstable Scenario (Grid Collapse)
- **Test Case:** Severe scenario with prolonged clearing time ($t_{\text{clear}} = 0.30\,\text{s}$) and catastrophic voltage depression ($v_{\text{mean}} < 0.90\,\text{p.u.}$).
- **Waterfall Mechanics:** Low mean system voltage drives a massive negative attribution ($\phi \approx -3.82$), and high voltage sag impulse drives $\phi \approx -0.45$. The model's log-odds output drops to $f(x) < -4.0$ ($P(\text{Stable}) < 1\%$, definite collapse).
- **Artifact:** Saved to [`reports/figures/shap_waterfall_unstable.png`](../reports/figures/shap_waterfall_unstable.png).

---

## 5. Visual Artifacts Generated

1. **Global Beeswarm Summary Plot:**
   - [`reports/figures/shap_summary_beeswarm.png`](../reports/figures/shap_summary_beeswarm.png)
   - Visualizes feature value spectrum (red = high, blue = low) and directional push.
2. **Feature Importance Bar Plot:**
   - [`reports/figures/shap_bar_importance.png`](../reports/figures/shap_bar_importance.png)
   - Quantifies the relative hierarchy of the top 10 features.
3. **Local Waterfall Plots:**
   - [`reports/figures/shap_waterfall_stable.png`](../reports/figures/shap_waterfall_stable.png)
   - [`reports/figures/shap_waterfall_unstable.png`](../reports/figures/shap_waterfall_unstable.png)
4. **SHAP Dependence & Interaction Plot:**
   - [`reports/figures/shap_dependence_clearing_vs_vsag.png`](../reports/figures/shap_dependence_clearing_vs_vsag.png)
   - Demonstrates the non-linear interaction between fault duration and voltage collapse.
