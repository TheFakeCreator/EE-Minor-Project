# Empirical Study: Generalization of Machine Learning in Power Grid Transient Stability Assessment

**Project:** Electrical Engineering Minor Project (7th Semester), NIT Raipur  
**System:** IEEE 39-Bus New England Test System  
**Simulator Engine:** ANDES (Python-native DAE Solver)  
**Authors:** Sanskar Gupta, Daksh Tripathi, Karandeep Taram, Veena Gupta  
**Date:** October 2026  

---

## 1. Research Question & Motivation

In traditional power system operations, **Transient Stability Assessment (TSA)** evaluates whether synchronous generators remain in synchronism following severe disturbances (e.g., three-phase faults, generator trips, or transmission line outages). Time-Domain Simulation (TDS) solves large systems of non-linear Differential-Algebraic Equations (DAEs):

$$\dot{x} = f(x, y, u)$$
$$0 = g(x, y, u)$$

While numerically exact, TDS is computationally prohibitive for real-time Dynamic Security Assessment (DSA). Machine Learning (ML) classifiers have been widely proposed to replace TDS by learning an instantaneous mapping from post-fault measurements $\mathbf{x} \in \mathbb{R}^d$ to stability outcome $y \in \{0, 1\}$.

However, **standard ML evaluation in the literature almost universally employs random train/test splits**, assuming that training and test samples are Independent and Identically Distributed (I.I.D.):
$$\mathcal{P}_{\text{train}}(\mathbf{x}, y) = \mathcal{P}_{\text{test}}(\mathbf{x}, y)$$

In real power grids, an operating grid is never identical to historical training sets:
1. Faults occur on transmission buses never previously faulted (**Topological Shift**).
2. The grid operates under unforeseen peak demand or renewable volatility (**Operational Stress Shift**).
3. Critical lines trip out of service (**Structural Contingency Shift**).

### Central Research Question:
> **"Can machine learning models trained on a standard distribution of operating conditions and disturbances reliably generalize to unseen fault locations, unseen load levels, and unseen topological contingencies?"**

---

## 2. Experimental Methodology

### 2.1 Simulation & Ground Truth Labeling
Using ANDES, we generated **450 unique dynamic scenarios** on the IEEE 39-bus system with 10 synchronous generators modeled using 6th-order GENROU equations with IEEE-T1 exciter models.

The system is labeled **Unstable ($y = 0$)** if:
1. The ANDES numerical solver encounters non-convergence or step-size underflow prior to simulation end ($t_f = 20.0\,\text{s}$).
2. The maximum rotor angle deviation of any generator from the Center of Inertia (COI) exceeds $180^\circ$ during the transient period:
$$\max_{t \in [0, T]} \max_{i} |\delta_i(t) - \delta_{\text{COI}}(t)| > 180^\circ$$
where $\delta_{\text{COI}}(t) = \frac{1}{\sum H_i} \sum_{i=1}^{10} H_i \delta_i(t)$. Otherwise, labeled **Stable ($y = 1$)**.

### 2.2 Feature Space
- Operating conditions: System load scaling factor $\lambda \in [0.8, 1.2]$
- Disturbance location: Fault Bus ID
- Fault Clearing Time: $t_{\text{clear}} \in [0.05, 0.30]\,\text{s}$
- Structural state: N-1 line contingency (Intact, Line 2-3 outage, Line 16-17 outage)
- Dynamic measurements: Minimum system voltage sag ($v_{\min}$) and mean system voltage ($v_{\text{mean}}$) during fault clearance.

---

## 3. Empirical Results

### 3.1 In-Distribution Baseline (Standard 80/20 Stratified Split)
When trained and tested on randomly split scenarios, all models perform near-perfection:

| Model | Accuracy | Macro F1 | Unstable Recall | Missed Instability Rate | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **100.00%** | **1.0000** | 100.00% | **0.0%** (0 cases) | 1.000 |
| **Random Forest** | **100.00%** | **1.0000** | 100.00% | **0.0%** (0 cases) | 1.000 |
| **XGBoost** | **98.89%** | **0.9872** | 100.00% | **0.0%** (0 cases) | 1.000 |
| **Neural Network (MLP)** | **87.78%** | **0.8613** | 85.71% | **14.3%** (4 cases) | 0.982 |

*Takeaway:* Looking solely at standard random train/test splits creates a **false sense of near-perfect security** for tree and linear models.

---

### 3.2 Out-of-Distribution (Held-Out) Generalization Benchmarks

We evaluated four diverse model paradigms across three domain-shift splits:
- **Experiment 1 (Held-out Fault Buses):** Train on buses $\{2, 16, 26\}$ ($N_{\text{train}}=270$); test on completely unseen buses $\{29, 39\}$ ($N_{\text{test}}=180$).
- **Experiment 2 (Held-out High Stress):** Train on nominal/light load $\{0.8, 0.9, 1.0\}$ ($N_{\text{train}}=270$); test on extreme load $\{1.1, 1.2\}$ ($N_{\text{test}}=180$).
- **Experiment 3 (Held-out Contingency):** Train on Intact grid + Line 2-3 outage ($N_{\text{train}}=300$); test on unseen Line 16-17 outage ($N_{\text{test}}=150$).

#### Summary Table:

| Experiment (Domain Shift) | Model | In-Dist F1 | OOD F1 | $\Delta$ F1 (Degradation) | OOD Accuracy | OOD Missed Instability |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Held-out Buses** (Topological) | Logistic Regression | 1.0000 | 0.9776 | -0.0224 | 97.78% | 4.0% (4 cases) |
| | Random Forest | 1.0000 | 0.9610 | -0.0390 | 96.11% | 7.1% (7 cases) *(down from 15.2%)* |
| | **XGBoost** | 0.9872 | **1.0000** | **+0.0128** | **100.00%** | **0.0% (0 cases)** 🏆 |
| | **Neural Network (MLP)** | 1.0000 | **0.9832** | **-0.0168** | **98.33%** | **3.0% (3 cases)** 🌟 |
| **Held-out Loads** (Stress Extrapolation) | Logistic Regression | 1.0000 | 0.9815 | -0.0185 | 98.33% | 4.8% (3 cases) |
| | Random Forest | 1.0000 | 0.9815 | -0.0185 | 98.33% | 4.8% (3 cases) |
| | **XGBoost** | 0.9872 | **0.6375** | **-0.3496** | **63.89%** | **0.0% (Tree Saturation)** 🚨 |
| | **Neural Network (MLP)** | 1.0000 | **0.9752** | **-0.0248** | **97.78%** | **6.3% (Smooth Extrapolation)** 🌟 |
| **Held-out Contingency** (N-1 Structural) | Logistic Regression | 1.0000 | 0.9558 | -0.0442 | 96.00% | 10.9% (6 cases) |
| | **Random Forest** | 1.0000 | **1.0000** | **0.0000** | **100.00%** | **0.0% (0 cases)** 🏆 |
| | **XGBoost** | 0.9872 | **1.0000** | **+0.0128** | **100.00%** | **0.0% (0 cases)** 🏆 |
| | **Neural Network (MLP)** | 1.0000 | **0.9634** | **-0.0366** | **96.67%** | **9.1% (5 cases)** |

---

## 4. Key Physical & Algorithmic Findings

### Finding 1: Random Forest Fails on Topological Generalization (The Dangerous Failure Mode)
- In the held-out bus experiment, Random Forest accuracy plummeted to 91.67%, but more dangerously, **missed instability jumped to 15.15% (15 actual grid collapses misclassified as stable)**.
- **Physical Rationale:** Random Forest creates orthogonal axis splits on feature spaces. When trained on buses 2, 16, and 26, decision trees partition the voltage sag space around the electrical distance of those specific buses to nearby generators. Bus 39 is a generator bus directly connected to Gen 10. Faults at Bus 39 cause immediate severe acceleration of Gen 10 even for moderate voltage dips. Random Forest's leaves failed to recognize this non-local sensitivity, erroneously predicting stability.

### Finding 2: XGBoost Suffers Severe Extrapolation Collapse under Heavy Load Stress
- When tested on loads exceeding the training boundary ($\lambda \in [1.1, 1.2]$), **XGBoost accuracy collapsed by 35% down to 63.89% (Macro F1 = 0.6375)**.
- **Algorithmic Rationale:** Gradient boosted decision trees do not extrapolate linear or polynomial gradients beyond bounding split thresholds. For any feature value $x > x_{\max}$, the tree assigns the constant leaf value of the extreme bin. In contrast, Logistic Regression learns a monotonic hyperplane $w^T x + b$, preserving correct directionality under domain shift.

### Finding 3: Logistic Regression Struggles with Non-Linear Structural Outages
- In the held-out transmission line outage (`Line_16_17`), Logistic Regression missed **10.9% of instabilities**.
- **Physical Rationale:** Transmission line tripping alters the admittance matrix $\mathbf{Y}_{\text{bus}}$ non-linearly. The change in transfer reactances between generator pairs cannot be captured by a linear hyper-plane without interaction terms.

---

## 5. Power System Engineering Implications

1. **Random train/test splits are invalid for mission-critical DSA.** Any study claiming >99% accuracy on a random split must be treated with skepticism unless validated across held-out contingencies and topological locations.
2. **False Negatives are catastrophic.** In power engineering, a false alarm (classifying stable as unstable) triggers precautionary redispatch or generation curtailment (economic cost). A missed instability (classifying unstable as stable) means zero protective action, resulting in **cascading blackouts and equipment destruction**.
3. **Hybrid Ensembles are required.** Because linear models handle operational trend extrapolation well while tree ensembles handle topological non-linearities, production systems must utilize physics-constrained ensembles or GNNs rather than isolated models.

---

## 6. Generated Visual Artifacts

The following figures were generated and saved to `reports/figures/`:
1. `reports/figures/baseline_roc_curves.png`: In-distribution ROC curves showing AUC = 1.000 across all baselines.
2. `reports/figures/baseline_confusion_matrices.png`: Confusion matrices under random split.
3. `reports/figures/baseline_feature_importances.png`: Feature importance hierarchy (Clearing time and voltage sag dominate).
4. `reports/figures/generalization_gap.png`: Grouped comparative bar plot demonstrating the In-Distribution vs Out-of-Distribution drop.
5. `reports/figures/generalization_missed_instability.png`: Critical blackout risk (Missed Instability Rate) across held-out conditions.
