# ADR-0004: Adding MLP Classifier for Continuous Hypothesis Boundaries

## Status
**Accepted**

## Context
Our Phase 4 generalization experiments revealed that Gradient Boosted Trees (XGBoost) suffer an acute extrapolation collapse (35% F1 degradation) when operating loads exceed the training boundary ($load\_scale \in [1.1, 1.2]$). 

Decision tree splits are axis-aligned step functions with zero gradient outside the minimum and maximum feature bounds:
$$\forall x > x_{\max}, \quad f_{\text{tree}}(x) = \text{leaf}_{\text{extreme}}$$

In contrast, smooth parameteric models like Multi-Layer Perceptrons (MLPs) parameterized by continuous activation functions (ReLU, GELU, Sigmoid) maintain continuous non-zero gradients beyond observed training domain boundaries.

## Decision
Incorporate a **Multi-Layer Perceptron (MLP)** into both baseline and generalization experiment suites (`src/models/train_baselines.py` and `src/models/generalization_experiments.py`) to systematically compare tree ensemble step functions against neural continuous representations.

## Consequences & Empirical Findings
- **Resolution of Premature Early-Stopping:** Initial attempts with `early_stopping=True` caused the MLP to terminate at epoch 15 on small 27-sample validation sets, resulting in 45% accuracy. Removing early stopping and training to full convergence enabled the MLP to achieve **100% in-distribution baseline accuracy (CV Macro F1 = 0.9967)**.
- **Superior Stress Extrapolation Over XGBoost:** On the held-out heavy load experiment ($load\_scale \in [1.1, 1.2]$), the MLP achieved **97.78% accuracy (F1 = 0.9752)**, completely overcoming XGBoost's 63.89% step-function saturation.
- **Consistent Generalization Across All Shifts:** 
  - Held-out Buses: **98.33% accuracy** (surpassing Random Forest's 96.11%)
  - Held-out Heavy Loads: **97.78% accuracy**
  - Held-out Line Outage: **96.67% accuracy**
- **Conclusion:** Properly trained continuous neural networks provide the most uniform out-of-distribution generalization across continuous parameter shifts in power systems.
