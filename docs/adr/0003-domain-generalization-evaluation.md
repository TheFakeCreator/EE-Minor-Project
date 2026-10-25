# ADR-0003: Evaluation Protocol: Rejecting Random Splits for OOD Benchmarks

## Status
**Accepted**

## Context
Standard machine learning evaluations in academic literature overwhelmingly use random 80/20 train/test splits. In our experiments, Random Forest, XGBoost, and Logistic Regression all achieved 98.9%–100.0% accuracy under random splitting. 

However, power grids never experience disturbances under identically distributed conditions. A random split leaks topology, fault locations, and load levels between train and test distributions, obscuring fatal model failure modes.

## Decision
We formally adopt **Domain Generalization / Held-Out Condition Protocols** as our primary evaluation methodology:
1. **Held-Out Fault Buses (Topological Shift):** Train on buses $\{2, 16, 26\}$; test on unseen buses $\{29, 39\}$.
2. **Held-Out System Stress (Operational Stress Extrapolation):** Train on load scales $\le 1.0$; test on high-stress grid conditions $\lambda \in \{1.1, 1.2\}$.
3. **Held-Out N-1 Outage (Structural Contingency):** Train on intact grid + Line 2-3 outage; test on unseen Line 16-17 outage.

We also mandate tracking the **Missed Instability Rate** (False Negative Rate for the unstable class), as predicting an unstable grid to be stable poses an unacceptable blackout risk.

## Consequences
- Uncovered real failure modes: Random Forest missed 15.2% of actual blackouts on unseen buses; XGBoost collapsed from 98.9% to 63.9% accuracy under heavy load stress.
- Provides defensible, novel academic contributions for presentation and publication.
