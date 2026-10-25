# Architecture & Research Decision Records (ADRs)

This directory contains the formal records of architectural, engineering, and research decisions made in the **Grid Stability Prediction** project.

| ID | Title | Status | Date | Decision Summary |
| :--- | :--- | :---: | :---: | :--- |
| **[ADR-0001](0001-simulation-engine-and-benchmark.md)** | Selection of ANDES and IEEE 39-Bus Benchmark | **Accepted** | 2026-09 | Use open-source Python simulator ANDES with 10-generator New England system |
| **[ADR-0002](0002-stability-ground-truth-labeling.md)** | Stability Ground Truth Labeling via COI 180° Rule | **Accepted** | 2026-09 | Label stability by Center-of-Inertia (COI) maximum relative rotor deviation > 180° |
| **[ADR-0003](0003-domain-generalization-evaluation.md)** | Evaluation Protocol: Rejecting Random Splits for OOD Benchmarks | **Accepted** | 2026-10 | Benchmark models on held-out buses, load scaling, and N-1 contingencies |
| **[ADR-0004](0004-neural-networks-and-model-diversity.md)** | Adding MLP Classifier for Continuous Hypothesis Boundaries | **Accepted** | 2026-10 | Compare neural networks with tree ensembles under extreme domain shifts |
| **[ADR-0005](0005-physics-informed-features-vs-black-box.md)** | Transition from Pure Black-Box to Physics-Informed Invariant Features | **Accepted** | 2026-10 | Introduce electrical invariants (generator proximity, rotor speed kinetic deviation) |
| **[ADR-0006](0006-shap-explainability-and-physics-validation.md)** | Model Interpretability via TreeSHAP and Physics Validation | **Accepted** | 2026-10 | Global & local SHAP attributions tied to Swing Equation and Equal Area Criterion |
| **[ADR-0007](0007-cct-bisection-search-and-regression.md)** | Critical Clearing Time (CCT) Bisection Search and ML Regression | **Accepted** | 2026-10 | Ground truth bisection search in ANDES, ML regression, and 10,000x speedup benchmark |
