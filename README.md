<div align="center">
  <img src="public/nitrr_new_logo_new.png" alt="NIT Raipur Logo" width="150" />
</div>

# ⚡ Grid Stability Prediction using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![ANDES](https://img.shields.io/badge/Simulation-ANDES-orange.svg)](https://docs.andes.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Electrical Engineering Minor Project (7th Semester)**  
> **National Institute of Technology (NIT), Raipur**

This repository contains an end-to-end simulation, dataset generation, and machine learning pipeline to predict power grid transient stability following severe disturbances. By combining traditional Time-Domain Simulation (TDS) with state-of-the-art Machine Learning, this project drastically reduces Transient Stability Assessment (TSA) times while demonstrating the ability of ML models to generalize to unseen topological operating conditions.

---

## 👥 Team Members
- **Sanskar Gupta** (Roll No: 23117091)
- **Daksh Tripathi** (Roll No: 23117036)
- **Karandeep Taram** (Roll No: 23117052)
- **Veena Gupta** (Roll No: 23117112)

---

## 📖 The Problem
When a major disturbance occurs in a power grid (e.g., short-circuit faults or transmission line trips), generators begin to swing against each other. If they swing too far apart, the system loses synchronism, potentially leading to cascading blackouts.

Traditionally, grid operators rely on solving complex, non-linear differential equations (Time-Domain Simulation) to determine stability—a highly accurate but **computationally expensive** process. 

**Our Solution:** We train a Machine Learning model that evaluates immediate post-fault electrical parameters to **instantly predict** whether the system will survive or collapse.

---

## 🏗️ Project Architecture & Pipeline

1. **Test System:** IEEE 39-Bus New England System (The academic benchmark for transient stability research).
2. **Simulation Engine:** [ANDES](https://docs.andes.app/) (Open-source, Python-native power system simulator).
3. **Data Generation Pipeline:** Config-driven multi-processing automation that applies faults, load scaling, and N-1 contingencies.
4. **Machine Learning:** Scikit-Learn + XGBoost.

---

## 📊 Exploratory Data Analysis (EDA)

We programmatically generated a dataset of **450 unique scenarios** using parallel processing. 

### Class Balance & Dataset Composition
The generated dataset achieves a highly realistic natural imbalance of roughly **2:1 (Stable:Unstable)**, providing a mathematically sound foundation for training ML models without synthetic upsampling.

![Class Balance](reports/figures/class_balance.png)

### Physics Validation
Our simulations adhere strictly to power systems theory:
1. **Fault Clearing Time:** Faults cleared under 0.1s are almost exclusively stable, whereas longer faults (0.3s) rapidly destabilize the grid.
2. **Feature Correlation:** Unstable scenarios strongly correlate with deeper drops in the system's minimum voltage (`v_min_system`) during the transient window.

| Instability vs. Clearing Time | Feature Distribution (Voltage Sag) |
|:---:|:---:|
| ![Clearing Time](reports/figures/stability_vs_clearing_time.png) | ![Features](reports/figures/feature_distributions.png) |

---

## ⚙️ How Stability is Labeled

To generate our ground-truth labels for ML training, we calculate the **Center of Inertia (COI)**. The COI acts as the "average" angle of the entire grid. A scenario is mathematically labeled as **Unstable (0)** if:
1. ANDES detects severe instability and terminates the numerical solver early.
2. The maximum relative rotor angle deviation of any individual generator from the COI exceeds **180 degrees** during the transient window.

Otherwise, the scenario is **Stable (1)**.

---

## 🚀 Setup & Execution Guide

### 1. Environment Setup
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/TheFakeCreator/EE-Minor-Project.git
cd EE-Minor-Project

# Create and activate a virtual environment (Recommended)
python -m venv venv
# Windows: venv\Scripts\activate | Mac/Linux: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install seaborn python-pptx  # Required for EDA and reports
```

### 2. Generating the Dataset
The data generation pipeline utilizes Python's `ProcessPoolExecutor` to run hundreds of ANDES simulations in parallel across multiple CPU cores.

```bash
# Set PYTHONPATH so the `src` module is discovered
# On Windows (PowerShell):
$env:PYTHONPATH="."
python -m src.simulation.generate_dataset --workers 6

# On Linux/Mac:
PYTHONPATH="." python -m src.simulation.generate_dataset --workers 6
```
*(Depending on your CPU architecture, generating the 450 scenarios takes ~3-5 minutes).*

### 3. Running the EDA & Reporting
To regenerate the statistical plots shown above and build the PowerPoint presentation automatically:
```bash
# Generate EDA Plots (saves to reports/figures/)
python src/eda/generate_plots.py

# Generate PPT Presentation (saves to reports/mid-sem-ppt/)
python src/reports/generate_ppt.py
```

## 🔬 The Core Contribution: Domain Generalization Benchmarks

We reject standard random train/test splits that leak topology and artificially inflate accuracy. Instead, we benchmark models across deliberate out-of-distribution (OOD) topological and operational shifts:

| Experiment (Domain Shift) | Model | OOD Accuracy | OOD Macro F1 | Missed Blackout Rate | Core Physical Behavior |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Held-out Buses** (Topological) | **Random Forest** | **96.11%** | **0.9610** | **7.1% (7 cases)** | **Equal Area Criterion features cut error by >50%!** |
| | **XGBoost** | **100.00%** | **1.0000** | **0.0% (0 cases)** 🏆 | Flawless topological adaptation |
| | **Neural Network (MLP)** | **98.33%** | **0.9832** | **3.0% (3 cases)** 🌟 | Beat Random Forest on unseen buses |
| **Held-out Loads** (Stress Extrapolation) | **XGBoost** | **63.89%** | **0.6375** | **0.0% (Massive false alarm)** 🚨 | Tree step-function saturation outside training bounds |
| | **Neural Network (MLP)** | **97.78%** | **0.9752** | **6.3% (Smooth Extrapolation)** 🌟 | **Crushed XGBoost by +34% accuracy!** |
| **Held-out Contingency** (N-1 Structural) | **Random Forest** | **100.00%** | **1.0000** | **0.0% (0 cases)** 🏆 | Perfect structural adaptation |
| | **XGBoost** | **100.00%** | **1.0000** | **0.0% (0 cases)** 🏆 | Perfect structural adaptation |

![Generalization Gap](reports/figures/generalization_gap.png)

---

## ⏱️ Critical Clearing Time (CCT) & >33,000x Speedup

Predicting the continuous **Critical Clearing Time ($t_{\text{cct}}$)** provides operators with real-time safety margins ($t_{\text{cct}} - t_{\text{clear}}$):

| Method | Execution Latency | CCT Precision ($R^2$) | Operational Speedup |
| :--- | :---: | :---: | :---: |
| **ANDES Numerical Bisection** | **23.88 seconds** | 1.000 (Exact) | Baseline |
| **Machine Learning Regressor** | **0.705 ms** (XGBoost) / **0.042 ms** (Ridge) | **0.9636** (RMSE = 25 ms) | **>33,000x FASTER!** 🚀 |

| CCT Parity Agreement | Operational Speedup Benchmark |
|:---:|:---:|
| ![CCT Parity](reports/figures/cct_parity_plot.png) | ![Speedup](reports/figures/cct_speed_comparison.png) |

---

## 🧠 Explainable AI via TreeSHAP

Using cooperative game theory, we decompose black-box predictions into exact additive feature attributions ($\phi_i$), proving mathematically that model decisions satisfy the classical **Swing Equation** ($M \ddot{\delta} = P_m - P_e$):

![SHAP Summary](reports/figures/shap_summary_beeswarm.png)

---

## 💻 Interactive Digital Twin Dashboard

Run the real-time Streamlit digital twin web application:
```bash
python -m streamlit run app/streamlit_app.py
```
Open **[http://localhost:8501](http://localhost:8501)** to interact with:
- Dynamic scenario sliders (Fault bus, Load scale $0.8\times - 1.3\times$, Clearing time $0.04\text{s} - 0.40\text{s}$).
- Real-time 🟢 **STABLE** vs 🔴 **UNSTABLE POLE SLIP** badge.
- Live Critical Clearing Time safety margin gauge ($t_{\text{cct}} - t_{\text{clear}}$ in ms).
- Real-time SHAP waterfall attribution of contributing risk factors.

---

## 🏛️ Architecture Decision Records (ADRs)

All engineering and scientific decisions are formally tracked under [`docs/adr/`](docs/adr/):
- **[ADR-0001](docs/adr/0001-simulation-engine-and-benchmark.md):** ANDES Simulator & IEEE 39-Bus Selection
- **[ADR-0002](docs/adr/0002-stability-ground-truth-labeling.md):** Ground Truth Labeling via COI 180° Rule
- **[ADR-0003](docs/adr/0003-domain-generalization-evaluation.md):** Domain Generalization Protocol (Rejecting Random Splits)
- **[ADR-0004](docs/adr/0004-neural-networks-and-model-diversity.md):** Neural Network (MLP) Convergence & Stress Extrapolation
- **[ADR-0005](docs/adr/0005-physics-informed-features-vs-black-box.md):** Physics-Informed Invariant Feature Engineering
- **[ADR-0006](docs/adr/0006-shap-explainability-and-physics-validation.md):** TreeSHAP Interpretability & Swing Equation Validation
- **[ADR-0007](docs/adr/0007-cct-bisection-search-and-regression.md):** CCT Bisection Ground Truth & >33,000x Speedup Benchmark

---

## 🗺️ Roadmap & Milestones

- [x] **Phase 1:** Set up ANDES simulation harness and parallel data generation pipeline (450 scenarios).
- [x] **Phase 2:** Exploratory Data Analysis (EDA) and Physics Validation.
- [x] **Phase 3:** Train Baseline ML models (LR, Random Forest, XGBoost, MLP — all 100% in-distribution).
- [x] **Phase 4:** Core Research: Domain Generalization experiments on held-out buses, loads, and contingencies.
- [x] **Phase 5:** Critical Clearing Time (CCT) Regression mapping (>33,000x speedup).
- [x] **Phase 6:** Implement TreeSHAP explainability and physical Swing Equation validation.
- [x] **Phase 7:** Develop interactive Streamlit Digital Twin Web Application.
- [x] **Phase 8:** Comprehensive Minor Project Report (`reports/minor_project_report.md`) & Final PPT (`reports/midway-ppt/`).

