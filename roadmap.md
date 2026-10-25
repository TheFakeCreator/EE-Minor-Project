# Roadmap — Grid Stability Prediction

## How to use this
This project runs the full academic year: this semester as the minor project,
continuing next semester as the major project. Three checkpoints in total —
mid-sem PPT, midway PPT (end of this semester), final PPT (end of the year).
Check items off as you go, or track them in your own generated task list and
reflect status back here.

---

## PART A — This Semester (Minor Project)

### Phase 0 — Setup & Research
- [x] Set up the repo structure (see "Suggested structure" below)
- [x] Install ANDES, confirm the version, run a sanity power flow + time-domain
      simulation on a bundled example case
- [x] Locate and confirm the IEEE 39-bus New England case within ANDES's bundled
      examples/docs (verified `ieee39/ieee39_full.xlsx`)
- [x] Literature pass: find and summarize 3–6 papers on ML-based transient
      stability / dynamic security assessment. Saved to [`research/literature-notes.md`](research/literature-notes.md)
- [x] Decide and document the scenario space: documented in [`config/scenario_space.yaml`](config/scenario_space.yaml)
      with COI 180° rotor angle deviation threshold.
- **Deliverable:** `research/literature-notes.md`, a working ANDES environment,
  a documented scenario-space spec

### Phase 1 — Simulation & Dataset Generation
- [x] Build a config-driven scenario generator (`src/simulation/scenario_generator.py`)
- [x] Build the ANDES simulation harness: multi-processing time-domain simulation (`src/simulation/harness.py`)
- [x] Implement the stability-labeling function as its own, separately tested
      module (`src/labeling/stability.py`, unit tested in `tests/test_stability.py`)
- [x] Run at a meaningful scale: 450 scenarios generated and saved to [`data/processed/dataset.csv`](data/processed/dataset.csv)
- [x] Log the generation run (`data/processed/generation_log.json`)
- **Deliverable:** reproducible dataset, generation script, and data logs

### Phase 2 — Exploratory Data Analysis
- [x] Class balance (308 stable vs 142 unstable, ~2:1), feature distributions, correlations
- [x] Sanity-check scenarios against physical intuition (clearing time vs instability, voltage sags)
- [x] Produce core EDA plots saved to `reports/figures/` via `src/eda/generate_plots.py`
- **Deliverable:** EDA figures + scripts

### Phase 3 — Baseline Models
- [x] Standard random train/test split baseline: Logistic Regression, Random
      Forest, XGBoost (`src/models/train_baselines.py`)
- [x] Evaluate with precision/recall/F1, ROC-AUC, Brier score, and confusion matrix
      (`src/models/evaluate.py`)
- [x] Log model run configurations and metrics to [`reports/baseline_results.json`](reports/baseline_results.json)
- **Deliverable:** baseline results table, ROC curves, confusion matrices, feature importance plots

### 🔶 MID-SEM CHECKPOINT — progress PPT
- [x] Mid-sem PPT generated and saved to `reports/mid-sem-ppt/Grid_Stability_Prediction_MidSem.pptx`

### Phase 4 — Generalization Experiments *(core research contribution — closes out the minor project)*
- [x] Design three held-out-condition splits:
      - Split 1: Held-out fault buses (Buses 29, 39)
      - Split 2: Held-out heavy load stress (1.1, 1.2)
      - Split 3: Held-out N-1 line contingency (Line 16-17 outage)
- [x] Compare generalization performance against the random-split baseline from
      Phase 3 (`src/models/generalization_experiments.py`)
- [x] Quantify Generalization Gap and Missed Instability (blackout) risks
- **Deliverable:** [`research/generalization-study.md`](research/generalization-study.md), [`reports/generalization_summary.csv`](reports/generalization_summary.csv), and figures

### 🔶 MIDWAY CHECKPOINT — end-of-semester PPT (minor project wrap-up)
- [x] Midway PPT drafted and saved to `reports/midway-ppt/Grid_Stability_Prediction_Final_Minor.pptx`


---

## PART B — Next Semester (Major Project Continuation)

### Phase 5 — CCT Regression *(stretch, high value)*
- [x] Ground truth: run a bisection search for Critical Clearing Time per scenario
      in ANDES (`src/simulation/cct_search.py`, 45 conditions, avg 23.88s/condition)
- [x] Train a regressor to predict CCT (`src/models/train_cct_regressor.py`, $R^2 = 0.9636$, RMSE = 25 ms)
- [x] Timing comparison: ML inference (0.7 ms) vs full bisection-search (23.88 s) — **>33,000x Speedup!**
- **Deliverable:** [`reports/figures/cct_parity_plot.png`](reports/figures/cct_parity_plot.png), [`reports/figures/cct_speed_comparison.png`](reports/figures/cct_speed_comparison.png), [`reports/cct_results.json`](reports/cct_results.json)

### Phase 6 — Explainability
- [x] Compute SHAP values on the best classifier via TreeSHAP (`src/explain/shap_analysis.py`)
- [x] Tie top features (`v_mean_system`, `v_min_system`, `fault_bus_39`, `v_sag_impulse`) back to Swing Equation & Equal Area Criterion
- **Deliverable:** SHAP beeswarm, waterfall, and dependence plots in `reports/figures/`, documented in [`research/shap-explainability-report.md`](research/shap-explainability-report.md) and [`docs/adr/0006-shap-explainability-and-physics-validation.md`](docs/adr/0006-shap-explainability-and-physics-validation.md)

### Phase 7 — Demo
- [x] Streamlit app: pick or generate a scenario, show real-time prediction, confidence, CCT estimate, and SHAP attribution
- [x] Interactive UI with preset operating conditions & reactive margin of safety gauge
- **Deliverable:** runnable [`app/streamlit_app.py`](app/streamlit_app.py)

### Phase 8 — Final Report & Final PPT *(end of year / minor wrap-up)*
- [x] Full report: problem, related work, methodology, baseline + generalization + CCT + XAI + limitations + conclusion
- [x] Final PPT: 16-slide narrative deck with embedded high-resolution figures and all 4 team members
- **Deliverable:** [`reports/minor_project_report.md`](reports/minor_project_report.md), [`reports/midway-ppt/Grid_Stability_Prediction_Final_Minor.pptx`](reports/midway-ppt/Grid_Stability_Prediction_Final_Minor.pptx)

---

## Suggested repo structure
```
grid-stability-prediction/
├── AGENTS.md
├── roadmap.md
├── README.md
├── requirements.txt          (or pyproject.toml)
├── config/
│   └── scenario_space.yaml
├── src/
│   ├── simulation/           # ANDES harness, scenario generation
│   ├── features/             # feature extraction
│   ├── labeling/             # stability-labeling logic (+ tests)
│   ├── models/                # training, evaluation
│   └── explain/                # SHAP
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/                   # EDA, exploration
├── app/                          # Streamlit demo
├── research/
│   └── literature-notes.md
├── reports/
│   ├── figures/
│   ├── mid-sem-ppt/
│   ├── midway-ppt/
│   └── final-ppt/
└── tests/
```