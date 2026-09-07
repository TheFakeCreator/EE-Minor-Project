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
- [ ] Set up the repo structure (see "Suggested structure" below)
- [ ] Install ANDES, confirm the version, run a sanity power flow + time-domain
      simulation on a bundled example case
- [ ] Locate and confirm the IEEE 39-bus New England case within ANDES's bundled
      examples/docs (verify against the installed version, don't assume a path)
- [ ] Literature pass: find and summarize 3–6 papers on ML-based transient
      stability / dynamic security assessment, ideally involving the 39-bus system
      or a comparably standard test system. Save to `research/literature-notes.md`
      with citations.
- [ ] Decide and document the scenario space: which buses get faults, fault
      clearing-time range, load-scaling range, which N-1 contingencies are in
      scope, and the exact stability-labeling rule (e.g. max rotor-angle deviation
      threshold)
- **Deliverable:** `research/literature-notes.md`, a working ANDES environment,
  a documented scenario-space spec

### Phase 1 — Simulation & Dataset Generation
- [ ] Build a config-driven scenario generator (YAML/JSON: fault bus list,
      clearing-time range, load-scaling range, contingency list, random seed)
- [ ] Build the ANDES simulation harness: run time-domain simulation per scenario,
      extract features (bus voltage magnitude/angle, generator active/reactive
      power, generator rotor angles, system frequency, line loading, fault
      location and clearing time)
- [ ] Implement the stability-labeling function as its own, separately tested
      module
- [ ] Run at a meaningful scale (hundreds to low thousands of scenarios, depending
      on runtime budget) and save the resulting dataset to `data/processed/`
- [ ] Log the generation run — seed, config, scenario count, wall-clock time — for
      reproducibility
- **Deliverable:** a reproducible dataset, the generation script, and a short data
  dictionary describing every column

### Phase 2 — Exploratory Data Analysis
- [ ] Class balance (stable vs unstable), feature distributions, correlations
- [ ] Sanity-check a handful of individual scenarios against physical intuition
      (e.g. faults near heavily loaded generators should skew unstable)
- [ ] Produce the core EDA plots (notebook or script), saved to `reports/figures/`
- **Deliverable:** EDA notebook + key figures

### Phase 3 — Baseline Models
- [ ] Standard random train/test split baseline: logistic regression, random
      forest, XGBoost
- [ ] Evaluate with precision/recall/F1 and a confusion matrix, not just accuracy
      (class imbalance is likely)
- [ ] Log every run's model, config, and metrics to a results file (e.g.
      `results/experiments.json`)
- **Deliverable:** a baseline results table

### 🔶 MID-SEM CHECKPOINT — progress PPT
Content: problem statement, why this test system and tooling were chosen, a
pipeline diagram (simulation → features → labeling → ML), literature-review
highlights, EDA highlights, baseline results if ready, and a roadmap of what's
left. Polished final models are not required for this deck.
- [ ] Mid-sem PPT drafted and saved to `reports/mid-sem-ppt/`

### Phase 4 — Generalization Experiments *(core research contribution — closes out the minor project)*
- [ ] Design at least two held-out-condition splits — e.g. train on a subset of
      fault buses/load range, test on the rest
- [ ] Compare generalization performance against the random-split baseline from
      Phase 3. This comparison is the main "so what" of the project.
- [ ] Try the neural-network variant if time allows
- **Deliverable:** generalization results + written discussion

### 🔶 MIDWAY CHECKPOINT — end-of-semester PPT (minor project wrap-up)
Content: everything from the mid-sem PPT, updated with final baseline results,
the generalization-experiment design and results, and what's planned for the
major-project continuation next semester (CCT, explainability, demo, final
report). This is the deliverable that closes out the minor project.
- [ ] Midway PPT drafted and saved to `reports/midway-ppt/`

---

## PART B — Next Semester (Major Project Continuation)

### Phase 5 — CCT Regression *(stretch, high value)*
- [ ] Ground truth: run a bisection search for Critical Clearing Time per scenario
      in ANDES
- [ ] Train a regressor (reusing the existing feature set where possible) to
      predict CCT
- [ ] Timing comparison: ML inference time vs full bisection-search time — a
      concrete, quotable headline result
- **Deliverable:** CCT results + a speed-comparison chart

### Phase 6 — Explainability
- [ ] Compute SHAP values on the best classifier (and the regressor, if built)
- [ ] Tie the top 2–3 features back to power-system intuition, in plain language
- **Deliverable:** SHAP plots + a short interpretation write-up

### Phase 7 — Demo
- [ ] Streamlit app: pick or generate a scenario, show the prediction (confidence,
      and CCT estimate if built)
- [ ] Keep it simple — this is a demo for a professor, not a shipped product
- **Deliverable:** a runnable `app/streamlit_app.py`

### Phase 8 — Final Report & Final PPT *(end of year)*
- [ ] Full report: problem, related work, methodology, results (baseline +
      generalization + CCT if built), limitations, conclusion
- [ ] Final PPT: same narrative, presentation length
- **Deliverable:** `reports/final-report.*`, `reports/final-ppt/`

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