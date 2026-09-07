# Project Context — Grid Stability Prediction
*(EE Minor Project, 7th Semester, NIT Raipur)*

## Who this is for
- Student is a 7th-semester Electrical Engineering student, but with a software/AI-ML
  background (Accenture internship + job offer, built a full data pipeline and
  dashboard: AWS S3 → Snowflake → dbt → Next.js/React). Not a core-power-systems
  specialist.
- This build is **solo**: the student + this agent. No teammate is validating the
  power-systems physics. That means you (the agent) are the primary check on domain
  correctness — via literature, ANDES documentation, and sanity checks against
  published results — not a human catching your errors.
- This is a **minor** project (distinct from the 8th-sem major project). It needs to
  be real, working, and defensible — not groundbreaking research. Favor solid,
  correct execution over invented complexity.
- Immediate deadline: a **mid-semester progress PPT** for the professor. Full
  deliverables (pipeline, results, demo, report, final PPT) are due at semester end.

## Problem statement
Predict whether a power system remains stable or goes unstable after a disturbance
(fault, generator outage, large load change), using power-system simulation to
generate labeled scenarios and ML to learn the relationship between electrical
operating conditions and stability outcome. Initial target: binary classification
(stable/unstable). Stretch target: Critical Clearing Time (CCT) regression — the
maximum fault-clearing time for which the system stays stable.

Central research question (this is the actual contribution, keep it central):
**Can a model trained on one set of operating conditions and disturbances
generalize to fault locations / load levels / contingencies it never saw during
training?** A random train/test split does not test this — the split has to be
built around deliberately held-out conditions.

## Locked-in technical decisions
- **Simulation engine:** ANDES — open-source Python, `pip install andes`, no license
  needed. Supports power flow, time-domain (transient) simulation, and small-signal
  analysis.
- **Test system:** IEEE 39-bus New England system (10 generators) — the standard
  benchmark for transient stability studies in the literature.
- **ML stack:** Python — scikit-learn + XGBoost for baselines, a small neural net if
  time allows, SHAP for explainability.
- **Demo:** a lightweight Streamlit app.
- Everything must run from pip-installable, license-free tooling. No MATLAB/PSAT/
  PowerFactory dependency — reproducibility without campus software access matters.

## What "production quality" means here (and what it doesn't)
**Do:**
- Modular code: simulation / feature extraction / labeling / training / evaluation /
  app as separate, clearly-interfaced components.
- Config-driven scenario generation — no magic numbers for fault buses, clearing
  times, or load ranges buried in code.
- Reproducibility: seeded randomness, versioned/saved datasets, logged run
  parameters.
- Basic automated tests, at minimum for feature extraction and the
  stability-labeling logic.
- Clear README, pinned dependencies.

**Don't over-invest in** (this is a semester project, not a company):
- Docker/Kubernetes, CI/CD, cloud deployment.
- Distributed training or heavy hyperparameter-search infrastructure.
- A custom dashboard framework instead of just using Streamlit.

## Novelty / stretch goals, in priority order
1. **Held-out-condition generalization experiments** (the core contribution — see
   research question above).
2. **CCT regression**, with ground truth from a bisection search in ANDES, plus a
   direct timing comparison: ML inference time vs full bisection-search time. This
   is the strongest single concrete result for the report ("X ms vs Y seconds").
3. **SHAP explainability**, tying 2–3 top features back to power-system intuition
   (e.g. rotor angle spread, generator loading) in plain language.
4. **Streamlit demo**: pick/generate a scenario, show the prediction and confidence
   (and CCT estimate if built).

## Known reference points (starting points for the literature research phase —
do not treat as a substitute for actually searching)
- Recent work has generated very similar datasets: ANDES simulation of the 39-bus
  system with full generator dynamics (GENROU model), load scaled to roughly
  80–120% of nominal, combined with random three-phase faults and systematic N-1
  line-outage contingencies.
- A classic labeling approach in this literature: label each simulated sample
  stable/unstable using the maximum relative rotor-angle deviation observed during
  the transient window.
- Find and cite 3–6 relevant papers (ML + transient stability assessment,
  ideally on the 39-bus system or a comparably standard test system) during
  research — this note is a starting point, not the literature review.

## Timeline
This project runs the full academic year — this semester as the minor project,
continuing next semester as the major project. Three checkpoints:
- **Now → mid-sem PPT** (this semester): problem framing, literature summary, a
  working simulation pipeline (even at small scale), initial EDA. Trained models
  are not required yet.
- **Mid-sem → end of this semester → midway PPT**: finished baselines plus the
  core generalization experiments (the project's central research contribution).
  This closes out the minor project.
- **Next semester (major-project continuation) → final PPT**: CCT regression,
  explainability, the demo, and the full final report.

---
*Tip: save/rename this file as `AGENTS.md` at the project root. Antigravity loads
that automatically as persistent context every session, on top of whatever's in
`roadmap.md` and the chat.*