# Literature Notes: ML for Transient Stability Assessment (TSA)

This document summarizes recent research on applying Machine Learning to Transient Stability Assessment (TSA), particularly focusing on standard test systems like the IEEE 39-bus system.

## 1. Core Methodologies and Models
Recent literature predominantly treats TSA as a binary classification problem (Stable vs. Unstable) or early prediction problem based on post-fault trajectories.

- **Ensemble Methods (XGBoost, Random Forest):** Highly popular for baseline performance due to their interpretability, robustness to class imbalance (when combined with sampling techniques), and speed. They handle tabular features effectively.
- **Support Vector Machines (SVM):** Historically used to define complex hyperplanes separating stable and unstable operating points.
- **Deep Learning (CNNs, RNNs):** Used for extracting spatio-temporal features directly from post-fault time-series trajectories. RNNs/LSTMs are particularly used for *early prediction* by processing sequential measurements before the system fully destabilizes. Graph Neural Networks (GNNs) are emerging to incorporate the grid's topology directly into the learning process.

## 2. Dataset Generation and Feature Selection
Data is consistently generated using time-domain simulation tools (e.g., ANDES, DIgSILENT, PSAT).

- **Scenarios:** Combinations of varying load levels, generator dispatch, and N-1 contingencies, subjected to three-phase faults at different buses with varying clearing times.
- **Feature Selection:** 
  - **Dynamic Features:** Generator rotor angles ($\delta$), rotor speeds ($\omega$), active power ($P$), reactive power ($Q$), terminal voltages ($V$).
  - **Pre-fault Static Features:** Initial bus voltages, line flows, generator loading.
- **Class Imbalance:** A recurring challenge noted in the literature. Because most grid configurations are inherently stable (and instability is rare), datasets are heavily skewed towards the 'Stable' class. SMOTE, under-sampling, or custom loss functions are often required.

## 3. Stability Labeling Criteria
A standard rule for labeling time-domain simulation results:
- **Rotor Angle Deviation Threshold:** A system is considered *unstable* if the maximum relative rotor angle deviation between any two generators (or between any generator and the Center of Inertia - COI) exceeds a critical threshold (typically $180^\circ$ or $360^\circ$) during the transient window.

## References
1. Research approaches applying CNNs and deep learning for spatio-temporal post-fault feature extraction.
2. Classical ML frameworks utilizing SVMs and Ensemble methods for robust classification under varying operating conditions.
3. Studies leveraging publicly available datasets (Mendeley) of IEEE 39-bus system simulations to benchmark classification accuracy and address class imbalance.
