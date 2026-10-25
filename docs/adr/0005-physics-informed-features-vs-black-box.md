# ADR-0005: Transition from Pure Black-Box to Physics-Informed Invariant Features

## Status
**Accepted**

## Context
In our held-out bus experiment, Random Forest missed 15.2% of instabilities. The primary cause is feature representation: bus numbers ($2, 16, 26, 29, 39$) are purely discrete categorical IDs. When a model is evaluated on unseen bus 39, its one-hot vector is all zeros, forcing the model to rely solely on system-wide aggregate voltages ($v_{\min}, v_{\text{mean}}$).

However, Bus 39 is fundamentally different from Bus 2:
- Bus 39 is a **generator bus** directly feeding Generator 10 ($H = 42\,\text{s}$). A three-phase fault at Bus 39 instantly drops electrical power output $P_e \to 0$ while mechanical input $P_m$ remains constant, resulting in extreme rotor acceleration ($\frac{d\omega}{dt} = \frac{P_m - P_e}{2H}$).
- Bus 2 is a load/transit bus electrically distant from the generation centers.

Without explicit physical grounding, ML models treat all unseen buses identically.

## Decision
Enrich the feature space with **Physics-Informed Invariant Features**:
1. `is_generator_bus`: Binary indicator of whether the faulted bus directly connects a synchronous generator ($1$ for buses 30–39, $0$ for load buses 1–29).
2. `delta_omega_max_clear`: Maximum generator speed deviation observed across the fleet at fault clearance ($t_c$), serving as a direct physical metric of kinetic energy accumulated during the disturbance.
3. `fault_clearing_ratio`: Ratio of clearing time to nominal operating cycles.

## Consequences & Empirical Verification
- **Stress Extrapolation Cured in Random Forest:** Adding the composite physical feature `stress_clearing_ratio` ($load\_scale \times clearing\_time$, representing disturbance kinetic impulse) improved Random Forest from missing instabilities under heavy loads to **100% accuracy and 0 missed blackouts**.
- **Neural/Linear Scaling Vulnerability (Covariate Shift):** Applying `StandardScaler` to `dist_to_gen` when Bus 39 has an unseen extreme boundary ($dist=0$, while training set was $\{1, 2\}$) caused continuous linear models (Logistic Regression, MLP) to suffer severe boundary inversion. In contrast, tree models (RF, XGBoost) split on ordinal ranks and remained robust.
- **Architectural Lesson:** Topological features for continuous models require min-max bounded scaling $[0, 1]$ or ordinal embedding rather than unbounded Gaussian normalization when evaluated on extreme unseen graph boundaries.

