# ADR-0002: Stability Ground Truth Labeling via COI 180° Rule

## Status
**Accepted**

## Context
Transient rotor angle stability determines whether synchronous generators maintain synchronism after a disturbance. In a multi-machine system, individual rotor angles drift together; evaluating absolute angle $\delta_i(t)$ is insufficient because the whole system may shift its synchronous speed.

## Decision
We label stability using the **Center of Inertia (COI)** reference frame:
$$\delta_{\text{COI}}(t) = \frac{\sum_{i=1}^{m} H_i \delta_i(t)}{\sum_{i=1}^{m} H_i} \approx \frac{1}{m} \sum_{i=1}^{m} \delta_i(t)$$

A scenario is classified as **Unstable ($y = 0$)** if:
1. The maximum deviation of any generator from the COI exceeds $180^\circ$ ($|\delta_i(t) - \delta_{\text{COI}}(t)| > 180^\circ$) at any point during the transient simulation window ($t \in [0, 20\,\text{s}]$).
2. The ANDES numerical solver encounters divergence, step-size breakdown, or terminates early ($t_{\text{final}} < 19.5\,\text{s}$).

Otherwise, the scenario is labeled **Stable ($y = 1$)**.

## Alternatives Considered
- **Pairwise Generator Angle Difference ($|\delta_i - \delta_j| > 180^\circ$ or $360^\circ$):** Computationally scales as $O(m^2)$, whereas COI deviation scales as $O(m)$ and better reflects whole-system separation.
- **Terminal Voltage Settling Threshold:** Fails when voltage recovers but rotor angle has already slipped poles.

## Consequences
- Produces physically meaningful binary labels validated against classical power system dynamics theory.
- Allows automated test coverage in `tests/test_stability.py`.
