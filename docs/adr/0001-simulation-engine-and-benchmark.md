# ADR-0001: Selection of ANDES and IEEE 39-Bus Benchmark

## Status
**Accepted**

## Context
Transient stability assessment requires accurate numerical solution of power system Differential-Algebraic Equations (DAEs) under large-disturbance dynamics (three-phase faults, generator exciter responses, line trips). Commercial tools (PSS/E, DIgSILENT PowerFactory) require expensive proprietary licenses, hindering reproducibility and headless programmatic scripting on personal machines.

## Decision
1. **Simulation Engine:** We selected **ANDES** (`andes>=1.8.0`), an open-source, Python-native hybrid symbolic-numeric power system simulator.
2. **Benchmark System:** We locked in the **IEEE 39-bus New England system** (`ieee39/ieee39_full.xlsx`) with 10 synchronous machines modeled with 6th-order GENROU dynamics and IEEE-T1 exciter models.

## Consequences
- **Positive:**
  - Fully scriptable in Python with multi-processing parallel execution.
  - Zero licensing constraints, enabling reproducible research on standard developer machines.
  - Native support for power flow, time-domain transient simulation, and event injection (faults, line tripping).
- **Negative / Risks:**
  - ANDES file I/O writes temporary `.lst` and `.npz` files relative to current working directory, requiring worker-level `tempfile.TemporaryDirectory` sandboxing during parallel processing.
