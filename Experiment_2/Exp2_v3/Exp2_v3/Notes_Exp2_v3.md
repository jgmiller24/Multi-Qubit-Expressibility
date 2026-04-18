
## Experiment 2 (v3): Increased 3-Qubit Circuit Depth

### Circuit Configuration:
- 3-qubit architecture
- Three rotation layers:
  - Layer 1: RY (3 parameters)
  - Entanglement: q0 → q1 → q2
  - Layer 2: RX (3 parameters)
  - Entanglement: q0 → q1 → q2
  - Layer 3: RY (3 parameters)
- Total parameters: 9
- Measurement:
  - Same 8 observables (aligned with 8-class task)


### Objective:
Evaluate whether further increasing circuit depth improves class separability and overall performance in 8-class classification.


### Results Summary:
- Test accuracy: ~64–65%
- Significant improvement over v2 (~46%)
- All classes now receive predictions (no collapse)


### Key Observations:
- Performance improves consistently with increased circuit depth
- Training curves show smooth convergence with no instability
- No clear plateau reached at 1000 epochs


### Per-Class Behavior:
- Strong performance for several classes (>75% recall)
- Some classes remain difficult (e.g., class '5' equivalent)
- Indicates uneven class separability despite improved capacity


### Interpretation:
- Additional circuit depth significantly enhances feature extraction
- 3-qubit system demonstrates higher usable capacity when sufficiently parameterized
- Remaining errors likely stem from:
  - overlapping class representations
  - limited observable expressivity


### Comparative Insight (v1 → v3):
- v1: shallow circuit → near-random performance
- v2: moderate depth → partial learning (~46%)
- v3: deeper circuit → strong learning (~65%)


### Conclusion:
- Circuit depth is a critical factor for scaling quantum model performance
- Increased qubit count + sufficient depth enables meaningful multi-class classification
- However, performance gains are still limited by representational and observable constraints

### Takeaway:
> **Effective scaling requires both increased Hilbert space (qubits) and sufficient circuit expressivity (depth)**