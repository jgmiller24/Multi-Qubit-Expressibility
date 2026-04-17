## Experiment 2 (v3): Extended Training Control (Same Circuit as v2, 600 → 1000 Epochs)

### Circuit Configuration:
- 3-qubit architecture (identical to v2)
- Two rotation layers with one entanglement block:
  - Layer 1: RY rotations (3 parameters)
  - Entanglement: q0 → q1 → q2 (ladder CNOT)
  - Layer 2: RX rotations (3 parameters)
- Total parameters: **6** (unchanged from v2)
- Measurement:
  - Same 8 observables as v1/v2:
    - Z(0), Z(1), Z(2)
    - Z(0)Z(1), Z(1)Z(2), Z(0)Z(2)
    - X(0)X(1), X(1)X(2)


### Objective:
Isolate the effect of training time by holding the v2 circuit fixed and extending epochs
from 600 to 1000. Determines whether v2's ~45–46% accuracy was training-time limited,
or whether the circuit has already converged — establishing a clean baseline before
adding circuit depth in v4.


### Run History:

| Attempt | sample_count | Outcome |
|---------|-------------|---------|
| v3 initial | 4000 | OOM-killed by OS at epoch 446/1000 after 3h 35m |
| v3 current | 2000 | Reduced to allow full 1000-epoch CPU run to complete |

**Memory optimization note:** The 4000-sample run was terminated by the OS memory manager at epoch 446/1000 (after ~3 hours 35 minutes of CPU training). `sample_count` has been reduced to 2000 to lower per-epoch memory pressure and allow the full 1000-epoch run to complete without hitting the OOM limit.


### Results Summary:
- Test accuracy: [TBD]
- Comparison to v2 (~45–46%): [TBD]
- Convergence behavior: [TBD]


### Key Observations:
- [TBD after run]


### Interpretation:
- [TBD after run]


### Comparative Insight (v2 → v3):
| Change | Impact |
|------|--------|
| 600 → 1000 epochs | [TBD] |
| Circuit (unchanged) | Controlled variable |
| Parameters (unchanged, 6) | Controlled variable |


### Conclusion:
- [TBD after run]


### Takeaway:
> [TBD after run]
