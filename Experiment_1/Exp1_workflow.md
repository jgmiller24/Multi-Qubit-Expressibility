## Experiment 1 Workflow

- v1  → Initial 2-qubit (4-class) hybrid QNN baseline
- v2  → Increased sample size (1000 → 2000) → negligible impact
- v3  → Modified observables → significant performance improvement
- v4  → Increased circuit depth → no immediate benefit
- v4b → Extended training (400 epochs) → depth becomes beneficial
- v5  → Improved observables → more efficient performance gains
- v5b → Both approaches (depth + observables) converge to similar performance ceiling
- v4c → Further training (600 epochs) → evaluate convergence behavior
- v5c → Further training (600 epochs) → confirm ceiling consistency

## Key Finding: (prelim)
- Performance saturates in the ~70–75% range across configurations

## Conclusion: (prelim)
- 2-qubit representational limit identified