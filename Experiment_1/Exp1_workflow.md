## Experiment 1 Workflow
v1 → Initial 2-qubit (4-class) hybrid QNN baseline
v2 → Increased sample size (1000 → 2000) → negligible impact
v3 → Modified observables → significant performance improvement
v4 → Increased circuit depth → no immediate benefit (undertrained)
v4b → Extended training (400 epochs) → depth becomes beneficial
v4c → Extended training (600 epochs) → delayed convergence observed
v5 → Improved observables → more efficient performance gains
v5b → Observables vs depth converge to similar range (~70–75%)
v5c → Extended training (600 epochs) → apparent ceiling ~80%
v4d → Extended training (1000 epochs) → depth reaches ~87–88%
v5d → Extended training (1000 epochs) → observables reach ~88–90%
v6 → Combined design (observables + depth) + extended training (1500 epochs) → ~92–93%

→ Final Insight: Early performance limits were caused by insufficient training, not architectural constraints.

## Factor contribution
  - Performance is governed by the interaction between circuit expressivity and optimization time, 
  rather than either factor independently.

| Factor        | Alone               | Combined |
| ------------- | ------------------- | -------- |
| Depth         | Helps (slowly)      | ✔️       |
| Observables   | Helps (efficiently) | ✔️       |
| Training time | Critical            | ✔️       |

## Tuning Attributes Not Fully Explored

- **Optimizer tuning**  
  - Learning rate  
  - Weight decay  

- **Sample size**  
  - Increased from 1000 → 2000 with minimal impact  
  - Further scaling may still provide marginal gains  

- **Quantum circuit design**  
  - Alternative gate combinations  
  - Different entanglement patterns  

- **Observable design**  
  - Additional interaction terms (e.g., ZZ, XX, XY)  
  - Higher-order or composite measurements  

## Key Finding (Final)

- Model performance is highly sensitive to observable design, which directly affects feature 
extraction from the quantum state
- Increasing circuit depth improves performance, but:
  - requires significantly longer training to converge
  - introduces a more complex optimization landscape
- Observable improvements:
  - accelerate learning (stronger early performance)
  - improve training efficiency
- With sufficient training (≥1000 epochs):
  - both approaches (depth and observables) converge to ~88–90% accuracy
- The previously observed ~80% ceiling was not a true representational limit,
but rather a result of premature convergence assumptions

## Implications

- Further improvements are unlikely to come from:
  - minor observable adjustments alone
  - shallow training regimes
- Instead, meaningful improvements are more likely to come from:
  - increasing qubit count (to evaluate scaling behavior)
  - improving convergence efficiency (optimizer tuning, learning rate schedules)
  - hybrid feature engineering (classical preprocessing + quantum encoding)

## Practical Insight

- The 2-qubit system already achieves strong performance (~90%+)
- Additional qubits should be evaluated in terms of:
  - efficiency (training time vs gain)
  - scalability
  - robustness
  - not just raw accuracy

  ## - These results suggest that optimization dynamics play a critical role in unlocking the representational capacity of low-qubit QNNs.