## Experiment 1 Workflow

- v1  → Initial 2-qubit (4-class) hybrid QNN baseline  
- v2  → Increased sample size (1000 → 2000) → negligible impact  
- v3  → Modified observables → significant performance improvement  
- v4  → Increased circuit depth → no immediate benefit  
- v4b → Extended training (400 epochs) → depth becomes beneficial  
- v5  → Improved observables → more efficient performance gains  
- v5b → Depth vs observables converge to similar performance ceiling  
- v4c → Extended training (600 epochs) → evaluate convergence behavior  
- v5c → Extended training (600 epochs) → confirm ceiling consistency  
- v6  → Extended training (1000 epochs) → final validation of 2-qubit performance ceiling  

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

## Key Finding (Preliminary)

- Model performance is **highly sensitive to observable design**, which directly impacts feature extraction from the quantum state  
- Increased circuit depth improves performance **only when sufficient training is provided**  
- With extended training, both approaches (depth vs observables) **converge to a similar performance range (~80%)**  
- This suggests the presence of a **representational bottleneck in the 2-qubit system**, rather than an optimization limitation  

→ Primary limitation is likely:
- Restricted Hilbert space (2 qubits)  
- Limited capacity for class separability in 4-class classification  

## Implication

Further improvements are unlikely to come from:
- Additional training time  
- Increased circuit depth alone  
- Minor observable adjustments  

Instead, meaningful gains will likely require:
- Increasing qubit count (higher-dimensional state space)  
- Enhancing classical feature extraction in the hybrid pipeline  