
## Experiment 1 (v5c): Modified Observables – EPOCHS = 600

### Results Summary:
- Test accuracy reached ~80–81%, slightly exceeding v4c (~78–79%)
- Training remained stable with continued but diminishing improvements
- Performance gains beyond ~500 epochs were minimal\

### Key Observation:
- Improved observables provide slightly better performance than depth-based approach,
  but both converge to a similar accuracy range (~75–80%)

### Interpretation:
- Observable design improves feature extraction efficiency, particularly in earlier training stages
- However, extended training allows deeper circuits (v4c) to achieve comparable performance
- Both approaches ultimately converge to a similar performance ceiling

### Per-Class Behavior:
- Balanced performance across all classes
- Strong performance on digits with clearer structure (e.g., '6', '4')
- Remaining errors reflect intrinsic overlap between digit representations

### Conclusion:
- Observable improvements and increased circuit depth both enhance model performance
- However, neither approach surpasses the ~80% accuracy ceiling

- This strongly indicates:
  - Performance is limited by the representational capacity of the 2-qubit system

### Key Insight:
- The limiting factor is not optimization, training duration, or architecture design
- The limiting factor is the size of the quantum state space (2 qubits)

### Final Outcome (Experiment 1):
- Multiple architectural and optimization strategies converge to the same performance range
- A clear representational ceiling has been identified

### Next Steps:
- Increase qubit count (3-qubit system) to expand Hilbert space
- Evaluate whether additional qubits improve class separability and raise performance ceiling