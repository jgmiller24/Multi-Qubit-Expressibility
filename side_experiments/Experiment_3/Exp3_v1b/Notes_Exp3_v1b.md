
## Experiment 3 (v1b):  4-Qubit Scaling Evaluation - Adjusted Observables to enrich feature extraction + extended training (100 epochs)

##

##

## Observables
- More diverse observables to facilitate feature seperation. 
- Included one higher-order term (zzz) one x-axis pairwise (xx) and one y-axis pairwise (yy)

## Extended training (100 epochs)
- Increased training to allow proper convergence

### Class 8 Behavior
- Class 8 exhibited extremely low recall but very high precision when predicted. This indicates that the model 
learned a narrow but correct decision region for digit 8, while most true 8 samples were projected into 
nearby classes such as 2 and 5.

### Interpretation
- This suggests that the current observable basis does not provide sufficient feature separation for class 8. The 
issue is not random misclassification, but an overly restrictive representation of class 8 within the measured 
quantum feature space.

### Observable Insight
- The current basis is dominated by local and pairwise Z correlations, with limited X/Y diversity and limited 
higher-order terms involving later qubits. This likely restricts the model’s ability to capture the looped and 
symmetric structure associated with digit 8.

