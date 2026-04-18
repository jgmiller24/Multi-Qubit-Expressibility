
## Experiment 3: 4-qubit scaling evaluation - Initial 10-class test run (25 epochs)

## 

## Early Scaling Result 
- The model achieved ~51% test accuracy after 25 epochs, demonstrating stable learning behavior without class collapse.
- Per-class analysis revealed uneven performance, with certain digits (e.g., 5, 2, 4, 8) significantly underperforming. This 
indicates that while the circuit has sufficient capacity to model the task, the observable set limits feature separability for specific classes.
- This suggests that observable design plays a critical role in scaling hybrid QNN performance, particularly as class complexity increases.