
## Experiment 2 (v3b) - Observable Variation - YY interaction

## Modification
- Replaced final observable:
    - From: spin.x(1) * spin.x(2)
    - To: spin.y(1) * spin.y(2)

## Results
- Test accuracy ~60.7% (Down from ~64-65%)
- Severe degradation in class-specific performance
    - Class 8 complete collapse (0% recall, 0% precision)

## Key Observations
- Replacing XX with YY altered the feature space sgnificantly
- Certain classes (e.g., digit 8) rely on correlation structures captured by XX interactions.
- YY interactions alone were insufficient to represent these features.

## Interpretation
- Observable choice directly immpacts class separability
- Different observables encode different quantum correlations
- Feature expressivity is highly sensitive to measurement basis

## Conclusion
- Observables are not interchangeable
- A richer observable set (e.g., combining XX and YY) may improve robustness
- Observable design is a crutial factor in Qnn performance.