## Experiment 3 Workflow: 4-Qubit Scaling and Observable Design

### Objective

Evaluate how scaling from 3 → 4 qubits impacts model performance in a 10-class MNIST setting, and determine whether performance limitations arise from circuit capacity or observable design.

---

## Phase 1 — Initial Scaling (Exp3_v1)

**Configuration**

* 4 qubits, 3-layer circuit
* 10-class classification (digits 0–9)
* 10 observables (primarily Z + pairwise terms)
* 25 epochs (short-run evaluation)

**Result**

* ~50% test accuracy after 25 epochs
* Stable learning behavior with **no class collapse**

**Key Observation**

* Significant **per-class imbalance** (notably digits 2, 4, 5, 8 underperforming)

**Interpretation**

* The circuit has sufficient capacity to model the task
* Performance limitations are driven by **poor feature separability**, not underfitting



---

## Phase 2 — Observable Expansion + Training (Exp3_v1b)

**Changes**

* Added observable diversity:

  * Introduced **XX, YY, and ZZZ terms**
* Increased training from 25 → 100 epochs

**Result**

* Improved overall accuracy
* Persistent **failure on digit 8**

  * Very low recall, but high precision when predicted

**Key Observation**

* Digit 8 forms a **narrow, correct decision region**
* Most true 8s are misclassified as nearby digits (2, 5)

**Interpretation**

* This is **not random error**
* It is a **representation bottleneck**

  * Observable space cannot adequately encode digit 8 structure

**Insight**

* Observable set is still biased toward:

  * Z-dominant correlations
  * Limited higher-order interactions across all qubits



---

## Phase 3 — Refined Observable Design (Exp3_v1c)

**Changes**

* Balanced observable set across:

  * Z, X, and Y bases
* Added richer correlations:

  * Pairwise + higher-order interactions
* Ensured **all qubits contribute to measurements**

**Result**

* Test accuracy ≈ **86.7%**
* Digit 8 recall improved from near-zero → **~84%**
* Strong per-class balance across most digits

**Additional Observation**

* Test/train gap (~11%) indicates **moderate overfitting**

**Interpretation**

* Observable redesign directly improved:

  * Feature richness
  * Class separability
* Previous failure modes were due to:

  * **Insufficient feature representation**, not circuit limitations

**Key Insight**

* Observable selection is a **primary driver of representational power** in hybrid QNNs
* Increasing qubits or depth alone is insufficient without proper measurement design



---

## Cross-Phase Conclusions

### 1. Scaling Alone Is Not Enough

* Moving from 3 → 4 qubits increased capacity
* But performance gains only materialized after improving observables

---

### 2. Observable Design Controls Feature Space

* Poor observables → class collapse / separability issues
* Balanced observables → strong recovery across all classes

---

### 3. Failure Modes Are Structured (Not Random)

* Digit 8 case showed:

  * Consistent misprojection into similar classes
  * Indicates geometric limitation in feature space

---

### 4. Training vs Representation

* Early experiments suggested undertraining
* Exp3 shows:

  * Even with sufficient training, **representation can still fail**
  * Observables define what the model *can learn*

---

### 5. Emerging Tradeoff

* Improved observables → better accuracy
* But also:

  * Increased model capacity → **overfitting risk**

---

## Final Takeaway

Hybrid QNN performance is governed by three interacting factors:

1. **Circuit capacity** (qubits + depth)
2. **Training dynamics** (optimizer, epochs)
3. **Observable design** (feature extraction)

> Among these, observable design acts as the **bottleneck and enabler** of effective learning.

---
