# OQBoost Research Notes — Bounded Energy Loss Framework

This document organizes OQBoost's theoretical research based on the **Bounded Energy Loss** framework, reflecting the discontinuity of tree splits (Indicator Function) for use in papers and reports.

---

## 1. Background: Limits of Previous Hypotheses and Corrections

- **Flaw of the Hard Orthogonality Hypothesis:** Just because an ancestor node has split on axis $w_i$, the linear component of that axis ($\langle v_G, w_i \rangle$) does not completely disappear ($=0$) in child nodes. This is because a tree split is a space partitioning of the form $\mathbb{I}(w_i^T X < \theta)$, not a linear projection. Therefore, forcing strict orthogonalization on child nodes is a theoretical error that overly restricts degrees of freedom (leading to capacity exhaustion).
- **Solution (Bounded Energy Loss Framework):** Instead of assuming absolute gradient exhaustion, we derive a theorem showing that the split gain loss rate is bounded by the residual gradient energy remaining within the lineage subspace ($\epsilon$).

---

## 2. Energy Decomposition-based Gain Preservation Theorem

### [Theorem]

> Even under the constraint of the lineage subspace $S$, the theoretical upper bound of the optimal gain is mostly preserved as long as the residual gradient energy does not concentrate inside $S$.

### [Proof]

#### Step 1: Orthogonal Decomposition of the Gradient Direction Vector

Let the lineage subspace formed by the ancestor split axes up to the current node be $S = \text{span}(W) \subset \mathbb{R}^D$, and let its orthogonal complement be $S^\perp$. The residual gradient direction vector $v_G = X^T g \in \mathbb{R}^D$ at the current node is uniquely decomposed as:

$$v_G = v_\parallel + v_\perp \quad (v_\parallel \in S, \quad v_\perp \in S^\perp)$$

- $v_\parallel$: The **residual linear energy** that ancestor axes failed to resolve.
- $v_\perp$: The **new orthogonal energy** that ancestors have not touched.

#### Step 2: Derivation of Maximum Gain under Constraints

Let the first-order Taylor approximation objective for split gain optimization by selecting the projection axis $w$ ($\|w\|_2=1$) in oblique trees be defined as $f(w) = (w^T v_G)^2$.

1. **Unconstrained Optimization (Unconstrained Upper Bound):**
   In the absence of constraints, the absolute upper bound is achieved by the Cauchy-Schwarz inequality when $w^* = \frac{v_G}{\|v_G\|_2}$.

$$f_{\text{free}} = \max_{\|w\|_2=1} (w^T v_G)^2 = \|v_G\|_2^2$$

2. **Lineage-Orthogonal Constrained Optimization:**
   If we impose the constraint that the new axis must be orthogonal to the lineage subspace ($w \in S^\perp$), we have $w^T v_\parallel = 0$. The objective function simplifies to:

$$f(w) = (w^T (v_\parallel + v_\perp))^2 = (w^T v_\perp)^2$$

Since $v_\perp \in S^\perp$, the optimal axis maximizing this is $w^* = \frac{v_\perp}{\|v_\perp\|_2}$, yielding the maximum gain $f^*$:

$$f^* = \max_{\|w\|_2=1, w \in S^\perp} (w^T v_\perp)^2 = \|v_\perp\|_2^2$$

#### Step 3: Derivation of the Gain Retention Ratio

The ratio of the gain preserved by the lineage-orthogonal axis ($f^*$) compared to the unconstrained upper bound ($f_{\text{free}}$) is:

$$\frac{f^*}{f_{\text{free}}} = \frac{\|v_\perp\|_2^2}{\|v_G\|_2^2} = \frac{\|v_G\|_2^2 - \|v_\parallel\|_2^2}{\|v_G\|_2^2} = 1 - \frac{\|v_\parallel\|_2^2}{\|v_G\|_2^2}$$

Defining the ratio of energy leaked into the lineage subspace to the total energy as $\epsilon = \frac{\|v_\parallel\|_2^2}{\|v_G\|_2^2}$, the final gain preservation equation holds:

$$f^* = (1 - \epsilon) f_{\text{free}}$$

---

## 3. Connection with OQBoost Architectural Advantages (EXP-D Isomorphism)

This theorem explains why avoiding the lineage subspace is safe and effective, connecting it with the experimental results (`EXP-D`).

- **Early Stage of Boosting (Leaked Energy $\epsilon$ is Non-Zero):**
   In early rounds, residual linear signals ($v_\parallel$) of existing axes may persist, meaning $\epsilon$ can be temporarily large. However, in this phase, the **256-bin Axis Scan** slot wins the tournament with overwhelming gain. Thus, even if the lineage-orthogonal slot takes a loss, it does not hurt overall tree quality.
- **Late Stage of Boosting (Energy Depletion $\epsilon \to 0$):**
   As rounds progress and the residual error becomes complex, the linear energy within the existing lineage subspace is fully depleted ($\|v_\parallel\| \to 0$). Therefore, in late-stage nodes where oblique axes are truly necessary, $\epsilon \approx 0$ is achieved, leading to $f^* \approx f_{\text{free}}$ (0% gain loss).

---

[Korean Version (한국어 버전)](research.md)
