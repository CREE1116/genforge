# study: Angle Distribution in Oblique Child Directions

This study investigates the distribution of angles between parent split directions ($\mathbf{w}_p$) and child candidate directions ($\mathbf{r}$) to explore if **orthogonality** is a cause of performance improvement or simply a geometric consequence of high-dimensional search spaces.

---

## 1. Theoretical Analysis & Mathematical Simulation

In a $d$-dimensional subspace, let $\mathbf{w}_p$ be a unit parent vector. If we draw a random Gaussian vector $\mathbf{r} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$, the square of the cosine of the angle $\theta$ between them follows a Beta distribution:
$$\cos^2 \theta \sim \text{Beta}\left(\frac{1}{2}, \frac{d-1}{2}\right)$$

For large $d$, the expected value of $\cos^2 \theta$ is $1/d$, and the expected absolute cosine is approximately $1/\sqrt{d}$.

We ran a simulation drawing 10,000 random vectors across different dimensions to measure the concentration of angles:

| Dimension $d$ | Expected $|\cos \theta|$ | Mean Angle (degrees) | Std Dev Angle |
|:---:|:---:|:---:|:---:|
| **2** | 0.6308 | 45.4° | 26.19° |
| **4** | 0.4231 | 63.3° | 18.60° |
| **8** | 0.2896 | 72.7° | 12.67° |
| **16** | 0.2022 | 78.2° | 8.78° |
| **30** | 0.1453 | 81.6° | 6.33° |
| **50** | 0.1133 | 83.5° | 4.94° |
| **100** | 0.0808 | 85.4° | 3.49° |

### Key Insight:
As dimension increases, the mass of the sphere concentrates around the equator. In 16 dimensions (a typical feature subspace size in OQBoost), **completely random directions are naturally $78.2^\circ$ orthogonal to the parent.** At 30 dimensions, they are $81.6^\circ$ orthogonal.

---

## 2. Empirical Analysis in OQBoost

We trained `OQBoostResearch` on a synthetic classification dataset ($D=30$) over 50 boosting rounds and analyzed the angles of the winning inherited candidate directions (`inherit_O`) relative to their parent split directions:

### A. `random` Strategy ($\beta = 0.0$, completely random in the subspace)
- **Inherited Candidate Win Rate**: **$3.6\%$** (26/721 splits)
- **Angle of Winning Directions**: Mean = **$74.92^\circ$** ($\pm 11.57^\circ$), Range = $[50.4^\circ, 89.2^\circ]$

### B. `full` Strategy ($\beta = 1.0$, strictly orthogonalized)
- **Inherited Candidate Win Rate**: **$3.5\%$** (25/718 splits)
- **Angle of Winning Directions**: Mean = **$90.00^\circ$** ($\pm 0.00^\circ$), Range = $[90.0^\circ, 90.0^\circ]$

---

## 3. Discussion & Scientific Conclusions

1. **Orthogonality as a Symptom (Phenomenon)**:
   The fact that the winning rate of inherited candidates is virtually identical between completely random ($3.6\%$) and strictly orthogonalized ($3.5\%$) indicates that enforcing exact $90^\circ$ projection does not alter the optimization dynamics. Random selection inside the subspace naturally results in directions that are $\sim 75^\circ$ orthogonal.
   
2. **Diversity is the True Driver**:
   The reason "orthogonal" inheritance (Strategy O) beats "mutate" (Strategy A/B) is because:
   - **Strategy A** forces the child to remain in a narrow $11^\circ$ cone around the parent direction. This severely restricts diversity.
   - The parent split has already partitioned the space along its normal vector, meaning the gradient signal in the direction of the parent is near zero.
   - Opening the search space to the entire subspace (whether exactly orthogonal or completely random) yields the same performance benefit because it allows the model to explore directions that still contain signal.
