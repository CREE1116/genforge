# OQBoost

**Gradient-boosted oblique decision trees with hereditary projection evolution.**

OQBoost replaces axis-aligned splits with gradient-guided oblique hyperplanes that are inherited and mutated from parent nodes — exploiting the geometric structure of the data without expensive numerical optimization.

<p align="center">
  <img src="docs/diverse_boundaries.png" alt="OQBoost Decision Boundaries" width="800">
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![CI](https://github.com/cree1116/oqboost/actions/workflows/ci.yml/badge.svg)](https://github.com/cree1116/oqboost/actions)
[![PyPI](https://img.shields.io/pypi/v/oqboost)](https://pypi.org/project/oqboost/)

---

## Key Properties

| Feature | OQBoost |
|---------|---------|
| Split type | Oblique (linear projection of multiple features) |
| Direction finding | GG-SRP: gradient-guided sparse random projection |
| Inheritance | Parent weight inheritance with depth-decayed mutation |
| Missing values | Native — NaN handled via mean imputation in C++ |
| Categorical features | Native — gradient-rank encoding per round |
| API | scikit-learn compatible |
| Backend | Compiled C++ with OpenMP parallelism |

---

## Install

```bash
pip install oqboost
```

Pre-compiled wheels are available for macOS (arm64, x86_64) and Linux (x86_64).
On unsupported platforms, `clang++` or `g++` is required to compile from source.

---

## Quickstart

```python
from oqboost import OQBoostClassifier

clf = OQBoostClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
)

clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
clf.predict(X_test)
clf.predict_proba(X_test)
```

### Native NaN support

```python
import numpy as np
X_train[50, 3] = np.nan   # NaN anywhere is fine
clf.fit(X_train, y_train)
clf.predict(X_test)        # NaN → column-mean imputation at inference
```

### Native categorical support

```python
import pandas as pd

df = pd.read_csv("data.csv")
X = df.drop(columns=["target"])
y = df["target"]

# auto-detects pandas Categorical / object columns
clf = OQBoostClassifier(n_estimators=500)
clf.fit(X, y)

# or specify explicitly
clf = OQBoostClassifier(cat_features=["city", "product"])
clf.fit(X, y)
```

### Early stopping

```python
clf = OQBoostClassifier(n_estimators=2000, early_stopping_rounds=50)
clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
print(f"Stopped at {clf.get_n_trees()} trees")
```

### Save / load

```python
clf.save("model.joblib")
from oqboost import load_model
clf2 = load_model("model.joblib")
```

---

## Benchmark Results

All benchmarks: 80/20 train-test split, 3 repetitions, mean ± standard deviation reported. All models (XGBoost, LightGBM, CatBoost, OQBoost) were hyperparameter-tuned using Optuna for 50 trials.

### Main Benchmark Table

| Dataset | Model | Accuracy | Bal. Acc. | F1 Macro | Log Loss | Train (s) | Infer (s) |
|---------|-------|----------|-----------|----------|----------|-----------|-----------|
| **Adult** | XGBoost | 0.8736±0.0033 | 0.7990±0.0065 | 0.8159±0.0055 | 0.2760±0.0018 | **0.83±0.10** | **0.01±0.00** |
| | LightGBM | **0.8746±0.0041** | **0.8002±0.0079** | **0.8173±0.0069** | 0.2755±0.0021 | 2.80±0.27 | 0.06±0.00 |
| | CatBoost | 0.8745±0.0040 | 0.7963±0.0065 | 0.8156±0.0063 | **0.2737±0.0018** | 7.83±0.86 | 0.05±0.00 |
| | OQBoost | 0.8717±0.0023 | 0.7985±0.0073 | 0.8141±0.0050 | 0.2768±0.0025 | 4.16±0.30 | 0.04±0.01 |
| **Credit Default** | XGBoost | 0.8206±0.0018 | 0.6584±0.0015 | 0.6836±0.0022 | 0.4280±0.0010 | **0.19±0.01** | **0.00±0.00** |
| | LightGBM | **0.8223±0.0019** | 0.6585±0.0018 | 0.6844±0.0025 | **0.4239±0.0012** | 5.56±0.32 | 0.07±0.01 |
| | CatBoost | 0.8220±0.0010 | **0.6603±0.0021** | **0.6859±0.0024** | 0.4274±0.0004 | 0.56±0.08 | **0.00±0.00** |
| | OQBoost | 0.8211±0.0025 | 0.6569±0.0057 | 0.6824±0.0064 | 0.4251±0.0006 | 0.75±0.03 | **0.00±0.00** |
| **Give Me Credit** | XGBoost | 0.7400±0.0312 | 0.6381±0.0349 | 0.6488±0.0398 | 0.5093±0.0266 | 0.16±0.03 | **0.00±0.00** |
| | LightGBM | 0.7417±0.0321 | 0.6250±0.0360 | 0.6341±0.0420 | 0.5051±0.0320 | 0.38±0.20 | **0.00±0.00** |
| | CatBoost | **0.7650±0.0173** | **0.6750±0.0167** | **0.6888±0.0196** | 0.4951±0.0382 | **0.12±0.02** | **0.00±0.00** |
| | OQBoost | 0.7600±0.0397 | 0.6651±0.0791 | 0.6719±0.0867 | **0.4940±0.0566** | 0.18±0.02 | **0.00±0.00** |
| **CoverType** | XGBoost | 0.9704±0.0008 | 0.9392±0.0042 | 0.9460±0.0034 | 0.0789±0.0017 | **74.01±0.27** | 4.06±0.08 |
| | LightGBM | 0.9704±0.0008 | 0.9397±0.0052 | 0.9466±0.0045 | 0.0823±0.0021 | 286.03±2.17 | 35.09±0.18 |
| | CatBoost | 0.9588±0.0005 | 0.9303±0.0038 | 0.9371±0.0038 | 0.1171±0.0017 | 130.84±0.51 | **0.19±0.01** |
| | OQBoost | **0.9752±0.0008** | **0.9495±0.0032** | **0.9543±0.0027** | **0.0762±0.0013** | 219.58±0.85 | 2.66±0.08 |
| **Higgs** | XGBoost | 0.7304±0.0052 | 0.7291±0.0054 | 0.7293±0.0053 | 0.5259±0.0045 | **5.74±0.81** | 0.08±0.01 |
| | LightGBM | **0.7319±0.0037** | **0.7307±0.0037** | **0.7309±0.0037** | 0.5255±0.0044 | 33.07±5.15 | 0.46±0.08 |
| | CatBoost | 0.7293±0.0055 | 0.7279±0.0055 | 0.7281±0.0055 | 0.5296±0.0043 | 11.91±2.24 | **0.01±0.00** |
| | OQBoost | 0.7311±0.0043 | 0.7300±0.0043 | 0.7301±0.0043 | **0.5243±0.0043** | 45.69±7.71 | 0.74±0.11 |
| **Rotated Synth.** | XGBoost | 0.9758±0.0014 | 0.9758±0.0015 | 0.9758±0.0014 | 0.0819±0.0023 | **6.46±0.29** | 0.11±0.02 |
| | LightGBM | 0.9763±0.0017 | 0.9763±0.0017 | 0.9763±0.0017 | 0.0835±0.0023 | 17.72±0.58 | 0.33±0.08 |
| | CatBoost | 0.9772±0.0019 | 0.9772±0.0019 | 0.9772±0.0019 | 0.0818±0.0034 | 20.75±9.69 | **0.02±0.01** |
| | OQBoost | **0.9795±0.0013** | **0.9795±0.0013** | **0.9795±0.0013** | **0.0724±0.0027** | 32.08±11.30 | 0.51±0.12 |

### Highlights

- **Outstanding on CoverType:** OQBoost achieves **0.9495±0.0032** Balanced Accuracy (outperforming LightGBM's 0.9397 and XGBoost's 0.9392) and the lowest Log Loss of **0.0762** (vs XGBoost's 0.0789 and CatBoost's 0.1171).
- **Superior on Rotated Synthetic:** OQBoost achieves **0.9795±0.0013** Balanced Accuracy and **0.0724±0.0027** Log Loss, outperforming all baselines (CatBoost 0.9772, LightGBM 0.9763, XGBoost 0.9758). This validates the oblique split advantage on rotated decision boundaries.
- **Lowest Log Loss on Give Me Credit & Higgs:** OQBoost achieves the lowest Log Loss on Give Me Credit (**0.4940±0.0566** vs CatBoost's 0.4951) and Higgs (**0.5243±0.0043** vs LightGBM's 0.5255).
- **Strong Performance on Give Me Credit & Adult:** OQBoost performs competitively, scoring **0.6651±0.0791** Balanced Accuracy on Give Me Credit (outperforming XGBoost's 0.6381 and LightGBM's 0.6250) and **0.7985±0.0073** on Adult Income.

### Figure 1 — Balanced Accuracy Comparison

![Balanced Accuracy](benchmark/results/figures/fig2_balanced_accuracy.png)

### Figure 2 — Performance vs Training Cost

![Perf vs Cost](benchmark/results/figures/fig5_perf_vs_cost.png)

### Figure 3 — Rotation Robustness

![Rotation Robustness](benchmark/results/figures/fig6_rotation_robustness.png)

### Figure 4 — Missing Value Robustness

![Missing Value](benchmark/results/figures/fig7_missing_value.png)

### Figure 5 — Categorical Cardinality Robustness

![Categorical](benchmark/results/figures/fig8_categorical.png)

---

## Algorithm

OQBoost uses three-stage hereditary projection evolution:

**Stage 1 — GG-SRP (Gradient-Guided Sparse Random Projection)**  
Features are sampled with probability proportional to SIS gradient-importance scores. Each selected feature gets a weight sign aligned with the steepest gradient descent direction. No Gram matrix, no linear system — $O(D)$ per node.

**Stage 2 — Parent Weight Inheritance**  
Child nodes inherit their parent's split direction and apply two mutation strategies:
- *Strategy A:* axis-maintaining noise (tilt the boundary by ±10%)
- *Strategy B:* new-axis borrowing (add a high-correlation feature not in the parent's support)

**Stage 3 — Global-Local Crossover + Depth Decay**  
- *Strategy C:* random blend of the current parent direction with a globally top-performing direction from the ring buffer (last 32 rounds)
- *Depth decay:* mutation strength decreases as $1/\sqrt{1 + d}$ — wide exploration at shallow depth, fine-tuning at deep nodes

See [`docs/algorithm.md`](docs/algorithm.md) for the full derivation and [`docs/THEORY.md`](docs/THEORY.md) for theoretical insights and experiment logs.

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_estimators` | 500 | Number of boosting rounds |
| `learning_rate` | 0.05 | Shrinkage per tree |
| `max_depth` | 6 | Leaf budget = 2^max_depth (64 leaves, matches XGBoost/CatBoost) |
| `reg_lambda` | 1.0 | L2 leaf regularization |
| `subsample` | 0.8 | Row fraction per tree |
| `early_stopping_rounds` | 50 | Stop if class-weighted val loss stagnates |
| `cat_features` | None | Categorical column names or indices |
| `class_weight` | None | Reweight by inverse class frequency (e.g. "balanced") |
| `inherited_rp_ratio` | 1.0 | Fraction of candidates from inheritance + cache |
| `mutation_rate` | 0.1 | Base noise scale for axis mutations |
| `mutation_strength` | 0.5 | Base weight for new-axis borrowing |
| `pobs` | False | Inject Haar-orthogonal POBS candidates into every node's tournament |
| `random_state` | None | Seed |
| `verbose` | False | Print per-round metrics |

---

## Running Benchmarks

```bash
cd benchmark

# Run all (takes ~hours for Higgs + CoverType)
python run_all.py

# Skip large datasets
python run_all.py --skip higgs covertype

# Individual benchmarks
python adult.py
python rotated_synthetic.py
python missing_value_robustness.py
python categorical_robustness.py

# Generate figures from completed results
python generate_figures.py

# Generate summary table (results/summary.md)
python generate_tables.py
```

Large datasets require manual download:
- **HIGGS**: https://archive.ics.uci.edu/dataset/280 → `benchmark/data/HIGGS.csv.gz`
- **Give Me Some Credit (Kaggle)**: https://www.kaggle.com/competitions/GiveMeSomeCredit → `benchmark/data/cs-training.csv`

---

## Repository Structure

```
oqboost/
├── src/oqboost/
│   ├── __init__.py
│   ├── _classifier.py      # OQBoostClassifier
│   ├── _oqboost.py        # C bindings + OQBoostTree, OQBoostContext
│   └── _ext/
│       ├── oqboost.cpp    # full engine: boosting round, routing, serialization
│       ├── oqboost_types.h # OQTree structure
│       ├── oqboost_core.h # Shared constants and helpers
│       └── liboqboost.dylib / .so / .dll  (compiled)
├── benchmark/
│   ├── *.py                # Per-dataset benchmark scripts
│   ├── _utils.py           # Shared train/eval utilities
│   ├── generate_figures.py
│   ├── generate_tables.py
│   ├── run_all.py
│   └── results/
│       ├── csv/            # Raw results
│       ├── figures/        # Generated plots
│       └── summary.md
├── docs/
│   ├── algorithm.md        # Theory and derivations
│   ├── api.md              # Full API reference
│   └── quickstart.md
├── tests/
└── pyproject.toml
```

---

## License

[MIT](LICENSE) — Copyright (c) 2025 cree1116
