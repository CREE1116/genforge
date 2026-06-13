# OQBoost

**유전적 사영 진화를 기반으로 한 경사 부스팅 사선 결정 트리 (Gradient-boosted oblique decision trees with hereditary projection evolution)**

OQBoost는 기존의 축 정렬 분기(axis-aligned splits) 대신 그라디언트(gradient)에 의해 정렬되는 사선 하이퍼플레인(oblique hyperplanes) 분기를 수행합니다. 이 분기 방향들은 부모 노드로부터 상속 및 변이되는 유전적 파이프라인을 거치며, 값비싼 수치 최적화 없이도 데이터의 고유한 기하학적 구조를 효과적으로 캡처합니다.

<p align="center">
  <img src="docs/diverse_boundaries.png" alt="OQBoost 결정 경계" width="800">
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![CI](https://github.com/cree1116/oqboost/actions/workflows/ci.yml/badge.svg)](https://github.com/cree1116/oqboost/actions)
[![PyPI](https://img.shields.io/pypi/v/oqboost)](https://pypi.org/project/oqboost/)

---

## 핵심 특징 (Key Properties)

| 특징 | 설명 |
|------|------|
| **분기 타입 (Split type)** | 사선 분기 (Oblique split, 여러 피처들의 선형 사영 결합) |
| **방향 탐색 (Direction finding)** | GG-SRP: 그라디언트 유도형 희소 무작위 사영 |
| **상속 구조 (Inheritance)** | 부모 노드의 분기 방향 가중치를 상속하고 깊이에 비례해 감쇠하는 변이 적용 |
| **결측치 지원 (Missing values)** | 네이티브 지원 — C++ 내부에서 실시간 평균 대체로 NaN 처리 |
| **범주형 피처 지원 (Categorical features)** | 네이티브 지원 — 라운드별 그라디언트 랭크(gradient-rank) 타겟 인코딩 |
| **API 호환성** | scikit-learn API 완벽 호환 |
| **백엔드 엔진** | OpenMP 병렬 컴파일된 C++ 백엔드 |

---

## 설치 (Install)

```bash
pip install oqboost
```

macOS (arm64, x86_64) 및 Linux (x86_64) 환경을 위한 사전 빌드된 휠(wheel) 패키지가 배포됩니다.
지원되지 않는 플랫폼의 경우, 소스코드 컴파일을 위해 `clang++` 또는 `g++`가 필요합니다.

---

## 퀵스타트 (Quickstart)

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

### 네이티브 결측치(NaN) 지원

```python
import numpy as np
X_train[50, 3] = np.nan   # 데이터의 임의 영역에 NaN 존재 허용
clf.fit(X_train, y_train)
clf.predict(X_test)        # 추론 시점에 컬럼 평균 대체 방식 적용
```

### 네이티브 범주형 피처 지원

```python
import pandas as pd

df = pd.read_csv("data.csv")
X = df.drop(columns=["target"])
y = df["target"]

# 방법 1: pandas의 Categorical 또는 object 컬럼 자동 감지
clf = OQBoostClassifier(n_estimators=500)
clf.fit(X, y)

# 방법 2: 명시적으로 범주형 컬럼 지정
clf = OQBoostClassifier(cat_features=["city", "product"])
clf.fit(X, y)
```

### 조기 종료 (Early Stopping)

```python
clf = OQBoostClassifier(n_estimators=2000, early_stopping_rounds=50)
clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
print(f"조기 종료 지점: {clf.get_n_trees()} 번째 트리")
```

### 모델 저장 및 로드 (Save / Load)

```python
clf.save("model.joblib")
from oqboost import load_model
clf2 = load_model("model.joblib")
```

---

## 벤치마크 결과 (Benchmark Results)

모든 벤치마크는 80/20 학습-평가 스플릿으로 3회 반복 실행하여 평균 및 표준편차(mean ± std)를 계산했습니다. 비교군 모델(XGBoost, LightGBM, CatBoost, OQBoost)의 하이퍼파라미터는 모두 동일하게 **Optuna를 사용하여 50회(trials) 튜닝**된 최적 값을 적용했습니다.

### 메인 벤치마크 테이블 (Main Benchmark Table)

| 데이터셋 | 모델 | 정확도 (Accuracy) | 균형 정확도 (Bal. Acc.) | F1 매크로 (F1 Macro) | 로그 손실 (Log Loss) | 학습 시간 (Train, 초) | 추론 시간 (Infer, 초) |
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

### 하이라이트 지표 (Highlights)

- **CoverType 데이터셋 우수한 성능:** OQBoost는 **0.9495±0.0032**의 균형 정확도를 달성하여 기존 베이스라인 모델인 LightGBM(0.9397) 및 XGBoost(0.9392)를 유의미하게 능가했으며, 로그 손실률 역시 최저 수치인 **0.0762**를 기록했습니다.
- **Rotated Synthetic 강건성 검증:** 회전 공간에서 오블리크 분기의 진가가 입증되어 베이스라인 전체(CatBoost 0.9772, LightGBM 0.9763, XGBoost 0.9758)를 누르고 **0.9795±0.0013**의 가장 높은 균형 정확도 및 **0.0724±0.0027**의 최저 로그 손실을 기록했습니다.
- **Give Me Credit & Higgs 로그 손실 최소화:** OQBoost는 Give Me Credit (**0.4940±0.0566**) 및 Higgs (**0.5243±0.0043**) 데이터셋에서 베이스라인 모델보다 낮은 로그 손실(Log Loss)을 기록하며 우수한 확률 보정 품질을 나타냈습니다.
- **Adult & Give Me Credit 경쟁력**: OQBoost는 Give Me Credit에서 균형 정확도 **0.6651±0.791**을 기록하여 XGBoost(0.6381) 및 LightGBM(0.6250)을 앞서는 등 탁월한 성능 경쟁력을 보였습니다.

### 그림 1 — 균형 정확도 비교 (Balanced Accuracy Comparison)

![Balanced Accuracy](benchmark/results/figures/fig2_balanced_accuracy.png)

### 그림 2 — 성능 대 학습 속도 비교 (Performance vs Training Cost)

![Perf vs Cost](benchmark/results/figures/fig5_perf_vs_cost.png)

### 그림 3 — 회전 강건성 평가 (Rotation Robustness)

![Rotation Robustness](benchmark/results/figures/fig6_rotation_robustness.png)

### 그림 4 — 결측치 내성 평가 (Missing Value Robustness)

![Missing Value](benchmark/results/figures/fig7_missing_value.png)

### 그림 5 — 범주 카디널리티 내성 평가 (Categorical Cardinality Robustness)

![Categorical](benchmark/results/figures/fig8_categorical.png)

---

## 알고리즘 개요 (Algorithm)

OQBoost는 다음과 같은 3단계 유전적 사영 진화 파이프라인을 가동합니다.

**1단계 — GG-SRP (그라디언트 유도형 희소 무작위 사영)**  
각 노드에서 샘플 오차 정보를 담은 SIS 그라디언트 중요도 스코어에 비례하여 피처들을 서브 공간에 샘플링합니다. 그 후 각 피처의 가중치 부호를 가장 가파른 오차 경사하강 방향과 일치하도록 정렬합니다. 이 모든 연산은 그람 행렬 구축이나 연립 방정식 계산 없이 $O(D)$의 선형 복잡도로 수행됩니다.

**2단계 — 부모 방향 상속 (Parent Weight Inheritance)**  
자식 노드는 부모 노드의 우수한 스플릿 방향 가중치를 물려받아 아래의 두 가지 유전적 변이를 거쳐 후보군으로 추가합니다.
- *Strategy A (축 유지 변이)*: 결정 경계선을 미세하게 비틀어 탐색 각도를 미세 조정합니다 (±10%).
- *Strategy B (상관 축 확장 변이)*: 부모 노드의 피처 세트에 신규 피처를 확장 병합합니다.

**3단계 — 글로벌-로컬 크로스오버 + 깊이 감쇠**  
- *Strategy C (크로스오버)*: 이전 라운드에서 우수했던 상위 32개 사영 축을 담은 전역 버퍼 `dir_cache`와 부모 노드의 방향을 무작위 비율로 섞어 하이브리드 후보를 생산합니다.
- *깊이 감쇠 (Depth Decay)*: 상위 노드에서는 탐색(exploration)이 중요하지만 하위 노드에서는 오버핏을 억제해야 하므로, 노드 깊이에 반비례하여 변이 노이즈 강도를 $1/\sqrt{1 + d}$ 로 감쇠시킵니다.

수리적 유도 과정은 [`docs/algorithm.md`](docs/algorithm.md)에서, 상세 연구 및 실험 로그는 [`docs/THEORY.md`](docs/THEORY.md)에서 확인하실 수 있습니다.

---

## 매개변수 명세 (Parameters)

| 매개변수 (Parameter) | 기본값 (Default) | 설명 (Description) |
|-----------|---------|-------------|
| `n_estimators` | 500 | 부스팅 반복 라운드 수 (트리 개수) |
| `learning_rate` | 0.05 | 트리가 앙상블에 기여하는 학습 축소 계수 |
| `max_depth` | 6 | 리프 노드 수 한도 = $2^{\text{max_depth}}$ (64개) |
| `reg_lambda` | 1.0 | 리프 스코어 가중치에 대한 L2 규제 강도 |
| `subsample` | 0.8 | 각 트리를 빌드할 때 무작위 샘플링할 샘플 비율 |
| `early_stopping_rounds` | 50 | 검증 데이터 손실이 정체될 시 조기 종료 임계치 |
| `cat_features` | None | 범주형 피처 컬럼 명칭(DataFrame) 또는 컬럼 인덱스 목록 |
| `class_weight` | None | `"balanced"` 설정 시 빈도 역수에 비례하여 의사결정 선값을 보정 |
| `inherited_rp_ratio` | 1.0 | 상속 및 캐시 패밀리가 스캔 후보 풀에서 차지하는 비율 |
| `mutation_rate` | 0.1 | Strategy A 축 변이의 노이즈 강도 파라미터 |
| `mutation_strength` | 0.5 | Strategy B 상관 피처 차용 시의 기본 가중치 |
| `pobs` | False | Haar-orthogonal 기반의 무작위 직교 블록 후보 주입 여부 |
| `random_state` | None | 난수 고정 시드 |
| `verbose` | False | 학습 과정 모니터링 출력 여부 |

---

## 벤치마크 실행 방법 (Running Benchmarks)

```bash
cd benchmark

# 전체 벤치마크 가동 (Higgs + CoverType 포함 시 몇 시간 소요될 수 있음)
python run_all.py

# 대용량 데이터셋 제외하고 실행
python run_all.py --skip higgs covertype

# 개별 데이터셋 전용 벤치마크 실행
python adult.py
python rotated_synthetic.py
python missing_value_robustness.py
python categorical_robustness.py

# 벤치마크 완료 후 결과 이미지(차트)들 생성
python generate_figures.py

# 벤치마크 완료 후 결과 테이블 생성 (results/summary.md)
python generate_tables.py
```

일부 대량 데이터셋은 아래 링크에서 수동 다운로드하여 데이터 경로에 위치시켜야 합니다.
- **HIGGS**: https://archive.ics.uci.edu/dataset/280 → `benchmark/data/HIGGS.csv.gz`
- **Give Me Some Credit (Kaggle)**: https://www.kaggle.com/competitions/GiveMeSomeCredit → `benchmark/data/cs-training.csv`

---

## 디렉터리 구조 (Repository Structure)

```
oqboost/
├── src/oqboost/
│   ├── __init__.py
│   ├── _classifier.py      # OQBoostClassifier 래퍼
│   ├── _oqboost.py        # C++ 바인딩 + OQBoostTree, OQBoostContext
│   └── _ext/
│       ├── oqboost.cpp    # 핵심 빌딩 엔진, 정렬 스캔, 직렬화
│       ├── oqboost_types.h # OQTree 노드 구조 선언
│       ├── oqboost_core.h # 핵심 수학 및 공유 상수 라이브러리
│       └── liboqboost.dylib / .so / .dll  (컴파일 바이너리)
├── benchmark/
│   ├── *.py                # 데이터셋별 평가 스크립트
│   ├── _utils.py           # 공통 학습/평가 지표 래퍼
│   ├── generate_figures.py
│   ├── generate_tables.py
│   ├── run_all.py
│   └── results/
│       ├── csv/            # 가공 이전 벤치마크 원본 데이터
│       ├── figures/        # 최종 시각화 차트 이미지
│       └── summary.md
├── docs/
│   ├── algorithm.md        # 사선 수학 및 알고리즘 명세
│   ├── api.md              # 파이썬 전체 API 명세서
│   └── quickstart.md       # 빠른 사용법 가이드
├── tests/
│   └── test_classifier.py
└── pyproject.toml
```

---

## 라이선스 (License)

[MIT](LICENSE) — Copyright (c) 2025 cree1116

---

[English Version (영문 버전)](README.md)
