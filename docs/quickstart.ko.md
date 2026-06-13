# OQBoost 퀵스타트 (Quickstart)

## 설치 (Installation)

```bash
pip install oqboost
```

C++ 컴파일러가 필요합니다 (macOS의 경우 `clang++`, Linux의 경우 `g++`). 지원되는 플랫폼의 경우 빌드 완료된 휠(wheel) 패키지가 제공되므로 별도의 컴파일 과정이 필요하지 않습니다.

---

## 기본 사용법 (Basic Usage)

```python
from oqboost import OQBoostClassifier

clf = OQBoostClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
)

clf.fit(X_train, y_train)
clf.predict(X_test)
clf.predict_proba(X_test)
```

---

## 검증 데이터 및 조기 종료 (With Validation and Early Stopping)

```python
clf = OQBoostClassifier(
    n_estimators=1000,
    early_stopping_rounds=50,
)
clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
print(f"학습 완료된 트리 개수: {clf.get_n_trees()} (조기 종료 적용)")
```

---

## Pandas DataFrame 및 범주형 피처 지원 (Pandas DataFrame with Categorical Features)

```python
import pandas as pd

df = pd.read_csv("dataset.csv")
X = df.drop(columns=["target"])
y = df["target"]

# 방법 1: pandas의 Categorical 또는 object 컬럼 자동 감지
clf = OQBoostClassifier(n_estimators=500)
clf.fit(X, y)

# 방법 2: 명시적으로 범주형 컬럼 지정
clf = OQBoostClassifier(
    n_estimators=500,
    cat_features=["city", "product_type", "category"],
)
clf.fit(X, y)
```

---

## 결측치 처리 (Handling Missing Values)

OQBoost는 결측치(NaN)를 네이티브로 처리하므로 별도의 대체(imputation) 단계가 필요하지 않습니다.

```python
import numpy as np

X_train[50, 3] = np.nan   # NaN 값 주입
X_test[10, 7] = np.nan    # 테스트 세트의 NaN 값도 허용됨

clf = OQBoostClassifier()
clf.fit(X_train, y_train)
clf.predict(X_test)        # 추론 시점에 컬럼 평균 값으로 대체 처리가 내부 수행됨
```

---

## 모델 저장 및 로드 (Save and Load)

```python
clf.save("model.joblib")

from oqboost import load_model
clf2 = load_model("model.joblib")
proba = clf2.predict_proba(X_test)
```

---

## scikit-learn Pipeline 연동

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", OQBoostClassifier(n_estimators=300, max_depth=5)),
])
pipe.fit(X_train, y_train)
pipe.predict(X_test)
```

---

## 유용한 팁 (Tips)

**하이퍼파라미터 튜닝:** OQBoost의 주요 파라미터는 `max_depth` (권장 범위 4–8) 및 `reg_lambda` (권장 범위 0.1–10)입니다. 기본값으로 먼저 테스트해보는 것을 권장합니다.

**속도 vs 정확도:** 학습 속도를 높이려면 `n_estimators`를 줄이고 `learning_rate`를 높이십시오. 최적의 트리 개수를 찾으려면 조기 종료(`early_stopping_rounds`)를 활용하는 것이 좋습니다.

**불균형 데이터:** `class_weight="balanced"`로 설정하면 클래스 빈도의 역수에 비례하여 그래디언트의 가중치를 조절합니다. 클래스가 매우 불균형한 경우 원본 정확도(Accuracy)보다 균형 정확도(`balanced_accuracy_score`)나 F1 스코어로 평가하십시오.

**회전 강건성 (Rotation-robust):** OQBoost의 사선 분기(Oblique splits)는 회전된 피처 공간에 자연스럽게 적응합니다. 데이터의 여러 피처들의 선형 결합이 중요한 정보를 가질 때, OQBoost는 별도의 피처 엔지니어링 없이도 축 정렬(Axis-aligned) 모델들보다 우수한 성능을 보여줍니다.

---

[English Version (영문 버전)](quickstart.md)
