# OQBoost API 레퍼런스 (API Reference)

## `OQBoostClassifier`

```python
from oqboost import OQBoostClassifier
```

경사 부스팅 사선 결정 트리(Gradient-boosted oblique decision tree) 분류기입니다. scikit-learn의 `BaseEstimator`, `ClassifierMixin`, `TransformerMixin` 인터페이스를 구현합니다.

### 생성자 (Constructor)

```python
OQBoostClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=6,
    reg_lambda=1.0,
    subsample=0.8,
    early_stopping_rounds=50,
    random_state=None,
    verbose=False,
    cat_features=None,
    class_weight=None,
    prior_alpha=0.5,
    inherited_rp_ratio=1.0,
    mutation_rate=0.1,
    mutation_strength=0.5,
    pobs=False,
)
```

#### 매개변수 (Parameters)

| 매개변수 | 타입 | 기본값 | 설명 |
|-----------|------|---------|-------------|
| `n_estimators` | int | 1000 | 부스팅 라운드(트리 개수) 수 |
| `learning_rate` | float | 0.03 | 각 트리의 단일 예측값에 적용할 학습률 (축소 계수) |
| `max_depth` | int | 6 | 트리 최대 깊이. 리프 노드 한도는 $2^{\text{max\_depth}}$ (64개 리프 노드) |
| `reg_lambda` | float | 1.0 | 리프 가중치에 대한 L2 규제 (Newton step 분모 항 추가) |
| `subsample` | float | 0.8 | 각 트리를 구성할 때 무작위 샘플링할 행 비율 ($0 < \text{subsample} \le 1$) |
| `early_stopping_rounds` | int or None | 50 | 검증 데이터 손실이 정체되면 조기 종료함. `eval_set` 필요 |
| `random_state` | int or None | None | 결과 재현을 위한 난수 시드 |
| `verbose` | bool | False | 라운드별 손실함수 및 지표 출력 여부 |
| `cat_features` | list or None | None | DataFrame의 경우 컬럼명 리스트, ndarray의 경우 컬럼 인덱스 리스트 |
| `class_weight` | str or None | None | `"balanced"`로 설정 시 클래스 빈도의 역수로 가중치를 재조정하여 예측 |
| `prior_alpha` | float | 0.5 | `"balanced"` 가중치 사전 보정 시 가중 세기 조절 (0~1 사이값, 0.5는 기하평균 절충안) |
| `inherited_rp_ratio` | float | 1.0 | 부모 노드 상속 및 글로벌 캐시에서 탐색할 방향 후보 비율 |
| `mutation_rate` | float | 0.1 | 부모로부터 상속받은 방향 축의 노이즈 강도 설정 |
| `mutation_strength` | float | 0.5 | 하이브리드 탐색 시 부모 노드 외의 상관 피처 차용 가중치 |
| `pobs` | bool | False | 토너먼트에 Haar-orthogonal 기반의 무작위 직교 블록 후보(pobs_sis)를 주입할지 여부 |

---

### 메서드 (Methods)

#### `fit(X, y, eval_set=None, sample_weight=None)`

모델을 학습합니다.

```python
clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
```

- `X`: array-like (N, D) 또는 pandas DataFrame. NaN 값 허용.
- `y`: array-like (N,), 0부터 시작하는 정수 클래스 레이블.
- `eval_set`: (X_val, y_val) 튜플의 리스트. 첫 번째 튜플이 조기 종료 스캔에 사용됨.

`self`를 반환합니다.

---

#### `predict(X)`

클래스 예측 레이블을 반환합니다.

```python
y_pred = clf.predict(X_test)  # → np.ndarray (N,)
```

---

#### `predict_proba(X)`

클래스별 예측 확률을 반환합니다.

```python
proba = clf.predict_proba(X_test)  # → np.ndarray (N, K)
```

각 행의 확률 합은 1이며, 앙상블 리프 노드 스코어에 소프트맥스(softmax) 함수를 씌워 계산합니다.

---

#### `save(path)` / `load(path)`

학습된 모델을 파일로 직렬화(저장)하거나 가져옵니다.

```python
clf.save("model.joblib")
clf2 = OQBoostClassifier.load("model.joblib")
```

---

#### `get_n_trees()`

실제 학습 완료된(조기 종료가 적용된) 트리 개수를 반환합니다.

---

### scikit-learn 호환성 (sklearn Compatibility)

`OQBoostClassifier`는 `sklearn.pipeline.Pipeline`, `clone`, `GridSearchCV`, `check_estimator`와 완벽히 호환됩니다.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", OQBoostClassifier(n_estimators=300)),
])
pipe.fit(X_train, y_train)
```

---

## `OQBoostTree`

```python
from oqboost import OQBoostTree
```

앙상블 부스팅 풀이 없는 단일 사선 결정 트리입니다. 직접 부스팅 루프를 커스텀하고 제어하고자 할 때 유용합니다.

### 생성자 (Constructor)

```python
OQBoostTree(
    max_depth=4,
    reg_lambda=1.0,
    subsample=1.0,
    random_state=None,
)
```

### 메서드 (Methods)

#### `fit_predict(X, G, H, D_num=None, subset=None, ...)`

그라디언트 $G$와 헤시안 $H$를 기반으로 하나의 트리를 학습하고 리프 노드 예측값을 반환합니다 `(N, K)`.

#### `predict(X)`

각 샘플에 매핑되는 리프 예측 값들을 반환합니다 `(N, K)`.

---

## `OQBoostContext`

```python
from oqboost._oqboost import OQBoostContext
```

부스팅 루프를 위한 재사용 가능한 비닝 컨텍스트(Binning Context)입니다. 첫 1회 비닝 후 다음 라운드들에 계속 재사용하여 계산 병목을 방지합니다.

### 생성자 (Constructor)

```python
ctx = OQBoostContext(X, D_num=None)
```

### 메서드 (Methods)

#### `build(G, H, sub, max_depth, reg_lambda, ...)`

단일 부스팅 라운드를 실행하고 빌드된 트리와 그에 해당하는 예측 결과를 반환합니다 `(OQBoostTree, np.ndarray)`.

#### `close()`

메모리 상의 C++ 관련 자원을 해제합니다. 가비지 컬렉터에 의해 `__del__` 호출 시 자동 해제되기도 합니다.

---

## `load_model(path)`

```python
from oqboost import load_model
clf = load_model("model.joblib")
```

`OQBoostClassifier.load(path)`를 더 편하게 호출하는 단축형 래퍼 함수입니다.

---

[English Version (영문 버전)](api.md)
