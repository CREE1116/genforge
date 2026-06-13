# OQBoost: 알고리즘 및 이론 (Algorithm and Theory)

OQBoost는 높은 분류 정확도, 강력한 일반화 성능, C++ 기반의 빠른 실행 속도를 위해 설계된 정형 데이터용 경사 부스팅 사선 결정 트리(oblique GBDT) 모델입니다. 본 문서는 최근의 C++ 엔진 최적화를 포함하여 OQBoost의 수리적 설계와 아키텍처적 설계를 명세합니다.

---

## 1. 뉴턴-랩슨 부스팅 프레임워크 (Newton-Raphson Boosting Framework)

OQBoost는 표준적인 뉴턴-랩슨 부스팅 공식을 따릅니다. 라운드(반복) $m$에서 기존 앙상블 모델의 예측을 $F^{(m)}(x)$라 할 때, 다음 트리 $f^{(m+1)}(x)$는 손실 함수의 2차 테일러 전개식을 최소화하도록 학습됩니다.

$$\mathcal{L}^{(m+1)} \approx \sum_{i=1}^{N} \left[ g_i f^{(m+1)}(x_i) + \frac{1}{2} h_i \left(f^{(m+1)}(x_i)\right)^2 \right] + \Omega(f^{(m+1)})$$

여기서 변수들의 수리적 정의는 다음과 같습니다.
*   $g_i = \left. \frac{\partial \ell(y_i, F(x_i))}{\partial F(x_i)} \right|_{F = F^{(m)}}$ 은 1차 경사(1st-order gradient)입니다.
*   $h_i = \left. \frac{\partial^2 \ell(y_i, F(x_i))}{\partial F(x_i)^2} \right|_{F = F^{(m)}}$ 은 2차 헤시안(2nd-order Hessian)입니다.
*   $\Omega(f) = \lambda \sum_{\text{leaves}} \|w_\text{leaf}\|^2$ 은 리프 노드 가중치에 대한 L2 정규화 페널티입니다.

$K$-클래스 분류 문제의 경우, 앙상블은 K개의 출력 헤드를 갖고 다중 클래스 Softmax 손실 함수를 사용합니다.

---

## 2. 분할 전략: 사선 결정 규칙 (Oblique Decision Rules)

단일 피처만을 비교하여 축 정렬 방식으로 분기하는 기존 트리($x_j < \theta$)와 달리, OQBoost는 각 분기 노드에서 여러 피처의 선형 결합을 바탕으로 하는 **사선 분기(Oblique splits)**를 수행합니다.

$$w^T x = \sum_{j \in \mathcal{S}} w_j x_j < \theta$$

여기서 $w$는 희소 가중치 벡터(활성 피처 서브공간의 크기를 $|\mathcal{S}| \le D_{\text{SUB\_MAX}} = 16$으로 엄격히 제한)이며, $\theta$는 분기 임계값입니다. 이를 통해 모델은 개별 노드 레벨에서 다차원 피처 간의 선형 상관 관계를 직접 캡처할 수 있습니다.

---

## 3. 방향 벡터 생성 및 최적화 파이프라인 (Direction Generation & Optimization)

OQBoost는 각 노드 분기 시점에서 값비싼 좌표 하강법(Coordinate Descent)이나 그람 행렬(Gram matrix)의 역행렬 연산을 피하기 위해, 경사(gradient) 정렬 정보를 담은 3단계 무작위화 파이프라인을 운영합니다.

### 1단계: GG-SRP (그라디언트 유도형 희소 무작위 사영)
모든 차원을 무차별로 탐색하는 대신, OQBoost는 현재 노드에 도달한 샘플 서브셋 $\mathcal{I}_t$를 기반으로 Sure Independence Screening (SIS) 스코어를 계산하여 피처 후보군을 추려냅니다.

$$s_f = \frac{\left| \sum_{i \in \mathcal{I}_t} x_{if} g_i \right|}{\sqrt{\sum_{i \in \mathcal{I}_t} h_i x_{if}^2 + \lambda}}$$

1.  **서브공간 선택**: 스크리닝 스코어 $s_f$에 비례하는 확률로 무작위 희소 서브공간 $\mathcal{S}$를 샘플링합니다.
2.  **활성 피처 개수 한도**: C++ `SparseVec` 메모리 한계를 초과하지 않도록 0이 아닌 값을 가지는 피처 수를 $D_{\text{SUB\_MAX}} = 16$으로 고정 캡핑합니다.
3.  **부호 정렬**: 선택된 각 피처 $f \in \mathcal{S}$에 대해, 가중치 부호가 오차가 줄어드는 가파른 경사하강 방향과 일치하도록 결정합니다.
    $$w_f = -\operatorname{sign}\left( \sum_{i \in \mathcal{I}_t} x_{if} g_i \right) \cdot |r_f|, \quad r_f \sim \mathcal{N}(0, 1)$$
4.  **정규화**: 가중치 벡터의 L2 노름을 1로 정규화합니다: $w \leftarrow \frac{w}{\|w\|_2}$.

### 2단계: 유전적 방향성 상속 (Hereditary Direction Inheritance)
자식 노드에 도달한 샘플들은 이미 부모 노드의 분기 평면에 의해 걸러진 상태입니다. 따라서 부모 노드의 가중치 벡터 $w_{\text{parent}}$는 매우 훌륭한 초기 상태(warm-start) 역할을 합니다. OQBoost는 다음 두 가지 변이 전략을 검토합니다.
*   **Strategy A (축 유지 변이)**: 부모의 결정 경계 축을 유지하면서 방향만 미세하게 조절합니다.
    $$w_{\text{mutated}} = w_{\text{parent}} + \text{rate} \cdot \epsilon, \quad \epsilon_j \sim \mathcal{U}(-1, 1)$$
*   **Strategy B (새로운 축 차용 변이)**: 부모의 피처 세트에 현 노드 오차와 상관성이 높은 새로운 피처 $f^*$ 하나를 확장 추가합니다.
    $$w_{\text{mutated}} = w_{\text{parent}} \oplus \{ f^* \mapsto \pm \text{strength} \}$$

### 3단계: 부모-캐시 크로스오버 및 깊이별 변이 감쇠 (Crossover & Depth-Decayed Mutation)
*   **Strategy C (글로벌-로컬 크로스오버)**: 이전 부스팅 라운드에서 우수한 게인을 기록했던 최적의 방향 벡터들을 링 버퍼 형태의 글로벌 캐시 `dir_cache` (최대 32개)에 보관하고, 이를 부모의 방향과 블렌딩하여 후보로 만듭니다.
    $$w_{\text{blend}} = \alpha w_{\text{parent}} + (1 - \alpha) w_{\text{cache}}, \quad \alpha \sim \mathcal{U}(0, 1)$$
*   **깊이별 변이 감쇠**: 트리가 얕은 노드에서는 탐색(Exploration)이 유익하지만, 트리가 깊어질수록 샘플 크기가 급감하여 큰 변이는 과적합(Overfitting)을 유발합니다. OQBoost는 깊이 $d$에 반비례하여 변이율과 변이 강도를 감쇠시킵니다.
    $$\text{rate}_d = \frac{\text{rate}_0}{\sqrt{1 + d}}, \quad \text{strength}_d = \frac{\text{strength}_0}{1 + d}$$
*   **난수 시드 전파**: 트리의 다양성을 보장하고 부스팅 라운드 간 중복 분할을 방지하기 위해, 파이썬 분류기는 매 트리 빌드 시 고유한 난수 시드(예: `rng.integers(1 << 30)`)를 C++ 엔진에 전달합니다. C++ 엔진은 노드별 생성기 시드를 `seed + t`로 결합하여 전체 프로세스의 재현성과 다양성을 동시에 확보합니다.

---

## 4. 작동 원리: 수학적 및 통계학적 직관 (Why It Works)

OQBoost가 기존 축 정렬 모델보다 우수한 성능을 내면서 동시에 무거운 경사 부스팅 트리의 학습 병목을 해결할 수 있었던 이유에는 다음과 같은 수학적 및 통계학적 배경이 있습니다.

### 4.1 오차의 최속 강하 방향 정렬 (Newton-Raphson Gradient Alignment)
GG-SRP 기작의 핵심은 사영 방향을 단순한 무작위 탐색에 방임하지 않고, 노드에 도달한 샘플들의 경사(gradient) 분포의 반대 방향으로 피처별 가중치의 부호를 명시적으로 부합시키는 것입니다 ($w_f \propto -\operatorname{sign}(\sum g_i x_{if})$).
수리적으로 이는 목적함수인 2차 Taylor 전개식(Newton step)의 최속 하강 방향(gradient descent path)과 초기 사영 축을 정렬시키는 동작으로, 이로 인해 탐색 풀 내부의 임의의 단일 사선 벡터조차 목적함수의 게인을 효율적으로 캡처하도록 보장됩니다.

### 4.2 오목 탐색 공간의 극값 정리 (Extreme Values of Gain Distribution)
노드 레벨의 후보 토너먼트 선택식은 최댓값 스캔 연산($G^* = \max_i \text{gain}(w_i)$)이므로 극값 통계학(Order Statistics) 법칙을 따릅니다.
각 후보 생성 기작(Axis, SRP, 상속, 캐시)은 상호 독립적이거나 다른 꼬리 분포(tail shape)를 가지므로, 이들을 하이브리드하여 풀에 섞어주면 토너먼트 최대 게인의 기대치 최댓값($\mathbb{E}[\max]$)은 수학적으로 단일 패밀리만 구성할 때보다 항상 커지게 됩니다. 이것이 순수 무작위 사영이나 순수 상속 모델이 하이브리드 프로덕션 구성을 이기지 못한 수학적 배경입니다.

### 4.3 직교 보완성 원리 (Orthogonality Principle & POBS)
특정 노드에서 결정 경계 $w$에 의해 데이터 공간이 쪼개지면, 그라디언트 신호의 $w$ 방향 선형 성분은 자식 노드에서 대부분 소진됩니다. 따라서 자식 노드로 물려받은 오차의 정보(Residuals)는 부모 분할 평면의 직교 여공간(orthogonal complement) 영역으로 강하게 편향되어 농축됩니다.
- OQBoost는 상호 직교하는 다차원 후보군 블록을 강제하는 **POBS (Parseval-Constrained Random Orthogonal Block Projections)** 기작을 통해 중복 검색 차원을 완벽히 소거하여, 한정된 토너먼트 풀 크기 안에서 탐색 공간의 다양성(diversity)을 극대화합니다. 이는 파르스발 항등식(Parseval's identity)에 근거하여 사영 에너지 총합을 보존하고, 토너먼트 내의 최댓값 하한선을 보장하는 수학적 안전장치 역할을 합니다.

### 4.4 데이터 스케일별 추정치 분산 제어 (Bias-Variance trade-off via Depth-Adaptive Budget)
트리 깊이가 깊어질수록 분기 노드에 유입되는 샘플의 수 $N_t$는 지수적으로 급감합니다.
- **깊은 노드의 노이즈**: 적은 샘플 환경에서 복잡한 최적화(예: CD 또는 WLS 역행렬 계산)로 유도한 정밀 사선 축은 국소 샘플 노이즈에 완벽히 오버핏되어 모델 분산(variance)을 심각하게 폭증시킵니다.
- **적응형 예산 분배**: OQBoost는 표본이 많아 시그널이 깨끗한 얕은 노드에는 64개 이상의 조밀한 토너먼트를 개최하고, 잡음이 심한 깊은 노드에는 후보군 크기를 8개 수준으로 대폭 조여 탐색을 제어합니다. 이는 학습 연산 속도를 대폭 단축함과 동시에 깊은 노드의 분기 분산을 억제하는 자동 정규화(regularization) 장치로 작동합니다.

---

## 5. 메모리 최적화 C++ 엔진 (Memory-Optimized C++ Engine)

### 객체 풀 기반 히스토그램 재사용 기법 (`hist_pool`)
히스토그램 생성은 GBDT 학습 연산에서 가장 많은 시간을 소모하는 병목입니다. 최적 우선(Best-first) 트리 성장 과정에서 잦은 힙 메모리 할당(`malloc`) 및 해제(`free`)를 막기 위해, OQBoost는 `gf_build` 함수 내에 히스토그램 버퍼 풀을 구축하여 재활용합니다.

```cpp
std::vector<std::vector<float>> hist_pool;

auto get_hist = [&]() -> std::vector<float> {
  if (!hist_pool.empty()) {
    auto h = std::move(hist_pool.back());
    hist_pool.pop_back();
    std::fill(h.begin(), h.end(), 0.0f);
    return h;
  }
  return std::vector<float>(HSZ, 0.0f);
};

auto recycle_hist = [&](std::vector<float>& h) {
  if (h.size() == HSZ) {
    hist_pool.push_back(std::move(h));
  }
  h.clear();
};
```

이 객체 풀은 256-bin 피처 히스토그램 버퍼를 재사용하여, 풀이 일정 수준 차오른 뒤에는 힙 메모리 재할당 오버헤드를 0으로 낮춥니다.

### 동적 하이브리드 히스토그램 병렬화 (Dynamic OpenMP Parallelization)
OQBoost는 CPU 스레드 수($T$)와 현재 피처 수($D$)의 관계에 따라 OpenMP 병렬화 전략을 유연하게 스위칭합니다.

1.  **피처 단위 병렬화 (피처 수 $D \ge T$ 일 때)**:
    *   피처들을 $\lfloor D/T \rfloor$ 크기의 청크로 쪼개어 스레드별로 할당합니다.
    *   각 스레드는 자신에게 할당된 글로벌 히스토그램의 특정 슬라이스 영역에 직접 히스토그램을 작성합니다.
    *   스레드별 로컬 버퍼 할당 및 병합(Merge) 절차가 필요 없어지므로 캐시 연속성(cache-contiguous)이 유지되고 최적의 성능을 냅니다.
2.  **샘플 단위 병렬화 (피처 수 $D < T$ 일 때)**:
    *   피처 수가 너무 적을 때 피처 단위 병렬화를 사용하면 사용되지 않고 노는 CPU 코어들이 생깁니다.
    *   이 경우 샘플들을 스레드별로 할당하여 연산한 뒤, 병렬 리덕션(reduction) 루프로 머지합니다.
    *   피처 수가 적기 때문에 로컬 버퍼가 L1/L2 캐시 공간에 완전히 담겨 캐시 경합(cache thrashing)이 방지됩니다.

---

## 6. 알고리즘 시간 복잡도 (Complexity)

| 단계 | 시간 복잡도 | 비고 |
| :--- | :--- | :--- |
| **컨텍스트 생성 (Context)** | $O(N \cdot D)$ | 학습 초기 1회 수행, 피처의 bin 경계 계산 |
| **범주형 피처 재인코딩** | $O(N \cdot D_{\text{cat}})$ | 매 부스팅 라운드별 1회 수행 |
| **GG-SRP 사영 연산** | $O(D + b)$ | 노드별 수행; 서브공간 피처 수 $b \le 16$ |
| **히스토그램 빌딩** | $O(n_t \cdot D \cdot 256)$ | 히스토그램 감산 기법 적용으로 실제 연산 범위 축소 |
| **추론 시 샘플 라우팅** | $O(N \cdot \text{depth} \cdot s)$ | 분기당 사용 피처 수 평균 $s \le 6$ |

---

## 7. 소거 연구 및 실험 결과 (Empirical Findings)

분류용 벤치마크 데이터(샘플 10만 개, 피처 50개)를 통한 소거 연구 결과는 다음과 같습니다.

*   **GG-SRP vs 좌표 하강법(CD)**: 정확한 수치 최적화인 Gauss-Seidel CD 좌표 탐색 대신, 해석적이고 단순 확률 사영에 기초한 GG-SRP를 적용하여도 균형 정확도 손실 없이 노드 연산 속도를 **10배 이상** 단축했습니다.
*   **부모 노드 상속 및 크로스오버**: 캐시 교차 탐색(Strategy C)과 깊이별 감쇠 변이를 조합 적용할 때 가장 낮고 안정적인 Log Loss를 달성하여 과적합을 효과적으로 통제했습니다.
*   **균형화된 Argmax 의사결정**: 클래스 비율 불균형 시 강제로 Newton-Raphson 경사를 왜곡하는 대신, 최종 `predict` 시점에 사전 보정(prior correction) 비율을 조정함으로써 확률 추정값(log loss)을 훼손하지 않으면서도 균형 정확도를 높였습니다.
*   **상속 기작의 실험적 통찰**: 인위적 데이터 실험을 통해 부모의 분할 기준을 그대로 물려받는 상계 변이가 국소 오차 소진(orthogonality principle)으로 인해 생각보다 비효율적임을 발견했습니다. 그럼에도 튜닝 파라미터 제약 하에서 고유 차원의 보완을 돕는 안전망 역할을 하고 있음을 실전 검증하였습니다. (`research/FINDINGS.md` 참고)

---

## 8. 향후 연구 과제: 그라디언트 공분산 기반 사선 유도 (Gradient Covariance)

2026-06-13, 프로토타입(`OQBoostCovClassifier`) 연구를 통해 사선 생성 파이프라인을 획기적으로 단순화하는 통계적 최적화 방향이 수립되었습니다.

### 핵심 단순화 제안:
노드 분기 마다 임의의 방향 벡터를 무수히 난립시켜 토너먼트를 여는 방식 대신, 다음 두 갈래로만 축을 좁혀 검토합니다.
1. 기존 $D$개의 단일 피처 축 정렬 벡터 (회전되지 않은 원본 공간의 척도 보호용).
2. 현 노드의 오차 기여도가 높은 피처들에 한정해 계산한 딱 **1개**의 그라디언트 공분산 벡터 $w_{\text{cov}}$:
   $$w_{\text{cov\_sub}} = \frac{G_{\text{sub}}}{\|G_{\text{sub}}\|_2 + \epsilon}, \quad \text{where } G_{\text{sub}} = -X_{\text{sub}}^T g$$

이를 통해 탐색 후보 풀의 크기를 매 노드마다 $D+1$개로 제한할 수 있습니다.

### 이론적 배경:
데이터가 소실되는 깊은 노드에서 노이즈가 낀 헤시안 역행렬 계산($A^{-1}G$ 방식의 Newton-Raphson 해)에 집착하지 않고, 단순히 오차의 주류 경향(공분산)만을 따라 사선 축을 유도함으로써 Hessian-free 형태의 내재적 규제(Implicit L2 Regularization) 효과를 얻습니다. 실험실 수준의 tabular 벤치마크(Adult, Credit Default) 테스트에서 이 단순한 $D+1$ 설계는 좌표 하강법이나 종전의 대규모 토너먼트 방식보다 일관되게 높은 정확도 및 낮은 Log Loss를 나타냈습니다. 차기 엔진 릴리스에서 이 그라디언트 공분산 사선 유도 메커니즘을 C++ 코어에 포팅하는 방안이 유력하게 논의되고 있습니다.

---

[English Version (영문 버전)](algorithm.md)
