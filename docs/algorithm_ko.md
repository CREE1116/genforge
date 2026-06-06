# HypForge — 알고리즘 레퍼런스 (한국어)

## 개요

HypForge(Hypothesis Pool Forge)는 **사선 분할(oblique split)**을 사용하는 그래디언트 부스팅 의사결정트리 프레임워크다. 각 트리 노드는 단일 특성이 아닌 특성들의 선형 결합(학습된 방향)으로 분할한다. 분할 방향은 UCB(Upper Confidence Bound) 밴딧 알고리즘으로 관리되는 *가설(hypothesis)* 풀에서 선택된다.

핵심 아이디어: 매 분할마다 개별 특성을 탐색하는 대신, HypForge는 crossover 교배와 적합도 압력을 통해 지속적으로 정제되는 투영 방향의 공유 풀을 유지한다. 더 나은 방향은 생존하고, 중복/약한 방향은 제거되며, 새로운 방향은 기존 최상위 방향들을 교배해 제안된다.

---

## 가설 타입

**가설(Hypothesis)** `h`는 희소 단위 노름 가중치 벡터 `w ∈ ℝᴰ`를 통해 하나의 사선 분할 방향을 정의한다:

| 타입 | ID | 투영식 | 비고 |
|------|----|--------|------|
| `linear` | 0 | `X @ w` | 표준 사선 분할 |
| `leaky_relu` | 1 | `max(X@w, 0.01·X@w)` | 비대칭; 양수 투영 강조 |
| `product` | 2 | `h₁(X) · h₂(X)` | 기존 두 가설의 복합 |

에블레이션 결과: `linear` + `leaky_relu` 혼합(op_mode="all")이 단독 사용보다 우수. `product`는 지원되나 다양성 pruning에서 두 주요 타입에 밀려 거의 생존하지 못함.

각 가설이 갖는 내부 상태:
- **UCB 통계**: `n_obs`, `mu_fitness`, `M2_fitness` (Welford 온라인 평균/분산)
- **사용 통계**: `use_count`, `rounds_since_last_use`
- **투영 캐시**: `full_cache[N]` (학습 전용, 저장 전 삭제), `thresholds[9]`
- **계통 및 혈통**:
  - `parent1`, `parent2`: 부모 가설의 global history ID (교배 혹은 product 구성요소, 없으면 -1)
  - `birth_round`: 해당 가설이 생성된 부스팅 라운드 인덱스
  - `family_id`: 고유 혈통 식별자 (교배/product 시 더 우수한 부모의 ID를 상속)
  - `family_fitness`: 모든 후손 가설들의 평균 fitness
  - `breeding_value`: 모든 직계 자식 가설들의 평균 fitness

희소 가중치: 최대 `k = max(2, floor(√D_num))`개 비영 항목, ℓ2 정규화.

---

## 풀 구조

**HypForgePool**은 D개의 축 정렬 기저 가설(특성 하나당 하나)로 초기화되며, `is_base=True`로 표시돼 영구 보존된다. 학습 중 풀은 crossover 교배로 성장하고, 다양성 dedup·엘리트주의·MAP-Elites 타입 쿼터·생존 필터로 축소된다.

### 에블레이션으로 확정된 기본값

| 파라미터 | 값 | 근거 |
|---------|-----|------|
| `pool_size` | 400 | 더 큰 후보 버퍼 → 다양성 선택 품질 향상 |
| `map_elites_slots` | 100 | linear/lrelu 50/50 균형 강제 (pool_size=400 기준) |
| `elitism_k` | 20 | 상위 20개는 유사도 제거에서 면제 |
| `alps_mode` | True | 관측 횟수 적은 가설(n_obs<5) 제거 방어 |
| `crossover_top_k` | 3 | 상위 3쌍 교배; 이상이면 품질 희석 |
| `subsample` | 0.8 | 트리당 행 서브샘플링; 정규화 효과 |
| `max_depth` | 4 | 최적값; 5 이상은 과적합 |
| `family_max_size` | 30 | 혈통 기반 MAP-Elites 쿼터; 특정 혈통의 독점 방지 |
| `meta_evolution` | True | 4가지 진화 정책 후보군 중 UCB 알고리즘을 통한 온라인 자동 선택 |
| `family_lambda` | 0.1 | 최종 UCB 점수 계산 시 family_fitness 반영 가중치 |
| `breeding_beta` | 0.3 | 부모 가설 선택 시 breeding_value 반영 가중치 |

에블레이션으로 하드코딩된 파라미터 (사용자 노출 불필요):

| 파라미터 | 고정값 | 이유 |
|---------|--------|------|
| `mutation_mode` | none | 그래디언트/랜덤 변이가 일관되게 해로움 |
| `feedback_mode` | scan_only | split_gain 재관측이 오버헤드만 추가 |
| `fitness_norm_mode` | none | UCB 탐색 보너스가 이미 스케일 변동 처리 |
| `recency_bonus_rounds` | 0 | 최근 사용 가설 보호 → 다양성 손상 |
| `use_rate_bonus` | 0.0 | 모든 값에서 해로움 확인 |
| `novelty_lambda` | 0.0 | dedup 후 novelty 재점수화가 해로움 |
| `evolve_mode` | crossover | standard·novelty 모드 모두 열등 |

---

## UCB 점수

매 evolve 라운드 후 각 가설의 점수 계산:

```
score = ucb_score + family_lambda × family_fitness
```
여기서 `ucb_score`는 다음과 같이 계산됩니다:
```
ucb_score = μ_fitness + (σ_fitness + 0.5) / √n_obs − η · complexity
```

- `μ_fitness` — 관측된 split-gain 값들의 Welford 온라인 평균
- `σ_fitness` — Welford 온라인 표준편차 (n_obs ≥ 2부터)
- `0.5` — 1/√n_obs로 감소하는 신규 가설 초기 부스트
- `η = 0.002` — 복잡도 패널티
- `complexity` — `w`의 비영 항목 수
- `family_fitness` — 모든 후손 가설들의 평균 fitness로, 우수한 유전 계통을 전파하는 것에 보상을 줍니다.

풀은 점수 내림차순으로 정렬된 후 다양성 pruning과 crossover 선택에 사용.

---

## 학습 루프

### 초기화

```
pool ← D개의 축 정렬 기저 가설 (영구 보존)
F    ← 로그-사전 확률 점수 (균등 클래스 로그 확률), shape [N, K]
```

### 라운드별 (m = 1 … n_estimators)

**1. 그래디언트 계산 (소프트맥스 교차 엔트로피)**

```
P = softmax(F)
G = sample_weight ⊙ (P − one_hot(y))    # [N, K]
H = sample_weight ⊙ P ⊙ (1 − P)        # [N, K]
```

**2. 풀 진화** (매 `evolve_every` 라운드마다, 최대 10,000행 랜덤 서브샘플)

아래 [풀 진화](#풀-진화) 참조.

**3. 사선 트리 빌드 (C++)**

```
Z          = pool.eval(X)        # [P, N] — 모든 가설 투영값
thresholds = pool.thresholds     # [9, P] — 사전 계산된 분위수 임계값

tree.bfs_build(Z, thresholds, G[sub], H[sub], max_depth, reg_lambda)
```

각 노드는 XGBoost 스타일 gain을 최대화하는 (가설, 임계값) 쌍 선택:

```
gain = 0.5 × (G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ))
```

**4. 업데이트**

```
F += learning_rate × tree.predict(Z)
pool.increment_use_counts(tree.split_indices)
```

**5. 검증 & 조기 종료**

`eval_set` 제공 시: val loss 계산, `best_val_loss` 추적. 개선되면 `(best_trees, best_pool_snaps)` 스냅샷 저장. `no_improv ≥ early_stopping_rounds`면 최적 스냅샷으로 복원 후 종료.

---

## 풀 진화

`pool.evolve()`는 매 라운드 랜덤 서브샘플에서 실행.

### Step 1 — 적합도 스캔 & UCB 업데이트

풀의 모든 가설에 대해 서브샘플에서 split-gain 적합도 계산(feedback_mode="scan_only"), Welford 업데이트. 이 통계를 글로벌 history DAG에 동기화하고 계통 메트릭(모든 history 가설들에 대해 family fitness는 후손들의 평균 fitness, breeding value는 직계 자식들의 평균 fitness로 계산)을 계산하고 점수 산출:
```
score = ucb_score + family_lambda × family_fitness
```

풀을 점수 내림차순으로 정렬.

### Step 2 — Crossover 교배

1. 부모 선택 점수 `parent_selection_score = fitness + breeding_beta * breeding_value` 기준 내림차순 정렬.
2. 상위 `crossover_top_k` 부모 추출.
3. 모든 가능한 부모 쌍을 형성하고, 역사적 조합 생존율(라플라스 스무딩 적용)에 비례하는 crossover 선택 가중치 계산:
```
weight(Ta, Tb) = (survivors[Ta][Tb] + 1) / (births[Ta][Tb] + 2)
```
4. 가중치 기반으로 룰렛 휠 선택(Roulette Wheel Selection)으로 부모 쌍을 샘플링. 각 샘플링된 쌍에 대해 자손 생성:
```
α       ~ Uniform(0.3, 0.7)
w_자손  = α × wᵢ + (1−α) × wⱼ
w_자손  = sparsify(w_자손)
type    = 점수가 높은 부모의 type
family_id = 점수가 높은 부모의 family_id
```

### Step 3 — 후보 적합도 스캔 & 입장 허가

후보마다 서브샘플에서 split-gain 계산. `gain > 1e-5`인 것만 입장 허가 및 글로벌 history에 추가(그래디언트 정렬 시 `family_id = global_id`로 새로운 계통 부여). 전체 투영값 계산 및 캐시: `full_cache[N] = apply_op(X @ w, type)`.

### Step 4 — 재점수화 & 정렬

기존 + 새로 입장한 모든 가설의 UCB 점수 및 family 보너스 재계산, 내림차순 정렬.

### Step 5 — ALPS 우선순위 부스트

`alps_mode=True`일 때: `n_obs < 5`인 가설의 점수에 일시적으로 `+1e6` 추가 후 정렬. UCB 추정이 안정화되기 전에 제거되지 않도록 보호. 재정렬 후 부스트 제거.

### Step 6 — 다양성 dedup 및 계통 쿼터 (Family Quota)

서브 인덱스 투영값으로 탐욕적 선택. 점수 최고 가설부터 시작. 이후 각 후보에 대해 계통 쿼터와 코사인 유사도 검증:
1. **계통 쿼터 (Family Quota)**: 이미 선택된 가설 중 동일한 `family_id`를 가진 가설의 수가 `family_max_size` (기본값 30)에 도달하면 해당 후보 스킵.
2. **코사인 유사도 (Cosine Similarity)**: 이미 선택된 가설들과의 절대 코사인 유사도가 같은 타입인 경우 `0.90` 초과, 다른 타입인 경우 `0.98` 초과 시 스킵.

`|kept| = pool_size (400)` 도달 시 종료.

### Step 7 — 엘리트주의

점수 기준 상위 `elitism_k = 20`개는 유사도 및 계통 쿼터 제거 단계에서 면제. 중복 여부와 무관하게 항상 보존.

### Step 8 — MAP-Elites 타입 쿼터

dedup 후, 타입당 슬롯 제한 적용: 각 타입 최대 `map_elites_slots = 100`개.

### Step 9 — 생존 필터 및 연산자 전이 행렬 업데이트 (Transition Update)

기저 가설이 아닌 가설 중 `n_obs >= 3`이고 `mu_fitness + score <= 1e-6`인 가설 제거. 이번 라운드에 입장한 crossover 후보들과 이들의 eviction 생존 여부를 기반으로 transition matrix의 births(출생)와 survivors(생존) 업데이트.

---

## 추론

```python
F = tile(log_prior, (N, 1))

for tree, pool_snap in zip(trees_, pool_snaps_):
    Z  = pool_snap.eval(X)         # [P_snap, N]
    F += learning_rate × tree.predict(Z)

return softmax(F)
```

각 트리는 자신만의 풀 스냅샷과 쌍을 이룬다. C++ 트리는 빌드 시점의 풀 인덱스를 정수로 저장하기 때문에, 최종 풀을 모든 트리에 쓰면 잘못된 가설에 매핑된다.

---

## C++ 인터페이스

모든 풀 연산은 `bfstree.cpp`(C++17, OpenMP)에서 실행. Python은 ctypes 브리지(`_pool.py`)로 호출.

| 심볼 | 역할 |
|------|-----|
| `pool_create(D, max_size, evolve_mode)` | D개 기저 가설로 풀 생성 |
| `pool_set_options(…)` | 풀 설정 옵션 지정 (op_mode, crossover_top_k, elitism_k, alps_mode, map_elites_slots, family_max_size, enable_meta_evolution, family_lambda, breeding_beta) |
| `pool_evolve(…)` | 전체 진화 파이프라인 실행 |
| `pool_eval(…)` | 배치 Z = X에 모든 가설 적용 |
| `pool_get_caches_and_thresholds(…)` | 트리 빌드용 투영값 및 분위수 임계값 반환 |
| `pool_update_use_counts(…)` | 선택된 분할 인덱스의 use_count 증가 |
| `pool_export(…)` | 풀 상태 → numpy 배열 직렬화 (부모 및 혈통 정보 포함) |
| `pool_import(…)` | numpy 배열 → C++ 풀 복원 |
| `pool_get_policy_stats(…)` | Meta-Evolution 밴딧 텔레메트리 조회 |
| `pool_get_transition_matrix(…)` | Crossover 연산자 조합 성공률 조회 |
| `pool_free(handle)` | C++ 메모리 해제 |

---

## 성능

에블레이션 벤치마크 (3개 데이터셋: spiral-2D, checker-2D, hiD-20D; 시드 3개):

| 설정 | 평균 정확도 |
|-----|-----------|
| 베이스라인 (standard 모드, 풀 기능 없음) | 0.9309 |
| + crossover (top_k=3) | 0.9354 |
| + elitism_k=20 | 0.9358 |
| + alps_mode=True | 0.9361 |
| + map_elites_slots=100 | 0.9364 |
| + subsample=0.8 | 0.9376 |
| + pool_size=400 | **0.9394** |

최종 비교 (동일 3개 데이터셋, n_estimators=300):

| 모델 | 평균 정확도 |
|------|-----------|
| **HypForge** | **0.9338** |
| XGBoost | 0.9318 |
| LightGBM | 0.9308 |
| ExtraTrees | 0.9306 |
| GradBoost-sklearn | 0.9282 |
| RandomForest | 0.9236 |

HypForge는 고차원 데이터(hiD-20D: 0.9160 vs XGBoost 0.8967)에서 두드러진 우위. 축 정렬 분할로는 표현할 수 없는 특성 선형 결합을 사선 분할이 발견하기 때문.

---

## 계획된 개선사항

별도 설계 문서 참조:

- [`plan_contribution_based_survival_ko.md`](plan_contribution_based_survival_ko.md) — 가설별 검증 gain 추적; 과적합 방향에 UCB 페널티 부과.
- [`plan_latent_pool_ko.md`](plan_latent_pool_ko.md) — primitive(기저)와 compound(교배/복합) 가설 풀 분리 진화.
