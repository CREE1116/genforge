# SALOT — 이론 문서

> Pool-free Local Stochastic Oblique Boosting (PLSOB)  
> 구현: `src/hypforge/_ext/salot.cpp`

---

## 1. 문제 설정

K-클래스 분류 문제에서 gradient boosting은 라운드 m마다 잔차를 설명하는 약학습기 $f_m$을 추가한다.

$$F_m(x) = F_{m-1}(x) + \eta \cdot f_m(x), \quad F_m \in \mathbb{R}^K$$

각 라운드에서 multinomial logistic loss의 1차/2차 도함수를 구한다:

$$g_i^{(k)} = p_i^{(k)} - \mathbf{1}[y_i = k], \qquad h_i^{(k)} = p_i^{(k)}(1 - p_i^{(k)})$$

트리 $f_m$의 리프값은 Newton step으로 설정한다:

$$v_\ell^{(k)} = -\frac{\sum_{i \in \ell} g_i^{(k)}}{\sum_{i \in \ell} h_i^{(k)} + \lambda}$$

---

## 2. 노드별 Block WLS (Per-Node Block WLS)

### 2.1 기존 접근과의 차이

| 방식                   | 분기 방향        | 특성                    |
| ---------------------- | ---------------- | ----------------------- |
| 축-정렬 (CART)         | 단일 피처        | O(DN log N) 탐색        |
| 레이어 공유 WLS        | 레이어 전체 공유 | 레이어 내 다양성 부족   |
| **Per-Node Block WLS** | 노드 독립        | 노드마다 다른 사선 방향 |

### 2.2 방향 벡터 도출

노드 $t$에서 $d$차원 피처 서브셋 $\mathcal{F} \subset [D]$를 샘플링한다 ($d = \max(2, \min(d_{\max}, \lceil\sqrt{D}\rceil))$).

WLS 인스턴스 서브셋 $\mathcal{S}_t \subseteq$ (노드 샘플)를 최대 $n_{\text{wls\_max}}$개 랜덤 샘플링한다.

대표 클래스 선택 (K-클래스 병목 해소):

$$c^* = \arg\max_{k} \sum_{i \in \mathcal{S}_t} |g_i^{(k)}|$$

$d \times d$ Hessian 행렬과 음의 gradient 벡터를 구성한다:

$$A_{jl} = \sum_{i \in \mathcal{S}_t} h_i^{(c^*)} x_{i,f_j} x_{i,f_l}, \qquad r_j = -\sum_{i \in \mathcal{S}_t} g_i^{(c^*)} x_{i,f_j}$$

$(A + (\lambda_0 + \lambda_{\text{adapt}})I)\,w = r$를 Cholesky 분해로 풀어 방향 벡터 $w \in \mathbb{R}^d$를 얻는다.

---

## 3. GBAOR: Gershgorin Bound Adaptive Oblique Regularization

### 3.1 동기

$A$의 최소 고유값이 작으면 (노드 샘플 수 적거나 피처 간 상관이 높을 때) 정규화 $\lambda_0$만으로는 부족하다. 고유값 분해 없이 $\lambda_{\min}(A)$를 하한 추정하는 방법이 Gershgorin 원 정리다.

### 3.2 Gershgorin 원 정리

행렬 $A$의 임의 고유값 $\mu$에 대해:

$$\mu \in \bigcup_{j=1}^{d} \{ z \in \mathbb{C} : |z - A_{jj}| \leq R_j \}, \qquad R_j = \sum_{k \neq j} |A_{jk}|$$

따라서:

$$\hat{\lambda}_{\min} = \min_j (A_{jj} - R_j) \leq \lambda_{\min}(A)$$
$$\hat{\lambda}_{\max} = \max_j (A_{jj} + R_j) \geq \lambda_{\max}(A)$$

### 3.3 적응적 정규화 계산

목표: $\lambda_{\min}(A + \lambda_{\text{adapt}} I) \geq \alpha \cdot \lambda_{\max}(A + \lambda_{\text{adapt}} I)$, 즉 조건수 $\leq 1/\alpha$.

$$\lambda_{\text{adapt}} = \max\!\left(0,\ \frac{\alpha \cdot \hat{\lambda}_{\max} - \hat{\lambda}_{\min}}{1 - \alpha}\right)$$

최종 솔버에 사용하는 정규화:

$$\lambda_{\text{total}} = \lambda_0 + \lambda_{\text{adapt}}$$

**핵심 성질**: 샘플이 적은 깊은 노드는 $A$가 자동으로 더 불량조건이 되어 $\lambda_{\text{adapt}}$가 커진다. 깊이별 스케줄링 없이 데이터 통계로 자동 적응한다.

**계산 비용**: Gershgorin 스캔은 $O(d^2)$로, 이미 구성한 $A$ 행렬의 행합을 한 번 더 훑는 수준이다.

---

## 4. 다중 후보 WLS (Multi-Candidate WLS)

### 4.1 단일 후보의 한계

피처 서브셋을 단 한 번만 샘플링하면 운이 나쁜 경우 (정보 피처가 서브셋에서 제외) 분기 품질이 저하된다. 노드별 독립 RNG를 사용하는 현재 구조에서 이 문제를 간단히 해결할 수 있다.

### 4.2 알고리즘

노드 $t$에서 `n_candidates` 번 반복:

1. 새로운 랜덤 피처 서브셋 $\mathcal{F}^{(c)}$ 샘플링
2. Block WLS로 방향 $w^{(c)}$ 도출
3. $w^{(c)}$로 노드 샘플을 1D 투영 후 히스토그램 gain 계산:

$$G^{(c)} = \max_b \left[ \sum_k \frac{(G_k^L)^2}{H_k^L + \lambda} + \frac{(G_k^R)^2}{H_k^R + \lambda} - \frac{(G_k)^2}{H_k + \lambda} \right]$$

4. $c^* = \arg\max_c G^{(c)}$인 방향을 채택

라우팅은 $w^{(c^*)}$로 재투영한다.

**계산 비용**: WLS $O(d^3)$ × n*candidates + 재투영 $O(n \cdot D*{\text{num}})$ 1회. `n_candidates=3`이면 오버헤드가 작다.

---

## 5. Zero-Cross-Term Feature Bundling

### 5.1 목적

원-핫 인코딩된 범주형 피처처럼 상호 배타적인 피처 쌍은 동시에 0이 아닐 수 없으므로 $A_{jl} = \sum_i h_i x_{ij} x_{il} = 0$이다. 이를 미리 파악하고 누적을 건너뛴다.

### 5.2 배타성 판단

피처 $j, l$의 co-occurrence rate:

$$\text{cooc}(j,l) = \frac{|\{i : x_{ij} \neq 0 \wedge x_{il} \neq 0\}|}{N_{\text{node}}}$$

$\text{cooc}(j,l) < 0.05$이면 배타적으로 간주하고 $A_{jl} = 0$ 처리.

### 5.3 두 가지 경로

| 조건                      | 방식                                         | 비용                                   |
| ------------------------- | -------------------------------------------- | -------------------------------------- |
| $D_{\text{num}} \leq 150$ | 학습 시작 시 $D \times D$ 전역 행렬 1회 계산 | $O(N \cdot D^2)$ 선불                  |
| $D_{\text{num}} > 150$    | 노드별 $d \times d$ 로컬 체크                | $O(n_{\text{wls}} \cdot d^2)$ per node |

---

## 6. Energy Pruning

WLS로 구한 $w \in \mathbb{R}^d$에서 에너지 비율이 작은 성분을 제거한다.

**에너지 분율**: $\|w\|^2$를 내림차순 정렬 후 누적합이 `energy_frac` × $\|w\|^2$에 도달하는 최소 성분 집합만 유지한다.

이후 $w$를 단위 벡터로 정규화한다 ($\|w\| = 1$).

**D ≤ 4 보호**: 저차원에서는 pruning이 과도하게 축-정렬로 퇴화시키므로 `prune_strength`를 자동으로 0으로 설정한다.

---

## 7. Honest Split (정직한 분할)

Athey & Imbens (2016)의 정직한 추정 기법을 사선 부스팅에 적용한다.

### 7.1 Data Reuse Bias

기존 트리는 동일 데이터로 분기 방향 탐색(구조)과 분기 이득 평가(검증)를 동시에 수행한다. 사선 트리는 탐색 자유도가 높아 이 편향이 더 크게 작용할 수 있다.

### 7.2 메커니즘

노드 샘플을 두 집합으로 분리한다:

- $\mathcal{S}_s$ (structure, 앞쪽 절반): WLS 방향 탐색에만 사용
- $\mathcal{S}_e$ (estimation, 뒤쪽 절반): Gt/Ht, 히스토그램 gain 계산, 임계값 결정에만 사용

라우팅은 전체 샘플 $\mathcal{S}_s \cup \mathcal{S}_e$를 사용하므로 자식 노드 크기가 유지된다.

$$\text{Gain}_{\text{honest}}(\tau) = \frac{1}{2} \left[ \frac{(G_e^L)^2}{H_e^L + \lambda} + \frac{(G_e^R)^2}{H_e^R + \lambda} - \frac{(G_e)^2}{H_e + \lambda} \right]$$

### 7.3 실험적 한계

에블레이션 결과 (→ `exp_ablation_regularization.md`):

| 기법 | linear50d vs baseline |
|------|----------------------|
| GBAOR only | +1.66% |
| honest only | **-1.50%** |
| GBAOR+honest | +0.66% |

소규모 데이터 (≤2000 샘플)에서 절반 분할은 편향 감소보다 분산 증가가 크다. **기본값 `honest_split=False`**, 대규모 데이터셋에서만 시험 권장.

---

## 8. 전체 노드 처리 흐름

```
노드 t 처리 (BFS 순서):
│
├─ [honest=True]  samp → samp_s (앞 절반) + samp_e (뒤 절반)
├─ [honest=False] samp_s = samp_e = samp
├─ wls_samp = subsample(samp_s, n_wls_max)
├─ Gt / Ht 계산 (samp_e)
│
└─ for cand in range(n_candidates):
   ├─ 피처 서브셋 샘플링 (feat_sub)
   ├─ [D_num > 150] 로컬 번들 체크 (d×d)
   ├─ Block WLS → w_cand  (wls_samp 사용)
   │   ├─ A 행렬 구성 (번들링 적용)
   │   ├─ GBAOR: Gershgorin 스캔 → λ_adapt
   │   ├─ A += (λ_0 + λ_adapt) I
   │   ├─ Cholesky 분해 + 전진/후진 대입
   │   └─ Energy pruning + 정규화
   ├─ 1D 투영 (samp_e) → 히스토그램 gain 계산
   └─ best_gain 갱신

→ 최우수 w로 samp 전체 재투영 → 샘플 라우팅
```

---

## 9. 복잡도 요약

노드 1개당 비용 (n = 노드 샘플 수, d = d_sub_max, n_w = n_wls_max):

| 단계                            | 비용                    |
| ------------------------------- | ----------------------- |
| 인스턴스 서브샘플               | O(n)                    |
| Gt/Ht 계산                      | O(n × K)                |
| A 행렬 구성 (×n_candidates)     | O(n_w × d²)             |
| GBAOR Gershgorin 스캔           | O(d²)                   |
| Cholesky 분해                   | O(d³)                   |
| Energy pruning                  | O(d log d)              |
| 히스토그램 gain (×n_candidates) | O(n × D_num + BINS × K) |
| 재투영                          | O(n × D_num)            |

$d \ll D_{\text{num}}$이므로 전체 비용은 $O(n_{\text{candidates}} \cdot (n_w d^2 + d^3) + n \cdot D_{\text{num}})$.
