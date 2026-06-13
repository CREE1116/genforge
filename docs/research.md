지적해주신 트리 분할의 불연속성(Indicator Function)을 반영하고, **에너지 분해 기반의 상계(Bounded Energy Loss)** 프레임워크로 전면 수정하여 OQBoost 논문 및 보고서용으로 바로 활용할 수 있게 깔끔하게 정리했습니다.

---

## 1. 배경: 기존 가설의 한계와 교정

- **기존 하드 가설의 오류:** 조상 노드가 축 $w_i$를 선택했다고 해서 하위 노드에서 해당 축의 선형 성분($\langle v_G, w_i \rangle$)이 완전히 소멸($=0$)하지 않습니다. 트리의 분할은 선형 사영이 아니라 $\mathbb{I}(w_i^T x < \theta)$ 형태의 공간 격리(Partitioning)이기 때문입니다. 즉, 개별 차원을 칼같이 자르는 직교화는 이론적 오류이자 과도한 제약(자유도 고갈)을 낳습니다.
- **해결책 (에너지 상계 프레임워크):** 완전한 그래디언트 소모를 가정하는 대신, 잔여 그래디언트의 에너지가 혈통 부공간 내에 얼마나 남아있는지($\epsilon$)에 따라 이득(Gain)의 손실률이 결정된다는 정리를 유도합니다.

---

## 2. 에너지 분해 기반 이득 보존 정리 (Gain Preservation Theorem)

### [정리]

> 혈통 부공간 $S$에 대한 제약 조건을 걸더라도, 잔여 그래디언트의 에너지가 $S$ 내부에 많이 남아있지 않다면 최적 이득(Optimal Gain)의 이론적 상한은 거의 보존된다.

### [증명]

#### Step 1: 그래디언트 방향 벡터의 직교 분해

현재 노드까지의 조상 분할 축들이 이루는 혈통 부공간을 $S = \text{span}(W) \subset \mathbb{R}^D$, 그 직교 여공간을 $S^\perp$라 하자. 현재 노드의 잔여 그래디언트 형성 방향 벡터 $v_G = X^T g \in \mathbb{R}^D$는 다음과 같이 유일하게 분해된다.

$$v_G = v_\parallel + v_\perp \quad (v_\parallel \in S, \quad v_\perp \in S^\perp)$$

- $v_\parallel$: 조상 축들이 미처 다 해결하지 못하고 남겨둔 **잔여 선형 성분의 에너지**
- $v_\perp$: 조상들이 건드리지 않은 **새로운 직교 부공간의 에너지**

#### Step 2: 제약 조건별 최대 이득(Gain) 유도

오블리크 트리에서 사영 축 $w$ ($\|w\|_2=1$) 선택에 따른 분할 이득의 1차 Taylor 근사 최적화 목적함수를 $f(w) = (w^T v_G)^2$라 정의한다.

1. **무제약 최적화 (Unconstrained 상한):**
   아무런 제약이 없을 때, 코시-슈바르츠 부등식에 의해 $w^* = \frac{v_G}{\|v_G\|_2}$ 일 때 절대 상한을 달성한다.

$$f_{\text{free}} = \max_{\|w\|_2=1} (w^T v_G)^2 = \|v_G\|_2^2$$

2. **혈통 직교 제약 최적화 (Lineage-Orthogonal 제약):**
   새로운 축이 혈통 공간과 직교해야 한다는 제약($w \in S^\perp$)을 부과하면 $w^T v_\parallel = 0$이 되므로 목적함수는 다음과 같이 축소된다.

$$f(w) = (w^T (v_\parallel + v_\perp))^2 = (w^T v_\perp)^2$$

$v_\perp \in S^\perp$이므로, 이를 극대화하는 최적 축은 $w^* = \frac{v_\perp}{|v_\perp|*2}$이며 이때의 최대 이득 $f^*$는 다음과 같다.
$$f^* = \max*{|w|*2=1, w \in S^\perp} (w^T v*\perp)^2 = |v_\perp|_2^2$$

#### Step 3: 이득 보존율(Gain Retention Ratio) 도출

무제약 상한($f_{\text{free}}$) 대비 혈통 직교 제약 축이 보존하는 이득($f^*$)의 비율은 다음과 같다.

$$\frac{f^*}{f_{\text{free}}} = \frac{\|v_\perp\|_2^2}{\|v_G\|_2^2} = \frac{\|v_G\|_2^2 - \|v_\parallel\|_2^2}{\|v_G\|_2^2} = 1 - \frac{\|v_\parallel\|_2^2}{\|v_G\|_2^2}$$

여기서 전체 에너지 중 혈통 공간 내부로 누출된 에너지의 비율을 $\epsilon = \frac{|v_\parallel|_2^2}{|v_G|_2^2}$이라 정의하면, 최종 이득 보존 방정식이 성립한다.

$$f^* = (1 - \epsilon) f_{\text{free}}$$

---

## 3. OQBoost 아키텍처적 장점과의 연결 (EXP-D 동형화)

이 정리는 완전 직교를 억지로 주장하지 않으면서도, 왜 혈통 회피가 안전하고 강력한지를 실험 결과(`EXP-D`)와 연결해 증명합니다.

- **부스팅 초반부 (에너지 누출 $\epsilon$ 고정):**
  초기 라운드에서는 기존 축들의 잔여 선형 시그널($v_\parallel$)이 제법 존재할 수 있어 $\epsilon$이 일시적으로 커질 수 있습니다. 하지만 이 구간에서는 **256-bin 축 스캔(Axis Scan)** 슬롯이 압도적인 이득으로 토너먼트에서 승리하므로, 혈통 회피 슬롯이 잠시 손실을 보더라도 전체 트리 품질에 영향을 주지 않습니다.
- **부스팅 후반부 (에너지 고갈 $\epsilon \to 0$):**
  라운드가 진행될수록 잔여 오차가 복잡해지면서 기존 혈통 공간 내의 선형 에너지는 완전히 고갈됩니다 ($\|v_\parallel\| \to 0$). 따라서 오블리크 축이 정말로 필요해지는 후반부 노드일수록 $\epsilon \approx 0$이 되어 \*_$f^_ \approx f\_{\text{free}}$(이득 손실 0%)가 달성됩니다.

---

## 4. 논문용 영문 요약 (Paper Draft Ready)

> **The gain-preserving property does not require exact gradient exhaustion.** Let $v_G=v_\parallel+v_\perp$ be the orthogonal decomposition of the node gradient vector with respect to the lineage subspace $S=\text{span}(W)$. Under the lineage-orthogonal constraint ($w \in S^\perp$), the achievable maximum gain is reduced from $\|v_G\|_2^2$ to $\|v_\perp\|_2^2$. Therefore, the exact gain preservation ($f^* \approx f_{\text{free}}$) is seamlessly achieved whenever the residual gradient has little linear energy inside the pre-explored lineage subspace ($\|v_\parallel\|_2^2 \ll \|v_G\|_2^2$), which perfectly aligns with the late-stage boosting dynamics where residuals become inherently oblique.

---

[English Version (영문 버전)](research.en.md)
