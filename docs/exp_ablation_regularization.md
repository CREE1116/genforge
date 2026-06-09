# 정규화 기법 에블레이션 (2025-06-09)

**설정**: n_estimators=300, learning_rate=0.05, max_depth=4, reg_lambda=1.0, subsample=0.8, n_candidates=3, early_stopping_rounds=30, random_state=42  
**데이터**: sklearn 합성 데이터, test_size=0.3

## 결과

| Config            | moons  | circles | linear20d | linear50d |
|-------------------|--------|---------|-----------|-----------|
| baseline (no reg) | 0.9917 | 0.9917  | 0.9483    | 0.9167    |
| GBAOR only        | 0.9917 | 0.9933  | **0.9617**| **0.9333**|
| prune only        | 0.9917 | 0.9917  | 0.9467    | 0.9083    |
| honest only       | 0.9900 | 0.9933  | 0.9517    | 0.9017    |
| GBAOR+prune       | 0.9917 | 0.9933  | 0.9467    | 0.9300    |
| GBAOR+honest      | 0.9917 | 0.9933  | 0.9517    | 0.9233    |
| prune+honest      | 0.9900 | 0.9933  | 0.9500    | 0.9000    |
| all three         | 0.9917 | 0.9933  | 0.9500    | 0.9167    |
| **LightGBM**      | 0.9883 | 0.9900  | 0.9017    | 0.8683    |

## 분석

### GBAOR
가장 강력한 단일 기법. linear20d +1.34%, linear50d +1.66%.  
고차원일수록 효과가 크다. 이유: 깊은 노드에서 Hessian이 자동으로 불량조건 → λ_adapt 증가. 별도 스케줄 없이 데이터 통계가 적응.

### Honest Split
**단독으로는 오히려 역효과.** linear50d에서 baseline 대비 -1.5%, moons도 소폭 하락.  
원인: 2000 샘플 × 0.7 = 1400 훈련 샘플을 절반으로 나누면 노드당 WLS/gain estimation에 사용하는 샘플이 ~700개 이하로 줄어든다. 소규모 데이터에서는 편향 감소보다 분산 증가가 더 크다. GBAOR와 조합해도 GBAOR 단독 대비 개선 없음.

### Energy Pruning
단독으로는 베이스라인 대비 열세. GBAOR와 조합하면 linear50d에서 경쟁력 있지만 (0.9300 vs 0.9333) linear20d에서 역효과.  
추가 파라미터(`prune_strength`)를 관리해야 하는 비용 대비 편익이 제한적.

## 결론

| 기법 | 유지 여부 | 이유 |
|------|-----------|------|
| **GBAOR** | ✅ 유지 | 명확하고 일관된 개선, O(d²) 오버헤드, 파라미터 1개 |
| Energy Pruning | ⚠️ 선택적 | 저차원 보호(D≤4 auto-disable)가 유용하지만 단독 효과 미미 |
| Honest Split | ❌ 기본 비활성 | 소규모 데이터에서 역효과, 파라미터 복잡도 증가 |

**권장 기본 설정**: `gbaor_alpha=0.05`, `prune_strength=0.1`(저차원 보호 목적), `honest_split=False`
