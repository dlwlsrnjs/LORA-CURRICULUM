# Curriculum Learning Experiments

## 실험 로그 및 결과

### Baseline (Random Sampling)
- **날짜**: 
- **설정**: 
- **결과**:

### Experiment 1: Easy-to-Hard
- **날짜**: 
- **설정**: 
  - Strategy: easy_to_hard
  - Difficulty metric: topk_mean
  - Top-K: 3
- **결과**:
  - 수렴 속도:
  - 최종 Loss:
  - Val Accuracy:

### Experiment 2: Layer-Wise Progressive
- **날짜**: 
- **설정**: 
  - Strategy: layer_wise_progressive
  - Stages: 3
  - Stage percentiles: [0.33, 0.66, 1.0]
- **결과**:

### Experiment 3: Dynamic Pacing
- **날짜**: 
- **설정**: 
- **결과**:

### Experiment 4: Hybrid
- **날짜**: 
- **설정**: 
- **결과**:

---

## Ablation Studies

### Metric Comparison
- L2 distance:
- Cosine distance:
- KL divergence:

### Top-K Comparison
- K=1:
- K=3:
- K=5:

### Stage Number Comparison
- 2 stages:
- 3 stages:
- 4 stages:

---

## Observations

### 메타토큰 차이 vs 실제 난이도
- 상관계수:
- Scatter plot: [링크]

### 레이어별 분석
- 초기 레이어:
- 중간 레이어:
- 후기 레이어:

### 학습 곡선
- [그래프 첨부]

---

## 결론

### 최적 전략

### 주요 발견

### 향후 연구
