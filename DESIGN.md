# 메타토큰 차이 기반 커리큘럼 러닝 설계

## 🎯 핵심 아이디어

**클라우드 LLM과 엣지 LLM의 레이어별 메타토큰 차이를 난이도 지표로 활용**

---

## 📐 메타토큰 차이 계산 방법

### 1. 메타토큰 추출
```
입력 샘플 x → Cloud LLM (LLaMA-3-8B) → T_cloud = {t^cloud_1, ..., t^cloud_L}
                → Edge LLM (Qwen2.5-1.5B) → T_edge = {t^edge_1, ..., t^edge_L}
```

**시스템 프롬프트**: 동일한 프롬프트를 양쪽 모델에 주입
- "Analyze the following dialogue and generate embeddings for task understanding."
- 대화 컨텍스트 + 현재 발화를 입력

### 2. 레이어별 차이 계산

각 레이어 i에 대해:
```
d_i(x) = ||t^cloud_i - t^edge_i||_2  (L2 distance)
```

**대안 메트릭**:
- Cosine Distance: `d_i(x) = 1 - cos(t^cloud_i, t^edge_i)`
- KL Divergence: Softmax 후 분포 차이 (더 민감)
- Mahalanobis Distance: 공분산 고려 (더 robust)

### 3. 종합 난이도 점수

**옵션 A: 평균 차이**
```
Difficulty(x) = (1/L) * Σ d_i(x)
```

**옵션 B: 가중 평균 (하위 레이어 중요도 높임)**
```
Difficulty(x) = Σ w_i * d_i(x)
where w_i = exp(-i/τ) / Z  (초반 레이어 가중치 높음)
```

**옵션 C: 최대 차이 (가장 어려운 레이어 기준)**
```
Difficulty(x) = max_i d_i(x)
```

**옵션 D: 상위-K 평균 (가장 큰 K개 차이)**
```
Difficulty(x) = (1/K) * Σ top_k(d_i(x))
```

**✅ 추천: 옵션 D (Top-K 평균)**
- 이유: 특정 레이어에서 큰 차이가 있으면 해당 샘플이 어렵다는 신호
- K=3~5 정도로 설정 (전체 레이어의 ~20%)

---

## 🎓 커리큘럼 전략

### 전략 1: Easy-to-Hard (기본)
```
난이도 순으로 정렬: Difficulty(x_1) ≤ ... ≤ Difficulty(x_N)
학습 순서: x_1 → x_2 → ... → x_N
```

### 전략 2: Layer-Wise Progressive (레이어별 점진적)

**Idea**: 레이어마다 난이도가 다르므로, 레이어별로 커리큘럼 적용

**Stage 1 (초기 레이어 집중)**: 
- d_1, d_2, d_3가 작은 샘플부터 학습
- LoRA Generator의 초기 레이어 가중치 먼저 학습

**Stage 2 (중간 레이어)**:
- d_4, ..., d_L/2가 작은 샘플 추가
- 점진적으로 모든 레이어 활성화

**Stage 3 (전체 레이어)**:
- 모든 레이어 고려, 어려운 샘플 포함
- 전체 d_avg가 큰 샘플까지 학습

```python
# Stage schedule
Stage 1: layers [0:L//3],    난이도 [0%, 33%]
Stage 2: layers [0:2*L//3],  난이도 [0%, 66%]
Stage 3: layers [0:L],       난이도 [0%, 100%]
```

### 전략 3: Dynamic Pacing (동적 속도 조절)

**Idea**: 현재 손실에 따라 난이도 증가 속도 조절

```python
if current_loss < threshold:
    difficulty_percentile += 5%  # 더 어려운 샘플 추가
else:
    difficulty_percentile -= 2%  # 난이도 낮춤 (복습)
```

### 전략 4: Layer-Specific Curriculum (레이어별 독립 커리큘럼)

**Idea**: 각 레이어가 독립적으로 학습

```python
for layer_idx in range(L):
    # 레이어 i에 대해 d_i가 작은 순서로 학습
    samples_sorted_by_layer_i = sort(samples, key=lambda x: d_i(x))
    train_on_layer(layer_idx, samples_sorted_by_layer_i)
```

**문제**: 레이어 간 상호작용 무시
**해결**: Hierarchical Training
- Phase 1: 레이어별 독립 학습
- Phase 2: Joint 학습 (모든 레이어)

---

## 🔬 실험 설계

### Baseline
- **Random Sampling**: 샘플 순서 무작위

### Ours - Variant 1: Easy-to-Hard
- Top-K 평균 차이로 정렬
- 쉬운 것부터 어려운 것까지

### Ours - Variant 2: Layer-Wise Progressive
- 3 Stage (초기/중간/전체 레이어)
- 각 Stage마다 난이도 33%씩 증가

### Ours - Variant 3: Dynamic Pacing
- 손실 기반 난이도 조절
- Adaptive threshold

### Ours - Variant 4: Hybrid
- Layer-Wise Progressive + Dynamic Pacing
- 가장 정교한 방법

---

## 📊 평가 지표

### 1. 학습 효율
- **Convergence Speed**: 동일 성능 도달 시간
- **Sample Efficiency**: 동일 성능에 필요한 샘플 수

### 2. 최종 성능
- **Validation Loss**: Phase 3 종료 시점
- **Generalization**: 테스트셋 성능

### 3. 레이어별 분석
- 각 레이어의 LoRA 품질
- 레이어별 수렴 속도

---

## 💾 데이터 구조

### 난이도 레이블 파일
```json
{
  "sample_id": "personachat_train_00001",
  "meta_token_diff": {
    "layer_0": 0.23,
    "layer_1": 0.31,
    ...
    "layer_31": 0.89
  },
  "difficulty_scores": {
    "mean": 0.45,
    "max": 0.89,
    "top3_mean": 0.76,
    "weighted_mean": 0.52
  },
  "difficulty_percentile": 0.67,  // 전체 중 67%
  "curriculum_stage": 2  // Stage 2에 속함
}
```

---

## 🎯 구현 우선순위

1. ✅ **Phase 1**: Easy-to-Hard (가장 단순)
   - Top-K 평균 차이로 정렬
   - 검증 후 다음 단계로

2. **Phase 2**: Layer-Wise Progressive
   - 3-Stage 커리큘럼
   - Ablation study

3. **Phase 3**: Dynamic Pacing
   - 적응형 난이도 조절
   - 최적 threshold 탐색

4. **Phase 4**: Hybrid
   - 모든 전략 통합
   - 최종 성능 극대화

---

## 🔧 하이퍼파라미터

```yaml
meta_token:
  distance_metric: "l2"  # l2, cosine, kl
  top_k: 3  # Top-K layers for difficulty
  system_prompt: "Analyze the dialogue context and generate task embeddings."

curriculum:
  strategy: "layer_wise_progressive"  # easy_to_hard, layer_wise, dynamic, hybrid
  num_stages: 3
  stage_percentiles: [0.33, 0.66, 1.0]
  
  # Dynamic pacing
  loss_threshold: 2.0
  difficulty_step_up: 0.05
  difficulty_step_down: 0.02

training:
  curriculum_start_epoch: 0  # 커리큘럼 시작 epoch
  curriculum_end_epoch: 5    # 이후 전체 데이터 사용
  warmup_epochs: 1  # 첫 N epoch은 쉬운 데이터만
```

---

## 📈 예상 결과

### 가설
1. **H1**: 메타토큰 차이가 큰 샘플 = 실제 어려운 샘플
   - 검증: 난이도와 학습 손실 상관관계 분석

2. **H2**: 커리큘럼 러닝으로 수렴 속도 향상
   - 예상: 20-30% 빠른 수렴

3. **H3**: Layer-Wise Progressive가 최적
   - 이유: LoRA-Gen의 레이어별 구조와 정렬

### 위험 요소
- **Risk 1**: Cloud/Edge 메타토큰 차이가 난이도를 반영하지 않을 수 있음
  - Mitigation: 여러 메트릭 실험 (L2, Cosine, KL)

- **Risk 2**: 너무 쉬운 샘플만 학습 → 과소적합
  - Mitigation: Warmup 후 점진적 난이도 증가

- **Risk 3**: 레이어별 독립 학습 → 레이어 간 불균형
  - Mitigation: Phase 2에서 Joint 학습
