# 프로젝트 요약: 메타토큰 차이 기반 커리큘럼 러닝

## 🎯 프로젝트 개요

**목표**: Cloud LLM과 Edge LLM의 레이어별 메타토큰 차이를 활용한 커리큘럼 러닝으로 LoRA 생성 모델의 학습 효율 향상

**핵심 아이디어**:
1. 동일한 시스템 프롬프트를 Cloud/Edge LLM에 입력하여 메타토큰 추출
2. 각 레이어별로 두 모델의 메타토큰 차이 계산
3. 차이가 큰 샘플 = 난이도 높음으로 가정
4. 난이도 순서로 커리큘럼 구성하여 학습 효율 향상

---

## 📁 프로젝트 구조

```
curriculum/
├── README.md                          # 전체 프로젝트 설명
├── DESIGN.md                          # 상세 설계 문서
├── EXPERIMENTS.md                     # 실험 로그
│
├── meta_token_difference.py           # 메타토큰 추출 및 차이 계산
│   ├── MetaTokenExtractor             # Cloud/Edge LLM에서 메타토큰 추출
│   ├── DifficultyScorer               # 난이도 점수 계산 (mean, max, top-k 등)
│   └── MetaTokenDifferenceAnalyzer    # 전체 데이터셋 분석
│
├── curriculum_scheduler.py            # 커리큘럼 스케줄러
│   ├── CurriculumConfig               # 설정 데이터클래스
│   ├── CurriculumSampler              # PyTorch Sampler
│   ├── LayerSpecificCurriculumScheduler  # 레이어별 독립 스케줄링
│   ├── CurriculumDataset              # Dataset wrapper
│   └── CurriculumTrainingScheduler    # Epoch별 전략 관리
│
├── train_with_curriculum.py           # 기존 학습에 커리큘럼 통합
│   └── create_curriculum_dataloader() # DataLoader 생성 헬퍼
│
├── configs/
│   └── curriculum_config.yaml         # 커리큘럼 설정
│
├── scripts/
│   ├── 01_label_difficulties.py       # 난이도 레이블링 파이프라인
│   └── 02_analyze_difficulties.py     # 난이도 분석 및 시각화
│
└── run_curriculum_pipeline.sh         # 전체 파이프라인 실행 스크립트
```

---

## 🔧 핵심 구현

### 1. 메타토큰 차이 계산

**MetaTokenExtractor**:
```python
# Cloud/Edge 모델 로드
extractor = MetaTokenExtractor(
    cloud_model_name="meta-llama/Meta-Llama-3-8B",
    edge_model_name="Qwen/Qwen2.5-1.5B",
    system_prompt="Analyze the dialogue...",
)

# 메타토큰 추출
cloud_tokens, edge_tokens = extractor.extract_meta_tokens(
    dialogue_history=["Turn 1", "Turn 2"],
    current_utterance="Current turn",
)

# 레이어별 차이 계산
layer_diffs = extractor.compute_layer_differences(
    cloud_tokens, edge_tokens, metric="l2"
)
# → {0: 0.23, 1: 0.31, ..., 31: 0.89}
```

**DifficultyScorer**:
```python
scorer = DifficultyScorer(num_layers=32, top_k=3)

# 여러 난이도 메트릭 계산
difficulty_scores = scorer.compute_difficulty_scores(layer_diffs)
# → {
#   'mean': 0.45,
#   'max': 0.89,
#   'topk_mean': 0.76,  # 가장 큰 3개 레이어 평균
#   'weighted_mean': 0.52,
# }
```

### 2. 커리큘럼 전략

**4가지 전략 구현**:

**A. Easy-to-Hard** (기본):
```python
config = CurriculumConfig(strategy="easy_to_hard")
# 난이도 순으로 정렬하여 학습
```

**B. Layer-Wise Progressive** (추천 ⭐):
```python
config = CurriculumConfig(
    strategy="layer_wise_progressive",
    num_stages=3,
    stage_percentiles=[0.33, 0.66, 1.0],
)
# Stage 1: 난이도 하위 33% + 초기 레이어
# Stage 2: 난이도 하위 66% + 중간 레이어
# Stage 3: 전체 데이터 + 모든 레이어
```

**C. Dynamic Pacing**:
```python
config = CurriculumConfig(
    strategy="dynamic_pacing",
    loss_threshold=2.0,
    difficulty_step_up=0.05,
)
# 손실에 따라 난이도 동적 조절
```

**D. Hybrid**:
```python
config = CurriculumConfig(strategy="hybrid")
# Layer-Wise + Dynamic 결합
```

### 3. 학습 통합

```python
from curriculum.train_with_curriculum import create_curriculum_dataloader
from curriculum.meta_token_difference import MetaTokenDifferenceAnalyzer

# 난이도 레이블 로드
difficulties = MetaTokenDifferenceAnalyzer.load_difficulties(
    'curriculum/data/difficulty_labels.json'
)

# 학습 루프
for epoch in range(num_epochs):
    # 커리큘럼 DataLoader 생성
    train_loader = create_curriculum_dataloader(
        base_dataset=train_dataset,
        difficulties=difficulties,
        config=curriculum_config,
        epoch=epoch,
        current_loss=avg_loss,
        batch_size=4,
        num_layers=32,
    )
    
    # 학습
    for batch in train_loader:
        # batch['difficulty_info'] 포함
        loss = model(batch)
        loss.backward()
        optimizer.step()
```

---

## 🚀 사용법

### Quick Start

```bash
# 1. 난이도 레이블링
python curriculum/scripts/01_label_difficulties.py \
    --data_path data/train.jsonl \
    --output_path curriculum/data/difficulty_labels.json \
    --cloud_model "meta-llama/Meta-Llama-3-8B" \
    --edge_model "Qwen/Qwen2.5-1.5B"

# 2. 난이도 분석
python curriculum/scripts/02_analyze_difficulties.py \
    --difficulty_path curriculum/data/difficulty_labels.json \
    --output_dir curriculum/analysis

# 3. 학습 (전체 파이프라인)
bash curriculum/run_curriculum_pipeline.sh
```

### 기존 코드에 통합

```python
# train_dialogue_lora.py에 추가
from curriculum.train_with_curriculum import create_curriculum_dataloader

# Phase 3 학습 부분 수정
if args.use_curriculum:
    train_loader = create_curriculum_dataloader(
        base_dataset=train_dataset,
        difficulties=difficulties,  # 사전 계산된 난이도
        config=curriculum_config,
        epoch=epoch,
        current_loss=avg_loss,
        batch_size=args.batch_size,
        num_layers=model.target_num_layers,
    )
else:
    train_loader = DataLoader(train_dataset, ...)
```

---

## 📊 실험 계획

### Phase 1: 기본 검증 (1-2주)
- [ ] Easy-to-Hard 구현 및 실험
- [ ] 메타토큰 차이 vs 실제 난이도 상관관계 검증
- [ ] Baseline 대비 수렴 속도 측정

**성공 기준**: 
- 상관계수 r > 0.5
- 수렴 속도 10% 이상 개선

### Phase 2: 레이어별 최적화 (2-3주)
- [ ] Layer-Wise Progressive 구현
- [ ] 3/4/5 스테이지 비교 실험
- [ ] 레이어별 수렴 속도 분석

**성공 기준**:
- Easy-to-Hard 대비 추가 5% 개선
- 레이어별 균형잡힌 수렴

### Phase 3: 동적 조절 (1-2주)
- [ ] Dynamic Pacing 구현
- [ ] 손실 임계값 최적화
- [ ] Hybrid 전략 실험

**성공 기준**:
- 최종 성능 Baseline 대비 20-30% 개선

### Phase 4: 대규모 검증 (1-2주)
- [ ] 전체 PersonaChat 데이터셋 적용
- [ ] Ablation study (메트릭, Top-K, 스테이지 수)
- [ ] 최종 벤치마크

---

## 📈 예상 결과

### 정량적 목표

| 지표 | Baseline | Easy-to-Hard | Layer-Wise | Hybrid |
|------|----------|--------------|------------|--------|
| 수렴 시간 | 100% | 85% | **70%** | 72% |
| 최종 Loss | 2.15 | 2.08 | **1.98** | 1.95 |
| Val Acc | 75.2% | 76.1% | **77.5%** | 77.8% |

### 정성적 기대

1. **메타토큰 차이가 실제 난이도를 반영함을 입증**
   - Cloud-Edge 격차 = 학습 난이도
   - 난이도 예측 정확도 > 70%

2. **레이어별 커리큘럼의 효과 검증**
   - LoRA-Gen의 레이어별 생성과 자연스럽게 정렬
   - 각 레이어가 최적 속도로 학습

3. **실용적 학습 효율 향상**
   - GPU 시간 20-30% 절감
   - 동일 성능 도달에 필요한 샘플 수 감소

---

## 🔬 주요 분석

### 1. 메타토큰 차이 분석
- **레이어별 분포**: 어느 레이어가 가장 차이가 큰가?
- **샘플별 분포**: 어떤 대화가 어려운가?
- **메트릭 비교**: L2 vs Cosine vs KL

### 2. 커리큘럼 효과 분석
- **학습 곡선**: Epoch별 손실 변화
- **수렴 속도**: 동일 성능 도달 시간
- **레이어별 수렴**: 각 레이어의 학습 속도

### 3. 전략 비교
- **4가지 전략 Ablation**
- **하이퍼파라미터 민감도**
- **최적 설정 탐색**

---

## 💡 핵심 Insight

### 1. 메타토큰 차이의 의미
> Cloud LLM과 Edge LLM의 메타토큰 차이는 단순한 모델 크기 격차가 아니라, 
> **Edge 모델이 해당 샘플을 처리하는 데 필요한 추가 노력**을 나타낸다.

### 2. 레이어별 커리큘럼의 장점
> LoRA-Gen은 레이어마다 독립적으로 LoRA를 생성한다.
> 따라서 **각 레이어가 자신의 난이도에 맞는 커리큘럼을 따르는 것**이 
> 전체 평균 난이도보다 효과적이다.

### 3. 동적 조절의 필요성
> 고정된 커리큘럼은 데이터 분포에 민감하다.
> **현재 학습 상태(손실)에 따라 난이도를 조절**하면 더 robust한 학습이 가능하다.

---

## 🎓 이론적 기여

1. **Meta-Token as Difficulty Proxy**
   - Cloud-Edge 메타토큰 차이를 난이도 지표로 사용하는 새로운 방법 제안

2. **Layer-Specific Curriculum Learning**
   - 레이어별 독립 커리큘럼 + Hierarchical 학습 전략

3. **Dynamic Curriculum Pacing**
   - 손실 기반 적응형 난이도 조절

---

## 📚 관련 연구

1. **Curriculum Learning** (Bengio et al., 2009)
   - 기본 개념: 쉬운 것부터 어려운 것까지

2. **Self-Paced Learning** (Kumar et al., 2010)
   - 모델 자신이 난이도 결정

3. **Progressive Growing** (Karras et al., 2017)
   - 레이어별 점진적 학습

4. **Knowledge Distillation** (Hinton et al., 2015)
   - Teacher-Student 격차 활용

---

## 🔄 다음 단계

### 즉시 실행 가능
1. PersonaChat 데이터로 Phase 1 실험
2. 메타토큰 차이 분포 분석
3. Easy-to-Hard baseline 구축

### 단기 (1-2주)
1. Layer-Wise Progressive 구현 완료
2. 4가지 전략 비교 실험
3. 최적 하이퍼파라미터 탐색

### 중기 (1개월)
1. 대규모 실험 완료
2. Ablation study 완료
3. 논문 초안 작성

### 장기 (2-3개월)
1. 다른 도메인/태스크 적용
2. 논문 제출 (ICML/NeurIPS 2026)
3. 코드 공개

---

## ✅ 체크리스트

### 구현 완료
- [x] 메타토큰 추출 모듈
- [x] 난이도 계산 모듈
- [x] 커리큘럼 스케줄러 (4가지 전략)
- [x] 데이터셋 레이블링 파이프라인
- [x] 분석 및 시각화 도구
- [x] 학습 통합 인터페이스
- [x] 설정 파일 및 문서

### 실험 대기
- [ ] Phase 1: Easy-to-Hard
- [ ] Phase 2: Layer-Wise Progressive
- [ ] Phase 3: Dynamic Pacing
- [ ] Phase 4: Hybrid & Ablation

### 분석 대기
- [ ] 메타토큰 차이 검증
- [ ] 레이어별 수렴 분석
- [ ] 전략 비교 분석
- [ ] 최종 벤치마크

---

## 📞 문의 및 협업

이 프로젝트는 LoRA-Gen의 학습 효율을 크게 향상시킬 수 있는 잠재력을 가지고 있습니다.

**핵심 가치**:
- 새로운 난이도 측정 방법 (메타토큰 차이)
- LoRA-Gen 구조에 최적화된 커리큘럼
- 실용적 학습 효율 향상 (20-30%)

**적용 가능 범위**:
- LoRA 생성 모델 전반
- Cloud-Edge 협업 시스템
- 대화 시스템 학습

---

**최종 업데이트**: 2026-01-31
**Status**: 구현 완료, 실험 대기
