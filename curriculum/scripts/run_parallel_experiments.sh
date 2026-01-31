#!/bin/bash

# 병렬로 두 실험 실행
# 사용법: bash run_parallel_experiments.sh

set -e

DATA_PATH="curriculum/data/reasoning_tasks/reasoning_tasks_with_difficulties.json"
OUTPUT_DIR="curriculum/experiments"
NUM_EPOCHS=2
BATCH_SIZE=8
MAX_SAMPLES=10000

echo "=================================================="
echo "Running Parallel Experiments"
echo "=================================================="
echo "Samples: $MAX_SAMPLES"
echo "Epochs: $NUM_EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo ""

# 작은 샘플로 빠른 데이터 준비
python3 -c "
import json
with open('$DATA_PATH', 'r') as f:
    data = json.load(f)
    
# 원본 데이터 매핑
with open('curriculum/data/reasoning_tasks/reasoning_tasks_combined.json', 'r') as f:
    original = json.load(f)
    
original_map = {item['id']: item for item in original}

# 결합
combined = []
for diff_item in data[:$MAX_SAMPLES]:
    sample_id = diff_item['sample_id']
    if sample_id in original_map:
        orig_item = original_map[sample_id]
        combined.append({
            'sample_id': sample_id,
            'full_text': orig_item['full_text'],
            'difficulty_percentile': diff_item['difficulty_percentile'],
        })

with open('$OUTPUT_DIR/temp_data.json', 'w') as f:
    json.dump(combined, f)
    
print(f'Prepared {len(combined)} samples')
"

# GPU 0에서 Baseline 실행 (백그라운드)
CUDA_VISIBLE_DEVICES=0 python curriculum/scripts/04_run_single_experiment.py \
    --data_path $OUTPUT_DIR/temp_data.json \
    --experiment_name baseline_random \
    --model_name Qwen/Qwen2.5-1.5B \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --output_dir $OUTPUT_DIR \
    > $OUTPUT_DIR/baseline.log 2>&1 &

BASELINE_PID=$!
echo "Started Baseline (PID: $BASELINE_PID)"

# GPU 1에서 Curriculum 실행 (백그라운드) - GPU가 2개 이상인 경우
if [ $(nvidia-smi -L | wc -l) -gt 1 ]; then
    CUDA_VISIBLE_DEVICES=1 python curriculum/scripts/04_run_single_experiment.py \
        --data_path $OUTPUT_DIR/temp_data.json \
        --experiment_name curriculum_easy_to_hard \
        --curriculum_order \
        --model_name Qwen/Qwen2.5-1.5B \
        --num_epochs $NUM_EPOCHS \
        --batch_size $BATCH_SIZE \
        --output_dir $OUTPUT_DIR \
        > $OUTPUT_DIR/curriculum.log 2>&1 &
    
    CURRICULUM_PID=$!
    echo "Started Curriculum (PID: $CURRICULUM_PID)"
    
    # 두 프로세스 대기
    echo ""
    echo "Waiting for both experiments to complete..."
    wait $BASELINE_PID
    echo "✓ Baseline completed"
    wait $CURRICULUM_PID
    echo "✓ Curriculum completed"
else
    # GPU 1개인 경우 순차 실행
    echo "Only 1 GPU detected, running sequentially..."
    wait $BASELINE_PID
    echo "✓ Baseline completed"
    
    CUDA_VISIBLE_DEVICES=0 python curriculum/scripts/04_run_single_experiment.py \
        --data_path $OUTPUT_DIR/temp_data.json \
        --experiment_name curriculum_easy_to_hard \
        --curriculum_order \
        --model_name Qwen/Qwen2.5-1.5B \
        --num_epochs $NUM_EPOCHS \
        --batch_size $BATCH_SIZE \
        --output_dir $OUTPUT_DIR
    
    echo "✓ Curriculum completed"
fi

# 결과 비교
echo ""
echo "=================================================="
echo "Comparison Results"
echo "=================================================="

python3 -c "
import json

with open('$OUTPUT_DIR/baseline_random_results.json', 'r') as f:
    baseline = json.load(f)

with open('$OUTPUT_DIR/curriculum_easy_to_hard_results.json', 'r') as f:
    curriculum = json.load(f)

print(f'Baseline Final Val Loss: {baseline[\"val_losses\"][-1]:.4f}')
print(f'Curriculum Final Val Loss: {curriculum[\"val_losses\"][-1]:.4f}')

improvement = (baseline['val_losses'][-1] - curriculum['val_losses'][-1]) / baseline['val_losses'][-1] * 100
print(f'')
print(f'Improvement: {improvement:.2f}%')

# 비교 저장
comparison = {
    'baseline': baseline,
    'curriculum': curriculum,
    'improvement_percent': improvement,
}

with open('$OUTPUT_DIR/comparison_results.json', 'w') as f:
    json.dump(comparison, f, indent=2)

print(f'')
print(f'✓ Saved to: $OUTPUT_DIR/comparison_results.json')
"

# 정리
rm $OUTPUT_DIR/temp_data.json

echo ""
echo "=================================================="
echo "All experiments completed!"
echo "=================================================="
