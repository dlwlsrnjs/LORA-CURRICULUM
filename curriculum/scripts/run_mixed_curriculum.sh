#!/bin/bash

# Mixed Curriculum vs Baseline 비교
set -e

DATA_PATH="curriculum/data/reasoning_tasks/reasoning_tasks_with_difficulties.json"
OUTPUT_DIR="curriculum/experiments"
NUM_EPOCHS=2
BATCH_SIZE=8
MAX_SAMPLES=10000

echo "=================================================="
echo "Mixed Curriculum Learning Experiment"
echo "=================================================="
echo "Samples: $MAX_SAMPLES"
echo "Epochs: $NUM_EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo ""

# 데이터 준비
echo "Preparing data..."
python3 << EOF
import json
import random

with open('$DATA_PATH') as f:
    data = json.load(f)

# 샘플링
random.seed(42)
data = random.sample(data, min($MAX_SAMPLES, len(data)))

with open('$OUTPUT_DIR/temp_data.json', 'w') as f:
    json.dump(data, f)

print(f"Prepared {len(data)} samples")
EOF

# GPU 개수 확인
NUM_GPUS=$(nvidia-smi -L | wc -l)

if [ $NUM_GPUS -ge 2 ]; then
    echo "Starting parallel experiments on 2 GPUs..."
    
    # Baseline on GPU 0
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
    
    # Mixed Curriculum on GPU 1
    CUDA_VISIBLE_DEVICES=1 python curriculum/scripts/05_run_mixed_curriculum.py \
        --data_path $OUTPUT_DIR/temp_data.json \
        --experiment_name mixed_curriculum \
        --model_name Qwen/Qwen2.5-1.5B \
        --num_epochs $NUM_EPOCHS \
        --batch_size $BATCH_SIZE \
        --output_dir $OUTPUT_DIR \
        > $OUTPUT_DIR/mixed.log 2>&1 &
    
    MIXED_PID=$!
    echo "Started Mixed Curriculum (PID: $MIXED_PID)"
    
    echo ""
    echo "Waiting for both experiments to complete..."
    wait $BASELINE_PID
    wait $MIXED_PID
    
else
    echo "Only 1 GPU available. Running sequentially..."
    
    # Baseline
    python curriculum/scripts/04_run_single_experiment.py \
        --data_path $OUTPUT_DIR/temp_data.json \
        --experiment_name baseline_random \
        --model_name Qwen/Qwen2.5-1.5B \
        --num_epochs $NUM_EPOCHS \
        --batch_size $BATCH_SIZE \
        --output_dir $OUTPUT_DIR \
        > $OUTPUT_DIR/baseline.log 2>&1
    
    # Mixed Curriculum
    python curriculum/scripts/05_run_mixed_curriculum.py \
        --data_path $OUTPUT_DIR/temp_data.json \
        --experiment_name mixed_curriculum \
        --model_name Qwen/Qwen2.5-1.5B \
        --num_epochs $NUM_EPOCHS \
        --batch_size $BATCH_SIZE \
        --output_dir $OUTPUT_DIR \
        > $OUTPUT_DIR/mixed.log 2>&1
fi

echo ""
echo "=================================================="
echo "Comparing Results"
echo "=================================================="

python3 << 'EOF'
import json

with open('curriculum/experiments/baseline_random_results.json') as f:
    baseline = json.load(f)
    
with open('curriculum/experiments/mixed_curriculum_results.json') as f:
    mixed = json.load(f)

print(f"\n{'='*60}")
print("Results Comparison")
print(f"{'='*60}\n")

print("Baseline (Random):")
print(f"  Final Val Loss: {baseline['val_losses'][-1]:.4f}")

print("\nMixed Curriculum:")
print(f"  Schedule: {mixed['curriculum_schedule']}")
print(f"  Final Val Loss: {mixed['val_losses'][-1]:.4f}")

diff = baseline['val_losses'][-1] - mixed['val_losses'][-1]
improvement = (diff / baseline['val_losses'][-1]) * 100

print(f"\n{'='*60}")
if diff > 0:
    print(f"✅ Mixed Curriculum improved by {improvement:.2f}%")
else:
    print(f"❌ Mixed Curriculum worse by {abs(improvement):.2f}%")
print(f"{'='*60}")

# 비교 저장
comparison = {
    'baseline': baseline,
    'mixed_curriculum': mixed,
    'improvement_pct': improvement
}

with open('curriculum/experiments/mixed_comparison.json', 'w') as f:
    json.dump(comparison, f, indent=2)

print("\n✓ Comparison saved to: curriculum/experiments/mixed_comparison.json")

EOF

echo ""
echo "Done!"
