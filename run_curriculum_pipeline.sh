#!/bin/bash
# 커리큘럼 러닝 실험 파이프라인

set -e  # Exit on error

echo "======================================"
echo "Curriculum Learning Pipeline"
echo "======================================"

# Configuration
CONFIG_FILE="${1:-curriculum/configs/curriculum_config.yaml}"
DATA_PATH="${2:-data/train.jsonl}"
OUTPUT_DIR="${3:-curriculum/outputs}"

echo ""
echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Data path: $DATA_PATH"
echo "  Output dir: $OUTPUT_DIR"
echo ""

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/difficulties"
mkdir -p "$OUTPUT_DIR/analysis"
mkdir -p "$OUTPUT_DIR/checkpoints"
mkdir -p "$OUTPUT_DIR/logs"

# Step 1: Label difficulties
echo ""
echo "======================================"
echo "Step 1: Labeling Difficulty"
echo "======================================"

DIFFICULTY_PATH="$OUTPUT_DIR/difficulties/difficulty_labels.json"

python curriculum/scripts/01_label_difficulties.py \
    --data_path "$DATA_PATH" \
    --output_path "$DIFFICULTY_PATH" \
    --cloud_model "meta-llama/Meta-Llama-3-8B" \
    --edge_model "Qwen/Qwen2.5-1.5B" \
    --metric "l2" \
    --top_k 3 \
    --difficulty_metric "topk_mean" \
    --num_stages 3 \
    --device "cuda"

echo "✓ Difficulty labeling completed"

# Step 2: Analyze difficulties
echo ""
echo "======================================"
echo "Step 2: Analyzing Difficulties"
echo "======================================"

python curriculum/scripts/02_analyze_difficulties.py \
    --difficulty_path "$DIFFICULTY_PATH" \
    --output_dir "$OUTPUT_DIR/analysis"

echo "✓ Difficulty analysis completed"

# Step 3: Training with curriculum
echo ""
echo "======================================"
echo "Step 3: Training with Curriculum"
echo "======================================"

# Option A: Run existing training script with curriculum integration
# This requires modifying train_dialogue_lora.py to accept curriculum parameters

python train_dialogue_lora.py \
    --config "$CONFIG_FILE" \
    --difficulty_path "$DIFFICULTY_PATH" \
    --use_curriculum \
    --phase 3 \
    --output_dir "$OUTPUT_DIR/checkpoints" \
    --log_dir "$OUTPUT_DIR/logs"

# Option B: Use standalone curriculum training script
# python curriculum/train_with_curriculum.py \
#     --difficulty_path "$DIFFICULTY_PATH" \
#     --curriculum_config "$CONFIG_FILE"

echo "✓ Training completed"

# Step 4: Summary
echo ""
echo "======================================"
echo "Pipeline Completed Successfully!"
echo "======================================"
echo ""
echo "Outputs:"
echo "  Difficulty labels: $DIFFICULTY_PATH"
echo "  Analysis plots: $OUTPUT_DIR/analysis/"
echo "  Checkpoints: $OUTPUT_DIR/checkpoints/"
echo "  Logs: $OUTPUT_DIR/logs/"
echo ""
