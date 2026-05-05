#!/bin/bash
# ============================================================
# PhaBERT-CNN: Full Experiment Pipeline
# 
# This script runs the complete experiment:
# 1. Data preparation
# 2. Training for all groups (A, B, C, D) and folds (0-4)
# 3. Evaluation for all groups
#
# Usage:
#   bash scripts/run_all.sh
#   bash scripts/run_all.sh --skip_prepare  # Skip data preparation
# ============================================================

set -e  # Exit on error

SKIP_PREPARE=false
GPU_ID=0
COMPILE_FLAG=""

# Parse arguments
for arg in "$@"; do
    case $arg in
        --skip_prepare)
            SKIP_PREPARE=true
            shift
            ;;
        --no_compile)
            COMPILE_FLAG="--no_compile"
            shift
            ;;
        --gpu=*)
            GPU_ID="${arg#*=}"
            shift
            ;;
    esac
done

export CUDA_VISIBLE_DEVICES=$GPU_ID
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "========================================================"
echo "PhaBERT-CNN Full Experiment Pipeline"
echo "GPU: $GPU_ID"
echo "========================================================"

# ============================================================
# Step 1: Prepare Data
# ============================================================
if [ "$SKIP_PREPARE" = false ]; then
    echo ""
    echo "========================================================"
    echo "Step 1: Preparing data..."
    echo "========================================================"
    python scripts/prepare_data.py --skip_download
fi

# ============================================================
# Step 2: Train all groups and folds
# ============================================================
echo ""
echo "========================================================"
echo "Step 2: Training models..."
echo "========================================================"

N_FOLDS=5

for group in A; do
    for ((fold=2; fold<N_FOLDS; fold++)); do
        echo ""
        echo "--------------------------------------------------------"
        echo "Training: Group $group, Fold $fold"
        echo "--------------------------------------------------------"
        
        python scripts/train.py \
            --group $group \
            --fold $fold \
            --batch_size 128 \
            --num_workers 0 \
            --max_seq_length 512 \
            --warmup_epochs 1 \
            --finetune_epochs 10 \
            --patience 3 \
            --class_balance undersample \
            $COMPILE_FLAG
        
        echo "Completed: Group $group, Fold $fold"
    done
done

# ============================================================
# Step 3: Evaluate all groups
# ============================================================
echo ""
echo "========================================================"
echo "Step 3: Evaluating models..."
echo "========================================================"

for group in A; do
    echo ""
    echo "--------------------------------------------------------"
    echo "Evaluating: Group $group"
    echo "--------------------------------------------------------"
    
    python scripts/evaluate.py --group $group
done

# ============================================================
# Summary
# ============================================================
echo ""
echo "========================================================"
echo "Experiment Complete!"
echo "========================================================"
echo ""
echo "Results saved in results/metrics/"
echo ""
echo "Paper reference results:"
echo "  Group A (100-400bp):  sn=82.00%, sp=80.15%, acc=81.59%"
echo "  Group B (400-800bp):  sn=89.91%, sp=80.44%, acc=87.91%"
echo "  Group C (800-1200bp): sn=91.12%, sp=85.93%, acc=90.01%"
echo "  Group D (1200-1800bp): sn=88.47%, sp=90.95%, acc=90.69%"
