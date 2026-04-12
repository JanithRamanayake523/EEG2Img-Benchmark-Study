#!/bin/bash

# Phase 3: Run all experiments on pooled subjects
# This script trains all transform+model combinations on all subjects pooled together

DEVICE="cuda:0"  # Change to "cpu" if needed
TRANSFORMS=(
    "gaf_summation"
    "gaf_difference"
    "mtf_q8"
    "mtf_q16"
    "recurrence_plot"
    "spectrogram"
    "scalogram_morlet"
    "scalogram_mexican"
    "topographic"
)
MODELS=(
    "resnet18"
    "resnet50"
    "lightweight_cnn"
    "vit_base"
    "vit_small"
)

TOTAL=$((${#TRANSFORMS[@]} * ${#MODELS[@]}))
COMPLETED=0

echo "==========================================="
echo "Phase 3: Complete Benchmark Experiments"
echo "Running on all subjects pooled together"
echo "==========================================="
echo "Transforms: ${#TRANSFORMS[@]}"
echo "Models: ${#MODELS[@]}"
echo "Total combinations: $TOTAL"
echo "Device: $DEVICE"
echo "==========================================="
echo ""

# Run all combinations on pooled subjects
for transform in "${TRANSFORMS[@]}"; do
    for model in "${MODELS[@]}"; do
        COMPLETED=$((COMPLETED + 1))
        PROGRESS=$(echo "scale=1; $COMPLETED * 100 / $TOTAL" | bc)

        printf "%-40s " "[$COMPLETED/$TOTAL - $PROGRESS%] $transform + $model..."

        python experiments/phase_3_benchmark_experiments/05_run_experiments.py \
            --transform "$transform" \
            --model "$model" \
            --subject "all" \
            --device "$DEVICE" > /dev/null 2>&1

        if [ $? -eq 0 ]; then
            echo "[OK]"
        else
            echo "[FAIL]"
        fi
    done
done

echo ""
echo "==========================================="
echo "All experiments completed!"
echo "Results saved to: results/phase3/experiments/"
echo "==========================================="
