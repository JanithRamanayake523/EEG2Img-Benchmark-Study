# Phase 3: Image Transformation & Deep Learning Experiments

Complete implementation of Phase 3 benchmark experiments for EEG-to-Image transformation and deep learning classification.

## 📁 Directory Structure

```
phase3/
├── 01_validate_transforms.py       # Validate all 6 transformation methods
├── 02_validate_models.py           # Validate all 7 model architectures
├── 03_test_pipeline.py             # End-to-end pipeline smoke test
├── 04_train_baselines.py           # Train raw-signal baseline models
├── 05_run_experiments.py           # Main experiment runner
├── 06_aggregate_results.py         # Aggregate and analyze results
├── 07_robustness_tests.py          # Robustness testing (noise, dropout, jitter)
├── 08_generate_results_figures.py  # Generate publication figures
└── README.md                       # This file

configs/phase3/
├── base_config.yaml                # Base configuration
├── transform_configs.yaml          # Transform parameters
├── model_configs.yaml              # Model architectures
└── example_experiment.yaml         # Example experiment config

notebooks/phase_3_image_transformations/
└── PHASE_3_COMPREHENSIVE_ANALYSIS.ipynb  # Complete analysis notebook
```

## 🚀 Quick Start

### Step 1: Validate Transforms (5 minutes)

Test all 6 transformation methods:

```bash
python experiments/phase_3_benchmark_experiments/01_validate_transforms.py
```

**Output:** `results/phase3/validation/transforms/`
- Validation report
- Sample transformation images
- Performance metrics

### Step 2: Validate Models (5 minutes)

Test all model architectures:

```bash
python experiments/phase_3_benchmark_experiments/02_validate_models.py
```

**Output:** `results/phase3/validation/models/`
- Model validation report
- Parameter counts
- Inference speed benchmarks

### Step 3: Test Pipeline (5 minutes)

Run end-to-end smoke test:

```bash
python experiments/phase_3_benchmark_experiments/03_test_pipeline.py
```

**Output:** `results/phase3/validation/pipeline_test/`
- Pipeline test results
- GPU memory usage
- Training verification

---

## 📊 Running Full Experiments

### Option A: Train Baselines

Train raw-signal baseline models (1D CNN, BiLSTM, Transformer):

```bash
# Train 1D CNN baseline
python experiments/phase_3_benchmark_experiments/04_train_baselines.py --model cnn1d --device cuda:0

# Train BiLSTM baseline
python experiments/phase_3_benchmark_experiments/04_train_baselines.py --model bilstm --device cuda:0

# Train Transformer baseline
python experiments/phase_3_benchmark_experiments/04_train_baselines.py --model transformer --device cuda:0
```

**Output:** `results/phase3/baselines/{model}/{subject}/`
- Training logs
- Results JSON
- Best model checkpoints

### Option B: Train Image Models

Run image transformation experiments:

```bash
# Example: GAF + ResNet-18
python experiments/phase_3_benchmark_experiments/05_run_experiments.py \
    --transform gaf_summation \
    --model resnet18 \
    --subject A01T \
    --device cuda:0 \
    --output_dir results/phase3/experiments

# Example: MTF + ViT-Base
python experiments/phase_3_benchmark_experiments/05_run_experiments.py \
    --transform mtf_q8 \
    --model vit_base \
    --subject A01T \
    --device cuda:0

# Using config file
python experiments/phase_3_benchmark_experiments/05_run_experiments.py \
    --config experiments/configs/phase3/example_experiment.yaml \
    --device cuda:0
```

**Output:** `results/phase3/experiments/{transform}_{model}_{subject}/`
- Experiment logs
- K-fold results
- Training history
- Best model checkpoints

### Option C: Batch Experiments

Run multiple experiments in sequence:

```bash
# Create a batch script
# experiments/phase_3_benchmark_experiments/run_batch.sh

#!/bin/bash

TRANSFORMS=("gaf_summation" "mtf_q8" "recurrence_plot" "spectrogram" "scalogram_morlet" "topographic")
MODELS=("resnet18" "resnet34" "vit_base" "lightweight_cnn")
SUBJECT="A01T"

for transform in "${TRANSFORMS[@]}"; do
    for model in "${MODELS[@]}"; do
        echo "Running: $transform + $model"
        python experiments/phase_3_benchmark_experiments/05_run_experiments.py \
            --transform $transform \
            --model $model \
            --subject $SUBJECT \
            --device cuda:0
    done
done
```

---

## 📈 Results Analysis

### Step 1: Aggregate Results

Collect all experiment results:

```bash
python experiments/phase_3_benchmark_experiments/06_aggregate_results.py \
    --results_dir results/phase3 \
    --output results/phase3/aggregated_results.csv \
    --excel
```

**Output:**
- `aggregated_results.csv` - All experiment results
- `aggregated_results_summary.csv` - Summary statistics
- `aggregated_results_pvalues.csv` - Statistical significance
- `summary_report.txt` - Text summary
- `aggregated_results.xlsx` - Excel file (if --excel used)

### Step 2: Generate Figures

Create publication-quality figures:

```bash
python experiments/phase_3_benchmark_experiments/08_generate_results_figures.py \
    --results_csv results/phase3/aggregated_results.csv \
    --output_dir results/phase3/figures \
    --format png
```

**Output:** `results/phase3/figures/`
- `accuracy_comparison.png`
- `accuracy_boxplots.png`
- `accuracy_heatmap.png`
- `statistical_significance.png`
- `top_10_performers.png`
- `summary_statistics.png`

### Step 3: Comprehensive Analysis

Open Jupyter notebook for complete analysis:

```bash
jupyter notebook notebooks/phase_3_image_transformations/PHASE_3_COMPREHENSIVE_ANALYSIS.ipynb
```

**Includes:**
- Transform method analysis
- Model architecture comparison
- Statistical significance testing
- Effect size analysis
- Top performers
- Publication-ready figures

---

## 🔬 Robustness Testing

Test model robustness under perturbations:

```bash
# Test noise robustness
python experiments/phase_3_benchmark_experiments/07_robustness_tests.py \
    --model_path results/phase3/gaf_resnet18/best_model.pth \
    --transform gaf_summation \
    --test_type noise \
    --device cuda:0

# Test channel dropout robustness
python experiments/phase_3_benchmark_experiments/07_robustness_tests.py \
    --model_path results/phase3/gaf_resnet18/best_model.pth \
    --transform gaf_summation \
    --test_type dropout \
    --device cuda:0

# Test temporal jitter robustness
python experiments/phase_3_benchmark_experiments/07_robustness_tests.py \
    --model_path results/phase3/gaf_resnet18/best_model.pth \
    --transform gaf_summation \
    --test_type jitter \
    --device cuda:0

# Test all perturbations
python experiments/phase_3_benchmark_experiments/07_robustness_tests.py \
    --model_path results/phase3/gaf_resnet18/best_model.pth \
    --transform gaf_summation \
    --test_type all \
    --device cuda:0
```

**Output:** `results/phase3/robustness/`
- Robustness results JSON
- Robustness curves plots

---

## 🎯 Experiment Configurations

### Transform Methods

- **GAF Summation** (`gaf_summation`) - Gramian Angular Summation Field
- **GAF Difference** (`gaf_difference`) - Gramian Angular Difference Field
- **MTF Q8** (`mtf_q8`) - Markov Transition Field (8 bins)
- **MTF Q16** (`mtf_q16`) - Markov Transition Field (16 bins)
- **Recurrence Plot** (`recurrence_plot`) - Phase-space recurrence
- **Spectrogram** (`spectrogram`) - STFT time-frequency
- **Scalogram Morlet** (`scalogram_morlet`) - CWT with Morlet wavelet
- **Scalogram Mexican** (`scalogram_mexican`) - CWT with Mexican Hat
- **Topographic** (`topographic`) - Spatial electrode mapping

### Model Architectures

**CNN Models:**
- `resnet18` - ResNet-18
- `resnet34` - ResNet-34
- `resnet50` - ResNet-50
- `lightweight_cnn` - Custom lightweight CNN

**Vision Transformers:**
- `vit_base` - Vision Transformer Base
- `vit_small` - Vision Transformer Small

**Baseline Models (Raw Time-Series):**
- `cnn1d` - 1D CNN
- `bilstm` - Bidirectional LSTM
- `transformer` - Transformer encoder

---

## 💾 GPU Requirements

### Memory Requirements (Approximate)

| Model | Image Size | Batch Size | GPU Memory |
|-------|-----------|-----------|------------|
| ResNet-18 | 128×128 | 32 | ~4 GB |
| ResNet-34 | 128×128 | 32 | ~6 GB |
| ResNet-50 | 128×128 | 32 | ~8 GB |
| ViT-Base | 224×224 | 16 | ~10 GB |
| Lightweight CNN | 64×64 | 64 | ~2 GB |

### Multi-GPU Training

To use multiple GPUs, set the device flag:

```bash
# Use GPU 0
--device cuda:0

# Use GPU 1
--device cuda:1

# For data parallel (requires code modification)
--device cuda  # Will use all available GPUs
```

---

## 📋 Expected Outputs

### Per Experiment

Each experiment produces:

```
results/phase3/experiments/{transform}_{model}_{subject}/
├── {experiment}_results.json       # Complete results
├── {experiment}.log                # Training log
├── fold_1_checkpoint.pth           # Model checkpoints (optional)
├── fold_2_checkpoint.pth
└── ...
```

### Aggregated Results

```
results/phase3/
├── aggregated_results.csv          # All experiments
├── aggregated_results_summary.csv  # Summary stats
├── aggregated_results_pvalues.csv  # Statistical tests
├── summary_report.txt              # Text summary
├── top_10_performers.csv           # Top performers
├── complete_results_table.csv      # Full results table
├── statistical_summary.json        # Statistical summary
└── figures/                        # All figures
    ├── accuracy_comparison.png
    ├── accuracy_boxplots.png
    ├── accuracy_heatmap.png
    ├── statistical_significance.png
    ├── top_10_performers.png
    ├── summary_statistics.png
    └── comprehensive_overview.png
```

---

## 🔧 Troubleshooting

### CUDA Out of Memory

Reduce batch size:

```bash
# Edit config file or use smaller batch size
--batch_size 16  # Instead of 32
```

### Transform Taking Too Long

Use smaller image size:

```yaml
# In transform_configs.yaml
gaf_summation:
  image_size: 64  # Instead of 128
```

### Model Not Found

Ensure model is properly installed:

```bash
pip install timm  # For Vision Transformers
```

---

## 📊 Validation Metrics

All experiments report:

- **Accuracy** - Classification accuracy
- **Precision** - Per-class precision
- **Recall** - Per-class recall
- **F1-Score** - Harmonic mean of precision/recall
- **Confusion Matrix** - Detailed classification matrix
- **Training Time** - Time per epoch and total
- **GPU Memory** - Peak memory usage

---

## 🎓 Citation

If you use this code for your research, please cite:

```bibtex
@software{eeg2img_benchmark_2026,
  title={EEG-to-Image Transformation Benchmark Study},
  author={[Your Name]},
  year={2026},
  url={https://github.com/[your-repo]}
}
```

---

## ✅ Phase 3 Completion Checklist

- [ ] Validate all transforms (01_validate_transforms.py)
- [ ] Validate all models (02_validate_models.py)
- [ ] Test end-to-end pipeline (03_test_pipeline.py)
- [ ] Train baseline models (04_train_baselines.py)
- [ ] Run image transformation experiments (05_run_experiments.py)
- [ ] Aggregate results (06_aggregate_results.py)
- [ ] Generate figures (08_generate_results_figures.py)
- [ ] Complete analysis notebook
- [ ] Document findings
- [ ] Prepare for publication

---

**Phase 3 Complete!** 🎉

Ready to begin Phase 4: Advanced Model Architectures and Optimization
