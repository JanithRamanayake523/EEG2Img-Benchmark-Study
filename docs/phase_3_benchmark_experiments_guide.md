# Phase 3: Benchmark Experiments - Execution Guide

**Status**: Ready to execute
**Prerequisites**: Phase 2 preprocessing complete ✓
**Data**: EEG-only dataset ready (22 channels, 2,216 trials)

---

## Overview

Phase 3 consists of running comprehensive benchmark experiments comparing:
- **3 baseline models** (raw time-series)
- **45 image-based combinations** (9 transformations × 5 models)

**Total experiments**: 48 combinations
**Expected runtime**: 12-24 hours on GPU

---

## Execution Roadmap

```
Phase 3 Execution
│
├─ Stage 1: VALIDATION & TESTING (10-15 min)
│  ├── 01_validate_transforms.py
│  ├── 02_validate_models.py
│  └── 03_test_pipeline.py
│
├─ Stage 2: BASELINE TRAINING (1-2 hours)
│  └── 04_train_baselines.py
│      ├── cnn1d
│      ├── bilstm
│      └── transformer
│
├─ Stage 3: IMAGE-BASED EXPERIMENTS (12-24 hours)
│  └── 05_run_experiments.py (×45 combinations)
│      or run_all_experiments.py (all at once)
│
└─ Stage 4: ANALYSIS & VISUALIZATION (30 min)
   ├── 06_aggregate_results.py
   └── 08_generate_results_figures.py
```

---

## Stage 1: Validation & Testing (10-15 min)

**Purpose**: Verify all components work before running long experiments

### 1.1 Validate All Transformations

```bash
python experiments/phase_3_benchmark_experiments/01_validate_transforms.py
```

**What it checks**:
- All 9 transformation implementations
- Output shapes and data types
- Performance on sample data

**Expected output**:
```
[OK] gaf_summation: (2216, 1, 128, 128)
[OK] gaf_difference: (2216, 1, 128, 128)
[OK] mtf: (2216, 1, 128, 128)
... (9 total)
Result: 9/9 PASSED
```

### 1.2 Validate All Model Architectures

```bash
python experiments/phase_3_benchmark_experiments/02_validate_models.py
```

**What it checks**:
- All 8 model architectures
- Forward pass validity
- Parameter counts
- GPU/CPU compatibility

**Expected output**:
```
[OK] cnn1d: 485,124 parameters
[OK] bilstm: 652,356 parameters
[OK] transformer: 1,245,188 parameters
[OK] resnet18: 11,185,282 parameters
[OK] resnet50: 23,508,618 parameters
[OK] lightweight_cnn: 342,916 parameters
[OK] vit_base: 86,567,424 parameters
[OK] vit_small: 22,050,598 parameters
Result: 8/8 PASSED
```

### 1.3 Test Complete Pipeline

```bash
python experiments/phase_3_benchmark_experiments/03_test_pipeline.py
```

**What it checks**:
- Data loading
- Transformation pipeline
- Model training (1 epoch)
- Results saving

**Expected output**:
```
[OK] Data loaded: (2216, 22, 751)
[OK] Transformation successful: (2216, 1, 128, 128)
[OK] Model training: 1 epoch completed
[OK] Results saved to disk
Result: PIPELINE FUNCTIONAL
```

---

## Stage 2: Baseline Training (1-2 hours)

**Purpose**: Train baseline models on raw time-series EEG

### 2.1 Train 1D CNN

```bash
python experiments/phase_3_benchmark_experiments/04_train_baselines.py --model cnn1d --device cuda:0
```

**What it does**:
- 5-fold cross-validation
- Trains on raw (22, 751) EEG
- Early stopping based on Cohen's Kappa (after 50 warmup epochs)
- Saves results to `results/phase3/baselines/cnn1d_all_fold_results.json`

**Expected output**:
```
Fold 1/5 - Best Val Kappa: 0.6132, Acc: 70.85% (Epoch 65)
Fold 2/5 - Best Val Kappa: 0.6078, Acc: 70.45% (Epoch 62)
Fold 3/5 - Best Val Kappa: 0.6245, Acc: 71.32% (Epoch 68)
Fold 4/5 - Best Val Kappa: 0.5987, Acc: 69.85% (Epoch 58)
Fold 5/5 - Best Val Kappa: 0.6089, Acc: 70.56% (Epoch 64)
───────────────────────────────────────────────────
Final Result - Accuracy: 70.61 ± 0.65%
Final Result - Kappa: 0.6106 ± 0.0095
```

**Time**: ~10-15 minutes per fold, ~50-75 minutes total

### 2.2 Train BiLSTM

```bash
python experiments/phase_3_benchmark_experiments/04_train_baselines.py --model bilstm --device cuda:0
```

**Same as 2.1 but with BiLSTM architecture**

**Expected**: Similar accuracy range (68-72%)
**Time**: ~15-20 minutes per fold, ~75-100 minutes total

### 2.3 Train Transformer

```bash
python experiments/phase_3_benchmark_experiments/04_train_baselines.py --model transformer --device cuda:0
```

**Same as 2.1 but with Transformer architecture**

**Expected**: Similar or better accuracy (70-74%)
**Time**: ~20-25 minutes per fold, ~100-125 minutes total

---

## Stage 3: Image-Based Experiments (12-24 hours)

**Purpose**: Train image-based models with various transformations

### Option A: Run Complete Benchmark (Recommended)

```bash
python experiments/phase_3_benchmark_experiments/run_all_experiments.py --device cuda:0
```

**What it does**:
- Runs all 45 combinations (9 transforms × 5 models)
- 5-fold cross-validation per combination
- Saves results for each combination
- Automatically handles failures and logging

**Output structure**:
```
results/phase3/experiments/
├── gaf_summation_resnet18_all_fold_results.json
├── gaf_summation_resnet50_all_fold_results.json
├── gaf_summation_lightweight_cnn_all_fold_results.json
├── gaf_summation_vit_base_all_fold_results.json
├── gaf_summation_vit_small_all_fold_results.json
├── gaf_difference_resnet18_all_fold_results.json
├── mtf_resnet18_all_fold_results.json
├── mtf_resnet50_all_fold_results.json
├── mtf_lightweight_cnn_all_fold_results.json
├── mtf_vit_base_all_fold_results.json
├── mtf_vit_small_all_fold_results.json
├── ... (45 total files)
└── logs/
    └── experiments_YYYYMMDD_HHMMSS.log
```

**Expected time**:
- GPU (RTX 3070): 12-18 hours
- GPU (RTX 4090): 6-10 hours
- CPU: 3-5 days (not recommended)

**Example output** (one experiment):
```
Experiment: gaf_summation + resnet18
Fold 1/5 - Transform: 45.2s, Train: 18m 34s
  Best Val Kappa: 0.7128, Acc: 76.45% (Epoch 64)
Fold 2/5 - Transform: 45.1s, Train: 18m 41s
  Best Val Kappa: 0.7089, Acc: 76.12% (Epoch 62)
Fold 3/5 - Transform: 45.3s, Train: 18m 52s
  Best Val Kappa: 0.7234, Acc: 77.03% (Epoch 68)
Fold 4/5 - Transform: 45.0s, Train: 18m 38s
  Best Val Kappa: 0.7045, Acc: 75.78% (Epoch 60)
Fold 5/5 - Transform: 45.2s, Train: 18m 45s
  Best Val Kappa: 0.7156, Acc: 76.78% (Epoch 66)
─────────────────────────────────────────────────
Final Result - Accuracy: 76.43 ± 0.89%
Final Result - Kappa: 0.7130 ± 0.0079
```

### Option B: Run Specific Combination

```bash
# Run one transformation + model combination
python experiments/phase_3_benchmark_experiments/05_run_experiments.py \
    --transform gaf_summation \
    --model resnet18 \
    --device cuda:0
```

**When to use**:
- Testing a specific combination
- Debugging issues
- Running on limited time

**Output**:
```
results/phase3/experiments/gaf_summation_resnet18_all_fold_results.json
```

### 45 Experiment Combinations

**Transformations (9)**:
1. gaf_summation (GASF)
2. gaf_difference (GADF)
3. mtf (Markov Transition Field)
4. recurrence (Recurrence Plot)
5. recurrence_quantified (RQA)
6. spectrogram (STFT)
7. cwt (Continuous Wavelet Transform)
8. cwt_morlet (CWT with Morlet)
9. stft (Short-Time Fourier Transform)

**Models (5 for each transformation)**:
1. resnet18 (pretrained)
2. resnet50 (pretrained)
3. lightweight_cnn (from scratch)
4. vit_base (pretrained)
5. vit_small (pretrained)

**Total**: 9 × 5 = 45 combinations

---

## Stage 4: Analysis & Visualization (30 min)

### 4.1 Aggregate All Results

```bash
python experiments/phase_3_benchmark_experiments/06_aggregate_results.py
```

**What it does**:
- Collects results from all experiments
- Computes statistics (mean, std)
- Compares transformations
- Compares models
- Identifies best performers

**Output**:
```
results/phase3/
├── aggregated_results.json
├── comparison_by_transformation.csv
├── comparison_by_model.csv
└── summary_statistics.txt
```

### 4.2 Generate Publication Figures

```bash
python experiments/phase_3_benchmark_experiments/08_generate_results_figures.py
```

**What it generates**:
- Accuracy comparison plots
- Kappa comparison plots
- Confusion matrices
- Training curves
- Statistical significance tests

**Output**:
```
results/phase3/figures/
├── comparison_accuracy.png
├── comparison_kappa.png
├── confusion_matrices/
│   ├── gaf_summation_resnet18_fold1.png
│   ├── gaf_summation_resnet18_fold2.png
│   └── ...
├── training_curves/
│   ├── cnn1d_fold1.png
│   ├── cnn1d_fold2.png
│   └── ...
└── statistical_analysis/
    ├── pairwise_comparisons.png
    └── significance_matrix.png
```

---

## Quick Reference Commands

### Minimal (Validation only)
```bash
# 10-15 minutes
python experiments/phase_3_benchmark_experiments/01_validate_transforms.py
python experiments/phase_3_benchmark_experiments/02_validate_models.py
python experiments/phase_3_benchmark_experiments/03_test_pipeline.py
```

### Baseline Only
```bash
# 1-2 hours
python experiments/phase_3_benchmark_experiments/04_train_baselines.py --model cnn1d --device cuda:0
python experiments/phase_3_benchmark_experiments/04_train_baselines.py --model bilstm --device cuda:0
python experiments/phase_3_benchmark_experiments/04_train_baselines.py --model transformer --device cuda:0
```

### Single Experiment
```bash
# 1-2 hours (one combination)
python experiments/phase_3_benchmark_experiments/05_run_experiments.py \
    --transform gaf_summation \
    --model resnet18 \
    --device cuda:0
```

### Complete Benchmark
```bash
# 12-24 hours (all 48 experiments)
python experiments/phase_3_benchmark_experiments/run_all_experiments.py --device cuda:0
```

### Post-Processing
```bash
# 30 minutes
python experiments/phase_3_benchmark_experiments/06_aggregate_results.py
python experiments/phase_3_benchmark_experiments/08_generate_results_figures.py
```

---

## Expected Results Range

Based on BCI IV-2a motor imagery classification:

| Model Type | Accuracy | Kappa | Notes |
|------------|----------|-------|-------|
| **Baseline** | 68-72% | 0.57-0.63 | Raw time-series |
| **GAF** | 73-77% | 0.64-0.69 | Best transformations |
| **MTF** | 72-76% | 0.63-0.68 | Good temporal info |
| **Recurrence** | 70-75% | 0.61-0.66 | Moderate performance |
| **Spectrogram** | 71-74% | 0.62-0.66 | Time-frequency info |
| **CWT** | 72-75% | 0.63-0.67 | Wavelet decomposition |
| **STFT** | 71-74% | 0.62-0.66 | Short-time analysis |

---

## Hardware & Time Estimates

### GPU (Recommended)

**RTX 3070 (8GB VRAM)**:
- Validation: 10-15 min
- Baselines: 1-2 hours
- Complete benchmark: 12-18 hours
- **Total**: ~15-20 hours

**RTX 4090 (24GB VRAM)**:
- Validation: 10-15 min
- Baselines: 1-2 hours
- Complete benchmark: 6-10 hours
- **Total**: ~9-13 hours

### CPU (Not Recommended)
- Validation: 5-10 min
- Baselines: 4-6 hours
- Complete benchmark: 2-5 days
- **Total**: Too long

---

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python experiments/phase_3_benchmark_experiments/05_run_experiments.py \
    --transform gaf_summation \
    --model resnet18 \
    --batch_size 16 \
    --device cuda:0
```

### Slow Transformation
- Use GPU for transformation (automatic)
- Reduce image size if needed
- Check disk I/O

### Model Not Found
```bash
# Re-run validation
python experiments/phase_3_benchmark_experiments/02_validate_models.py
```

---

## Results Interpretation

### Kappa Ranges
- 0.81-1.00: Almost perfect
- 0.61-0.80: Substantial ← Our target range
- 0.41-0.60: Moderate
- 0.21-0.40: Fair
- <0.20: Poor

### What to Look For
1. **Best transformation**: Which produces highest accuracy/Kappa?
2. **Best model**: Which architecture performs best?
3. **Stability**: Low standard deviation = consistent performance
4. **Trade-offs**: Speed vs. accuracy

---

## Data Management During Experiments

**Results saved**:
```
results/phase3/
├── baselines/          ← 3 files (~1 MB)
├── experiments/        ← 45 files (~15 MB)
└── figures/           ← Generated figures (~50 MB)
```

**Generated images** (in data/images/):
```
data/images/
├── gaf/               ← ~11 GB (2216 images × 45 models)
├── mtf/               ← ~11 GB
└── ...
```

**Storage needed**: ~100 GB total during experiments (temporary)

---

## Next Steps (In Order)

1. **[TODAY]** Run validation (10 min)
   ```bash
   python experiments/phase_3_benchmark_experiments/01_validate_transforms.py
   python experiments/phase_3_benchmark_experiments/02_validate_models.py
   python experiments/phase_3_benchmark_experiments/03_test_pipeline.py
   ```

2. **[TODAY]** Run baselines (2 hours)
   ```bash
   python experiments/phase_3_benchmark_experiments/04_train_baselines.py --model cnn1d --device cuda:0
   python experiments/phase_3_benchmark_experiments/04_train_baselines.py --model bilstm --device cuda:0
   python experiments/phase_3_benchmark_experiments/04_train_baselines.py --model transformer --device cuda:0
   ```

3. **[LONG RUN]** Run complete benchmark (12-24 hours)
   ```bash
   python experiments/phase_3_benchmark_experiments/run_all_experiments.py --device cuda:0
   ```

4. **[AFTER]** Analyze results (30 min)
   ```bash
   python experiments/phase_3_benchmark_experiments/06_aggregate_results.py
   python experiments/phase_3_benchmark_experiments/08_generate_results_figures.py
   ```

---

**Ready to start?** Begin with validation:
```bash
python experiments/phase_3_benchmark_experiments/01_validate_transforms.py
```

---

**Last Updated**: 2026-04-12
**Status**: Ready to execute Phase 3
