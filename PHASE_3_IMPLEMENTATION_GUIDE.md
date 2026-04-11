# Phase 3: Time-Series-to-Image Transformation & Deep Learning
## Implementation Guide & Status

**Current Date:** April 11, 2026
**Phase Status:** Ready to Begin
**Foundation:** Phase 2 Preprocessing Complete (99% data retention, 83.4% noise reduction)

---

## 🎯 Phase 3 Objectives

1. **Image Transformation Module:** Implement and validate 6 transformation methods
2. **Model Training Infrastructure:** Set up training pipelines for CNN and ViT models
3. **Baseline Comparisons:** Establish performance benchmarks with raw-signal baselines
4. **Comprehensive Benchmarking:** Test across multiple datasets and paradigms
5. **Statistical Analysis:** Identify best-performing transformation-model combinations

---

## 📦 Available Infrastructure (Verified)

### ✅ Transform Module (`src/transforms/`)
All 6 transformation methods are already implemented:
- **GAFTransformer** - Gramian Angular Fields (GASF/GADF)
- **MTFTransformer** - Markov Transition Fields
- **RecurrencePlotTransformer** - Phase-space recurrence analysis
- **SpectrogramTransformer** - STFT time-frequency representation
- **ScalogramTransformer** - CWT wavelet analysis (Morlet, Mexican Hat)
- **TopographicTransformer** - Spatial electrode mapping (SSFI)

Registry includes 18 pre-configured variants:
```python
from src.transforms import get_transformer

# Example usage:
transformer = get_transformer('gaf_summation', image_size=128)
images = transformer.fit_transform(eeg_epochs)
```

### ✅ Model Module (`src/models/`)
Three model architectures ready:
- **CNNs** - ResNet variants (18/34/50) + lightweight custom CNN
- **ViT** - Vision Transformers (Base/Small variants with timm)
- **Baselines** - 1D CNN, BiLSTM, Transformer (raw time-series)

Registry for easy creation:
```python
from src.models import create_model

model = create_model('resnet18', num_classes=4, pretrained=True)
model = create_model('vit_base', num_classes=4)
```

### ✅ Training Infrastructure (`src/training/`)
Complete training pipeline:
- **Trainer** - Main training loop with validation, checkpointing, early stopping
- **Augmentation** - Image + time-series data augmentation
- **Callbacks** - Progress tracking, learning rate scheduling, metric logging

---

## 📋 Implementation Tasks (Prioritized)

### **Phase 3.1: Validation & Testing (Week 1)**

#### Task 1.1: Transform Validation Script
**File:** `experiments/phase3/01_validate_transforms.py`

Create comprehensive validation script that:
- Loads sample preprocessed EEG data
- Tests each transformation method
- Verifies output shapes and value ranges
- Generates sample images for visual inspection
- Tests on both single and multi-channel epochs
- Estimates computational time per transform

**Expected Output:**
```
Transform Validation Results
=============================
GAF Summation:      Output shape (128, 128), Time: 2.3ms, Value range [0, 1]
GAF Difference:     Output shape (128, 128), Time: 2.1ms, Value range [-1, 1]
MTF (Q=8):          Output shape (64, 64), Time: 1.8ms, Value range [0, 1]
MTF (Q=16):         Output shape (64, 64), Time: 2.1ms, Value range [0, 1]
Recurrence Plot:    Output shape (128, 128), Time: 4.5ms, Value range [0, 1]
Spectrogram:        Output shape (128, 128), Time: 3.2ms, Value range [0, 1]
Scalogram Morlet:   Output shape (64, 64), Time: 5.1ms, Value range [0, 1]
Topographic (SSFI): Output shape (64, 64, 5), Time: 8.3ms, Value range [0, 1]
```

#### Task 1.2: Model Architecture Validation
**File:** `experiments/phase3/02_validate_models.py`

Verify all models work with sample inputs:
- Test forward pass with dummy data
- Count parameters
- Test on different image sizes
- Verify output dimensions
- Check GPU/CPU compatibility

#### Task 1.3: End-to-End Pipeline Test
**File:** `experiments/phase3/03_test_pipeline.py`

Complete pipeline test:
1. Load BCI IV-2a preprocessed data
2. Apply GAF transformation
3. Create dataset and dataloader
4. Train ResNet18 for 1 epoch
5. Verify no data leakage
6. Check memory usage

### **Phase 3.2: Experiment Configuration (Week 1-2)**

#### Task 2.1: Configuration Templates
**Directory:** `experiments/configs/phase3/`

Create YAML configuration files for:
- `base_config.yaml` - Shared parameters
- `dataset_configs.yaml` - Dataset-specific settings
- `transform_configs.yaml` - Transform parameters
- `model_configs.yaml` - Model architectures
- `training_configs.yaml` - Hyperparameters

Example structure:
```yaml
experiment:
  name: "BCI_IV_2a_GAF_ResNet18_KFold5"
  dataset: "bci_iv_2a"
  transform: "gaf_summation"
  model: "resnet18"
  validation_strategy: "stratified_kfold"

transform:
  image_size: 128
  channel_strategy: "stacking"  # or 'per_channel', 'topographic'
  normalize: true

model:
  architecture: "resnet18"
  pretrained: true
  dropout: 0.5

training:
  learning_rate: 0.001
  batch_size: 32
  epochs: 50
  optimizer: "adam"
  early_stopping_patience: 5
```

#### Task 2.2: Experiment Generator
**File:** `experiments/phase3/generate_configs.py`

Script to generate all experiment combinations:
- 3 datasets (BCI IV-2a, PhysioNet, P300)
- 6 transforms (GAF, MTF, RP, Spectrogram, Scalogram, Topographic)
- 4 image models (ResNet18/34, ViT-B, LightweightCNN)
- 3 baseline models (1D CNN, LSTM, Transformer)
- Stratified K-Fold (5) + LOSO validations

**Total configurations:** 3 × 6 × 4 × 5 folds = 360 image model experiments
**Plus baselines:** 3 × 3 × 5 folds = 45 baseline experiments
**Grand total:** ~405 training runs

### **Phase 3.3: Baseline Training (Week 2)**

#### Task 3.1: Raw-Signal Baseline Experiments
**File:** `experiments/phase3/04_train_baselines.py`

Establish performance ceiling with raw time-series models:
- 1D CNN (reference CNN for time-series)
- BiLSTM (temporal dependencies)
- Transformer (attention on time-series)

These establish what can be achieved without image transformation.

### **Phase 3.4: Image Transformation Training (Week 3-4)**

#### Task 4.1: Batch Experiment Runner
**File:** `experiments/phase3/05_run_experiments.py`

Main orchestration script:
- Load configuration
- Load and preprocess data
- Apply image transformation
- Initialize model
- Run k-fold or LOSO validation
- Save metrics and checkpoints
- Log to Weights & Biases

Usage:
```bash
python experiments/phase3/05_run_experiments.py \
    --config experiments/configs/phase3/bci_iv_2a_gaf_resnet18.yaml \
    --device cuda:0 \
    --output results/phase3/
```

#### Task 4.2: Distributed Training Support (Optional)
If using multiple GPUs:
- Wrap trainer with DistributedDataParallel
- Implement DDP initialization
- Handle synchronized metrics

### **Phase 3.5: Analysis & Visualization (Week 4-5)**

#### Task 5.1: Results Aggregation
**File:** `experiments/phase3/06_aggregate_results.py`

Collect all experiment results:
- Load metric JSON files
- Create master DataFrame
- Compute statistics (mean, std, SEM)
- Perform pairwise statistical tests

Output: `results/phase3/all_results.csv`

#### Task 5.2: Analysis Notebook
**File:** `notebooks/05_phase3_analysis.ipynb`

Comprehensive analysis:
1. **Descriptive Statistics:** Summary tables per dataset/transform/model
2. **Comparisons:** Pairwise Wilcoxon tests, multiple comparison correction
3. **ANOVA:** Main effects of transform, model, dataset
4. **Visualization:**
   - Bar charts (accuracy across transforms)
   - Box plots (distribution across subjects/folds)
   - Heatmaps (transform × model performance)
   - Best performers table

#### Task 5.3: Robustness Analysis
**File:** `experiments/phase3/07_robustness_tests.py`

Test generalization under perturbations:
- **Noise injection:** SNR levels [20, 15, 10, 5, 0, -5 dB]
- **Channel dropout:** [0%, 10%, 20%, 30%, 50%]
- **Temporal jitter:** ±50ms random shifts
- **Cross-session:** Train on session 1, test on session 2

#### Task 5.4: Results Visualization
**File:** `experiments/phase3/08_generate_results_figures.py`

Publication-quality figures:
1. **Accuracy Comparison** - Main results across methods
2. **Robustness Curves** - Noise/dropout performance
3. **Confusion Matrices** - Best performers per dataset
4. **Statistical Significance** - Heatmap of p-values
5. **Feature Embeddings** - t-SNE of learned representations
6. **Attention Maps** - ViT attention visualization

---

## 🔄 Workflow Summary

```
Phase 3 Implementation Workflow
================================

Week 1: VALIDATION & TESTING
├── Validate all 6 transforms
├── Verify model architectures
└── Test end-to-end pipeline

Week 2: CONFIGURATION & BASELINES
├── Create experiment configs
├── Generate config combinations
└── Train raw-signal baselines

Week 3-4: IMAGE TRANSFORMATION EXPERIMENTS
├── Run full experiment grid
│   ├── 360 image model experiments
│   └── Distributed across GPUs
└── Monitor progress with Weights & Biases

Week 5: ANALYSIS & VISUALIZATION
├── Aggregate all results
├── Statistical analysis
├── Generate figures
└── Create final report

Post-Phase: PUBLICATION PREPARATION
├── Write manuscript draft
├── Create supplementary materials
├── Archive code and data
└── Submit to journal/conference
```

---

## 📊 Expected Results Preview

### Performance Range (BCI IV-2a Dataset)
Based on literature and Phase 2 preprocessing quality:

| Method | Typical Accuracy | Expected Range |
|--------|-----------------|-----------------|
| Random Baseline | 25% | — |
| Raw 1D CNN | 75-80% | 70-85% |
| GAF + ResNet18 | 82-87% | 80-90% |
| GAF + ResNet50 | 84-89% | 82-91% |
| MTF + ResNet18 | 80-85% | 78-88% |
| Spectrogram + ResNet18 | 81-86% | 79-88% |
| Scalogram + ResNet18 | 79-84% | 77-87% |
| Topographic + ResNet18 | 78-83% | 76-86% |
| ViT + GAF | 83-88% | 81-90% |

---

## 🔧 Critical Implementation Notes

### **Data Leakage Prevention**
- Always split data BEFORE applying transforms
- Never compute transform statistics on test data
- Verify no subject overlap in train/test for LOSO

### **Computational Resources**
- GPU strongly recommended (TPU optional)
- Estimated: 400 runs × 2 hours = 800 GPU-hours
- With 4 GPUs: ~200 GPU-hours (1-2 weeks)
- Implement checkpointing for resumable experiments

### **Statistical Rigor**
- Use stratified K-fold (preserve class balance)
- Report mean ± SEM with confidence intervals
- Wilcoxon signed-rank tests for pairwise comparisons
- Bonferroni correction for multiple comparisons
- ANOVA with post-hoc tests for main effects

### **Reproducibility**
- Fix random seeds (numpy, torch, python)
- Document library versions
- Save all configurations
- Archive trained models
- Provide complete reproduction scripts

---

## 📁 Phase 3 Directory Structure

```
experiments/
├── phase3/                          # NEW
│   ├── __init__.py
│   ├── 01_validate_transforms.py   # Validate all 6 transforms
│   ├── 02_validate_models.py       # Verify model architectures
│   ├── 03_test_pipeline.py         # End-to-end test
│   ├── 04_train_baselines.py       # Raw-signal baselines
│   ├── 05_run_experiments.py       # Main experiment runner
│   ├── 06_aggregate_results.py     # Collect results
│   ├── 07_robustness_tests.py      # Perturbation testing
│   ├── 08_generate_results_figures.py  # Visualization
│   └── config_generator.py         # Generate all configs
│
├── configs/
│   └── phase3/                      # NEW
│       ├── base_config.yaml
│       ├── dataset_configs.yaml
│       ├── transform_configs.yaml
│       ├── model_configs.yaml
│       ├── training_configs.yaml
│       └── all_experiments.txt      # Generated list
│
notebooks/
├── 05_phase3_analysis.ipynb         # Results analysis
└── 06_phase3_robustness.ipynb       # Robustness analysis (optional)

results/
└── phase3/                          # NEW
    ├── metrics/                     # CSV/JSON results
    ├── models/                      # Saved checkpoints
    ├── figures/                     # Generated plots
    └── logs/                        # Training logs
```

---

## ✅ Next Immediate Steps

1. **Create validation scripts** (Phase 3.1)
   - Transform validation
   - Model validation
   - Pipeline test

2. **Generate experiment configurations** (Phase 3.2)
   - All 405 experiment configs
   - Distributed runner setup

3. **Train baselines** (Phase 3.3)
   - Raw-signal models
   - Establish performance ceiling

4. **Launch full experiments** (Phase 3.4)
   - Image transformation experiments
   - Monitor progress

5. **Analyze and visualize** (Phase 3.5)
   - Aggregate results
   - Statistical analysis
   - Generate publication figures

---

## 📞 Key Files & Dependencies

**Core Modules (Ready):**
- `src/transforms/` - All 6 transformers implemented
- `src/models/` - CNN, ViT, Baselines ready
- `src/training/` - Trainer, augmentation, callbacks ready
- `src/evaluation/` - Metrics and statistical tests

**Data (From Phase 2):**
- `data/BCI_IV_2a.hdf5` - Preprocessed epochs
- `data/preprocessed/` - Other datasets (if available)

**Dependencies (requirements.txt):**
- torch, torchvision, timm (deep learning)
- mne, scikit-learn (signal processing)
- scipy, numpy, pandas (scientific computing)
- matplotlib, seaborn, plotly (visualization)
- wandb (experiment tracking - optional but recommended)

---

## 📈 Success Metrics

✅ Phase 3 will be considered complete when:

1. **All transforms validated** - Each produces expected output shapes and value ranges
2. **All models verified** - Forward pass successful on different input sizes
3. **Pipeline tested** - End-to-end training runs without errors
4. **Baselines trained** - Raw-signal models establish performance ceiling
5. **Full experiments run** - All 405 configurations complete
6. **Results aggregated** - Master CSV with all metrics
7. **Statistical analysis done** - Significance tests completed
8. **Figures generated** - Publication-quality visualizations ready
9. **Analysis documented** - Comprehensive findings notebook

---

**Status:** Ready to Begin Phase 3
**Last Updated:** April 11, 2026
**Implementation Owner:** Claude Code Agent
