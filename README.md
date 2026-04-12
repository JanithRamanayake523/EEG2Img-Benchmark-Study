# EEG-to-Image Transformation Benchmark Study

A comprehensive benchmark study comparing different EEG-to-image transformation methods for motor imagery classification using deep learning.

## Overview

This project investigates whether transforming EEG signals into 2D images can improve classification performance using state-of-the-art computer vision models.

**Dataset**: BCI Competition IV-2a (9 subjects, 4 motor imagery classes)
**EEG Channels**: 22 channels (EEG only, EOG excluded)
**Total Trials**: 2,216 (pooled across all subjects)
**Task**: 4-class motor imagery classification (left hand, right hand, feet, tongue)

---

## Quick Start

### 1. Prerequisites

```bash
# Python 3.8+
pip install -r requirements.txt

# For GPU support (recommended)
# CUDA 11.8+ and compatible PyTorch
```

### 2. Data Preparation

The preprocessed EEG-only dataset is required:

```bash
# If not already created, generate EEG-only dataset (22 channels)
python experiments/phase_2_preprocessing/create_eeg_only_dataset.py
```

**Output**: `data/BCI_IV_2a_EEG_only.hdf5` (270 MB)
- Shape: (2216, 22, 751)
- 22 EEG channels, 751 timepoints (3.004 sec @ 250 Hz)

### 3. Run Experiments

#### Baseline Models (Raw Time-Series)

```bash
# 1D CNN
python experiments/phase_3_benchmark_experiments/04_train_baselines.py --model cnn1d --device cuda:0

# BiLSTM
python experiments/phase_3_benchmark_experiments/04_train_baselines.py --model bilstm --device cuda:0

# Transformer
python experiments/phase_3_benchmark_experiments/04_train_baselines.py --model transformer --device cuda:0
```

#### Image-Based Models

```bash
# ResNet-18 with GAF transformation
python experiments/phase_3_benchmark_experiments/05_run_experiments.py \
    --transform gaf_summation \
    --model resnet18 \
    --device cuda:0

# Vision Transformer with MTF transformation
python experiments/phase_3_benchmark_experiments/05_run_experiments.py \
    --transform mtf \
    --model vit_base \
    --device cuda:0
```

#### Run All Experiments

```bash
# Run complete benchmark (9 transforms × 5 models = 45 combinations)
python experiments/phase_3_benchmark_experiments/run_all_experiments.py --device cuda:0
```

---

## Project Structure

```
EEG2Img-Benchmark-Study/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── data/                              # Data files
│   ├── BCI_IV_2a_EEG_only.hdf5       # Primary dataset (22 EEG channels)
│   ├── raw/                           # Raw BCI IV-2a files (downloaded)
│   └── preprocessed/                  # Individual preprocessed files
│
├── src/                               # Source code
│   ├── data/                          # Data loaders and preprocessors
│   ├── models/                        # Model architectures
│   │   ├── baselines.py              # 1D CNN, BiLSTM, Transformer
│   │   ├── cnn_models.py             # ResNet, Lightweight CNN
│   │   └── vit_models.py             # Vision Transformer variants
│   ├── transforms/                    # EEG-to-image transformations
│   │   ├── gaf.py                    # Gramian Angular Field
│   │   ├── mtf.py                    # Markov Transition Field
│   │   └── recurrence.py             # Recurrence Plot
│   └── utils/                         # Utilities
│
├── experiments/                       # Experiment scripts
│   ├── phase3/                        # Phase 3: Benchmark experiments
│   │   ├── 01_validate_transforms.py # Validate transformations
│   │   ├── 02_validate_models.py     # Validate model architectures
│   │   ├── 03_test_pipeline.py       # Test end-to-end pipeline
│   │   ├── 04_train_baselines.py     # Train baseline models
│   │   ├── 05_run_experiments.py     # Run image-based experiments
│   │   ├── 06_aggregate_results.py   # Aggregate results
│   │   ├── 07_robustness_tests.py    # Robustness testing
│   │   ├── 08_generate_results_figures.py  # Generate figures
│   │   └── run_all_experiments.py    # Run complete benchmark
│   │
│   └── preprocessing/                 # Data preprocessing
│       ├── preprocess_bci_iv_2a.py   # Preprocess single subject
│       ├── preprocess_all_bci_iv_2a.py  # Preprocess all subjects
│       ├── combine_preprocessed_data.py  # Combine into single HDF5
│       └── create_eeg_only_dataset.py    # Extract 22 EEG channels
│
├── results/                           # Results (generated during runs)
│   ├── phase3/                        # Phase 3 results
│   │   ├── baselines/                # Baseline model results
│   │   └── experiments/              # Image-based model results
│   ├── figures/                       # Generated figures
│   └── logs/                          # Training logs
│
├── docs/                              # Documentation
│   ├── README.md                      # Documentation index
│   ├── reference/                     # Reference documentation
│   └── archive/                       # Archived documentation
│
└── notebooks/                         # Jupyter notebooks (reference)
    └── phase_2_data_preprocessing/   # Phase 2 preprocessing notebooks
```

---

## Experiment Details

### Transformations (9 types)

1. **Gramian Angular Field (GAF)**
   - `gaf_summation`: GASF
   - `gaf_difference`: GADF

2. **Markov Transition Field (MTF)**
   - `mtf`: Standard MTF

3. **Recurrence Plot (RP)**
   - `recurrence`: Standard RP
   - `recurrence_quantified`: With RQA features

4. **Spectrogram**
   - `spectrogram`: Time-frequency representation

5. **Continuous Wavelet Transform (CWT)**
   - `cwt`: Scalogram
   - `cwt_morlet`: Using Morlet wavelet

6. **Short-Time Fourier Transform (STFT)**
   - `stft`: Time-frequency representation

### Models (8 architectures)

**Baseline Models (Raw Time-Series)**
- 1D CNN: Convolutional network for time-series
- BiLSTM: Bidirectional LSTM
- Transformer: Self-attention on time-series

**Image-Based Models (2D)**
- ResNet-18: Pretrained on ImageNet
- ResNet-50: Larger pretrained ResNet
- Lightweight CNN: Custom small CNN (from scratch)
- ViT Base: Vision Transformer (pretrained)
- ViT Small: Smaller Vision Transformer (pretrained)

### Training Configuration

| Parameter | Value |
|-----------|-------|
| **Optimizer** | AdamW |
| **Learning Rate** | 0.001 |
| **Weight Decay** | 0.0001 (image models) |
| **Loss Function** | CrossEntropyLoss |
| **Early Stopping Metric** | Cohen's Kappa |
| **Warmup Epochs** | 50 |
| **Patience** | 15 epochs |
| **Max Epochs** | 100 |
| **Batch Size** | 32 |
| **Cross-Validation** | 5-fold stratified |

### Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **F1 Score (Weighted)**: Accounts for class imbalance
- **Cohen's Kappa**: Agreement adjusted for chance (used for early stopping)
- **Confusion Matrix**: Per-class performance
- **Training Time**: Time per epoch and total

---

## Key Features

### Data Pipeline
- ✅ ICA-based artifact removal
- ✅ Band-pass filtering (0.5-40 Hz)
- ✅ Notch filtering (50/60 Hz)
- ✅ Epoching with baseline correction
- ✅ Amplitude-based artifact rejection
- ✅ Z-score normalization per channel
- ✅ **EEG-only (22 channels, EOG excluded)**

### Transformation Pipeline
- ✅ Per-channel transformation (22 channels → 22 images)
- ✅ Automatic averaging to single-channel image (22 images → 1 image)
- ✅ Consistent image size (128×128)
- ✅ Normalization to [0, 1] range

### Training Infrastructure
- ✅ AdamW optimizer for better generalization
- ✅ Kappa-based early stopping (more robust than accuracy)
- ✅ 50-epoch warmup before early stopping
- ✅ 5-fold stratified cross-validation
- ✅ Automatic GPU/CPU selection
- ✅ Comprehensive logging and checkpointing

---

## Results Directory Structure

After running experiments, results are organized as:

```
results/
├── phase3/
│   ├── baselines/
│   │   ├── cnn1d_all_fold_results.json
│   │   ├── bilstm_all_fold_results.json
│   │   └── transformer_all_fold_results.json
│   │
│   └── experiments/
│       ├── gaf_summation_resnet18_all_fold_results.json
│       ├── mtf_vit_base_all_fold_results.json
│       └── ... (45 combinations)
│
└── figures/
    ├── comparison_accuracy.png
    ├── confusion_matrices/
    └── training_curves/
```

Each result JSON contains:
- Mean accuracy ± std
- Mean Kappa ± std
- Per-fold results
- Training history
- Best epoch information
- Training time

---

## Common Commands

### Validate Installation

```bash
# Validate all transformations
python experiments/phase_3_benchmark_experiments/01_validate_transforms.py

# Validate all models
python experiments/phase_3_benchmark_experiments/02_validate_models.py

# Test complete pipeline
python experiments/phase_3_benchmark_experiments/03_test_pipeline.py
```

### Run Specific Experiments

```bash
# Single transformation + model combination
python experiments/phase_3_benchmark_experiments/05_run_experiments.py \
    --transform <transform_name> \
    --model <model_name> \
    --device cuda:0

# Examples:
python experiments/phase_3_benchmark_experiments/05_run_experiments.py --transform gaf_summation --model resnet18 --device cuda:0
python experiments/phase_3_benchmark_experiments/05_run_experiments.py --transform mtf --model vit_base --device cuda:0
python experiments/phase_3_benchmark_experiments/05_run_experiments.py --transform recurrence --model lightweight_cnn --device cuda:0
```

### Aggregate and Visualize Results

```bash
# Aggregate all results
python experiments/phase_3_benchmark_experiments/06_aggregate_results.py

# Generate figures
python experiments/phase_3_benchmark_experiments/08_generate_results_figures.py
```

---

## Hardware Requirements

### Minimum
- **CPU**: 4 cores
- **RAM**: 16 GB
- **Storage**: 5 GB free space

### Recommended
- **GPU**: NVIDIA GPU with 8+ GB VRAM (RTX 3070 or better)
- **CPU**: 8+ cores
- **RAM**: 32 GB
- **Storage**: 10 GB free space

### Expected Runtime
- **Baseline model (1 fold)**: ~5-10 minutes on GPU
- **Image-based model (1 fold)**: ~15-30 minutes on GPU
- **Complete benchmark (45 combinations)**: ~12-24 hours on GPU

---

## Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
python experiments/phase_3_benchmark_experiments/05_run_experiments.py \
    --transform gaf_summation \
    --model resnet18 \
    --batch_size 16 \
    --device cuda:0
```

### Data File Not Found
```bash
# Ensure EEG-only dataset exists
python experiments/phase_2_preprocessing/create_eeg_only_dataset.py
```

### Import Errors
```bash
# Reinstall package in development mode
pip install -e .
```

---

## Citation

If you use this codebase, please cite:

```bibtex
@article{eeg2img_benchmark_2026,
  title={Benchmarking EEG-to-Image Transformations for Motor Imagery Classification},
  author={Your Name},
  journal={Under Review},
  year={2026}
}
```

---

## License

MIT License - See LICENSE file for details

---

## Changelog

### Version 1.0 (2026-04-12)
- ✅ Phase 2: Complete preprocessing pipeline
- ✅ Phase 3: Benchmark experiments implemented
- ✅ EEG-only data extraction (22 channels)
- ✅ AdamW optimizer + Kappa-based early stopping
- ✅ 9 transformations × 5 image models + 3 baselines
- ✅ Comprehensive documentation and code organization

---

**Status**: Ready for production runs
**Last Updated**: 2026-04-12
