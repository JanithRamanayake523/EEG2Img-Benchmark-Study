# Documentation Index

Complete documentation for the EEG-to-Image Transformation Benchmark Study.

---

## Getting Started

- **[Quick Start Guide](QUICK_START.md)** - Get up and running in minutes
- **[Project README](../README.md)** - Main project overview

---

## Phase Guides

All phase execution guides are organized by phase with consistent naming:

### Phase 2: EEG Preprocessing
- **[Phase 2 Preprocessing Guide](phase_2_preprocessing_guide.md)**
  - Complete preprocessing steps
  - Data pipeline details
  - How to re-run preprocessing
- **Status**: ✓ COMPLETE & LOCKED

### Phase 3: Benchmark Experiments
- **[Phase 3 Benchmark Experiments Guide](phase_3_benchmark_experiments_guide.md)**
  - 4-stage execution roadmap
  - Validation, baselines, experiments, analysis
  - Expected results and troubleshooting
- **Status**: READY TO EXECUTE
- **Quick Start**: See [NEXT_STEPS.md](../NEXT_STEPS.md)

---

## Setup Guides

- **[GPU Setup Guide](gpu_setup_guide.md)** - Configure CUDA and GPU for training

---

## Quick Commands

```bash
# Phase 3 Validation (10-15 min)
python experiments/phase_3_benchmark_experiments/01_validate_transforms.py
python experiments/phase_3_benchmark_experiments/02_validate_models.py
python experiments/phase_3_benchmark_experiments/03_test_pipeline.py

# Phase 3 Training Baselines (1-2 hours)
python experiments/phase_3_benchmark_experiments/04_train_baselines.py --model cnn1d --device cuda:0

# Phase 3 Run Experiments (1-2 hours per combo)
python experiments/phase_3_benchmark_experiments/05_run_experiments.py \
  --transform gaf_summation --model resnet18 --device cuda:0

# Phase 3 Run All (12-24 hours)
python experiments/phase_3_benchmark_experiments/run_all_experiments.py --device cuda:0
```

---

**Last Updated**: 2026-04-12
**Status**: Phase 2 Complete, Phase 3 Ready
