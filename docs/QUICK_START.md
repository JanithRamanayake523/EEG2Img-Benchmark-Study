# Quick Start Guide

Get up and running with the EEG-to-Image benchmark in minutes.

---

## Prerequisites

1. **Python 3.8+** installed
2. **NVIDIA GPU** with CUDA 11.8+ (recommended)
3. **16GB+ RAM**

---

## Installation

```bash
# Clone the repository
cd EEG2Img-Benchmark-Study

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

---

## Data Setup

### Option 1: Use Existing EEG-Only Dataset

If `data/BCI_IV_2a_EEG_only.hdf5` exists (270 MB), you're ready to go!

### Option 2: Create EEG-Only Dataset

```bash
# Extract 22 EEG channels from preprocessed data
python experiments/phase_2_preprocessing/create_eeg_only_dataset.py
```

**Output**: `data/BCI_IV_2a_EEG_only.hdf5`
- 2,216 trials (pooled from 9 subjects)
- 22 EEG channels (EOG excluded)
- 751 timepoints per trial (3.004 sec @ 250 Hz)

---

## Run Your First Experiment

### 1. Validate Everything Works

```bash
# Test transformations (should pass 9/9)
python experiments/phase_3_benchmark_experiments/01_validate_transforms.py

# Test models (should pass 8/8)
python experiments/phase_3_benchmark_experiments/02_validate_models.py

# Test complete pipeline
python experiments/phase_3_benchmark_experiments/03_test_pipeline.py
```

### 2. Train a Baseline Model

```bash
# Train 1D CNN on raw EEG signals (~10 min on GPU)
python experiments/phase_3_benchmark_experiments/04_train_baselines.py --model cnn1d --device cuda:0
```

**Expected output**:
```
Fold 1/5 - Best Val Kappa: 0.6132, Acc: 70.85% (Epoch 65)
Fold 2/5 - Best Val Kappa: 0.6078, Acc: 70.45% (Epoch 62)
...
Final Result - Accuracy: 70.58 ± 0.36%
Final Result - Kappa: 0.6078 ± 0.0048
```

### 3. Train an Image-Based Model

```bash
# Train ResNet-18 with GAF transformation (~25 min on GPU)
python experiments/phase_3_benchmark_experiments/05_run_experiments.py \
    --transform gaf_summation \
    --model resnet18 \
    --device cuda:0
```

**Expected output**:
```
Transforming 2216 signals to images...
Transformation complete: (2216, 1, 128, 128) in 45.32s
...
Fold 1/5 - Best Val Kappa: 0.7128, Acc: 76.45% (Epoch 64)
...
Final Result - Accuracy: 75.23 ± 2.15%
Final Result - Kappa: 0.7025 ± 0.0287
```

---

## Understanding the Output

### Results Location

```
results/phase3/
├── baselines/
│   └── cnn1d_all_fold_results.json
└── experiments/
    └── gaf_summation_resnet18_all_fold_results.json
```

### Result JSON Format

```json
{
  "model": "resnet18",
  "transform": "gaf_summation",
  "mean_accuracy": 75.23,
  "std_accuracy": 2.15,
  "mean_kappa": 0.7025,
  "std_kappa": 0.0287,
  "fold_results": [
    {
      "fold": 1,
      "best_val_acc": 76.45,
      "best_val_kappa": 0.7128,
      "best_epoch": 64,
      "training_time_s": 1823.4
    },
    ...
  ]
}
```

---

## Run All Experiments

```bash
# Run complete benchmark (9 transforms × 5 models = 45 combinations)
# Estimated time: 12-24 hours on GPU
python experiments/phase_3_benchmark_experiments/run_all_experiments.py --device cuda:0
```

This will run:
- 3 baseline models (1D CNN, BiLSTM, Transformer)
- 45 image-based combinations (9 transforms × 5 models)

---

## Common Issues

### 1. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solution**: Reduce batch size
```bash
python experiments/phase_3_benchmark_experiments/05_run_experiments.py \
    --transform gaf_summation \
    --model resnet18 \
    --batch_size 16 \  # Default is 32
    --device cuda:0
```

### 2. Data File Not Found

**Error**: `FileNotFoundError: data/BCI_IV_2a_EEG_only.hdf5`

**Solution**: Create the EEG-only dataset
```bash
python experiments/phase_2_preprocessing/create_eeg_only_dataset.py
```

### 3. Import Errors

**Error**: `ModuleNotFoundError: No module named 'src'`

**Solution**: Reinstall in development mode
```bash
pip install -e .
```

---

## Next Steps

1. **Explore different models**:
   ```bash
   # BiLSTM
   python experiments/phase_3_benchmark_experiments/04_train_baselines.py --model bilstm --device cuda:0

   # Vision Transformer
   python experiments/phase_3_benchmark_experiments/05_run_experiments.py --transform mtf --model vit_base --device cuda:0
   ```

2. **Try different transformations**:
   - `gaf_summation`, `gaf_difference` (GAF variants)
   - `mtf` (Markov Transition Field)
   - `recurrence` (Recurrence Plot)
   - `spectrogram`, `cwt`, `stft` (Time-frequency)

3. **Aggregate and visualize results**:
   ```bash
   python experiments/phase_3_benchmark_experiments/06_aggregate_results.py
   python experiments/phase_3_benchmark_experiments/08_generate_results_figures.py
   ```

---

## Performance Expectations

Based on BCI IV-2a motor imagery classification:

| Model Type | Expected Accuracy | Expected Kappa |
|------------|------------------|----------------|
| **Baselines** | 68-72% | 0.57-0.63 |
| **Image-Based** | 72-78% | 0.63-0.71 |

**Note**: Results vary depending on transformation and model architecture.

---

## Getting Help

- **Documentation**: See [`docs/`](docs/) for detailed guides
- **Issues**: Check [`docs/phase_3_benchmark_experiments_guide.md`](phase_3_benchmark_experiments_guide.md) for troubleshooting
- **Code Examples**: See [`notebooks/`](notebooks/) for Jupyter notebooks

---

**Ready to experiment? Start with the baseline model and work your way up to image-based models!**
