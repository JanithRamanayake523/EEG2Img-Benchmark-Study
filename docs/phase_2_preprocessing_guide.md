# Preprocessing Guide

Complete guide to running the BCI IV-2a preprocessing pipeline.

---

## Overview

The preprocessing pipeline transforms raw BCI IV-2a EEG data into a clean, analysis-ready dataset.

**Input**: Raw GDF files (18 files, 575 MB)
**Output**: EEG-only dataset (22 channels, 2,216 trials, 270 MB)

### Data Flow

```
Raw GDF Files (data/raw/bci_iv_2a/)
    ↓
Step 1: Preprocess each subject (9 subjects × 2 sessions)
    ↓
Individual H5 files (data/preprocessed/bci_iv_2a/)
    ↓
Step 2: Combine all subjects into single HDF5
    ↓
Combined file (data/BCI_IV_2a.hdf5) - 25 channels (22 EEG + 3 EOG)
    ↓
Step 3: Extract 22 EEG channels (exclude 3 EOG channels)
    ↓
Final dataset (data/BCI_IV_2a_EEG_only.hdf5) - READY FOR EXPERIMENTS
```

---

## Current Status

✅ **All preprocessing is already complete!**

```
Raw Data:           ✓ Exists (18 GDF files, 575 MB)
Preprocessed Files: ✓ Complete (9 subjects)
Combined HDF5:      ✓ Exists (25 channels)
EEG-Only Dataset:   ✓ Exists (22 channels)
```

You can skip to **[Running Experiments](#running-experiments)** section.

---

## If You Need to Preprocess from Scratch

### Step 1: Preprocess All Subjects

This step loads raw GDF files, applies preprocessing, and saves individual H5 files.

```bash
python experiments/phase_2_preprocessing/preprocess_all_bci_iv_2a.py
```

**What it does**:
- Loads raw GDF files from `data/raw/bci_iv_2a/`
- Applies ICA for artifact removal
- Filters signals (0.5-40 Hz bandpass, 50/60 Hz notch)
- Epochs data (0.5-3.5 sec post-cue)
- Rejects artifacts (|amplitude| > 100 µV)
- Z-score normalization per channel
- Saves to `data/preprocessed/bci_iv_2a/A0xT_preprocessed.h5`

**Time**: ~30-60 minutes (depends on CPU)

**Output**:
```
data/preprocessed/bci_iv_2a/
├── A01T_preprocessed.h5 (240 trials, 25 channels)
├── A02T_preprocessed.h5 (278 trials, 25 channels)
├── ... (9 files total)
└── A09T_preprocessed.h5 (199 trials, 25 channels)
```

### Step 2: Combine into Single HDF5

This step merges all subject files into one convenient file.

```bash
python experiments/phase_2_preprocessing/combine_preprocessed_data.py
```

**What it does**:
- Reads all individual H5 files
- Combines them into single HDF5
- Preserves metadata (sampling rate, channel names, etc.)
- Applies compression (gzip level 4)

**Time**: ~5 minutes

**Output**:
```
data/BCI_IV_2a.hdf5
├── subject_A01T/
│   ├── signals: (240, 25, 751)
│   └── labels: (240,)
├── subject_A02T/
│   ├── signals: (278, 25, 751)
│   └── labels: (278,)
└── ... (9 subjects)
```

### Step 3: Extract EEG-Only Channels

This step removes the 3 EOG channels, keeping only 22 EEG channels.

```bash
python experiments/phase_2_preprocessing/create_eeg_only_dataset.py
```

**What it does**:
- Loads combined 25-channel HDF5 file
- Extracts channels 0-21 (EEG only)
- Discards channels 22-24 (EOG)
- Saves new EEG-only HDF5 file
- Adds metadata about EEG-only status

**Time**: ~30 seconds

**Output**:
```
data/BCI_IV_2a_EEG_only.hdf5
├── subject_A01T/
│   ├── signals: (240, 22, 751)  ← 22 channels instead of 25
│   └── labels: (240,)
├── subject_A02T/
│   ├── signals: (278, 22, 751)
│   └── labels: (278,)
└── ... (9 subjects)
```

---

## Preprocessing Details

### Input Data Structure

**Raw GDF Files**: 18 files (A01T, A01E, A02T, A02E, ..., A09T)
- **T** = Training session (T for Training)
- **E** = Evaluation session (E for Evaluation)
- **This study uses only T (training) sessions**

Each file contains:
- **Channels**: 25 (22 EEG + 3 EOG)
- **Sampling rate**: 250 Hz
- **Duration**: ~4 minutes per file
- **Classes**: 4 motor imagery (left hand, right hand, feet, tongue)

### Preprocessing Pipeline

1. **ICA-Based Artifact Removal**
   - Independent Component Analysis
   - Removes EOG, EMG, muscle artifacts
   - 25 independent components extracted

2. **Filtering**
   - Bandpass: 0.5-40 Hz (removes DC and high-frequency noise)
   - Notch: 50/60 Hz (removes line noise)

3. **Epoching**
   - Time window: 0.5-3.5 seconds post-cue
   - Baseline correction: mean subtraction (baseline: 0-0.5 sec)
   - Result: 751 timepoints per epoch at 250 Hz

4. **Artifact Rejection**
   - Removes epochs with |amplitude| > 100 µV
   - Automatic quality control

5. **Normalization**
   - Z-score per channel: (x - mean) / std
   - Applied after all other steps

### Output Data Structure

**Per Subject Training Session** (e.g., A01T):
- **Trials**: 288 total (before artifact rejection)
- **Kept**: ~240-280 trials (after artifact rejection)
- **Channels**: 25 or 22 (depending on EOG inclusion)
- **Timepoints**: 751 (0.5-3.5 sec @ 250 Hz)

**All 9 Subjects Pooled**:
- **Total trials**: 2,216 (across all 9 subjects)
- **Channels**: 22 (EEG only, after filtering)
- **Shape**: (2216, 22, 751)

---

## Channel Information

### All 25 Channels (Before Filtering)

**EEG Channels (0-21)** - 22 channels
```
Fz, FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2, C4, C6,
CP3, CP1, CPz, CP2, CP4, P1, Pz, P2, POz
```

**EOG Channels (22-24)** - 3 channels (REMOVED)
```
EOG-left, EOG-central, EOG-right
```

### Final 22 Channels (After EOG Removal)

Only the EEG channels are used:
```
Fz, FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2, C4, C6,
CP3, CP1, CPz, CP2, CP4, P1, Pz, P2, POz
```

---

## Running Experiments

Once preprocessing is complete, you're ready to run experiments!

### Verify the Data

```bash
python << 'EOF'
import h5py
import numpy as np

with h5py.File('data/BCI_IV_2a_EEG_only.hdf5', 'r') as f:
    print(f"Dataset ready!")
    print(f"Subjects: {len(f.keys())}")
    for subj in sorted(f.keys()):
        print(f"  {subj}: {f[subj]['signals'].shape}")
EOF
```

### Run Baseline Model

```bash
python experiments/phase_3_benchmark_experiments/04_train_baselines.py --model cnn1d --device cuda:0
```

### Run Image-Based Model

```bash
python experiments/phase_3_benchmark_experiments/05_run_experiments.py \
    --transform gaf_summation \
    --model resnet18 \
    --device cuda:0
```

---

## Troubleshooting

### Issue: "File not found: data/raw/bci_iv_2a/*.gdf"

**Cause**: Raw GDF files are missing

**Solution**: Download BCI IV-2a dataset from:
- Official: https://www.bbci.de/competition/iv/
- PhysioNet: https://physionet.org/

### Issue: "Out of memory during preprocessing"

**Cause**: Insufficient RAM for ICA computation

**Solution**:
1. Preprocess subjects one at a time:
   ```bash
   python experiments/phase_2_preprocessing/preprocess_bci_iv_2a.py --subject 1 --session T
   ```
2. Then combine them all:
   ```bash
   python experiments/phase_2_preprocessing/combine_preprocessed_data.py
   ```

### Issue: "Module not found: mne"

**Cause**: MNE library not installed

**Solution**:
```bash
pip install mne
```

---

## Expected Results

### Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total subjects** | 9 |
| **Sessions per subject** | 2 (Training + Evaluation, only using Training) |
| **Trials per subject** | 240-280 |
| **Total pooled trials** | 2,216 |
| **EEG Channels** | 22 |
| **EOG Channels (removed)** | 3 |
| **Timepoints per trial** | 751 |
| **Time per trial** | 3.004 seconds (@ 250 Hz) |
| **Frequency range** | 0.5-40 Hz |
| **Sampling rate** | 250 Hz |
| **Classes** | 4 (left hand, right hand, feet, tongue) |

### Class Distribution (Approximately Equal)

Each subject trained on ~4 motor imagery classes:
- Left hand: ~60 trials/subject
- Right hand: ~60 trials/subject
- Feet: ~60 trials/subject
- Tongue: ~60 trials/subject

Total across 9 subjects: ~540 trials per class

---

## Files Involved

### Preprocessing Scripts

1. **`preprocess_bci_iv_2a.py`**
   - Preprocesses single subject
   - Usage: `--subject 1 --session T`

2. **`preprocess_all_bci_iv_2a.py`** ⭐
   - Preprocesses all 9 subjects
   - Usage: No arguments needed

3. **`combine_preprocessed_data.py`** ⭐
   - Combines per-subject files
   - Usage: No arguments needed

4. **`create_eeg_only_dataset.py`** ⭐
   - Extracts 22 EEG channels
   - Usage: No arguments needed

### Input Files

```
data/raw/bci_iv_2a/
├── A01T.gdf (33 MB)
├── A01E.gdf (33 MB)
├── A02T.gdf (33 MB)
├── ...
└── A09T.gdf (32 MB)
Total: 18 files, 575 MB
```

### Output Files

```
data/
├── BCI_IV_2a.hdf5           (307 MB) - 25 channels
├── BCI_IV_2a_EEG_only.hdf5  (270 MB) - 22 channels (MAIN)
└── preprocessed/bci_iv_2a/
    ├── A01T_preprocessed.h5
    ├── A02T_preprocessed.h5
    └── ... (9 files)
```

---

## Next Steps

Once preprocessing is complete:

1. **Validate the data**:
   ```bash
   python experiments/phase_3_benchmark_experiments/03_test_pipeline.py
   ```

2. **Run first experiment**:
   ```bash
   python experiments/phase_3_benchmark_experiments/04_train_baselines.py --model cnn1d --device cuda:0
   ```

3. **Run complete benchmark**:
   ```bash
   python experiments/phase_3_benchmark_experiments/run_all_experiments.py --device cuda:0
   ```

---

## Summary

✅ **Preprocessing Status**: COMPLETE

- Raw data: 575 MB (18 GDF files)
- Preprocessed: 307 MB (25 channels)
- EEG-only: 270 MB (22 channels) ← **Use this for experiments**

All data is ready for experiments. No additional preprocessing needed!

**Ready to run?** Start with: `python experiments/phase_3_benchmark_experiments/04_train_baselines.py --model cnn1d --device cuda:0`

---

**Last Updated**: 2026-04-12
