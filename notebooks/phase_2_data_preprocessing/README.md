# Phase 2: Data Preprocessing - Notebooks Guide

This folder contains notebooks for understanding the EEG data preprocessing pipeline.

---

## 📓 Available Notebooks

### 1. **PHASE_2_DETAILED_EXPLANATION.ipynb**
*Quick Overview - No Computation*

**Purpose:** Educational explanation of the preprocessing pipeline
**Runtime:** < 1 minute
**Execution:** Reads pre-existing preprocessed data

**What it does:**
- Loads the combined preprocessed dataset (`data/BCI_IV_2a.hdf5`)
- Shows data specifications and structure
- Explains each preprocessing step with diagrams
- Provides code examples for data access

**When to use:**
- Quick reference for preprocessing steps
- Understanding the final data format
- Learning how to access preprocessed data

**Does NOT include:**
- Actual preprocessing execution
- Visual comparisons of signals
- Step-by-step analysis

---

### 2. **PHASE_2_COMPLETE_ANALYSIS.ipynb** ⭐ **RECOMMENDED**
*Comprehensive Analysis with Visualizations*

**Purpose:** Complete step-by-step preprocessing with visual analysis
**Runtime:** 5-10 minutes (depending on hardware)
**Execution:** Runs actual preprocessing on one subject (A01T)

**What it does:**
- Loads **raw** EEG data from GDF files
- Executes each preprocessing step:
  1. Band-pass filtering (0.5-40 Hz)
  2. Notch filtering (50 Hz)
  3. ICA artifact removal
  4. Epoch extraction
  5. Artifact rejection
  6. Z-score normalization
- **Visual comparisons** before/after each step
- Power spectral density analysis
- ICA component visualization
- Signal quality metrics
- Final data summary with comprehensive plots

**When to use:**
- **First time learning** about the preprocessing pipeline
- Understanding **what changes** at each step
- Seeing **visual proof** of preprocessing effects
- Analyzing signal quality and artifact removal

**Requirements:**
- Raw data must be downloaded: `data/raw/bci_iv_2a/A01T.gdf`
- MNE-Python library installed
- ~5-10 minutes of computation time

**Output:**
- 15+ visualizations showing preprocessing effects
- Statistical comparisons
- Quality metrics
- Complete understanding of the pipeline

---

## 🚀 Quick Start

### For Complete Understanding (Recommended):

```bash
# 1. Open the comprehensive analysis notebook
jupyter notebook PHASE_2_COMPLETE_ANALYSIS.ipynb

# 2. Run all cells (Kernel → Restart & Run All)
# 3. See visual comparisons at each preprocessing step
```

### For Quick Reference:

```bash
# 1. Open the explanation notebook
jupyter notebook PHASE_2_DETAILED_EXPLANATION.ipynb

# 2. Read through the explanations
# 3. Run cells to see data structure
```

---

## 📊 What You'll Learn

### PHASE_2_DETAILED_EXPLANATION.ipynb
- ✓ Dataset specifications (9 subjects, 2,216 trials)
- ✓ Data structure (epochs × channels × samples)
- ✓ Class distribution (4 motor imagery tasks)
- ✓ How to access preprocessed data
- ✓ Next steps for Phase 3

### PHASE_2_COMPLETE_ANALYSIS.ipynb
- ✓ **Visual proof** of filtering effects (PSD before/after)
- ✓ **ICA component maps** (identify artifacts)
- ✓ **Artifact removal comparison** (frontal channels)
- ✓ **Epoch extraction** (average per class)
- ✓ **Artifact rejection** (amplitude distribution)
- ✓ **Normalization effect** (z-score comparison)
- ✓ **Final data quality** (comprehensive summary)

---

## 🎯 Which Notebook Should I Use?

| Scenario | Recommended Notebook |
|----------|---------------------|
| **First time learning about preprocessing** | PHASE_2_COMPLETE_ANALYSIS.ipynb |
| **Want to see visual comparisons** | PHASE_2_COMPLETE_ANALYSIS.ipynb |
| **Need to understand what each step does** | PHASE_2_COMPLETE_ANALYSIS.ipynb |
| **Quick reference for data format** | PHASE_2_DETAILED_EXPLANATION.ipynb |
| **Just need to load preprocessed data** | PHASE_2_DETAILED_EXPLANATION.ipynb |
| **Limited time (< 2 minutes)** | PHASE_2_DETAILED_EXPLANATION.ipynb |
| **Have 10 minutes and want full understanding** | PHASE_2_COMPLETE_ANALYSIS.ipynb ⭐ |

---

## 📁 Data Files Required

### For PHASE_2_DETAILED_EXPLANATION.ipynb:
```
data/BCI_IV_2a.hdf5  (preprocessed, 321 MB)
```

### For PHASE_2_COMPLETE_ANALYSIS.ipynb:
```
data/raw/bci_iv_2a/A01T.gdf  (raw data, must download)
```

If you don't have the raw data, you can still run the explanation notebook.

---

## 🔧 Troubleshooting

### "Data file not found"
**For PHASE_2_DETAILED_EXPLANATION.ipynb:**
```bash
# Run preprocessing script first
python experiments/scripts/preprocess_all_bci_iv_2a.py

# Then combine per-subject files
python experiments/scripts/combine_preprocessed_data.py
```

**For PHASE_2_COMPLETE_ANALYSIS.ipynb:**
- Download BCI Competition IV-2a dataset
- Place in `data/raw/bci_iv_2a/`
- Or run the download script (if available)

### "Module not found: mne"
```bash
pip install mne
```

### Jupyter not installed
```bash
pip install jupyter notebook
```

---

## 📖 Key Concepts Explained

Both notebooks explain these preprocessing steps:

1. **Band-Pass Filtering (0.5-40 Hz)**
   - Removes DC drift and slow fluctuations
   - Removes high-frequency muscle noise
   - Keeps motor imagery relevant frequencies

2. **Notch Filtering (50 Hz)**
   - Removes power line electrical noise
   - Essential for clean European data

3. **ICA Artifact Removal**
   - Identifies independent signal sources
   - Removes eye blinks (frontal components)
   - Removes muscle artifacts (lateral components)

4. **Epoch Extraction**
   - Creates fixed-length trials (0.5 to 3.5 seconds)
   - Aligns to motor imagery cue onset
   - Enables trial-by-trial analysis

5. **Artifact Rejection**
   - Removes trials with amplitude > 100 µV
   - Ensures data quality
   - Typical rejection rate: 1-2%

6. **Z-Score Normalization**
   - Standardizes each channel: mean=0, std=1
   - Enables fair comparison across channels
   - Required for neural network training

---

## ✅ Learning Outcomes

After running **PHASE_2_COMPLETE_ANALYSIS.ipynb**, you will:

- ✓ Understand **exactly** what preprocessing does to EEG signals
- ✓ See **visual proof** of filtering, ICA, and normalization effects
- ✓ Know how to **identify artifacts** in raw data
- ✓ Understand **why** each preprocessing step is necessary
- ✓ Be able to **explain** the preprocessing pipeline to others
- ✓ Have **confidence** that the data is properly prepared

---

## 🔜 Next Steps

After completing Phase 2 notebooks:

1. ✅ Understand preprocessing pipeline
2. ✅ Verify data quality (2,216 trials, 25 channels, 751 samples)
3. ➡️ **Move to Phase 3:** Image Transformation
   - Location: `notebooks/phase_3_image_transformations/`
   - Transform EEG signals into 6 image types
   - Prepare for CNN/ViT training

---

## 📚 Additional Resources

- **Phase 2 Validation:** `docs/validation/PHASE3_VALIDATION.md`
- **Phase 2 Verification:** `PHASE_2_VERIFICATION.txt` (project root)
- **Source Code:** `src/data/preprocessors.py`
- **Preprocessing Script:** `experiments/scripts/preprocess_all_bci_iv_2a.py`

---

## 💡 Tips

- **Start with PHASE_2_COMPLETE_ANALYSIS.ipynb** for best learning experience
- Run cells sequentially (don't skip steps)
- Read the markdown explanations between code cells
- Examine the visualizations carefully
- Compare before/after plots to understand effects
- Save the notebook with outputs for future reference

---

**Questions?** Check the main project [README.md](../../README.md) or [DOCUMENTATION_MAP.md](../../DOCUMENTATION_MAP.md)
