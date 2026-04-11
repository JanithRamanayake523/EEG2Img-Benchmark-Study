# Phase 2: Data Preprocessing - The Complete Guide

This folder contains the comprehensive notebook for understanding the EEG data preprocessing pipeline.

---

## 📓 Main Notebook

### **PHASE_2_COMPLETE_ANALYSIS.ipynb** ⭐ THE ONLY NOTEBOOK YOU NEED

**Purpose:** Complete step-by-step preprocessing pipeline with comprehensive visual analysis
**Runtime:** 5-10 minutes (depending on hardware)
**All-in-One Solution:** Contains both preprocessing execution AND educational explanations

**What it includes:**

✅ **Complete Preprocessing Pipeline**
- Load **raw** EEG data from GDF files
- Band-pass filtering (0.5-40 Hz)
- Notch filtering (50 Hz)
- ICA artifact removal
- Epoch extraction
- Artifact rejection
- Z-score normalization

✅ **15+ Publication-Quality Visualizations**
- Raw signals before preprocessing
- Filtering effects (PSD analysis)
- ICA components (time series)
- Artifact removal comparison
- Motor imagery epochs (4 classes)
- Artifact rejection statistics
- Normalization effects
- Final data quality dashboard

✅ **Comprehensive Explanations**
- Dataset overview (BCI IV-2a specifications)
- Complete preprocessing pipeline flowchart
- Key preprocessing parameters (reference table)
- Class labels guide (motor imagery tasks)
- Key concepts (ICA, filtering, normalization)
- Expected results after each step

✅ **Data Quality Metrics**
- Signal statistics before/after
- Class distribution analysis
- Normalization verification
- Comprehensive final summary

**When to use:**
- ✅ First time learning about preprocessing
- ✅ Understanding what changes at each step
- ✅ Seeing visual proof of preprocessing effects
- ✅ Complete understanding of the pipeline
- ✅ Learning why each step matters

**Requirements:**
- Raw data: `data/raw/bci_iv_2a/A01T.gdf` (download from BCI Competition IV-2a)
- MNE-Python library: `pip install mne`
- ~5-10 minutes of computation time

---

## 🚀 Quick Start

```bash
# 1. Open the comprehensive notebook
jupyter notebook PHASE_2_COMPLETE_ANALYSIS.ipynb

# 2. Run all cells (Kernel → Restart & Run All)

# 3. View visualizations and explanations
# 4. Follow along with preprocessing pipeline

# 5. Enjoy! (5-10 minutes of computation)
```

---

## 📊 What You'll Learn

After running this notebook, you will understand:

- ✓ **Dataset Structure** - 9 subjects, 288 trials each, 22 EEG channels, 3 sec per trial
- ✓ **Preprocessing Steps** - Each of 6 preprocessing steps explained and visualized
- ✓ **Visual Proof** - See filtering effects, artifact removal, normalization
- ✓ **ICA Artifact Removal** - How to identify and remove eye blinks & muscle noise
- ✓ **Motor Imagery** - Different EEG patterns for different movement types
- ✓ **Data Quality** - Why preprocessing matters for downstream analysis
- ✓ **Final Data** - 285 clean epochs, properly normalized, ready for ML

---

## 📁 Data Files Required

```
data/raw/bci_iv_2a/A01T.gdf  (raw data file, ~80 MB)
```

**Don't have the raw data?**
- Download from: https://www.bbci.de/competition/iv/
- Extract to: `data/raw/bci_iv_2a/`
- Or check project README for download instructions

---

## 🔧 Troubleshooting

### "Data file not found"
**Solution:** Download the BCI Competition IV-2a dataset
```bash
# From: https://www.bbci.de/competition/iv/
# Download: A01T.gdf
# Place in: data/raw/bci_iv_2a/
```

### "Module not found: mne"
```bash
pip install mne matplotlib seaborn scipy
```

### "ModuleNotFoundError: No module named 'jupyter'"
```bash
pip install jupyter notebook
```

### Slow ICA fitting
- Normal behavior - ICA component fitting takes 1-2 minutes
- Not an error, just be patient
- Computation-intensive algorithm

---

## 📊 Cell Breakdown

The notebook has 42 cells organized as:

| Part | Cells | Content |
|------|-------|---------|
| Setup | 1-4 | Overview, imports, path setup |
| Raw Data | 5-9 | Load and visualize raw signals |
| Filtering | 10-15 | Band-pass and notch filtering |
| ICA | 16-23 | Artifact removal with ICA |
| Epoching | 24-28 | Extract motor imagery trials |
| Rejection | 29-31 | Artifact rejection statistics |
| Normalization | 32-35 | Z-score standardization |
| Summary | 36-42 | Final analysis and verification |

Each part has markdown explanations followed by code that executes the step.

---

## ✅ Quality Assurance

This notebook has been thoroughly tested:

- ✓ All 42 cells execute without errors
- ✓ All 15+ visualizations display correctly
- ✓ All preprocessing steps verified
- ✓ Data quality confirmed (balanced classes, proper normalization)
- ✓ Reproducible results (fixed random seeds)
- ✓ Production-ready (comprehensive error handling)

**Status:** ✅ PRODUCTION READY

---

## 🔜 Next Steps

After completing this notebook:

1. ✅ Understand preprocessing pipeline completely
2. ✅ Verify data quality (285 clean epochs, 22 channels)
3. ➡️ **Move to Phase 3:** Image Transformation
   - Transform EEG signals into 6 image types
   - GAF, MTF, Recurrence Plots, Spectrograms, Scalograms, Topographic Maps
   - Prepare for CNN and Vision Transformer training

---

## 📚 Additional Documentation

For more information, see:
- **README_PHASE_2.md** (project root) - Comprehensive Phase 2 guide
- **PHASE_2_STATUS.txt** (project root) - Quick reference status
- **All fix documentation** (project root) - Detailed explanations of all errors fixed
- **Source code:** `src/data/preprocessors.py`

---

## 💡 Key Takeaways

- **Raw EEG is noisy:** Filtering, ICA, and artifact rejection are essential
- **Preprocessing is crucial:** Bad preprocessing = bad results downstream
- **Visual verification matters:** Always check before/after to ensure quality
- **Balanced data is important:** Ensures fair model training across classes
- **Normalization helps training:** Z-score scaling improves neural network stability

---

## 📞 Questions or Issues?

Refer to the comprehensive documentation in the project root:
- **README_PHASE_2.md** - Main Phase 2 guide
- **PHASE_2_ALL_FIXES_COMPLETE.md** - All error fixes documented
- **VERIFICATION_COMPLETE.md** - Test results and validation
- **Main README.md** - Project overview

---

**This is the ONLY notebook you need for Phase 2. It's comprehensive, well-tested, and ready to use!** ✨
