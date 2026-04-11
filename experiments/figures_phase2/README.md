# Phase 2 Publication Figure Generation Scripts

This directory contains scripts to generate publication-ready figures for **Phase 2: Data Preprocessing** of the EEG Time-Series-to-Image Benchmark Study.

---

## 📊 Available Figures

| Figure | Description | File |
|--------|-------------|------|
| **Figure 1** | Dataset Overview and Characteristics | `fig1_dataset_overview.py` |
| **Figure 2** | Preprocessing Pipeline Effects (Filtering + ICA) | `fig2_preprocessing_pipeline.py` |
| **Figure 3** | ICA Component Analysis | `fig3_ica_components.py` |
| **Figure 4** | Epoch Analysis and Artifact Rejection | `fig4_epoch_analysis.py` |
| **Figure 5** | Z-Score Normalization Effects | `fig5_normalization.py` |
| **Figure 6** | Summary Dashboard | `fig6_summary_dashboard.py` |

---

## 🚀 Quick Start

### Generate All Figures (Recommended)

```bash
# From project root
python experiments/figures_phase2/generate_all_figures.py
```

This will:
- Generate all 6 figures sequentially
- Save outputs to `results/figures/phase2/`
- Display progress and timing information
- Create both PNG (300 DPI) and PDF (vector) formats

**Expected runtime:** 5-10 minutes (depending on hardware)

---

### Generate Individual Figures

If you only need specific figures:

```bash
# Figure 1: Dataset Overview
python experiments/figures_phase2/fig1_dataset_overview.py

# Figure 2: Preprocessing Pipeline
python experiments/figures_phase2/fig2_preprocessing_pipeline.py

# Figure 3: ICA Components
python experiments/figures_phase2/fig3_ica_components.py

# Figure 4: Epoch Analysis
python experiments/figures_phase2/fig4_epoch_analysis.py

# Figure 5: Normalization
python experiments/figures_phase2/fig5_normalization.py

# Figure 6: Summary Dashboard
python experiments/figures_phase2/fig6_summary_dashboard.py
```

---

## 📋 Requirements

### Data Files

1. **Raw EEG data:**
   - Location: `data/raw/bci_iv_2a/A01T.gdf`
   - Required for: Figures 1-5
   - Download from: https://www.bbci.de/competition/iv/

2. **Preprocessed data (combined):**
   - Location: `data/BCI_IV_2a.hdf5`
   - Required for: Figure 6
   - Generate with: `python experiments/scripts/combine_preprocessed_data.py`

### Python Packages

All required packages are in `requirements.txt`:

```bash
pip install numpy pandas matplotlib seaborn scipy mne h5py
```

**Key dependencies:**
- `mne >= 1.5.0` - EEG processing
- `matplotlib >= 3.7.0` - Plotting
- `seaborn >= 0.12.0` - Statistical visualizations
- `scipy >= 1.10.0` - Scientific computing

---

## 📁 Output

All figures are saved to: `results/figures/phase2/`

### Output Formats

Each figure is saved in **two formats**:

1. **PNG** (300 DPI)
   - High-resolution raster image
   - Suitable for manuscripts, presentations
   - Files: `fig1_dataset_overview.png`, etc.

2. **PDF** (Vector Graphics)
   - Scalable vector format
   - Recommended for publication
   - Files: `fig1_dataset_overview.pdf`, etc.

---

## 📊 Figure Descriptions

### Figure 1: Dataset Overview
**Purpose:** Introduce the BCI Competition IV-2a dataset characteristics

**Contents:**
- Dataset specifications table
- Motor imagery trial timeline
- EEG channel layout (10-20 system)
- Trial distribution by class
- Representative raw EEG signals

---

### Figure 2: Preprocessing Pipeline
**Purpose:** Show signal transformation effects through preprocessing steps

**Contents:**
- Raw vs Filtered signals (time domain)
- Power Spectral Density comparison (frequency domain)
- ICA before/after comparison
- Demonstrates filtering and artifact removal effectiveness

---

### Figure 3: ICA Components
**Purpose:** Detailed analysis of Independent Component Analysis

**Contents:**
- Time series of first 10 ICA components
- Component variance distribution
- Component kurtosis analysis
- PCA explained variance
- Visual identification of artifact components (eye blinks, muscle noise)

---

### Figure 4: Epoch Analysis
**Purpose:** Motor imagery epoch extraction and class-specific patterns

**Contents:**
- Average epochs for each of 4 motor imagery classes
- Artifact rejection amplitude distribution
- Box plot of epoch amplitudes
- Class distribution after artifact rejection
- Class balance verification

---

### Figure 5: Normalization
**Purpose:** Z-score normalization effects and statistical verification

**Contents:**
- Before/after comparison for 4 EEG channels
- Overall distribution histograms
- Statistical summary table (mean, std, min, max)
- Visual proof of mean ≈ 0, std ≈ 1

---

### Figure 6: Summary Dashboard
**Purpose:** Comprehensive final data quality assessment

**Contents:**
- Final data dimensions
- Class distribution (bar chart + pie chart)
- Data distribution verification
- Sample epochs for all 4 classes
- Statistical summary table
- Per-channel statistics
- Preprocessing pipeline checklist

---

## 🎨 Publication Quality Settings

All figures are configured for **publication standards**:

- **DPI:** 300 (high resolution)
- **Font:** Times New Roman (serif, professional)
- **Figure size:** 14-18 inches wide (publication ready)
- **Style:** Seaborn paper theme (clean, minimal)
- **Colors:** Color-blind friendly palettes
- **Labels:** Clear axis labels, legends, and titles
- **Grid:** Subtle gridlines for readability

---

## 🔧 Customization

### Modify Figure Appearance

Each script has configurable parameters at the top:

```python
# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.3)  # Adjust font size
plt.rcParams['figure.dpi'] = 300  # Change DPI
plt.rcParams['font.family'] = 'serif'  # Change font
```

### Change Output Directory

Modify the `output_dir` variable in each script:

```python
output_dir = Path('results/figures/phase2')  # Change this
```

### Adjust Time Windows

For time-series plots, modify `time_window` parameters:

```python
time_window = [50, 55]  # 5-second window starting at 50s
```

---

## ⚠️ Troubleshooting

### "Data file not found"
**Solution:** Ensure raw data is downloaded
```bash
# Check if file exists
ls data/raw/bci_iv_2a/A01T.gdf

# If missing, download from:
# https://www.bbci.de/competition/iv/
```

### "Module not found: mne"
**Solution:** Install required packages
```bash
pip install mne matplotlib seaborn scipy h5py
```

### "Figure 6 fails - HDF5 file not found"
**Solution:** Generate combined preprocessed data
```bash
python experiments/scripts/combine_preprocessed_data.py
```

### Slow ICA fitting
- **Expected behavior** - ICA fitting takes 1-2 minutes
- This is normal for 20-component ICA
- Be patient, it's not frozen

### Memory errors
- Ensure at least 8 GB RAM available
- Close other applications
- Process one figure at a time instead of running `generate_all_figures.py`

---

## 📝 Citation

If you use these figures in publications, please cite:

```bibtex
@article{eeg2img_benchmark_2026,
  title={Time-Series-to-Image Transformations for EEG Classification: A Systematic Benchmark Study},
  author={[Your Name]},
  journal={[Journal Name]},
  year={2026}
}
```

---

## 📧 Support

For questions or issues:
- Check the main project README: `README.md`
- Review Phase 2 documentation: `notebooks/phase_2_data_preprocessing/README.md`
- Open an issue on GitHub (if applicable)

---

## ✅ Verification Checklist

Before using figures in publication:

- [ ] All 6 figures generated successfully
- [ ] Figures are 300 DPI (check file properties)
- [ ] PDF versions are vector graphics (zoom in without pixelation)
- [ ] Axis labels are readable
- [ ] Legends are clear and unambiguous
- [ ] Color schemes are accessible (color-blind friendly)
- [ ] Figure captions written (not auto-generated by scripts)
- [ ] Statistical results verified against notebook outputs
- [ ] Data dimensions match expected values

---

**Last Updated:** April 2026
**Version:** 1.0.0
**Status:** ✅ Production Ready
