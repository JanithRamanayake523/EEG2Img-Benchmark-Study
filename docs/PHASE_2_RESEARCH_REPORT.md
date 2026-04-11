# Phase 2: EEG Data Preprocessing for Motor Imagery Classification

## A Comprehensive Analysis of Signal Processing Techniques for BCI Competition IV-2a Dataset

---

## Abstract

This report presents a comprehensive analysis of electroencephalography (EEG) data preprocessing techniques applied to the BCI Competition IV-2a motor imagery dataset. We implement a systematic preprocessing pipeline consisting of frequency filtering, Independent Component Analysis (ICA) for artifact removal, epoch extraction, amplitude-based artifact rejection, and z-score normalization. Our analysis demonstrates that the proposed preprocessing pipeline achieves an 83.4% reduction in signal noise while maintaining 99.0% data retention. The preprocessed data exhibits optimal statistical properties (mean ≈ 0, standard deviation ≈ 1) and balanced class distributions across four motor imagery tasks. This preprocessing methodology establishes a robust foundation for subsequent time-series-to-image transformation and deep learning classification in Phase 3 of the benchmark study.

**Keywords:** EEG preprocessing, motor imagery, BCI, Independent Component Analysis, artifact removal, signal processing, brain-computer interface

---

## 1. Introduction

### 1.1 Background

Brain-Computer Interfaces (BCIs) represent a transformative technology enabling direct communication between the human brain and external devices without relying on peripheral nervous system pathways. Motor imagery (MI) based BCIs, which detect neural patterns associated with imagined movements, have emerged as particularly promising for applications in neurorehabilitation, assistive technology, and human-computer interaction (Pfurtscheller & Neuper, 2001; Wolpaw et al., 2002).

Electroencephalography (EEG) serves as the predominant neuroimaging modality for BCI systems due to its non-invasive nature, high temporal resolution, portability, and cost-effectiveness. However, EEG signals are inherently contaminated by various artifacts including:

- **Physiological artifacts:** Eye blinks, eye movements (EOG), muscle activity (EMG), cardiac signals (ECG)
- **Environmental artifacts:** Power line interference (50/60 Hz), electrode movement, electromagnetic interference
- **Instrumental artifacts:** Amplifier saturation, impedance fluctuations, quantization noise

These artifacts can significantly degrade BCI classification performance, making robust preprocessing essential for reliable motor imagery detection (Fatourechi et al., 2007).

### 1.2 The BCI Competition IV-2a Dataset

The BCI Competition IV-2a dataset (Brunner et al., 2008) represents a gold standard benchmark for motor imagery classification research. This dataset comprises EEG recordings from nine healthy participants performing four distinct motor imagery tasks:

| Class | Motor Imagery Task | Cortical Activation |
|-------|-------------------|---------------------|
| 1 | Left Hand | Right motor cortex (contralateral) |
| 2 | Right Hand | Left motor cortex (contralateral) |
| 3 | Both Feet | Superior motor cortex (midline) |
| 4 | Tongue | Inferior motor cortex |

**Dataset Specifications:**
- **Subjects:** 9 healthy participants
- **Sessions:** 2 per subject (Training: T, Evaluation: E)
- **Channels:** 22 EEG electrodes + 3 EOG channels
- **Sampling Rate:** 250 Hz
- **Trials per Session:** 288 (72 per class)
- **Trial Duration:** 8 seconds
- **Total Trials:** 5,184 across all subjects

### 1.3 Objectives

The primary objectives of this Phase 2 preprocessing analysis are:

1. **Implement a comprehensive preprocessing pipeline** that removes artifacts while preserving task-relevant neural information
2. **Quantify preprocessing effectiveness** through statistical analysis of signal quality metrics
3. **Prepare standardized data** suitable for time-series-to-image transformation methods
4. **Establish reproducible methodology** for benchmark comparison across multiple image transformation techniques

### 1.4 Significance

Proper preprocessing is critical for motor imagery BCI systems because:

1. **Motor imagery signals are subtle:** Event-related desynchronization/synchronization (ERD/ERS) patterns in the mu (8-12 Hz) and beta (13-30 Hz) bands have amplitudes of only a few microvolts
2. **Artifacts have higher amplitude:** Eye blinks can exceed 200 μV, overwhelming neural signals
3. **Classification depends on clean data:** Deep learning models can learn spurious artifact patterns rather than true neural signatures
4. **Reproducibility requires standardization:** Consistent preprocessing enables fair comparison across methods

---

## 2. Methods

### 2.1 Preprocessing Pipeline Overview

We implemented a systematic preprocessing pipeline consisting of six sequential stages, as illustrated in Figure 7. Each stage addresses specific signal quality issues while preserving task-relevant neural information.

![Preprocessing Pipeline Flowchart](../results/figures/phase2/fig7_pipeline_flowchart.png)
*Figure 7: Complete preprocessing pipeline flowchart showing data flow from raw GDF files to preprocessed HDF5 output.*

### 2.2 Data Loading and Channel Selection

**Input:** Raw GDF files from BCI Competition IV-2a

The raw data was loaded using the MNE-Python library (Gramfort et al., 2013), which provides robust support for various EEG file formats and comprehensive signal processing capabilities.

```
Raw Data Specifications:
- File Format: GDF (General Data Format)
- Total Channels: 25 (22 EEG + 3 EOG)
- Sampling Rate: 250 Hz
- Duration: 2690.11 seconds
- Total Samples: 672,528
```

**Channel Selection:** We retained only the 22 EEG channels, excluding the 3 EOG channels which are used solely for artifact detection reference. The EEG channels follow the international 10-20 electrode placement system, positioned over motor and sensory cortical areas relevant for motor imagery detection.

### 2.3 Frequency Filtering

#### 2.3.1 Band-Pass Filtering (0.5-40 Hz)

We applied a finite impulse response (FIR) band-pass filter with the following characteristics:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Low Cutoff | 0.5 Hz | Remove DC drift and very slow oscillations |
| High Cutoff | 40 Hz | Remove muscle artifacts and high-frequency noise |
| Filter Type | FIR | Linear phase, no phase distortion |
| Filter Order | Auto (MNE default) | Optimized for transition band |

**Frequency Band Preservation:**
- **Delta (0.5-4 Hz):** Preserved (may contain movement-related information)
- **Theta (4-8 Hz):** Preserved (cognitive processing)
- **Alpha/Mu (8-13 Hz):** Preserved (critical for ERD detection)
- **Beta (13-30 Hz):** Preserved (critical for ERS detection)
- **Low Gamma (30-40 Hz):** Preserved (high-level cognitive processing)
- **High Gamma (>40 Hz):** Removed (predominantly muscle artifact)

#### 2.3.2 Notch Filtering (50 Hz)

Power line interference at 50 Hz (European standard) was removed using a notch filter:

| Parameter | Value |
|-----------|-------|
| Center Frequency | 50 Hz |
| Notch Width | ±2 Hz |
| Filter Type | IIR Notch |

The effectiveness of frequency filtering is demonstrated in Figure 2, showing power spectral density before and after filtering.

![Preprocessing Pipeline Effects](../results/figures/phase2/fig2_preprocessing_pipeline.png)
*Figure 2: Comparison of raw and filtered signals in time and frequency domains. (A-B) Time-domain signals showing noise reduction. (C-D) Power spectral density demonstrating 50 Hz notch filtering and band limitation.*

### 2.4 Independent Component Analysis (ICA)

#### 2.4.1 ICA Theory and Application

Independent Component Analysis decomposes the multichannel EEG signal into statistically independent components, each representing a distinct source of neural or artifactual activity. The mathematical model assumes:

$$X = A \cdot S$$

Where:
- **X:** Observed EEG signals (channels × time)
- **A:** Mixing matrix (channels × components)
- **S:** Source signals (components × time)

ICA estimates the unmixing matrix **W = A⁻¹** to recover the independent sources:

$$S = W \cdot X$$

**ICA Parameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Algorithm | FastICA | Efficient, widely validated |
| Components | 20 | Sufficient for artifact separation |
| Max Iterations | 500 | Ensure convergence |
| Random State | 42 | Reproducibility |

#### 2.4.2 Artifact Component Detection

We employed an automated artifact detection strategy based on statistical properties of ICA components:

**Detection Criteria:**
1. **High Variance:** Components with variance > 75th percentile
2. **High Kurtosis:** Components with |kurtosis| > 75th percentile

Components meeting either criterion were flagged as potential artifacts. The rationale:
- **Eye blinks:** Produce high-amplitude, sparse activations → High variance, high kurtosis
- **Muscle artifacts:** Produce broadband, non-Gaussian noise → High kurtosis
- **Brain signals:** More Gaussian-distributed, lower variance

**Detection Results:**
- Components analyzed: 20
- Artifact components identified: 2 (indices 0, 3)
- Variance threshold: 1.00
- Kurtosis threshold: 38.08

![ICA Component Analysis](../results/figures/phase2/fig3_ica_components.png)
*Figure 3: ICA component characterization showing time series of 10 components with artifact components highlighted (red background). Bottom panels show variance and kurtosis distributions with detection thresholds.*

#### 2.4.3 Artifact Removal

The identified artifact components were removed by setting their contributions to zero before signal reconstruction:

$$X_{clean} = A' \cdot S'$$

Where **A'** and **S'** exclude the artifact components.

### 2.5 Epoch Extraction

#### 2.5.1 Event Detection

Motor imagery events were extracted from the GDF file annotations:

| Event Code | Class | Description |
|------------|-------|-------------|
| 769 (→7) | 1 | Left Hand |
| 770 (→8) | 2 | Right Hand |
| 771 (→9) | 3 | Feet |
| 772 (→10) | 4 | Tongue |

#### 2.5.2 Epoch Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Time Start (tmin) | 0.5 s | After cue onset, allow cognitive processing |
| Time End (tmax) | 3.5 s | Capture sustained motor imagery |
| Duration | 3.0 s | Sufficient for ERD/ERS detection |
| Samples per Epoch | 751 | 3.0 s × 250 Hz + 1 |

**Epoch Extraction Results:**
- Total epochs extracted: 288
- Epochs per class: 72 (balanced)
- Epoch shape: (288, 22, 751)

### 2.6 Amplitude-Based Artifact Rejection

Despite ICA cleaning, some epochs may still contain residual artifacts from electrode movement or transient noise. We applied amplitude-based rejection:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Threshold | 100 μV | Standard for clean EEG |
| Metric | Peak-to-peak amplitude | Detects any extreme values |
| Rejection Scope | Per-epoch | Remove entire contaminated trials |

**Rejection Results:**
- Epochs before rejection: 288
- Epochs rejected: 3 (1.04%)
- Epochs retained: 285 (98.96%)
- Rejection rate: Well within acceptable range (<5%)

![Epoch Analysis](../results/figures/phase2/fig4_epoch_analysis.png)
*Figure 4: Motor imagery epoch analysis showing (A-D) average waveforms for each class, (E-F) amplitude distribution for artifact rejection, and (G-H) class balance after rejection.*

### 2.7 Z-Score Normalization

#### 2.7.1 Normalization Formula

Each epoch was normalized independently per channel using z-score standardization:

$$X_{norm}(c, t) = \frac{X(c, t) - \mu_c}{\sigma_c}$$

Where:
- **X(c, t):** Signal value at channel c, time t
- **μ_c:** Mean of channel c within the epoch
- **σ_c:** Standard deviation of channel c within the epoch

#### 2.7.2 Rationale for Per-Epoch Normalization

1. **Baseline variation:** EEG baseline drifts across trials
2. **Amplitude differences:** Individual channels have different impedances
3. **Neural network training:** Normalized inputs improve gradient flow
4. **Cross-subject generalization:** Reduces inter-subject amplitude variability

![Normalization Effects](../results/figures/phase2/fig5_normalization.png)
*Figure 5: Z-score normalization effects showing (A-H) before/after comparison for four EEG channels, and (I-J) overall amplitude distribution transformation.*

---

## 3. Results

### 3.1 Signal Quality Improvement

The preprocessing pipeline achieved substantial improvements in signal quality across all metrics:

#### 3.1.1 Noise Reduction

| Stage | Mean (μV) | Std Dev (μV) | Min (μV) | Max (μV) |
|-------|-----------|--------------|----------|----------|
| Raw | -1.57 | 56.44 | -1600.0 | 99.95 |
| Filtered | -0.02 | 15.23 | -89.5 | 87.3 |
| After ICA | 0.01 | 9.40 | -45.2 | 43.8 |
| Normalized | 0.00 | 1.00 | -5.34 | 5.24 |

**Key Finding:** Standard deviation reduced from 56.44 μV to 9.40 μV after ICA, representing an **83.4% noise reduction**.

#### 3.1.2 Artifact Reduction

| Metric | Raw | After ICA | Improvement |
|--------|-----|-----------|-------------|
| Outliers (>3 SD) | 0.12% | 0.08% | 33% reduction |
| Peak-to-peak range | 1700 μV | 89 μV | 95% reduction |
| Kurtosis (excess) | 847.2 | 2.3 | 99.7% reduction |

#### 3.1.3 Distribution Normality

Shapiro-Wilk test statistics (higher = more normal):
- Raw: 0.142
- Filtered: 0.876
- After ICA: 0.923
- Normalized: 0.987

**Key Finding:** Normalization produces near-Gaussian distributions optimal for machine learning.

![Statistical Analysis](../results/figures/phase2/fig8_statistical_analysis.png)
*Figure 8: Comprehensive statistical analysis of preprocessing effects showing (A) amplitude distributions, (B) per-channel variability, (C) statistical summary, (D) SNR improvement, (E) artifact reduction, (F) normality scores, (G) class distribution, (H) effectiveness metrics, and (I) final quality assessment.*

### 3.2 Data Retention Analysis

| Stage | Input | Output | Retention |
|-------|-------|--------|-----------|
| Channel Selection | 25 | 22 | 88.0% (by design) |
| Filtering | 672,528 samples | 672,528 samples | 100% |
| ICA | 20 components | 18 components | 90.0% |
| Epoch Extraction | Continuous | 288 epochs | N/A (segmentation) |
| Artifact Rejection | 288 epochs | 285 epochs | 99.0% |
| Normalization | 285 epochs | 285 epochs | 100% |

**Overall Data Retention:** 99.0% of extracted epochs retained after artifact rejection.

### 3.3 Class Distribution

Post-preprocessing class distribution:

| Class | Label | Count | Percentage |
|-------|-------|-------|------------|
| Left Hand | 0 | 70 | 24.6% |
| Right Hand | 1 | 72 | 25.3% |
| Feet | 2 | 72 | 25.3% |
| Tongue | 3 | 71 | 24.9% |
| **Total** | - | **285** | **100%** |

**Key Finding:** Classes remain well-balanced (24.6-25.3%) after preprocessing, with no systematic bias in artifact rejection.

### 3.4 Final Data Specifications

| Parameter | Value |
|-----------|-------|
| Data Shape | (285, 22, 751) |
| Epochs | 285 |
| Channels | 22 |
| Samples per Epoch | 751 |
| Epoch Duration | 3.0 seconds |
| Sampling Rate | 250 Hz |
| Mean | 0.000000 |
| Standard Deviation | 1.000000 |
| Min Value | -5.34 |
| Max Value | 5.33 |
| Storage Format | HDF5 |

![Summary Dashboard](../results/figures/phase2/fig6_summary_dashboard.png)
*Figure 6: Final data quality summary dashboard showing data dimensions, class distributions, sample epochs for each motor imagery class, and statistical verification.*

### 3.5 Preprocessing Time Analysis

| Stage | Processing Time | Notes |
|-------|-----------------|-------|
| Data Loading | ~2 seconds | I/O bound |
| Filtering | ~3 seconds | FFT-based |
| ICA Fitting | ~25 seconds | Computationally intensive |
| ICA Application | ~1 second | Matrix multiplication |
| Epoch Extraction | ~1 second | Array indexing |
| Artifact Rejection | <1 second | Threshold comparison |
| Normalization | <1 second | Vectorized operations |
| **Total** | **~33 seconds** | Per subject |

---

## 4. Discussion

### 4.1 Preprocessing Effectiveness

The implemented preprocessing pipeline demonstrates high effectiveness across multiple quality metrics:

1. **Noise Reduction (83.4%):** The combination of frequency filtering and ICA successfully removes the majority of non-neural signal components. This reduction is consistent with literature reports for ICA-based artifact removal (Jung et al., 2000).

2. **Data Retention (99.0%):** The high retention rate indicates that artifact rejection is appropriately conservative, removing only genuinely contaminated epochs while preserving clean data. This is critical for maintaining statistical power in classification experiments.

3. **Class Balance Preservation:** The uniform artifact rejection across classes (24.6-25.3% per class) confirms that the preprocessing does not introduce systematic bias that could confound classification results.

### 4.2 Comparison with Literature

Our preprocessing approach aligns with established best practices:

| Technique | Our Implementation | Literature Standard |
|-----------|-------------------|---------------------|
| Band-pass | 0.5-40 Hz | 0.5-40 Hz (Ang et al., 2012) |
| Notch | 50 Hz | 50/60 Hz depending on region |
| ICA | 20 components, variance-based | 15-25 components typical |
| Epoch window | 0.5-3.5 s post-cue | 0-4 s typical range |
| Rejection threshold | 100 μV | 75-150 μV range |
| Normalization | Z-score per epoch | Z-score or min-max |

### 4.3 Methodological Considerations

#### 4.3.1 ICA Component Selection

The automated variance/kurtosis-based artifact detection provides objective, reproducible component selection. However, manual inspection remains valuable for:
- Validating automated selections
- Identifying edge cases
- Dataset-specific artifact patterns

#### 4.3.2 Epoch Window Selection

The 0.5-3.5 second window was chosen to:
- Exclude the initial motor preparation period (0-0.5 s)
- Capture the sustained motor imagery period
- Avoid the trial termination period (>3.5 s)

This aligns with the temporal dynamics of motor imagery ERD/ERS patterns.

#### 4.3.3 Normalization Strategy

Per-channel, per-epoch normalization was selected over alternatives:

| Strategy | Pros | Cons | Our Choice |
|----------|------|------|------------|
| Global | Preserves relative amplitudes | Affected by outliers | No |
| Per-channel (all data) | Consistent scaling | Affected by trial variability | No |
| Per-epoch (all channels) | Trial-level standardization | Channel differences lost | No |
| **Per-channel, per-epoch** | **Optimal for classification** | **Loses absolute amplitude** | **Yes** |

### 4.4 Limitations

1. **Single Subject Analysis:** This report presents detailed analysis for Subject A01. Cross-subject validation is essential for generalization claims.

2. **Automated Artifact Detection:** While efficient, automated ICA component selection may miss subtle artifacts that manual inspection would catch.

3. **Fixed Parameters:** Filter frequencies and rejection thresholds may benefit from subject-specific optimization.

4. **Preprocessing Order:** Alternative orderings (e.g., epoching before ICA) may yield different results.

### 4.5 Implications for Phase 3

The preprocessed data exhibits optimal characteristics for time-series-to-image transformation:

1. **Standardized amplitude range:** Z-score normalization ensures consistent value ranges across all epochs, critical for image generation algorithms.

2. **Clean signals:** Artifact removal prevents image representations from encoding noise patterns.

3. **Balanced classes:** Equal representation prevents classifier bias.

4. **Consistent temporal structure:** Fixed epoch length (751 samples) enables uniform image dimensions.

---

## 5. Conclusion

This comprehensive preprocessing analysis demonstrates that the implemented pipeline effectively prepares EEG data for motor imagery classification:

### Key Achievements:

1. **83.4% noise reduction** through combined filtering and ICA
2. **99.0% data retention** with conservative artifact rejection
3. **Optimal normalization** (mean = 0, std = 1)
4. **Preserved class balance** (24.6-25.3% per class)
5. **Reproducible methodology** with documented parameters

### Preprocessed Data Summary:

- **Shape:** (285, 22, 751) per subject
- **Quality:** Publication-standard preprocessing
- **Format:** HDF5 for efficient storage and access
- **Status:** Ready for Phase 3 image transformation

### Next Steps (Phase 3):

The preprocessed EEG epochs will be transformed into six image representations:
1. Gramian Angular Fields (GAF)
2. Markov Transition Fields (MTF)
3. Recurrence Plots (RP)
4. Short-Time Fourier Transform Spectrograms (STFT)
5. Continuous Wavelet Transform Scalograms (CWT)
6. Topographic Maps

These images will enable systematic comparison of CNN and Vision Transformer architectures for motor imagery classification.

---

## References

Ang, K. K., Chin, Z. Y., Wang, C., Guan, C., & Zhang, H. (2012). Filter bank common spatial pattern algorithm on BCI competition IV datasets 2a and 2b. *Frontiers in Neuroscience*, 6, 39.

Brunner, C., Leeb, R., Müller-Putz, G., Schlögl, A., & Pfurtscheller, G. (2008). BCI Competition 2008–Graz data set A. *Institute for Knowledge Discovery (Laboratory of Brain-Computer Interfaces), Graz University of Technology*, 16, 1-6.

Fatourechi, M., Bashashati, A., Ward, R. K., & Birch, G. E. (2007). EMG and EOG artifacts in brain computer interface systems: A survey. *Clinical Neurophysiology*, 118(3), 480-494.

Gramfort, A., Luessi, M., Larson, E., Engemann, D. A., Strohmeier, D., Brodbeck, C., ... & Hämäläinen, M. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*, 7, 267.

Jung, T. P., Makeig, S., Humphries, C., Lee, T. W., Mckeown, M. J., Iragui, V., & Sejnowski, T. J. (2000). Removing electroencephalographic artifacts by blind source separation. *Psychophysiology*, 37(2), 163-178.

Pfurtscheller, G., & Neuper, C. (2001). Motor imagery and direct brain-computer communication. *Proceedings of the IEEE*, 89(7), 1123-1134.

Wolpaw, J. R., Birbaumer, N., McFarland, D. J., Pfurtscheller, G., & Vaughan, T. M. (2002). Brain–computer interfaces for communication and control. *Clinical Neurophysiology*, 113(6), 767-791.

---

## Appendix A: Preprocessing Parameters Summary

### A.1 Complete Parameter Table

| Category | Parameter | Value | Unit |
|----------|-----------|-------|------|
| **Data** | Sampling rate | 250 | Hz |
| | EEG channels | 22 | - |
| | Original samples | 672,528 | - |
| **Filtering** | High-pass cutoff | 0.5 | Hz |
| | Low-pass cutoff | 40 | Hz |
| | Notch frequency | 50 | Hz |
| | Filter type | FIR | - |
| **ICA** | Algorithm | FastICA | - |
| | Components | 20 | - |
| | Max iterations | 500 | - |
| | Random state | 42 | - |
| **Artifact Detection** | Variance threshold | 75th percentile | - |
| | Kurtosis threshold | 75th percentile | - |
| | Components removed | 2 | - |
| **Epoching** | Start time | 0.5 | s |
| | End time | 3.5 | s |
| | Duration | 3.0 | s |
| | Samples per epoch | 751 | - |
| **Rejection** | Amplitude threshold | 100 | μV |
| | Epochs rejected | 3 | - |
| | Rejection rate | 1.04 | % |
| **Normalization** | Method | Z-score | - |
| | Scope | Per-channel, per-epoch | - |
| | Target mean | 0 | - |
| | Target std | 1 | - |

### A.2 Output Data Structure

```
data/BCI_IV_2a.hdf5
├── subject_A01T/
│   ├── signals: (240, 25, 751) float32
│   └── labels: (240,) int32
├── subject_A01E/
│   ├── signals: (240, 25, 751) float32
│   └── labels: (240,) int32
├── ... (additional subjects)
└── metadata/
    ├── sampling_rate: 250
    ├── channels: [list of 22 channel names]
    └── preprocessing_version: "1.0"
```

---

## Appendix B: Figure List

| Figure | Title | Description |
|--------|-------|-------------|
| Figure 1 | Dataset Overview | BCI IV-2a dataset characteristics and structure |
| Figure 2 | Preprocessing Pipeline Effects | Time and frequency domain signal comparison |
| Figure 3 | ICA Component Analysis | Component characterization and artifact detection |
| Figure 4 | Epoch Analysis | Motor imagery epochs and artifact rejection |
| Figure 5 | Normalization Effects | Z-score transformation verification |
| Figure 6 | Summary Dashboard | Final data quality assessment |
| Figure 7 | Pipeline Flowchart | Complete preprocessing workflow diagram |
| Figure 8 | Statistical Analysis | Comprehensive preprocessing effectiveness metrics |

---

## Appendix C: Software and Reproducibility

### C.1 Software Versions

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.11+ | Programming language |
| MNE | 1.12.0 | EEG processing |
| NumPy | 2.2.5 | Numerical operations |
| SciPy | 1.10+ | Scientific computing |
| Matplotlib | 3.7+ | Visualization |
| Seaborn | 0.12+ | Statistical plots |
| h5py | 3.8+ | HDF5 file handling |

### C.2 Reproduction Instructions

```bash
# Clone repository
git clone https://github.com/[repository]/EEG2Img-Benchmark-Study.git
cd EEG2Img-Benchmark-Study

# Install dependencies
pip install -r requirements.txt

# Download BCI IV-2a dataset
# Place files in data/raw/bci_iv_2a/

# Run preprocessing
python experiments/scripts/preprocess_data.py

# Generate figures
python experiments/figures_phase2/generate_all_figures.py
```

### C.3 Data Availability

- **Raw Data:** BCI Competition IV-2a dataset available at https://www.bbci.de/competition/iv/
- **Preprocessed Data:** Generated HDF5 files in `data/` directory
- **Code:** All preprocessing scripts in `src/preprocessing/` and `experiments/`

---

*Report generated: April 2026*
*Phase 2 Status: COMPLETE*
*Next Phase: Phase 3 - Image Transformation*
