# EEG Time-Series-to-Image Benchmark Study - Implementation Plan

**Project Repository:** EEG2Img-Benchmark-Study
**Timeline:** 6 months (April 2026 - October 2026)
**Status:** Implementation Phase

---

## 🎯 Executive Summary

This plan outlines the complete implementation of a systematic benchmark study comparing multiple time-series-to-image (T2I) transformations for EEG classification across diverse BCI paradigms. The project will evaluate GAF, MTF, Recurrence Plots, Spectrograms, and Scalograms using CNN and Vision Transformer models against raw-signal baselines.

**Current State:** Greenfield project with comprehensive research specification document but no existing code.

**⚠️ CRITICAL RULE:** This implementation MUST strictly follow this document phase by phase. Do not skip phases or implement out of order. Each phase must be completed and validated before moving to the next.

---

## 📋 Table of Contents

1. [Phase 1: Project Infrastructure & Environment Setup](#phase-1-project-infrastructure--environment-setup)
2. [Phase 2: Data Acquisition & Preprocessing](#phase-2-data-acquisition--preprocessing)
3. [Phase 3: Image Transformation Implementation](#phase-3-image-transformation-implementation)
4. [Phase 4: Model Architecture Implementation](#phase-4-model-architecture-implementation)
5. [Phase 5: Training Infrastructure](#phase-5-training-infrastructure)
6. [Phase 6: Evaluation & Analysis](#phase-6-evaluation--analysis)
7. [Phase 7: Experiment Orchestration](#phase-7-experiment-orchestration)
8. [Phase 8: Results Analysis & Reporting](#phase-8-results-analysis--reporting)

---

## Phase 1: Project Infrastructure & Environment Setup

**Duration:** Week 1
**Dependencies:** None
**Completion Criteria:** All infrastructure files created, environment tested

### 1.1 Directory Structure

Create the following directory structure:

```
EEG2Img-Benchmark-Study/
├── data/
│   ├── raw/                    # Raw downloaded datasets
│   ├── preprocessed/           # Preprocessed EEG epochs
│   └── images/                 # Generated T2I images
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── downloaders.py      # Dataset download scripts
│   │   ├── loaders.py          # Dataset loading utilities
│   │   └── preprocessors.py    # EEG preprocessing pipeline
│   ├── transforms/
│   │   ├── __init__.py
│   │   ├── gaf.py             # Gramian Angular Fields
│   │   ├── mtf.py             # Markov Transition Fields
│   │   ├── recurrence.py      # Recurrence Plots
│   │   ├── spectrogram.py     # STFT spectrograms
│   │   ├── scalogram.py       # CWT scalograms
│   │   └── topographic.py     # Spatial mapping (SSFI)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn.py             # CNN architectures (ResNet variants)
│   │   ├── vit.py             # Vision Transformer
│   │   ├── baselines.py       # Raw-signal models (1D CNN, LSTM, Transformer)
│   │   └── eegnet.py          # EEGNet (optional)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py         # Training loop
│   │   ├── augmentation.py    # Data augmentation
│   │   └── callbacks.py       # Early stopping, checkpointing
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py         # Accuracy, F1, AUC, confusion matrix
│   │   ├── statistical.py     # Wilcoxon, ANOVA tests
│   │   └── robustness.py      # Noise injection, channel dropout
│   └── utils/
│       ├── __init__.py
│       ├── config.py          # Configuration management
│       ├── logging_utils.py   # Experiment logging
│       └── visualization.py   # Plotting utilities
├── experiments/
│   ├── configs/               # YAML/JSON experiment configs
│   └── scripts/               # Experiment running scripts
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_validation.ipynb
│   ├── 03_transform_examples.ipynb
│   └── 04_results_analysis.ipynb
├── results/
│   ├── models/                # Saved model checkpoints
│   ├── logs/                  # Training logs
│   ├── metrics/               # Evaluation results (CSV/JSON)
│   └── figures/               # Generated plots
├── tests/
│   ├── test_transforms.py
│   ├── test_models.py
│   └── test_preprocessing.py
├── Plans/                     # Research and implementation plans
├── requirements.txt
├── setup.py
├── README.md
├── .gitignore
└── LICENSE
```

**Action Items:**
- ✅ Create all directories using `mkdir -p` commands
- ✅ Verify directory structure with `tree` or `ls -R`

### 1.2 Python Dependencies (requirements.txt)

Create `requirements.txt` with the following exact specifications:

```txt
# Core scientific computing
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0

# EEG processing
mne>=1.5.0
mne-bids>=0.13.0

# Time-series to image transforms
pyts>=0.13.0
PyWavelets>=1.4.0

# Deep learning
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0  # For pretrained ViT models

# Machine learning utilities
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0

# Data management
h5py>=3.8.0
pyarrow>=12.0.0

# Experiment tracking
wandb>=0.15.0
tensorboard>=2.13.0

# Configuration
pyyaml>=6.0
omegaconf>=2.3.0

# Testing
pytest>=7.3.0
pytest-cov>=4.1.0

# Utilities
tqdm>=4.65.0
joblib>=1.3.0
```

**Action Items:**
- ✅ Create requirements.txt file
- ✅ Test installation: `pip install -r requirements.txt`
- ✅ Document any installation issues and resolutions

### 1.3 Git Configuration (.gitignore)

Create `.gitignore` for Python projects:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/
.venv

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb_checkpoints/

# Data files (large datasets)
data/raw/*
data/preprocessed/*
data/images/*
!data/raw/.gitkeep
!data/preprocessed/.gitkeep
!data/images/.gitkeep

# Model checkpoints
results/models/*.pth
results/models/*.pt
results/models/*.h5

# Logs
results/logs/*.log
*.log

# Experiment tracking
wandb/
.wandb/

# pytest
.pytest_cache/
.coverage
htmlcov/

# MyPy
.mypy_cache/

# Environment variables
.env
.env.local

# Temporary files
*.tmp
*.temp
```

**Action Items:**
- ✅ Create .gitignore file
- ✅ Create .gitkeep files in data subdirectories to preserve structure

### 1.4 Package Setup (setup.py)

Create `setup.py` for package installation:

```python
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="eeg2img-benchmark",
    version="0.1.0",
    author="Research Team",
    author_email="your.email@example.com",
    description="Benchmark study of EEG time-series-to-image transformations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JanithRamanayake523/EEG2Img-Benchmark-Study",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "sphinx>=6.0.0",
        ],
    },
)
```

**Action Items:**
- ✅ Create setup.py file
- ✅ Test installation: `pip install -e .`

### 1.5 README Documentation

Create `README.md` with project overview:

```markdown
# EEG Time-Series-to-Image Transformation Benchmark Study

A comprehensive benchmark study comparing multiple time-series-to-image (T2I) transformations for EEG classification across diverse Brain-Computer Interface (BCI) paradigms.

## 🎯 Research Objectives

- Systematically evaluate GAF, MTF, Recurrence Plots, Spectrograms, and Scalograms for EEG classification
- Compare CNN and Vision Transformer models against raw-signal baselines
- Test across Motor Imagery, P300, and SSVEP paradigms
- Assess robustness to noise, channel dropout, and cross-subject generalization

## 📊 Datasets

1. **BCI Competition IV-2a**: 4-class motor imagery (22 channels, 9 subjects)
2. **PhysioNet EEGMMI**: Motor movement/imagery (64 channels, 109 subjects)
3. **BCI Competition III - P300 Speller**: P300 ERP task (64 channels)
4. **SSVEP Datasets**: Frequency-tagged visual stimuli (64-ch and 8-ch variants)

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM

### Setup
```bash
# Clone repository
git clone https://github.com/JanithRamanayake523/EEG2Img-Benchmark-Study.git
cd EEG2Img-Benchmark-Study

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## 🚀 Quick Start

### 1. Download Datasets
```bash
python src/data/downloaders.py --dataset bci_iv_2a --output data/raw
```

### 2. Preprocess EEG Data
```bash
python src/data/preprocessors.py --dataset bci_iv_2a --config experiments/configs/preprocessing.yaml
```

### 3. Generate Image Transformations
```bash
python src/transforms/gaf.py --input data/preprocessed/bci_iv_2a --output data/images/gaf
```

### 4. Train Models
```bash
python experiments/scripts/run_experiment.py --config experiments/configs/bci_iv_2a_gaf_resnet18.yaml
```

## 📁 Project Structure

```
├── data/               # Datasets (raw, preprocessed, images)
├── src/                # Source code
│   ├── data/          # Data loading and preprocessing
│   ├── transforms/    # T2I transformations
│   ├── models/        # Neural network architectures
│   ├── training/      # Training loops and utilities
│   ├── evaluation/    # Metrics and analysis
│   └── utils/         # Helper functions
├── experiments/        # Experiment configs and scripts
├── notebooks/          # Jupyter notebooks for exploration
├── results/            # Model checkpoints, logs, metrics
└── tests/             # Unit tests
```

## 📝 Implementation Phases

This project follows a strict phase-by-phase implementation plan:

1. **Phase 1**: Project Infrastructure & Environment Setup ✅
2. **Phase 2**: Data Acquisition & Preprocessing
3. **Phase 3**: Image Transformation Implementation
4. **Phase 4**: Model Architecture Implementation
5. **Phase 5**: Training Infrastructure
6. **Phase 6**: Evaluation & Analysis
7. **Phase 7**: Experiment Orchestration
8. **Phase 8**: Results Analysis & Reporting

See `Plans/IMPLEMENTATION_PLAN.md` for detailed specifications.

## 🔬 Transformations Implemented

- **GAF (Gramian Angular Fields)**: GASF and GADF variants
- **MTF (Markov Transition Fields)**: State transition encoding
- **Recurrence Plots**: Phase space recurrence visualization
- **Spectrograms**: STFT time-frequency representation
- **Scalograms**: Continuous wavelet transform
- **Topographic Maps**: Spatio-spectral feature images (SSFI)

## 🤖 Models

- **CNNs**: ResNet-18/34/50, Custom Lightweight CNN
- **Vision Transformers**: ViT-B/16, ViT-S/16
- **Raw-Signal Baselines**: 1D CNN, LSTM/BiLSTM, Transformer, EEGNet

## 📊 Evaluation Metrics

- Accuracy, F1-score (macro/weighted), AUC
- Confusion matrices
- Wilcoxon signed-rank tests for pairwise comparisons
- Repeated-measures ANOVA
- Robustness analysis (noise, channel dropout, temporal shifts)

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## 📖 Citation

If you use this code or methodology in your research, please cite:

```bibtex
@misc{eeg2img-benchmark-2026,
  title={Comparative Benchmark Study: EEG Time-Series-to-Image Transformations},
  author={Your Name},
  year={2026},
  publisher={GitHub},
  url={https://github.com/JanithRamanayake523/EEG2Img-Benchmark-Study}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Follow the implementation plan strictly
4. Add tests for new functionality
5. Submit a pull request

## 📧 Contact

For questions or collaborations, please open an issue or contact [your.email@example.com]

## 🙏 Acknowledgments

This research builds upon:
- Kessler et al. (2025) - EEG preprocessing effects
- Prabhukumar et al. (2025) - MTF in EEG
- Hao et al. (2021) - Recurrence plot CNNs
- Mastandrea et al. (2023) - Spatio-spectral images
```

**Action Items:**
- ✅ Create README.md file
- ✅ Update contact information and citation details

### 1.6 Initialize Python Package Structure

Create `__init__.py` files for all Python packages:

**Files to create:**
- `src/__init__.py`
- `src/data/__init__.py`
- `src/transforms/__init__.py`
- `src/models/__init__.py`
- `src/training/__init__.py`
- `src/evaluation/__init__.py`
- `src/utils/__init__.py`

**Content for `src/__init__.py`:**
```python
"""
EEG2Img Benchmark Study
~~~~~~~~~~~~~~~~~~~~~~~

A comprehensive benchmark study comparing time-series-to-image transformations
for EEG classification across diverse BCI paradigms.

:copyright: (c) 2026 by Research Team.
:license: MIT, see LICENSE for more details.
"""

__version__ = '0.1.0'
__author__ = 'Research Team'
__license__ = 'MIT'

from . import data
from . import transforms
from . import models
from . import training
from . import evaluation
from . import utils

__all__ = [
    'data',
    'transforms',
    'models',
    'training',
    'evaluation',
    'utils',
]
```

**Content for module-level `__init__.py` files:**
```python
"""
[Module name] module
"""

# This will be populated as we implement each module
```

**Action Items:**
- ✅ Create all `__init__.py` files
- ✅ Add version and metadata to main `src/__init__.py`

### 1.7 Phase 1 Validation Checklist

Before proceeding to Phase 2, verify:

- [ ] All directories created successfully
- [ ] `requirements.txt` created with exact specifications
- [ ] `.gitignore` created and configured
- [ ] `setup.py` created and package installable
- [ ] `README.md` created with comprehensive documentation
- [ ] All `__init__.py` files created
- [ ] Virtual environment created and activated
- [ ] All dependencies installed without errors
- [ ] Package installed in development mode (`pip install -e .`)
- [ ] Git repository initialized and first commit made

**Validation Commands:**
```bash
# Check directory structure
ls -R

# Verify Python environment
python --version
pip list

# Test package import
python -c "import src; print(src.__version__)"

# Check git status
git status
```

**Phase 1 Exit Criteria:**
- ✅ All validation checklist items passed
- ✅ No installation errors
- ✅ Package importable in Python

---

## Phase 2: Data Acquisition & Preprocessing

**Duration:** Weeks 2-4
**Dependencies:** Phase 1 completed
**Completion Criteria:** All datasets downloaded, preprocessed, and validated

### 2.1 Dataset Download Scripts

**File:** `src/data/downloaders.py`

Implement downloaders for each priority dataset following this structure:

```python
"""
Dataset downloading utilities for EEG benchmark study.

Supports:
- BCI Competition IV-2a (Motor Imagery)
- PhysioNet EEGMMI (Motor Movement/Imagery)
- BCI Competition III - P300 Speller
- SSVEP datasets (Gu et al. 2024, Wang et al. 2021)
"""

import os
import argparse
from pathlib import Path
from typing import Optional
import requests
from tqdm import tqdm
import mne

class DatasetDownloader:
    """Base class for dataset downloaders."""

    def __init__(self, output_dir: str = 'data/raw'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download(self):
        """Download dataset. To be implemented by subclasses."""
        raise NotImplementedError


class BCICompetitionIV2aDownloader(DatasetDownloader):
    """
    Download BCI Competition IV Dataset 2a
    4-class motor imagery: left hand, right hand, feet, tongue
    22 EEG channels, 250 Hz, 9 subjects
    """

    DATASET_URL = "https://www.bbci.de/competition/iv/desc_2a.pdf"
    # Note: Actual data requires manual download or academic access

    def download(self):
        print("BCI Competition IV-2a dataset requires manual download.")
        print(f"Please visit: {self.DATASET_URL}")
        print(f"Download GDF files to: {self.output_dir / 'bci_iv_2a'}")
        print("\nExpected files:")
        for subject in range(1, 10):
            print(f"  - A0{subject}T.gdf (training)")
            print(f"  - A0{subject}E.gdf (evaluation)")


class PhysioNetEEGMMIDownloader(DatasetDownloader):
    """
    Download PhysioNet EEG Motor Movement/Imagery Dataset
    64 channels, 160 Hz, 109 subjects
    Uses MNE-Python's built-in downloader
    """

    def download(self, subjects: Optional[list] = None):
        """
        Download PhysioNet EEGMMI dataset.

        Args:
            subjects: List of subject IDs to download. If None, downloads all.
        """
        if subjects is None:
            subjects = list(range(1, 110))  # All 109 subjects

        output_path = self.output_dir / 'physionet_eegmmi'
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Downloading PhysioNet EEGMMI for {len(subjects)} subjects...")

        for subject in tqdm(subjects, desc="Downloading subjects"):
            try:
                # Download all runs for this subject
                mne.datasets.eegbci.load_data(
                    subject=subject,
                    runs=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                    path=str(output_path),
                    update_path=False
                )
            except Exception as e:
                print(f"Error downloading subject {subject}: {e}")


class BCICompetitionIIIP300Downloader(DatasetDownloader):
    """
    Download BCI Competition III Dataset II (P300 Speller)
    64 channels, ~240 Hz, P300 ERP paradigm
    """

    DATASET_URL = "https://www.bbci.de/competition/iii/"

    def download(self):
        print("BCI Competition III P300 Speller dataset requires manual download.")
        print(f"Please visit: {self.DATASET_URL}")
        print(f"Download to: {self.output_dir / 'bci_iii_p300'}")


class SSVEPDatasetDownloader(DatasetDownloader):
    """
    Download SSVEP datasets
    - Gu et al. 2024: 64-channel, 1-60 Hz
    - Wang et al. 2021: 8-channel wearable, 102 subjects
    """

    def download(self):
        print("SSVEP datasets require manual download from:")
        print("  - Gu et al. (2024): https://www.nature.com/articles/s41597-024-03023-7")
        print("  - Wang et al. (2021): https://pmc.ncbi.nlm.nih.gov/articles/PMC7916479/")
        print(f"Download to: {self.output_dir / 'ssvep'}")


def main():
    parser = argparse.ArgumentParser(description='Download EEG datasets')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['bci_iv_2a', 'physionet', 'bci_iii_p300', 'ssvep', 'all'],
                        help='Dataset to download')
    parser.add_argument('--output', type=str, default='data/raw',
                        help='Output directory')
    parser.add_argument('--subjects', type=int, nargs='+',
                        help='Specific subjects to download (PhysioNet only)')

    args = parser.parse_args()

    downloaders = {
        'bci_iv_2a': BCICompetitionIV2aDownloader,
        'physionet': PhysioNetEEGMMIDownloader,
        'bci_iii_p300': BCICompetitionIIIP300Downloader,
        'ssvep': SSVEPDatasetDownloader,
    }

    if args.dataset == 'all':
        for name, downloader_class in downloaders.items():
            print(f"\n{'='*60}")
            print(f"Downloading {name}...")
            print('='*60)
            downloader = downloader_class(args.output)
            if name == 'physionet' and args.subjects:
                downloader.download(subjects=args.subjects)
            else:
                downloader.download()
    else:
        downloader = downloaders[args.dataset](args.output)
        if args.dataset == 'physionet' and args.subjects:
            downloader.download(subjects=args.subjects)
        else:
            downloader.download()


if __name__ == '__main__':
    main()
```

**Action Items:**
- ✅ Create `src/data/downloaders.py`
- ✅ Test PhysioNet downloader (downloads automatically)
- ✅ Document manual download instructions for competition datasets
- ✅ Verify downloaded data integrity

### 2.2 EEG Preprocessing Pipeline

**File:** `src/data/preprocessors.py`

Implement standardized preprocessing pipeline:

```python
"""
EEG preprocessing pipeline following research plan specifications.

Pipeline steps:
1. Loading & Basic QC
2. Filtering (band-pass 0.5-40 Hz, notch 50/60 Hz)
3. Referencing (common average)
4. Resampling (to 250 Hz)
5. Epoching (paradigm-specific)
6. Baseline correction
7. Artifact removal (ICA/AutoReject)
8. Normalization (z-score, min-max)
"""

import numpy as np
import mne
from mne.preprocessing import ICA
from autoreject import AutoReject
from pathlib import Path
import h5py
from typing import Dict, Optional, Tuple
import argparse
import yaml


class EEGPreprocessor:
    """
    Standardized EEG preprocessing pipeline.
    """

    def __init__(self, config: Dict):
        """
        Initialize preprocessor with configuration.

        Args:
            config: Dictionary with preprocessing parameters
                - filter: {l_freq, h_freq, notch}
                - resample: target sampling rate
                - epoch: {tmin, tmax, baseline}
                - artifact: {ica, amplitude_threshold}
        """
        self.config = config

    def load_raw(self, file_path: str) -> mne.io.Raw:
        """Load raw EEG data."""
        print(f"Loading: {file_path}")

        # Auto-detect file format
        if file_path.endswith('.gdf'):
            raw = mne.io.read_raw_gdf(file_path, preload=True)
        elif file_path.endswith('.edf'):
            raw = mne.io.read_raw_edf(file_path, preload=True)
        elif file_path.endswith('.fif'):
            raw = mne.io.read_raw_fif(file_path, preload=True)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        # Drop EOG channels, keep only EEG
        raw.pick_types(eeg=True, exclude='bads')

        print(f"  Channels: {len(raw.ch_names)}")
        print(f"  Sampling rate: {raw.info['sfreq']} Hz")
        print(f"  Duration: {raw.times[-1]:.1f} s")

        return raw

    def apply_filtering(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Apply band-pass and notch filtering."""
        filter_config = self.config['filter']

        # Band-pass filter
        print(f"Band-pass filtering: {filter_config['l_freq']}-{filter_config['h_freq']} Hz")
        raw.filter(
            l_freq=filter_config['l_freq'],
            h_freq=filter_config['h_freq'],
            filter_length='auto',
            l_trans_bandwidth='auto',
            h_trans_bandwidth='auto',
            method='fir',
            phase='zero',
            fir_window='hamming'
        )

        # Notch filter (power line noise)
        if 'notch' in filter_config and filter_config['notch']:
            print(f"Notch filtering: {filter_config['notch']} Hz")
            raw.notch_filter(freqs=filter_config['notch'])

        return raw

    def apply_referencing(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Apply common average reference."""
        print("Applying common average reference")
        raw.set_eeg_reference('average', projection=False)
        return raw

    def apply_resampling(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Resample to target frequency."""
        target_sfreq = self.config.get('resample', 250)

        if raw.info['sfreq'] != target_sfreq:
            print(f"Resampling: {raw.info['sfreq']} Hz -> {target_sfreq} Hz")
            raw.resample(target_sfreq, npad='auto')

        return raw

    def create_epochs(self, raw: mne.io.Raw, events: np.ndarray,
                     event_id: Dict) -> mne.Epochs:
        """Create epochs from continuous data."""
        epoch_config = self.config['epoch']

        print(f"Epoching: {epoch_config['tmin']} to {epoch_config['tmax']} s")

        epochs = mne.Epochs(
            raw,
            events=events,
            event_id=event_id,
            tmin=epoch_config['tmin'],
            tmax=epoch_config['tmax'],
            baseline=epoch_config.get('baseline', None),
            preload=True,
            reject_by_annotation=True
        )

        print(f"  Created {len(epochs)} epochs")

        return epochs

    def apply_artifact_removal(self, epochs: mne.Epochs) -> mne.Epochs:
        """Remove artifacts using ICA and amplitude-based rejection."""
        artifact_config = self.config.get('artifact', {})

        # ICA for artifact removal
        if artifact_config.get('ica', True):
            print("Applying ICA for artifact removal")
            ica = ICA(
                n_components=15,
                random_state=42,
                method='fastica',
                max_iter=200
            )
            ica.fit(epochs)

            # Automatic component selection (simplified)
            # In practice, use more sophisticated methods like ADJUST
            ica.exclude = []  # Manual inspection or automatic detection
            epochs = ica.apply(epochs)

        # Amplitude-based rejection
        threshold = artifact_config.get('amplitude_threshold', 100e-6)  # 100 µV
        print(f"Amplitude-based rejection: threshold = {threshold*1e6} µV")

        reject_criteria = {'eeg': threshold}
        epochs.drop_bad(reject=reject_criteria)

        print(f"  Epochs remaining: {len(epochs)}")

        return epochs

    def apply_normalization(self, epochs: mne.Epochs,
                           method: str = 'zscore') -> np.ndarray:
        """
        Normalize epoch data.

        Args:
            epochs: MNE Epochs object
            method: 'zscore' or 'minmax'

        Returns:
            Normalized data array (n_epochs, n_channels, n_times)
        """
        data = epochs.get_data()

        if method == 'zscore':
            # Z-score normalization per channel per epoch
            mean = data.mean(axis=2, keepdims=True)
            std = data.std(axis=2, keepdims=True)
            data = (data - mean) / (std + 1e-8)

        elif method == 'minmax':
            # Min-max to [-1, 1] for GAF compatibility
            min_val = data.min(axis=2, keepdims=True)
            max_val = data.max(axis=2, keepdims=True)
            data = 2 * (data - min_val) / (max_val - min_val + 1e-8) - 1

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        print(f"Applied {method} normalization")

        return data

    def save_preprocessed(self, data: np.ndarray, labels: np.ndarray,
                         metadata: Dict, output_path: str):
        """Save preprocessed data to HDF5."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Saving to: {output_path}")

        with h5py.File(output_path, 'w') as f:
            f.create_dataset('data', data=data, compression='gzip')
            f.create_dataset('labels', data=labels)

            # Save metadata
            for key, value in metadata.items():
                if isinstance(value, (str, int, float)):
                    f.attrs[key] = value
                elif isinstance(value, list):
                    f.attrs[key] = str(value)

        print(f"  Shape: {data.shape}")
        print(f"  Labels: {np.unique(labels)}")

    def process(self, input_file: str, events: np.ndarray, event_id: Dict,
                output_file: str):
        """Run full preprocessing pipeline."""
        print("="*60)
        print(f"Preprocessing: {input_file}")
        print("="*60)

        # Load raw data
        raw = self.load_raw(input_file)

        # Preprocessing steps
        raw = self.apply_filtering(raw)
        raw = self.apply_referencing(raw)
        raw = self.apply_resampling(raw)

        # Epoching
        epochs = self.create_epochs(raw, events, event_id)

        # Artifact removal
        epochs = self.apply_artifact_removal(epochs)

        # Normalization
        data = self.apply_normalization(epochs, method='zscore')
        labels = epochs.events[:, -1]

        # Save
        metadata = {
            'sfreq': epochs.info['sfreq'],
            'n_channels': len(epochs.ch_names),
            'ch_names': epochs.ch_names,
            'tmin': epochs.tmin,
            'tmax': epochs.tmax,
        }

        self.save_preprocessed(data, labels, metadata, output_file)

        print("Preprocessing complete!\n")


def main():
    parser = argparse.ArgumentParser(description='Preprocess EEG data')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to preprocessing config YAML')
    parser.add_argument('--input', type=str, required=True,
                        help='Input raw EEG file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output preprocessed HDF5 file')

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize preprocessor
    preprocessor = EEGPreprocessor(config['preprocessing'])

    # TODO: Extract events from specific dataset format
    # This is dataset-specific and needs to be implemented
    events = None
    event_id = None

    # Process
    preprocessor.process(args.input, events, event_id, args.output)


if __name__ == '__main__':
    main()
```

**Action Items:**
- ✅ Create `src/data/preprocessors.py`
- ✅ Create example preprocessing config YAML
- ✅ Test on sample EEG data
- ✅ Validate output HDF5 format

### 2.3 Dataset Loaders

**File:** `src/data/loaders.py`

Implement dataset-specific loaders:

```python
"""
Dataset loading utilities for different EEG datasets.
"""

import numpy as np
import mne
from pathlib import Path
from typing import Tuple, Dict
import h5py


class BCICompetitionIV2aLoader:
    """Load BCI Competition IV-2a dataset."""

    EVENT_IDS = {
        'left_hand': 1,
        'right_hand': 2,
        'feet': 3,
        'tongue': 4,
    }

    @staticmethod
    def load_subject(subject_id: int, data_path: str,
                     session: str = 'T') -> Tuple[mne.io.Raw, np.ndarray, Dict]:
        """
        Load data for one subject.

        Args:
            subject_id: Subject number (1-9)
            data_path: Path to raw data directory
            session: 'T' for training, 'E' for evaluation

        Returns:
            raw: MNE Raw object
            events: Event array
            event_id: Event ID dictionary
        """
        file_path = Path(data_path) / f'A0{subject_id}{session}.gdf'

        raw = mne.io.read_raw_gdf(str(file_path), preload=True)
        events, event_id = mne.events_from_annotations(raw)

        # Map to standard event IDs
        # This mapping depends on the specific dataset annotations

        return raw, events, BCICompetitionIV2aLoader.EVENT_IDS


class PhysioNetLoader:
    """Load PhysioNet EEGMMI dataset."""

    RUNS = {
        'baseline_eyes_open': [1],
        'baseline_eyes_closed': [2],
        'motor_execution_lr': [3, 7, 11],  # Left/right hand
        'motor_imagery_lr': [4, 8, 12],
        'motor_execution_fists_feet': [5, 9, 13],
        'motor_imagery_fists_feet': [6, 10, 14],
    }

    @staticmethod
    def load_subject(subject_id: int, data_path: str,
                     task: str = 'motor_imagery_lr') -> Tuple[mne.io.Raw, np.ndarray, Dict]:
        """Load data for specific task."""
        runs = PhysioNetLoader.RUNS[task]

        raw_files = mne.datasets.eegbci.load_data(
            subject=subject_id,
            runs=runs,
            path=data_path
        )

        raw = mne.io.read_raw_edf(raw_files[0], preload=True)
        for raw_file in raw_files[1:]:
            raw.append(mne.io.read_raw_edf(raw_file, preload=True))

        events, event_id = mne.events_from_annotations(raw)

        return raw, events, event_id


def load_preprocessed(file_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load preprocessed HDF5 data.

    Returns:
        data: (n_epochs, n_channels, n_times)
        labels: (n_epochs,)
        metadata: Dictionary with dataset info
    """
    with h5py.File(file_path, 'r') as f:
        data = f['data'][:]
        labels = f['labels'][:]

        metadata = {key: f.attrs[key] for key in f.attrs.keys()}

    return data, labels, metadata
```

**Action Items:**
- ✅ Create `src/data/loaders.py`
- ✅ Implement loaders for each dataset
- ✅ Test loading and verify data integrity

### 2.4 Create Preprocessing Configuration

**File:** `experiments/configs/preprocessing.yaml`

```yaml
# Standard EEG preprocessing configuration

preprocessing:
  # Filtering parameters
  filter:
    l_freq: 0.5      # High-pass cutoff (Hz)
    h_freq: 40.0     # Low-pass cutoff (Hz)
    notch: 50        # Notch filter frequency (50 or 60 Hz)

  # Resampling
  resample: 250      # Target sampling rate (Hz)

  # Epoching parameters
  epoch:
    tmin: 0.0        # Start time relative to event (s)
    tmax: 4.0        # End time relative to event (s)
    baseline: null   # Baseline correction window [start, end] or null

  # Artifact removal
  artifact:
    ica: true                    # Apply ICA
    amplitude_threshold: 100e-6  # Amplitude rejection threshold (V)

# Dataset-specific configurations
datasets:
  bci_iv_2a:
    epoch:
      tmin: 0.5
      tmax: 3.5
      baseline: null

  bci_iii_p300:
    epoch:
      tmin: 0.0
      tmax: 1.0
      baseline: [-0.2, 0.0]

  ssvep:
    epoch:
      tmin: 0.0
      tmax: 5.0
      baseline: null
```

**Action Items:**
- ✅ Create `experiments/configs/preprocessing.yaml`
- ✅ Create dataset-specific config variants

### 2.5 Create Preprocessing Validation Notebook

**File:** `notebooks/02_preprocessing_validation.ipynb`

Create Jupyter notebook to:
- Load raw EEG data
- Visualize raw signals
- Apply preprocessing steps one by one
- Show before/after comparisons
- Validate artifact removal
- Check signal quality metrics

**Action Items:**
- ✅ Create validation notebook
- ✅ Generate preprocessing report for each dataset
- ✅ Document any dataset-specific issues

### 2.6 Phase 2 Validation Checklist

Before proceeding to Phase 3, verify:

- [ ] `src/data/downloaders.py` implemented and tested
- [ ] At least one dataset downloaded successfully (PhysioNet recommended for auto-download)
- [ ] `src/data/preprocessors.py` implemented with all pipeline steps
- [ ] `src/data/loaders.py` implemented for each dataset
- [ ] Preprocessing config YAML created
- [ ] Preprocessing pipeline tested on real EEG data
- [ ] Validation notebook created with visualizations
- [ ] Preprocessed data saved in HDF5 format
- [ ] Signal quality verified (SNR, artifact removal effectiveness)
- [ ] No data leakage in epoching/preprocessing

**Validation Commands:**
```bash
# Test downloader
python src/data/downloaders.py --dataset physionet --output data/raw --subjects 1

# Test preprocessor
python src/data/preprocessors.py --config experiments/configs/preprocessing.yaml \
    --input data/raw/sample.gdf --output data/preprocessed/sample.h5

# Verify HDF5 output
python -c "import h5py; f = h5py.File('data/preprocessed/sample.h5', 'r'); print(f.keys()); print(f['data'].shape)"
```

**Phase 2 Exit Criteria:**
- ✅ All validation checklist items passed
- ✅ At least one complete dataset preprocessed
- ✅ Preprocessing reports generated
- ✅ HDF5 files verified and loadable

---

## Phase 3: Image Transformation Implementation

**Duration:** Weeks 5-7
**Dependencies:** Phase 2 completed
**Completion Criteria:** All 6 transforms implemented, tested, and validated

### 3.1 Gramian Angular Fields

**File:** `src/transforms/gaf.py`

Implement GAF (GASF and GADF) transformations:

```python
"""
Gramian Angular Field (GAF) transformations for EEG time-series.

Implements:
- GASF (Gramian Angular Summation Field)
- GADF (Gramian Angular Difference Field)
"""

import numpy as np
from pyts.image import GramianAngularField
from typing import Literal, Tuple
import argparse
import h5py
from pathlib import Path
from tqdm import tqdm


class GAFTransformer:
    """
    Transform EEG time-series to GAF images.
    """

    def __init__(self, image_size: int = 128,
                 method: Literal['summation', 'difference'] = 'summation'):
        """
        Initialize GAF transformer.

        Args:
            image_size: Output image size (NxN)
            method: 'summation' for GASF, 'difference' for GADF
        """
        self.image_size = image_size
        self.method = method
        self.transformer = GramianAngularField(
            image_size=image_size,
            method=method,
            sample_range=(-1, 1)
        )

    def normalize_to_range(self, x: np.ndarray) -> np.ndarray:
        """Normalize to [-1, 1] for GAF."""
        x_min = x.min(axis=-1, keepdims=True)
        x_max = x.max(axis=-1, keepdims=True)
        return 2 * (x - x_min) / (x_max - x_min + 1e-8) - 1

    def transform_epoch(self, epoch: np.ndarray) -> np.ndarray:
        """
        Transform single epoch to GAF image.

        Args:
            epoch: (n_channels, n_times) array

        Returns:
            gaf_images: (n_channels, image_size, image_size) array
        """
        # Normalize each channel to [-1, 1]
        epoch_norm = self.normalize_to_range(epoch)

        # Apply GAF to each channel
        gaf_images = self.transformer.fit_transform(epoch_norm)

        return gaf_images

    def transform_batch(self, data: np.ndarray,
                       strategy: str = 'per_channel') -> np.ndarray:
        """
        Transform batch of epochs.

        Args:
            data: (n_epochs, n_channels, n_times) array
            strategy: 'per_channel' or 'average'

        Returns:
            Transformed images based on strategy
        """
        n_epochs = data.shape[0]
        results = []

        for i in tqdm(range(n_epochs), desc=f"GAF-{self.method}"):
            gaf_image = self.transform_epoch(data[i])

            if strategy == 'per_channel':
                # Keep all channels as separate image layers
                results.append(gaf_image)
            elif strategy == 'average':
                # Average across channels to get single image
                results.append(gaf_image.mean(axis=0, keepdims=True))

        return np.array(results)

    def save_images(self, images: np.ndarray, labels: np.ndarray,
                   output_path: str):
        """Save GAF images to HDF5."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_path, 'w') as f:
            f.create_dataset('images', data=images, compression='gzip')
            f.create_dataset('labels', data=labels)
            f.attrs['transform'] = f'gaf_{self.method}'
            f.attrs['image_size'] = self.image_size

        print(f"Saved GAF images: {output_path}")
        print(f"  Shape: {images.shape}")


def main():
    parser = argparse.ArgumentParser(description='Generate GAF images from EEG')
    parser.add_argument('--input', type=str, required=True,
                        help='Input preprocessed HDF5 file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output GAF images HDF5 file')
    parser.add_argument('--method', type=str, default='summation',
                        choices=['summation', 'difference'],
                        help='GAF method: summation (GASF) or difference (GADF)')
    parser.add_argument('--size', type=int, default=128,
                        help='Image size (NxN)')
    parser.add_argument('--strategy', type=str, default='per_channel',
                        choices=['per_channel', 'average'],
                        help='Multi-channel strategy')

    args = parser.parse_args()

    # Load preprocessed data
    with h5py.File(args.input, 'r') as f:
        data = f['data'][:]
        labels = f['labels'][:]

    # Transform
    transformer = GAFTransformer(image_size=args.size, method=args.method)
    images = transformer.transform_batch(data, strategy=args.strategy)

    # Save
    transformer.save_images(images, labels, args.output)


if __name__ == '__main__':
    main()
```

**Action Items:**
- ✅ Create `src/transforms/gaf.py`
- ✅ Test GASF and GADF variants
- ✅ Validate images visually
- ✅ Test different image sizes (64, 128, 256)

### 3.2-3.6 Remaining Transformations

Create similar implementations for:

- **MTF** (`src/transforms/mtf.py`): Markov Transition Fields
- **Recurrence Plots** (`src/transforms/recurrence.py`)
- **Spectrograms** (`src/transforms/spectrogram.py`)
- **Scalograms** (`src/transforms/scalogram.py`)
- **Topographic Maps** (`src/transforms/topographic.py`)

Each file should follow the same structure:
1. Transformer class with configurable parameters
2. `transform_epoch()` method
3. `transform_batch()` method
4. `save_images()` method
5. CLI interface via `main()`

**Action Items for each transform:**
- ✅ Implement transformer class
- ✅ Add parameter validation
- ✅ Test on synthetic signals (sine waves, chirps)
- ✅ Validate on real EEG data
- ✅ Create example images

### 3.7 Transform Registry

**File:** `src/transforms/__init__.py`

```python
"""
Time-series-to-image transformations.
"""

from .gaf import GAFTransformer
from .mtf import MTFTransformer
from .recurrence import RecurrencePlotTransformer
from .spectrogram import SpectrogramTransformer
from .scalogram import ScalogramTransformer
from .topographic import TopographicTransformer

TRANSFORM_REGISTRY = {
    'gaf_summation': lambda **kwargs: GAFTransformer(method='summation', **kwargs),
    'gaf_difference': lambda **kwargs: GAFTransformer(method='difference', **kwargs),
    'mtf': MTFTransformer,
    'recurrence': RecurrencePlotTransformer,
    'spectrogram': SpectrogramTransformer,
    'scalogram': ScalogramTransformer,
    'topographic': TopographicTransformer,
}

__all__ = [
    'GAFTransformer',
    'MTFTransformer',
    'RecurrencePlotTransformer',
    'SpectrogramTransformer',
    'ScalogramTransformer',
    'TopographicTransformer',
    'TRANSFORM_REGISTRY',
]
```

### 3.8 Transform Validation Notebook

**File:** `notebooks/03_transform_examples.ipynb`

Create notebook to:
- Load sample preprocessed EEG epoch
- Apply all transforms
- Visualize all 6 transform outputs side-by-side
- Test on synthetic signals (validate expected patterns)
- Compare computational time
- Save example figures

### 3.9 Phase 3 Validation Checklist

- [ ] All 6 transforms implemented
- [ ] GAF produces symmetric images with diagonal structure
- [ ] MTF shows transition patterns
- [ ] Recurrence plots show diagonal lines for periodic signals
- [ ] Spectrograms show correct frequency peaks
- [ ] Scalograms show multi-resolution time-frequency
- [ ] Topographic maps preserve electrode spatial layout
- [ ] All transforms tested on synthetic signals
- [ ] Transform validation notebook completed
- [ ] Example images saved for documentation
- [ ] Computational benchmarks recorded

**Phase 3 Exit Criteria:**
- ✅ All transforms validated against expected patterns
- ✅ No numerical errors or NaN values
- ✅ Images saved in correct format
- ✅ Transform registry functional

---

## Phase 4: Model Architecture Implementation

**Duration:** Weeks 8-10
**Dependencies:** Phase 3 completed
**Completion Criteria:** All model architectures implemented and tested

### 4.1 CNN Models

**File:** `src/models/cnn.py`

Implement ResNet variants and lightweight CNN.

### 4.2 Vision Transformers

**File:** `src/models/vit.py`

Implement ViT using timm library.

### 4.3 Raw-Signal Baselines

**File:** `src/models/baselines.py`

Implement 1D CNN, LSTM, Transformer for raw time-series.

### 4.4 Model Registry

**File:** `src/models/__init__.py`

Create model factory and registry.

### 4.5 Phase 4 Validation Checklist

- [ ] All models instantiate without errors
- [ ] Forward pass works with dummy inputs
- [ ] Output dimensions correct for number of classes
- [ ] Pretrained weights load successfully
- [ ] Memory usage acceptable
- [ ] Model summary shows correct architecture

---

## Phase 5: Training Infrastructure

**Duration:** Weeks 11-12
**Dependencies:** Phase 4 completed
**Completion Criteria:** Training loop functional, cross-validation working

### 5.1 Data Augmentation

**File:** `src/training/augmentation.py`

### 5.2 Training Loop

**File:** `src/training/trainer.py`

### 5.3 Callbacks

**File:** `src/training/callbacks.py`

### 5.4 Phase 5 Validation Checklist

- [ ] Training loop runs without errors
- [ ] Loss decreases over epochs
- [ ] Validation metrics computed correctly
- [ ] Early stopping works
- [ ] Model checkpoints saved
- [ ] Augmentation applies correctly
- [ ] Cross-validation implements correctly (no data leakage)

---

## Phase 6: Evaluation & Analysis

**Duration:** Weeks 13-14
**Dependencies:** Phase 5 completed
**Completion Criteria:** All evaluation metrics and tests implemented

### 6.1 Metrics

**File:** `src/evaluation/metrics.py`

### 6.2 Statistical Tests

**File:** `src/evaluation/statistical.py`

### 6.3 Robustness Testing

**File:** `src/evaluation/robustness.py`

### 6.4 Phase 6 Validation Checklist

- [ ] All metrics compute correctly
- [ ] Statistical tests produce valid p-values
- [ ] Robustness tests run successfully
- [ ] Confusion matrices visualize correctly

---

## Phase 7: Experiment Orchestration

**Duration:** Weeks 15-18
**Dependencies:** Phase 6 completed
**Completion Criteria:** Full experimental grid running

### 7.1 Experiment Configs

Create YAML configs for each experiment combination.

### 7.2 Experiment Runner

**File:** `experiments/scripts/run_experiment.py`

### 7.3 Batch Execution

**File:** `experiments/scripts/run_grid.py`

### 7.4 Phase 7 Validation Checklist

- [ ] Single experiment runs end-to-end
- [ ] Results saved correctly
- [ ] Batch execution works
- [ ] Resource usage acceptable
- [ ] Progress tracking functional

---

## Phase 8: Results Analysis & Reporting

**Duration:** Weeks 19-24
**Dependencies:** Phase 7 completed
**Completion Criteria:** Paper draft complete, code documented

### 8.1 Result Aggregation

**File:** `src/evaluation/aggregate_results.py`

### 8.2 Analysis Notebook

**File:** `notebooks/04_results_analysis.ipynb`

### 8.3 Paper Writing

Draft manuscript following research plan outline.

### 8.4 Phase 8 Validation Checklist

- [ ] All results aggregated
- [ ] Statistical analysis complete
- [ ] Figures generated
- [ ] Tables formatted
- [ ] Paper draft written
- [ ] Code documented
- [ ] Repository cleaned

---

## ⚠️ IMPLEMENTATION RULES

**MANDATORY RULES - MUST BE FOLLOWED:**

1. **Sequential Execution**: Phases MUST be completed in order. Do not skip ahead.

2. **Phase Completion**: Each phase's validation checklist MUST be fully completed before starting the next phase.

3. **No Deviations**: Implement exactly as specified in this plan. Any deviations must be documented and justified.

4. **Testing**: Every module MUST be tested before moving forward.

5. **Documentation**: Code MUST be documented with docstrings following this plan's examples.

6. **File Structure**: Create files exactly as specified in directory structure.

7. **Dependencies**: Verify all dependencies installed before starting each phase.

8. **Validation**: Run all validation commands for each phase.

9. **Version Control**: Commit after completing each major component.

10. **Data Integrity**: Verify data quality at each preprocessing step.

---

## 📊 Progress Tracking

**Current Phase:** Phase 1 - Project Infrastructure
**Current Status:** In Progress

**Completed Phases:**
- [ ] Phase 1: Project Infrastructure & Environment Setup
- [ ] Phase 2: Data Acquisition & Preprocessing
- [ ] Phase 3: Image Transformation Implementation
- [ ] Phase 4: Model Architecture Implementation
- [ ] Phase 5: Training Infrastructure
- [ ] Phase 6: Evaluation & Analysis
- [ ] Phase 7: Experiment Orchestration
- [ ] Phase 8: Results Analysis & Reporting

**Next Milestone:** Complete Phase 1 validation checklist

---

## 📞 Support & References

**Key References:**
- Research Plan PDF: `Plans/Comparative Benchmark Study_ EEG Time-Series-to-Image Transformations (Research Plan).pdf`
- Implementation Plan: This document

**Libraries Documentation:**
- MNE-Python: https://mne.tools
- pyts: https://pyts.readthedocs.io
- PyTorch: https://pytorch.org
- timm: https://github.com/huggingface/pytorch-image-models

---

**Last Updated:** 2026-04-02
**Version:** 1.0
**Status:** Active Implementation
