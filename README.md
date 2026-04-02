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
