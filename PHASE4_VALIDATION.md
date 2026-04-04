# Phase 4 Validation Checklist

**Phase:** Model Architecture Implementation
**Date Started:** 2026-04-02
**Status:** 🔄 IN PROGRESS

---

## Implementation Requirements

### 4.1 CNN Models ✅

**File:** `src/models/cnn.py` (378 lines)

- [x] **ResNet-18**
  - [x] Pretrained ImageNet weights support
  - [x] Modified first conv layer for multi-channel input
  - [x] Weight initialization strategy (averaging/repeating RGB)
  - [x] Configurable dropout before FC layer
  - [x] Output dimension matches num_classes

- [x] **ResNet-50**
  - [x] Deeper architecture with bottleneck blocks
  - [x] Pretrained weights support
  - [x] Multi-channel input adaptation
  - [x] Configurable parameters

- [x] **Lightweight CNN**
  - [x] Custom 4-block architecture
  - [x] BatchNorm + ReLU + MaxPool in each block
  - [x] Global average pooling
  - [x] Fewer parameters than ResNet
  - [x] Base filters configurable

- [x] **Utilities**
  - [x] `get_cnn_model()` factory function
  - [x] `count_parameters()` utility
  - [x] Command-line testing script

**Architecture Details:**
- ResNet-18: ~11M parameters (without pretrained adjustments)
- ResNet-50: ~23M parameters
- Lightweight CNN: ~4-8M parameters (depends on base_filters)

### 4.2 Vision Transformers ✅

**File:** `src/models/vit.py` (334 lines)

- [x] **ViT Base Classes**
  - [x] Generic `ViTEEG` wrapper for timm models
  - [x] Patch embedding modification for multi-channel input
  - [x] Pretrained weight adaptation
  - [x] Configurable image size

- [x] **ViT Variants**
  - [x] ViT-Base/16 (12 layers, 768 dim, 12 heads, ~86M params)
  - [x] ViT-Small/16 (12 layers, 384 dim, 6 heads, ~22M params)
  - [x] ViT-Tiny/16 (12 layers, 192 dim, 3 heads, ~5M params)

- [x] **Utilities**
  - [x] `get_vit_model()` factory function
  - [x] Attention map extraction (placeholder for visualization)
  - [x] Command-line testing script

**Dependencies:**
- timm>=0.9.0 for pretrained ViT models

### 4.3 Raw-Signal Baselines ✅

**File:** `src/models/baselines.py` (587 lines)

- [x] **1D CNN**
  - [x] 4 temporal convolution blocks
  - [x] Increasing filter sizes (base_filters * 1/2/4/8)
  - [x] Global average pooling
  - [x] Handles multi-channel time-series

- [x] **LSTM Classifier**
  - [x] Bidirectional and unidirectional variants
  - [x] Multi-layer support
  - [x] Uses final hidden state for classification
  - [x] Configurable hidden size and layers

- [x] **Transformer Classifier**
  - [x] Self-attention on time sequences
  - [x] Positional encoding
  - [x] Configurable heads and layers
  - [x] Global average pooling over time

- [x] **EEGNet**
  - [x] Specialized architecture for EEG signals
  - [x] Depthwise spatial convolution
  - [x] Separable convolution
  - [x] Based on Lawhern et al. (2018)
  - [x] Expects 4D input (batch, 1, channels, time)

- [x] **Utilities**
  - [x] `get_baseline_model()` factory function
  - [x] Positional encoding module
  - [x] Command-line testing script

**Architecture Details:**
- 1D CNN: ~2-5M parameters (depends on base_filters)
- LSTM: ~1-3M parameters (depends on hidden_size, num_layers)
- BiLSTM: ~2-6M parameters (2x LSTM)
- Transformer: ~1-4M parameters (depends on d_model, num_layers)
- EEGNet: ~2-5K parameters (very compact!)

### 4.4 Model Registry ✅

**File:** `src/models/__init__.py` (198 lines, updated)

- [x] **Unified Registry**
  - [x] `MODEL_REGISTRY` dictionary with all models
  - [x] 11 total models registered:
    - 3 CNN models (resnet18, resnet50, lightweight_cnn)
    - 3 ViT models (vit_base, vit_small, vit_tiny)
    - 5 baseline models (cnn1d, lstm, bilstm, transformer, eegnet)

- [x] **Factory Functions**
  - [x] `get_model()`: Unified model instantiation
  - [x] `list_models()`: List models by category
  - [x] `count_parameters()`: Parameter counting
  - [x] `get_model_info()`: Comprehensive model information

- [x] **Exports**
  - [x] All model classes exported in `__all__`
  - [x] All factory functions exported
  - [x] Consistent API across all models

---

## Testing & Validation

### 4.5 Functionality Tests ⏳

**Test Script:** `experiments/scripts/test_models.py`

- [ ] **All models instantiate without errors**
  - [ ] CNN models (resnet18, resnet50, lightweight_cnn)
  - [ ] ViT models (vit_base, vit_small, vit_tiny)
  - [ ] Baseline models (cnn1d, lstm, bilstm, transformer, eegnet)

- [ ] **Forward pass works with dummy inputs**
  - [ ] Image models with (batch, 25, 64, 64) input
  - [ ] Image models with (batch, 25, 128, 128) input
  - [ ] Image models with (batch, 25, 224, 224) input
  - [ ] Baseline models with (batch, 25, 751) input
  - [ ] EEGNet with (batch, 1, 25, 751) input

- [ ] **Output dimensions correct for number of classes**
  - [ ] All models output (batch_size, num_classes)
  - [ ] Logits not probabilities (no softmax in forward)

- [ ] **Pretrained weights load successfully**
  - [ ] ResNet-18 with ImageNet weights
  - [ ] ResNet-50 with ImageNet weights
  - [ ] ViT models with ImageNet weights
  - [ ] Weight adaptation for multi-channel works correctly

- [ ] **Memory usage acceptable**
  - [ ] Models fit in typical GPU memory (8-16 GB)
  - [ ] No memory leaks during forward/backward

- [ ] **Model summary shows correct architecture**
  - [ ] Layer structure matches design
  - [ ] Parameter counts reasonable

### 4.6 Parameter Count Validation ⏳

Expected parameter ranges:

| Model | Expected Parameters | Status |
|-------|---------------------|--------|
| ResNet-18 | ~11M | ⏳ |
| ResNet-50 | ~23M | ⏳ |
| Lightweight CNN | 4-8M | ⏳ |
| ViT-Base | ~86M | ⏳ |
| ViT-Small | ~22M | ⏳ |
| ViT-Tiny | ~5M | ⏳ |
| 1D CNN | 2-5M | ⏳ |
| LSTM | 1-3M | ⏳ |
| BiLSTM | 2-6M | ⏳ |
| Transformer | 1-4M | ⏳ |
| EEGNet | 2-5K | ⏳ |

### 4.7 Code Quality ✅

- [x] **Documentation**
  - [x] All classes have comprehensive docstrings
  - [x] All methods documented with args/returns
  - [x] Usage examples in docstrings
  - [x] References to papers included

- [x] **Code Organization**
  - [x] Consistent class structure across all models
  - [x] Proper inheritance and modularity
  - [x] Clear separation of concerns
  - [x] Factory pattern for easy instantiation

- [x] **Error Handling**
  - [x] Input validation in factory functions
  - [x] Clear error messages
  - [x] Helpful suggestions when model not found

---

## Deliverables

### 4.8 Files Created ✅

**Core Implementations:**
- [x] `src/models/cnn.py` (378 lines)
- [x] `src/models/vit.py` (334 lines)
- [x] `src/models/baselines.py` (587 lines)
- [x] `src/models/__init__.py` (198 lines, updated)

**Scripts:**
- [x] `experiments/scripts/test_models.py` (comprehensive test script)

**Documentation:**
- [x] `PHASE4_VALIDATION.md` (this file)

### 4.9 Version Control ⏳

- [ ] All files committed to git
- [ ] Commit message with detailed description
- [ ] Co-authored attribution included

---

## Phase 4 Exit Criteria

From `IMPLEMENTATION_PLAN.md` - all must be checked before proceeding:

- [ ] All models instantiate without errors
- [ ] Forward pass works with dummy inputs
- [ ] Output dimensions correct for number of classes
- [ ] Pretrained weights load successfully
- [ ] Memory usage acceptable
- [ ] Model summary shows correct architecture

---

## Dependencies

### 4.10 Required Packages ⏳

- [ ] **PyTorch** (`torch>=2.0.0`)
- [ ] **torchvision** (`torchvision>=0.15.0`)
- [ ] **timm** (`timm>=0.9.0`) - For ViT models

Installation:
```bash
pip install torch torchvision timm --index-url https://download.pytorch.org/whl/cpu
# OR for GPU:
pip install torch torchvision timm
```

---

## Testing Commands

```bash
# Test individual model modules
python src/models/cnn.py
python src/models/vit.py
python src/models/baselines.py

# Run comprehensive validation
python experiments/scripts/test_models.py
```

---

## Known Issues & Future Work

### Minor Issues
- [ ] ViT attention map extraction not yet implemented (placeholder)
- [ ] Could add more ViT variants (ViT-Large, different patch sizes)
- [ ] Could add more ResNet variants (ResNet-34, ResNet-101)

### Future Enhancements
1. **Model Improvements**
   - Mixed precision training support (automatic mixed precision)
   - Gradient checkpointing for larger models
   - Model ensemble utilities
   - Custom model architectures specific to EEG

2. **Visualization**
   - Attention map extraction for ViT
   - Grad-CAM for CNNs
   - Feature map visualization

3. **Optimization**
   - Knowledge distillation from larger to smaller models
   - Pruning and quantization
   - Neural architecture search

4. **Additional Baselines**
   - Wavelet-based models
   - Graph neural networks for electrode connectivity
   - Hybrid CNN-RNN architectures

---

## Sign-Off

**Phase 4 Status:** 🔄 **IN PROGRESS**

**Completed:**
- ✅ All model architectures implemented (11 models)
- ✅ Model registry and factory functions created
- ✅ Comprehensive documentation
- ✅ Test script prepared

**Pending:**
- ⏳ PyTorch installation
- ⏳ Model testing with dummy inputs
- ⏳ Pretrained weight loading validation
- ⏳ Git commit

**Implementation Time So Far:** ~2 hours
**Lines of Code:** 1,497 (4 files)

---

## Next Steps

1. **Complete Installation**
   - Finish PyTorch installation
   - Verify timm library installed

2. **Run Tests**
   - Execute `test_models.py`
   - Verify all models pass validation
   - Document parameter counts

3. **Finalize Phase 4**
   - Update this checklist with test results
   - Commit all files to git
   - Mark phase as complete

4. **Proceed to Phase 5**
   - Training infrastructure
   - Data augmentation
   - Training loop implementation

Refer to `IMPLEMENTATION_PLAN.md` for Phase 5 detailed requirements.
