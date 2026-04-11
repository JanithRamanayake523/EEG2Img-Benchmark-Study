# GPU Setup Guide for PyTorch

## Overview

This guide helps you install **PyTorch with GPU support** for the EEG2Img-Benchmark-Study project.

**GPU support enables:**
- ✅ 10-50× faster training (3-5 minutes vs 2-3 hours per model)
- ✅ Faster data transformation and evaluation
- ✅ Ability to use larger batch sizes and models

---

## Prerequisites

### Check Your GPU

Open Command Prompt and run:

```bash
nvidia-smi
```

**Output example:**
```
+-------------------------+
| NVIDIA-SMI 535.104.05   |
+-------------------------+
| GPU  Name                    | VRAM  |
|  0   NVIDIA GeForce RTX 3080 | 10GB  |
+-------------------------+
```

**Note:** If you don't see this output, you may not have:
- NVIDIA GPU
- NVIDIA drivers installed
- CUDA toolkit installed

### Check System Specifications

Find your:
1. **GPU Model** - From `nvidia-smi` output
2. **CUDA Version** - From `nvidia-smi` output (last line usually shows it)
3. **cudnn Version** - Run `nvidia-smi` and check "CUDA Version" field

---

## Installation Steps

### Step 1: Create Conda Environment

```bash
# Create new environment with Python 3.10
conda create -n eeg2img python=3.10

# Activate the environment
conda activate eeg2img
```

### Step 2: Install PyTorch with GPU Support

**Choose your CUDA version and run the appropriate command:**

#### Option A: CUDA 12.1 (Recommended for newer GPUs)

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

#### Option B: CUDA 11.8 (For older GPUs or systems)

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### Option C: CPU Only (No GPU - slower but works)

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

**Installation will take 5-15 minutes** (downloading 2-3 GB of packages)

### Step 3: Install Project Dependencies

```bash
# Navigate to project root
cd d:\EEG2Img-Benchmark-Study

# Install remaining dependencies
pip install -r requirements.txt
```

### Step 4: Verify GPU Installation

```bash
python -c "
import torch
print('PyTorch Version:', torch.__version__)
print('GPU Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU Name:', torch.cuda.get_device_name(0))
    print('GPU Count:', torch.cuda.device_count())
    print('GPU Memory:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')
else:
    print('WARNING: GPU not available - using CPU (will be slow)')
"
```

**Expected output with GPU:**
```
PyTorch Version: 2.0.0+cu121
GPU Available: True
GPU Name: NVIDIA GeForce RTX 3080
GPU Count: 1
GPU Memory: 10.0 GB
```

**Expected output without GPU:**
```
PyTorch Version: 2.0.0+cpu
GPU Available: False
WARNING: GPU not available - using CPU (will be slow)
```

---

## Troubleshooting

### Problem 1: "GPU Available: False"

**Possible Causes:**
1. Wrong CUDA version installed
2. NVIDIA drivers not updated
3. CUDA not in system PATH

**Solutions:**

**A) Update NVIDIA Drivers:**
```bash
# Visit: https://www.nvidia.com/Download/driverDetails.aspx
# Find your GPU model
# Download and install latest driver
# Restart computer
```

**B) Install CUDA Toolkit:**
```bash
# Visit: https://developer.nvidia.com/cuda-downloads
# Download CUDA 12.1 (or matching your PyTorch version)
# Run installer and follow prompts
# Restart computer
```

**C) Reinstall PyTorch with Correct Version:**
```bash
# First remove old PyTorch
conda remove pytorch torchvision torchaudio pytorch-cuda -y

# Then reinstall with correct CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Problem 2: "ModuleNotFoundError: No module named 'torch'"

**Solution:**
1. Make sure conda environment is activated: `conda activate eeg2img`
2. Check PyTorch is installed: `pip list | grep torch`
3. Reinstall if needed: `pip install torch torchvision torchaudio`

### Problem 3: CUDA Out of Memory During Training

**This is normal and happens when:**
- Batch size is too large
- Model is too large for GPU memory
- Running multiple models simultaneously

**Solutions:**
1. **Reduce batch size** in config:
   ```yaml
   training:
     batch_size: 16  # Reduce from 32
   ```

2. **Use gradient accumulation** in config:
   ```yaml
   training:
     gradient_accumulation_steps: 2  # Simulate larger batch
   ```

3. **Clear GPU cache** before training:
   ```bash
   python -c "import torch; torch.cuda.empty_cache()"
   ```

### Problem 4: "nvcc: command not found"

**This means CUDA toolkit is not in PATH**

**Solution:**
```bash
# Add CUDA to PATH (Windows)
# System Properties → Environment Variables → Add:
# C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin
# C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\libnvvm\bin

# Restart Command Prompt and try again
```

---

## Verifying GPU Performance

### Test 1: GPU Utilization During Training

During Phase 5 training, you should see GPU usage:

```bash
# Open second terminal and run (while training):
nvidia-smi -l 1
```

**Good output (GPU in use):**
```
| GPU  Name    |  Temp  | Power | Memory-Usage |
|  0   RTX3080 |  65°C  | 250W | 8000MiB      |
```

**Bad output (GPU idle):**
```
| GPU  Name    |  Temp  | Power | Memory-Usage |
|  0   RTX3080 |  30°C  |  5W  | 100MiB       |
```

### Test 2: Training Speed Comparison

**Phase 5 training one model:**
- **CPU only:** 2-3 hours
- **GPU enabled:** 3-5 minutes
- **Speedup:** 20-40×

---

## Optimizing GPU Performance

### 1. Enable Mixed Precision Training

Already enabled in Phase 5 config:
```yaml
training:
  mixed_precision: true  # Uses FP16 + FP32 for ~2× speedup
```

### 2. Use Gradient Accumulation

For larger effective batch size on limited VRAM:
```yaml
training:
  gradient_accumulation_steps: 2
```

### 3. Enable Cudnn Benchmarking

Add to training script for automatic optimization:
```python
import torch
torch.backends.cudnn.benchmark = True
```

### 4. Monitor GPU Memory

During training:
```bash
# Watch GPU memory in real-time
watch nvidia-smi
```

---

## GPU Memory Requirements by Model

| Model | VRAM Required | Batch Size | Note |
|-------|---------------|-----------|------|
| ResNet-18 | 2 GB | 64 | Can use larger batches |
| ResNet-50 | 4 GB | 32 | Standard batch size |
| ViT-Tiny | 2 GB | 64 | Efficient |
| ViT-Small | 4 GB | 32 | Standard batch size |
| ViT-Base | 8 GB | 16 | Requires larger GPU |
| EEGNet | 1 GB | 128 | Very efficient |

---

## Conda GPU Environment Commands

```bash
# Create GPU environment
conda create -n eeg2img python=3.10
conda activate eeg2img

# Install PyTorch with GPU
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install project dependencies
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print('GPU:', torch.cuda.is_available())"

# List all conda environments
conda env list

# Activate environment before work
conda activate eeg2img

# Deactivate when done
conda deactivate

# Remove environment (if needed)
conda remove -n eeg2img --all
```

---

## Quick Start Checklist

- [ ] GPU detected: `nvidia-smi` shows your GPU
- [ ] Conda environment created: `conda create -n eeg2img python=3.10`
- [ ] Environment activated: `conda activate eeg2img` (shows `(eeg2img)` in prompt)
- [ ] PyTorch installed with GPU: `pip list | grep torch` (shows `+cu121` or `+cu118`)
- [ ] Verification passed: `torch.cuda.is_available()` returns `True`
- [ ] Dependencies installed: `pip install -r requirements.txt` completes
- [ ] All tests pass: Run Phase 4 model validation

---

## Performance Expectations

### Phase 5: Training One Model

**Without GPU (CPU only):**
- Time: 2-3 hours
- Memory: 16+ GB RAM needed
- Not recommended

**With GPU (CUDA 12.1, RTX 3080):**
- Time: 3-5 minutes
- Memory: 8-10 GB VRAM
- **Recommended** ✅

### Phase 7: Grid Search (9 Models)

**Without GPU:**
- Time: 18-24 hours
- Not practical

**With GPU:**
- Time: 30-45 minutes
- Practical and efficient ✅

---

## Additional Resources

- **PyTorch Official:** https://pytorch.org/get-started/locally/
- **CUDA Downloads:** https://developer.nvidia.com/cuda-downloads
- **NVIDIA Drivers:** https://www.nvidia.com/Download/index.aspx
- **cuDNN:** https://developer.nvidia.com/cudnn

---

## Support

If you encounter issues:

1. **Check GPU availability:** `nvidia-smi`
2. **Check PyTorch setup:** `python -c "import torch; print(torch.cuda.is_available())"`
3. **Check CUDA version:** `nvcc --version`
4. **Review this guide:** Check "Troubleshooting" section
5. **Update drivers:** Install latest NVIDIA drivers
6. **Reinstall PyTorch:** Remove and reinstall with correct CUDA version

---

**GPU Setup Complete!** ✅

You're now ready to run Phase 5 training with GPU acceleration.

Expected speedup: **20-40× faster training**
