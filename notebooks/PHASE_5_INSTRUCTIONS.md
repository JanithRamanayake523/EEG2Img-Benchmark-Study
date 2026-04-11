# Phase 5: Training Infrastructure

## Overview

Phase 5 implements the complete training pipeline including augmentation, optimization, callbacks, and cross-validation strategies.

**Commit:** `4dd1c47`
**Status:** ✅ Complete
**Estimated Runtime:** 1-2 hours per model (full training)

---

## What This Phase Does

### 1. Data Augmentation
Techniques to artificially increase training data diversity and improve generalization.

#### Image Augmentation
```python
class ImageAugmentation:
    ├─ Rotation: ±5 degrees
    ├─ Shifts: ±5% horizontal/vertical
    ├─ Gaussian Noise: σ ∈ [0.01, 0.05]
    ├─ Intensity Scaling: ×[0.9, 1.1]
    └─ Brightness/Contrast: ±10%

Example:
  Original image → Apply random rotation
                → Apply random shift
                → Apply random noise
                → Output: Augmented image (different each epoch)
```

#### MixUp Augmentation
```
Combines two training samples linearly:
  x_mix = λ·x_i + (1-λ)·x_j
  y_mix = λ·y_i + (1-λ)·y_j

where λ ~ Beta(α, α), α=0.2

Benefits:
├─ Smoother decision boundaries
├─ Better generalization
├─ Regularization effect
└─ 2-3% accuracy improvement typical
```

#### CutMix Augmentation
```
Mixes samples via region replacement:
  ├─ Select random rectangular region in image
  ├─ Replace with corresponding region from another image
  ├─ Soft label mixing

Mask M ∈ {0, 1} for mixed region:
  x_cutmix = M·x_i + (1-M)·x_j
  y_cutmix = λ·y_i + (1-λ)·y_j

Benefits:
├─ Learns region importance
├─ Handles occlusion
├─ Data-dependent regularization
└─ 2-5% improvement possible
```

#### Time-Series Augmentation
For raw signal models:
```python
class TimeSeriesAugmentation:
    ├─ Jitter: Add small random noise (σ=0.01)
    ├─ Scaling: Multiply by [0.9, 1.1]
    ├─ Magnitude Warping: Smooth random walk on amplitude
    ├─ Time Warping: DTW-based temporal distortion
    └─ Window Slicing: Extract random window
```

### 2. Training Loop

#### Basic Training Step
```python
for epoch in range(num_epochs):
    # Training phase
    for batch_X, batch_y in train_loader:
        # Forward pass
        logits = model(batch_X)
        loss = criterion(logits, batch_y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        track_loss(loss)
        track_accuracy(logits, batch_y)

    # Validation phase
    for batch_X, batch_y in val_loader:
        with torch.no_grad():
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            track_val_loss(loss)
            track_val_accuracy(logits, batch_y)

    # Callbacks (early stopping, checkpointing, LR scheduling)
    early_stopping(val_loss)
    if val_loss improved:
        save_checkpoint(model)
```

#### Mixed Precision Training (AMP)
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(num_epochs):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()

        # Mixed precision: FP16 forward pass
        with autocast():
            logits = model(batch_X)
            loss = criterion(logits, batch_y)

        # Scale loss and backward (FP32)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

Benefits:
├─ 2-3× memory reduction
├─ 1.5-2× faster training
├─ Minimal accuracy loss
└─ Better GPU utilization
```

#### Gradient Accumulation
```python
# Simulate larger batch size without OOM

accumulation_steps = 4  # Accumulate 4 gradient steps
effective_batch_size = batch_size * accumulation_steps

for i, (batch_X, batch_y) in enumerate(train_loader):
    logits = model(batch_X)
    loss = criterion(logits, batch_y)

    # Scale loss by accumulation steps
    loss = loss / accumulation_steps
    loss.backward()  # Accumulate gradients

    # Step optimizer every accumulation_steps
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

Benefits:
├─ Effective batch size increased
├─ Better convergence for small batches
├─ Memory efficient
└─ Simulate larger GPUs
```

### 3. Callbacks

#### Early Stopping
```python
class EarlyStopping:
    def __init__(self, patience=5, metric='val_loss', mode='min'):
        self.patience = patience
        self.counter = 0
        self.best_value = None

    def __call__(self, current_value):
        if self.best_value is None:
            self.best_value = current_value
        elif self.improved(current_value):
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Stop training
        return False
```

Usage:
```
Epoch 1: val_loss = 1.234
Epoch 2: val_loss = 1.100 ✓ improved, reset counter
Epoch 3: val_loss = 1.105 ✗ counter = 1
Epoch 4: val_loss = 1.110 ✗ counter = 2
Epoch 5: val_loss = 1.115 ✗ counter = 3
Epoch 6: val_loss = 1.120 ✗ counter = 4
Epoch 7: val_loss = 1.125 ✗ counter = 5 → STOP training
```

Result: Save model from Epoch 2 (best validation loss)

#### Model Checkpointing
```python
# Save best model during training
if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'loss': val_loss,
        'metrics': metrics
    }, checkpoint_path)
```

#### Learning Rate Scheduling
```python
# Option 1: ReduceLROnPlateau
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3
)

for epoch in range(num_epochs):
    train(...)
    val_loss = validate(...)
    scheduler.step(val_loss)  # Reduce LR if val_loss plateaus

# Example: LR = 0.001 → 0.0005 → 0.00025 → ...

# Option 2: Cosine Annealing
scheduler = CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(num_epochs):
    train(...)
    scheduler.step()  # Gradually decay LR as cos

# Learning rate follows: η(t) = η_min + (η_max - η_min) * (1 + cos(πt/T)) / 2
```

### 4. Cross-Validation Strategies

#### Stratified K-Fold
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_results = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold+1}/5:")

    # Split data
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate
    metrics = evaluate(model, X_val, y_val)
    fold_results.append(metrics)

    print(f"  Accuracy: {metrics['accuracy']:.4f}")

# Aggregate results
mean_acc = np.mean([r['accuracy'] for r in fold_results])
std_acc = np.std([r['accuracy'] for r in fold_results])
print(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
```

#### Leave-One-Subject-Out (LOSO)
```python
# For BCI data: test generalization across subjects

for subject_idx in range(n_subjects):
    # Training: all subjects except current
    train_subjects = [s for s in range(n_subjects) if s != subject_idx]
    X_train = X[subjects in train_subjects]
    y_train = y[subjects in train_subjects]

    # Validation: current subject
    X_test = X[subjects == subject_idx]
    y_test = y[subjects == subject_idx]

    # Train and evaluate
    model = train_model(X_train, y_train)
    acc = evaluate(model, X_test, y_test)['accuracy']
    print(f"Subject {subject_idx}: {acc:.4f}")

# Result: Subject-independent performance
```

### 5. Hyperparameter Defaults

```
ResNet-18/50:
├─ Learning rate: 0.001
├─ Batch size: 32
├─ Epochs: 100
├─ Weight decay: 1e-4
├─ Dropout: 0.5
├─ Optimizer: Adam
└─ Loss: CrossEntropyLoss

ViT-Tiny/Small/Base:
├─ Learning rate: 5e-5 (lower for transformers)
├─ Batch size: 16 (smaller due to memory)
├─ Epochs: 100
├─ Weight decay: 1e-4
├─ Dropout: 0.1
├─ Optimizer: Adam
├─ Warmup epochs: 5
└─ Loss: CrossEntropyLoss

1D CNN/LSTM/Transformer/EEGNet:
├─ Learning rate: 1e-3
├─ Batch size: 32
├─ Epochs: 100
├─ Weight decay: 1e-4
├─ Dropout: 0.5
├─ Optimizer: Adam
└─ Loss: CrossEntropyLoss
```

---

## Training Script Example

### Script Usage
```bash
python -m src.training.trainer \
    --model resnet18 \
    --data data/transformed/gaf_summation_subject_001.hdf5 \
    --output results/resnet18_gaf/ \
    --epochs 100 \
    --batch_size 32 \
    --augmentation mixup cutmix
```

### Expected Training Output
```
Training ResNet-18 on GAF images (fold 1/5)
═════════════════════════════════════════════════════════════════

Configuration:
├─ Model: ResNet-18
├─ Data: GAF transformed images (64×64)
├─ Samples: 952 train, 119 val, 117 test
├─ Augmentation: MixUp + CutMix
├─ Learning rate: 0.001
├─ Batch size: 32
└─ Epochs: 100

Training Progress:
Epoch 1/100   [████░░░░░░░░░░░░░░░] loss: 1.387 val_loss: 1.234 acc: 25.5% val_acc: 28.6% ✓
Epoch 2/100   [████░░░░░░░░░░░░░░░] loss: 1.156 val_loss: 0.987 acc: 45.3% val_acc: 48.2% ✓ (best)
Epoch 3/100   [████░░░░░░░░░░░░░░░] loss: 0.943 val_loss: 0.876 acc: 58.2% val_acc: 60.1% ✓ (best)
...
Epoch 50/100  [█████████████░░░░░░░] loss: 0.089 val_loss: 0.234 acc: 95.6% val_acc: 91.2%
Epoch 51/100  [█████████████░░░░░░░] loss: 0.076 val_loss: 0.243 acc: 96.2% val_acc: 90.8%
Epoch 52/100  [█████████████░░░░░░░] loss: 0.068 val_loss: 0.256 acc: 96.8% val_acc: 90.4%
Epoch 53/100  [█████████████░░░░░░░] loss: 0.063 val_loss: 0.271 acc: 97.1% val_acc: 89.9% (counter: 1)
Epoch 54/100  [█████████████░░░░░░░] loss: 0.058 val_loss: 0.287 acc: 97.4% val_acc: 89.2% (counter: 2)
...
Epoch 60/100  [██████████████░░░░░░] loss: 0.031 val_loss: 0.345 acc: 98.5% val_acc: 88.1% (counter: 8)
Epoch 61/100  [██████████████░░░░░░] loss: 0.027 val_loss: 0.358 acc: 98.7% val_acc: 87.6% (counter: 9)
Epoch 62/100  [██████████████░░░░░░] loss: 0.024 val_loss: 0.372 acc: 98.9% val_acc: 87.1% (counter: 10)
→ EARLY STOPPING (patience reached)

Final Results:
├─ Best epoch: 50
├─ Best val_loss: 0.234
├─ Test accuracy: 92.3%
├─ Test F1-score: 0.921
├─ Training time: 8m 45s
├─ Model saved: results/resnet18_gaf/best_model.pt
└─ Metrics saved: results/resnet18_gaf/metrics.json
```

---

## Test Validation (`experiments/scripts/test_training.py`)

```
Test: ImageAugmentation                                PASSED ✓
Test: TimeSeriesAugmentation                           PASSED ✓
Test: MixUp                                            PASSED ✓
Test: CutMix                                           PASSED ✓
Test: EarlyStopping                                    PASSED ✓
Test: ModelCheckpoint                                  PASSED ✓
Test: LearningRateScheduler                            PASSED ✓
Test: Trainer forward pass                             PASSED ✓
Test: Trainer training loop (1 epoch)                  PASSED ✓
Test: Prediction functionality                         PASSED ✓
═════════════════════════════════════════════════════════════════
SUMMARY: All 10 training tests PASSED (100%)
```

---

## Training Outputs

### Files Saved
```
results/resnet18_gaf/
├── best_model.pt              # Best model weights
├── training_history.json      # Loss/accuracy per epoch
├── metrics.json              # Final test metrics
├── config.yaml               # Training configuration
└── logs/                     # Training logs
    └── training_YYYYMMDD.log
```

### Metrics JSON Format
```json
{
  "model": "resnet18",
  "transformation": "gaf_summation",
  "fold": 1,
  "train_metrics": {
    "accuracy": 0.989,
    "f1": 0.988,
    "loss": 0.024
  },
  "val_metrics": {
    "accuracy": 0.912,
    "f1": 0.910,
    "loss": 0.234
  },
  "test_metrics": {
    "accuracy": 0.923,
    "f1": 0.921,
    "auc": 0.975,
    "kappa": 0.897,
    "mcc": 0.894
  },
  "training_time": "8m45s",
  "best_epoch": 50,
  "total_epochs": 62,
  "early_stopped": true,
  "augmentation": ["mixup", "cutmix"]
}
```

---

## Phase 5 Checklist

- ✅ **Image Augmentation** - Rotation, shifts, noise, scaling
- ✅ **MixUp** - Linear mixing of samples (α=0.2)
- ✅ **CutMix** - Region-based mixing
- ✅ **Time-Series Augmentation** - Jitter, scaling, warping
- ✅ **Training Loop** - Forward/backward/optimize
- ✅ **Mixed Precision Training** - AMP with GradScaler
- ✅ **Gradient Accumulation** - Larger effective batch sizes
- ✅ **Early Stopping** - Patience-based termination
- ✅ **Model Checkpointing** - Save best model
- ✅ **Learning Rate Scheduling** - ReduceLROnPlateau + Cosine
- ✅ **Stratified K-Fold** - Balanced cross-validation
- ✅ **LOSO Cross-Validation** - Subject-independent validation
- ✅ **Validation** - All tests passing (100%)

---

## Key Takeaways

| Component | Details |
|-----------|---------|
| **Augmentation Methods** | 4 strategies (MixUp, CutMix, geometric, time-series) |
| **Expected Improvements** | 2-5% from augmentation |
| **Training Strategies** | Mixed precision, gradient accumulation |
| **Callbacks** | Early stopping, checkpointing, LR scheduling |
| **CV Strategies** | 5-fold, LOSO for BCI data |
| **Test Pass Rate** | 100% |

---

**Phase 5 Status:** ✅ COMPLETE AND VERIFIED

Complete training infrastructure is implemented and ready. All augmentation strategies, optimization techniques, and callbacks are working correctly.
