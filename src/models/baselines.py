"""
Baseline models for raw EEG time-series classification.

Implements models that operate directly on 1D time-series data without
image transformation, serving as baselines for comparison.

Models:
- 1D CNN: Temporal convolutions
- LSTM/BiLSTM: Recurrent models
- Transformer: Self-attention on sequences
- EEGNet: Specialized CNN for EEG

References:
- Lawhern et al. (2018): EEGNet - Compact CNN for EEG-based BCIs
- Schirrmeister et al. (2017): Deep learning with convolutional neural networks for EEG
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional


class CNN1D(nn.Module):
    """
    1D CNN for raw EEG time-series classification.

    Uses temporal convolutions to extract features directly from
    multi-channel time-series data.
    """

    def __init__(self, num_classes: int = 4,
                 in_channels: int = 25,
                 seq_length: int = 751,
                 base_filters: int = 64,
                 dropout: float = 0.5):
        """
        Initialize 1D CNN.

        Args:
            num_classes: Number of output classes
            in_channels: Number of EEG channels
            seq_length: Length of time sequence
            base_filters: Number of filters in first conv layer
            dropout: Dropout rate

        Example:
            >>> model = CNN1D(num_classes=4, in_channels=25, seq_length=751)
            >>> x = torch.randn(8, 25, 751)  # (batch, channels, time)
            >>> out = model(x)  # (8, 4)
        """
        super(CNN1D, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.seq_length = seq_length

        # Temporal convolution blocks
        self.conv1 = self._make_conv_block(in_channels, base_filters, kernel_size=25)
        self.conv2 = self._make_conv_block(base_filters, base_filters * 2, kernel_size=15)
        self.conv3 = self._make_conv_block(base_filters * 2, base_filters * 4, kernel_size=9)
        self.conv4 = self._make_conv_block(base_filters * 4, base_filters * 8, kernel_size=5)

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(base_filters * 8, num_classes)
        )

    def _make_conv_block(self, in_channels, out_channels, kernel_size):
        """Create conv block: Conv1d -> BatchNorm -> ReLU -> MaxPool."""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, in_channels, seq_length)

        Returns:
            Output logits (batch_size, num_classes)
        """
        x = self.conv1(x)  # Length / 2
        x = self.conv2(x)  # Length / 4
        x = self.conv3(x)  # Length / 8
        x = self.conv4(x)  # Length / 16

        x = self.global_pool(x)  # (batch, filters, 1)
        x = torch.flatten(x, 1)  # (batch, filters)
        x = self.classifier(x)   # (batch, num_classes)

        return x


class LSTMClassifier(nn.Module):
    """
    LSTM/BiLSTM for raw EEG time-series classification.

    Uses recurrent layers to model temporal dependencies.
    """

    def __init__(self, num_classes: int = 4,
                 in_channels: int = 25,
                 seq_length: int = 751,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 bidirectional: bool = True,
                 dropout: float = 0.5):
        """
        Initialize LSTM classifier.

        Args:
            num_classes: Number of output classes
            in_channels: Number of EEG channels (input features)
            seq_length: Length of time sequence
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            bidirectional: Use bidirectional LSTM
            dropout: Dropout rate

        Example:
            >>> model = LSTMClassifier(num_classes=4, in_channels=25, bidirectional=True)
            >>> x = torch.randn(8, 25, 751)  # (batch, channels, time)
            >>> out = model(x)  # (8, 4)
        """
        super(LSTMClassifier, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )

        # Classifier
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(lstm_output_size, num_classes)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, in_channels, seq_length)

        Returns:
            Output logits (batch_size, num_classes)
        """
        # LSTM expects (batch, seq, features)
        x = x.transpose(1, 2)  # (batch, seq_length, in_channels)

        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use final hidden state
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            h_final = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            h_final = h_n[-1]

        # Classify
        out = self.classifier(h_final)

        return out


class TransformerClassifier(nn.Module):
    """
    Transformer for raw EEG time-series classification.

    Uses self-attention mechanism to capture temporal dependencies.
    """

    def __init__(self, num_classes: int = 4,
                 in_channels: int = 25,
                 seq_length: int = 751,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        """
        Initialize Transformer classifier.

        Args:
            num_classes: Number of output classes
            in_channels: Number of EEG channels
            seq_length: Length of time sequence
            d_model: Transformer embedding dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout rate

        Example:
            >>> model = TransformerClassifier(num_classes=4, in_channels=25)
            >>> x = torch.randn(8, 25, 751)  # (batch, channels, time)
            >>> out = model(x)  # (8, 4)
        """
        super(TransformerClassifier, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.seq_length = seq_length
        self.d_model = d_model

        # Input projection
        self.input_projection = nn.Linear(in_channels, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=seq_length)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, in_channels, seq_length)

        Returns:
            Output logits (batch_size, num_classes)
        """
        # Transformer expects (batch, seq, features)
        x = x.transpose(1, 2)  # (batch, seq_length, in_channels)

        # Project to d_model
        x = self.input_projection(x)  # (batch, seq_length, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer_encoder(x)  # (batch, seq_length, d_model)

        # Global average pooling over time
        x = x.mean(dim=1)  # (batch, d_model)

        # Classify
        out = self.classifier(x)

        return out


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class EEGNet(nn.Module):
    """
    EEGNet: Compact CNN for EEG-based BCIs.

    Specialized architecture designed for EEG signals with
    depthwise and separable convolutions.

    Reference:
    Lawhern et al. (2018): EEGNet: a compact convolutional neural network
    for EEG-based brain-computer interfaces.
    """

    def __init__(self, num_classes: int = 4,
                 in_channels: int = 25,
                 seq_length: int = 751,
                 sfreq: int = 250,
                 F1: int = 8,
                 D: int = 2,
                 F2: int = 16,
                 dropout: float = 0.5):
        """
        Initialize EEGNet.

        Args:
            num_classes: Number of output classes
            in_channels: Number of EEG channels
            seq_length: Length of time sequence
            sfreq: Sampling frequency (Hz)
            F1: Number of temporal filters
            D: Depth multiplier (spatial filters per temporal filter)
            F2: Number of pointwise filters
            dropout: Dropout rate

        Example:
            >>> model = EEGNet(num_classes=4, in_channels=25, seq_length=751)
            >>> x = torch.randn(8, 1, 25, 751)  # (batch, 1, channels, time)
            >>> out = model(x)  # (8, 4)
        """
        super(EEGNet, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.seq_length = seq_length
        self.F1 = F1
        self.D = D
        self.F2 = F2

        # Block 1: Temporal convolution
        self.conv1 = nn.Conv2d(1, F1, kernel_size=(1, sfreq // 2), padding=(0, sfreq // 4), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        # Block 2: Depthwise spatial convolution
        self.conv2 = nn.Conv2d(F1, F1 * D, kernel_size=(in_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout2 = nn.Dropout(p=dropout)

        # Block 3: Separable convolution
        self.conv3 = nn.Conv2d(F1 * D, F2, kernel_size=(1, 16), padding=(0, 8), bias=False, groups=F1 * D)
        self.conv4 = nn.Conv2d(F2, F2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.elu3 = nn.ELU()
        self.pool3 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout3 = nn.Dropout(p=dropout)

        # Classifier
        # Calculate output size after convolutions/pooling
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, in_channels, seq_length)
            dummy_output = self._forward_features(dummy_input)
            num_features = dummy_output.numel()

        self.classifier = nn.Linear(num_features, num_classes)

    def _forward_features(self, x):
        """Extract features."""
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Block 3
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.elu3(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        return x

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, 1, in_channels, seq_length)
               Note: EEGNet expects 4D input with channel dimension = 1

        Returns:
            Output logits (batch_size, num_classes)
        """
        # Extract features
        x = self._forward_features(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Classify
        out = self.classifier(x)

        return out


def get_baseline_model(
    model_name: Literal['cnn1d', 'lstm', 'bilstm', 'transformer', 'eegnet'],
    num_classes: int = 4,
    in_channels: int = 25,
    seq_length: int = 751,
    **kwargs
) -> nn.Module:
    """
    Factory function to get baseline model by name.

    Args:
        model_name: Model type
        num_classes: Number of output classes
        in_channels: Number of EEG channels
        seq_length: Length of time sequence
        **kwargs: Additional model-specific arguments

    Returns:
        Baseline model instance

    Example:
        >>> model = get_baseline_model('cnn1d', num_classes=4, in_channels=25)
        >>> model = get_baseline_model('bilstm', num_classes=4, hidden_size=128)

    Raises:
        ValueError: If model_name not recognized
    """
    if model_name == 'cnn1d':
        return CNN1D(num_classes, in_channels, seq_length, **kwargs)
    elif model_name == 'lstm':
        return LSTMClassifier(num_classes, in_channels, seq_length, bidirectional=False, **kwargs)
    elif model_name == 'bilstm':
        return LSTMClassifier(num_classes, in_channels, seq_length, bidirectional=True, **kwargs)
    elif model_name == 'transformer':
        return TransformerClassifier(num_classes, in_channels, seq_length, **kwargs)
    elif model_name == 'eegnet':
        return EEGNet(num_classes, in_channels, seq_length, **kwargs)
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: cnn1d, lstm, bilstm, transformer, eegnet"
        )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test models
    print("="*60)
    print("Testing Baseline Models")
    print("="*60)

    batch_size = 8
    in_channels = 25
    seq_length = 751
    num_classes = 4

    # Test data for 1D models
    x_1d = torch.randn(batch_size, in_channels, seq_length)
    print(f"\nInput shape (1D): {x_1d.shape}")

    # Test 1D CNN
    print("\n--- 1D CNN ---")
    model = CNN1D(num_classes, in_channels, seq_length)
    out = model(x_1d)
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {count_parameters(model):,}")

    # Test LSTM
    print("\n--- LSTM ---")
    model = LSTMClassifier(num_classes, in_channels, seq_length, bidirectional=False)
    out = model(x_1d)
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {count_parameters(model):,}")

    # Test BiLSTM
    print("\n--- BiLSTM ---")
    model = LSTMClassifier(num_classes, in_channels, seq_length, bidirectional=True)
    out = model(x_1d)
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {count_parameters(model):,}")

    # Test Transformer
    print("\n--- Transformer ---")
    model = TransformerClassifier(num_classes, in_channels, seq_length)
    out = model(x_1d)
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {count_parameters(model):,}")

    # Test EEGNet (needs 4D input)
    print("\n--- EEGNet ---")
    x_4d = torch.randn(batch_size, 1, in_channels, seq_length)
    print(f"Input shape (4D): {x_4d.shape}")
    model = EEGNet(num_classes, in_channels, seq_length)
    out = model(x_4d)
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {count_parameters(model):,}")

    # Test factory function
    print("\n--- Factory Function ---")
    model = get_baseline_model('cnn1d', num_classes, in_channels, seq_length)
    out = model(x_1d)
    print(f"1D CNN via factory: {out.shape}")

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
