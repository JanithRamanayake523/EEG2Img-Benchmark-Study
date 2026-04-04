"""
CNN architectures for EEG image classification.

Implements ResNet variants and a lightweight custom CNN for processing
transformed EEG images.

References:
- He et al. (2016): Deep Residual Learning for Image Recognition
- Standard ResNet architectures adapted for EEG image classification
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Literal


class ResNet18EEG(nn.Module):
    """
    ResNet-18 architecture adapted for EEG images.

    Uses ImageNet-pretrained weights with modified first conv layer
    to handle multi-channel EEG inputs (e.g., 25 channels).
    """

    def __init__(self, num_classes: int = 4,
                 in_channels: int = 25,
                 pretrained: bool = True,
                 dropout: float = 0.5):
        """
        Initialize ResNet-18 for EEG classification.

        Args:
            num_classes: Number of output classes
            in_channels: Number of input channels (EEG channels or 1 for single image)
            pretrained: Use ImageNet pretrained weights
            dropout: Dropout rate before final FC layer

        Example:
            >>> model = ResNet18EEG(num_classes=4, in_channels=25)
            >>> x = torch.randn(8, 25, 64, 64)  # (batch, channels, H, W)
            >>> out = model(x)  # (8, 4)
        """
        super(ResNet18EEG, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels

        # Load pretrained ResNet-18
        if pretrained:
            self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.resnet = models.resnet18(weights=None)

        # Modify first conv layer to accept in_channels
        if in_channels != 3:
            self.resnet.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            # If pretrained, initialize new conv1 by averaging RGB weights
            if pretrained:
                with torch.no_grad():
                    # Get pretrained weights
                    pretrained_resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                    pretrained_weight = pretrained_resnet.conv1.weight  # (64, 3, 7, 7)

                    # Repeat or average to match in_channels
                    if in_channels < 3:
                        # Average RGB channels
                        new_weight = pretrained_weight.mean(dim=1, keepdim=True)
                        self.resnet.conv1.weight.copy_(new_weight.repeat(1, in_channels, 1, 1))
                    else:
                        # Repeat RGB channels and normalize
                        repeats = (in_channels // 3) + 1
                        new_weight = pretrained_weight.repeat(1, repeats, 1, 1)[:, :in_channels, :, :]
                        new_weight = new_weight * (3.0 / in_channels)  # Normalize
                        self.resnet.conv1.weight.copy_(new_weight)

        # Modify final FC layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, in_channels, H, W)

        Returns:
            Output logits (batch_size, num_classes)
        """
        return self.resnet(x)


class ResNet50EEG(nn.Module):
    """
    ResNet-50 architecture adapted for EEG images.

    Deeper variant with bottleneck blocks for more complex patterns.
    """

    def __init__(self, num_classes: int = 4,
                 in_channels: int = 25,
                 pretrained: bool = True,
                 dropout: float = 0.5):
        """
        Initialize ResNet-50 for EEG classification.

        Args:
            num_classes: Number of output classes
            in_channels: Number of input channels
            pretrained: Use ImageNet pretrained weights
            dropout: Dropout rate before final FC layer

        Example:
            >>> model = ResNet50EEG(num_classes=4, in_channels=25)
            >>> x = torch.randn(8, 25, 128, 128)
            >>> out = model(x)  # (8, 4)
        """
        super(ResNet50EEG, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels

        # Load pretrained ResNet-50
        if pretrained:
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.resnet = models.resnet50(weights=None)

        # Modify first conv layer
        if in_channels != 3:
            self.resnet.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            if pretrained:
                with torch.no_grad():
                    pretrained_resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                    pretrained_weight = pretrained_resnet.conv1.weight

                    if in_channels < 3:
                        new_weight = pretrained_weight.mean(dim=1, keepdim=True)
                        self.resnet.conv1.weight.copy_(new_weight.repeat(1, in_channels, 1, 1))
                    else:
                        repeats = (in_channels // 3) + 1
                        new_weight = pretrained_weight.repeat(1, repeats, 1, 1)[:, :in_channels, :, :]
                        new_weight = new_weight * (3.0 / in_channels)
                        self.resnet.conv1.weight.copy_(new_weight)

        # Modify final FC layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        """Forward pass."""
        return self.resnet(x)


class LightweightCNN(nn.Module):
    """
    Lightweight custom CNN for EEG image classification.

    Designed for faster training and inference with fewer parameters
    than ResNet architectures.

    Architecture:
    - 4 convolutional blocks with batch norm and max pooling
    - Global average pooling
    - Dropout + FC layer
    """

    def __init__(self, num_classes: int = 4,
                 in_channels: int = 25,
                 base_filters: int = 32,
                 dropout: float = 0.5):
        """
        Initialize Lightweight CNN.

        Args:
            num_classes: Number of output classes
            in_channels: Number of input channels
            base_filters: Number of filters in first conv layer (doubles each block)
            dropout: Dropout rate before final FC layer

        Example:
            >>> model = LightweightCNN(num_classes=4, in_channels=25)
            >>> x = torch.randn(8, 25, 64, 64)
            >>> out = model(x)  # (8, 4)
        """
        super(LightweightCNN, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels

        # Convolutional blocks
        self.conv_block1 = self._make_conv_block(in_channels, base_filters)
        self.conv_block2 = self._make_conv_block(base_filters, base_filters * 2)
        self.conv_block3 = self._make_conv_block(base_filters * 2, base_filters * 4)
        self.conv_block4 = self._make_conv_block(base_filters * 4, base_filters * 8)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(base_filters * 8, num_classes)
        )

    def _make_conv_block(self, in_channels, out_channels):
        """Create a conv block: Conv -> BatchNorm -> ReLU -> MaxPool."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, in_channels, H, W)

        Returns:
            Output logits (batch_size, num_classes)
        """
        x = self.conv_block1(x)  # H/2, W/2
        x = self.conv_block2(x)  # H/4, W/4
        x = self.conv_block3(x)  # H/8, W/8
        x = self.conv_block4(x)  # H/16, W/16

        x = self.global_pool(x)  # (batch, filters, 1, 1)
        x = torch.flatten(x, 1)  # (batch, filters)
        x = self.classifier(x)   # (batch, num_classes)

        return x


def get_cnn_model(
    model_name: Literal['resnet18', 'resnet50', 'lightweight'],
    num_classes: int = 4,
    in_channels: int = 25,
    pretrained: bool = True,
    dropout: float = 0.5
) -> nn.Module:
    """
    Factory function to get CNN model by name.

    Args:
        model_name: Model architecture ('resnet18', 'resnet50', 'lightweight')
        num_classes: Number of output classes
        in_channels: Number of input channels
        pretrained: Use ImageNet pretrained weights (ResNets only)
        dropout: Dropout rate

    Returns:
        CNN model instance

    Example:
        >>> model = get_cnn_model('resnet18', num_classes=4, in_channels=25)
        >>> model = get_cnn_model('lightweight', num_classes=4, in_channels=1)

    Raises:
        ValueError: If model_name not recognized
    """
    models_dict = {
        'resnet18': ResNet18EEG,
        'resnet50': ResNet50EEG,
        'lightweight': LightweightCNN,
    }

    if model_name not in models_dict:
        available = ', '.join(models_dict.keys())
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {available}"
        )

    model_class = models_dict[model_name]

    # LightweightCNN doesn't have pretrained parameter
    if model_name == 'lightweight':
        return model_class(
            num_classes=num_classes,
            in_channels=in_channels,
            dropout=dropout
        )
    else:
        return model_class(
            num_classes=num_classes,
            in_channels=in_channels,
            pretrained=pretrained,
            dropout=dropout
        )


def count_parameters(model: nn.Module) -> int:
    """
    Count total and trainable parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters

    Example:
        >>> model = ResNet18EEG()
        >>> print(f"Parameters: {count_parameters(model):,}")
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test models
    print("="*60)
    print("Testing CNN Models")
    print("="*60)

    batch_size = 8
    in_channels = 25
    image_size = 64
    num_classes = 4

    # Test data
    x = torch.randn(batch_size, in_channels, image_size, image_size)
    print(f"\nInput shape: {x.shape}")

    # Test ResNet-18
    print("\n--- ResNet-18 ---")
    model = ResNet18EEG(num_classes=num_classes, in_channels=in_channels, pretrained=False)
    out = model(x)
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {count_parameters(model):,}")

    # Test ResNet-50
    print("\n--- ResNet-50 ---")
    model = ResNet50EEG(num_classes=num_classes, in_channels=in_channels, pretrained=False)
    out = model(x)
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {count_parameters(model):,}")

    # Test Lightweight CNN
    print("\n--- Lightweight CNN ---")
    model = LightweightCNN(num_classes=num_classes, in_channels=in_channels)
    out = model(x)
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {count_parameters(model):,}")

    # Test factory function
    print("\n--- Factory Function ---")
    model = get_cnn_model('resnet18', num_classes=num_classes, in_channels=in_channels, pretrained=False)
    out = model(x)
    print(f"ResNet-18 via factory: {out.shape}")

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
