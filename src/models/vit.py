"""
Vision Transformer (ViT) models for EEG image classification.

Implements ViT architectures using the timm library with adaptations
for multi-channel EEG inputs.

References:
- Dosovitskiy et al. (2020): An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
- timm library: https://github.com/huggingface/pytorch-image-models
"""

import torch
import torch.nn as nn
import timm
from typing import Optional, Literal


class ViTEEG(nn.Module):
    """
    Vision Transformer adapted for EEG images.

    Wraps timm's ViT models with custom input projection for
    multi-channel EEG data.
    """

    def __init__(self,
                 model_name: str = 'vit_base_patch16_224',
                 num_classes: int = 4,
                 in_channels: int = 25,
                 img_size: int = 224,
                 pretrained: bool = True,
                 dropout: float = 0.1):
        """
        Initialize Vision Transformer for EEG.

        Args:
            model_name: ViT variant from timm
                       ('vit_base_patch16_224', 'vit_small_patch16_224', etc.)
            num_classes: Number of output classes
            in_channels: Number of input channels (EEG channels)
            img_size: Input image size
            pretrained: Use ImageNet pretrained weights
            dropout: Dropout rate in head

        Example:
            >>> model = ViTEEG('vit_base_patch16_224', num_classes=4, in_channels=25)
            >>> x = torch.randn(8, 25, 224, 224)
            >>> out = model(x)  # (8, 4)
        """
        super(ViTEEG, self).__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.img_size = img_size

        # Load ViT model from timm
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            img_size=img_size,
            drop_rate=dropout
        )

        # Modify patch embedding for multi-channel input
        if in_channels != 3:
            # Get patch embed parameters
            original_patch_embed = self.vit.patch_embed.proj
            embed_dim = original_patch_embed.out_channels
            kernel_size = original_patch_embed.kernel_size
            stride = original_patch_embed.stride

            # Create new patch embedding
            new_patch_embed = nn.Conv2d(
                in_channels,
                embed_dim,
                kernel_size=kernel_size,
                stride=stride
            )

            # Initialize with pretrained weights if available
            if pretrained:
                with torch.no_grad():
                    pretrained_weight = original_patch_embed.weight  # (embed_dim, 3, patch_size, patch_size)

                    if in_channels < 3:
                        # Average RGB channels
                        new_weight = pretrained_weight.mean(dim=1, keepdim=True)
                        new_patch_embed.weight.copy_(new_weight.repeat(1, in_channels, 1, 1))
                    else:
                        # Repeat RGB channels
                        repeats = (in_channels // 3) + 1
                        new_weight = pretrained_weight.repeat(1, repeats, 1, 1)[:, :in_channels, :, :]
                        new_weight = new_weight * (3.0 / in_channels)
                        new_patch_embed.weight.copy_(new_weight)

                    # Copy bias
                    if original_patch_embed.bias is not None:
                        new_patch_embed.bias.copy_(original_patch_embed.bias)

            # Replace patch embedding
            self.vit.patch_embed.proj = new_patch_embed

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, in_channels, H, W)

        Returns:
            Output logits (batch_size, num_classes)
        """
        return self.vit(x)

    def get_attention_maps(self, x):
        """
        Extract attention maps from transformer blocks.

        Args:
            x: Input tensor (batch_size, in_channels, H, W)

        Returns:
            List of attention maps from each transformer block
        """
        # This requires accessing internal transformer blocks
        # Implementation depends on timm version and model
        # Placeholder for future visualization needs
        raise NotImplementedError("Attention map extraction not yet implemented")


class ViTBase16(ViTEEG):
    """ViT-Base with 16x16 patches."""

    def __init__(self, num_classes: int = 4,
                 in_channels: int = 25,
                 img_size: int = 224,
                 pretrained: bool = True):
        """
        ViT-Base/16 model.

        12 layers, 768 hidden dim, 12 attention heads.
        ~86M parameters.

        Example:
            >>> model = ViTBase16(num_classes=4, in_channels=25, img_size=224)
        """
        super().__init__(
            model_name='vit_base_patch16_224',
            num_classes=num_classes,
            in_channels=in_channels,
            img_size=img_size,
            pretrained=pretrained
        )


class ViTSmall16(ViTEEG):
    """ViT-Small with 16x16 patches."""

    def __init__(self, num_classes: int = 4,
                 in_channels: int = 25,
                 img_size: int = 224,
                 pretrained: bool = True):
        """
        ViT-Small/16 model.

        12 layers, 384 hidden dim, 6 attention heads.
        ~22M parameters.

        Example:
            >>> model = ViTSmall16(num_classes=4, in_channels=25, img_size=224)
        """
        super().__init__(
            model_name='vit_small_patch16_224',
            num_classes=num_classes,
            in_channels=in_channels,
            img_size=img_size,
            pretrained=pretrained
        )


class ViTTiny16(ViTEEG):
    """ViT-Tiny with 16x16 patches."""

    def __init__(self, num_classes: int = 4,
                 in_channels: int = 25,
                 img_size: int = 224,
                 pretrained: bool = True):
        """
        ViT-Tiny/16 model.

        12 layers, 192 hidden dim, 3 attention heads.
        ~5M parameters.

        Example:
            >>> model = ViTTiny16(num_classes=4, in_channels=25, img_size=224)
        """
        super().__init__(
            model_name='vit_tiny_patch16_224',
            num_classes=num_classes,
            in_channels=in_channels,
            img_size=img_size,
            pretrained=pretrained
        )


def get_vit_model(
    model_name: Literal['vit_base', 'vit_small', 'vit_tiny'],
    num_classes: int = 4,
    in_channels: int = 25,
    img_size: int = 224,
    pretrained: bool = True
) -> nn.Module:
    """
    Factory function to get ViT model by name.

    Args:
        model_name: Model variant ('vit_base', 'vit_small', 'vit_tiny')
        num_classes: Number of output classes
        in_channels: Number of input channels
        img_size: Input image size (must be divisible by patch size)
        pretrained: Use ImageNet pretrained weights

    Returns:
        ViT model instance

    Example:
        >>> model = get_vit_model('vit_base', num_classes=4, in_channels=25)
        >>> model = get_vit_model('vit_small', num_classes=4, img_size=128)

    Raises:
        ValueError: If model_name not recognized
    """
    models_dict = {
        'vit_base': ViTBase16,
        'vit_small': ViTSmall16,
        'vit_tiny': ViTTiny16,
    }

    if model_name not in models_dict:
        available = ', '.join(models_dict.keys())
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {available}"
        )

    model_class = models_dict[model_name]

    return model_class(
        num_classes=num_classes,
        in_channels=in_channels,
        img_size=img_size,
        pretrained=pretrained
    )


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test models
    print("="*60)
    print("Testing Vision Transformer Models")
    print("="*60)

    batch_size = 4
    in_channels = 25
    img_size = 224
    num_classes = 4

    # Test data
    x = torch.randn(batch_size, in_channels, img_size, img_size)
    print(f"\nInput shape: {x.shape}")

    # Test ViT-Tiny (fastest to test)
    print("\n--- ViT-Tiny/16 ---")
    model = ViTTiny16(
        num_classes=num_classes,
        in_channels=in_channels,
        img_size=img_size,
        pretrained=False  # Faster for testing
    )
    out = model(x)
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {count_parameters(model):,}")

    # Test ViT-Small
    print("\n--- ViT-Small/16 ---")
    model = ViTSmall16(
        num_classes=num_classes,
        in_channels=in_channels,
        img_size=img_size,
        pretrained=False
    )
    out = model(x)
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {count_parameters(model):,}")

    # Test factory function
    print("\n--- Factory Function ---")
    model = get_vit_model(
        'vit_tiny',
        num_classes=num_classes,
        in_channels=in_channels,
        img_size=img_size,
        pretrained=False
    )
    out = model(x)
    print(f"ViT-Tiny via factory: {out.shape}")

    # Test with different image sizes
    print("\n--- Different Image Sizes ---")
    for size in [64, 128, 224]:
        x_test = torch.randn(2, in_channels, size, size)
        model = get_vit_model(
            'vit_tiny',
            num_classes=num_classes,
            in_channels=in_channels,
            img_size=size,
            pretrained=False
        )
        out = model(x_test)
        print(f"  Size {size}x{size}: Input {x_test.shape} -> Output {out.shape}")

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
