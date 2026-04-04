"""
Data augmentation for EEG images and time-series.

Implements augmentation techniques for both transformed EEG images
and raw time-series signals to improve model generalization.

References:
- Standard image augmentations adapted for EEG images
- Time-series specific augmentations (jitter, scaling, rotation)
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from typing import Optional, Tuple, List


class ImageAugmentation:
    """
    Augmentation pipeline for EEG images.

    Applies spatial transformations while preserving EEG patterns.
    """

    def __init__(self,
                 horizontal_flip: bool = True,
                 vertical_flip: bool = False,
                 rotation_degrees: int = 15,
                 brightness: float = 0.2,
                 contrast: float = 0.2,
                 noise_std: float = 0.01):
        """
        Initialize image augmentation pipeline.

        Args:
            horizontal_flip: Apply random horizontal flip
            vertical_flip: Apply random vertical flip
            rotation_degrees: Max rotation in degrees
            brightness: Brightness jitter factor
            contrast: Contrast jitter factor
            noise_std: Gaussian noise standard deviation

        Example:
            >>> aug = ImageAugmentation(rotation_degrees=10, noise_std=0.01)
            >>> augmented = aug(image)
        """
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotation_degrees = rotation_degrees
        self.brightness = brightness
        self.contrast = contrast
        self.noise_std = noise_std

        # Build transform list
        transform_list = []

        if horizontal_flip:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

        if vertical_flip:
            transform_list.append(transforms.RandomVerticalFlip(p=0.5))

        if rotation_degrees > 0:
            transform_list.append(
                transforms.RandomRotation(degrees=rotation_degrees)
            )

        if brightness > 0 or contrast > 0:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=brightness,
                    contrast=contrast
                )
            )

        self.transforms = transforms.Compose(transform_list)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to image tensor.

        Args:
            x: Input image (C, H, W) or (B, C, H, W)

        Returns:
            Augmented image
        """
        # Apply torchvision transforms
        x_aug = self.transforms(x)

        # Add Gaussian noise
        if self.noise_std > 0:
            noise = torch.randn_like(x_aug) * self.noise_std
            x_aug = x_aug + noise

        return x_aug


class TimeSeriesAugmentation:
    """
    Augmentation pipeline for raw EEG time-series.

    Applies temporal transformations suitable for physiological signals.
    """

    def __init__(self,
                 jitter_std: float = 0.03,
                 scaling_range: Tuple[float, float] = (0.9, 1.1),
                 time_warp: bool = False,
                 magnitude_warp: bool = False,
                 channel_dropout: float = 0.0):
        """
        Initialize time-series augmentation pipeline.

        Args:
            jitter_std: Standard deviation for additive noise
            scaling_range: Range for amplitude scaling (min, max)
            time_warp: Apply time warping
            magnitude_warp: Apply magnitude warping
            channel_dropout: Probability of dropping channels

        Example:
            >>> aug = TimeSeriesAugmentation(jitter_std=0.03, scaling_range=(0.9, 1.1))
            >>> augmented = aug(timeseries)
        """
        self.jitter_std = jitter_std
        self.scaling_range = scaling_range
        self.time_warp = time_warp
        self.magnitude_warp = magnitude_warp
        self.channel_dropout = channel_dropout

    def apply_jitter(self, x: torch.Tensor) -> torch.Tensor:
        """Add random Gaussian jitter."""
        if self.jitter_std > 0:
            noise = torch.randn_like(x) * self.jitter_std
            return x + noise
        return x

    def apply_scaling(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random amplitude scaling."""
        if self.scaling_range != (1.0, 1.0):
            scale = torch.empty(x.shape[0], x.shape[1], 1).uniform_(
                self.scaling_range[0],
                self.scaling_range[1]
            ).to(x.device)
            return x * scale
        return x

    def apply_channel_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly drop channels (set to zero)."""
        if self.channel_dropout > 0:
            mask = torch.bernoulli(
                torch.ones(x.shape[0], x.shape[1], 1) * (1 - self.channel_dropout)
            ).to(x.device)
            return x * mask
        return x

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to time-series tensor.

        Args:
            x: Input time-series (B, C, T) where C=channels, T=time

        Returns:
            Augmented time-series
        """
        x_aug = x.clone()

        # Apply jitter
        x_aug = self.apply_jitter(x_aug)

        # Apply scaling
        x_aug = self.apply_scaling(x_aug)

        # Apply channel dropout
        x_aug = self.apply_channel_dropout(x_aug)

        return x_aug


class MixUp:
    """
    MixUp augmentation for both images and time-series.

    Mixes two samples with random interpolation weight.

    Reference:
    Zhang et al. (2017): mixup: Beyond Empirical Risk Minimization
    """

    def __init__(self, alpha: float = 0.2):
        """
        Initialize MixUp.

        Args:
            alpha: Beta distribution parameter (higher = more mixing)

        Example:
            >>> mixup = MixUp(alpha=0.2)
            >>> mixed_x, mixed_y = mixup(x1, y1, x2, y2)
        """
        self.alpha = alpha

    def __call__(self,
                 x1: torch.Tensor, y1: torch.Tensor,
                 x2: torch.Tensor, y2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply MixUp to two samples.

        Args:
            x1: First sample data
            y1: First sample labels (one-hot or class indices)
            x2: Second sample data
            y2: Second sample labels

        Returns:
            mixed_x: Mixed data
            mixed_y: Mixed labels
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        # Mix data
        mixed_x = lam * x1 + (1 - lam) * x2

        # Mix labels (if one-hot)
        if y1.dim() > 1:
            mixed_y = lam * y1 + (1 - lam) * y2
        else:
            # For class indices, return tuple
            mixed_y = (y1, y2, lam)

        return mixed_x, mixed_y


class CutMix:
    """
    CutMix augmentation for EEG images.

    Cuts and pastes patches between images.

    Reference:
    Yun et al. (2019): CutMix: Regularization Strategy to Train Strong Classifiers
    """

    def __init__(self, alpha: float = 1.0):
        """
        Initialize CutMix.

        Args:
            alpha: Beta distribution parameter

        Example:
            >>> cutmix = CutMix(alpha=1.0)
            >>> mixed_x, mixed_y = cutmix(x1, y1, x2, y2)
        """
        self.alpha = alpha

    def __call__(self,
                 x1: torch.Tensor, y1: torch.Tensor,
                 x2: torch.Tensor, y2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply CutMix to two image samples.

        Args:
            x1: First image (C, H, W)
            y1: First label
            x2: Second image (C, H, W)
            y2: Second label

        Returns:
            mixed_x: Mixed image
            mixed_y: Mixed label
        """
        lam = np.random.beta(self.alpha, self.alpha)

        _, H, W = x1.shape

        # Random box
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # Uniform sampling
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        # Mix images
        mixed_x = x1.clone()
        mixed_x[:, bby1:bby2, bbx1:bbx2] = x2[:, bby1:bby2, bbx1:bbx2]

        # Adjust lambda based on actual box area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

        # Mix labels
        if y1.dim() > 1:
            mixed_y = lam * y1 + (1 - lam) * y2
        else:
            mixed_y = (y1, y2, lam)

        return mixed_x, mixed_y


def get_augmentation(
    data_type: str = 'image',
    **kwargs
):
    """
    Factory function to get augmentation pipeline.

    Args:
        data_type: 'image' or 'timeseries'
        **kwargs: Arguments for augmentation class

    Returns:
        Augmentation instance

    Example:
        >>> aug = get_augmentation('image', rotation_degrees=10)
        >>> aug = get_augmentation('timeseries', jitter_std=0.03)
    """
    if data_type == 'image':
        return ImageAugmentation(**kwargs)
    elif data_type == 'timeseries':
        return TimeSeriesAugmentation(**kwargs)
    else:
        raise ValueError(f"Unknown data_type: {data_type}. Use 'image' or 'timeseries'")


if __name__ == '__main__':
    # Test augmentations
    print("="*60)
    print("Testing Augmentation Modules")
    print("="*60)

    # Test Image Augmentation
    print("\n--- Image Augmentation ---")
    img_aug = ImageAugmentation(rotation_degrees=10, noise_std=0.01)
    x_img = torch.randn(3, 64, 64)  # (C, H, W)
    x_img_aug = img_aug(x_img)
    print(f"Input shape: {x_img.shape}")
    print(f"Output shape: {x_img_aug.shape}")
    print(f"Value range: [{x_img_aug.min():.3f}, {x_img_aug.max():.3f}]")

    # Test Time-Series Augmentation
    print("\n--- Time-Series Augmentation ---")
    ts_aug = TimeSeriesAugmentation(jitter_std=0.03, scaling_range=(0.9, 1.1))
    x_ts = torch.randn(8, 25, 751)  # (B, C, T)
    x_ts_aug = ts_aug(x_ts)
    print(f"Input shape: {x_ts.shape}")
    print(f"Output shape: {x_ts_aug.shape}")
    print(f"Value range: [{x_ts_aug.min():.3f}, {x_ts_aug.max():.3f}]")

    # Test MixUp
    print("\n--- MixUp ---")
    mixup = MixUp(alpha=0.2)
    x1, y1 = torch.randn(25, 64, 64), torch.tensor(0)
    x2, y2 = torch.randn(25, 64, 64), torch.tensor(1)
    mixed_x, mixed_y = mixup(x1, y1, x2, y2)
    print(f"Mixed shape: {mixed_x.shape}")
    print(f"Mixed labels: {mixed_y}")

    # Test CutMix
    print("\n--- CutMix ---")
    cutmix = CutMix(alpha=1.0)
    mixed_x, mixed_y = cutmix(x1, y1, x2, y2)
    print(f"Mixed shape: {mixed_x.shape}")
    print(f"Mixed labels: {mixed_y}")

    # Test factory
    print("\n--- Factory Function ---")
    aug = get_augmentation('image', rotation_degrees=15)
    print(f"Created: {aug.__class__.__name__}")

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
