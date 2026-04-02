"""
Markov Transition Field (MTF) transformations for EEG time-series.

Implements quantization-based MTF encoding that captures temporal dynamics
through first-order Markov transition probabilities.

References:
- Wang & Oates (2015): "Encoding Time Series as Images for Visual Inspection and Classification"
"""

import numpy as np
from pyts.image import MarkovTransitionField
from typing import Tuple, Optional
import argparse
import h5py
from pathlib import Path
from tqdm import tqdm


class MTFTransformer:
    """
    Transform EEG time-series to MTF images.

    MTF quantizes time series into bins and encodes temporal transitions
    as a Markov transition matrix, preserving sequential dynamics in 2D format.
    """

    def __init__(self, image_size: int = 128, n_bins: int = 8):
        """
        Initialize MTF transformer.

        Args:
            image_size: Output image size (NxN). Common sizes: 64, 128, 256
            n_bins: Number of quantization bins (Q). Common: 8, 16
                   Higher Q captures more detail but may be sparse

        Example:
            >>> transformer = MTFTransformer(image_size=128, n_bins=8)
        """
        self.image_size = image_size
        self.n_bins = n_bins
        self.transformer = MarkovTransitionField(
            image_size=image_size,
            n_bins=n_bins,
            strategy='quantile'  # Equal-frequency binning
        )

    def normalize_to_range(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize to [0, 1] for MTF computation.

        Args:
            x: Input array (..., n_times)

        Returns:
            Normalized array in range [0, 1]
        """
        x_min = x.min(axis=-1, keepdims=True)
        x_max = x.max(axis=-1, keepdims=True)

        # Avoid division by zero
        range_val = x_max - x_min
        range_val = np.where(range_val == 0, 1.0, range_val)

        # Normalize to [0, 1]
        normalized = (x - x_min) / range_val

        return normalized

    def transform_epoch(self, epoch: np.ndarray) -> np.ndarray:
        """
        Transform single epoch to MTF image.

        Args:
            epoch: (n_channels, n_times) array

        Returns:
            mtf_images: (n_channels, image_size, image_size) array

        Example:
            >>> epoch = np.random.randn(22, 751)  # 22 channels, 751 timepoints
            >>> mtf_images = transformer.transform_epoch(epoch)
            >>> print(mtf_images.shape)  # (22, 128, 128)
        """
        # Normalize each channel to [0, 1]
        epoch_norm = self.normalize_to_range(epoch)

        # Apply MTF to each channel
        mtf_images = self.transformer.fit_transform(epoch_norm)

        return mtf_images

    def transform_batch(self, data: np.ndarray,
                       strategy: str = 'per_channel') -> np.ndarray:
        """
        Transform batch of epochs.

        Args:
            data: (n_epochs, n_channels, n_times) array
            strategy: 'per_channel', 'average', or 'first_channel'
                - 'per_channel': Keep all channels as separate layers
                - 'average': Average across channels to get single image
                - 'first_channel': Use only first channel

        Returns:
            Transformed images:
            - per_channel: (n_epochs, n_channels, image_size, image_size)
            - average: (n_epochs, 1, image_size, image_size)
            - first_channel: (n_epochs, 1, image_size, image_size)

        Example:
            >>> data = np.random.randn(100, 22, 751)  # 100 epochs
            >>> images = transformer.transform_batch(data, strategy='per_channel')
            >>> print(images.shape)  # (100, 22, 128, 128)
        """
        n_epochs = data.shape[0]
        results = []

        for i in tqdm(range(n_epochs), desc=f"MTF-Q{self.n_bins}"):
            mtf_image = self.transform_epoch(data[i])

            if strategy == 'per_channel':
                # Keep all channels as separate image layers
                results.append(mtf_image)
            elif strategy == 'average':
                # Average across channels to get single image
                avg_image = mtf_image.mean(axis=0, keepdims=True)
                results.append(avg_image)
            elif strategy == 'first_channel':
                # Use only first channel
                results.append(mtf_image[0:1])
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

        return np.array(results)

    def save_images(self, images: np.ndarray, labels: np.ndarray,
                   output_path: str, metadata: Optional[dict] = None):
        """
        Save MTF images to HDF5.

        Args:
            images: Transformed images array
            labels: Class labels (n_epochs,)
            output_path: Path to save HDF5 file
            metadata: Optional metadata dictionary

        Example:
            >>> transformer.save_images(images, labels, 'data/images/mtf/A01T_mtf.h5')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Saving to: {output_path}")

        with h5py.File(output_path, 'w') as f:
            f.create_dataset('images', data=images, compression='gzip')
            f.create_dataset('labels', data=labels)

            # Save transform metadata
            f.attrs['transform'] = 'mtf'
            f.attrs['image_size'] = self.image_size
            f.attrs['n_bins'] = self.n_bins

            # Save additional metadata if provided
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float)):
                        f.attrs[key] = value
                    elif isinstance(value, (list, np.ndarray)):
                        f.attrs[key] = str(value)

        print(f"  Shape: {images.shape}")
        print(f"  Size: {output_path.stat().st_size / 1024**2:.2f} MB")

    def load_images(self, file_path: str) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Load MTF images from HDF5.

        Args:
            file_path: Path to HDF5 file

        Returns:
            images: Image array
            labels: Label array
            metadata: Metadata dictionary
        """
        with h5py.File(file_path, 'r') as f:
            images = f['images'][:]
            labels = f['labels'][:]
            metadata = {key: f.attrs[key] for key in f.attrs.keys()}

        return images, labels, metadata


def transform_preprocessed_file(input_file: str, output_file: str,
                                image_size: int = 128,
                                n_bins: int = 8,
                                strategy: str = 'per_channel'):
    """
    Transform a preprocessed HDF5 file to MTF images.

    Args:
        input_file: Path to preprocessed HDF5 file
        output_file: Path to output MTF images HDF5 file
        image_size: MTF image size (NxN)
        n_bins: Number of quantization bins (Q)
        strategy: Multi-channel strategy

    Example:
        >>> transform_preprocessed_file(
        ...     'data/preprocessed/bci_iv_2a/A01T_preprocessed.h5',
        ...     'data/images/mtf/A01T_mtf_q8.h5',
        ...     image_size=128,
        ...     n_bins=8
        ... )
    """
    print("="*60)
    print(f"MTF Transformation: Q={n_bins}")
    print("="*60)

    # Load preprocessed data
    print(f"Loading: {input_file}")
    with h5py.File(input_file, 'r') as f:
        data = f['data'][:]
        labels = f['labels'][:]

        # Get metadata
        metadata = {key: f.attrs[key] for key in f.attrs.keys()}

    print(f"  Data shape: {data.shape}")
    print(f"  Labels: {np.unique(labels)} (counts: {np.bincount(labels)})")

    # Transform
    transformer = MTFTransformer(image_size=image_size, n_bins=n_bins)
    images = transformer.transform_batch(data, strategy=strategy)

    print(f"\nTransformed images shape: {images.shape}")

    # Add transform info to metadata
    metadata['original_shape'] = str(data.shape)
    metadata['transform_n_bins'] = n_bins
    metadata['transform_strategy'] = strategy

    # Save
    transformer.save_images(images, labels, output_file, metadata)

    print("\nTransformation complete!")
    print("="*60)


def main():
    """Command-line interface for MTF transformation."""
    parser = argparse.ArgumentParser(description='Generate MTF images from preprocessed EEG')
    parser.add_argument('--input', type=str, required=True,
                        help='Input preprocessed HDF5 file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output MTF images HDF5 file')
    parser.add_argument('--bins', type=int, default=8,
                        choices=[4, 8, 16, 32],
                        help='Number of quantization bins (Q)')
    parser.add_argument('--size', type=int, default=128,
                        help='Image size (NxN)')
    parser.add_argument('--strategy', type=str, default='per_channel',
                        choices=['per_channel', 'average', 'first_channel'],
                        help='Multi-channel strategy')

    args = parser.parse_args()

    transform_preprocessed_file(
        input_file=args.input,
        output_file=args.output,
        image_size=args.size,
        n_bins=args.bins,
        strategy=args.strategy
    )


if __name__ == '__main__':
    main()
