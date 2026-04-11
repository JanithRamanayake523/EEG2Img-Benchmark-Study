"""
Scalogram (CWT) transformations for EEG time-series.

Implements Continuous Wavelet Transform (CWT) scalograms for
multi-resolution time-frequency analysis of EEG signals.

References:
- Standard CWT for time-frequency analysis
- Morlet wavelet commonly used in EEG studies
"""

import numpy as np
import pywt
from scipy.ndimage import zoom
from typing import Tuple, Optional
import argparse
import h5py
from pathlib import Path
from tqdm import tqdm


class ScalogramTransformer:
    """
    Transform EEG time-series to CWT scalogram images.

    Scalograms provide multi-resolution time-frequency representation
    using continuous wavelet transforms with Morlet wavelets.
    """

    def __init__(self, sfreq: float = 250.0,
                 freq_range: Tuple[float, float] = (1.0, 50.0),
                 n_scales: int = 64,
                 wavelet: str = 'morl'):
        """
        Initialize Scalogram transformer.

        Args:
            sfreq: Sampling frequency in Hz
            freq_range: Frequency range (min_freq, max_freq) in Hz
            n_scales: Number of scales (frequency bins)
            wavelet: Wavelet type. Common: 'morl' (Morlet), 'mexh' (Mexican hat)

        Example:
            >>> transformer = ScalogramTransformer(sfreq=250.0, n_scales=64)
        """
        self.sfreq = sfreq
        self.freq_range = freq_range
        self.n_scales = n_scales
        self.wavelet = wavelet

        # Compute scales from frequency range
        self.scales = self._compute_scales()

    def _compute_scales(self) -> np.ndarray:
        """
        Compute scales corresponding to desired frequency range.

        Returns:
            scales: Array of scales for CWT
        """
        # Get wavelet center frequency
        # Use pywt.scale2frequency to convert scales to frequencies
        sampling_period = 1.0 / self.sfreq

        # For CWT, we compute scales that correspond to frequencies in freq_range
        # Using pywt.scale2frequency, we have: frequency = scale2frequency(wavelet, scale) / sampling_period
        # So: scale = 1.0 / (scale2frequency(wavelet, 1) * frequency)

        # Get center frequency by computing frequency at scale=1
        try:
            center_freq = pywt.scale2frequency(self.wavelet, 1.0)
        except:
            # Fallback for wavelets without center frequency
            center_freq = 1.0

        if center_freq is None or center_freq == 0:
            center_freq = 1.0

        # Convert frequency range to scale range
        # scale = center_freq / (freq * sampling_period)
        scale_min = center_freq / (self.freq_range[1] * sampling_period)
        scale_max = center_freq / (self.freq_range[0] * sampling_period)

        # Generate logarithmically-spaced scales
        scales = np.logspace(np.log10(scale_min), np.log10(scale_max), self.n_scales)

        return scales

    def compute_scalogram(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute CWT scalogram for single channel.

        Args:
            x: 1D time series (n_times,)

        Returns:
            coefficients: CWT coefficients (n_scales, n_times)
            frequencies: Corresponding frequencies (n_scales,)

        Example:
            >>> x = np.random.randn(751)
            >>> coeffs, freqs = transformer.compute_scalogram(x)
        """
        # Compute CWT
        coefficients, frequencies = pywt.cwt(
            x,
            scales=self.scales,
            wavelet=self.wavelet,
            sampling_period=1.0 / self.sfreq
        )

        return coefficients, frequencies

    def compute_power_scalogram(self, x: np.ndarray) -> np.ndarray:
        """
        Compute power scalogram |CWT|^2.

        Args:
            x: 1D time series (n_times,)

        Returns:
            Power scalogram: (n_scales, n_times)
        """
        coefficients, _ = self.compute_scalogram(x)

        # Compute power (magnitude squared)
        power = np.abs(coefficients) ** 2

        return power

    def normalize_scalogram(self, power: np.ndarray) -> np.ndarray:
        """
        Normalize scalogram to [0, 1] range.

        Args:
            power: Power scalogram (n_scales, n_times)

        Returns:
            Normalized scalogram in [0, 1]
        """
        # Convert to log-scale
        power_log = np.log10(power + 1e-10)  # Add small epsilon to avoid log(0)

        # Normalize to [0, 1]
        min_val = power_log.min()
        max_val = power_log.max()

        if max_val > min_val:
            power_norm = (power_log - min_val) / (max_val - min_val)
        else:
            power_norm = np.zeros_like(power_log)

        return power_norm

    def resize_to_target(self, scalogram: np.ndarray, target_size: int) -> np.ndarray:
        """
        Resize scalogram to target image size.

        Args:
            scalogram: Scalogram (n_scales, n_times)
            target_size: Target image size (NxN)

        Returns:
            Resized scalogram: (target_size, target_size)
        """
        # Compute zoom factors
        zoom_freq = target_size / scalogram.shape[0]
        zoom_time = target_size / scalogram.shape[1]

        # Resize using bilinear interpolation
        scalogram_resized = zoom(scalogram, (zoom_freq, zoom_time), order=1)

        # Ensure exact target size
        if scalogram_resized.shape != (target_size, target_size):
            scalogram_resized = scalogram_resized[:target_size, :target_size]

        return scalogram_resized

    def transform_channel(self, channel_data: np.ndarray,
                         target_size: int = 128) -> np.ndarray:
        """
        Transform single channel to scalogram.

        Args:
            channel_data: 1D time series (n_times,)
            target_size: Target image size (NxN)

        Returns:
            Scalogram image: (target_size, target_size)

        Example:
            >>> channel = np.random.randn(751)
            >>> scalo = transformer.transform_channel(channel, target_size=128)
            >>> print(scalo.shape)  # (128, 128)
        """
        # Compute power scalogram
        power = self.compute_power_scalogram(channel_data)

        # Normalize
        power_norm = self.normalize_scalogram(power)

        # Resize to target size
        power_resized = self.resize_to_target(power_norm, target_size)

        return power_resized

    def transform_epoch(self, epoch: np.ndarray,
                       target_size: int = 128) -> np.ndarray:
        """
        Transform multi-channel epoch to scalograms.

        Args:
            epoch: (n_channels, n_times) array
            target_size: Target image size (NxN)

        Returns:
            scalo_images: (n_channels, target_size, target_size) array

        Example:
            >>> epoch = np.random.randn(22, 751)  # 22 channels, 751 timepoints
            >>> scalo_images = transformer.transform_epoch(epoch, target_size=128)
            >>> print(scalo_images.shape)  # (22, 128, 128)
        """
        n_channels = epoch.shape[0]
        scalo_images = np.zeros((n_channels, target_size, target_size))

        for ch in range(n_channels):
            scalo_images[ch] = self.transform_channel(epoch[ch], target_size)

        return scalo_images

    def transform_batch(self, data: np.ndarray,
                       target_size: int = 128,
                       strategy: str = 'per_channel') -> np.ndarray:
        """
        Transform batch of epochs.

        Args:
            data: (n_epochs, n_channels, n_times) array
            target_size: Target image size (NxN)
            strategy: 'per_channel', 'average', or 'first_channel'
                - 'per_channel': Keep all channels as separate layers
                - 'average': Average across channels to get single image
                - 'first_channel': Use only first channel

        Returns:
            Transformed images:
            - per_channel: (n_epochs, n_channels, target_size, target_size)
            - average: (n_epochs, 1, target_size, target_size)
            - first_channel: (n_epochs, 1, target_size, target_size)

        Example:
            >>> data = np.random.randn(100, 22, 751)  # 100 epochs
            >>> images = transformer.transform_batch(data, target_size=128)
            >>> print(images.shape)  # (100, 22, 128, 128)
        """
        n_epochs = data.shape[0]
        results = []

        for i in tqdm(range(n_epochs), desc=f"CWT-{self.wavelet}"):
            scalo_image = self.transform_epoch(data[i], target_size)

            if strategy == 'per_channel':
                # Keep all channels as separate image layers
                results.append(scalo_image)
            elif strategy == 'average':
                # Average across channels to get single image
                avg_image = scalo_image.mean(axis=0, keepdims=True)
                results.append(avg_image)
            elif strategy == 'first_channel':
                # Use only first channel
                results.append(scalo_image[0:1])
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

        return np.array(results)

    def save_images(self, images: np.ndarray, labels: np.ndarray,
                   output_path: str, metadata: Optional[dict] = None):
        """
        Save scalogram images to HDF5.

        Args:
            images: Transformed images array
            labels: Class labels (n_epochs,)
            output_path: Path to save HDF5 file
            metadata: Optional metadata dictionary

        Example:
            >>> transformer.save_images(images, labels, 'data/images/cwt/A01T_cwt.h5')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Saving to: {output_path}")

        with h5py.File(output_path, 'w') as f:
            f.create_dataset('images', data=images, compression='gzip')
            f.create_dataset('labels', data=labels)

            # Save transform metadata
            f.attrs['transform'] = 'scalogram'
            f.attrs['sfreq'] = self.sfreq
            f.attrs['freq_range'] = str(self.freq_range)
            f.attrs['n_scales'] = self.n_scales
            f.attrs['wavelet'] = self.wavelet

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
        Load scalogram images from HDF5.

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
                                sfreq: float = 250.0,
                                freq_range: Tuple[float, float] = (1.0, 50.0),
                                n_scales: int = 64,
                                wavelet: str = 'morl',
                                target_size: int = 128,
                                strategy: str = 'per_channel'):
    """
    Transform a preprocessed HDF5 file to scalogram images.

    Args:
        input_file: Path to preprocessed HDF5 file
        output_file: Path to output scalogram images HDF5 file
        sfreq: Sampling frequency
        freq_range: Frequency range (min, max) in Hz
        n_scales: Number of CWT scales
        wavelet: Wavelet type ('morl', 'mexh')
        target_size: Target image size (NxN)
        strategy: Multi-channel strategy

    Example:
        >>> transform_preprocessed_file(
        ...     'data/preprocessed/bci_iv_2a/A01T_preprocessed.h5',
        ...     'data/images/cwt/A01T_cwt.h5',
        ...     sfreq=250.0,
        ...     n_scales=64
        ... )
    """
    print("="*60)
    print(f"CWT Scalogram Transformation ({wavelet.upper()})")
    print("="*60)

    # Load preprocessed data
    print(f"Loading: {input_file}")
    with h5py.File(input_file, 'r') as f:
        data = f['data'][:]
        labels = f['labels'][:]

        # Get metadata
        metadata = {key: f.attrs[key] for key in f.attrs.keys()}

        # Get sampling frequency from metadata if available
        if 'sfreq' in metadata:
            sfreq = float(metadata['sfreq'])
            print(f"  Using sfreq from metadata: {sfreq} Hz")

    print(f"  Data shape: {data.shape}")
    print(f"  Labels: {np.unique(labels)} (counts: {np.bincount(labels)})")

    # Transform
    transformer = ScalogramTransformer(
        sfreq=sfreq,
        freq_range=freq_range,
        n_scales=n_scales,
        wavelet=wavelet
    )
    images = transformer.transform_batch(data, target_size, strategy)

    print(f"\nTransformed images shape: {images.shape}")

    # Add transform info to metadata
    metadata['original_shape'] = str(data.shape)
    metadata['transform_freq_range'] = str(freq_range)
    metadata['transform_n_scales'] = n_scales
    metadata['transform_wavelet'] = wavelet
    metadata['transform_strategy'] = strategy

    # Save
    transformer.save_images(images, labels, output_file, metadata)

    print("\nTransformation complete!")
    print("="*60)


def main():
    """Command-line interface for Scalogram transformation."""
    parser = argparse.ArgumentParser(description='Generate CWT Scalogram images from preprocessed EEG')
    parser.add_argument('--input', type=str, required=True,
                        help='Input preprocessed HDF5 file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output scalogram images HDF5 file')
    parser.add_argument('--sfreq', type=float, default=250.0,
                        help='Sampling frequency in Hz')
    parser.add_argument('--freq-min', type=float, default=1.0,
                        help='Minimum frequency (Hz)')
    parser.add_argument('--freq-max', type=float, default=50.0,
                        help='Maximum frequency (Hz)')
    parser.add_argument('--n-scales', type=int, default=64,
                        help='Number of CWT scales')
    parser.add_argument('--wavelet', type=str, default='morl',
                        choices=['morl', 'mexh', 'cgau1', 'cgau2'],
                        help='Wavelet type')
    parser.add_argument('--size', type=int, default=128,
                        help='Target image size (NxN)')
    parser.add_argument('--strategy', type=str, default='per_channel',
                        choices=['per_channel', 'average', 'first_channel'],
                        help='Multi-channel strategy')

    args = parser.parse_args()

    transform_preprocessed_file(
        input_file=args.input,
        output_file=args.output,
        sfreq=args.sfreq,
        freq_range=(args.freq_min, args.freq_max),
        n_scales=args.n_scales,
        wavelet=args.wavelet,
        target_size=args.size,
        strategy=args.strategy
    )


if __name__ == '__main__':
    main()
