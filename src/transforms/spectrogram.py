"""
Spectrogram (STFT) transformations for EEG time-series.

Implements Short-Time Fourier Transform (STFT) spectrograms for
frequency-domain analysis of EEG signals.

References:
- Standard STFT for time-frequency analysis
- Commonly used in EEG classification (e.g., motor imagery, SSVEP)
"""

import numpy as np
from scipy import signal
from scipy.ndimage import zoom
from typing import Tuple, Optional
import argparse
import h5py
from pathlib import Path
from tqdm import tqdm


class SpectrogramTransformer:
    """
    Transform EEG time-series to STFT spectrogram images.

    Spectrograms provide time-frequency representation using sliding-window
    Fourier transforms, capturing temporal evolution of frequency content.
    """

    def __init__(self, sfreq: float = 250.0,
                 window_length: float = 0.5,
                 overlap: float = 0.5,
                 freq_range: Tuple[float, float] = (1.0, 50.0),
                 window_type: str = 'hamming',
                 nfft: Optional[int] = None):
        """
        Initialize Spectrogram transformer.

        Args:
            sfreq: Sampling frequency in Hz
            window_length: Window length in seconds (e.g., 0.5s = 125 samples at 250 Hz)
            overlap: Overlap fraction (0-1). 0.5 = 50% overlap
            freq_range: Frequency range to keep (min_freq, max_freq) in Hz
            window_type: Window function ('hamming', 'hann', 'blackman')
            nfft: FFT size. If None, uses nperseg (window length in samples)

        Example:
            >>> transformer = SpectrogramTransformer(sfreq=250.0, window_length=0.5)
        """
        self.sfreq = sfreq
        self.window_length = window_length
        self.overlap = overlap
        self.freq_range = freq_range
        self.window_type = window_type
        self.nfft = nfft

        # Compute window parameters
        self.nperseg = int(window_length * sfreq)
        self.noverlap = int(self.nperseg * overlap)

        if self.nfft is None:
            self.nfft = self.nperseg

    def compute_spectrogram(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute STFT spectrogram for single channel.

        Args:
            x: 1D time series (n_times,)

        Returns:
            f: Frequency bins (Hz)
            t: Time bins (seconds)
            Sxx: Spectrogram power (freq_bins, time_bins)

        Example:
            >>> x = np.random.randn(751)
            >>> f, t, Sxx = transformer.compute_spectrogram(x)
        """
        # Compute STFT
        f, t, Sxx = signal.spectrogram(
            x,
            fs=self.sfreq,
            window=self.window_type,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            nfft=self.nfft,
            mode='psd',  # Power spectral density
            scaling='density'
        )

        # Filter frequency range
        freq_mask = (f >= self.freq_range[0]) & (f <= self.freq_range[1])
        f = f[freq_mask]
        Sxx = Sxx[freq_mask, :]

        return f, t, Sxx

    def normalize_spectrogram(self, Sxx: np.ndarray) -> np.ndarray:
        """
        Normalize spectrogram to [0, 1] range.

        Args:
            Sxx: Power spectrogram (freq_bins, time_bins)

        Returns:
            Normalized spectrogram in [0, 1]
        """
        # Convert to log-scale (dB)
        Sxx_log = 10 * np.log10(Sxx + 1e-10)  # Add small epsilon to avoid log(0)

        # Normalize to [0, 1]
        min_val = Sxx_log.min()
        max_val = Sxx_log.max()

        if max_val > min_val:
            Sxx_norm = (Sxx_log - min_val) / (max_val - min_val)
        else:
            Sxx_norm = np.zeros_like(Sxx_log)

        return Sxx_norm

    def resize_to_target(self, Sxx: np.ndarray, target_size: int) -> np.ndarray:
        """
        Resize spectrogram to target image size.

        Args:
            Sxx: Spectrogram (freq_bins, time_bins)
            target_size: Target image size (NxN)

        Returns:
            Resized spectrogram: (target_size, target_size)
        """
        # Compute zoom factors
        zoom_freq = target_size / Sxx.shape[0]
        zoom_time = target_size / Sxx.shape[1]

        # Resize using bilinear interpolation
        Sxx_resized = zoom(Sxx, (zoom_freq, zoom_time), order=1)  # order=1 for bilinear

        # Ensure exact target size (zoom may introduce small errors)
        if Sxx_resized.shape != (target_size, target_size):
            Sxx_resized = Sxx_resized[:target_size, :target_size]

        return Sxx_resized

    def transform_channel(self, channel_data: np.ndarray,
                         target_size: int = 128) -> np.ndarray:
        """
        Transform single channel to spectrogram.

        Args:
            channel_data: 1D time series (n_times,)
            target_size: Target image size (NxN)

        Returns:
            Spectrogram image: (target_size, target_size)

        Example:
            >>> channel = np.random.randn(751)
            >>> spec = transformer.transform_channel(channel, target_size=128)
            >>> print(spec.shape)  # (128, 128)
        """
        # Compute spectrogram
        f, t, Sxx = self.compute_spectrogram(channel_data)

        # Normalize
        Sxx_norm = self.normalize_spectrogram(Sxx)

        # Resize to target size
        Sxx_resized = self.resize_to_target(Sxx_norm, target_size)

        return Sxx_resized

    def transform_epoch(self, epoch: np.ndarray,
                       target_size: int = 128) -> np.ndarray:
        """
        Transform multi-channel epoch to spectrograms.

        Args:
            epoch: (n_channels, n_times) array
            target_size: Target image size (NxN)

        Returns:
            spec_images: (n_channels, target_size, target_size) array

        Example:
            >>> epoch = np.random.randn(22, 751)  # 22 channels, 751 timepoints
            >>> spec_images = transformer.transform_epoch(epoch, target_size=128)
            >>> print(spec_images.shape)  # (22, 128, 128)
        """
        n_channels = epoch.shape[0]
        spec_images = np.zeros((n_channels, target_size, target_size))

        for ch in range(n_channels):
            spec_images[ch] = self.transform_channel(epoch[ch], target_size)

        return spec_images

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

        for i in tqdm(range(n_epochs), desc="STFT"):
            spec_image = self.transform_epoch(data[i], target_size)

            if strategy == 'per_channel':
                # Keep all channels as separate image layers
                results.append(spec_image)
            elif strategy == 'average':
                # Average across channels to get single image
                avg_image = spec_image.mean(axis=0, keepdims=True)
                results.append(avg_image)
            elif strategy == 'first_channel':
                # Use only first channel
                results.append(spec_image[0:1])
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

        return np.array(results)

    def save_images(self, images: np.ndarray, labels: np.ndarray,
                   output_path: str, metadata: Optional[dict] = None):
        """
        Save spectrogram images to HDF5.

        Args:
            images: Transformed images array
            labels: Class labels (n_epochs,)
            output_path: Path to save HDF5 file
            metadata: Optional metadata dictionary

        Example:
            >>> transformer.save_images(images, labels, 'data/images/spec/A01T_spec.h5')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Saving to: {output_path}")

        with h5py.File(output_path, 'w') as f:
            f.create_dataset('images', data=images, compression='gzip')
            f.create_dataset('labels', data=labels)

            # Save transform metadata
            f.attrs['transform'] = 'spectrogram'
            f.attrs['sfreq'] = self.sfreq
            f.attrs['window_length'] = self.window_length
            f.attrs['overlap'] = self.overlap
            f.attrs['freq_range'] = str(self.freq_range)
            f.attrs['window_type'] = self.window_type

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
        Load spectrogram images from HDF5.

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
                                window_length: float = 0.5,
                                overlap: float = 0.5,
                                freq_range: Tuple[float, float] = (1.0, 50.0),
                                target_size: int = 128,
                                strategy: str = 'per_channel'):
    """
    Transform a preprocessed HDF5 file to spectrogram images.

    Args:
        input_file: Path to preprocessed HDF5 file
        output_file: Path to output spectrogram images HDF5 file
        sfreq: Sampling frequency
        window_length: STFT window length in seconds
        overlap: Overlap fraction
        freq_range: Frequency range (min, max) in Hz
        target_size: Target image size (NxN)
        strategy: Multi-channel strategy

    Example:
        >>> transform_preprocessed_file(
        ...     'data/preprocessed/bci_iv_2a/A01T_preprocessed.h5',
        ...     'data/images/spec/A01T_spec.h5',
        ...     sfreq=250.0,
        ...     window_length=0.5
        ... )
    """
    print("="*60)
    print(f"STFT Spectrogram Transformation")
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
    transformer = SpectrogramTransformer(
        sfreq=sfreq,
        window_length=window_length,
        overlap=overlap,
        freq_range=freq_range
    )
    images = transformer.transform_batch(data, target_size, strategy)

    print(f"\nTransformed images shape: {images.shape}")

    # Add transform info to metadata
    metadata['original_shape'] = str(data.shape)
    metadata['transform_window_length'] = window_length
    metadata['transform_overlap'] = overlap
    metadata['transform_freq_range'] = str(freq_range)
    metadata['transform_strategy'] = strategy

    # Save
    transformer.save_images(images, labels, output_file, metadata)

    print("\nTransformation complete!")
    print("="*60)


def main():
    """Command-line interface for Spectrogram transformation."""
    parser = argparse.ArgumentParser(description='Generate STFT Spectrogram images from preprocessed EEG')
    parser.add_argument('--input', type=str, required=True,
                        help='Input preprocessed HDF5 file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output spectrogram images HDF5 file')
    parser.add_argument('--sfreq', type=float, default=250.0,
                        help='Sampling frequency in Hz')
    parser.add_argument('--window-length', type=float, default=0.5,
                        help='STFT window length in seconds')
    parser.add_argument('--overlap', type=float, default=0.5,
                        help='Overlap fraction (0-1)')
    parser.add_argument('--freq-min', type=float, default=1.0,
                        help='Minimum frequency (Hz)')
    parser.add_argument('--freq-max', type=float, default=50.0,
                        help='Maximum frequency (Hz)')
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
        window_length=args.window_length,
        overlap=args.overlap,
        freq_range=(args.freq_min, args.freq_max),
        target_size=args.size,
        strategy=args.strategy
    )


if __name__ == '__main__':
    main()
