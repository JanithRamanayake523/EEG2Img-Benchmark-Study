"""
Topographic mapping (SSFI) transformations for EEG time-series.

Implements Spatio-Spectral Feature Images (SSFI) that preserve spatial
electrode locations and frequency band information.

References:
- Mastandrea et al. (2023): "Spatio-spectral feature images for EEG classification"
- Leverages spatial information from electrode montage
"""

import numpy as np
import mne
from scipy.interpolate import griddata
from scipy.signal import welch
from scipy.integrate import trapezoid
from typing import Tuple, Optional, Dict, List
import argparse
import h5py
from pathlib import Path
from tqdm import tqdm


class TopographicTransformer:
    """
    Transform EEG time-series to topographic feature images.

    Creates 2D spatial maps by interpolating electrode values onto a grid,
    preserving spatial relationships. Can generate separate maps for different
    frequency bands (SSFI approach).
    """

    def __init__(self, ch_names: List[str],
                 grid_size: int = 64,
                 bands: Optional[Dict[str, Tuple[float, float]]] = None,
                 sfreq: float = 250.0):
        """
        Initialize Topographic transformer.

        Args:
            ch_names: List of channel names (e.g., ['Fz', 'C3', 'C4', ...])
            grid_size: Size of 2D grid for interpolation (NxN)
            bands: Dictionary of frequency bands {name: (fmin, fmax)}
                  Default: standard EEG bands
            sfreq: Sampling frequency (needed for band-power computation)

        Example:
            >>> ch_names = ['Fz', 'C3', 'Cz', 'C4', 'Pz']
            >>> transformer = TopographicTransformer(ch_names, grid_size=64)
        """
        self.ch_names = ch_names
        self.grid_size = grid_size
        self.sfreq = sfreq

        # Default EEG frequency bands
        if bands is None:
            self.bands = {
                'delta': (1, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 50),
            }
        else:
            self.bands = bands

        # Get electrode positions
        self.electrode_positions = self._get_electrode_positions()

        # Create interpolation grid
        self.grid_x, self.grid_y = self._create_grid()

    def _get_electrode_positions(self) -> np.ndarray:
        """
        Get 2D electrode positions from MNE montage.

        Returns:
            positions: (n_channels, 2) array with x, y coordinates
        """
        # Create info object with channel names
        info = mne.create_info(
            ch_names=self.ch_names,
            sfreq=self.sfreq,
            ch_types='eeg'
        )

        # Set standard montage (10-20 or 10-10 system)
        try:
            montage = mne.channels.make_standard_montage('standard_1020')
            info.set_montage(montage, on_missing='ignore')
        except Exception:
            # Fallback: try biosemi64
            try:
                montage = mne.channels.make_standard_montage('biosemi64')
                info.set_montage(montage, on_missing='ignore')
            except Exception:
                # If montage fails, create simple grid positions
                print("Warning: Could not load standard montage. Using grid positions.")
                return self._create_simple_grid_positions()

        # Get 2D positions (project to 2D)
        pos_3d = np.array([info['chs'][i]['loc'][:3] for i in range(len(self.ch_names))])

        # Project to 2D using azimuthal equidistant projection
        # (standard for EEG topographic maps)
        x = pos_3d[:, 0]
        y = pos_3d[:, 1]
        z = pos_3d[:, 2]

        # Project sphere to plane
        # Use standard projection: (x, y) from 3D to 2D
        radius = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / (radius + 1e-10))
        phi = np.arctan2(y, x)

        # Convert to 2D Cartesian
        r = theta / np.pi  # Normalize radius to [0, 1]
        pos_x = r * np.cos(phi)
        pos_y = r * np.sin(phi)

        positions = np.column_stack([pos_x, pos_y])

        return positions

    def _create_simple_grid_positions(self) -> np.ndarray:
        """
        Create simple grid positions when montage is unavailable.

        Returns:
            positions: (n_channels, 2) array
        """
        n_channels = len(self.ch_names)
        n_cols = int(np.ceil(np.sqrt(n_channels)))
        n_rows = int(np.ceil(n_channels / n_cols))

        positions = []
        for i in range(n_channels):
            row = i // n_cols
            col = i % n_cols
            x = (col / (n_cols - 1)) * 2 - 1  # Normalize to [-1, 1]
            y = (row / (n_rows - 1)) * 2 - 1
            positions.append([x, y])

        return np.array(positions)

    def _create_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create 2D interpolation grid.

        Returns:
            grid_x, grid_y: Meshgrid arrays (grid_size, grid_size)
        """
        # Create regular grid in [-1, 1] x [-1, 1]
        x = np.linspace(-1, 1, self.grid_size)
        y = np.linspace(-1, 1, self.grid_size)
        grid_x, grid_y = np.meshgrid(x, y)

        return grid_x, grid_y

    def compute_band_power(self, data: np.ndarray,
                          band: Tuple[float, float]) -> np.ndarray:
        """
        Compute band power for each channel using Welch's method.

        Args:
            data: (n_channels, n_times) array
            band: Frequency band (fmin, fmax) in Hz

        Returns:
            band_power: (n_channels,) array with power in the band
        """
        n_channels = data.shape[0]
        band_power = np.zeros(n_channels)

        for ch in range(n_channels):
            # Compute power spectral density using Welch's method
            freqs, psd = welch(
                data[ch],
                fs=self.sfreq,
                nperseg=min(256, data.shape[1]),
                noverlap=None,
                scaling='density'
            )

            # Select frequencies in band
            freq_mask = (freqs >= band[0]) & (freqs <= band[1])

            # Integrate power in band
            band_power[ch] = trapezoid(psd[freq_mask], freqs[freq_mask])

        return band_power

    def interpolate_to_grid(self, channel_values: np.ndarray) -> np.ndarray:
        """
        Interpolate channel values to 2D grid.

        Args:
            channel_values: (n_channels,) array with values per channel

        Returns:
            grid_values: (grid_size, grid_size) interpolated map
        """
        # Interpolate using cubic interpolation
        grid_values = griddata(
            points=self.electrode_positions,
            values=channel_values,
            xi=(self.grid_x, self.grid_y),
            method='cubic',
            fill_value=0.0  # Fill outside convex hull with 0
        )

        # Apply circular mask (head shape)
        radius = np.sqrt(self.grid_x**2 + self.grid_y**2)
        mask = radius <= 1.0
        grid_values = grid_values * mask

        return grid_values

    def normalize_map(self, grid_map: np.ndarray) -> np.ndarray:
        """
        Normalize grid map to [0, 1].

        Args:
            grid_map: (grid_size, grid_size) array

        Returns:
            Normalized map in [0, 1]
        """
        min_val = grid_map.min()
        max_val = grid_map.max()

        if max_val > min_val:
            normalized = (grid_map - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(grid_map)

        return normalized

    def transform_epoch_ssfi(self, epoch: np.ndarray) -> np.ndarray:
        """
        Transform epoch to Spatio-Spectral Feature Image (SSFI).

        Creates separate topographic maps for each frequency band
        and stacks them as channels.

        Args:
            epoch: (n_channels, n_times) array

        Returns:
            ssfi: (n_bands, grid_size, grid_size) multi-band topographic image

        Example:
            >>> epoch = np.random.randn(22, 751)
            >>> ssfi = transformer.transform_epoch_ssfi(epoch)
            >>> print(ssfi.shape)  # (5, 64, 64) for 5 bands
        """
        n_bands = len(self.bands)
        ssfi = np.zeros((n_bands, self.grid_size, self.grid_size))

        for i, (band_name, band_range) in enumerate(self.bands.items()):
            # Compute band power for each channel
            band_power = self.compute_band_power(epoch, band_range)

            # Interpolate to grid
            grid_map = self.interpolate_to_grid(band_power)

            # Normalize
            grid_map_norm = self.normalize_map(grid_map)

            ssfi[i] = grid_map_norm

        return ssfi

    def transform_batch(self, data: np.ndarray,
                       mode: str = 'ssfi') -> np.ndarray:
        """
        Transform batch of epochs.

        Args:
            data: (n_epochs, n_channels, n_times) array
            mode: Transformation mode
                - 'ssfi': Multi-band spatio-spectral images
                - 'single': Single average topographic map

        Returns:
            Transformed images:
            - ssfi: (n_epochs, n_bands, grid_size, grid_size)
            - single: (n_epochs, 1, grid_size, grid_size)

        Example:
            >>> data = np.random.randn(100, 22, 751)
            >>> images = transformer.transform_batch(data, mode='ssfi')
            >>> print(images.shape)  # (100, 5, 64, 64)
        """
        n_epochs = data.shape[0]
        results = []

        for i in tqdm(range(n_epochs), desc="Topographic"):
            if mode == 'ssfi':
                topo_image = self.transform_epoch_ssfi(data[i])
            elif mode == 'single':
                # Create single map from all frequency content
                all_bands_power = self.compute_band_power(
                    data[i],
                    (self.bands['delta'][0], self.bands['gamma'][1])
                )
                grid_map = self.interpolate_to_grid(all_bands_power)
                grid_map_norm = self.normalize_map(grid_map)
                topo_image = grid_map_norm[np.newaxis, :, :]  # Add channel dim
            else:
                raise ValueError(f"Unknown mode: {mode}")

            results.append(topo_image)

        return np.array(results)

    def save_images(self, images: np.ndarray, labels: np.ndarray,
                   output_path: str, metadata: Optional[dict] = None):
        """
        Save topographic images to HDF5.

        Args:
            images: Transformed images array
            labels: Class labels (n_epochs,)
            output_path: Path to save HDF5 file
            metadata: Optional metadata dictionary

        Example:
            >>> transformer.save_images(images, labels, 'data/images/topo/A01T_topo.h5')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Saving to: {output_path}")

        with h5py.File(output_path, 'w') as f:
            f.create_dataset('images', data=images, compression='gzip')
            f.create_dataset('labels', data=labels)

            # Save transform metadata
            f.attrs['transform'] = 'topographic'
            f.attrs['grid_size'] = self.grid_size
            f.attrs['n_bands'] = len(self.bands)
            f.attrs['bands'] = str(self.bands)
            f.attrs['ch_names'] = str(self.ch_names)

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
        Load topographic images from HDF5.

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
                                grid_size: int = 64,
                                mode: str = 'ssfi'):
    """
    Transform a preprocessed HDF5 file to topographic images.

    Args:
        input_file: Path to preprocessed HDF5 file
        output_file: Path to output topographic images HDF5 file
        grid_size: Size of 2D grid
        mode: 'ssfi' for multi-band or 'single' for single map

    Example:
        >>> transform_preprocessed_file(
        ...     'data/preprocessed/bci_iv_2a/A01T_preprocessed.h5',
        ...     'data/images/topo/A01T_topo_ssfi.h5',
        ...     grid_size=64,
        ...     mode='ssfi'
        ... )
    """
    print("="*60)
    print(f"Topographic Transformation ({mode.upper()})")
    print("="*60)

    # Load preprocessed data
    print(f"Loading: {input_file}")
    with h5py.File(input_file, 'r') as f:
        data = f['data'][:]
        labels = f['labels'][:]

        # Get metadata
        metadata = {key: f.attrs[key] for key in f.attrs.keys()}

        # Get channel names
        ch_names_str = metadata.get('ch_names', '')
        if isinstance(ch_names_str, str):
            # Parse string representation of list
            ch_names = eval(ch_names_str) if ch_names_str else []
        else:
            ch_names = list(ch_names_str)

        # Get sampling frequency
        sfreq = float(metadata.get('sfreq', 250.0))

    print(f"  Data shape: {data.shape}")
    print(f"  Channels: {len(ch_names)}")
    print(f"  Labels: {np.unique(labels)} (counts: {np.bincount(labels)})")

    # Transform
    transformer = TopographicTransformer(
        ch_names=ch_names,
        grid_size=grid_size,
        sfreq=sfreq
    )
    images = transformer.transform_batch(data, mode=mode)

    print(f"\nTransformed images shape: {images.shape}")

    # Add transform info to metadata
    metadata['original_shape'] = str(data.shape)
    metadata['transform_mode'] = mode
    metadata['transform_grid_size'] = grid_size

    # Save
    transformer.save_images(images, labels, output_file, metadata)

    print("\nTransformation complete!")
    print("="*60)


def main():
    """Command-line interface for Topographic transformation."""
    parser = argparse.ArgumentParser(description='Generate Topographic images from preprocessed EEG')
    parser.add_argument('--input', type=str, required=True,
                        help='Input preprocessed HDF5 file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output topographic images HDF5 file')
    parser.add_argument('--grid-size', type=int, default=64,
                        help='Size of 2D grid (NxN)')
    parser.add_argument('--mode', type=str, default='ssfi',
                        choices=['ssfi', 'single'],
                        help='Transformation mode: ssfi (multi-band) or single')

    args = parser.parse_args()

    transform_preprocessed_file(
        input_file=args.input,
        output_file=args.output,
        grid_size=args.grid_size,
        mode=args.mode
    )


if __name__ == '__main__':
    main()
