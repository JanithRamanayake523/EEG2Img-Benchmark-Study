"""
Recurrence Plot (RP) transformations for EEG time-series.

Implements phase space reconstruction and recurrence analysis for
capturing complex nonlinear dynamics in EEG signals.

References:
- Eckmann et al. (1987): "Recurrence Plots of Dynamical Systems"
- Marwan et al. (2007): "Recurrence plots for the analysis of complex systems"
- Hao et al. (2021): "Deep learning with CNNs for EEG using recurrence plots"
"""

import numpy as np
from typing import Tuple, Optional
import argparse
import h5py
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform


class RecurrencePlotTransformer:
    """
    Transform EEG time-series to Recurrence Plot images.

    Recurrence plots visualize recurrences in phase space, revealing
    hidden patterns and nonlinear dynamics in time series data.
    """

    def __init__(self, embedding_dim: int = 3, time_delay: int = 1,
                 threshold_percentile: float = 10.0,
                 distance_metric: str = 'euclidean'):
        """
        Initialize Recurrence Plot transformer.

        Args:
            embedding_dim: Phase space embedding dimension (m). Common: 3-10
            time_delay: Time delay for embedding (τ). Common: 1 or auto-computed
            threshold_percentile: Threshold as percentile of distances (ε).
                                 Common: 10% for ~10% recurrence rate
            distance_metric: Distance metric for phase space. Common: 'euclidean'

        Example:
            >>> transformer = RecurrencePlotTransformer(embedding_dim=3, time_delay=1)
        """
        self.embedding_dim = embedding_dim
        self.time_delay = time_delay
        self.threshold_percentile = threshold_percentile
        self.distance_metric = distance_metric

    def phase_space_reconstruction(self, x: np.ndarray) -> np.ndarray:
        """
        Reconstruct phase space using time-delay embedding.

        Args:
            x: 1D time series (n_times,)

        Returns:
            Embedded trajectory: (n_vectors, embedding_dim)
                where n_vectors = n_times - (embedding_dim - 1) * time_delay

        Example:
            >>> x = np.sin(np.linspace(0, 4*np.pi, 100))
            >>> embedded = transformer.phase_space_reconstruction(x)
            >>> print(embedded.shape)  # (98, 3) for m=3, τ=1
        """
        n_times = len(x)
        n_vectors = n_times - (self.embedding_dim - 1) * self.time_delay

        if n_vectors <= 0:
            raise ValueError(
                f"Time series too short for embedding. "
                f"Need at least {(self.embedding_dim - 1) * self.time_delay + 1} points, "
                f"got {n_times}"
            )

        # Create embedded vectors
        embedded = np.zeros((n_vectors, self.embedding_dim))
        for i in range(self.embedding_dim):
            start_idx = i * self.time_delay
            end_idx = start_idx + n_vectors
            embedded[:, i] = x[start_idx:end_idx]

        return embedded

    def compute_distance_matrix(self, embedded: np.ndarray) -> np.ndarray:
        """
        Compute pairwise distance matrix in phase space.

        Args:
            embedded: Phase space trajectory (n_vectors, embedding_dim)

        Returns:
            Distance matrix: (n_vectors, n_vectors)
        """
        # Compute pairwise distances
        distances = squareform(pdist(embedded, metric=self.distance_metric))

        return distances

    def apply_threshold(self, distances: np.ndarray) -> np.ndarray:
        """
        Apply threshold to create binary recurrence plot.

        Args:
            distances: Distance matrix (n_vectors, n_vectors)

        Returns:
            Binary recurrence plot: (n_vectors, n_vectors)
                1 where distance < threshold, 0 otherwise
        """
        # Compute threshold from percentile
        threshold = np.percentile(distances, self.threshold_percentile)

        # Create binary recurrence plot
        rp = (distances <= threshold).astype(np.float32)

        return rp

    def resize_to_square(self, rp: np.ndarray, target_size: int) -> np.ndarray:
        """
        Resize recurrence plot to target image size.

        Args:
            rp: Recurrence plot (n_vectors, n_vectors)
            target_size: Target image size (NxN)

        Returns:
            Resized recurrence plot: (target_size, target_size)
        """
        from scipy.ndimage import zoom

        current_size = rp.shape[0]

        if current_size == target_size:
            return rp

        # Compute zoom factor
        zoom_factor = target_size / current_size

        # Resize using nearest-neighbor to preserve binary structure
        rp_resized = zoom(rp, zoom_factor, order=0)  # order=0 for nearest

        return rp_resized

    def transform_channel(self, channel_data: np.ndarray,
                         target_size: int = 128) -> np.ndarray:
        """
        Transform single channel to recurrence plot.

        Args:
            channel_data: 1D time series (n_times,)
            target_size: Target image size (NxN)

        Returns:
            Recurrence plot image: (target_size, target_size)

        Example:
            >>> channel = np.random.randn(751)
            >>> rp = transformer.transform_channel(channel, target_size=128)
            >>> print(rp.shape)  # (128, 128)
        """
        # Phase space reconstruction
        embedded = self.phase_space_reconstruction(channel_data)

        # Compute distance matrix
        distances = self.compute_distance_matrix(embedded)

        # Apply threshold to get binary RP
        rp = self.apply_threshold(distances)

        # Resize to target size
        rp_resized = self.resize_to_square(rp, target_size)

        return rp_resized

    def transform_epoch(self, epoch: np.ndarray,
                       target_size: int = 128) -> np.ndarray:
        """
        Transform multi-channel epoch to recurrence plots.

        Args:
            epoch: (n_channels, n_times) array
            target_size: Target image size (NxN)

        Returns:
            rp_images: (n_channels, target_size, target_size) array

        Example:
            >>> epoch = np.random.randn(22, 751)  # 22 channels, 751 timepoints
            >>> rp_images = transformer.transform_epoch(epoch, target_size=128)
            >>> print(rp_images.shape)  # (22, 128, 128)
        """
        n_channels = epoch.shape[0]
        rp_images = np.zeros((n_channels, target_size, target_size))

        for ch in range(n_channels):
            rp_images[ch] = self.transform_channel(epoch[ch], target_size)

        return rp_images

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

        for i in tqdm(range(n_epochs), desc=f"RP-m{self.embedding_dim}"):
            rp_image = self.transform_epoch(data[i], target_size)

            if strategy == 'per_channel':
                # Keep all channels as separate image layers
                results.append(rp_image)
            elif strategy == 'average':
                # Average across channels to get single image
                avg_image = rp_image.mean(axis=0, keepdims=True)
                results.append(avg_image)
            elif strategy == 'first_channel':
                # Use only first channel
                results.append(rp_image[0:1])
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

        return np.array(results)

    def save_images(self, images: np.ndarray, labels: np.ndarray,
                   output_path: str, metadata: Optional[dict] = None):
        """
        Save recurrence plot images to HDF5.

        Args:
            images: Transformed images array
            labels: Class labels (n_epochs,)
            output_path: Path to save HDF5 file
            metadata: Optional metadata dictionary

        Example:
            >>> transformer.save_images(images, labels, 'data/images/rp/A01T_rp.h5')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Saving to: {output_path}")

        with h5py.File(output_path, 'w') as f:
            f.create_dataset('images', data=images, compression='gzip')
            f.create_dataset('labels', data=labels)

            # Save transform metadata
            f.attrs['transform'] = 'recurrence_plot'
            f.attrs['embedding_dim'] = self.embedding_dim
            f.attrs['time_delay'] = self.time_delay
            f.attrs['threshold_percentile'] = self.threshold_percentile
            f.attrs['distance_metric'] = self.distance_metric

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
        Load recurrence plot images from HDF5.

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
                                embedding_dim: int = 3,
                                time_delay: int = 1,
                                threshold_percentile: float = 10.0,
                                target_size: int = 128,
                                strategy: str = 'per_channel'):
    """
    Transform a preprocessed HDF5 file to recurrence plot images.

    Args:
        input_file: Path to preprocessed HDF5 file
        output_file: Path to output RP images HDF5 file
        embedding_dim: Phase space embedding dimension (m)
        time_delay: Time delay for embedding (τ)
        threshold_percentile: Threshold percentile for recurrence
        target_size: Target image size (NxN)
        strategy: Multi-channel strategy

    Example:
        >>> transform_preprocessed_file(
        ...     'data/preprocessed/bci_iv_2a/A01T_preprocessed.h5',
        ...     'data/images/rp/A01T_rp_m3.h5',
        ...     embedding_dim=3,
        ...     time_delay=1
        ... )
    """
    print("="*60)
    print(f"Recurrence Plot Transformation: m={embedding_dim}, τ={time_delay}")
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
    transformer = RecurrencePlotTransformer(
        embedding_dim=embedding_dim,
        time_delay=time_delay,
        threshold_percentile=threshold_percentile
    )
    images = transformer.transform_batch(data, target_size, strategy)

    print(f"\nTransformed images shape: {images.shape}")

    # Add transform info to metadata
    metadata['original_shape'] = str(data.shape)
    metadata['transform_embedding_dim'] = embedding_dim
    metadata['transform_time_delay'] = time_delay
    metadata['transform_threshold_percentile'] = threshold_percentile
    metadata['transform_strategy'] = strategy

    # Save
    transformer.save_images(images, labels, output_file, metadata)

    print("\nTransformation complete!")
    print("="*60)


def main():
    """Command-line interface for Recurrence Plot transformation."""
    parser = argparse.ArgumentParser(description='Generate Recurrence Plot images from preprocessed EEG')
    parser.add_argument('--input', type=str, required=True,
                        help='Input preprocessed HDF5 file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output RP images HDF5 file')
    parser.add_argument('--embedding-dim', type=int, default=3,
                        help='Phase space embedding dimension (m)')
    parser.add_argument('--time-delay', type=int, default=1,
                        help='Time delay for embedding (τ)')
    parser.add_argument('--threshold-percentile', type=float, default=10.0,
                        help='Threshold percentile for recurrence (0-100)')
    parser.add_argument('--size', type=int, default=128,
                        help='Target image size (NxN)')
    parser.add_argument('--strategy', type=str, default='per_channel',
                        choices=['per_channel', 'average', 'first_channel'],
                        help='Multi-channel strategy')

    args = parser.parse_args()

    transform_preprocessed_file(
        input_file=args.input,
        output_file=args.output,
        embedding_dim=args.embedding_dim,
        time_delay=args.time_delay,
        threshold_percentile=args.threshold_percentile,
        target_size=args.size,
        strategy=args.strategy
    )


if __name__ == '__main__':
    main()
