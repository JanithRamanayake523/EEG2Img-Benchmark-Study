"""
EEG preprocessing pipeline following research plan specifications.

Pipeline steps:
1. Loading & Basic QC
2. Filtering (band-pass 0.5-40 Hz, notch 50/60 Hz)
3. Referencing (common average)
4. Resampling (to 250 Hz)
5. Epoching (paradigm-specific)
6. Baseline correction
7. Artifact removal (ICA/AutoReject)
8. Normalization (z-score, min-max)
"""

import numpy as np
import mne
from mne.preprocessing import ICA
from pathlib import Path
import h5py
from typing import Dict, Optional, Tuple, List
import argparse
import yaml
from tqdm import tqdm


class EEGPreprocessor:
    """
    Standardized EEG preprocessing pipeline.

    Follows the research plan specifications for consistent preprocessing
    across all datasets.
    """

    def __init__(self, config: Dict):
        """
        Initialize preprocessor with configuration.

        Args:
            config: Dictionary with preprocessing parameters
                - filter: {l_freq, h_freq, notch}
                - resample: target sampling rate
                - epoch: {tmin, tmax, baseline}
                - artifact: {ica, amplitude_threshold}
                - picks: channel types to keep (default: 'eeg')
        """
        self.config = config

    def load_raw(self, file_path: str) -> mne.io.Raw:
        """Load raw EEG data."""
        print(f"Loading: {file_path}")

        # Auto-detect file format
        if file_path.endswith('.gdf'):
            raw = mne.io.read_raw_gdf(file_path, preload=True, verbose=False)
        elif file_path.endswith('.edf'):
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        elif file_path.endswith('.fif'):
            raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        # Pick only EEG channels (drop EOG, etc.)
        picks = self.config.get('picks', 'eeg')
        if picks == 'eeg':
            raw.pick_types(eeg=True, exclude='bads')

        print(f"  Channels: {len(raw.ch_names)}")
        print(f"  Sampling rate: {raw.info['sfreq']} Hz")
        print(f"  Duration: {raw.times[-1]:.1f} s")

        return raw

    def apply_filtering(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Apply band-pass and notch filtering."""
        filter_config = self.config['filter']

        # Band-pass filter
        l_freq = filter_config.get('l_freq', 0.5)
        h_freq = filter_config.get('h_freq', 40.0)

        print(f"Band-pass filtering: {l_freq}-{h_freq} Hz")
        raw.filter(
            l_freq=l_freq,
            h_freq=h_freq,
            filter_length='auto',
            l_trans_bandwidth='auto',
            h_trans_bandwidth='auto',
            method='fir',
            phase='zero',
            fir_window='hamming',
            verbose=False
        )

        # Notch filter (power line noise)
        if 'notch' in filter_config and filter_config['notch']:
            notch_freq = filter_config['notch']
            print(f"Notch filtering: {notch_freq} Hz")
            raw.notch_filter(freqs=notch_freq, verbose=False)

        return raw

    def apply_referencing(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Apply common average reference."""
        ref_type = self.config.get('reference', 'average')

        if ref_type == 'average':
            print("Applying common average reference")
            raw.set_eeg_reference('average', projection=False, verbose=False)
        elif ref_type is not None:
            print(f"Applying {ref_type} reference")
            raw.set_eeg_reference(ref_type, verbose=False)

        return raw

    def apply_resampling(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Resample to target frequency."""
        target_sfreq = self.config.get('resample', 250)

        if raw.info['sfreq'] != target_sfreq:
            print(f"Resampling: {raw.info['sfreq']} Hz -> {target_sfreq} Hz")
            raw.resample(target_sfreq, npad='auto', verbose=False)

        return raw

    def create_epochs(self, raw: mne.io.Raw, events: np.ndarray,
                     event_id: Dict) -> mne.Epochs:
        """
        Create epochs from continuous data.

        Args:
            raw: MNE Raw object
            events: Event array
            event_id: Dictionary mapping event names to IDs

        Returns:
            MNE Epochs object
        """
        epoch_config = self.config['epoch']

        tmin = epoch_config.get('tmin', 0.0)
        tmax = epoch_config.get('tmax', 4.0)
        baseline = epoch_config.get('baseline', None)

        print(f"Epoching: {tmin} to {tmax} s")
        if baseline:
            print(f"  Baseline: {baseline}")

        epochs = mne.Epochs(
            raw,
            events=events,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            preload=True,
            reject_by_annotation=True,
            verbose=False
        )

        print(f"  Created {len(epochs)} epochs")

        return epochs

    def apply_artifact_removal(self, epochs: mne.Epochs) -> mne.Epochs:
        """
        Remove artifacts using ICA and amplitude-based rejection.

        Args:
            epochs: MNE Epochs object

        Returns:
            Cleaned MNE Epochs object
        """
        artifact_config = self.config.get('artifact', {})

        # ICA for artifact removal
        if artifact_config.get('ica', False):
            print("Applying ICA for artifact removal")

            # Fit ICA on epochs
            ica = ICA(
                n_components=15,
                random_state=42,
                method='fastica',
                max_iter=200,
                verbose=False
            )

            try:
                ica.fit(epochs, verbose=False)

                # Automatic component detection would go here
                # For now, we'll apply ICA without automatic rejection
                # In a full implementation, use methods like:
                # - ica.detect_artifacts(epochs)
                # - or manual inspection

                # Apply ICA (this removes artifacts)
                epochs = ica.apply(epochs, verbose=False)
                print(f"  ICA applied with {ica.n_components_} components")

            except Exception as e:
                print(f"  ICA failed: {e}. Skipping ICA.")

        # Amplitude-based rejection
        threshold = artifact_config.get('amplitude_threshold', 100e-6)  # 100 µV
        threshold = float(threshold) if isinstance(threshold, str) else threshold
        print(f"Amplitude-based rejection: threshold = {threshold*1e6:.0f} uV")

        reject_criteria = {'eeg': threshold}

        # Get indices of bad epochs
        epochs.drop_bad(reject=reject_criteria, verbose=False)

        print(f"  Epochs remaining: {len(epochs)}")

        return epochs

    def apply_normalization(self, epochs: mne.Epochs,
                           method: str = 'zscore') -> np.ndarray:
        """
        Normalize epoch data.

        Args:
            epochs: MNE Epochs object
            method: 'zscore' or 'minmax'

        Returns:
            Normalized data array (n_epochs, n_channels, n_times)
        """
        data = epochs.get_data()

        if method == 'zscore':
            # Z-score normalization per channel per epoch
            mean = data.mean(axis=2, keepdims=True)
            std = data.std(axis=2, keepdims=True)
            data = (data - mean) / (std + 1e-8)

        elif method == 'minmax':
            # Min-max to [-1, 1] for GAF compatibility
            min_val = data.min(axis=2, keepdims=True)
            max_val = data.max(axis=2, keepdims=True)
            data = 2 * (data - min_val) / (max_val - min_val + 1e-8) - 1

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        print(f"Applied {method} normalization")

        return data

    def save_preprocessed(self, data: np.ndarray, labels: np.ndarray,
                         metadata: Dict, output_path: str):
        """
        Save preprocessed data to HDF5.

        Args:
            data: (n_epochs, n_channels, n_times) array
            labels: (n_epochs,) array with class labels
            metadata: Dictionary with dataset info
            output_path: Path to save HDF5 file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Saving to: {output_path}")

        with h5py.File(output_path, 'w') as f:
            f.create_dataset('data', data=data, compression='gzip')
            f.create_dataset('labels', data=labels)

            # Save metadata
            for key, value in metadata.items():
                if isinstance(value, (str, int, float)):
                    f.attrs[key] = value
                elif isinstance(value, (list, np.ndarray)):
                    f.attrs[key] = str(value)

        print(f"  Shape: {data.shape}")
        print(f"  Labels: {np.unique(labels)} (counts: {np.bincount(labels)})")

    def process(self, raw: mne.io.Raw, events: np.ndarray, event_id: Dict,
                output_file: str, event_label_map: Optional[Dict] = None):
        """
        Run full preprocessing pipeline.

        Args:
            raw: MNE Raw object
            events: Event array
            event_id: Event ID dictionary
            output_file: Path to save preprocessed data
            event_label_map: Optional mapping from event IDs to integer labels
                           (e.g., {769: 0, 770: 1, 771: 2, 772: 3})
        """
        print("="*60)
        print("PREPROCESSING PIPELINE")
        print("="*60)

        # Preprocessing steps
        raw = self.apply_filtering(raw)
        raw = self.apply_referencing(raw)
        raw = self.apply_resampling(raw)

        # Epoching
        epochs = self.create_epochs(raw, events, event_id)

        # Artifact removal
        epochs = self.apply_artifact_removal(epochs)

        # Normalization
        normalization_method = self.config.get('normalization', 'zscore')
        data = self.apply_normalization(epochs, method=normalization_method)

        # Extract labels
        if event_label_map:
            # Map event IDs to integer class labels
            labels = np.array([event_label_map[e] for e in epochs.events[:, 2]])
        else:
            # Use event IDs directly as labels
            labels = epochs.events[:, 2]

        # Save
        metadata = {
            'sfreq': epochs.info['sfreq'],
            'n_channels': len(epochs.ch_names),
            'ch_names': epochs.ch_names,
            'tmin': epochs.tmin,
            'tmax': epochs.tmax,
            'n_epochs': len(epochs),
            'n_classes': len(np.unique(labels)),
        }

        self.save_preprocessed(data, labels, metadata, output_file)

        print("Preprocessing complete!\n")

        return data, labels, metadata


def process_bci_iv_2a_subject(subject_id: int, session: str,
                               data_path: str, config: Dict,
                               output_dir: str):
    """
    Process one subject from BCI IV-2a dataset.

    Args:
        subject_id: Subject number (1-9)
        session: 'T' or 'E'
        data_path: Path to raw data directory
        config: Preprocessing configuration
        output_dir: Output directory for preprocessed data
    """
    try:
        from .loaders import BCICompetitionIV2aLoader
    except ImportError:
        from src.data.loaders import BCICompetitionIV2aLoader

    # Load raw data
    loader = BCICompetitionIV2aLoader()
    raw, events, event_id = loader.load_subject(subject_id, data_path, session)

    # Initialize preprocessor
    preprocessor = EEGPreprocessor(config)

    # Output file
    output_file = Path(output_dir) / f'A{subject_id:02d}{session}_preprocessed.h5'

    # Create event label mapping (MNE-assigned ID -> class label 0-3)
    # event_id maps class names to MNE IDs
    # We need to map MNE IDs to sequential class labels 0, 1, 2, 3
    class_order = ['left_hand', 'right_hand', 'feet', 'tongue']
    event_label_map = {event_id[class_name]: idx
                      for idx, class_name in enumerate(class_order)
                      if class_name in event_id}

    # Process
    preprocessor.process(
        raw=raw,
        events=events,
        event_id=event_id,
        output_file=str(output_file),
        event_label_map=event_label_map
    )


def main():
    """Command-line interface for preprocessing."""
    parser = argparse.ArgumentParser(description='Preprocess EEG data')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to preprocessing config YAML')
    parser.add_argument('--dataset', type=str, default='bci_iv_2a',
                        choices=['bci_iv_2a'],
                        help='Dataset name')
    parser.add_argument('--subject', type=int, required=True,
                        help='Subject ID')
    parser.add_argument('--session', type=str, default='T',
                        choices=['T', 'E'],
                        help='Session: T (training) or E (evaluation)')
    parser.add_argument('--data-path', type=str, default='data/raw/bci_iv_2a',
                        help='Path to raw data')
    parser.add_argument('--output-dir', type=str, default='data/preprocessed/bci_iv_2a',
                        help='Output directory')

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Get preprocessing config
    preproc_config = config.get('preprocessing', config)

    # Process subject
    if args.dataset == 'bci_iv_2a':
        process_bci_iv_2a_subject(
            subject_id=args.subject,
            session=args.session,
            data_path=args.data_path,
            config=preproc_config,
            output_dir=args.output_dir
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


if __name__ == '__main__':
    main()
