"""
Dataset loading utilities for different EEG datasets.

This module provides dataset-specific loaders for:
- BCI Competition IV-2a (Motor Imagery)
- PhysioNet EEGMMI (Motor Movement/Imagery)
- BCI Competition III - P300 Speller
- SSVEP datasets
"""

import numpy as np
import mne
from pathlib import Path
from typing import Tuple, Dict, Optional
import h5py


class BCICompetitionIV2aLoader:
    """
    Load BCI Competition IV-2a dataset.

    Dataset Info:
    - 4-class motor imagery: left hand, right hand, feet, tongue
    - 22 EEG channels + 3 EOG channels
    - 250 Hz sampling rate
    - 9 subjects (A01-A09)
    - Sessions: T (training), E (evaluation)

    Event Codes (from dataset documentation):
    - 768: Start of trial
    - 769: Cue onset - Class 1 (left hand)
    - 770: Cue onset - Class 2 (right hand)
    - 771: Cue onset - Class 3 (feet)
    - 772: Cue onset - Class 4 (tongue)
    - 783: Cue unknown/undefined
    - 1023: Rejected trial
    - 1072: Eye movements
    - 32766: Start of new run
    """

    # Event ID mapping for motor imagery classes
    EVENT_IDS = {
        'left_hand': 769,
        'right_hand': 770,
        'feet': 771,
        'tongue': 772,
    }

    # Reverse mapping for integer labels
    EVENT_LABELS = {
        769: 0,  # left_hand -> class 0
        770: 1,  # right_hand -> class 1
        771: 2,  # feet -> class 2
        772: 3,  # tongue -> class 3
    }

    @staticmethod
    def load_subject(subject_id: int, data_path: str,
                     session: str = 'T') -> Tuple[mne.io.Raw, np.ndarray, Dict]:
        """
        Load data for one subject.

        Args:
            subject_id: Subject number (1-9)
            data_path: Path to raw data directory (e.g., 'data/raw/bci_iv_2a')
            session: 'T' for training, 'E' for evaluation

        Returns:
            raw: MNE Raw object with EEG data
            events: Event array (n_events, 3) with [sample, duration, event_id]
            event_id: Dictionary mapping event names to IDs

        Example:
            >>> loader = BCICompetitionIV2aLoader()
            >>> raw, events, event_id = loader.load_subject(1, 'data/raw/bci_iv_2a', 'T')
            >>> print(f"Loaded subject 1, {len(raw.ch_names)} channels, {len(events)} events")
        """
        # Construct file path
        file_path = Path(data_path) / f'A{subject_id:02d}{session}.gdf'

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        print(f"Loading: {file_path}")

        # Load GDF file
        raw = mne.io.read_raw_gdf(
            str(file_path),
            preload=True,
            verbose=False
        )

        # Extract events from annotations
        events, event_dict_from_file = mne.events_from_annotations(raw, verbose=False)

        # MNE remaps event codes when reading from annotations
        # We need to find which remapped IDs correspond to our MI cues (769-772)
        # Create reverse mapping from original event codes to MNE-assigned IDs
        reverse_mapping = {}
        for event_name_str, mne_id in event_dict_from_file.items():
            original_code = int(str(event_name_str))  # Convert string to int
            reverse_mapping[original_code] = mne_id

        # Build event_id dict with MNE-assigned IDs
        event_id = {}
        for class_name, original_code in BCICompetitionIV2aLoader.EVENT_IDS.items():
            if original_code in reverse_mapping:
                event_id[class_name] = reverse_mapping[original_code]

        # Filter to keep only motor imagery cue events (mapped versions of 769-772)
        mi_event_codes = list(event_id.values())
        mi_event_mask = np.isin(events[:, 2], mi_event_codes)
        events = events[mi_event_mask]

        print(f"  Subject: A{subject_id:02d}{session}")
        print(f"  Channels: {len(raw.ch_names)} ({raw.info['nchan']} total)")
        print(f"  Sampling rate: {raw.info['sfreq']} Hz")
        print(f"  Duration: {raw.times[-1]:.1f} s")
        print(f"  Motor imagery events: {len(events)}")

        # Count events per class
        for class_name, event_code in event_id.items():
            count = np.sum(events[:, 2] == event_code)
            print(f"    {class_name}: {count} trials")

        return raw, events, event_id

    @staticmethod
    def load_all_subjects(data_path: str, session: str = 'T',
                         subjects: Optional[list] = None) -> Dict:
        """
        Load multiple subjects.

        Args:
            data_path: Path to raw data directory
            session: 'T' or 'E'
            subjects: List of subject IDs (1-9). If None, loads all.

        Returns:
            Dictionary with subject_id as key and (raw, events, event_id) as value

        Example:
            >>> loader = BCICompetitionIV2aLoader()
            >>> data = loader.load_all_subjects('data/raw/bci_iv_2a', session='T', subjects=[1, 2])
            >>> print(f"Loaded {len(data)} subjects")
        """
        if subjects is None:
            subjects = list(range(1, 10))  # All 9 subjects

        all_data = {}

        for subject_id in subjects:
            try:
                raw, events, event_id = BCICompetitionIV2aLoader.load_subject(
                    subject_id, data_path, session
                )
                all_data[subject_id] = {
                    'raw': raw,
                    'events': events,
                    'event_id': event_id
                }
            except Exception as e:
                print(f"Error loading subject {subject_id}: {e}")

        return all_data


class PhysioNetLoader:
    """Load PhysioNet EEGMMI dataset."""

    RUNS = {
        'baseline_eyes_open': [1],
        'baseline_eyes_closed': [2],
        'motor_execution_lr': [3, 7, 11],  # Left/right hand
        'motor_imagery_lr': [4, 8, 12],
        'motor_execution_fists_feet': [5, 9, 13],
        'motor_imagery_fists_feet': [6, 10, 14],
    }

    @staticmethod
    def load_subject(subject_id: int, data_path: str,
                     task: str = 'motor_imagery_lr') -> Tuple[mne.io.Raw, np.ndarray, Dict]:
        """
        Load data for specific task.

        Note: This requires MNE's PhysioNet downloader.
        To be implemented when needed.
        """
        raise NotImplementedError("PhysioNet loader to be implemented in future phases")


def load_preprocessed(file_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load preprocessed HDF5 data.

    Args:
        file_path: Path to preprocessed HDF5 file

    Returns:
        data: (n_epochs, n_channels, n_times) array
        labels: (n_epochs,) array with class labels
        metadata: Dictionary with dataset info

    Example:
        >>> data, labels, metadata = load_preprocessed('data/preprocessed/bci_iv_2a/A01T_preprocessed.h5')
        >>> print(f"Shape: {data.shape}, Classes: {np.unique(labels)}")
    """
    with h5py.File(file_path, 'r') as f:
        data = f['data'][:]
        labels = f['labels'][:]

        metadata = {key: f.attrs[key] for key in f.attrs.keys()}

    return data, labels, metadata


# Example usage
if __name__ == '__main__':
    # Test BCI IV-2a loader
    loader = BCICompetitionIV2aLoader()

    # Load subject 1 training data
    raw, events, event_id = loader.load_subject(
        subject_id=1,
        data_path='data/raw/bci_iv_2a',
        session='T'
    )

    print("\n" + "="*60)
    print("BCI IV-2a Loader Test")
    print("="*60)
    print(f"Raw data shape: {raw.get_data().shape}")
    print(f"Events shape: {events.shape}")
    print(f"Event IDs: {event_id}")
    print(f"Sample event codes: {np.unique(events[:, 2])}")
