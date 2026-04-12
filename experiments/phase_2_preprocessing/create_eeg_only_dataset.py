"""
Create EEG-only dataset by extracting 22 EEG channels from preprocessed data.

The preprocessed data contains 25 channels (22 EEG + 3 EOG).
This script extracts only the 22 EEG channels as specified in the research plan.

Channel structure:
- Indices 0-21: EEG channels (keep these)
- Indices 22-24: EOG channels (remove these)

Usage:
    python experiments/scripts/create_eeg_only_dataset.py
"""

import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def create_eeg_only_dataset():
    """Extract 22 EEG channels from 25-channel preprocessed data."""

    input_file = Path('data/BCI_IV_2a.hdf5')
    output_file = Path('data/BCI_IV_2a_EEG_only.hdf5')

    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        return False

    print("="*70)
    print("Creating EEG-Only Dataset (22 channels)")
    print("="*70)
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print()

    try:
        with h5py.File(input_file, 'r') as in_file:
            subjects = list(in_file.keys())

            # Get channel names from first subject to identify EEG channels
            first_subject = subjects[0]
            sample_data = in_file[first_subject]['signals'][:]
            print(f"Original data shape: {sample_data.shape}")
            print(f"  Channels: {sample_data.shape[1]}")
            print(f"  Timepoints: {sample_data.shape[2]}")
            print()

            # EEG channels are indices 0-21 (first 22 channels)
            # EOG channels are indices 22-24 (last 3 channels)
            eeg_channel_indices = list(range(22))

            with h5py.File(output_file, 'w') as out_file:
                total_trials = 0

                for subject in tqdm(subjects, desc="Processing subjects"):
                    # Load signals and labels
                    signals = in_file[subject]['signals'][:]
                    labels = in_file[subject]['labels'][:]

                    # Extract only EEG channels (indices 0-21)
                    eeg_signals = signals[:, eeg_channel_indices, :]

                    # Create subject group
                    subject_group = out_file.create_group(subject)

                    # Store EEG-only data
                    subject_group.create_dataset(
                        'signals',
                        data=eeg_signals,
                        compression='gzip',
                        compression_opts=4
                    )
                    subject_group.create_dataset('labels', data=labels)

                    # Store metadata
                    subject_group.attrs['n_channels'] = 22
                    subject_group.attrs['n_trials'] = len(labels)
                    subject_group.attrs['n_timepoints'] = eeg_signals.shape[2]
                    subject_group.attrs['sfreq'] = 250.0

                    total_trials += len(labels)

                # Store global metadata
                out_file.attrs['n_subjects'] = len(subjects)
                out_file.attrs['total_trials'] = total_trials
                out_file.attrs['n_channels'] = 22
                out_file.attrs['channel_type'] = 'EEG-only'
                out_file.attrs['sfreq'] = 250.0
                out_file.attrs['n_classes'] = 4
                out_file.attrs['preprocessing'] = 'ICA, filtering (0.5-40 Hz), epoching, normalization'
                out_file.attrs['eog_removed'] = 'Yes (3 EOG channels excluded)'

        # Verify output
        print()
        print("="*70)
        print("VERIFICATION")
        print("="*70)

        with h5py.File(output_file, 'r') as f:
            subjects = list(f.keys())
            print(f"Subjects: {len(subjects)}")

            for subject in subjects:
                signals_shape = f[subject]['signals'].shape
                labels_shape = f[subject]['labels'].shape
                print(f"  {subject}:")
                print(f"    signals: {signals_shape}")
                print(f"    labels:  {labels_shape}")

            print()
            print(f"Total trials: {f.attrs['total_trials']:,}")
            print(f"Channels: {f.attrs['n_channels']} (EEG only)")
            print(f"Channel type: {f.attrs['channel_type']}")
            print(f"Sampling rate: {f.attrs['sfreq']} Hz")
            print(f"Classes: {f.attrs['n_classes']}")

        print()
        print("="*70)
        print("✓ SUCCESS: EEG-only dataset created!")
        print("="*70)
        print(f"File: {output_file}")
        print(f"Size: {output_file.stat().st_size / 1e6:.1f} MB")
        print()
        print("Next steps:")
        print("  1. Update Phase 3 scripts to use this new file")
        print("  2. Run validation to verify 22-channel models work")
        print("="*70)

        return True

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = create_eeg_only_dataset()
    sys.exit(0 if success else 1)
