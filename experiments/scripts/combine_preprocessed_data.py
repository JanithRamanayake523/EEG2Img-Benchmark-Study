"""
Combine individual preprocessed BCI IV-2a subject files into a single HDF5 file.

This script takes the per-subject preprocessed files from data/preprocessed/bci_iv_2a/
and combines them into a single data/BCI_IV_2a.hdf5 file for easier access.

Usage:
    python experiments/scripts/combine_preprocessed_data.py
"""

import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def combine_preprocessed_files():
    """Combine per-subject preprocessed files into single HDF5."""

    input_dir = Path('data/preprocessed/bci_iv_2a')
    output_file = Path('data/BCI_IV_2a.hdf5')

    # Check if input directory exists
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return False

    # Find all preprocessed files
    preprocessed_files = sorted(input_dir.glob('*.h5'))

    if not preprocessed_files:
        print(f"❌ No preprocessed files found in {input_dir}")
        return False

    print(f"Found {len(preprocessed_files)} preprocessed files")
    print(f"Output file: {output_file}")
    print()

    # Create combined file
    try:
        with h5py.File(output_file, 'w') as out_file:
            for file_path in tqdm(preprocessed_files, desc="Combining files"):
                subject_name = file_path.stem.split('_')[0]  # e.g., 'A01T' from 'A01T_preprocessed'

                # Read source file
                with h5py.File(file_path, 'r') as in_file:
                    # Get data and labels
                    signals = in_file['data'][:]
                    labels = in_file['labels'][:]

                    # Create subject group
                    subject_group = out_file.create_group(f'subject_{subject_name}')

                    # Store data directly in subject group
                    subject_group.create_dataset('signals', data=signals, compression='gzip')
                    subject_group.create_dataset('labels', data=labels)

        # Verify output file
        with h5py.File(output_file, 'r') as f:
            subjects = list(f.keys())
            total_trials = 0
            for subject in subjects:
                count = len(f[f'{subject}/labels'])
                total_trials += count

        print()
        print("SUCCESS: Combined {0} subjects".format(len(subjects)))
        print("   Subjects: {0}".format(', '.join(subjects)))
        print("   Total trials: {0:,}".format(total_trials))
        print("   File size: {0:.1f} MB".format(output_file.stat().st_size / 1e6))

        return True

    except Exception as e:
        print("ERROR combining files: {0}".format(e))
        return False


if __name__ == '__main__':
    success = combine_preprocessed_files()
    sys.exit(0 if success else 1)
