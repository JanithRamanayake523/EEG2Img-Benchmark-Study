"""
Batch preprocessing script for all BCI Competition IV-2a subjects.

Usage:
    python experiments/scripts/preprocess_all_bci_iv_2a.py --session T
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import yaml
from tqdm import tqdm
from src.data.preprocessors import process_bci_iv_2a_subject


def main():
    parser = argparse.ArgumentParser(description='Preprocess all BCI IV-2a subjects')
    parser.add_argument('--session', type=str, default='T',
                        choices=['T', 'E', 'both'],
                        help='Session to process: T, E, or both')
    parser.add_argument('--subjects', type=int, nargs='+',
                        default=list(range(1, 10)),
                        help='Subject IDs to process (default: all 1-9)')
    parser.add_argument('--config', type=str,
                        default='experiments/configs/preprocessing_bci_iv_2a.yaml',
                        help='Path to config file')
    parser.add_argument('--data-path', type=str,
                        default='data/raw/bci_iv_2a',
                        help='Path to raw data')
    parser.add_argument('--output-dir', type=str,
                        default='data/preprocessed/bci_iv_2a',
                        help='Output directory')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    preproc_config = config['preprocessing']

    # Determine sessions to process
    if args.session == 'both':
        sessions = ['T', 'E']
    else:
        sessions = [args.session]

    # Process each subject
    total = len(args.subjects) * len(sessions)
    print(f"\n{'='*60}")
    print(f"BCI IV-2a Batch Preprocessing")
    print(f"{'='*60}")
    print(f"Subjects: {args.subjects}")
    print(f"Sessions: {sessions}")
    print(f"Total files to process: {total}")
    print(f"{'='*60}\n")

    success_count = 0
    failed = []

    with tqdm(total=total, desc="Processing") as pbar:
        for subject_id in args.subjects:
            for session in sessions:
                try:
                    pbar.set_description(f"Subject {subject_id}{session}")

                    process_bci_iv_2a_subject(
                        subject_id=subject_id,
                        session=session,
                        data_path=args.data_path,
                        config=preproc_config,
                        output_dir=args.output_dir
                    )

                    success_count += 1

                except Exception as e:
                    print(f"\nERROR processing Subject {subject_id}{session}: {e}")
                    failed.append(f"{subject_id}{session}")

                finally:
                    pbar.update(1)

    # Summary
    print(f"\n{'='*60}")
    print("BATCH PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Successful: {success_count}/{total}")
    if failed:
        print(f"Failed: {len(failed)} -> {failed}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
