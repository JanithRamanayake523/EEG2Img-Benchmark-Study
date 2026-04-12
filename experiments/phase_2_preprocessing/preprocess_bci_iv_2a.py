"""
Preprocessing script for BCI Competition IV-2a dataset.

Usage:
    python experiments/scripts/preprocess_bci_iv_2a.py --subject 1 --session T
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import yaml
from src.data.loaders import BCICompetitionIV2aLoader
from src.data.preprocessors import EEGPreprocessor, process_bci_iv_2a_subject


def main():
    parser = argparse.ArgumentParser(description='Preprocess BCI IV-2a dataset')
    parser.add_argument('--subject', type=int, required=True,
                        help='Subject ID (1-9)')
    parser.add_argument('--session', type=str, default='T',
                        choices=['T', 'E'],
                        help='Session: T (training) or E (evaluation)')
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

    # Process subject
    process_bci_iv_2a_subject(
        subject_id=args.subject,
        session=args.session,
        data_path=args.data_path,
        config=preproc_config,
        output_dir=args.output_dir
    )

    print("\n" + "="*60)
    print("SUCCESS: Preprocessing completed!")
    print("="*60)


if __name__ == '__main__':
    main()
