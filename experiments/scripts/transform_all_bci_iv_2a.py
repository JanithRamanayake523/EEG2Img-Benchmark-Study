"""
Batch transformation script for all BCI Competition IV-2a subjects.

Applies all time-series-to-image transformations to preprocessed data.

Usage:
    python experiments/scripts/transform_all_bci_iv_2a.py --session T --transforms all
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
from tqdm import tqdm
from src.transforms.gaf import transform_preprocessed_file as transform_gaf
from src.transforms.mtf import transform_preprocessed_file as transform_mtf
from src.transforms.recurrence import transform_preprocessed_file as transform_rp
from src.transforms.spectrogram import transform_preprocessed_file as transform_spec
from src.transforms.scalogram import transform_preprocessed_file as transform_cwt
from src.transforms.topographic import transform_preprocessed_file as transform_topo


def main():
    parser = argparse.ArgumentParser(description='Transform all BCI IV-2a subjects')
    parser.add_argument('--session', type=str, default='T',
                        choices=['T', 'E', 'both'],
                        help='Session to process: T, E, or both')
    parser.add_argument('--subjects', type=int, nargs='+',
                        default=list(range(1, 10)),
                        help='Subject IDs to process (default: all 1-9)')
    parser.add_argument('--transforms', type=str, nargs='+',
                        default=['all'],
                        choices=['all', 'gaf', 'mtf', 'rp', 'spec', 'cwt', 'topo'],
                        help='Transformations to apply (default: all)')
    parser.add_argument('--preprocessed-dir', type=str,
                        default='data/preprocessed/bci_iv_2a',
                        help='Preprocessed data directory')
    parser.add_argument('--output-dir', type=str,
                        default='data/images',
                        help='Output images directory')
    parser.add_argument('--image-size', type=int, default=64,
                        help='Image size (NxN)')

    args = parser.parse_args()

    # Determine sessions to process
    if args.session == 'both':
        sessions = ['T', 'E']
    else:
        sessions = [args.session]

    # Determine transformations to apply
    if 'all' in args.transforms:
        transforms = ['gaf', 'mtf', 'rp', 'spec', 'cwt', 'topo']
    else:
        transforms = args.transforms

    # Count total operations
    total = len(args.subjects) * len(sessions) * len(transforms)
    print(f"\n{'='*60}")
    print(f"BCI IV-2a Batch Transformation")
    print(f"{'='*60}")
    print(f"Subjects: {args.subjects}")
    print(f"Sessions: {sessions}")
    print(f"Transforms: {transforms}")
    print(f"Total transformations: {total}")
    print(f"{'='*60}\n")

    success_count = 0
    failed = []

    with tqdm(total=total, desc="Processing") as pbar:
        for subject_id in args.subjects:
            for session in sessions:
                # Input file
                input_file = Path(args.preprocessed_dir) / f'A{subject_id:02d}{session}_preprocessed.h5'

                if not input_file.exists():
                    print(f"\nWARNING: {input_file} not found, skipping...")
                    pbar.update(len(transforms))
                    continue

                for transform in transforms:
                    try:
                        pbar.set_description(f"A{subject_id:02d}{session}-{transform.upper()}")

                        if transform == 'gaf':
                            # GAF Summation (GASF)
                            output_file = Path(args.output_dir) / 'gaf' / f'A{subject_id:02d}{session}_gaf_summation.h5'
                            transform_gaf(
                                str(input_file),
                                str(output_file),
                                image_size=args.image_size,
                                method='summation',
                                strategy='per_channel'
                            )

                        elif transform == 'mtf':
                            # MTF with Q=8 bins
                            output_file = Path(args.output_dir) / 'mtf' / f'A{subject_id:02d}{session}_mtf_q8.h5'
                            transform_mtf(
                                str(input_file),
                                str(output_file),
                                image_size=args.image_size,
                                n_bins=8,
                                strategy='per_channel'
                            )

                        elif transform == 'rp':
                            # Recurrence Plot (m=3, τ=1)
                            output_file = Path(args.output_dir) / 'rp' / f'A{subject_id:02d}{session}_rp_m3.h5'
                            transform_rp(
                                str(input_file),
                                str(output_file),
                                embedding_dim=3,
                                time_delay=1,
                                threshold_percentile=10.0,
                                target_size=args.image_size,
                                strategy='per_channel'
                            )

                        elif transform == 'spec':
                            # Spectrogram (STFT)
                            output_file = Path(args.output_dir) / 'spec' / f'A{subject_id:02d}{session}_spec.h5'
                            transform_spec(
                                str(input_file),
                                str(output_file),
                                sfreq=250.0,
                                window_length=0.5,
                                overlap=0.5,
                                freq_range=(1.0, 50.0),
                                target_size=args.image_size,
                                strategy='per_channel'
                            )

                        elif transform == 'cwt':
                            # Scalogram (CWT with Morlet wavelet)
                            output_file = Path(args.output_dir) / 'cwt' / f'A{subject_id:02d}{session}_cwt_morlet.h5'
                            transform_cwt(
                                str(input_file),
                                str(output_file),
                                sfreq=250.0,
                                freq_range=(1.0, 50.0),
                                n_scales=64,
                                wavelet='morl',
                                target_size=args.image_size,
                                strategy='per_channel'
                            )

                        elif transform == 'topo':
                            # Topographic (SSFI multi-band)
                            output_file = Path(args.output_dir) / 'topo' / f'A{subject_id:02d}{session}_topo_ssfi.h5'
                            transform_topo(
                                str(input_file),
                                str(output_file),
                                grid_size=args.image_size,
                                mode='ssfi'
                            )

                        success_count += 1

                    except Exception as e:
                        print(f"\nERROR processing {subject_id}{session}-{transform}: {e}")
                        failed.append(f"{subject_id}{session}-{transform}")

                    finally:
                        pbar.update(1)

    # Summary
    print(f"\n{'='*60}")
    print("BATCH TRANSFORMATION COMPLETE")
    print(f"{'='*60}")
    print(f"Successful: {success_count}/{total}")
    if failed:
        print(f"Failed: {len(failed)} -> {failed}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
