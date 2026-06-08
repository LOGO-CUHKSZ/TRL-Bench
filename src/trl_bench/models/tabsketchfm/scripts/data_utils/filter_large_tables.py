#!/usr/bin/env python3
"""
Filter out tables with too many columns to prevent NaN loss.
Creates a new split file with reasonable-sized tables only.
"""

import bz2
import json
from pathlib import Path

def count_columns(json_path):
    """Count columns in a processed JSON file."""
    try:
        with bz2.open(json_path, 'rt') as f:
            data = json.load(f)
        return len(data.get('columns', {}))
    except:
        return None

def filter_splits(input_file, output_file, max_columns=512):
    """
    Filter data splits to only include tables with <= max_columns.

    Args:
        input_file: Path to original data_splits.json.bz2
        output_file: Path to save filtered splits
        max_columns: Maximum number of columns to allow
    """
    print(f"Loading {input_file}...")
    with bz2.open(input_file, 'rt') as f:
        splits = json.load(f)

    filtered_splits = {'train': [], 'valid': [], 'test': []}
    stats = {'train': {'total': 0, 'kept': 0, 'filtered': 0},
             'valid': {'total': 0, 'kept': 0, 'filtered': 0},
             'test': {'total': 0, 'kept': 0, 'filtered': 0}}

    # Track which JSON files we've already checked
    column_counts = {}

    for split_name in ['train', 'valid', 'test']:
        print(f"\nProcessing {split_name} split...")
        samples = splits[split_name]
        stats[split_name]['total'] = len(samples)

        for idx, sample in enumerate(samples):
            if idx % 10000 == 0:
                print(f"  Processed {idx}/{len(samples)}...")

            json_path = sample['json']

            # Check column count (cache results)
            if json_path not in column_counts:
                column_counts[json_path] = count_columns(json_path)

            col_count = column_counts[json_path]

            if col_count is not None and col_count <= max_columns:
                filtered_splits[split_name].append(sample)
                stats[split_name]['kept'] += 1
            else:
                stats[split_name]['filtered'] += 1

    # Save filtered splits
    print(f"\nSaving filtered splits to {output_file}...")
    with bz2.open(output_file, 'wt') as f:
        json.dump(filtered_splits, f, indent=2)

    # Print statistics
    print(f"\n{'='*60}")
    print("FILTERING COMPLETE")
    print(f"{'='*60}")
    for split_name in ['train', 'valid', 'test']:
        s = stats[split_name]
        kept_pct = (s['kept'] / s['total'] * 100) if s['total'] > 0 else 0
        print(f"\n{split_name.upper()}:")
        print(f"  Total entries: {s['total']}")
        print(f"  Kept: {s['kept']} ({kept_pct:.1f}%)")
        print(f"  Filtered: {s['filtered']} ({100-kept_pct:.1f}%)")

    print(f"\n✅ Filtered file saved: {output_file}")
    print(f"File size: {Path(output_file).stat().st_size / 1024 / 1024:.1f} MB")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Filter large tables from data splits')
    parser.add_argument('--input', default='data_splits.json.bz2', help='Input split file')
    parser.add_argument('--output', default='data_splits_filtered.json.bz2', help='Output split file')
    parser.add_argument('--max_columns', type=int, default=256,
                        help='Maximum columns to keep (default: 256)')
    args = parser.parse_args()

    filter_splits(args.input, args.output, args.max_columns)
