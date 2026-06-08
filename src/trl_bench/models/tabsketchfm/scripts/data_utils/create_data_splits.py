#!/usr/bin/env python3
"""
Create train/test/validation splits for TabSketchFM pretraining.
This script generates a JSON.bz2 file mapping processed files to their sources.
"""

import json
import bz2
import os
import random
from collections import defaultdict
from pathlib import Path
import argparse


def load_processed_file_mapping(opendata_dir, metadata_dir, processed_dir):
    """
    Map processed JSON.bz2 files back to their source CSV and metadata files.
    Returns a list of entries with table, metadata, json, and column info.
    """
    print("Scanning processed files...")

    entries = []
    processed_files = list(Path(processed_dir).glob("*.json.bz2"))

    print(f"Found {len(processed_files)} processed JSON.bz2 files")

    # Load each processed file to extract source information
    for idx, json_path in enumerate(processed_files):
        if idx % 1000 == 0:
            print(f"Processing {idx}/{len(processed_files)}...")

        try:
            with bz2.open(json_path, 'rt', encoding='utf-8') as f:
                data = json.load(f)

            # Extract source file info from metadata
            table_metadata = data.get('table_metadata', {})
            source_file = table_metadata.get('file_name', '')

            if not source_file:
                continue

            # Build metadata path
            rel_path = os.path.relpath(source_file, opendata_dir)
            metadata_path = os.path.join(metadata_dir, rel_path + '.meta')

            # Check if metadata exists
            if not os.path.exists(metadata_path):
                continue

            # Extract column information
            columns = data.get('columns', {})
            for col_idx, col_name in enumerate(columns.keys()):
                entry = {
                    "table": source_file,
                    "metadata": metadata_path,
                    "json": str(json_path),
                    "column": col_idx
                }
                entries.append(entry)

        except Exception as e:
            print(f"Error processing {json_path}: {e}")
            continue

    print(f"Created {len(entries)} column entries from {len(processed_files)} files")
    return entries


def create_splits(entries, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=0):
    """
    Split entries into train/val/test sets.
    """
    random.seed(random_seed)

    # Group by source table to ensure augmented versions stay together
    table_groups = defaultdict(list)
    for entry in entries:
        table_groups[entry['table']].append(entry)

    # Get list of unique tables
    tables = list(table_groups.keys())
    random.shuffle(tables)

    # Calculate split points
    n_tables = len(tables)
    train_end = int(n_tables * train_ratio)
    val_end = train_end + int(n_tables * val_ratio)

    # Split tables
    train_tables = tables[:train_end]
    val_tables = tables[train_end:val_end]
    test_tables = tables[val_end:]

    # Collect all entries for each split
    splits = {
        'train': [],
        'valid': [],
        'test': []
    }

    for table in train_tables:
        splits['train'].extend(table_groups[table])

    for table in val_tables:
        splits['valid'].extend(table_groups[table])

    for table in test_tables:
        splits['test'].extend(table_groups[table])

    print(f"\nSplit summary:")
    print(f"  Train: {len(splits['train'])} entries from {len(train_tables)} tables")
    print(f"  Valid: {len(splits['valid'])} entries from {len(val_tables)} tables")
    print(f"  Test:  {len(splits['test'])} entries from {len(test_tables)} tables")

    return splits


def main():
    parser = argparse.ArgumentParser(description='Create train/test/val splits for TabSketchFM')
    parser.add_argument('--opendata_dir', default='opendata', help='Path to raw data directory')
    parser.add_argument('--metadata_dir', default='opendata_metadata', help='Path to metadata directory')
    parser.add_argument('--processed_dir', default='opendata_processed', help='Path to processed data directory')
    parser.add_argument('--output', default='data_splits.json.bz2', help='Output split file')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train split ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test split ratio')
    parser.add_argument('--random_seed', type=int, default=0, help='Random seed')

    args = parser.parse_args()

    print("=" * 60)
    print("TabSketchFM Data Split Generator")
    print("=" * 60)

    # Load and map files
    entries = load_processed_file_mapping(
        args.opendata_dir,
        args.metadata_dir,
        args.processed_dir
    )

    if not entries:
        print("ERROR: No valid entries found!")
        return

    # Create splits
    splits = create_splits(
        entries,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed
    )

    # Save to compressed JSON
    print(f"\nSaving splits to {args.output}...")
    with bz2.open(args.output, 'wt', encoding='utf-8') as f:
        json.dump(splits, f, indent=2)

    print(f"✅ Splits saved successfully!")
    print(f"File size: {os.path.getsize(args.output) / 1024 / 1024:.2f} MB")


if __name__ == '__main__':
    main()
