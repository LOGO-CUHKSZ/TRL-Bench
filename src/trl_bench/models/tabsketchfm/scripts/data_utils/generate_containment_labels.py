#!/usr/bin/env python3
"""
Generate finetuning labels.json for wiki-containment task (regression or classification).

Per TabSketchFM paper: "Wiki Containment is formulated as regression task to estimate
containment value" using R² metric. This script converts the JSONL ground truth format
to the finetuning labels.json format with continuous score labels (regression) or
binary labels (classification).

Usage:
    # Regression (default)
    python scripts/data_utils/generate_containment_labels.py \
        --input wiki-join-search/labels/join_search_containment_min_gt.jsonl \
        --tables_dir wiki_containment/tables \
        --output wiki_containment/labels.json \
        --task_type regression

    # Classification
    python scripts/data_utils/generate_containment_labels.py \
        --input wiki-join-search/labels/join_search_containment_min_gt.jsonl \
        --tables_dir wiki_containment/tables \
        --output wiki_containment/labels.json \
        --task_type classification \
        --threshold 0.5

Input format (JSONL - one JSON object per line):
    {"source": {"filename": "XXX", "col": "0"},
     "joinable_list": [{"filename": "YYY", "col": "0", "score": 0.2}, ...]}

Output format (JSON):
    - Regression: label is continuous score (0.0 to 1.0)
    - Classification: label is binary (0 or 1) based on threshold
    {
        "train": [{"table1": {"filename": "XXX.csv"}, "table2": {"filename": "YYY.csv"},
                   "label": 0.85, "join_col_table1": "col_name", "join_col_table2": "col_name"}, ...],
        "valid": [...],
        "test": [...]
    }
"""

import argparse
import json
import os
import random
from collections import defaultdict


def load_jsonl(filepath):
    """Load JSONL file line by line."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def get_available_tables(tables_dir):
    """Get set of available table filenames (without extension)."""
    tables = set()
    for f in os.listdir(tables_dir):
        if f.endswith('.csv') or f.endswith('.CSV'):
            tables.add(f.replace('.csv', '').replace('.CSV', ''))
    return tables


def generate_labels(ground_truth, available_tables, task_type='regression', threshold=0.5, negative_ratio=1.0, max_negatives_per_source=10):
    """
    Generate pairwise labels from search ground truth.

    Args:
        ground_truth: List of search ground truth entries
        available_tables: Set of available table names
        task_type: 'regression' (continuous scores) or 'classification' (binary labels)
        threshold: Threshold for converting scores to binary labels (used in classification)
        negative_ratio: Ratio of negative to positive samples
        max_negatives_per_source: Max negative samples per source table

    Returns:
        List of label dictionaries with continuous or binary labels
    """
    labels = []
    positive_pairs = set()
    all_tables = list(available_tables)

    # Generate pairs from ground truth with continuous scores
    for entry in ground_truth:
        source_table = entry['source']['filename']
        source_col = entry['source']['col']

        # Skip if source table not in available tables
        if source_table not in available_tables:
            continue

        for joinable in entry.get('joinable_list', []):
            target_table = joinable['filename']
            target_col = joinable['col']
            score = joinable['score']

            # Skip if target not in available tables
            if target_table not in available_tables:
                continue

            # Skip self-joins
            if source_table == target_table:
                continue

            # Create canonical pair to avoid duplicates
            pair_key = tuple(sorted([f"{source_table}:{source_col}", f"{target_table}:{target_col}"]))
            if pair_key in positive_pairs:
                continue

            positive_pairs.add(pair_key)

            # Convert score based on task type
            if task_type == 'classification':
                label = 1 if score >= threshold else 0
            else:  # regression
                label = score

            labels.append({
                'table1': {'filename': f"{source_table}.csv"},
                'table2': {'filename': f"{target_table}.csv"},
                'label': label,
                'join_col_table1': source_col,
                'join_col_table2': target_col
            })

    if task_type == 'classification':
        print(f"Generated {len(labels)} pairs with binary labels (threshold={threshold})")
    else:
        print(f"Generated {len(labels)} pairs with continuous scores")

    # Generate negative pairs (score=0.0 for non-joinable pairs)
    num_negatives = int(len(labels) * negative_ratio)
    negative_pairs = set()

    # Build positive pairs lookup for quick negative sampling
    positive_table_pairs = set()
    for label in labels:
        t1 = label['table1']['filename'].replace('.csv', '')
        t2 = label['table2']['filename'].replace('.csv', '')
        positive_table_pairs.add((t1, t2))
        positive_table_pairs.add((t2, t1))

    attempts = 0
    max_attempts = num_negatives * 10

    while len(negative_pairs) < num_negatives and attempts < max_attempts:
        attempts += 1
        t1 = random.choice(all_tables)
        t2 = random.choice(all_tables)

        if t1 == t2:
            continue
        if (t1, t2) in positive_table_pairs:
            continue

        pair_key = tuple(sorted([t1, t2]))
        if pair_key in negative_pairs:
            continue

        negative_pairs.add(pair_key)

        # Negative pairs always have label 0 (score=0.0 for regression, class=0 for classification)
        labels.append({
            'table1': {'filename': f"{t1}.csv"},
            'table2': {'filename': f"{t2}.csv"},
            'label': 0 if task_type == 'classification' else 0.0,
            'join_col_table1': '0',  # Default to first column
            'join_col_table2': '0'
        })

    if task_type == 'classification':
        print(f"Generated {len(negative_pairs)} negative pairs (label=0)")
    else:
        print(f"Generated {len(negative_pairs)} negative pairs (score=0.0)")
    print(f"Total pairs: {len(labels)}")

    return labels


def split_labels(labels, train_ratio=0.7, valid_ratio=0.15, seed=42):
    """
    Split labels into train/valid/test sets with TABLE-LEVEL separation.

    This ensures no table appears in multiple splits, preventing data leakage.
    Tables are split first, then pairs are assigned based on table membership.

    Assignment strategy:
    - If both tables in train set → train split
    - If at least one table in valid set (and none in test) → valid split
    - If at least one table in test set → test split

    Args:
        labels: List of label dictionaries with table pairs
        train_ratio: Ratio of tables for training (default: 0.7)
        valid_ratio: Ratio of tables for validation (default: 0.15)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with train/valid/test splits
    """
    random.seed(seed)

    # Step 1: Collect all unique tables from all pairs
    all_tables = set()
    for label in labels:
        t1 = label['table1']['filename'].replace('.csv', '').replace('.CSV', '')
        t2 = label['table2']['filename'].replace('.csv', '').replace('.CSV', '')
        all_tables.add(t1)
        all_tables.add(t2)

    # Step 2: Split tables into disjoint sets (no overlap!)
    all_tables = list(all_tables)
    random.shuffle(all_tables)

    n_tables = len(all_tables)
    train_end = int(n_tables * train_ratio)
    valid_end = int(n_tables * (train_ratio + valid_ratio))

    train_tables = set(all_tables[:train_end])
    valid_tables = set(all_tables[train_end:valid_end])
    test_tables = set(all_tables[valid_end:])

    print(f"\nTable-level split:")
    print(f"  Train tables: {len(train_tables):,}")
    print(f"  Valid tables: {len(valid_tables):,}")
    print(f"  Test tables:  {len(test_tables):,}")
    print(f"  Total tables: {len(all_tables):,}")

    # Verify no overlap
    assert len(train_tables & valid_tables) == 0, "Train-Valid table overlap detected!"
    assert len(train_tables & test_tables) == 0, "Train-Test table overlap detected!"
    assert len(valid_tables & test_tables) == 0, "Valid-Test table overlap detected!"
    print("  ✓ No table overlap between splits")

    # Step 3: Assign pairs based on table membership
    # STRICT RULE: Both tables must be from the SAME split
    # This ensures complete table-level separation
    train_pairs = []
    valid_pairs = []
    test_pairs = []
    cross_split_pairs = 0

    for label in labels:
        t1 = label['table1']['filename'].replace('.csv', '').replace('.CSV', '')
        t2 = label['table2']['filename'].replace('.csv', '').replace('.CSV', '')

        # Determine which split(s) the tables belong to
        t1_in_train = t1 in train_tables
        t1_in_valid = t1 in valid_tables
        t1_in_test = t1 in test_tables

        t2_in_train = t2 in train_tables
        t2_in_valid = t2 in valid_tables
        t2_in_test = t2 in test_tables

        # Assignment logic: BOTH tables must be from the same split
        if t1_in_train and t2_in_train:
            # Both in train → train split
            train_pairs.append(label)
        elif t1_in_valid and t2_in_valid:
            # Both in valid → valid split
            valid_pairs.append(label)
        elif t1_in_test and t2_in_test:
            # Both in test → test split
            test_pairs.append(label)
        else:
            # Cross-split pair (e.g., one table in train, other in valid)
            # These are discarded to ensure complete table-level separation
            cross_split_pairs += 1

    if cross_split_pairs > 0:
        print(f"  ℹ Discarded {cross_split_pairs:,} cross-split pairs (tables from different splits)")

    print(f"\nPair-level distribution:")
    print(f"  Train pairs: {len(train_pairs):,}")
    print(f"  Valid pairs: {len(valid_pairs):,}")
    print(f"  Test pairs:  {len(test_pairs):,}")
    print(f"  Total pairs: {len(train_pairs) + len(valid_pairs) + len(test_pairs):,}")

    return {
        'train': train_pairs,
        'valid': valid_pairs,
        'test': test_pairs
    }


def main():
    parser = argparse.ArgumentParser(description='Generate finetuning labels from containment ground truth')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to containment ground truth JSONL file')
    parser.add_argument('--tables_dir', type=str, required=True,
                       help='Directory containing available tables')
    parser.add_argument('--output', type=str, required=True,
                       help='Output labels.json path')
    parser.add_argument('--task_type', type=str, default='regression', choices=['regression', 'classification'],
                       help='Task type: regression (continuous scores) or classification (binary labels)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for converting scores to binary labels in classification (default: 0.5)')
    parser.add_argument('--negative_ratio', type=float, default=1.0,
                       help='Ratio of negative to positive samples (default: 1.0)')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training set ratio (default: 0.7)')
    parser.add_argument('--valid_ratio', type=float, default=0.15,
                       help='Validation set ratio (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')

    args = parser.parse_args()

    print(f"Loading ground truth from: {args.input}")
    ground_truth = load_jsonl(args.input)
    print(f"Loaded {len(ground_truth)} entries")

    print(f"Loading available tables from: {args.tables_dir}")
    available_tables = get_available_tables(args.tables_dir)
    print(f"Found {len(available_tables)} tables")

    print(f"Generating labels (task_type={args.task_type})")
    if args.task_type == 'classification':
        print(f"Using threshold={args.threshold} for binary classification")
    labels = generate_labels(
        ground_truth,
        available_tables,
        task_type=args.task_type,
        threshold=args.threshold,
        negative_ratio=args.negative_ratio
    )

    print(f"Splitting into train/valid/test...")
    split_data = split_labels(
        labels,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        seed=args.seed
    )

    print(f"Train: {len(split_data['train'])} samples")
    print(f"Valid: {len(split_data['valid'])} samples")
    print(f"Test: {len(split_data['test'])} samples")

    # Print label distribution for classification
    if args.task_type == 'classification':
        for split_name, split_data_list in split_data.items():
            pos_count = sum(1 for item in split_data_list if item['label'] == 1)
            neg_count = len(split_data_list) - pos_count
            print(f"{split_name.capitalize()}: {pos_count} positive, {neg_count} negative")

    print(f"Saving to: {args.output}")
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(split_data, f, indent=4)

    print("Done!")


if __name__ == '__main__':
    main()
