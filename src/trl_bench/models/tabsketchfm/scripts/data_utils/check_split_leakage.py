#!/usr/bin/env python3
"""
Check for data leakage in train/valid/test splits.

This script validates that splits are properly separated at the table level
to prevent data leakage that artificially inflates evaluation metrics.

Usage:
    python scripts/data_utils/check_split_leakage.py \
        --labels wiki_containment/labels.json

Output:
    - Reports table overlap between splits
    - Reports identical pair overlap between splits
    - Returns exit code 1 if leakage detected, 0 otherwise
"""

import argparse
import json
import sys
from collections import defaultdict


def load_labels(filepath):
    """Load labels.json file."""
    print(f"Loading labels from: {filepath}")
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def extract_tables_and_pairs(split_data):
    """Extract unique tables and pairs from a split."""
    tables = set()
    pairs = set()

    for item in split_data:
        t1 = item['table1']['filename'].replace('.csv', '')
        t2 = item['table2']['filename'].replace('.csv', '')

        tables.add(t1)
        tables.add(t2)

        # Create canonical pair (sorted order)
        pair = tuple(sorted([
            f"{t1}:{item['join_col_table1']}",
            f"{t2}:{item['join_col_table2']}"
        ]))
        pairs.add(pair)

    return tables, pairs


def check_overlap(set1, set2, name1, name2, item_type="items"):
    """Check and report overlap between two sets."""
    overlap = set1 & set2
    if overlap:
        pct1 = 100 * len(overlap) / len(set1) if set1 else 0
        pct2 = 100 * len(overlap) / len(set2) if set2 else 0
        print(f"  ❌ {name1}-{name2} overlap: {len(overlap):,} {item_type} "
              f"({pct1:.2f}% of {name1}, {pct2:.2f}% of {name2})")
        return len(overlap)
    else:
        print(f"  ✅ {name1}-{name2} overlap: 0 {item_type} (0.0%)")
        return 0


def print_label_statistics(split_data, split_name):
    """Print label distribution statistics for a split."""
    labels = [item['label'] for item in split_data]

    if not labels:
        print(f"  {split_name}: No samples")
        return

    # Check if binary or continuous
    unique_labels = set(labels)
    if unique_labels <= {0, 1, 0.0, 1.0}:
        # Binary classification
        pos_count = sum(1 for l in labels if l in [1, 1.0])
        neg_count = len(labels) - pos_count
        print(f"  {split_name}: {pos_count:,} positive ({100*pos_count/len(labels):.1f}%), "
              f"{neg_count:,} negative ({100*neg_count/len(labels):.1f}%)")
    else:
        # Regression
        zeros = sum(1 for l in labels if l == 0.0)
        mean_val = sum(labels) / len(labels)
        min_val = min(labels)
        max_val = max(labels)
        print(f"  {split_name}: {len(labels):,} samples | "
              f"mean={mean_val:.4f} | min={min_val:.4f} | max={max_val:.4f} | "
              f"zeros={zeros:,} ({100*zeros/len(labels):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Check for data leakage in splits')
    parser.add_argument('--labels', type=str, required=True,
                       help='Path to labels.json file')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed statistics')

    args = parser.parse_args()

    # Load data
    data = load_labels(args.labels)

    print("\n" + "="*80)
    print("DATA LEAKAGE CHECK")
    print("="*80)

    # Extract tables and pairs for each split
    print("\nExtracting tables and pairs...")
    train_tables, train_pairs = extract_tables_and_pairs(data['train'])
    valid_tables, valid_pairs = extract_tables_and_pairs(data['valid'])
    test_tables, test_pairs = extract_tables_and_pairs(data['test'])

    print(f"  Train: {len(data['train']):,} pairs, {len(train_tables):,} unique tables")
    print(f"  Valid: {len(data['valid']):,} pairs, {len(valid_tables):,} unique tables")
    print(f"  Test:  {len(data['test']):,} pairs, {len(test_tables):,} unique tables")

    # Check for table overlap
    print("\n" + "-"*80)
    print("TABLE-LEVEL OVERLAP CHECK")
    print("-"*80)

    leakage_detected = False

    overlap_tv = check_overlap(train_tables, valid_tables, "Train", "Valid", "tables")
    overlap_tt = check_overlap(train_tables, test_tables, "Train", "Test", "tables")
    overlap_vt = check_overlap(valid_tables, test_tables, "Valid", "Test", "tables")

    if overlap_tv > 0 or overlap_tt > 0 or overlap_vt > 0:
        leakage_detected = True

    # Check for identical pair overlap
    print("\n" + "-"*80)
    print("PAIR-LEVEL OVERLAP CHECK")
    print("-"*80)

    overlap_tv_pairs = check_overlap(train_pairs, valid_pairs, "Train", "Valid", "pairs")
    overlap_tt_pairs = check_overlap(train_pairs, test_pairs, "Train", "Test", "pairs")
    overlap_vt_pairs = check_overlap(valid_pairs, test_pairs, "Valid", "Test", "pairs")

    if overlap_tv_pairs > 0 or overlap_tt_pairs > 0 or overlap_vt_pairs > 0:
        leakage_detected = True

    # Label distribution
    if args.verbose:
        print("\n" + "-"*80)
        print("LABEL DISTRIBUTION")
        print("-"*80)
        print_label_statistics(data['train'], "Train")
        print_label_statistics(data['valid'], "Valid")
        print_label_statistics(data['test'], "Test")

    # Summary
    print("\n" + "="*80)
    if leakage_detected:
        print("⚠️  DATA LEAKAGE DETECTED!")
        print("="*80)
        print("\nThe splits have overlapping tables or identical pairs between")
        print("train/valid/test sets. This causes data leakage and artificially")
        print("inflates evaluation metrics.")
        print("\nRecommendation: Re-generate splits with table-level separation.")
        print("="*80)
        return 1
    else:
        print("✅ NO DATA LEAKAGE DETECTED")
        print("="*80)
        print("\nThe splits are properly separated - no table overlap and no")
        print("identical pairs between train/valid/test sets.")
        print("="*80)
        return 0


if __name__ == '__main__':
    sys.exit(main())
