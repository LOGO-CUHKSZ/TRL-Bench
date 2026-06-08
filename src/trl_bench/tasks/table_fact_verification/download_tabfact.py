#!/usr/bin/env python3
"""
Download and prepare the TabFact dataset.

TabFact is a large-scale dataset for table-based fact verification containing:
- 117,854 manually annotated statements
- 16,573 Wikipedia tables
- Binary labels: ENTAILED (1) or REFUTED (0)

Source: https://github.com/wenhuchen/Table-Fact-Checking

Usage:
    python download_tabfact.py --output_dir datasets/tabfact
"""

import os
import json
import argparse
import urllib.request
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def download_tabfact(output_dir: str, max_tables: int = None) -> dict:
    """
    Download and prepare TabFact dataset from GitHub.

    Args:
        output_dir: Directory to save the dataset
        max_tables: Maximum number of tables to download (for testing)

    Returns:
        Dict with dataset statistics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tables_dir = output_path / "tables"
    tables_dir.mkdir(exist_ok=True)

    base_url = "https://raw.githubusercontent.com/wenhuchen/Table-Fact-Checking/master"

    # Download tokenized examples for each split
    splits = {
        'train': f"{base_url}/tokenized_data/train_examples.json",
        'validation': f"{base_url}/tokenized_data/val_examples.json",
        'test': f"{base_url}/tokenized_data/test_examples.json",
    }

    stats = {
        'train': {'total': 0, 'entailed': 0, 'refuted': 0},
        'validation': {'total': 0, 'entailed': 0, 'refuted': 0},
        'test': {'total': 0, 'entailed': 0, 'refuted': 0},
    }

    all_table_ids = set()
    all_examples = {}

    # Download and parse each split
    for split_name, url in splits.items():
        print(f"\nDownloading {split_name} split...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=60) as response:
                data = json.loads(response.read().decode('utf-8'))
        except Exception as e:
            print(f"Error downloading {split_name}: {e}")
            continue

        examples = []
        example_idx = 0

        # Data format: {table_id: [[statements], [labels], caption]}
        for table_id, (statements, labels, caption) in tqdm(data.items(), desc=f"Processing {split_name}"):
            all_table_ids.add(table_id)

            for stmt, label in zip(statements, labels):
                example = {
                    'id': f"{split_name}_{example_idx}",
                    'table_id': table_id,
                    'statement': stmt,
                    'label': label,  # 1 = ENTAILED, 0 = REFUTED
                    'caption': caption,
                }
                examples.append(example)
                example_idx += 1

                # Update stats
                stats[split_name]['total'] += 1
                if label == 1:
                    stats[split_name]['entailed'] += 1
                else:
                    stats[split_name]['refuted'] += 1

        # Save examples as JSONL
        output_file = output_path / f"{split_name}.jsonl"
        with open(output_file, 'w') as f:
            for ex in examples:
                f.write(json.dumps(ex) + '\n')

        print(f"Saved {len(examples)} examples to {output_file}")
        all_examples[split_name] = examples

    # Download tables
    print(f"\nDownloading {len(all_table_ids)} tables...")

    if max_tables:
        table_ids_to_download = list(all_table_ids)[:max_tables]
        print(f"  (Limited to {max_tables} tables for testing)")
    else:
        table_ids_to_download = list(all_table_ids)

    def download_table(table_id):
        table_url = f"{base_url}/data/all_csv/{table_id}"
        table_path = tables_dir / table_id
        if table_path.exists():
            return table_id, True, "exists"
        try:
            req = urllib.request.Request(table_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as response:
                content = response.read().decode('utf-8')
            with open(table_path, 'w') as f:
                f.write(content)
            return table_id, True, None
        except Exception as e:
            return table_id, False, str(e)

    successful = 0
    failed = 0
    errors = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(download_table, tid): tid for tid in table_ids_to_download}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading tables"):
            table_id, success, error = future.result()
            if success:
                successful += 1
            else:
                failed += 1
                errors.append({'table_id': table_id, 'error': error})

    print(f"Downloaded {successful} tables, {failed} failed")

    if errors:
        error_file = output_path / "download_errors.json"
        with open(error_file, 'w') as f:
            json.dump(errors, f, indent=2)
        print(f"Errors saved to {error_file}")

    # Save table metadata
    tables_meta_file = output_path / "tables_metadata.json"
    with open(tables_meta_file, 'w') as f:
        json.dump({
            'num_tables': len(all_table_ids),
            'downloaded': successful,
            'failed': failed,
            'table_ids': list(all_table_ids),
        }, f, indent=2)

    # Save labels file for run_task.py compatibility
    labels = {}
    for split_name in ['train', 'validation', 'test']:
        filepath = output_path / f"{split_name}.jsonl"
        if filepath.exists():
            labels[split_name] = [{'id': ex['id'], 'label': ex['label']}
                                  for ex in _load_jsonl(filepath)]
    labels_file = output_path / "labels.json"
    with open(labels_file, 'w') as f:
        json.dump(labels, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("TabFact Dataset Downloaded Successfully")
    print("="*60)
    print(f"Output directory: {output_path}")
    print(f"Total tables: {len(all_table_ids)} (downloaded: {successful})")
    print(f"\nSplit statistics:")
    for split_name, split_stats in stats.items():
        if split_stats['total'] > 0:
            print(f"  {split_name}:")
            print(f"    Total: {split_stats['total']}")
            print(f"    Entailed: {split_stats['entailed']} ({100*split_stats['entailed']/split_stats['total']:.1f}%)")
            print(f"    Refuted: {split_stats['refuted']} ({100*split_stats['refuted']/split_stats['total']:.1f}%)")

    stats['num_tables'] = len(all_table_ids)
    stats['tables_downloaded'] = successful
    stats['tables_failed'] = failed
    return stats


def _load_jsonl(filepath: Path) -> list:
    """Load JSONL file."""
    examples = []
    with open(filepath, 'r') as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare TabFact dataset"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='datasets/tabfact',
        help='Directory to save the dataset'
    )
    parser.add_argument(
        '--max_tables',
        type=int,
        default=None,
        help='Maximum number of tables to download (for testing)'
    )

    args = parser.parse_args()

    stats = download_tabfact(
        output_dir=args.output_dir,
        max_tables=args.max_tables,
    )

    print("\nDone!")


if __name__ == '__main__':
    main()
