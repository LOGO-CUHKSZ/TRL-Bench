#!/usr/bin/env python3
"""Generate table list files for sharded embedding generation.

Splits a dataset's CSV files into N roughly-equal partitions and writes
one text file per shard.  Table lists are per-dataset (shared across
models) so that every model processes the exact same partition.

Output directory: ``slurm/scripts/generated/table_lists/``

Usage (standalone):
    python generate_table_lists.py --dataset dlte_v1_all --input-dir /path/to/csvs --shards 3

The main consumer is ``generate_scripts.py`` which calls
:func:`generate_table_lists` programmatically.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _list_csv_basenames(input_dir: str | Path) -> list[str]:
    """Return sorted basenames of ``.csv`` files in *input_dir*."""
    results = []
    with os.scandir(input_dir) as it:
        for entry in it:
            if entry.is_file() and entry.name.endswith('.csv'):
                results.append(entry.name)
    return sorted(results)


def generate_table_lists(
    dataset_name: str,
    input_dir: str | Path,
    num_shards: int,
    output_dir: str | Path | None = None,
) -> list[Path]:
    """Partition CSV basenames into *num_shards* list files.

    Args:
        dataset_name: Dataset identifier (used in filenames).
        input_dir: Directory containing the CSV files.
        num_shards: Number of shards to create.
        output_dir: Where to write the list files.  Defaults to
            ``slurm/scripts/generated/table_lists`` relative to the
            project root.

    Returns:
        List of Paths to the generated table list files.
    """
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")

    if output_dir is None:
        project_root = Path(__file__).resolve().parent.parent
        output_dir = project_root / 'slurm' / 'scripts' / 'generated' / 'table_lists'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    basenames = _list_csv_basenames(input_dir)
    if not basenames:
        raise ValueError(f"No CSV files found in {input_dir}")

    if num_shards > len(basenames):
        raise ValueError(
            f"num_shards ({num_shards}) exceeds table count ({len(basenames)}) "
            f"for {dataset_name} — would produce empty shard files"
        )

    # Divide into N roughly-equal chunks (earlier chunks get +1 if remainder)
    chunk_size, remainder = divmod(len(basenames), num_shards)
    chunks: list[list[str]] = []
    offset = 0
    for i in range(num_shards):
        size = chunk_size + (1 if i < remainder else 0)
        chunks.append(basenames[offset:offset + size])
        offset += size

    # Write list files (idempotent: overwrites existing)
    paths: list[Path] = []
    for i, chunk in enumerate(chunks):
        filename = f"{dataset_name}_shard{i}of{num_shards}.txt"
        filepath = output_dir / filename
        with open(filepath, 'w') as f:
            for name in chunk:
                f.write(name + '\n')
        paths.append(filepath)

    return paths


def main():
    parser = argparse.ArgumentParser(
        description='Generate table list files for sharded embedding generation'
    )
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--input-dir', required=True, help='Directory containing CSV files')
    parser.add_argument('--shards', type=int, required=True, help='Number of shards')
    parser.add_argument('--output-dir', default=None, help='Output directory for list files')

    args = parser.parse_args()
    paths = generate_table_lists(args.dataset, args.input_dir, args.shards, args.output_dir)
    for p in paths:
        print(f"  {p}")
    print(f"Generated {len(paths)} table list files")


if __name__ == '__main__':
    main()
