#!/usr/bin/env python3
"""Merge shard pkl files into a single output pkl.

Preserves the legacy list-of-dicts pickle schema expected by all
downstream consumers (``generate_table_embeddings.py``, model resume
loaders, ``run_task.py``, etc.).

Usage:
    # Explicit shard paths
    python merge_shards.py --output merged.pkl --shards s0.pkl s1.pkl s2.pkl

    # Auto-discover from naming convention
    python merge_shards.py --output merged.pkl \\
        --shard-dir embeddings/column/gte \\
        --dataset dlte_v1_all --num-shards 3
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
from pathlib import Path


def _canonical_table_id(entry: dict) -> str:
    """Extract a canonical table identifier for deduplication.

    Only strips directory components and a trailing ``.csv`` extension.
    Does NOT use ``os.path.splitext`` because table IDs often contain
    dots (e.g. ``Hotel_destinia.ad_September2020_CPA``) that would be
    mis-parsed as file extensions.
    """
    # Prefer table_id if present, else fall back to table_name then table
    raw = entry.get('table_id') or entry.get('table_name') or entry.get('table', '')
    raw = str(raw)
    # Strip directory components
    basename = os.path.basename(raw)
    # Only strip .csv extension specifically
    if basename.endswith('.csv'):
        basename = basename[:-4]
    return basename


def merge_shard_files(
    shard_paths: list[str | Path],
    output_path: str | Path,
    delete_shards: bool = True,
) -> int:
    """Merge N shard pkl files into one.

    Args:
        shard_paths: Ordered list of shard pkl file paths.
        output_path: Path for the merged output pkl.
        delete_shards: If True, delete shard pkls (and their checkpoints)
            after a successful merge.

    Returns:
        Number of tables in the merged output.
    """
    shard_paths = [Path(p) for p in shard_paths]
    output_path = Path(output_path)

    if not shard_paths:
        raise ValueError("shard_paths is empty — nothing to merge")

    # Validate all shard files exist before starting
    missing = [p for p in shard_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing shard files: {[str(p) for p in missing]}"
        )

    # Load and concatenate (raw pickle — preserves exact dict schema)
    all_entries: list[dict] = []
    for shard in shard_paths:
        with open(shard, 'rb') as f:
            data = pickle.load(f)
        if not isinstance(data, list):
            raise TypeError(
                f"Expected list[dict] in {shard}, got {type(data).__name__}"
            )
        all_entries.extend(data)

    # Deduplicate: keep last occurrence (handles resume edge cases)
    seen: dict[str, int] = {}
    for idx, entry in enumerate(all_entries):
        tid = _canonical_table_id(entry)
        seen[tid] = idx  # last occurrence wins
    keep_indices = sorted(seen.values())
    deduped = [all_entries[i] for i in keep_indices]

    # Sort by canonical table_id for deterministic output
    deduped.sort(key=_canonical_table_id)

    # Atomic write: .tmp then os.replace
    tmp_path = output_path.with_suffix('.tmp')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, 'wb') as f:
        pickle.dump(deduped, f, protocol=4)
    os.replace(tmp_path, output_path)

    # Validate: re-load and check count
    with open(output_path, 'rb') as f:
        check = pickle.load(f)
    if len(check) != len(deduped):
        raise RuntimeError(
            f"Merge validation failed: wrote {len(deduped)} entries but "
            f"re-read {len(check)}"
        )

    print(f"Merged {len(shard_paths)} shards -> {len(deduped)} tables in {output_path}")

    # Clean up shard files + their checkpoints
    # Resolve output_path to avoid deleting it if it overlaps with a shard path
    resolved_output = output_path.resolve()
    if delete_shards:
        for shard in shard_paths:
            if shard.resolve() == resolved_output:
                print(f"  Skipped delete: {shard.name} (is the output file)")
                continue
            checkpoint = shard.with_name(
                shard.stem + '.checkpoint.pkl'
            )
            for f in (shard, checkpoint):
                if f.exists():
                    f.unlink()
                    print(f"  Deleted: {f.name}")

    return len(deduped)


def _discover_shard_paths(
    shard_dir: str | Path,
    dataset: str,
    num_shards: int,
) -> list[Path]:
    """Build expected shard paths from naming convention."""
    shard_dir = Path(shard_dir)
    paths = []
    for i in range(num_shards):
        paths.append(shard_dir / f"{dataset}_shard{i}of{num_shards}.pkl")
    return paths


def main():
    parser = argparse.ArgumentParser(
        description='Merge shard pkl files into a single output'
    )
    parser.add_argument('--output', required=True, help='Output pkl path')

    # Option 1: explicit shard paths
    parser.add_argument('--shards', nargs='+', help='Explicit shard pkl paths')

    # Option 2: auto-discover
    parser.add_argument('--shard-dir', help='Directory containing shard pkls')
    parser.add_argument('--dataset', help='Dataset name for auto-discovery')
    parser.add_argument('--num-shards', type=int, help='Number of shards')

    parser.add_argument('--no-delete', action='store_true',
                        help='Keep shard files after merge')

    args = parser.parse_args()

    if args.shards:
        shard_paths = [Path(p) for p in args.shards]
    elif args.shard_dir and args.dataset and args.num_shards:
        shard_paths = _discover_shard_paths(args.shard_dir, args.dataset, args.num_shards)
    else:
        parser.error("Provide either --shards or (--shard-dir, --dataset, --num-shards)")

    try:
        merge_shard_files(shard_paths, args.output, delete_shards=not args.no_delete)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
