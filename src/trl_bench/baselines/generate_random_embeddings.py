#!/usr/bin/env python3
"""
Generate random embedding baselines for cross-task negative control experiments.

Reads a reference model's embeddings and creates dimension-matched random versions.
The SLURM pipeline auto-discovers the output model directory with zero changes to
downstream tasks.

Usage:
    # Generate random embeddings for all levels using bert as reference
    python utils/baselines/generate_random_embeddings.py --ref_model bert --seed 42

    # Specific levels and datasets
    python utils/baselines/generate_random_embeddings.py \\
        --ref_model bert --seed 42 --model_name random \\
        --levels column table --datasets ckan_subset wiki_union

    # Custom dimensionality
    python utils/baselines/generate_random_embeddings.py \\
        --ref_model bert --seed 42 --dim 256

    # Dry run
    python utils/baselines/generate_random_embeddings.py --ref_model bert --dry-run
"""

import argparse
import json
import pickle
import shutil
from pathlib import Path

import numpy as np


def get_project_root() -> Path:
    """Get the project root directory (two levels up from this script)."""
    return Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def randomize_column_table_pkl(ref_path, out_path, rng, model_name, dim_override):
    """
    Randomize column-level or table-level embedding pkl files.

    Handles both embeddings/column/{model}/{dataset}.pkl and
    embeddings/table/{model}/{dataset}.pkl (same unified v2.0 format).

    Preserves all metadata; replaces embedding arrays with random normal values.
    """
    with open(ref_path, 'rb') as f:
        data = pickle.load(f)

    dim_used = None
    for entry in data:
        # --- column_embeddings: dict mapping int -> ndarray ---
        col_embs = entry.get('column_embeddings', {})
        for col_idx, arr in col_embs.items():
            if arr is not None and hasattr(arr, 'shape'):
                shape = list(arr.shape)
                if dim_override:
                    shape[-1] = dim_override
                col_embs[col_idx] = rng.standard_normal(shape).astype(np.float32)
                dim_used = shape[-1]

        # --- table_embedding: dict mapping variant_name -> ndarray or None ---
        table_emb = entry.get('table_embedding')
        if isinstance(table_emb, dict):
            for key in table_emb:
                val = table_emb[key]
                if val is not None and hasattr(val, 'shape'):
                    shape = list(val.shape)
                    if dim_override:
                        shape[-1] = dim_override
                    table_emb[key] = rng.standard_normal(shape).astype(np.float32)
                    dim_used = shape[-1] if len(shape) == 1 else shape[-1]

        entry['model_name'] = model_name
        if dim_used is not None:
            entry['embedding_dim'] = dim_used

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(data, f, protocol=4)

    return len(data), dim_used


def randomize_row_pkl(ref_path, out_path, rng, model_name, dim_override):
    """
    Randomize row-level embedding pkl files.

    Handles embeddings/row/{model}/{dataset}.pkl.
    Replaces row_embeddings array (n_rows, dim) with random normal values.
    """
    with open(ref_path, 'rb') as f:
        data = pickle.load(f)

    dim_used = None
    for entry in data:
        row_embs = entry.get('row_embeddings')
        if row_embs is not None and hasattr(row_embs, 'shape'):
            shape = list(row_embs.shape)
            if dim_override:
                shape[-1] = dim_override
            entry['row_embeddings'] = rng.standard_normal(shape).astype(np.float32)
            dim_used = shape[-1]

        entry['model_name'] = model_name
        if dim_used is not None:
            entry['embedding_dim'] = dim_used

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(data, f, protocol=4)

    return len(data), dim_used


def randomize_row_prediction_dir(ref_dir, out_dir, rng, model_name, dim_override):
    """
    Randomize row_prediction embedding directories.

    Handles embeddings/row_prediction/{model}/{dataset}/ directories.
    Metadata-driven: reads metadata.json to determine which files to process.
    Embedding .npy files get randomized; label and index files are copied verbatim.
    """
    metadata_path = ref_dir / 'metadata.json'
    if not metadata_path.exists():
        print(f"  WARNING: No metadata.json in {ref_dir}, skipping")
        return 0, None

    with open(metadata_path) as f:
        meta = json.load(f)

    out_dir.mkdir(parents=True, exist_ok=True)

    dim_used = None
    splits = meta.get('splits', {})

    for split_name in sorted(splits.keys()):
        split_info = splits[split_name]

        # Randomize embeddings file
        emb_file = split_info.get('embeddings_file')
        if emb_file:
            ref_emb_path = ref_dir / emb_file
            out_emb_path = out_dir / emb_file
            if ref_emb_path.exists():
                arr = np.load(ref_emb_path)
                shape = list(arr.shape)
                if dim_override:
                    shape[-1] = dim_override
                random_arr = rng.standard_normal(shape).astype(np.float32)
                np.save(out_emb_path, random_arr)
                dim_used = shape[-1]

        # Copy label files verbatim
        # Handle both singular labels_file and plural labels_files
        labels_file = split_info.get('labels_file')
        if labels_file:
            src = ref_dir / labels_file
            if src.exists():
                shutil.copy2(src, out_dir / labels_file)

        labels_files = split_info.get('labels_files', {})
        for label_name, label_file in labels_files.items():
            src = ref_dir / label_file
            if src.exists():
                shutil.copy2(src, out_dir / label_file)

        # Copy row indices verbatim
        row_indices_file = split_info.get('row_indices_file')
        if row_indices_file:
            src = ref_dir / row_indices_file
            if src.exists():
                shutil.copy2(src, out_dir / row_indices_file)

    # Update metadata
    meta['model_name'] = model_name
    if dim_used is not None:
        meta['embedding_dim'] = dim_used

    with open(out_dir / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    n_splits = len(splits)
    return n_splits, dim_used


# ---------------------------------------------------------------------------
# Level dispatch
# ---------------------------------------------------------------------------

LEVELS = {
    'column':         ('embeddings/column',         'pkl', randomize_column_table_pkl),
    'table':          ('embeddings/table',           'pkl', randomize_column_table_pkl),
    'row':            ('embeddings/row',             'pkl', randomize_row_pkl),
    'row_prediction': ('embeddings/row_prediction',  'dir', randomize_row_prediction_dir),
}


def process_level(level_name, base_path, file_type, handler, project_root,
                  ref_model, model_name, rng, dim_override,
                  datasets_filter, dry_run):
    """Process a single embedding level (column, table, row, or row_prediction)."""
    ref_dir = project_root / base_path / ref_model
    if not ref_dir.exists():
        print(f"WARNING: Reference directory does not exist: {ref_dir}")
        return 0

    out_dir = project_root / base_path / model_name

    if file_type == 'pkl':
        items = sorted([
            p for p in ref_dir.iterdir()
            if p.suffix == '.pkl' and p.is_file()
        ], key=lambda p: p.name)
    else:
        # Directory-based (row_prediction) — only dirs with metadata.json
        items = sorted([
            p for p in ref_dir.iterdir()
            if p.is_dir() and (p / 'metadata.json').exists()
        ], key=lambda p: p.name)

    processed = 0
    for item in items:
        dataset_name = item.stem if file_type == 'pkl' else item.name

        # Apply dataset filter
        if datasets_filter and dataset_name not in datasets_filter:
            continue

        if file_type == 'pkl':
            out_path = out_dir / item.name
            if dry_run:
                print(f"  [DRY-RUN] Would process: {level_name}/{dataset_name}")
                processed += 1
                continue
            n_entries, dim = handler(item, out_path, rng, model_name, dim_override)
            print(f"  Processing {level_name}/{dataset_name}... "
                  f"{n_entries} entries, dim={dim}")
            processed += 1
        else:
            out_item_dir = out_dir / item.name
            if dry_run:
                print(f"  [DRY-RUN] Would process: {level_name}/{dataset_name}")
                processed += 1
                continue
            n_entries, dim = handler(item, out_item_dir, rng, model_name, dim_override)
            if n_entries > 0:
                print(f"  Processing {level_name}/{dataset_name}... "
                      f"{n_entries} splits, dim={dim}")
                processed += 1
            else:
                print(f"  Skipped {level_name}/{dataset_name} (no data produced)")

    return processed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Generate random embedding baselines from reference model embeddings.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--ref_model', type=str, required=True,
        help='Reference model to read embeddings from (e.g., bert, gte)')
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42)')
    parser.add_argument(
        '--model_name', type=str, default='random',
        help='Name for the output model directory (default: random)')
    parser.add_argument(
        '--dim', type=int, default=None,
        help='Override embedding dimensionality (default: match reference)')
    parser.add_argument(
        '--levels', nargs='+', default=None,
        choices=list(LEVELS.keys()),
        help='Embedding levels to process (default: all available)')
    parser.add_argument(
        '--datasets', nargs='+', default=None,
        help='Only process these datasets (default: all)')
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Show what would be generated without writing files')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    project_root = get_project_root()

    print(f"Project root: {project_root}")
    print(f"Reference model: {args.ref_model}")
    print(f"Output model name: {args.model_name}")
    print(f"Seed: {args.seed}")
    if args.dim:
        print(f"Dimension override: {args.dim}")
    if args.datasets:
        print(f"Dataset filter: {args.datasets}")
    if args.dry_run:
        print("DRY RUN MODE")
    print()

    rng = np.random.default_rng(args.seed)

    levels_to_process = args.levels or list(LEVELS.keys())
    total_processed = 0

    for level_name in levels_to_process:
        base_path, file_type, handler = LEVELS[level_name]
        print(f"=== Level: {level_name} ===")

        count = process_level(
            level_name=level_name,
            base_path=base_path,
            file_type=file_type,
            handler=handler,
            project_root=project_root,
            ref_model=args.ref_model,
            model_name=args.model_name,
            rng=rng,
            dim_override=args.dim,
            datasets_filter=set(args.datasets) if args.datasets else None,
            dry_run=args.dry_run,
        )
        print(f"  {count} dataset(s) processed\n")
        total_processed += count

    print(f"Total: {total_processed} dataset(s) across {len(levels_to_process)} level(s)")


if __name__ == '__main__':
    main()
