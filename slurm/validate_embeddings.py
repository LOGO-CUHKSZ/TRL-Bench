#!/usr/bin/env python3
"""
Validate embedding completeness for model/dataset outputs.

Supports both column embeddings (--type column) and row embeddings (--type row).

Column checks:
  - Embedding file exists
  - All dataset tables have embeddings
  - No extra embeddings for unknown tables
  - Column embeddings count matches table columns
  - Column index keys are contiguous

Row checks:
  - Embedding file exists
  - All dataset tables have embeddings
  - No extra embeddings for unknown tables
  - Row count matches CSV rows
  - Embedding shape matches (num_rows, embedding_dim)
  - No NaN/Inf values in embeddings
  - Consistent embedding dimension across tables
"""

import argparse
import csv
import json
import pickle
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from trl_bench.utils.pickle_compat import load_pickle

try:
    import numpy as np
except Exception:
    np = None

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def get_project_root() -> Path:
    """Get the project root directory.

    File at slurm/validate_embeddings.py; one .parent reaches the repo root
    (matches _PROJECT_ROOT above).
    """
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent


def load_yaml(path: Path) -> dict:
    """Load a YAML configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def find_tables_dir(dataset_path: Path, config_tables_dir: str | None) -> str:
    """Find the directory containing CSV files for a dataset."""
    if config_tables_dir is not None:
        return config_tables_dir

    candidates = ['tables', 'csv', 'datalake', 'datasets', '.']
    for subdir in candidates:
        candidate = dataset_path / subdir if subdir != '.' else dataset_path
        if candidate.exists() and list_csv_files(candidate, recursive=False):
            return subdir if subdir != '.' else '.'

    for d in dataset_path.iterdir():
        if d.is_dir() and list_csv_files(d, recursive=False):
            return d.name

    return '.'


def resolve_tables_dir(dataset_name: str, dataset_config: dict, datasets_root: Path) -> Path:
    """Resolve the actual tables directory path for a dataset."""
    tables_source = dataset_config.get('tables_source')
    if tables_source:
        source_path = datasets_root / tables_source
        tables_dir = find_tables_dir(source_path, dataset_config.get('tables_dir'))
        return source_path / tables_dir

    dataset_path = datasets_root / dataset_name
    tables_dir = find_tables_dir(dataset_path, dataset_config.get('tables_dir'))
    return dataset_path / tables_dir


def normalize_table_path(table_path: str, project_root: Path) -> str:
    """Normalize table path to absolute string for matching."""
    if not table_path:
        return ""
    try:
        p = Path(table_path)
        if p.is_absolute():
            return str(p.resolve())
        return str((project_root / p).resolve())
    except Exception:
        return table_path


def list_csv_files(directory: Path, recursive: bool) -> list[Path]:
    """List CSV files in a directory, including dot-prefixed files."""
    if recursive:
        results: list[Path] = []
        for root, _, files in os.walk(directory):
            for name in files:
                if name.endswith(".csv"):
                    results.append(Path(root) / name)
        return sorted(results)

    results = []
    with os.scandir(directory) as it:
        for entry in it:
            if entry.is_file() and entry.name.endswith(".csv"):
                results.append(Path(entry.path))
    return sorted(results)


def discover_model_dirs(base_dir: Path, *,
                        exclude_query_encoders: bool = False,
                        exclude_hybrids: bool = False) -> list[str]:
    """Discover model directory names from an embeddings output tree."""
    query_encoder_models = {'sentence_t5', 'mpnet'}

    if not base_dir.exists():
        return []

    names = []
    for child in base_dir.iterdir():
        if not child.is_dir() or child.name.startswith('.'):
            continue
        name = child.name
        if exclude_query_encoders and name in query_encoder_models:
            continue
        if exclude_hybrids and (name.endswith('_hybrid') or '_backup_' in name):
            continue
        names.append(name)
    return sorted(names)


def resolve_optional_path(path_like: str | Path | None) -> Path | None:
    """Resolve an optional path relative to the project root."""
    if not path_like:
        return None
    p = Path(path_like)
    return p if p.is_absolute() else _PROJECT_ROOT / p


def discover_trained_models(models_cfg: dict) -> set[str]:
    """Return trained row models whose alternate dimensions live in overlay roots."""
    return {
        model_name
        for model_name, model_cfg in (models_cfg.get('models', {}) or {}).items()
        if model_cfg.get('model_type') == 'trained'
    }


def resolve_model_overlay_roots(
    default_root: Path,
    overlay_root: Path | None,
    strict_overlay_models: set[str],
    model_name: str,
) -> list[Path]:
    """Resolve precedence roots for a model, keeping trained overlays strict."""
    if overlay_root and model_name in strict_overlay_models:
        return [overlay_root]
    return [default_root]


def resolve_row_embeddings_path(
    default_root: Path,
    overlay_root: Path | None,
    strict_overlay_models: set[str],
    model_name: str,
    dataset_name: str,
) -> Path:
    """Resolve the concrete row embedding file for validation."""
    roots = resolve_model_overlay_roots(default_root, overlay_root, strict_overlay_models, model_name)
    for root in roots:
        candidate = root / model_name / f"{dataset_name}.pkl"
        if candidate.exists():
            return candidate
    return roots[0] / model_name / f"{dataset_name}.pkl"


def load_embeddings(path: Path):
    """Load embeddings from pickle file."""
    data = load_pickle(path)
    if isinstance(data, dict) and 'results' in data:
        data = data['results']
    if isinstance(data, list):
        return data
    return []


def get_column_info(path: Path) -> tuple[int, list]:
    """Return (column_count, column_names) for a CSV file."""
    if pd is not None:
        try:
            df = pd.read_csv(path, nrows=0)
            return len(df.columns), list(df.columns)
        except Exception:
            pass

    try:
        with open(path, 'r', encoding='utf-8', errors='replace', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                return 0, []
            return len(header), header
    except Exception:
        return 0, []


def get_row_count(path: Path) -> int:
    """Return the number of data rows (excluding header) in a CSV file."""
    if pd is not None:
        try:
            return len(pd.read_csv(path))
        except Exception:
            pass
    try:
        with open(path, 'r', encoding='utf-8', errors='replace', newline='') as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            return sum(1 for _ in reader)
    except Exception:
        return -1


def build_expected_tables(tables_dir: Path, recursive: bool = False) -> list[Path]:
    """Collect expected CSV files for a dataset."""
    return list_csv_files(tables_dir, recursive=recursive)


def validate_model_dataset(
    model: str,
    dataset: str,
    embeddings_path: Path,
    tables_dir: Path,
    project_root: Path,
    check_columns: bool,
    workers: int,
    recursive: bool,
    use_progress: bool,
    embedding_type: str = 'column',
) -> dict:
    """Validate a single model/dataset pair."""
    result = {
        'model': model,
        'dataset': dataset,
        'embeddings_path': str(embeddings_path),
        'tables_dir': str(tables_dir),
        'embedding_type': embedding_type,
        'status': 'ok',
        'counts': {},
        'issues': {
            'missing_embeddings_file': False,
            'missing_tables': [],
            'extra_tables': [],
            'duplicate_embeddings': [],
            'invalid_entries': [],
            'column_mismatches': [],
            'column_name_mismatches': [],
            'column_key_gaps': [],
            # Row-specific issues
            'row_count_mismatches': [],
            'shape_mismatches': [],
            'nan_inf_entries': [],
            'dim_inconsistencies': [],
            # Table-specific issues
            'missing_table_embedding': [],
            'empty_table_embedding': [],
            'table_nan_inf_entries': [],
            'table_dim_inconsistencies': [],
        },
    }

    if not embeddings_path.exists():
        result['status'] = 'missing_embeddings_file'
        result['issues']['missing_embeddings_file'] = True
        return result

    expected_tables = build_expected_tables(tables_dir, recursive=recursive)
    expected_paths = [p.resolve() for p in expected_tables]
    expected_by_path = {str(p): p for p in expected_paths}

    expected_by_id = {}
    duplicate_ids = set()
    for p in expected_paths:
        stem = p.stem
        if stem in expected_by_id:
            duplicate_ids.add(stem)
        else:
            expected_by_id[stem] = p

    try:
        embeddings = load_embeddings(embeddings_path)
    except Exception as e:
        result['status'] = 'failed_to_load_embeddings'
        result['issues']['invalid_entries'].append(f"load_error: {e}")
        return result

    matched = {}
    duplicate_embeddings = []
    extra_tables = []
    invalid_entries = []

    for idx, item in enumerate(embeddings):
        if not isinstance(item, dict):
            invalid_entries.append(f"index {idx}: {type(item)}")
            continue

        table_path = item.get('table', '')
        table_id = item.get('table_id', '')
        matched_key = None

        if table_path:
            norm_path = normalize_table_path(table_path, project_root)
            if norm_path in expected_by_path:
                matched_key = norm_path

        if matched_key is None:
            candidate_id = table_id
            if not candidate_id and table_path:
                try:
                    candidate_id = Path(table_path).stem
                except Exception:
                    candidate_id = ''
            if candidate_id and candidate_id in expected_by_id and candidate_id not in duplicate_ids:
                matched_key = str(expected_by_id[candidate_id].resolve())

        if matched_key is None:
            extra_tables.append(table_path or table_id or f"index {idx}")
            continue

        if matched_key in matched:
            duplicate_embeddings.append(matched_key)
        else:
            matched[matched_key] = item

    missing_tables = [str(p) for p in expected_paths if str(p) not in matched]

    column_mismatches = []
    column_name_mismatches = []
    column_key_gaps = []
    row_count_mismatches = []
    shape_mismatches = []
    nan_inf_entries = []
    dim_inconsistencies = []
    missing_table_embedding = []
    empty_table_embedding = []
    table_nan_inf_entries = []
    table_dim_inconsistencies = []

    if check_columns and matched:
        expected_paths_list = list(matched.keys())

        if embedding_type == 'table':
            # --- Table embedding checks ---
            _TABLE_VARIANTS = ('cls_embedding', 'table_embedding', 'column_mean', 'token_mean')
            seen_dims = set()
            items_iter = matched.items()
            if use_progress and tqdm is not None:
                items_iter = tqdm(
                    list(items_iter),
                    total=len(matched),
                    desc=f"Validate {model}/{dataset}",
                    leave=False,
                )
            for path_str, item in items_iter:
                te = item.get('table_embedding')
                if te is None or not isinstance(te, dict):
                    missing_table_embedding.append(path_str)
                    continue

                # Check at least one non-None variant exists
                variants = {k: v for k, v in te.items() if k in _TABLE_VARIANTS and v is not None}
                if not variants:
                    empty_table_embedding.append(path_str)
                    continue

                # Check NaN/Inf and collect dims
                for k, v in variants.items():
                    if not isinstance(v, np.ndarray):
                        continue
                    seen_dims.add(v.shape[0])
                    if np.isnan(v).any() or np.isinf(v).any():
                        nan_count = int(np.isnan(v).sum())
                        inf_count = int(np.isinf(v).sum())
                        table_nan_inf_entries.append({
                            'table': path_str,
                            'variant': k,
                            'nan_count': nan_count,
                            'inf_count': inf_count,
                        })

            if len(seen_dims) > 1:
                table_dim_inconsistencies.append({
                    'model': model,
                    'dataset': dataset,
                    'dims_found': sorted(seen_dims),
                })

        elif embedding_type == 'row':
            # --- Row embedding checks ---
            # Gather CSV row counts in parallel
            def compute_row_info(path_str: str):
                return path_str, get_row_count(Path(path_str))

            if workers > 1:
                row_info = {}
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futures = {ex.submit(compute_row_info, p): p for p in expected_paths_list}
                    future_iter = as_completed(futures)
                    if use_progress and tqdm is not None:
                        future_iter = tqdm(
                            future_iter,
                            total=len(futures),
                            desc=f"Rows {model}/{dataset}",
                            leave=False
                        )
                    for fut in future_iter:
                        path_str, count = fut.result()
                        row_info[path_str] = count
            else:
                path_iter = expected_paths_list
                if use_progress and tqdm is not None:
                    path_iter = tqdm(
                        path_iter,
                        total=len(expected_paths_list),
                        desc=f"Rows {model}/{dataset}",
                        leave=False
                    )
                row_info = {p: get_row_count(Path(p)) for p in path_iter}

            seen_dims = set()
            items_iter = matched.items()
            if use_progress and tqdm is not None:
                items_iter = tqdm(
                    list(items_iter),
                    total=len(matched),
                    desc=f"Validate {model}/{dataset}",
                    leave=False
                )
            for path_str, item in items_iter:
                row_emb = item.get('row_embeddings')

                # Check row_embeddings exists and is ndarray
                if row_emb is None or not hasattr(row_emb, 'shape'):
                    shape_mismatches.append({
                        'table': path_str,
                        'issue': 'missing or invalid row_embeddings',
                    })
                    continue

                # Check shape matches metadata
                claimed_rows = item.get('num_rows', -1)
                claimed_dim = item.get('embedding_dim', -1)
                if row_emb.ndim != 2:
                    shape_mismatches.append({
                        'table': path_str,
                        'issue': f'expected 2D, got {row_emb.ndim}D shape={row_emb.shape}',
                    })
                    continue

                actual_rows, actual_dim = row_emb.shape
                if claimed_rows >= 0 and actual_rows != claimed_rows:
                    shape_mismatches.append({
                        'table': path_str,
                        'issue': f'num_rows={claimed_rows} but shape[0]={actual_rows}',
                    })
                if claimed_dim >= 0 and actual_dim != claimed_dim:
                    shape_mismatches.append({
                        'table': path_str,
                        'issue': f'embedding_dim={claimed_dim} but shape[1]={actual_dim}',
                    })

                # Check row count against CSV
                exp_rows = row_info.get(path_str, -1)
                if exp_rows >= 0 and actual_rows != exp_rows:
                    row_count_mismatches.append({
                        'table': path_str,
                        'expected_rows': exp_rows,
                        'found_rows': actual_rows,
                    })

                # Check for NaN/Inf
                if np is not None:
                    if np.isnan(row_emb).any() or np.isinf(row_emb).any():
                        nan_count = int(np.isnan(row_emb).sum())
                        inf_count = int(np.isinf(row_emb).sum())
                        nan_inf_entries.append({
                            'table': path_str,
                            'nan_count': nan_count,
                            'inf_count': inf_count,
                        })

                seen_dims.add(actual_dim)

            # Check embedding dimension consistency
            if len(seen_dims) > 1:
                dim_inconsistencies.append({
                    'model': model,
                    'dataset': dataset,
                    'dims_found': sorted(seen_dims),
                })

        else:
            # --- Column embedding checks ---
            def compute_col_info(path_str: str):
                return path_str, get_column_info(Path(path_str))

            if workers > 1:
                col_info = {}
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futures = {ex.submit(compute_col_info, p): p for p in expected_paths_list}
                    future_iter = as_completed(futures)
                    if use_progress and tqdm is not None:
                        future_iter = tqdm(
                            future_iter,
                            total=len(futures),
                            desc=f"Columns {model}/{dataset}",
                            leave=False
                        )
                    for fut in future_iter:
                        path_str, info = fut.result()
                        col_info[path_str] = info
            else:
                path_iter = expected_paths_list
                if use_progress and tqdm is not None:
                    path_iter = tqdm(
                        path_iter,
                        total=len(expected_paths_list),
                        desc=f"Columns {model}/{dataset}",
                        leave=False
                    )
                col_info = {p: get_column_info(Path(p)) for p in path_iter}

            items_iter = matched.items()
            if use_progress and tqdm is not None:
                items_iter = tqdm(
                    list(items_iter),
                    total=len(matched),
                    desc=f"Validate {model}/{dataset}",
                    leave=False
                )
            for path_str, item in items_iter:
                exp_count, exp_names = col_info.get(path_str, (0, []))
                col_emb = item.get('column_embeddings') or item.get('column_embedding') or {}
                if not isinstance(col_emb, dict):
                    column_mismatches.append({
                        'table': path_str,
                        'expected_cols': exp_count,
                        'found_cols': 'invalid',
                    })
                    continue

                found_count = len(col_emb)
                if exp_count != found_count:
                    column_mismatches.append({
                        'table': path_str,
                        'expected_cols': exp_count,
                        'found_cols': found_count,
                    })

                col_names = item.get('column_names')
                if isinstance(col_names, list) and exp_count != len(col_names):
                    column_name_mismatches.append({
                        'table': path_str,
                        'expected_cols': exp_count,
                        'found_cols': len(col_names),
                    })

                try:
                    keys = sorted(int(k) for k in col_emb.keys())
                    if keys != list(range(len(keys))):
                        column_key_gaps.append({
                            'table': path_str,
                            'keys': keys[:10],
                            'total_keys': len(keys),
                        })
                except Exception:
                    column_key_gaps.append({
                        'table': path_str,
                        'keys': 'non-integer',
                        'total_keys': len(col_emb),
                    })

    result['counts'] = {
        'expected_tables': len(expected_paths),
        'embedding_entries': len(embeddings),
        'matched_tables': len(matched),
        'missing_tables': len(missing_tables),
        'extra_tables': len(extra_tables),
        'duplicate_embeddings': len(duplicate_embeddings),
        'invalid_entries': len(invalid_entries),
        'column_mismatches': len(column_mismatches),
        'column_name_mismatches': len(column_name_mismatches),
        'column_key_gaps': len(column_key_gaps),
        'row_count_mismatches': len(row_count_mismatches),
        'shape_mismatches': len(shape_mismatches),
        'nan_inf_entries': len(nan_inf_entries),
        'dim_inconsistencies': len(dim_inconsistencies),
        'missing_table_embedding': len(missing_table_embedding),
        'empty_table_embedding': len(empty_table_embedding),
        'table_nan_inf_entries': len(table_nan_inf_entries),
        'table_dim_inconsistencies': len(table_dim_inconsistencies),
    }

    result['issues']['missing_tables'] = missing_tables
    result['issues']['extra_tables'] = extra_tables
    result['issues']['duplicate_embeddings'] = duplicate_embeddings
    result['issues']['invalid_entries'] = invalid_entries
    result['issues']['column_mismatches'] = column_mismatches
    result['issues']['column_name_mismatches'] = column_name_mismatches
    result['issues']['column_key_gaps'] = column_key_gaps
    result['issues']['row_count_mismatches'] = row_count_mismatches
    result['issues']['shape_mismatches'] = shape_mismatches
    result['issues']['nan_inf_entries'] = nan_inf_entries
    result['issues']['dim_inconsistencies'] = dim_inconsistencies
    result['issues']['missing_table_embedding'] = missing_table_embedding
    result['issues']['empty_table_embedding'] = empty_table_embedding
    result['issues']['table_nan_inf_entries'] = table_nan_inf_entries
    result['issues']['table_dim_inconsistencies'] = table_dim_inconsistencies

    all_issues = [
        missing_tables, extra_tables, duplicate_embeddings, invalid_entries,
        column_mismatches, column_name_mismatches, column_key_gaps,
        row_count_mismatches, shape_mismatches, nan_inf_entries, dim_inconsistencies,
        missing_table_embedding, empty_table_embedding, table_nan_inf_entries, table_dim_inconsistencies,
    ]
    if any(all_issues):
        result['status'] = 'issues_found'

    return result


def _validate_task(args: tuple) -> dict:
    """Worker wrapper for multiprocessing."""
    (
        model,
        dataset,
        embeddings_path,
        tables_dir,
        project_root,
        check_columns,
        workers,
        recursive,
        use_progress,
        embedding_type,
    ) = args
    return validate_model_dataset(
        model=model,
        dataset=dataset,
        embeddings_path=Path(embeddings_path),
        tables_dir=Path(tables_dir),
        project_root=Path(project_root),
        check_columns=check_columns,
        workers=workers,
        recursive=recursive,
        use_progress=use_progress,
        embedding_type=embedding_type,
    )


def _resolve_row_tables_dir(dataset_name: str, dataset_config: dict, datasets_root: Path) -> Path:
    """Resolve input directory for row embeddings.

    Row embedding datasets use 'row_input_dir' from dataset config, which
    points to a flat directory of CSVs (e.g. column_shuffle/, row_shuffle/).
    Falls back to the column-style resolution if not specified.
    """
    row_input = dataset_config.get('row_input_dir')
    if row_input:
        p = Path(row_input)
        if p.is_absolute():
            return p
        return datasets_root / row_input
    return resolve_tables_dir(dataset_name, dataset_config, datasets_root)


def main():
    parser = argparse.ArgumentParser(
        description='Validate embedding completeness for model/dataset outputs'
    )
    parser.add_argument('--type', choices=['column', 'row', 'table'], default='column',
                        help='Embedding type to validate (default: column)')
    parser.add_argument('--models', nargs='+', help='Models to validate (default: all)')
    parser.add_argument('--datasets', nargs='+', help='Datasets to validate (default: all)')
    parser.add_argument('--only-existing', action='store_true',
                        help='Only validate pairs with existing embedding files')
    parser.add_argument('--skip-checks', action='store_true',
                        help='Skip content checks — column counts (column) or row counts/shapes (row)')
    parser.add_argument('--recursive', action='store_true',
                        help='Recursively search for CSV files in tables dir')
    parser.add_argument('--workers', type=int, default=4,
                        help='Worker threads for content checks (default: 4)')
    parser.add_argument('--jobs', type=int, default=32,
                        help='Processes for per-pair validation (default: 32)')
    parser.add_argument('--show', type=int, default=10,
                        help='Show up to N examples per issue (default: 10)')
    parser.add_argument('--report', type=str, default="report.json", help='Write full JSON report to file')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print per-pair summaries')
    parser.add_argument('--no-progress', action='store_true',
                        help='Disable progress bar')
    parser.add_argument('--embedding-root', type=str, default=None,
                        help='Override the base embedding root for the selected type')
    parser.add_argument('--row-overlay-root', '--row-embedding-root', dest='row_overlay_root', type=str, default=None,
                        help='Overlay row embedding root to merge on top of embeddings/row. '
                             'Trained row models are read strictly from this root when set.')
    # Backward compat
    parser.add_argument('--skip-columns', action='store_true',
                        help=argparse.SUPPRESS)

    args = parser.parse_args()
    embedding_type = args.type
    skip_checks = args.skip_checks or args.skip_columns

    project_root = get_project_root()
    config_dir = project_root / 'slurm' / 'config'
    resources = load_yaml(config_dir / 'resources.yaml')
    datasets_cfg = load_yaml(config_dir / 'datasets.yaml')

    explicit_output_root = resolve_optional_path(args.embedding_root)
    row_overlay_root = None if explicit_output_root else resolve_optional_path(args.row_overlay_root)

    if embedding_type == 'row':
        models_cfg = load_yaml(config_dir / 'row_models.yaml')
        output_dir = explicit_output_root or project_root / resources.get('paths', {}).get(
            'row_output_dir', 'embeddings/row')
        strict_row_overlay_models = discover_trained_models(models_cfg)
    elif embedding_type == 'table':
        # Table embeddings come from both column models (derived) and native table models
        col_models = load_yaml(config_dir / 'models.yaml').get('models', {})
        table_models_path = config_dir / 'table_models.yaml'
        native_models = load_yaml(table_models_path).get('models', {}) if table_models_path.exists() else {}
        merged = {**col_models, **native_models}
        models_cfg = {'models': merged}
        output_dir = explicit_output_root or project_root / resources.get('paths', {}).get(
            'table_output_dir', 'embeddings/table')
    else:
        models_cfg = load_yaml(config_dir / 'models.yaml')
        output_dir = explicit_output_root or project_root / resources.get('paths', {}).get(
            'output_dir', 'embeddings/column')

    datasets_root = project_root / resources.get('paths', {}).get('datasets_dir', 'datasets')

    discovered_models = set()
    if embedding_type == 'table':
        discovered_models.update(discover_model_dirs(
            output_dir,
            exclude_query_encoders=True,
            exclude_hybrids=True,
        ))
    else:
        discovered_models.update(discover_model_dirs(output_dir))
        if embedding_type == 'row' and row_overlay_root is not None:
            discovered_models.update(discover_model_dirs(row_overlay_root))

    all_models = sorted(set(models_cfg['models'].keys()) | discovered_models)
    all_datasets = list(datasets_cfg['datasets'].keys())

    # For table type, default to datasets with table_embedding: true
    if embedding_type == 'table' and not args.datasets:
        all_datasets = [
            name for name, cfg in datasets_cfg['datasets'].items()
            if cfg.get('table_embedding', False)
        ]

    models = args.models or all_models
    datasets = args.datasets or all_datasets

    invalid_models = set(models) - set(all_models)
    if invalid_models:
        print(f"Unknown models: {sorted(invalid_models)}")
        sys.exit(1)
    invalid_datasets = set(datasets) - set(all_datasets)
    if invalid_datasets:
        print(f"Unknown datasets: {sorted(invalid_datasets)}")
        sys.exit(1)

    results = []
    use_progress = not args.no_progress and tqdm is not None and sys.stdout.isatty()

    pairs = []
    for model in sorted(models):
        for dataset in sorted(datasets):
            dataset_cfg = datasets_cfg['datasets'][dataset]
            # Skip if dataset is restricted to specific models
            allowed_models = dataset_cfg.get('models')
            if allowed_models and model not in allowed_models:
                continue
            if embedding_type == 'row':
                tables_dir = _resolve_row_tables_dir(dataset, dataset_cfg, datasets_root)
                embeddings_path = resolve_row_embeddings_path(
                    output_dir,
                    row_overlay_root,
                    strict_row_overlay_models,
                    model,
                    dataset,
                )
            else:
                tables_dir = resolve_tables_dir(dataset, dataset_cfg, datasets_root)
                embeddings_path = output_dir / model / f"{dataset}.pkl"
            if args.only_existing and not embeddings_path.exists():
                continue
            pairs.append((model, dataset, str(embeddings_path), str(tables_dir)))

    if not args.no_progress and tqdm is None:
        print("Note: tqdm not installed; run `pip install tqdm` for a progress bar.")

    if args.jobs > 1 and pairs:
        worker_progress = False
        tasks = [
            (
                model,
                dataset,
                embeddings_path,
                tables_dir,
                str(project_root),
                not skip_checks,
                max(1, args.workers),
                args.recursive,
                worker_progress,
                embedding_type,
            )
            for model, dataset, embeddings_path, tables_dir in pairs
        ]

        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futures = [ex.submit(_validate_task, t) for t in tasks]
            future_iter = as_completed(futures)
            if use_progress:
                future_iter = tqdm(future_iter, total=len(futures), desc='Validating')
            for fut in future_iter:
                res = fut.result()
                results.append(res)
                if args.verbose:
                    _print_verbose_line(res, embedding_type)
    else:
        iterator = pairs
        if use_progress:
            iterator = tqdm(pairs, total=len(pairs), desc='Validating')

        for model, dataset, embeddings_path, tables_dir in iterator:
            res = validate_model_dataset(
                model=model,
                dataset=dataset,
                embeddings_path=Path(embeddings_path),
                tables_dir=Path(tables_dir),
                project_root=project_root,
                check_columns=not skip_checks,
                workers=max(1, args.workers),
                recursive=args.recursive,
                use_progress=use_progress,
                embedding_type=embedding_type,
            )
            results.append(res)

            if args.verbose:
                _print_verbose_line(res, embedding_type)

    # Summary
    print("=" * 70)
    type_label = {"row": "ROW", "column": "COLUMN", "table": "TABLE"}.get(embedding_type, "COLUMN")
    print(f"{type_label} EMBEDDING COMPLETENESS REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    total = len(results)
    ok = [r for r in results if r['status'] == 'ok']
    missing_files = [r for r in results if r['issues']['missing_embeddings_file']]
    issues = [r for r in results if r['status'] == 'issues_found']

    print(f"Checked: {total} ({embedding_type} embeddings)")
    print(f"OK: {len(ok)}")
    print(f"Missing embedding files: {len(missing_files)}")
    print(f"Pairs with issues: {len(issues)}")

    def show_examples(label: str, items: list[str]):
        if not items:
            return
        print(f"\n{label} (showing up to {args.show}):")
        for x in items[:args.show]:
            print(f"  {x}")

    if missing_files:
        show_examples(
            "Missing embedding files",
            [f"{r['model']}/{r['dataset']}" for r in missing_files]
        )

    # Per-pair issue summary
    for r in issues:
        counts = r['counts']
        print(f"\n{r['model']}/{r['dataset']}:")

        # Common counts
        parts = [
            f"missing_tables={counts.get('missing_tables', 0)}",
            f"extra_tables={counts.get('extra_tables', 0)}",
            f"duplicates={counts.get('duplicate_embeddings', 0)}",
            f"invalid_entries={counts.get('invalid_entries', 0)}",
        ]
        if embedding_type == 'column':
            parts += [
                f"column_mismatches={counts.get('column_mismatches', 0)}",
                f"column_name_mismatches={counts.get('column_name_mismatches', 0)}",
                f"column_key_gaps={counts.get('column_key_gaps', 0)}",
            ]
        elif embedding_type == 'table':
            parts += [
                f"missing_te_dict={counts.get('missing_table_embedding', 0)}",
                f"empty_te={counts.get('empty_table_embedding', 0)}",
                f"nan_inf={counts.get('table_nan_inf_entries', 0)}",
                f"dim_inconsistencies={counts.get('table_dim_inconsistencies', 0)}",
            ]
        else:
            parts += [
                f"row_count_mismatches={counts.get('row_count_mismatches', 0)}",
                f"shape_mismatches={counts.get('shape_mismatches', 0)}",
                f"nan_inf={counts.get('nan_inf_entries', 0)}",
                f"dim_inconsistencies={counts.get('dim_inconsistencies', 0)}",
            ]
        print(f"  {' '.join(parts)}")

        show_examples(
            "  missing tables",
            r['issues']['missing_tables']
        )
        show_examples(
            "  extra tables",
            r['issues']['extra_tables']
        )

        if embedding_type == 'column':
            show_examples(
                "  column mismatches",
                [f"{x['table']} (expected {x['expected_cols']}, found {x['found_cols']})"
                 for x in r['issues']['column_mismatches']]
            )
        elif embedding_type == 'table':
            show_examples(
                "  missing table_embedding dict",
                r['issues']['missing_table_embedding']
            )
            show_examples(
                "  empty table_embedding (all variants None)",
                r['issues']['empty_table_embedding']
            )
            show_examples(
                "  NaN/Inf entries",
                [f"{x['table']} variant={x['variant']} (nan={x['nan_count']}, inf={x['inf_count']})"
                 for x in r['issues']['table_nan_inf_entries']]
            )
            show_examples(
                "  dimension inconsistencies",
                [f"{x['model']}/{x['dataset']}: dims={x['dims_found']}"
                 for x in r['issues']['table_dim_inconsistencies']]
            )
        else:
            show_examples(
                "  row count mismatches",
                [f"{x['table']} (expected {x['expected_rows']}, found {x['found_rows']})"
                 for x in r['issues']['row_count_mismatches']]
            )
            show_examples(
                "  shape mismatches",
                [f"{x['table']}: {x['issue']}" for x in r['issues']['shape_mismatches']]
            )
            show_examples(
                "  NaN/Inf entries",
                [f"{x['table']} (nan={x['nan_count']}, inf={x['inf_count']})"
                 for x in r['issues']['nan_inf_entries']]
            )
            show_examples(
                "  dimension inconsistencies",
                [f"{x['model']}/{x['dataset']}: dims={x['dims_found']}"
                 for x in r['issues']['dim_inconsistencies']]
            )

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote report to: {report_path}")


def _print_verbose_line(res: dict, embedding_type: str):
    """Print a per-pair summary line."""
    counts = res['counts']
    parts = [
        f"{res['model']}/{res['dataset']}:",
        f"expected={counts.get('expected_tables', 0)}",
        f"embeddings={counts.get('embedding_entries', 0)}",
        f"missing={counts.get('missing_tables', 0)}",
        f"extra={counts.get('extra_tables', 0)}",
    ]
    if embedding_type == 'column':
        parts.append(f"col_mismatch={counts.get('column_mismatches', 0)}")
    elif embedding_type == 'table':
        parts.append(f"missing_te={counts.get('missing_table_embedding', 0)}")
        parts.append(f"nan_inf={counts.get('table_nan_inf_entries', 0)}")
    else:
        parts.append(f"row_mismatch={counts.get('row_count_mismatches', 0)}")
        parts.append(f"shape={counts.get('shape_mismatches', 0)}")
        parts.append(f"nan_inf={counts.get('nan_inf_entries', 0)}")
    parts.append(f"status={res['status']}")
    print(" ".join(parts))


if __name__ == '__main__':
    main()
