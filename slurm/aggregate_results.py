#!/usr/bin/env python3
"""
Aggregate downstream task results into summary files.

Collects JSON results from results/evaluation/{task}/{model}/.../*.json
(recursive discovery) and generates:
- results/evaluation/summary/all_results.csv
- results/evaluation/summary/leaderboard.md

Supports multi-seed results: when multiple seeds exist for a (model, dataset)
combination, the leaderboard shows mean±std.

Usage:
    # Aggregate all results
    python aggregate_results.py

    # Aggregate specific tasks
    python aggregate_results.py --tasks column_clustering column_type_prediction

    # Output to custom directory
    python aggregate_results.py --output-dir /path/to/output

    # Verbose output
    python aggregate_results.py -v
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np


def get_project_root() -> Path:
    """Get the project root directory.

    File at slurm/aggregate_results.py; one .parent reaches the repo root.
    """
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent


def collect_results(results_dir: Path, tasks: list[str] | None = None) -> list[dict]:
    """
    Collect all JSON result files using recursive discovery.

    Discovers results in:
    - {task}/{model}/*.json (standard)
    - {task}/{model}/{variant}/*.json (variant subdirs)
    - {task}/{model}/{dataset}/{label_name}/results.json (row_prediction nested)

    Deduplicates on (task, model, dataset, variant, seed, label_column) key,
    preferring canonical files (from extract_metrics) over generic results.json.

    Args:
        results_dir: Base results directory
        tasks: List of tasks to include (None = all)

    Returns:
        List of result dictionaries
    """
    raw_results = []

    if not results_dir.exists():
        print(f"Warning: Results directory not found: {results_dir}")
        return raw_results

    for task_dir in results_dir.iterdir():
        if not task_dir.is_dir():
            continue

        task_name = task_dir.name
        if task_name == 'summary':
            continue

        if tasks and task_name not in tasks:
            continue

        for model_dir in task_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name

            # Recursive discovery: pick up variant subdirs and nested results
            for json_file in model_dir.rglob('*.json'):
                # Skip strict-split results to avoid contaminating legacy
                # aggregations.  Strict results live under .../strict/ and
                # should be aggregated separately (--split-protocol strict).
                rel = json_file.relative_to(model_dir)
                if 'strict' in rel.parts:
                    continue

                try:
                    with open(json_file, 'r') as f:
                        result = json.load(f)

                    # Ensure required fields
                    result.setdefault('task', task_name)
                    result.setdefault('model', model_name)
                    result.setdefault('source_file', str(json_file))

                    # Parse seed from JSON (preferred) or filename
                    result.setdefault('seed', None)
                    if result['seed'] is None:
                        m = re.search(r'_seed(\d+)', json_file.stem)
                        if m:
                            result['seed'] = int(m.group(1))

                    # ── Variant inference ──
                    # Known variant names used across the project
                    known_variants = {
                        'cls_embedding', 'column_mean', 'column_sum',  # column_sum: legacy-only; remove after asset refresh
                        'first_token', 'last_token', 'table_embedding', 'token_mean',
                    }
                    # embedding_type values → variant name mapping
                    _emb_type_to_variant = {
                        'cls': 'cls_embedding',
                        'cls_embedding': 'cls_embedding',
                        'column_mean': 'column_mean',
                        'column_sum': 'column_sum',  # legacy-only; remove after asset refresh
                        'first_token': 'first_token',
                        'last_token': 'last_token',
                        'table': 'table_embedding',
                        'table_embedding': 'table_embedding',
                        'token_mean': 'token_mean',
                    }

                    if 'variant' not in result:
                        # 1. Try path-based inference
                        rel_parts = json_file.relative_to(model_dir).parts
                        path_variant = None
                        for part in rel_parts[:-1]:  # skip filename
                            if part in known_variants:
                                path_variant = part
                                break
                        if path_variant:
                            result['variant'] = path_variant
                        # 2. Fall back to embedding_type if present
                        elif result.get('embedding_type') in _emb_type_to_variant:
                            result['variant'] = _emb_type_to_variant[result['embedding_type']]
                        # 3. Check hyperparameters.embedding_type
                        elif isinstance(result.get('hyperparameters'), dict):
                            ht = result['hyperparameters'].get('embedding_type')
                            if ht in _emb_type_to_variant:
                                result['variant'] = _emb_type_to_variant[ht]
                    result.setdefault('variant', None)

                    # ── Retrieval mode inference ──
                    if 'retrieval_mode' not in result:
                        rel_parts = json_file.relative_to(model_dir).parts
                        if 'model_only' in rel_parts[:-1]:
                            result['retrieval_mode'] = 'model_only'
                        elif result.get('task') == 'table_retrieval' and result.get('model') == 'bert':
                            result['retrieval_mode'] = 'model_only'
                        elif result.get('task') == 'table_retrieval':
                            result['retrieval_mode'] = 'hybrid'
                    result.setdefault('retrieval_mode', None)

                    # ── Statement-only inference ──
                    if 'statement_only' not in result:
                        rel_parts = json_file.relative_to(model_dir).parts
                        if 'statement_only' in rel_parts[:-1]:
                            result['statement_only'] = True
                        elif result.get('task') == 'table_fact_verification':
                            result['statement_only'] = False
                    result.setdefault('statement_only', None)

                    # ── Infer dataset from path when missing ──
                    if not result.get('dataset'):
                        if task_name == 'row_prediction':
                            # Path: model_dir/{dataset}/{label}/results.json
                            rel_parts = json_file.relative_to(model_dir).parts
                            # Skip seed dirs: parts like 'seed42'
                            non_seed = [p for p in rel_parts[:-1]
                                        if not re.match(r'^seed\d+$', p)]
                            if len(non_seed) >= 1:
                                result['dataset'] = non_seed[0]
                        else:
                            # Try sibling canonical file: {model}_{dataset}[_seed*].json
                            for sibling in json_file.parent.glob(f'{model_name}_*.json'):
                                if sibling.name == json_file.name:
                                    continue
                                stem = sibling.stem
                                prefix = f'{model_name}_'
                                if stem.startswith(prefix):
                                    ds = stem[len(prefix):]
                                    # Strip _seed suffix if present
                                    ds = re.sub(r'_seed\d+$', '', ds)
                                    if ds:
                                        result['dataset'] = ds
                                        break

                    # Detect if this is a canonical file (from extract_metrics)
                    # vs a generic results.json (from Python script)
                    result['_is_canonical'] = (
                        'slurm_job_id' in result or json_file.name != 'results.json'
                    )

                    raw_results.append(result)
                except (json.JSONDecodeError, Exception) as e:
                    print(f"Warning: Failed to parse {json_file}: {e}")

    # Deduplicate on composite key
    dedup = {}
    for r in raw_results:
        key = (
            r.get('task'),
            r.get('model'),
            r.get('dataset'),
            r.get('variant'),
            r.get('seed'),
            r.get('label_column'),
            r.get('head_type'),
            r.get('retrieval_mode'),
            r.get('statement_only'),
        )
        if key in dedup:
            # Prefer canonical files
            existing = dedup[key]
            if r.get('_is_canonical') and not existing.get('_is_canonical'):
                dedup[key] = r
            # Otherwise keep existing
        else:
            dedup[key] = r

    # Clean up internal fields
    results = []
    for r in dedup.values():
        r.pop('_is_canonical', None)
        results.append(r)

    return results


def flatten_result(result: dict) -> dict:
    """Flatten nested result dict for CSV export."""
    flat = {}

    for key, value in result.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                flat[f"{key}_{subkey}"] = subvalue
        else:
            flat[key] = value

    return flat


def generate_csv(results: list[dict], output_path: Path) -> pd.DataFrame:
    """
    Generate CSV summary of all results.

    Args:
        results: List of result dictionaries
        output_path: Path to output CSV file

    Returns:
        DataFrame of results
    """
    if not results:
        print("No results to aggregate")
        return pd.DataFrame()

    # Flatten results
    flat_results = [flatten_result(r) for r in results]

    # Create DataFrame
    df = pd.DataFrame(flat_results)

    # Reorder columns: task, model, variant, dataset, seed, label_column first
    priority_cols = ['task', 'model', 'variant', 'retrieval_mode', 'statement_only', 'dataset', 'seed', 'label_column', 'head_type', 'status']
    metric_cols = ['purity', 'nmi', 'ari', 'recall_at_gt', 'gt_coverage',
                   'micro_f1', 'macro_f1', 'MAP', 'best_MAP', 'best_micro_f1',
                   'accuracy', 'Recall@1', 'Recall@5', 'Recall@10', 'Recall@100', 'MRR',
                   'col_f1_at_10', 'col_recall_at_10', 'col_precision_at_10', 'col_map',
                   'map_at_k',
                   'test_results_f1', 'test_results_accuracy', 'test_results_weighted_f1',
                   'test_results_mse', 'test_results_r2', 'test_results_mae',
                   'test_results_macro_f1', 'test_results_precision', 'test_results_recall']

    ordered_cols = []
    for col in priority_cols + metric_cols:
        if col in df.columns:
            ordered_cols.append(col)

    remaining_cols = [c for c in df.columns if c not in ordered_cols]
    df = df[ordered_cols + sorted(remaining_cols)]

    # Sort by task, model, dataset, seed
    sort_cols = [c for c in ['task', 'model', 'variant', 'retrieval_mode', 'statement_only', 'dataset', 'seed', 'head_type'] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    # Save CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"CSV saved to: {output_path}")

    return df


# Primary metric lookup per task
TASK_METRICS = {
    'column_clustering': ('nmi', 'NMI'),
    'column_relation_prediction': ('micro_f1', 'Micro F1'),
    'column_type_prediction': ('micro_f1', 'Micro F1'),
    'join_classification': ('test_results_accuracy', 'Accuracy'),
    'join_containment': ('test_results_r2', 'R²'),
    'union_classification': ('test_results_accuracy', 'Accuracy'),
    'union_regression': ('test_results_r2', 'R²'),
    'table_subset': ('test_results_accuracy', 'Accuracy'),
    'table_fact_verification': ('accuracy', 'Accuracy'),
    'table_retrieval': ('Recall@100', 'Recall@100'),
    'semantic_parsing': ('accuracy', 'Accuracy'),
    'record_linkage': ('test_results_f1', 'F1'),
    'join_search': ('col_f1_at_10', 'COL F1@10'),
    'union_search': ('map_at_k', 'MAP@K'),
    'schema_matching': ('recall_at_gt', 'Recall@GT'),
    # row_prediction handled specially below (mixed classification/regression)
}


def _format_cell(mean_val, std_val, multi_seed: bool) -> str:
    """Format a cell as mean±std or plain value."""
    if pd.isna(mean_val):
        return "-"
    if multi_seed and pd.notna(std_val) and std_val > 0:
        return f"{mean_val:.4f}±{std_val:.4f}"
    return f"{mean_val:.4f}"


def generate_leaderboard(df: pd.DataFrame, output_path: Path) -> str:
    """
    Generate markdown leaderboard from results DataFrame.

    Shows mean±std when multiple seeds exist for a combination.
    Variants are shown as model/variant (e.g., starmie/cls_embedding).
    Row prediction labels are shown as dataset:label.

    Args:
        df: Results DataFrame
        output_path: Path to output markdown file

    Returns:
        Markdown content
    """
    if df.empty:
        return "# Downstream Task Results\n\nNo results available.\n"

    md_lines = [
        "# Downstream Task Results",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
    ]

    # Build display columns
    df = df.copy()

    # display_model: model/variant when variant is present, plus head_type suffix
    # when head_type != 'mlp' (default) to distinguish probe tiers in the leaderboard.
    def _build_display_model(r):
        base = r['model']
        if 'variant' in r and pd.notna(r.get('variant')):
            base = f"{base}/{r['variant']}"
        if 'head_type' in r and pd.notna(r.get('head_type')) and r.get('head_type') != 'mlp':
            base = f"{base} [{r['head_type']}]"
        if r.get('retrieval_mode') == 'model_only' and r.get('model') != 'bert':
            base = f"{base} [model_only]"
        if r.get('statement_only') is True:
            base = f"{base} [stmt_only]"
        return base

    df['display_model'] = df.apply(_build_display_model, axis=1)

    # display_dataset: dataset:label_column for row_prediction
    if 'label_column' in df.columns:
        df['display_dataset'] = df.apply(
            lambda r: f"{r.get('dataset', '')}:{r['label_column']}"
                if pd.notna(r.get('label_column')) and r.get('task') == 'row_prediction'
                else r.get('dataset', ''),
            axis=1
        )
    else:
        df['display_dataset'] = df.get('dataset', '')

    # Detect if we have multi-seed data
    has_multi_seed = 'seed' in df.columns and df['seed'].notna().any()

    # Group by task
    tasks = df['task'].unique() if 'task' in df.columns else []

    for task in sorted(tasks):
        task_df = df[df['task'] == task].copy()

        md_lines.append(f"## {task.replace('_', ' ').title()}")
        md_lines.append("")

        # Determine primary metric
        if task == 'row_prediction':
            # Mixed classification/regression: synthesize a unified metric
            # Flattened names: test_results → test_X becomes test_results_test_X
            metric_col = '_primary_metric'
            metric_name = 'Score (acc or R²)'
            # Find the actual column names (handle both naming conventions)
            acc_col = next((c for c in task_df.columns
                           if c in ('test_results_test_accuracy', 'test_results_accuracy')), None)
            r2_col = next((c for c in task_df.columns
                          if c in ('test_results_test_r2', 'test_results_r2')), None)
            task_df[metric_col] = task_df.apply(
                lambda r: r.get(acc_col)
                    if r.get('task_type') == 'classification' and acc_col
                    else r.get(r2_col) if r2_col else None,
                axis=1
            )
        elif task in TASK_METRICS:
            metric_col, metric_name = TASK_METRICS[task]
            # Fallback for alternative column names (e.g. historical runs)
            if metric_col not in task_df.columns:
                if task == 'column_relation_prediction' and 'best_micro_f1' in task_df.columns:
                    metric_col = 'best_micro_f1'
                elif task == 'column_type_prediction' and 'best_MAP' in task_df.columns:
                    metric_col = 'best_MAP'
                    metric_name = 'MAP'
                elif task == 'join_search' and 'f1_at_10' in task_df.columns:
                    # Fallback for pre-COL/TBL prefix runs
                    metric_col = 'f1_at_10'
                    metric_name = 'F1@10'
                elif task == 'record_linkage' and 'test_results_macro_f1' in task_df.columns:
                    # Pure-legacy fallback: no runs have f1 yet
                    metric_col = 'test_results_macro_f1'
                    metric_name = 'Macro F1 (legacy)'
                elif task == 'schema_matching' and 'micro_f1' in task_df.columns:
                    metric_col = 'micro_f1'
                    metric_name = 'Micro F1 (legacy)'
                else:
                    metric_col = None
        else:
            metric_col = None
            metric_name = 'Score'

        if metric_col and metric_col in task_df.columns:
            # Ensure metric is numeric
            task_df[metric_col] = pd.to_numeric(task_df[metric_col], errors='coerce')

            # Mixed-state coalescing: during metric transitions, fill NaN
            # gaps with the legacy column so all rows are represented.
            if task == 'record_linkage' and metric_col == 'test_results_f1':
                legacy_col = 'test_results_macro_f1'
                if legacy_col in task_df.columns:
                    task_df[legacy_col] = pd.to_numeric(task_df[legacy_col], errors='coerce')
                    task_df[metric_col] = task_df[metric_col].fillna(task_df[legacy_col])

            # Row-level coalescing for schema_matching: fill NaN recall_at_gt with micro_f1
            if task == 'schema_matching' and metric_col == 'recall_at_gt' and 'micro_f1' in task_df.columns:
                legacy_mask = task_df[metric_col].isna()
                if legacy_mask.any():
                    task_df.loc[legacy_mask, metric_col] = pd.to_numeric(
                        task_df.loc[legacy_mask, 'micro_f1'], errors='coerce')
                    metric_name = 'Recall@GT (legacy rows use Micro F1)'
                    print(f"  Note: {legacy_mask.sum()} schema_matching rows used micro_f1 fallback (legacy results)")

            # Compute mean and std across seeds
            group_cols = ['display_model', 'display_dataset']
            pivot_mean = task_df.pivot_table(
                index='display_model',
                columns='display_dataset',
                values=metric_col,
                aggfunc='mean'
            )
            pivot_std = task_df.pivot_table(
                index='display_model',
                columns='display_dataset',
                values=metric_col,
                aggfunc='std'
            )

            # Check if any group has >1 seed
            seed_counts = task_df.groupby(group_cols)['seed'].nunique() if 'seed' in task_df.columns else pd.Series()
            multi_seed = (seed_counts > 1).any() if len(seed_counts) > 0 else False

            # Sort by average performance
            pivot_mean['Average'] = pivot_mean.mean(axis=1)
            pivot_mean = pivot_mean.sort_values('Average', ascending=False)
            # Align std with mean
            pivot_std = pivot_std.reindex(index=pivot_mean.index, columns=pivot_mean.columns[:-1])

            # Format as markdown table
            datasets = [c for c in pivot_mean.columns if c != 'Average']
            header = f"| Model | {' | '.join(str(d) for d in datasets)} | Average |"
            separator = f"|-------|{' | '.join(['------:'] * len(datasets))} | ------:|"

            md_lines.append(f"**{metric_name}**")
            md_lines.append("")
            md_lines.append(header)
            md_lines.append(separator)

            for model in pivot_mean.index:
                row_values = []
                for ds in datasets:
                    mean_v = pivot_mean.loc[model, ds]
                    std_v = pivot_std.loc[model, ds] if ds in pivot_std.columns else np.nan
                    row_values.append(_format_cell(mean_v, std_v, multi_seed))
                avg = pivot_mean.loc[model, 'Average']
                row_values.append(f"**{avg:.4f}**" if pd.notna(avg) else "-")
                md_lines.append(f"| {model} | {' | '.join(row_values)} |")

            md_lines.append("")

            # Seed count warnings
            if has_multi_seed and 'seed' in task_df.columns:
                identity_cols = ['task', 'display_model', 'display_dataset']
                identity_cols = [c for c in identity_cols if c in task_df.columns]
                if identity_cols:
                    seed_counts_per_combo = task_df.groupby(identity_cols)['seed'].nunique()
                    max_seeds = seed_counts_per_combo.max()
                    if max_seeds > 1:
                        incomplete = seed_counts_per_combo[seed_counts_per_combo < max_seeds]
                        if len(incomplete) > 0:
                            md_lines.append(f"*Warning: {len(incomplete)} combination(s) have fewer than {max_seeds} seeds*")
                            md_lines.append("")

        # Add secondary metrics table for relation prediction
        if task == 'column_relation_prediction' and 'macro_f1' in task_df.columns:
            task_df['macro_f1'] = pd.to_numeric(task_df['macro_f1'], errors='coerce')
            pivot_macro = task_df.pivot_table(
                index='display_model',
                columns='display_dataset',
                values='macro_f1',
                aggfunc='mean'
            )
            pivot_macro_std = task_df.pivot_table(
                index='display_model',
                columns='display_dataset',
                values='macro_f1',
                aggfunc='std'
            )

            if not pivot_macro.empty:
                pivot_macro['Average'] = pivot_macro.mean(axis=1)
                pivot_macro = pivot_macro.sort_values('Average', ascending=False)
                pivot_macro_std = pivot_macro_std.reindex(
                    index=pivot_macro.index, columns=pivot_macro.columns[:-1]
                )

                datasets = [c for c in pivot_macro.columns if c != 'Average']
                header = f"| Model | {' | '.join(str(d) for d in datasets)} | Average |"
                separator = f"|-------|{' | '.join(['------:'] * len(datasets))} | ------:|"

                md_lines.append("**Macro F1**")
                md_lines.append("")
                md_lines.append(header)
                md_lines.append(separator)

                for model in pivot_macro.index:
                    row_values = []
                    for ds in datasets:
                        mean_v = pivot_macro.loc[model, ds]
                        std_v = pivot_macro_std.loc[model, ds] if ds in pivot_macro_std.columns else np.nan
                        row_values.append(_format_cell(mean_v, std_v, multi_seed))
                    avg = pivot_macro.loc[model, 'Average']
                    row_values.append(f"**{avg:.4f}**" if pd.notna(avg) else "-")
                    md_lines.append(f"| {model} | {' | '.join(row_values)} |")

                md_lines.append("")

    md_content = '\n'.join(md_lines)

    # Save markdown
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(md_content)
    print(f"Leaderboard saved to: {output_path}")

    return md_content


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate downstream task results into summary files',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--tasks', nargs='+',
                       help='Aggregate only these tasks')
    parser.add_argument('--output-dir',
                       help='Output directory for summary files')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--chance-baselines', action='store_true',
                       help='Include analytical chance baselines')

    args = parser.parse_args()

    # Get paths
    project_root = get_project_root()
    results_dir = project_root / 'results' / 'evaluation'

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_dir / 'summary'

    # Collect results
    print("Collecting results...")
    results = collect_results(results_dir, args.tasks)

    # Inject chance baselines BEFORE empty-results guard so
    # --chance-baselines works even when no real results exist yet.
    if args.chance_baselines:
        _root = str(project_root)
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from trl_bench.baselines.chance import compute_chance_baselines
        chance_results = compute_chance_baselines(project_root, args.tasks)
        results.extend(chance_results)
        if args.verbose:
            print(f"  Added {len(chance_results)} chance baseline entries")

    if not results:
        print("No results found to aggregate")
        sys.exit(0)

    print(f"Found {len(results)} result files")

    if args.verbose:
        tasks_found = set(r.get('task', 'unknown') for r in results)
        models_found = set(r.get('model', 'unknown') for r in results)
        seeds_found = set(r.get('seed') for r in results if r.get('seed') is not None)
        print(f"  Tasks: {sorted(tasks_found)}")
        print(f"  Models: {sorted(models_found)}")
        if seeds_found:
            print(f"  Seeds: {sorted(seeds_found)}")

    # Generate CSV
    print("\nGenerating CSV summary...")
    csv_path = output_dir / 'all_results.csv'
    df = generate_csv(results, csv_path)

    # Generate leaderboard
    print("\nGenerating leaderboard...")
    md_path = output_dir / 'leaderboard.md'
    generate_leaderboard(df, md_path)

    print("\nDone!")


if __name__ == '__main__':
    main()
