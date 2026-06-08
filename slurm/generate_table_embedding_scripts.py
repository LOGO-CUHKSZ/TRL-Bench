#!/usr/bin/env python3
"""
Unified SLURM script generator for table embedding generation.

Handles BOTH types of table embedding jobs:

1. Derived models (BERT, TAPAS, etc.): CPU-only jobs that read existing column
   embedding pkls and derive table embeddings via scripts/generate_table_embeddings.py.

2. Native models (TUTA): GPU jobs that run a model-specific script directly on
   CSV files to produce table embeddings with native [CLS] tokens.

Usage:
    # Generate all table embedding scripts (derived + native)
    python generate_table_embedding_scripts.py

    # Generate for specific models
    python generate_table_embedding_scripts.py --models tuta bert

    # Generate for specific datasets
    python generate_table_embedding_scripts.py --datasets sato wtq

    # Dry run
    python generate_table_embedding_scripts.py --dry-run
"""

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

_SHARD_RE = re.compile(r'.*_shard\d+of\d+\.pkl$')

def _is_shard_file(pkl_name: str) -> bool:
    """Return True if the pkl filename looks like a shard or checkpoint file."""
    return bool(_SHARD_RE.match(pkl_name))

import local_overrides as local

from generate_scripts import (
    get_project_root,
    load_yaml,
    list_csv_files,
    resolve_tables_dir,
    parse_memory,
    format_memory,
)
from generate_table_lists import generate_table_lists


# ---------------------------------------------------------------------------
# Derived table embeddings (from column pkls) — moved from generate_scripts.py
# ---------------------------------------------------------------------------

def generate_derived_table_scripts(
    project_root: Path,
    resources_config: dict,
    datasets_config: dict,
    args,
) -> int:
    """Generate sbatch scripts for derived table embeddings (column→table).

    Returns the number of scripts generated.
    """
    table_template_path = (
        project_root / 'slurm' / 'scripts' / 'templates' / 'table_embedding_job.sbatch.template'
    )

    if not table_template_path.exists():
        print(f"\nWARN: Table embedding template not found: {table_template_path}")
        return 0

    with open(table_template_path, 'r') as f:
        table_template_content = f.read()

    paths_config = resources_config.get('paths', {})
    col_emb_dir = project_root / paths_config.get('output_dir', 'embeddings/column')
    table_scripts_dir = project_root / paths_config.get(
        'table_scripts_dir', 'slurm/scripts/generated/table_embeddings'
    )

    if not args.dry_run:
        table_scripts_dir.mkdir(parents=True, exist_ok=True)

    # Load native model names so we skip them from the derived path
    config_dir = project_root / 'slurm' / 'config'
    table_models_path = config_dir / 'table_models.yaml'
    native_model_names = set()
    if table_models_path.exists():
        table_models_config = local.apply(load_yaml(table_models_path))
        native_model_names = set((table_models_config.get('models') or {}).keys())

    # Only include datasets marked with table_embedding: true (unless --datasets is explicit)
    if args.datasets:
        table_emb_datasets = set(args.datasets)
    else:
        table_emb_datasets = {
            name for name, cfg in datasets_config['datasets'].items()
            if cfg.get('table_embedding', False)
        }

    print(f"\nGenerating derived table embedding scripts...")
    print(f"  Table embedding datasets: {sorted(table_emb_datasets)}")
    print("-" * 60)

    generated = 0
    if not col_emb_dir.exists():
        print(f"  No column embeddings found at {col_emb_dir}")
        return 0

    for model_dir in sorted(col_emb_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        # Skip native models — they have their own generation path
        if model_name in native_model_names:
            if args.verbose:
                print(f"  SKIP {model_name}: native model (handled separately)")
            continue

        # Skip backup directories (e.g. tuta_backup_20260305_100139)
        if '_backup_' in model_name:
            if args.verbose:
                print(f"  SKIP {model_name}: backup directory")
            continue

        if args.models and model_name not in args.models:
            continue

        for pkl_file in sorted(model_dir.glob('*.pkl')):
            if pkl_file.name.endswith('.checkpoint.pkl'):
                continue
            if _is_shard_file(pkl_file.name):
                continue
            dataset_name = pkl_file.stem
            if dataset_name not in table_emb_datasets:
                continue

            logs_dir = project_root / paths_config.get('logs_dir', 'slurm/logs')
            status_file = project_root / paths_config.get(
                'status_file', 'slurm/logs/status/job_status.json'
            )

            # Build SLURM directives
            extra_directives = []
            slurm_defaults = resources_config.get('slurm_defaults', {})
            if slurm_defaults.get('account'):
                extra_directives.append(f"#SBATCH --account={slurm_defaults['account']}")
            if slurm_defaults.get('mail_user'):
                extra_directives.append(
                    f"#SBATCH --mail-type={slurm_defaults.get('mail_type', 'FAIL')}"
                )
                extra_directives.append(f"#SBATCH --mail-user={slurm_defaults['mail_user']}")

            # Table embedding jobs are CPU-only and lightweight
            substitutions = {
                'JOB_NAME': local.job_name(f"table_{model_name}_{dataset_name}"),
                'MODEL': model_name,
                'DATASET': dataset_name,
                'TIME_LIMIT': '01:00:00',
                'MEMORY': '16G',
                'CPUS': '4',
                'PARTITION': local.partition('cpubase_bycore_b2'),
                'EXTRA_SLURM_DIRECTIVES': '\n'.join(extra_directives) if extra_directives else '',
                'PROJECT_ROOT': str(project_root),
                'COLUMN_EMB_PATH': str(pkl_file),
                'OUTPUT_PATH': str(
                    project_root
                    / paths_config.get('table_output_dir', 'embeddings/table')
                    / model_name
                    / f'{dataset_name}.pkl'
                ),
                'STATUS_FILE': str(status_file),
                'LOG_DIR': str(logs_dir),
                'TIMESTAMP': datetime.now().isoformat(),
                'ENV_SETUP': local.env_setup('source load_env'),
            }

            script_name = f"{model_name}_{dataset_name}.sbatch"
            script_path = table_scripts_dir / script_name

            if args.dry_run:
                print(f"  [DRY-RUN] Would generate: {script_name}")
            else:
                result = table_template_content
                for key, value in substitutions.items():
                    result = result.replace(f'${{{key}}}', str(value))

                with open(script_path, 'w') as f:
                    f.write(result)
                os.chmod(script_path, 0o755)

                if args.verbose:
                    print(f"  Generated: {script_name}")
                else:
                    print(f"  {script_name}")

            generated += 1

    return generated


# ---------------------------------------------------------------------------
# Native table embeddings (model runs directly on CSVs)
# ---------------------------------------------------------------------------

def _fill_template(template_content: str, substitutions: dict) -> str:
    """Apply substitutions to a template string."""
    result = template_content
    for key, value in substitutions.items():
        result = result.replace(f'${{{key}}}', str(value))
    return result


def generate_native_table_script(
    model_name: str,
    model_config: dict,
    dataset_name: str,
    dataset_config: dict,
    resources: dict,
    project_root: Path,
    template_content: str,
    additional_args: list[str] | None = None,
    output_path_override: str | None = None,
    job_name_override: str | None = None,
    shard_type: str = '',
    shard_index: int = 0,
    num_shards: int = 1,
) -> str:
    """Generate a single sbatch script for a native table model-dataset combination."""

    # Resource allocation
    size_category = dataset_config.get('size_category', 'MEDIUM')
    base_resources = dict(resources['size_categories'][size_category])

    # Apply model-specific resource overrides
    model_overrides = model_config.get('resource_overrides', {}) or {}
    resource_override = model_overrides.get(dataset_name, {}) or {}
    for key in ['time_limit', 'gpus', 'cpus', 'partition', 'memory']:
        if key in resource_override:
            base_resources[key] = resource_override[key]

    final_memory_mb = parse_memory(base_resources['memory'])

    # Resolve paths
    paths_config = resources.get('paths', {})
    datasets_dir = project_root / paths_config.get('datasets_dir', 'data')
    table_output_dir = project_root / paths_config.get(
        'table_output_dir', 'embeddings/table'
    )
    logs_dir = project_root / paths_config.get('logs_dir', 'slurm/logs')
    status_file = project_root / paths_config.get(
        'status_file', 'slurm/logs/status/job_status.json'
    )
    timing_file = project_root / paths_config.get(
        'timing_file', 'slurm/logs/timing/timing_summary.csv'
    )

    # Resolve input directory (same logic as row scripts)
    row_input = dataset_config.get("row_input_dir")
    if row_input:
        p = Path(row_input)
        input_dir = p if p.is_absolute() else datasets_dir / row_input
    else:
        input_dir = resolve_tables_dir(dataset_name, dataset_config, datasets_dir)

    output_path = output_path_override or str(
        table_output_dir / model_name / f"{dataset_name}.pkl"
    )

    # Build SLURM directives
    extra_directives = []
    slurm_defaults = resources.get('slurm_defaults', {})
    if slurm_defaults.get('account'):
        extra_directives.append(f"#SBATCH --account={slurm_defaults['account']}")
    if slurm_defaults.get('mail_user'):
        extra_directives.append(
            f"#SBATCH --mail-type={slurm_defaults.get('mail_type', 'FAIL')}"
        )
        extra_directives.append(f"#SBATCH --mail-user={slurm_defaults['mail_user']}")

    # Build extra arguments
    defaults = dict(model_config.get('defaults', {}))
    args_mapping = model_config.get('args', {})
    extra_args_list = []

    # Add model checkpoint path
    model_checkpoint = model_config.get('checkpoint')
    if model_checkpoint and 'model_path' in args_mapping:
        ckpt_path = str(project_root / model_checkpoint)
        extra_args_list.append(f"{args_mapping['model_path']} \"{ckpt_path}\"")

    # Add default arguments
    skip_params = {'input_dir', 'output_path', 'model_path'}
    for param, value in defaults.items():
        if param in skip_params:
            continue
        if param in args_mapping and value is not None:
            if isinstance(value, list):
                extra_args_list.append(
                    f"{args_mapping[param]} {' '.join(str(v) for v in value)}"
                )
            else:
                extra_args_list.append(f"{args_mapping[param]} {value}")

    # Append additional args (e.g., --table_list for sharded runs)
    if additional_args:
        extra_args_list.extend(additional_args)

    # Build JOB_KEY: unique per shard, base dataset for non-sharded
    if shard_type == 'shard':
        job_key = f"ntable_{model_name}_{dataset_name}_shard{shard_index}of{num_shards}"
    else:
        job_key = f"ntable_{model_name}_{dataset_name}"

    substitutions = {
        'JOB_NAME': local.job_name(job_name_override or f"ntable_{model_name}_{dataset_name}"),
        'JOB_KEY': job_key,
        'MODEL': model_name,
        'DATASET': dataset_name,
        'TIME_LIMIT': base_resources['time_limit'],
        'MEMORY': format_memory(final_memory_mb),
        'GPUS': str(base_resources.get('gpus', 1)),
        'CPUS': str(base_resources.get('cpus', 4)),
        'PARTITION': base_resources.get('partition', 'gpu'),
        'EXTRA_SLURM_DIRECTIVES': '\n'.join(extra_directives) if extra_directives else '',
        'PROJECT_ROOT': str(project_root),
        'INPUT_DIR': str(input_dir),
        'OUTPUT_PATH': str(output_path),
        'STATUS_FILE': str(status_file),
        'TIMING_FILE': str(timing_file),
        'LOG_DIR': str(logs_dir),
        'TIMESTAMP': datetime.now().isoformat(),
        'ENV_SETUP': local.env_setup(model_config.get('env_setup', 'source load_env')),
        'SCRIPT_PATH': model_config['script'],
        'EXTRA_ARGS': ' \\\n    '.join(extra_args_list) if extra_args_list else '',
        'SHARD_TYPE': shard_type,
        'SHARD_INDEX': str(shard_index),
        'NUM_SHARDS': str(num_shards),
    }

    return _fill_template(template_content, substitutions)


def _build_merge_substitutions(
    model_name: str,
    dataset_name: str,
    resources_config: dict,
    project_root: Path,
    shard_output_paths: list[str],
    final_output_path: str,
    num_shards: int,
    embedding_type: str = 'native_table',
) -> dict:
    """Build template substitutions for a native table merge sbatch script."""
    paths_config = resources_config.get('paths', {})
    logs_dir = project_root / paths_config.get('logs_dir', 'slurm/logs')
    status_file = project_root / paths_config.get(
        'status_file', 'slurm/logs/status/job_status.json'
    )

    extra_directives = []
    slurm_defaults = resources_config.get('slurm_defaults', {})
    if slurm_defaults.get('account'):
        extra_directives.append(f"#SBATCH --account={slurm_defaults['account']}")
    if slurm_defaults.get('mail_user'):
        extra_directives.append(
            f"#SBATCH --mail-type={slurm_defaults.get('mail_type', 'FAIL')}"
        )
        extra_directives.append(f"#SBATCH --mail-user={slurm_defaults['mail_user']}")

    cpu_partition = resources_config.get('cpu_partition', 'cpu')

    return {
        'MODEL': model_name,
        'DATASET': dataset_name,
        'MERGE_JOB_NAME': local.job_name(f'merge_{model_name}_{dataset_name}'),
        'MERGE_JOB_KEY': f'{model_name}_{dataset_name}_merge',
        'MERGE_LOG_PREFIX': 'ntable_',
        'PROJECT_ROOT': str(project_root),
        'STATUS_FILE': str(status_file),
        'OUTPUT_PATH': final_output_path,
        'SHARD_PATHS': ' '.join(f'"{p}"' for p in shard_output_paths),
        'NUM_SHARDS': str(num_shards),
        'LOG_DIR': str(logs_dir),
        'TIMESTAMP': datetime.now().isoformat(),
        'ENV_SETUP': local.env_setup('source load_env'),
        'CPU_PARTITION': cpu_partition,
        'EMBEDDING_TYPE': embedding_type,
        'EXTRA_SLURM_DIRECTIVES': '\n'.join(extra_directives) if extra_directives else '',
    }


def generate_native_table_scripts(
    project_root: Path,
    resources_config: dict,
    args,
) -> int:
    """Generate sbatch scripts for native table embedding models.

    Supports sharding: when a dataset has shards > 1, generates N shard
    scripts (each with --table_list) plus a merge script.

    Returns the number of scripts generated.
    """
    config_dir = project_root / 'slurm' / 'config'
    table_models_path = config_dir / 'table_models.yaml'

    if not table_models_path.exists():
        if args.verbose:
            print(f"\nNo table_models.yaml found at {table_models_path}")
        return 0

    template_path = (
        project_root / 'slurm' / 'scripts' / 'templates'
        / 'native_table_embedding_job.sbatch.template'
    )
    if not template_path.exists():
        print(f"\nWARN: Native table embedding template not found: {template_path}")
        return 0

    merge_template_path = (
        project_root / 'slurm' / 'scripts' / 'templates' / 'merge_shards.sbatch.template'
    )
    merge_template = merge_template_path.read_text() if merge_template_path.exists() else None

    table_models_config = local.apply(load_yaml(table_models_path))
    datasets_config = local.apply(load_yaml(config_dir / 'datasets.yaml'))

    with open(template_path, 'r') as f:
        template_content = f.read()

    available_models = table_models_config.get('models', {}) or {}
    models = args.models or list(available_models.keys())

    # Only include datasets marked with table_embedding: true (unless --datasets is explicit)
    if args.datasets:
        datasets = args.datasets
    else:
        datasets = [
            name for name, cfg in datasets_config['datasets'].items()
            if cfg.get('table_embedding', False)
        ]

    # Filter to requested models that are actually native
    models = [m for m in models if m in available_models]
    if not models:
        return 0

    # Output directories
    paths_config = resources_config.get('paths', {})
    table_scripts_dir = project_root / paths_config.get(
        'table_scripts_dir', 'slurm/scripts/generated/table_embeddings'
    )
    table_output_dir = project_root / paths_config.get(
        'table_output_dir', 'embeddings/table'
    )
    datasets_dir = project_root / paths_config.get('datasets_dir', 'data')
    table_lists_dir = project_root / 'slurm' / 'scripts' / 'generated' / 'table_lists'

    if not args.dry_run:
        table_scripts_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating native table embedding scripts...")
    print("-" * 60)

    generated = 0
    skipped = 0
    table_lists_generated: set[str] = set()

    for model_name in sorted(models):
        model_config = available_models[model_name]

        # Allow per-model shard overrides
        model_shard_overrides = model_config.get('shard_overrides', {}) or {}

        for dataset_name in sorted(datasets):
            dataset_config = datasets_config['datasets'].get(dataset_name)
            if dataset_config is None:
                if args.verbose:
                    print(f"  SKIP {model_name}/{dataset_name}: dataset not in config")
                skipped += 1
                continue

            # Resolve and check input directory
            row_input = dataset_config.get("row_input_dir")
            if row_input:
                p = Path(row_input)
                input_dir = p if p.is_absolute() else datasets_dir / row_input
            else:
                try:
                    input_dir = resolve_tables_dir(dataset_name, dataset_config, datasets_dir)
                except Exception as e:
                    if args.verbose:
                        print(f"  SKIP {model_name}/{dataset_name}: {e}")
                    skipped += 1
                    continue

            if not input_dir.exists():
                if args.verbose:
                    print(f"  SKIP {model_name}/{dataset_name}: tables dir not found: {input_dir}")
                skipped += 1
                continue

            csv_count = len(list_csv_files(input_dir))
            if csv_count == 0:
                if args.verbose:
                    print(f"  SKIP {model_name}/{dataset_name}: no CSV files in {input_dir}")
                skipped += 1
                continue

            num_shards = dataset_config.get('shards', 1)
            if dataset_name in model_shard_overrides:
                num_shards = model_shard_overrides[dataset_name]

            if num_shards > 1:
                # =============================================================
                # SHARDED PATH: N shard scripts + 1 merge script
                # =============================================================

                # Generate table list files (once per dataset+shard_count combo)
                table_list_key = f"{dataset_name}_{num_shards}"
                if table_list_key not in table_lists_generated:
                    if not args.dry_run:
                        generate_table_lists(dataset_name, str(input_dir), num_shards)
                    table_lists_generated.add(table_list_key)

                shard_output_paths = []
                for i in range(num_shards):
                    shard_suffix = f"_shard{i}of{num_shards}"
                    table_list_file = table_lists_dir / f"{dataset_name}_shard{i}of{num_shards}.txt"
                    shard_output = table_output_dir / model_name / f"{dataset_name}{shard_suffix}.pkl"
                    shard_output_paths.append(str(shard_output))

                    script_name = f"{model_name}_{dataset_name}_shard{i}of{num_shards}.sbatch"
                    script_path = table_scripts_dir / script_name

                    if args.dry_run:
                        print(f"  [DRY-RUN] Would generate: {script_name}")
                    else:
                        script_content = generate_native_table_script(
                            model_name, model_config,
                            dataset_name, dataset_config,
                            resources_config,
                            project_root, template_content,
                            additional_args=[f'--table_list {table_list_file}'],
                            output_path_override=str(shard_output),
                            job_name_override=f"ntable_{model_name}_{dataset_name}_s{i}",
                            shard_type='shard',
                            shard_index=i,
                            num_shards=num_shards,
                        )
                        with open(script_path, 'w') as f:
                            f.write(script_content)
                        os.chmod(script_path, 0o755)
                        if args.verbose:
                            print(f"  Generated: {script_name}")
                        else:
                            print(f"  {script_name}")

                    generated += 1

                # Generate merge script
                final_output = table_output_dir / model_name / f"{dataset_name}.pkl"
                merge_script_name = f"{model_name}_{dataset_name}_merge.sbatch"
                merge_script_path = table_scripts_dir / merge_script_name

                if args.dry_run:
                    print(f"  [DRY-RUN] Would generate: {merge_script_name}")
                elif merge_template:
                    merge_subs = _build_merge_substitutions(
                        model_name, dataset_name, resources_config, project_root,
                        shard_output_paths, str(final_output), num_shards,
                    )
                    merge_content = _fill_template(merge_template, merge_subs)
                    with open(merge_script_path, 'w') as f:
                        f.write(merge_content)
                    os.chmod(merge_script_path, 0o755)
                    if args.verbose:
                        print(f"  Generated: {merge_script_name} (merge {num_shards} shards)")
                    else:
                        print(f"  {merge_script_name}")
                else:
                    print(f"  WARN: merge template not found, skipping {merge_script_name}")

                generated += 1  # count merge script

                if args.verbose:
                    print(f"    {model_name}/{dataset_name}: {num_shards} shards + merge ({csv_count} CSVs)")
            else:
                # =============================================================
                # UNSHARDED PATH: single script
                # =============================================================
                script_name = f"{model_name}_{dataset_name}.sbatch"
                script_path = table_scripts_dir / script_name

                if args.dry_run:
                    print(f"  [DRY-RUN] Would generate: {script_name}")
                    if args.verbose:
                        print(f"            Input: {input_dir} ({csv_count} CSVs)")
                else:
                    script_content = generate_native_table_script(
                        model_name, model_config,
                        dataset_name, dataset_config,
                        resources_config,
                        project_root, template_content,
                    )

                    with open(script_path, 'w') as f:
                        f.write(script_content)
                    os.chmod(script_path, 0o755)

                    if args.verbose:
                        print(f"  Generated: {script_name} ({csv_count} CSVs)")
                    else:
                        print(f"  {script_name}")

                generated += 1

    if skipped and args.verbose:
        print(f"  Skipped: {skipped} native table combinations")

    return generated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate SLURM sbatch scripts for table embedding generation (derived + native)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--models', nargs='+', help='Generate scripts only for these models')
    parser.add_argument('--datasets', nargs='+', help='Generate scripts only for these datasets')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be generated without writing')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    project_root = get_project_root()
    config_dir = project_root / 'slurm' / 'config'
    resources_config = local.apply(load_yaml(config_dir / 'resources.yaml'))
    datasets_config = local.apply(load_yaml(config_dir / 'datasets.yaml'))

    # Validate --models and --datasets upfront
    if args.models:
        known_models = set()
        # Configured column models
        col_models_path = config_dir / 'models.yaml'
        if col_models_path.exists():
            col_cfg = local.apply(load_yaml(col_models_path))
            known_models |= set((col_cfg.get('models') or {}).keys())
        # Native table models
        table_models_path = config_dir / 'table_models.yaml'
        if table_models_path.exists():
            tm = local.apply(load_yaml(table_models_path))
            known_models |= set((tm.get('models') or {}).keys())
        # Materialized column embedding dirs (includes hybrids not in config)
        col_emb_dir = project_root / resources_config.get('paths', {}).get(
            'output_dir', 'embeddings/column'
        )
        if col_emb_dir.exists():
            known_models |= {
                d.name for d in col_emb_dir.iterdir()
                if d.is_dir() and '_backup_' not in d.name
            }
        invalid = set(args.models) - known_models
        if invalid:
            print(f"Error: Unknown models: {invalid}")
            print(f"Available: {sorted(known_models)}")
            sys.exit(1)

    if args.datasets:
        known_datasets = set(datasets_config['datasets'].keys())
        invalid = set(args.datasets) - known_datasets
        if invalid:
            print(f"Error: Unknown datasets: {invalid}")
            print(f"Available: {sorted(known_datasets)}")
            sys.exit(1)

    derived_count = generate_derived_table_scripts(project_root, resources_config, datasets_config, args)
    native_count = generate_native_table_scripts(project_root, resources_config, args)

    total = derived_count + native_count
    print("=" * 60)
    if derived_count:
        print(f"Generated: {derived_count} derived table embedding scripts")
    if native_count:
        print(f"Generated: {native_count} native table embedding scripts")
    print(f"Total:     {total} table embedding scripts")
    if not args.dry_run:
        paths_config = resources_config.get('paths', {})
        table_scripts_dir = project_root / paths_config.get(
            'table_scripts_dir', 'slurm/scripts/generated/table_embeddings'
        )
        print(f"Output:    {table_scripts_dir}")


if __name__ == '__main__':
    main()
