#!/usr/bin/env python3
"""
Generate SLURM sbatch scripts from configuration files.

This script reads the model, dataset, and resource configurations and generates
individual sbatch scripts for each model-dataset combination.

Usage:
    # Generate all scripts
    python generate_scripts.py

    # Generate for specific models
    python generate_scripts.py --models starmie tapas

    # Generate for specific datasets
    python generate_scripts.py --datasets wtq sato

    # Dry run (show what would be generated)
    python generate_scripts.py --dry-run
"""

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from string import Template

import yaml

import local_overrides as local


def get_project_root() -> Path:
    """Get the project root directory.

    File at slurm/generate_scripts.py; script_dir is slurm/, so one .parent
    reaches the repo root. Two .parent would land ABOVE the repo.
    """
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent


def load_yaml(path: Path) -> dict:
    """Load a YAML configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def list_csv_files(directory: Path) -> list[Path]:
    """List CSV files in a directory, including dot-prefixed files."""
    results = []
    with os.scandir(directory) as it:
        for entry in it:
            if entry.is_file() and entry.name.endswith(".csv"):
                results.append(Path(entry.path))
    return sorted(results)


def find_tables_dir(dataset_path: Path, config_tables_dir: str | None) -> str:
    """
    Find the directory containing CSV files for a dataset.

    Args:
        dataset_path: Path to the dataset directory
        config_tables_dir: Configured tables_dir, or None for auto-detection

    Returns:
        Relative path to tables directory from dataset root
    """
    if config_tables_dir is not None:
        return config_tables_dir

    # Auto-detection: check common locations
    candidates = ['tables', 'csv', 'datalake', 'datasets', '.']
    for subdir in candidates:
        candidate = dataset_path / subdir if subdir != '.' else dataset_path
        if candidate.exists() and list_csv_files(candidate):
            return subdir if subdir != '.' else '.'

    # Fallback: find any directory with CSVs
    for d in dataset_path.iterdir():
        if d.is_dir() and list_csv_files(d):
            return d.name

    return '.'


def resolve_tables_dir(
    dataset_name: str,
    dataset_config: dict,
    datasets_root: Path
) -> Path:
    """
    Resolve the actual tables directory path for a dataset.

    Handles tables_source redirects (e.g., wiki_containment uses wiki-join-search-deepjoin tables).

    Args:
        dataset_name: Name of the dataset
        dataset_config: Dataset configuration dict
        datasets_root: Root path for all datasets

    Returns:
        Absolute path to the tables directory
    """
    # Check if this dataset uses another dataset's tables
    tables_source = dataset_config.get('tables_source')
    if tables_source:
        source_path = datasets_root / tables_source
        tables_dir = find_tables_dir(source_path, dataset_config.get('tables_dir'))
        return source_path / tables_dir

    # Use this dataset's own tables
    dataset_path = datasets_root / dataset_name
    tables_dir = find_tables_dir(dataset_path, dataset_config.get('tables_dir'))
    return dataset_path / tables_dir


def parse_memory(memory_str: str) -> int:
    """Parse memory string (e.g., '32G') to megabytes."""
    memory_str = memory_str.upper().strip()
    if memory_str.endswith('G'):
        return int(memory_str[:-1]) * 1024
    elif memory_str.endswith('M'):
        return int(memory_str[:-1])
    elif memory_str.endswith('T'):
        return int(memory_str[:-1]) * 1024 * 1024
    return int(memory_str)


def format_memory(mb: int) -> str:
    """Format memory in megabytes to human-readable string."""
    if mb >= 1024:
        return f"{mb // 1024}G"
    return f"{mb}M"


def generate_script(
    model_name: str,
    model_config: dict,
    dataset_name: str,
    dataset_config: dict,
    resources: dict,
    project_root: Path,
    template_content: str,
    repair_defaults: dict,
    shard_overrides: dict | None = None,
    additional_args: list[str] | None = None,
) -> str:
    """
    Generate a single sbatch script for a model-dataset combination.

    Args:
        model_name: Name of the model
        model_config: Model configuration dict
        dataset_name: Name of the dataset
        dataset_config: Dataset configuration dict
        resources: Resource configuration dict
        project_root: Project root path
        template_content: SBATCH template content

    Returns:
        Generated script content
    """
    # Get resource allocation for this dataset size
    size_category = dataset_config.get('size_category', 'MEDIUM')
    base_resources = dict(resources['size_categories'][size_category])
    dataset_overrides = resources.get('dataset_overrides', {}) or {}
    dataset_override = dataset_overrides.get(dataset_name, {}) or {}
    # Apply non-memory overrides to base resources (memory handled separately)
    for key in ['time_limit', 'gpus', 'cpus', 'partition']:
        if key in dataset_override:
            base_resources[key] = dataset_override[key]

    # Apply model-specific memory multiplier
    memory_multipliers = resources.get('model_memory_multipliers', {})
    model_multiplier = memory_multipliers.get(model_name, 1.0)
    base_memory_mb = parse_memory(base_resources['memory'])
    override_memory = dataset_override.get('memory')
    if override_memory:
        final_memory_mb = parse_memory(override_memory)
    else:
        final_memory_mb = int(base_memory_mb * model_multiplier)

    # Resolve paths
    paths_config = resources.get('paths', {})
    datasets_dir = project_root / paths_config.get('datasets_dir', 'datasets')
    output_dir = project_root / paths_config.get('output_dir', 'embeddings/column')
    logs_dir = project_root / paths_config.get('logs_dir', 'slurm/logs')
    status_file = project_root / paths_config.get('status_file', 'slurm/logs/status/job_status.json')
    timing_file = project_root / paths_config.get('timing_file', 'slurm/logs/timing/timing_summary.csv')

    # Resolve input directory (tables location)
    input_dir = resolve_tables_dir(dataset_name, dataset_config, datasets_dir)

    # Build output path
    output_path = output_dir / model_name / f"{dataset_name}.pkl"

    # Resolve checkpoint path and pretraining config
    checkpoint = model_config['checkpoint']
    checkpoint_type = model_config.get('checkpoint_type')
    pretrain_enabled = False
    pretrain_script = ''
    pretrain_args = ''

    if checkpoint_type == 'huggingface':
        pass  # HuggingFace model ID, used as-is
    elif checkpoint == 'auto':
        # Resolve to dataset-specific checkpoint
        # Pattern: checkpoints/{model}/{dataset}/{tables_subdir}/{checkpoint_filename}
        pretrain_config = model_config.get('pretrain', {})
        checkpoint_filename = pretrain_config.get('checkpoint_filename', 'model.pt')
        tables_subdir = input_dir.name  # e.g., 'csv', 'tables', 'datalake'
        auto_checkpoint = project_root / 'checkpoints' / model_name / dataset_name / tables_subdir / checkpoint_filename
        checkpoint = str(auto_checkpoint)

        # Enable pretraining if model has pretrain config
        if pretrain_config:
            pretrain_enabled = True
            pretrain_script = pretrain_config.get('script', '')
            # Build pretrain args from config
            pretrain_args_list = []
            pretrain_defaults = pretrain_config.get('defaults', {})
            pretrain_arg_mapping = pretrain_config.get('args', {})
            pretrain_flag_mapping = pretrain_config.get('flag_args', {})

            # Handle regular args (--arg value)
            for param, value in pretrain_defaults.items():
                if param in pretrain_arg_mapping:
                    arg_name = pretrain_arg_mapping[param]
                    pretrain_args_list.append(f'{arg_name} {value}')
                elif param in pretrain_flag_mapping:
                    # Handle flag args (--flag when true)
                    if value:
                        pretrain_args_list.append(pretrain_flag_mapping[param])

            pretrain_args = ' '.join(pretrain_args_list)
    elif not checkpoint.startswith('google/'):
        checkpoint = str(project_root / checkpoint)

    # Build extra SLURM directives
    extra_directives = []
    slurm_defaults = resources.get('slurm_defaults', {})
    if slurm_defaults.get('account'):
        extra_directives.append(f"#SBATCH --account={slurm_defaults['account']}")
    if slurm_defaults.get('qos'):
        extra_directives.append(f"#SBATCH --qos={slurm_defaults['qos']}")
    if slurm_defaults.get('constraint'):
        extra_directives.append(f"#SBATCH --constraint={slurm_defaults['constraint']}")
    if slurm_defaults.get('mail_user'):
        extra_directives.append(f"#SBATCH --mail-type={slurm_defaults.get('mail_type', 'FAIL')}")
        extra_directives.append(f"#SBATCH --mail-user={slurm_defaults['mail_user']}")
    if dataset_override.get('extra_slurm_directives'):
        extra_directives.extend(dataset_override['extra_slurm_directives'])

    # Build extra arguments for the Python script
    defaults = model_config.get('defaults', {})
    dataset_overrides = model_config.get('dataset_overrides', {}) or {}
    override_defaults = dataset_overrides.get(dataset_name, {}) or {}
    merged_defaults = dict(defaults)
    merged_defaults.update(override_defaults)
    args = model_config.get('args', {})
    extra_args_list = []

    # Add mode argument if required (TURL, Doduo)
    if model_config.get('extra_args'):
        extra_args_list.append(model_config['extra_args'])

    # Add default arguments
    for param, value in merged_defaults.items():
        if param in args and value is not None:
            extra_args_list.append(f"{args[param]} {value}")

    # Append any additional args (e.g., --table_list for sharded runs)
    if additional_args:
        extra_args_list.extend(additional_args)

    # Repair configuration
    repair_cfg = dict(repair_defaults or {})
    repair_cfg.update(model_config.get('repair', {}) or {})
    repair_enabled = bool(repair_cfg.get('enabled', False))
    repair_chunk_size = repair_cfg.get('chunk_size', 64)
    repair_max_rows = repair_cfg.get('max_rows')
    if repair_max_rows is None:
        repair_max_rows = merged_defaults.get('max_rows')
    repair_report_dir = repair_cfg.get('report_dir', 'slurm/logs/repair')
    repair_report_dir = str(project_root / repair_report_dir)
    repair_max_rows_str = "" if repair_max_rows is None else str(repair_max_rows)

    # Template substitutions
    substitutions = {
        'JOB_NAME': local.job_name(f"emb_{model_name}_{dataset_name}"),
        'MODEL': model_name,
        'DATASET': dataset_name,
        'TIME_LIMIT': base_resources['time_limit'],
        'MEMORY': format_memory(final_memory_mb),
        'GPUS': str(base_resources.get('gpus', 1)),
        'CPUS': str(base_resources.get('cpus', 4)),
        'PARTITION': base_resources.get('partition', 'gpu'),
        'EXTRA_SLURM_DIRECTIVES': '\n'.join(extra_directives) if extra_directives else '',
        'PROJECT_ROOT': str(project_root),
        'CHECKPOINT': checkpoint,
        'INPUT_DIR': str(input_dir),
        'OUTPUT_PATH': str(output_path),
        'STATUS_FILE': str(status_file),
        'TIMING_FILE': str(timing_file),
        'LOG_DIR': str(logs_dir),
        'TIMESTAMP': datetime.now().isoformat(),
        'ENV_SETUP': local.env_setup(model_config.get('env_setup', 'source load_env')),
        'SCRIPT_PATH': model_config['script'],
        'INPUT_ARG': args.get('input', '--input'),
        'OUTPUT_ARG': args.get('output', '--output'),
        'CHECKPOINT_ARG': args.get('checkpoint', '--checkpoint'),
        'EXTRA_ARGS': ' \\\n    '.join(extra_args_list) if extra_args_list else '',
        # Hyperparameters for traceability (recorded in job_status.json)
        'MAX_ROWS': str(merged_defaults.get('max_rows', 1000)),
        'MAX_ENTITIES': (
            'None' if merged_defaults.get('max_entities') is None else str(merged_defaults.get('max_entities'))
        ),
        'BATCH_SIZE': str(merged_defaults.get('batch_size', 32)),
        'CHECKPOINT_INTERVAL': str(merged_defaults.get('checkpoint_interval', 100)),
        # Repair settings
        'REPAIR_ENABLED': str(repair_enabled).lower(),
        'REPAIR_CHUNK_SIZE': str(repair_chunk_size),
        'REPAIR_MAX_ROWS': repair_max_rows_str,
        'REPAIR_REPORT_DIR': repair_report_dir,
        # Pretraining settings (for models like Starmie with per-dataset training)
        'PRETRAIN_ENABLED': str(pretrain_enabled).lower(),
        'PRETRAIN_SCRIPT': pretrain_script,
        'PRETRAIN_ARGS': pretrain_args,
        # Checkpoint type (huggingface or local path)
        'CHECKPOINT_TYPE': checkpoint_type or '',
        # Shard defaults (overridden by caller for sharded jobs)
        'SHARD_SUFFIX': '',
        'SHARD_TYPE': '',
        'SHARD_INDEX': '-1',
        'NUM_SHARDS': '1',
    }

    # Apply shard overrides if provided
    if shard_overrides:
        substitutions.update(shard_overrides)

    # Use simple string replacement instead of Template to avoid issues with shell variables
    result = template_content
    for key, value in substitutions.items():
        result = result.replace(f'${{{key}}}', str(value))

    return result


def cleanup_scripts_for(model: str, dataset: str, scripts_dir: Path):
    """Delete all generated scripts for this exact (model, dataset) pair."""
    # Known fixed names
    for suffix in ['', '_merge', '_pretrain']:
        path = scripts_dir / f"{model}_{dataset}{suffix}.sbatch"
        if path.exists():
            path.unlink()
    # Shard scripts: use anchored regex to avoid prefix collisions
    shard_re = re.compile(rf'^{re.escape(model)}_{re.escape(dataset)}_shard\d+of\d+\.sbatch$')
    for f in scripts_dir.iterdir():
        if shard_re.match(f.name):
            f.unlink()


def _fill_template(template_content: str, substitutions: dict) -> str:
    """Apply substitutions to a template string."""
    result = template_content
    for key, value in substitutions.items():
        result = result.replace(f'${{{key}}}', str(value))
    return result


def _build_merge_substitutions(
    model_name: str,
    dataset_name: str,
    resources_config: dict,
    project_root: Path,
    shard_output_paths: list[str],
    final_output_path: str,
    num_shards: int,
) -> dict:
    """Build template substitutions for a merge sbatch script."""
    paths_config = resources_config.get('paths', {})
    logs_dir = project_root / paths_config.get('logs_dir', 'slurm/logs')
    status_file = project_root / paths_config.get('status_file', 'slurm/logs/status/job_status.json')

    extra_directives = []
    slurm_defaults = resources_config.get('slurm_defaults', {})
    if slurm_defaults.get('account'):
        extra_directives.append(f"#SBATCH --account={slurm_defaults['account']}")
    if slurm_defaults.get('mail_user'):
        extra_directives.append(f"#SBATCH --mail-type={slurm_defaults.get('mail_type', 'FAIL')}")
        extra_directives.append(f"#SBATCH --mail-user={slurm_defaults['mail_user']}")

    # CPU partition for merge jobs (no GPU needed)
    cpu_partition = resources_config.get('cpu_partition', 'cpu')

    return {
        'MODEL': model_name,
        'DATASET': dataset_name,
        'MERGE_JOB_NAME': local.job_name(f'merge_{model_name}_{dataset_name}'),
        'MERGE_JOB_KEY': f'{model_name}_{dataset_name}_merge',
        'MERGE_LOG_PREFIX': '',
        'LOG_DIR': str(logs_dir),
        'CPU_PARTITION': cpu_partition,
        'EXTRA_SLURM_DIRECTIVES': '\n'.join(extra_directives) if extra_directives else '',
        'PROJECT_ROOT': str(project_root),
        'STATUS_FILE': str(status_file),
        'OUTPUT_PATH': final_output_path,
        'SHARD_PATHS': ' '.join(f'"{p}"' for p in shard_output_paths),
        'NUM_SHARDS': str(num_shards),
        'TIMESTAMP': datetime.now().isoformat(),
        'ENV_SETUP': local.env_setup('source load_env'),
        'EMBEDDING_TYPE': 'column',
    }


def _build_pretrain_substitutions(
    model_name: str,
    model_config: dict,
    dataset_name: str,
    dataset_config: dict,
    resources_config: dict,
    project_root: Path,
    num_shards: int,
) -> dict:
    """Build template substitutions for a pretrain-only sbatch script."""
    paths_config = resources_config.get('paths', {})
    datasets_dir = project_root / paths_config.get('datasets_dir', 'data')
    logs_dir = project_root / paths_config.get('logs_dir', 'slurm/logs')
    status_file = project_root / paths_config.get('status_file', 'slurm/logs/status/job_status.json')

    input_dir = resolve_tables_dir(dataset_name, dataset_config, datasets_dir)

    # Resolve checkpoint
    pretrain_config = model_config.get('pretrain', {})
    checkpoint_filename = pretrain_config.get('checkpoint_filename', 'model.pt')
    tables_subdir = input_dir.name
    checkpoint = str(project_root / 'checkpoints' / model_name / dataset_name / tables_subdir / checkpoint_filename)

    # Build pretrain args
    pretrain_args_list = []
    pretrain_defaults = pretrain_config.get('defaults', {})
    pretrain_arg_mapping = pretrain_config.get('args', {})
    pretrain_flag_mapping = pretrain_config.get('flag_args', {})
    for param, value in pretrain_defaults.items():
        if param in pretrain_arg_mapping:
            pretrain_args_list.append(f'{pretrain_arg_mapping[param]} {value}')
        elif param in pretrain_flag_mapping:
            if value:
                pretrain_args_list.append(pretrain_flag_mapping[param])
    pretrain_args = ' '.join(pretrain_args_list)

    # Resources
    size_category = dataset_config.get('size_category', 'MEDIUM')
    base_resources = dict(resources_config['size_categories'][size_category])

    extra_directives = []
    slurm_defaults = resources_config.get('slurm_defaults', {})
    if slurm_defaults.get('account'):
        extra_directives.append(f"#SBATCH --account={slurm_defaults['account']}")
    if slurm_defaults.get('mail_user'):
        extra_directives.append(f"#SBATCH --mail-type={slurm_defaults.get('mail_type', 'FAIL')}")
        extra_directives.append(f"#SBATCH --mail-user={slurm_defaults['mail_user']}")

    memory_multipliers = resources_config.get('model_memory_multipliers', {})
    model_multiplier = memory_multipliers.get(model_name, 1.0)
    base_memory_mb = parse_memory(base_resources['memory'])
    final_memory_mb = int(base_memory_mb * model_multiplier)

    return {
        'MODEL': model_name,
        'DATASET': dataset_name,
        'TIME_LIMIT': base_resources['time_limit'],
        'MEMORY': format_memory(final_memory_mb),
        'GPUS': str(base_resources.get('gpus', 1)),
        'CPUS': str(base_resources.get('cpus', 4)),
        'PARTITION': base_resources.get('partition', 'gpu'),
        'EXTRA_SLURM_DIRECTIVES': '\n'.join(extra_directives) if extra_directives else '',
        'PROJECT_ROOT': str(project_root),
        'CHECKPOINT': checkpoint,
        'INPUT_DIR': str(input_dir),
        'STATUS_FILE': str(status_file),
        'LOG_DIR': str(logs_dir),
        'PRETRAIN_SCRIPT': pretrain_config.get('script', ''),
        'PRETRAIN_ARGS': pretrain_args,
        'NUM_SHARDS': str(num_shards),
        'TIMESTAMP': datetime.now().isoformat(),
        'ENV_SETUP': local.env_setup(model_config.get('env_setup', 'source load_env')),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Generate SLURM sbatch scripts for embedding generation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--models', nargs='+', help='Generate scripts only for these models')
    parser.add_argument('--datasets', nargs='+', help='Generate scripts only for these datasets')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be generated without writing')
    parser.add_argument('--output-dir', help='Output directory for column/text embedding scripts (table scripts always use configured table_scripts_dir)')
    parser.add_argument('--no-text-jobs', action='store_true', help='Skip text embedding job scripts')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Get paths
    project_root = get_project_root()
    config_dir = project_root / 'slurm' / 'config'
    template_path = project_root / 'slurm' / 'scripts' / 'templates' / 'embedding_job.sbatch.template'

    # Load configurations
    print("Loading configurations...")
    models_config = local.apply(load_yaml(config_dir / 'models.yaml'))
    datasets_config = load_yaml(config_dir / 'datasets.yaml')
    resources_config = local.apply(load_yaml(config_dir / 'resources.yaml'))

    # Get output directory from config or args
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / resources_config['paths']['embeddings_scripts_dir']
    repair_defaults = models_config.get('repair_defaults', {})

    # Load templates
    with open(template_path, 'r') as f:
        template_content = f.read()

    merge_template_path = project_root / 'slurm' / 'scripts' / 'templates' / 'merge_shards.sbatch.template'
    merge_template = merge_template_path.read_text() if merge_template_path.exists() else None

    pretrain_template_path = project_root / 'slurm' / 'scripts' / 'templates' / 'pretrain_only.sbatch.template'
    pretrain_template = pretrain_template_path.read_text() if pretrain_template_path.exists() else None

    # Filter models and datasets
    models = args.models or list(models_config['models'].keys())
    datasets = args.datasets or list(datasets_config['datasets'].keys())

    # Validate selections — allow models known to the table embedding generator too
    table_models_path = config_dir / 'table_models.yaml'
    table_model_names = set()
    if table_models_path.exists():
        table_models_cfg = load_yaml(table_models_path)
        table_model_names = set((table_models_cfg.get('models') or {}).keys())

    known_models = set(models_config['models'].keys()) | table_model_names
    invalid_models = set(models) - known_models
    if invalid_models:
        print(f"Error: Unknown models: {invalid_models}")
        print(f"Available column models: {list(models_config['models'].keys())}")
        if table_model_names:
            print(f"Available table models: {sorted(table_model_names)}")
        sys.exit(1)

    invalid_datasets = set(datasets) - set(datasets_config['datasets'].keys())
    if invalid_datasets:
        print(f"Error: Unknown datasets: {invalid_datasets}")
        print(f"Available datasets: {list(datasets_config['datasets'].keys())}")
        sys.exit(1)

    # Create output directory
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Generate scripts
    print(f"\nGenerating scripts for {len(models)} models x {len(datasets)} datasets = {len(models) * len(datasets)} jobs")
    print("=" * 60)

    generated = 0
    skipped = 0

    # Import table list generator for sharded datasets
    tools_dir = Path(__file__).resolve().parent
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    from generate_table_lists import generate_table_lists

    # Track which datasets have had table lists generated (per-dataset, shared across models)
    table_lists_generated: set[str] = set()

    paths_config = resources_config.get('paths', {})
    output_base_dir = project_root / paths_config.get('output_dir', 'embeddings/column')
    datasets_dir = project_root / paths_config.get('datasets_dir', 'data')

    for model_name in sorted(models):
        # Skip models not in column config (e.g. table-only models like tuta)
        if model_name not in models_config['models']:
            continue
        model_config = models_config['models'][model_name]

        for dataset_name in sorted(datasets):
            dataset_config = datasets_config['datasets'][dataset_name]

            # Skip if dataset is restricted to specific models
            allowed_models = dataset_config.get('models')
            if allowed_models and model_name not in allowed_models:
                continue

            # Check if tables source exists
            tables_source = dataset_config.get('tables_source')
            if tables_source and not (datasets_dir / tables_source).exists():
                print(f"  SKIP {model_name}/{dataset_name}: tables_source '{tables_source}' not found")
                skipped += 1
                continue

            # Resolve tables directory (re-use datasets_dir resolved above from paths_config)
            try:
                input_dir = resolve_tables_dir(dataset_name, dataset_config, datasets_dir)
                if not input_dir.exists():
                    print(f"  SKIP {model_name}/{dataset_name}: tables dir not found: {input_dir}")
                    skipped += 1
                    continue
                csv_count = len(list_csv_files(input_dir))
                if csv_count == 0:
                    print(f"  SKIP {model_name}/{dataset_name}: no CSV files in {input_dir}")
                    skipped += 1
                    continue
            except Exception as e:
                print(f"  SKIP {model_name}/{dataset_name}: {e}")
                skipped += 1
                continue

            num_shards = dataset_config.get('shards', 1)
            # Allow per-model shard overrides
            model_shard_overrides = model_config.get('shard_overrides', {})
            if dataset_name in model_shard_overrides:
                num_shards = model_shard_overrides[dataset_name]

            # Clean up stale scripts for this (model, dataset) pair
            if not args.dry_run:
                cleanup_scripts_for(model_name, dataset_name, output_dir)

            if num_shards > 1:
                # =============================================================
                # SHARDED PATH
                # =============================================================

                # Generate table list files (once per dataset+shard_count combo)
                table_list_key = f"{dataset_name}_{num_shards}"
                if table_list_key not in table_lists_generated:
                    if not args.dry_run:
                        table_list_paths = generate_table_lists(
                            dataset_name, str(input_dir), num_shards
                        )
                        if args.verbose:
                            for p in table_list_paths:
                                print(f"  Table list: {p.name}")
                    table_lists_generated.add(table_list_key)

                table_lists_dir = project_root / 'slurm' / 'scripts' / 'generated' / 'table_lists'

                # Check if Starmie needs separate pretrain job
                needs_pretrain = (
                    model_name == 'starmie'
                    and model_config.get('checkpoint') == 'auto'
                    and model_config.get('pretrain', {})
                )

                if needs_pretrain:
                    pretrain_script_name = f"{model_name}_{dataset_name}_pretrain.sbatch"
                    if args.dry_run:
                        print(f"  [DRY-RUN] Would generate: {pretrain_script_name}")
                    else:
                        # Generate pretrain-only script
                        pretrain_subs = _build_pretrain_substitutions(
                            model_name, model_config, dataset_name, dataset_config,
                            resources_config, project_root, num_shards
                        )
                        pretrain_script_path = output_dir / pretrain_script_name
                        pretrain_content = _fill_template(pretrain_template, pretrain_subs)
                        pretrain_script_path.write_text(pretrain_content)
                        os.chmod(pretrain_script_path, 0o755)
                        print(f"  {pretrain_script_name} (pretrain)")

                # Generate N shard scripts
                shard_output_paths = []
                for i in range(num_shards):
                    shard_suffix = f"_shard{i}of{num_shards}"
                    table_list_file = table_lists_dir / f"{dataset_name}_shard{i}of{num_shards}.txt"
                    shard_output = output_base_dir / model_name / f"{dataset_name}{shard_suffix}.pkl"
                    shard_output_paths.append(str(shard_output))

                    script_name = f"{model_name}_{dataset_name}_shard{i}of{num_shards}.sbatch"
                    script_path = output_dir / script_name

                    if args.dry_run:
                        print(f"  [DRY-RUN] Would generate: {script_name}")
                    else:
                        shard_ov = {
                            'JOB_NAME': local.job_name(f"emb_{model_name}_{dataset_name}_s{i}"),
                            'SHARD_SUFFIX': shard_suffix,
                            'SHARD_TYPE': 'shard',
                            'SHARD_INDEX': str(i),
                            'NUM_SHARDS': str(num_shards),
                            'OUTPUT_PATH': str(shard_output),
                        }

                        # Disable pretraining in shard scripts when pretrain is separate
                        if needs_pretrain:
                            shard_ov['PRETRAIN_ENABLED'] = 'false'

                        script_content = generate_script(
                            model_name, model_config,
                            dataset_name, dataset_config,
                            resources_config,
                            project_root, template_content,
                            repair_defaults,
                            shard_overrides=shard_ov,
                            additional_args=[f'--table_list {table_list_file}'],
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
                merge_script_name = f"{model_name}_{dataset_name}_merge.sbatch"
                merge_script_path = output_dir / merge_script_name
                final_output = output_base_dir / model_name / f"{dataset_name}.pkl"

                if args.dry_run:
                    print(f"  [DRY-RUN] Would generate: {merge_script_name}")
                else:
                    merge_subs = _build_merge_substitutions(
                        model_name, dataset_name, resources_config, project_root,
                        shard_output_paths, str(final_output), num_shards
                    )
                    merge_content = _fill_template(merge_template, merge_subs)
                    merge_script_path.write_text(merge_content)
                    os.chmod(merge_script_path, 0o755)
                    print(f"  {merge_script_name} (merge)")
                generated += 1

            else:
                # =============================================================
                # NON-SHARDED PATH (existing behavior)
                # =============================================================
                script_name = f"{model_name}_{dataset_name}.sbatch"
                script_path = output_dir / script_name

                if args.dry_run:
                    print(f"  [DRY-RUN] Would generate: {script_name}")
                    print(f"            Input: {input_dir} ({csv_count} CSVs)")
                else:
                    script_content = generate_script(
                        model_name, model_config,
                        dataset_name, dataset_config,
                        resources_config,
                        project_root, template_content,
                        repair_defaults
                    )

                    with open(script_path, 'w') as f:
                        f.write(script_content)

                    os.chmod(script_path, 0o755)

                    if args.verbose:
                        print(f"  Generated: {script_name} ({csv_count} CSVs)")
                    else:
                        print(f"  {script_name}")

                generated += 1

    # -------------------------------------------------------------------------
    # Text embedding jobs
    # -------------------------------------------------------------------------
    text_jobs = models_config.get('text_embedding_jobs', {}) or {}
    text_generated = 0

    # Generate text embedding scripts by default (skip with --no-text-jobs)
    if text_jobs and not args.no_text_jobs:
        text_template_path = project_root / 'slurm' / 'scripts' / 'templates' / 'text_embedding_job.sbatch.template'
        if text_template_path.exists():
            with open(text_template_path, 'r') as f:
                text_template_content = f.read()

            print(f"\nGenerating text embedding scripts...")
            print("-" * 60)

            for job_name, job_config in sorted(text_jobs.items()):

                size_category = job_config.get('size_category', 'SMALL')
                base_resources = dict(resources_config['size_categories'][size_category])

                # Build SLURM directives
                extra_directives = []
                slurm_defaults = resources_config.get('slurm_defaults', {})
                if slurm_defaults.get('account'):
                    extra_directives.append(f"#SBATCH --account={slurm_defaults['account']}")
                if slurm_defaults.get('mail_user'):
                    extra_directives.append(f"#SBATCH --mail-type={slurm_defaults.get('mail_type', 'FAIL')}")
                    extra_directives.append(f"#SBATCH --mail-user={slurm_defaults['mail_user']}")

                paths_config = resources_config.get('paths', {})
                logs_dir = project_root / paths_config.get('logs_dir', 'slurm/logs')
                status_file = project_root / paths_config.get('status_file', 'slurm/logs/status/job_status.json')
                timing_file = project_root / paths_config.get('timing_file', 'slurm/logs/timing/timing_summary.csv')

                substitutions = {
                    'JOB_NAME': local.job_name(f"text_{job_name}"),
                    'TIME_LIMIT': base_resources['time_limit'],
                    'MEMORY': base_resources['memory'],
                    'GPUS': str(base_resources.get('gpus', 1)),
                    'CPUS': str(base_resources.get('cpus', 4)),
                    'PARTITION': base_resources.get('partition', 'gpu'),
                    'EXTRA_SLURM_DIRECTIVES': '\n'.join(extra_directives) if extra_directives else '',
                    'PROJECT_ROOT': str(project_root),
                    'SCRIPT_PATH': job_config['script'],
                    'MODE': job_config['mode'],
                    'INPUT_JSON': str(project_root / job_config['input_json']),
                    'TEXT_FIELD': job_config.get('text_field', ''),
                    'TOKENS_FIELD': job_config.get('tokens_field', ''),
                    'MODEL': job_config['model'],
                    'BATCH_SIZE': str(job_config.get('batch_size', 32)),
                    'MAX_LENGTH': str(job_config.get('max_length', 512)),
                    'OUTPUT_PATH': str(project_root / job_config['output']),
                    'STATUS_FILE': str(status_file),
                    'TIMING_FILE': str(timing_file),
                    'LOG_DIR': str(logs_dir),
                    'TIMESTAMP': datetime.now().isoformat(),
                    'ENV_SETUP': local.env_setup(job_config.get('env_setup', 'source load_env')),
                    'ID_FIELD': job_config.get('id_field', ''),
                }

                script_name = f"text_{job_name}.sbatch"
                script_path = output_dir / script_name

                if args.dry_run:
                    print(f"  [DRY-RUN] Would generate: {script_name}")
                else:
                    result = text_template_content
                    for key, value in substitutions.items():
                        result = result.replace(f'${{{key}}}', str(value))

                    with open(script_path, 'w') as f:
                        f.write(result)
                    os.chmod(script_path, 0o755)

                    if args.verbose:
                        print(f"  Generated: {script_name}")
                    else:
                        print(f"  {script_name}")

                text_generated += 1
        else:
            print(f"\nWARN: Text embedding template not found: {text_template_path}")

    # -------------------------------------------------------------------------
    # Table embedding jobs — delegated to unified generator
    # -------------------------------------------------------------------------
    import subprocess

    print(f"\nDelegating table embedding script generation...")
    table_cmd = [
        sys.executable,
        str(project_root / 'slurm' / 'generate_table_embedding_scripts.py'),
    ]
    if args.models:
        table_cmd += ['--models'] + args.models
    if args.datasets:
        table_cmd += ['--datasets'] + args.datasets
    # Note: --output-dir is NOT forwarded. Column scripts and table scripts use
    # separate directories to avoid filename collisions ({model}_{dataset}.sbatch).
    # The table generator uses its own configured table_scripts_dir.
    if args.dry_run:
        table_cmd.append('--dry-run')
    if args.verbose:
        table_cmd.append('--verbose')

    table_result = subprocess.run(table_cmd)
    if table_result.returncode != 0:
        print(f"ERROR: Table embedding script generation failed (exit code {table_result.returncode})")
        sys.exit(table_result.returncode)

    print("=" * 60)
    print(f"Generated: {generated} column embedding scripts")
    if text_generated:
        print(f"Generated: {text_generated} text embedding scripts")
    print(f"Table embedding scripts: see delegated output above")
    if skipped:
        print(f"Skipped:   {skipped} combinations")
    if not args.dry_run:
        print(f"Output:    {output_dir}")


if __name__ == '__main__':
    main()
