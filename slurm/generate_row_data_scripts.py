#!/usr/bin/env python3
"""
Generate SLURM sbatch scripts for row data (canonical dataset) embedding generation.

Discovers datasets from datasets/row_data/openml_*/dataset.json and reads model
configs from row_data_models.yaml. Produces one sbatch script per (model, dataset)
pair in slurm/scripts/generated/row_data_embeddings/.

Usage:
    # Generate all scripts
    python generate_row_data_scripts.py

    # Generate for specific models
    python generate_row_data_scripts.py --models tabpfn scarf

    # Generate for specific datasets
    python generate_row_data_scripts.py --datasets openml_1486 openml_3

    # Dry run (show what would be generated)
    python generate_row_data_scripts.py --dry-run
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Reuse utility functions from the column embedding generator
import local_overrides as local

from generate_scripts import (
    get_project_root,
    load_yaml,
    parse_memory,
    format_memory,
)


def discover_datasets(datasets_root: Path, filter_names: list[str] | None = None) -> dict[str, Path]:
    """
    Discover canonical datasets from datasets/row_data/openml_*/dataset.json.

    Returns:
        dict mapping dataset name -> dataset directory path
    """
    row_data_dir = datasets_root / 'row_data'
    if not row_data_dir.exists():
        return {}

    datasets = {}
    for dataset_dir in sorted(row_data_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        manifest = dataset_dir / 'dataset.json'
        if not manifest.exists():
            continue
        name = dataset_dir.name
        if filter_names and name not in filter_names:
            continue
        datasets[name] = dataset_dir

    return datasets


def build_execution_block(
    model_name: str,
    model_config: dict,
    dataset_name: str,
    data_dir: str,
    embedding_dir: str,
    checkpoint_dir: str,
    project_root: Path,
) -> str:
    """Build the EXECUTION_BLOCK for the sbatch template.

    For pretrained models: single generate command with all defaults.
    For trained models: train command gets all defaults (hyperparams + label_policy),
    generate command gets only path args. Generation loads training artifacts
    (e.g., preprocessor/checkpoint metadata) from training_config.pkl, while
    runtime flags (e.g., batch_size) use generator defaults unless passed.
    """
    model_type = model_config.get('model_type', 'pretrained')
    args_mapping = model_config.get('args', {})
    defaults = dict(model_config.get('defaults', {}))
    model_defaults = dict(model_config.get('model_defaults', {}))

    # Apply per-dataset overrides (e.g. smaller batch_size for large datasets)
    ds_overrides = (model_config.get('dataset_overrides') or {}).get(dataset_name) or {}
    if ds_overrides:
        defaults.update(ds_overrides)
        model_defaults.update(ds_overrides)

    # Merge model_defaults into defaults (model_defaults override)
    all_defaults = {**defaults, **model_defaults}

    def _format_flag(param: str, value) -> str:
        if isinstance(value, list):
            return f'    --{param} {" ".join(str(v) for v in value)}'
        return f'    --{param} {value}'

    def build_path_args(script: str, include_embedding: bool) -> list[str]:
        """Build path-only argument list (data_dir, checkpoint_dir, embedding_dir, model_path)."""
        parts = [f'python {script}']

        if 'data_dir' in args_mapping:
            parts.append(f'    {args_mapping["data_dir"]} "{data_dir}"')

        if 'checkpoint_dir' in args_mapping and checkpoint_dir:
            parts.append(f'    {args_mapping["checkpoint_dir"]} "{checkpoint_dir}"')

        if include_embedding and 'embedding_dir' in args_mapping:
            parts.append(f'    {args_mapping["embedding_dir"]} "{embedding_dir}"')

        model_checkpoint = model_config.get('checkpoint')
        if model_checkpoint and 'model_path' in args_mapping:
            ckpt_path = str(project_root / model_checkpoint)
            parts.append(f'    {args_mapping["model_path"]} "{ckpt_path}"')

        return parts

    def build_train_command(script: str) -> str:
        """Build train command: paths + all defaults (hyperparams, label_policy, etc.)."""
        parts = build_path_args(script, include_embedding=False)

        skip_params = {'data_dir', 'checkpoint_dir', 'embedding_dir', 'model_path'}
        for param, value in all_defaults.items():
            if param in skip_params:
                continue
            if value is not None:
                parts.append(_format_flag(param, value))

        return ' \\\n'.join(parts)

    def build_generate_command(script: str) -> str:
        """Build generate command: paths + dataset_overrides (e.g. batch_size)."""
        parts = build_path_args(script, include_embedding=True)

        # Forward dataset_overrides to generate command (e.g. batch_size for OOM-prone datasets)
        if ds_overrides:
            skip_params = {'data_dir', 'checkpoint_dir', 'embedding_dir', 'model_path'}
            for param, value in ds_overrides.items():
                if param not in skip_params and value is not None:
                    parts.append(_format_flag(param, value))

        return ' \\\n'.join(parts)

    def build_pretrained_command(script: str) -> str:
        """Build pretrained command: paths + all defaults."""
        parts = build_path_args(script, include_embedding=True)

        skip_params = {'data_dir', 'checkpoint_dir', 'embedding_dir', 'model_path'}
        for param, value in all_defaults.items():
            if param in skip_params:
                continue
            if value is not None:
                parts.append(_format_flag(param, value))

        return ' \\\n'.join(parts)

    lines = []

    if model_type == 'pretrained':
        generate_script = model_config['generate_script']
        lines.append('log_info "Generating embeddings..."')
        lines.append(build_pretrained_command(generate_script))
    else:
        # Trained: Phase 1 (train) + Phase 2 (generate)
        # Train gets all hyperparams; generate only gets paths + generator runtime defaults
        train_script = model_config['train_script']
        generate_script = model_config['generate_script']

        lines.append('log_info "Phase 1: Training model..."')
        lines.append(build_train_command(train_script))
        lines.append('')
        lines.append('log_info "Phase 1 complete. Phase 2: Generating embeddings..."')
        lines.append(build_generate_command(generate_script))

    return '\n'.join(lines)


def generate_row_data_script(
    model_name: str,
    model_config: dict,
    dataset_name: str,
    dataset_dir: Path,
    resources: dict,
    project_root: Path,
    template_content: str,
) -> str:
    """Generate a single sbatch script for a row_data model-dataset combination."""

    # Resource allocation
    default_size = resources.get('default_size_category', 'SMALL')
    size_category = model_config.get('size_category', default_size)
    base_resources = dict(resources['size_categories'][size_category])

    final_memory_mb = parse_memory(base_resources['memory'])

    # Resolve paths
    paths_config = resources.get('paths', {})
    row_data_output_dir = project_root / paths_config.get('row_data_output_dir', 'embeddings/row_prediction')
    row_data_checkpoint_dir = project_root / paths_config.get('row_data_checkpoint_dir', 'checkpoints/row_data')
    logs_dir = project_root / paths_config.get('logs_dir', 'slurm/logs')
    status_file = project_root / paths_config.get('status_file', 'slurm/logs/status/job_status.json')
    timing_file = project_root / paths_config.get('timing_file', 'slurm/logs/timing/timing_summary.csv')

    # Build paths
    data_dir = str(dataset_dir)
    embedding_dir = str(row_data_output_dir / model_name / dataset_name)

    model_type = model_config.get('model_type', 'pretrained')
    if model_type == 'trained':
        checkpoint_dir = str(row_data_checkpoint_dir / model_name / dataset_name)
    else:
        checkpoint_dir = ""

    # Build execution block
    execution_block = build_execution_block(
        model_name, model_config, dataset_name,
        data_dir, embedding_dir, checkpoint_dir,
        project_root,
    )

    # Build SLURM directives
    extra_directives = []
    slurm_defaults = resources.get('slurm_defaults', {})
    if slurm_defaults.get('account'):
        extra_directives.append(f"#SBATCH --account={slurm_defaults['account']}")
    if slurm_defaults.get('mail_user'):
        extra_directives.append(f"#SBATCH --mail-type={slurm_defaults.get('mail_type', 'FAIL')}")
        extra_directives.append(f"#SBATCH --mail-user={slurm_defaults['mail_user']}")
    local.append_extra_slurm_directives(
        extra_directives,
        gpu_requested=bool(base_resources.get('gpus')),
    )

    env_setup = model_config.get('env_setup') or resources.get('default_env_setup', 'source load_env')

    # Template substitutions
    substitutions = {
        'JOB_NAME': local.job_name(f"rowdata_{model_name}_{dataset_name}"),
        'MODEL': model_name,
        'DATASET': dataset_name,
        'TIME_LIMIT': base_resources['time_limit'],
        'MEMORY': format_memory(final_memory_mb),
        'GPUS': str(base_resources.get('gpus', 1)),
        'CPUS': str(base_resources.get('cpus', 4)),
        'PARTITION': base_resources.get('partition', 'gpu'),
        'EXTRA_SLURM_DIRECTIVES': '\n'.join(extra_directives) if extra_directives else '',
        'PROJECT_ROOT': str(project_root),
        'DATA_DIR': data_dir,
        'EMBEDDING_DIR': embedding_dir,
        'CHECKPOINT_DIR': checkpoint_dir,
        'EXECUTION_BLOCK': execution_block,
        'STATUS_FILE': str(status_file),
        'TIMING_FILE': str(timing_file),
        'LOG_DIR': str(logs_dir),
        'TIMESTAMP': datetime.now().isoformat(),
        'ENV_SETUP': local.env_setup(env_setup),
    }

    result = template_content
    for key, value in substitutions.items():
        result = result.replace(f'${{{key}}}', str(value))

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Generate SLURM sbatch scripts for row data embedding generation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--models', nargs='+', help='Generate scripts only for these models')
    parser.add_argument('--datasets', nargs='+', help='Generate scripts only for these datasets')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be generated without writing')
    parser.add_argument('--output-dir', help='Output directory for generated scripts')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--dim', type=int, default=None,
                        help='Override embedding dimension (hidden_dim/emb_dim). '
                             'Auto-filters to SSL models only. Output dir becomes '
                             'row_prediction_dim{N}/')
    parser.add_argument('--embedding-output-dir', default=None,
                        help='Override the base embedding output directory')

    args = parser.parse_args()

    project_root = get_project_root()
    config_dir = project_root / 'slurm' / 'config'
    template_path = project_root / 'slurm' / 'scripts' / 'templates' / 'row_data_embedding_job.sbatch.template'

    # Load configurations
    print("Loading configurations...")
    row_data_config = local.apply(load_yaml(config_dir / 'row_data_models.yaml'))
    resources_config = local.apply(load_yaml(config_dir / 'resources.yaml'))

    # Merge row_data_models default_size_category into resources for generate_row_data_script
    resources_config['default_size_category'] = row_data_config.get('default_size_category', 'SMALL')

    # Discover datasets from filesystem
    datasets_dir = project_root / resources_config.get('paths', {}).get('datasets_dir', 'datasets')
    discovered = discover_datasets(datasets_dir, args.datasets)

    if not discovered:
        if args.datasets:
            print(f"Error: No matching datasets found for: {args.datasets}")
            print(f"Searched in: {datasets_dir / 'row_data'}")
        else:
            print(f"No datasets found in: {datasets_dir / 'row_data'}")
        sys.exit(1)

    # Get output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        row_data_scripts = resources_config['paths'].get(
            'row_data_scripts_dir', 'slurm/scripts/generated/row_data_embeddings'
        )
        output_dir = project_root / row_data_scripts

    # Load template
    with open(template_path, 'r') as f:
        template_content = f.read()

    # Filter models
    available_models = row_data_config.get('models', {}) or {}
    models = args.models or list(available_models.keys())

    # --dim mode: filter to SSL (trained) models only and override dimension
    if args.dim is not None:
        ssl_models = [m for m in models
                      if available_models.get(m, {}).get('model_type') == 'trained']
        if not ssl_models:
            print("Error: No SSL (trained) models found for --dim override")
            sys.exit(1)
        models = ssl_models
        print(f"--dim {args.dim}: filtered to {len(models)} SSL models")

        # Override embedding dimension in each model's config (deep copy to avoid
        # mutating the original). Uses hidden_dim for most models, emb_dim for saint.
        import copy
        for m in models:
            available_models[m] = copy.deepcopy(available_models[m])
            cfg = available_models[m]
            defaults = cfg.setdefault('defaults', {})
            model_defaults = cfg.setdefault('model_defaults', {})
            if 'hidden_dim' in defaults or 'hidden_dim' in model_defaults:
                defaults['hidden_dim'] = args.dim
            if 'emb_dim' in model_defaults:
                model_defaults['emb_dim'] = args.dim

        # Override embedding output dir
        dim_output = f'embeddings/row_prediction_dim{args.dim}'
        resources_config.setdefault('paths', {})['row_data_output_dir'] = dim_output
        # Also use separate checkpoints so different dims don't collide
        resources_config['paths']['row_data_checkpoint_dir'] = f'checkpoints/row_data_dim{args.dim}'
        print(f"  Output: {dim_output}/")

    # Allow --embedding-output-dir to override (takes precedence over --dim)
    if args.embedding_output_dir:
        resources_config.setdefault('paths', {})['row_data_output_dir'] = args.embedding_output_dir

    # Validate models
    invalid_models = set(models) - set(available_models.keys())
    if invalid_models:
        print(f"Error: Unknown models: {invalid_models}")
        print(f"Available: {list(available_models.keys())}")
        sys.exit(1)

    # Validate datasets (if explicit filter was given)
    if args.datasets:
        missing = set(args.datasets) - set(discovered.keys())
        if missing:
            print(f"Warning: Datasets not found on filesystem: {missing}")

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating row data embedding scripts for {len(models)} models x {len(discovered)} datasets = {len(models) * len(discovered)} jobs")
    print("=" * 60)

    generated = 0
    skipped = 0

    for model_name in sorted(models):
        model_config = available_models[model_name]

        for dataset_name, dataset_dir in sorted(discovered.items()):
            script_name = f"{model_name}_{dataset_name}.sbatch"
            script_path = output_dir / script_name

            if args.dry_run:
                print(f"  [DRY-RUN] Would generate: {script_name}")
                if args.verbose:
                    print(f"            Data dir: {dataset_dir}")
            else:
                script_content = generate_row_data_script(
                    model_name, model_config,
                    dataset_name, dataset_dir,
                    resources_config,
                    project_root, template_content,
                )

                with open(script_path, 'w') as f:
                    f.write(script_content)
                os.chmod(script_path, 0o755)

                if args.verbose:
                    print(f"  Generated: {script_name}")
                else:
                    print(f"  {script_name}")

            generated += 1

    print("=" * 60)
    print(f"Generated: {generated} row data embedding scripts")
    if skipped:
        print(f"Skipped:   {skipped} combinations")
    if not args.dry_run:
        print(f"Output:    {output_dir}")


if __name__ == '__main__':
    main()
