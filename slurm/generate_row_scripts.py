#!/usr/bin/env python3
"""
Generate SLURM sbatch scripts for row embedding generation.

Reads row_models.yaml x datasets.yaml and produces one sbatch script
per (model, dataset) pair in slurm/scripts/generated/row_embeddings/.

Usage:
    # Generate all scripts
    python generate_row_scripts.py

    # Generate for specific models
    python generate_row_scripts.py --models tabicl tabpfn

    # Generate for specific datasets
    python generate_row_scripts.py --datasets sato wtq

    # Dry run (show what would be generated)
    python generate_row_scripts.py --dry-run
"""

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import local_overrides as local

# Reuse utility functions from the column embedding generator
from generate_scripts import (
    get_project_root,
    load_yaml,
    list_csv_files,
    resolve_tables_dir,
    parse_memory,
    format_memory,
    _fill_template,
    _build_merge_substitutions,
)


def resolve_row_input_dir(dataset_name: str, dataset_config: dict, datasets_root: Path) -> Path:
    """Resolve input directory for row embedding scripts.

    If ``row_input_dir`` is provided in the dataset config, use it. Relative
    paths are interpreted as relative to ``datasets_root`` (same behavior as
    ``validate_embeddings.py``). Otherwise fall back to the standard
    ``resolve_tables_dir`` resolution used for column embeddings.
    """
    row_input = dataset_config.get("row_input_dir")
    if row_input:
        p = Path(row_input)
        return p if p.is_absolute() else datasets_root / row_input
    return resolve_tables_dir(dataset_name, dataset_config, datasets_root)


def override_row_model_dimensions(available_models: dict, models: list[str], dim: int) -> None:
    """Override the effective embedding dimension for trained row models in-place."""
    import copy

    for model_name in models:
        available_models[model_name] = copy.deepcopy(available_models[model_name])
        cfg = available_models[model_name]
        defaults = cfg.setdefault('defaults', {})
        model_defaults = cfg.setdefault('model_defaults', {})

        if 'hidden_dim' in defaults or 'hidden_dim' in model_defaults:
            defaults['hidden_dim'] = dim
        if 'emb_dim' in defaults or 'emb_dim' in model_defaults:
            defaults['emb_dim'] = dim


def derive_result_tag(
    output_dir: str | Path | None,
    default_root: str | Path,
    explicit_tag: str | None = None,
) -> str | None:
    """Derive a stable result tag from an alternate row output root."""
    if explicit_tag:
        return explicit_tag
    if not output_dir:
        return None

    output_name = Path(output_dir).name
    default_name = Path(default_root).name
    if output_name == default_name:
        return None
    if output_name.startswith(default_name + "_"):
        return output_name[len(default_name) + 1:]
    return output_name


def generate_row_script(
    model_name: str,
    model_config: dict,
    dataset_name: str,
    dataset_config: dict,
    resources: dict,
    project_root: Path,
    template_content: str,
    shard_overrides: dict | None = None,
    additional_args: list[str] | None = None,
    result_tag: str | None = None,
) -> str:
    """Generate a single sbatch script for a row model-dataset combination."""

    # Resource allocation
    size_category = dataset_config.get('size_category', 'MEDIUM')
    base_resources = dict(resources['size_categories'][size_category])

    # Apply model-specific resource overrides
    model_overrides = model_config.get('resource_overrides', {}) or {}
    dataset_override = model_overrides.get(dataset_name, {}) or {}
    for key in ['time_limit', 'gpus', 'cpus', 'partition', 'memory']:
        if key in dataset_override:
            base_resources[key] = dataset_override[key]

    # Memory (no model multipliers for row models by default)
    final_memory_mb = parse_memory(base_resources['memory'])

    # Resolve paths
    paths_config = resources.get('paths', {})
    datasets_dir = project_root / paths_config.get('datasets_dir', 'datasets')
    row_output_dir = project_root / paths_config.get('row_output_dir', 'embeddings/row')
    row_checkpoint_dir = project_root / paths_config.get('row_checkpoint_dir', 'checkpoints/row')
    logs_dir = project_root / paths_config.get('logs_dir', 'slurm/logs')
    status_file = project_root / paths_config.get('status_file', 'slurm/logs/status/job_status.json')
    timing_file = project_root / paths_config.get('timing_file', 'slurm/logs/timing/timing_summary.csv')

    # Resolve input directory
    input_dir = resolve_row_input_dir(dataset_name, dataset_config, datasets_dir)

    # Build output path
    output_path = row_output_dir / model_name / f"{dataset_name}.pkl"

    # Build checkpoint base dir (only for trained models)
    model_type = model_config.get('model_type', 'trained')
    if model_type == 'trained':
        checkpoint_base_dir = str(row_checkpoint_dir / model_name / dataset_name)
    else:
        checkpoint_base_dir = ""

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

    # Build extra arguments for the Python script
    defaults = dict(model_config.get('defaults', {}))
    args_mapping = model_config.get('args', {})
    extra_args_list = []

    # Apply dataset-level overrides (e.g. row_embedding_args.phase1_epochs)
    dataset_overrides = dataset_config.get('row_embedding_args', {}) or {}
    for param, value in dataset_overrides.items():
        if param not in args_mapping:
            print(f"  WARN: row_embedding_args key '{param}' has no matching CLI flag for model '{model_name}' (skipped)")
            continue
        if value is not None:
            defaults[param] = value

    # Add checkpoint_base_dir for trained models
    if model_type == 'trained' and 'checkpoint_base_dir' in args_mapping:
        extra_args_list.append(f"{args_mapping['checkpoint_base_dir']} \"{checkpoint_base_dir}\"")

    # Add model checkpoint path for pretrained models that have one (e.g. TUTA)
    model_checkpoint = model_config.get('checkpoint')
    if model_checkpoint and 'model_path' in args_mapping:
        ckpt_path = str(project_root / model_checkpoint)
        extra_args_list.append(f"{args_mapping['model_path']} \"{ckpt_path}\"")

    # Add default arguments (skip params handled above)
    skip_params = {'input_dir', 'output_path', 'checkpoint_base_dir', 'model_path'}
    for param, value in defaults.items():
        if param in skip_params:
            continue
        if param in args_mapping and value is not None:
            if isinstance(value, list):
                extra_args_list.append(f"{args_mapping[param]} {' '.join(str(v) for v in value)}")
            else:
                extra_args_list.append(f"{args_mapping[param]} {value}")

    # Append any additional args (e.g., --table_list for sharded runs)
    if additional_args:
        extra_args_list.extend(additional_args)

    # Template substitutions
    result_tag_suffix = f"_{result_tag}" if result_tag else ""
    substitutions = {
        'JOB_NAME': local.job_name(f"row_{model_name}_{dataset_name}{result_tag_suffix}"),
        'MODEL': model_name,
        'DATASET': dataset_name,
        'RESULT_TAG': result_tag or '',
        'RESULT_TAG_SUFFIX': result_tag_suffix,
        'TIME_LIMIT': base_resources['time_limit'],
        'MEMORY': format_memory(final_memory_mb),
        'GPUS': str(base_resources.get('gpus', 1)),
        'CPUS': str(base_resources.get('cpus', 4)),
        'PARTITION': base_resources.get('partition', 'gpu'),
        'EXTRA_SLURM_DIRECTIVES': '\n'.join(extra_directives) if extra_directives else '',
        'PROJECT_ROOT': str(project_root),
        'INPUT_DIR': str(input_dir),
        'OUTPUT_PATH': str(output_path),
        'CHECKPOINT_BASE_DIR': checkpoint_base_dir,
        'STATUS_FILE': str(status_file),
        'TIMING_FILE': str(timing_file),
        'LOG_DIR': str(logs_dir),
        'TIMESTAMP': datetime.now().isoformat(),
        'ENV_SETUP': local.env_setup(model_config.get('env_setup', 'source load_env')),
        'SCRIPT_PATH': model_config['script'],
        'EXTRA_ARGS': ' \\\n    '.join(extra_args_list) if extra_args_list else '',
        # Shard defaults (overridden by caller for sharded jobs)
        'SHARD_SUFFIX': '',
        'SHARD_TYPE': '',
        'SHARD_INDEX': '-1',
        'NUM_SHARDS': '1',
    }

    # Apply shard overrides if provided
    if shard_overrides:
        substitutions.update(shard_overrides)

    result = template_content
    for key, value in substitutions.items():
        result = result.replace(f'${{{key}}}', str(value))

    return result


def cleanup_row_scripts_for(model: str, dataset: str, scripts_dir: Path, result_tag: str | None = None):
    """Delete all generated row scripts for this exact (model, dataset) pair."""
    tag_suffix = f"_{result_tag}" if result_tag else ""
    for suffix in [tag_suffix, f'{tag_suffix}_merge']:
        path = scripts_dir / f"{model}_{dataset}{suffix}.sbatch"
        if path.exists():
            path.unlink()
    shard_re = re.compile(
        rf'^{re.escape(model)}_{re.escape(dataset)}{re.escape(tag_suffix)}_shard\d+of\d+\.sbatch$'
    )
    for f in scripts_dir.iterdir():
        if shard_re.match(f.name):
            f.unlink()


def main():
    parser = argparse.ArgumentParser(
        description='Generate SLURM sbatch scripts for row embedding generation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--models', nargs='+', help='Generate scripts only for these models')
    parser.add_argument('--datasets', nargs='+', help='Generate scripts only for these datasets')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be generated without writing')
    parser.add_argument('--output-dir', help='Output directory for generated scripts')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--dim', type=int, default=None,
                        help='Override embedding dimension (hidden_dim/emb_dim) for '
                             'trained row models only. Output dir becomes row_dim{N}/')
    parser.add_argument('--embedding-output-dir', default=None,
                        help='Override the base row embedding output directory')
    parser.add_argument('--result-tag', default=None,
                        help='Suffix used for script names and status keys when generating '
                             'alternate row embeddings (default: derived from output dir)')

    args = parser.parse_args()

    project_root = get_project_root()
    config_dir = project_root / 'slurm' / 'config'
    template_path = project_root / 'slurm' / 'scripts' / 'templates' / 'row_embedding_job.sbatch.template'

    # Load configurations
    print("Loading configurations...")
    row_models_config = local.apply(load_yaml(config_dir / 'row_models.yaml'))
    datasets_config = local.apply(load_yaml(config_dir / 'datasets.yaml'))
    resources_config = local.apply(load_yaml(config_dir / 'resources.yaml'))

    # Resolve datasets directory from config
    datasets_dir = project_root / resources_config.get('paths', {}).get('datasets_dir', 'datasets')

    # Get output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        row_scripts_dir = resources_config['paths'].get(
            'row_scripts_dir', 'slurm/scripts/generated/row_embeddings'
        )
        output_dir = project_root / row_scripts_dir

    # Load templates
    with open(template_path, 'r') as f:
        template_content = f.read()

    merge_template_path = project_root / 'slurm' / 'scripts' / 'templates' / 'merge_shards.sbatch.template'
    merge_template = merge_template_path.read_text() if merge_template_path.exists() else None

    # Filter models and datasets
    available_models = row_models_config.get('models', {}) or {}
    models = args.models or list(available_models.keys())

    if args.dim is not None:
        ssl_models = [m for m in models
                      if available_models.get(m, {}).get('model_type') == 'trained']
        if not ssl_models:
            print("Error: No trained row models found for --dim override")
            sys.exit(1)
        models = ssl_models
        print(f"--dim {args.dim}: filtered to {len(models)} trained row models")

        override_row_model_dimensions(available_models, models, args.dim)
        resources_config.setdefault('paths', {})['row_output_dir'] = f'embeddings/row_dim{args.dim}'
        resources_config['paths']['row_checkpoint_dir'] = f'checkpoints/row_dim{args.dim}'
        print(f"  Output: {resources_config['paths']['row_output_dir']}/")

    if args.embedding_output_dir:
        resources_config.setdefault('paths', {})['row_output_dir'] = args.embedding_output_dir

    result_tag = derive_result_tag(
        resources_config.get('paths', {}).get('row_output_dir'),
        'embeddings/row',
        args.result_tag,
    )

    # Only include datasets marked with row_embedding: true (unless --datasets is explicit)
    if args.datasets:
        datasets = args.datasets
    else:
        datasets = [
            name for name, cfg in datasets_config['datasets'].items()
            if cfg.get('row_embedding', False)
        ]

    # Validate
    invalid_models = set(models) - set(available_models.keys())
    if invalid_models:
        print(f"Error: Unknown row models: {invalid_models}")
        print(f"Available: {list(available_models.keys())}")
        sys.exit(1)

    invalid_datasets = set(datasets) - set(datasets_config['datasets'].keys())
    if invalid_datasets:
        print(f"Error: Unknown datasets: {invalid_datasets}")
        print(f"Available: {list(datasets_config['datasets'].keys())}")
        sys.exit(1)

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating row embedding scripts for {len(models)} models x {len(datasets)} datasets = {len(models) * len(datasets)} jobs")
    print("=" * 60)

    generated = 0
    skipped = 0

    # Import table list generator for sharded datasets
    tools_dir = Path(__file__).resolve().parent
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    from generate_table_lists import generate_table_lists

    # Track which datasets have had table lists generated (per-dataset, shared across models)
    paths_config = resources_config.get('paths', {})
    table_lists_generated: set[str] = set()
    table_lists_dir = project_root / 'slurm' / 'scripts' / 'generated' / 'table_lists'
    row_output_dir = project_root / paths_config.get('row_output_dir', 'embeddings/row')

    for model_name in sorted(models):
        model_config = available_models[model_name]

        # Skip models that require labels (unless label_column is in defaults)
        if model_config.get('requires_label_columns') and not model_config.get('defaults', {}).get('label_columns'):
            print(f"  SKIP {model_name}: requires --label_columns (not configured)")
            skipped += len(datasets)
            continue

        for dataset_name in sorted(datasets):
            dataset_config = datasets_config['datasets'][dataset_name]

            # Check tables source exists
            if not dataset_config.get("row_input_dir"):
                tables_source = dataset_config.get('tables_source')
                if tables_source and not (datasets_dir / tables_source).exists():
                    if args.verbose:
                        print(f"  SKIP {model_name}/{dataset_name}: tables_source '{tables_source}' not found")
                    skipped += 1
                    continue

            # Resolve tables directory
            try:
                input_dir = resolve_row_input_dir(dataset_name, dataset_config, datasets_dir)
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
            except Exception as e:
                if args.verbose:
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
                cleanup_row_scripts_for(model_name, dataset_name, output_dir, result_tag=result_tag)

            if num_shards > 1:
                # =============================================================
                # SHARDED PATH
                # =============================================================
                # NOTE: Table lists are keyed by dataset_name only and shared
                # between column and row pipelines. This assumes row_input_dir
                # resolves to the same CSV directory as column's tables_dir.
                # If that changes, table list naming would need an embedding-type prefix.

                # Generate table list files (once per dataset+shard_count combo)
                table_list_key = f"{dataset_name}_{num_shards}"
                if table_list_key not in table_lists_generated:
                    if not args.dry_run:
                        tl_paths = generate_table_lists(
                            dataset_name, str(input_dir), num_shards, table_lists_dir
                        )
                        if args.verbose:
                            for p in tl_paths:
                                print(f"  Table list: {p.name}")
                    table_lists_generated.add(table_list_key)

                # Generate N shard scripts
                shard_output_paths = []
                for i in range(num_shards):
                    shard_suffix = f"_shard{i}of{num_shards}"
                    table_list_file = table_lists_dir / f"{dataset_name}_shard{i}of{num_shards}.txt"
                    shard_output = row_output_dir / model_name / f"{dataset_name}{shard_suffix}.pkl"
                    shard_output_paths.append(str(shard_output))

                    tag_suffix = f"_{result_tag}" if result_tag else ""
                    script_name = f"{model_name}_{dataset_name}{tag_suffix}_shard{i}of{num_shards}.sbatch"
                    script_path = output_dir / script_name

                    if args.dry_run:
                        print(f"  [DRY-RUN] Would generate: {script_name}")
                    else:
                        shard_ov = {
                            'JOB_NAME': local.job_name(f"row_{model_name}_{dataset_name}{tag_suffix}_s{i}"),
                            'SHARD_SUFFIX': shard_suffix,
                            'SHARD_TYPE': 'shard',
                            'SHARD_INDEX': str(i),
                            'NUM_SHARDS': str(num_shards),
                            'OUTPUT_PATH': str(shard_output),
                        }

                        script_content = generate_row_script(
                            model_name, model_config,
                            dataset_name, dataset_config,
                            resources_config,
                            project_root, template_content,
                            shard_overrides=shard_ov,
                            additional_args=[f'--table_list "{table_list_file}"'],
                            result_tag=result_tag,
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
                tag_suffix = f"_{result_tag}" if result_tag else ""
                merge_script_name = f"{model_name}_{dataset_name}{tag_suffix}_merge.sbatch"
                merge_script_path = output_dir / merge_script_name
                final_output = row_output_dir / model_name / f"{dataset_name}.pkl"

                if args.dry_run:
                    print(f"  [DRY-RUN] Would generate: {merge_script_name}")
                else:
                    merge_subs = _build_merge_substitutions(
                        model_name, dataset_name, resources_config, project_root,
                        shard_output_paths, str(final_output), num_shards,
                    )
                    # Override merge keys for row pipeline
                    merge_subs['MERGE_JOB_NAME'] = local.job_name(f'merge_row_{model_name}_{dataset_name}{tag_suffix}')
                    merge_subs['MERGE_JOB_KEY'] = f'row_{model_name}_{dataset_name}{tag_suffix}_merge'
                    merge_subs['MERGE_LOG_PREFIX'] = 'row_'
                    merge_content = _fill_template(merge_template, merge_subs)
                    merge_script_path.write_text(merge_content)
                    os.chmod(merge_script_path, 0o755)
                    print(f"  {merge_script_name} (merge)")
                generated += 1

            else:
                # =============================================================
                # NON-SHARDED PATH (existing behavior)
                # =============================================================
                tag_suffix = f"_{result_tag}" if result_tag else ""
                script_name = f"{model_name}_{dataset_name}{tag_suffix}.sbatch"
                script_path = output_dir / script_name

                if args.dry_run:
                    print(f"  [DRY-RUN] Would generate: {script_name}")
                    if args.verbose:
                        print(f"            Input: {input_dir} ({csv_count} CSVs)")
                else:
                    script_content = generate_row_script(
                        model_name, model_config,
                        dataset_name, dataset_config,
                        resources_config,
                        project_root, template_content,
                        result_tag=result_tag,
                    )

                    with open(script_path, 'w') as f:
                        f.write(script_content)
                    os.chmod(script_path, 0o755)

                    if args.verbose:
                        print(f"  Generated: {script_name} ({csv_count} CSVs)")
                    else:
                        print(f"  {script_name}")

                generated += 1

    print("=" * 60)
    print(f"Generated: {generated} row embedding scripts")
    if skipped:
        print(f"Skipped:   {skipped} combinations")
    if not args.dry_run:
        print(f"Output:    {output_dir}")


if __name__ == '__main__':
    main()
