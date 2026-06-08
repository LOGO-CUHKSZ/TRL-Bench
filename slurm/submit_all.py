#!/usr/bin/env python3
"""
Submit embedding generation jobs to SLURM.

This script generates sbatch scripts (if needed) and submits them to SLURM.
It supports filtering by model/dataset, backing up existing embeddings,
and dry-run mode.

Usage:
    # Submit all jobs
    python submit_all.py

    # Submit specific models/datasets
    python submit_all.py --models starmie tapas --datasets wtq sato

    # Submit only small datasets first
    python submit_all.py --size-category SMALL

    # Dry run (generate scripts but don't submit)
    python submit_all.py --dry-run

    # Backup existing embeddings before submission
    python submit_all.py --backup

    # Submit row embedding jobs
    python submit_all.py --type row

    # Submit row data (canonical dataset) embedding jobs
    python submit_all.py --type row_data

    # Submit row_prediction downstream jobs
    python submit_all.py --type row_data_downstream
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


def get_project_root() -> Path:
    """Get the project root directory.

    File at slurm/submit_all.py; script_dir is slurm/, so .parent is the
    repo root. Using .parent.parent would land ABOVE the repo.
    """
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent


def load_yaml(path: Path) -> dict:
    """Load a YAML configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def backup_embeddings(
    project_root: Path,
    models: list,
    datasets: list,
    verbose: bool = False,
    embedding_type: str = 'column',
    embeddings_dir_override: Path | None = None,
):
    """
    Backup existing embeddings before regeneration.

    Args:
        project_root: Project root path
        models: List of model names
        datasets: List of dataset names
        verbose: Print detailed info
    """
    if embeddings_dir_override is not None:
        embeddings_dir = embeddings_dir_override
    elif embedding_type == 'row_data':
        embeddings_dir = project_root / 'embeddings' / 'row_prediction'
    elif embedding_type == 'row':
        embeddings_dir = project_root / 'embeddings' / 'row'
    else:
        embeddings_dir = project_root / 'embeddings' / 'column'
    backup_base = project_root / 'embeddings' / 'backups'
    backup_dir = backup_base / datetime.now().strftime('%Y-%m-%d_%H%M%S_pre_regeneration')

    items_to_backup = []
    for model in models:
        model_dir = embeddings_dir / model
        if model_dir.exists():
            for dataset in datasets:
                if embedding_type == 'row_data':
                    # row_data outputs are directories
                    dataset_dir = model_dir / dataset
                    if dataset_dir.is_dir():
                        items_to_backup.append(dataset_dir)
                else:
                    pkl_file = model_dir / f"{dataset}.pkl"
                    if pkl_file.exists():
                        items_to_backup.append(pkl_file)

    if not items_to_backup:
        print("No existing embeddings to backup.")
        return

    print(f"\nBacking up {len(items_to_backup)} existing embedding items...")
    backup_dir.mkdir(parents=True, exist_ok=True)

    for src in items_to_backup:
        rel_path = src.relative_to(embeddings_dir)
        dst = backup_dir / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)

        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        if verbose:
            print(f"  Backed up: {rel_path}")

    print(f"Backup complete: {backup_dir}")


def submit_job(script_path: Path, dry_run: bool = False, dependency: str | None = None) -> tuple[bool, str]:
    """
    Submit a single job to SLURM.

    Args:
        script_path: Path to sbatch script
        dry_run: If True, don't actually submit
        dependency: Optional SLURM dependency string (e.g. 'afterok:123:456')

    Returns:
        Tuple of (success, job_id or error message)
    """
    if dry_run:
        return True, "DRY-RUN"

    try:
        cmd = ['sbatch']
        if dependency:
            cmd.append(f'--dependency={dependency}')
        cmd.append(str(script_path))
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        # Parse job ID from "Submitted batch job 12345"
        output = result.stdout.strip()
        job_id = output.split()[-1] if 'Submitted' in output else output
        return True, job_id
    except subprocess.CalledProcessError as e:
        return False, e.stderr.strip()
    except FileNotFoundError:
        return False, "sbatch command not found (not on a SLURM cluster?)"


def _update_job_id(status_file: Path, status_key: str, slurm_job_id: str):
    """Persist SLURM job ID to status file immediately after submission."""
    import fcntl

    status_file.parent.mkdir(parents=True, exist_ok=True)
    lock_path = str(status_file) + '.lock'

    with open(lock_path, 'w') as lock_fd:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            if status_file.exists():
                with open(status_file, 'r') as f:
                    status = json.load(f)
            else:
                status = {}
            if status_key in status:
                status[status_key]['slurm_job_id'] = slurm_job_id
            with open(status_file, 'w') as f:
                json.dump(status, f, indent=2)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)


def _mark_status_keys(status_file: Path, keys: list[str], status_value: str, message: str):
    """Batch-update status for multiple keys in job_status.json."""
    import fcntl
    from datetime import datetime

    if not keys:
        return

    status_file.parent.mkdir(parents=True, exist_ok=True)
    lock_path = str(status_file) + '.lock'

    with open(lock_path, 'w') as lock_fd:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            if status_file.exists():
                with open(status_file, 'r') as f:
                    status = json.load(f)
            else:
                status = {}
            now = datetime.now().isoformat()
            for key in keys:
                if key in status:
                    status[key]['status'] = status_value
                    status[key]['message'] = message
                    status[key]['timestamp'] = now
            with open(status_file, 'w') as f:
                json.dump(status, f, indent=2)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)


def initialize_status_file(status_file: Path, jobs: list, embedding_type: str = 'column'):
    """
    Initialize or update the job status file.

    Args:
        status_file: Path to status JSON file
        jobs: List of (model, dataset) tuples or dicts with shard metadata
    """
    status_file.parent.mkdir(parents=True, exist_ok=True)

    # Load existing status
    if status_file.exists():
        with open(status_file, 'r') as f:
            status = json.load(f)
    else:
        status = {}

    # Add new jobs as PENDING
    timestamp = datetime.now().isoformat()
    for job in jobs:
        if isinstance(job, dict):
            # Shard-aware entry
            key = job['key']
            entry = {
                'status': 'PENDING',
                'message': 'Submitted',
                'timestamp': timestamp,
                'model': job['model'],
                'dataset': job['dataset'],
                'embedding_type': embedding_type,
                'shard_type': job.get('shard_type', ''),
                'shard_index': job.get('shard_index', -1),
                'num_shards': job.get('num_shards', 1),
                'result_tag': job.get('result_tag', ''),
            }
        else:
            model, dataset = job
            if embedding_type == 'row_data_downstream':
                key = f"ds_rowpred_{model}_{dataset}"
            elif embedding_type == 'row_data':
                key = f"rowdata_{model}_{dataset}"
            elif embedding_type == 'row':
                key = f"row_{model}_{dataset}"
            elif embedding_type == 'pca':
                # dataset is actually plan_name for PCA; match runtime JOB_KEY
                key = f"pca_{dataset}_{model}"
            else:
                key = f"{model}_{dataset}"
            entry = {
                'status': 'PENDING',
                'message': 'Submitted',
                'timestamp': timestamp,
                'model': model,
                'dataset': dataset,
                'embedding_type': embedding_type,
                'result_tag': '',
            }
        if key not in status or status[key].get('status') != 'RUNNING':
            status[key] = entry

    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Submit embedding generation jobs to SLURM',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--models', nargs='+', help='Submit jobs only for these models')
    parser.add_argument('--datasets', nargs='+', help='Submit jobs only for these datasets')
    parser.add_argument('--size-category', choices=['SMALL', 'MEDIUM', 'LARGE'],
                        help='Submit only datasets of this size category')
    parser.add_argument('--type', choices=['column', 'row', 'row_data', 'row_data_downstream', 'pca', 'pca_downstream'],
                        default='column',
                        help='Submit column, row, row_data, row_data_downstream, pca, or pca_downstream jobs (default: column)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Generate scripts and show what would be submitted')
    parser.add_argument('--backup', action='store_true',
                        help='Backup existing embeddings before submission')
    parser.add_argument('--no-generate', action='store_true',
                        help='Skip script generation (use existing scripts)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='Delay between submissions in seconds (default: 0.5)')
    parser.add_argument('--dim', type=int, default=None,
                        help='Override embedding dimension for trained row or row_data models. '
                             'Auto-filters to trained models. Output dir becomes '
                             'row_dim{N}/ or row_prediction_dim{N}/')
    parser.add_argument('--embedding-root', type=str, default=None,
                        help='Override embedding root for downstream discovery '
                             '(e.g., embeddings/row_prediction_pca128_from768)')
    parser.add_argument('--result-tag', type=str, default=None,
                        help='Tag for result namespacing in downstream evaluation '
                             '(e.g., "pca128_from768")')
    parser.add_argument('--plans', nargs='+', default=None,
                        help='PCA plans to run (default: all). Only used with --type pca')
    parser.add_argument('--force', action='store_true',
                        help='Force overwrite (PCA: overwrite existing outputs)')
    parser.add_argument('--head-type', nargs='+', default=['mlp'],
                        choices=['mlp', 'linear', 'dummy'],
                        help='Probe types for downstream evaluation (default: [mlp]). '
                             'Multiple values run separate passes, e.g. --head-type mlp linear')

    args = parser.parse_args()

    # Get paths
    project_root = get_project_root()
    config_dir = project_root / 'slurm' / 'config'
    resources_config = load_yaml(config_dir / 'resources.yaml')
    paths_config = resources_config['paths']
    status_file = project_root / paths_config['status_file']

    # ---- PCA compression pipeline ----
    if args.type == 'pca':
        _run_pca_pipeline(args, project_root, config_dir, resources_config, paths_config, status_file)
        return

    # ---- PCA downstream: run downstream on all PCA embedding directories ----
    if args.type == 'pca_downstream':
        _run_pca_downstream_pipeline(args, project_root, config_dir, resources_config, paths_config, status_file)
        return

    # ---- row_data and row_data_downstream use filesystem discovery ----
    if args.type in ('row_data', 'row_data_downstream'):
        _run_row_data_pipeline(args, project_root, config_dir, resources_config, paths_config, status_file)
        return

    # ---- column / row (original YAML-driven flow) ----
    print("Loading configurations...")
    table_model_names: set[str] = set()
    if args.type == 'row':
        models_config = load_yaml(config_dir / 'row_models.yaml')
    else:
        models_config = load_yaml(config_dir / 'models.yaml')
        # Merge table-direct models (mpnet, sentence_t5, tapex) so users can
        # submit them via the same --type column flow. generate_scripts.py
        # delegates to generate_table_embedding_scripts.py for these.
        table_models_path = config_dir / 'table_models.yaml'
        if table_models_path.exists():
            table_cfg = load_yaml(table_models_path)
            for name, cfg in (table_cfg.get('models') or {}).items():
                table_model_names.add(name)
                if name not in models_config.get('models', {}):
                    models_config.setdefault('models', {})[name] = cfg
    datasets_config = load_yaml(config_dir / 'datasets.yaml')

    if args.type == 'row':
        scripts_dir = project_root / paths_config['row_scripts_dir']
        table_scripts_dir = None
    else:
        scripts_dir = project_root / paths_config['embeddings_scripts_dir']
        # generate_table_embedding_scripts.py writes table-direct sbatch files
        # to a separate directory; submitter must search there too.
        table_scripts_dir = project_root / paths_config.get(
            'table_scripts_dir', 'slurm/scripts/generated/table_embeddings'
        )

    # Determine which models/datasets to process
    all_models = list(models_config['models'].keys())
    all_datasets = list(datasets_config['datasets'].keys())

    models = args.models or all_models
    datasets = args.datasets or all_datasets
    row_result_tag = f"dim{args.dim}" if args.type == 'row' and args.dim is not None else None

    if args.type == 'row' and args.dim is not None:
        ssl_models = [m for m, cfg in models_config.get('models', {}).items()
                      if cfg.get('model_type') == 'trained']
        if models:
            models = [m for m in models if m in ssl_models]
        else:
            models = ssl_models
        if not models:
            print("Error: --dim requires at least one trained row model")
            sys.exit(1)
        all_models = ssl_models
        print(f"--dim {args.dim}: filtered to {len(models)} trained row models")

    # Filter by size category if specified
    if args.size_category:
        datasets = [
            d for d in datasets
            if datasets_config['datasets'][d].get('size_category') == args.size_category
        ]
        print(f"Filtered to {len(datasets)} datasets with size category: {args.size_category}")

    # Validate selections
    invalid_models = set(models) - set(all_models)
    if invalid_models:
        print(f"Error: Unknown models: {invalid_models}")
        sys.exit(1)

    invalid_datasets = set(datasets) - set(all_datasets)
    if invalid_datasets:
        print(f"Error: Unknown datasets: {invalid_datasets}")
        sys.exit(1)

    total_jobs = len(models) * len(datasets)
    print(f"\nPreparing {len(models)} models x {len(datasets)} datasets = {total_jobs} jobs")

    # Backup existing embeddings if requested
    if args.backup:
        backup_root = None
        if args.type == 'row' and args.dim is not None:
            backup_root = project_root / 'embeddings' / f'row_dim{args.dim}'
        backup_embeddings(
            project_root,
            models,
            datasets,
            args.verbose,
            embedding_type=args.type,
            embeddings_dir_override=backup_root,
        )

    # Generate scripts if needed
    if not args.no_generate:
        print("\nGenerating sbatch scripts...")
        if args.type == 'row':
            generate_script = project_root / 'slurm' / 'generate_row_scripts.py'
        else:
            generate_script = project_root / 'slurm' / 'generate_scripts.py'

        cmd = [sys.executable, str(generate_script)]
        if models != all_models:
            cmd.extend(['--models'] + models)
        if datasets != all_datasets:
            cmd.extend(['--datasets'] + datasets)
        if args.type == 'row' and args.dim is not None:
            cmd.extend(['--dim', str(args.dim)])
            if row_result_tag:
                cmd.extend(['--result-tag', row_result_tag])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("Error generating scripts:")
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
            sys.exit(1)
        if args.verbose:
            print(result.stdout)

    # Collect and submit scripts (config-driven for sharding)
    _submit_column_scripts(args, models, datasets, datasets_config, models_config, scripts_dir, status_file,
                           embedding_type=args.type, result_tag=row_result_tag if args.type == 'row' else None,
                           table_scripts_dir=table_scripts_dir, table_model_names=table_model_names)


def _submit_column_scripts(args, models, datasets, datasets_config, models_config, scripts_dir, status_file,
                           embedding_type='column', result_tag: str | None = None,
                           table_scripts_dir=None, table_model_names: set[str] | None = None):
    """Submit embedding scripts, handling sharded and non-sharded datasets.

    Sharding is only supported for column embeddings. Row mode always takes the
    non-sharded path regardless of the datasets.yaml shards field.
    """
    import time

    # Status key prefix must match the template's JOB_KEY so that
    # the sbatch script's status updates and the submitter's updates
    # write to the same key.
    _key_prefixes = {
        'row': 'row_',
        'row_data': 'rowdata_',
        'row_data_downstream': 'ds_rowpred_',
    }
    key_prefix = _key_prefixes.get(embedding_type, '')
    tag_suffix = f"_{result_tag}" if result_tag else ""

    # Build jobs list and status entries
    status_entries = []
    submission_plan = []  # list of (model, dataset, plan_type, ...)
    _table_models = table_model_names or set()

    for model in sorted(models):
        model_config = models_config.get('models', {}).get(model, {})
        # Table-direct models have their sbatch scripts written by
        # generate_table_embedding_scripts.py to table_scripts_dir, not the
        # default column-embedding scripts_dir.
        if model in _table_models and table_scripts_dir is not None:
            model_scripts_dir = table_scripts_dir
        else:
            model_scripts_dir = scripts_dir
        for dataset in sorted(datasets):
            dataset_config = datasets_config['datasets'].get(dataset, {})
            # Skip if dataset is restricted to specific models
            allowed_models = dataset_config.get('models')
            if allowed_models and model not in allowed_models:
                continue
            # Sharding applies to column and row embeddings
            num_shards = dataset_config.get('shards', 1) if embedding_type in ('column', 'row') else 1
            # Allow per-model shard overrides
            if embedding_type in ('column', 'row'):
                model_shard_overrides = model_config.get('shard_overrides', {})
                if dataset in model_shard_overrides:
                    num_shards = model_shard_overrides[dataset]

            if num_shards > 1:
                # Sharded submission — validate the full expected script set
                pretrain_script = model_scripts_dir / f"{model}_{dataset}{tag_suffix}_pretrain.sbatch"
                merge_script = model_scripts_dir / f"{model}_{dataset}{tag_suffix}_merge.sbatch"

                # Determine if pretrain is required from model config
                # (same logic as generate_scripts.py)
                needs_pretrain = (
                    model_config.get('checkpoint') == 'auto'
                    and model_config.get('pretrain', {})
                )
                # Only treat pretrain as active if config requires it AND script exists.
                # A stale pretrain script from a previous config must not drive submission.
                has_pretrain = needs_pretrain and pretrain_script.exists()

                # Collect ALL expected shard scripts and check for missing ones
                shard_scripts = []
                missing_scripts = []

                if needs_pretrain and not has_pretrain:
                    missing_scripts.append(pretrain_script.name)

                for i in range(num_shards):
                    sp = model_scripts_dir / f"{model}_{dataset}{tag_suffix}_shard{i}of{num_shards}.sbatch"
                    if sp.exists():
                        shard_scripts.append((i, sp))
                    else:
                        missing_scripts.append(sp.name)

                # Require the merge script
                if not merge_script.exists():
                    missing_scripts.append(merge_script.name)

                if not shard_scripts:
                    if args.verbose:
                        print(f"  SKIP: {model}_{dataset} (no shard scripts found)")
                    continue

                if missing_scripts:
                    print(f"  SKIP: {model}_{dataset} (incomplete script set, missing: {', '.join(missing_scripts)})")
                    continue

                # Status entries for shards
                if has_pretrain:
                    status_entries.append({
                        'key': f"{key_prefix}{model}_{dataset}{tag_suffix}_pretrain",
                        'model': model, 'dataset': dataset,
                        'shard_type': 'pretrain', 'shard_index': -1,
                        'num_shards': num_shards,
                        'result_tag': result_tag or '',
                    })
                for i, _ in shard_scripts:
                    status_entries.append({
                        'key': f"{key_prefix}{model}_{dataset}{tag_suffix}_shard{i}of{num_shards}",
                        'model': model, 'dataset': dataset,
                        'shard_type': 'shard', 'shard_index': i,
                        'num_shards': num_shards,
                        'result_tag': result_tag or '',
                    })
                status_entries.append({
                    'key': f"{key_prefix}{model}_{dataset}{tag_suffix}_merge",
                    'model': model, 'dataset': dataset,
                    'shard_type': 'merge', 'shard_index': -1,
                    'num_shards': num_shards,
                    'result_tag': result_tag or '',
                })

                submission_plan.append(('sharded', model, dataset, num_shards,
                                        has_pretrain, pretrain_script,
                                        shard_scripts, merge_script))
            else:
                # Non-sharded
                script_path = model_scripts_dir / f"{model}_{dataset}{tag_suffix}.sbatch"
                if script_path.exists():
                    status_entries.append({
                        'key': f"{key_prefix}{model}_{dataset}{tag_suffix}",
                        'model': model,
                        'dataset': dataset,
                        'result_tag': result_tag or '',
                    })
                    submission_plan.append(('single', model, dataset, script_path, f"{key_prefix}{model}_{dataset}{tag_suffix}"))
                else:
                    if args.verbose:
                        print(f"  SKIP: {model}_{dataset} (script not found)")

    if not submission_plan:
        print("No scripts to submit.")
        sys.exit(0)

    # Initialize status file
    initialize_status_file(status_file, status_entries, embedding_type=embedding_type)

    # Submit jobs
    total_count = sum(
        1 if p[0] == 'single' else
        (1 if p[4] else 0) + len(p[6]) + 1  # shards + merge (validated upfront)
        for p in submission_plan
    )
    print(f"\nSubmitting {total_count} jobs...")
    print("=" * 60)

    submitted = 0
    failed = 0

    for plan in submission_plan:
        if plan[0] == 'single':
            _, model, dataset, script_path, status_key = plan
            success, result = submit_job(script_path, args.dry_run)
            display_label = f"{model}_{dataset}{tag_suffix}"
            if success:
                submitted += 1
                status_str = "[DRY-RUN]" if args.dry_run else f"Job {result}"
                print(f"  {display_label}: {status_str}")
                if not args.dry_run and result != "DRY-RUN":
                    _update_job_id(status_file, status_key, result)
            else:
                failed += 1
                print(f"  {display_label}: FAILED - {result}")
                if not args.dry_run:
                    _mark_status_keys(status_file, [status_key], 'FAILED',
                                      f'Submission failed: {result}')

            if not args.dry_run and args.delay > 0:
                time.sleep(args.delay)

        elif plan[0] == 'sharded':
            _, model, dataset, num_shards, has_pretrain, pretrain_script, shard_scripts, merge_script = plan

            pretrain_id = None
            shard_ids = []

            # 1. Submit pretrain (if applicable)
            chain_aborted = False
            if has_pretrain:
                success, result = submit_job(pretrain_script, args.dry_run)
                status_key = f"{key_prefix}{model}_{dataset}{tag_suffix}_pretrain"
                if success:
                    submitted += 1
                    pretrain_id = result
                    status_str = "[DRY-RUN]" if args.dry_run else f"Job {result}"
                    print(f"  {key_prefix}{model}_{dataset}{tag_suffix}_pretrain: {status_str}")
                    if not args.dry_run and result != "DRY-RUN":
                        _update_job_id(status_file, status_key, result)
                else:
                    failed += 1
                    chain_aborted = True
                    print(f"  {key_prefix}{model}_{dataset}{tag_suffix}_pretrain: FAILED - {result}")
                    print(f"  {key_prefix}{model}_{dataset}{tag_suffix}: SKIPPING shards + merge (pretrain failed)")
                    if not args.dry_run:
                        _mark_status_keys(status_file, [status_key], 'FAILED',
                                          f'Submission failed: {result}')
                if not args.dry_run and args.delay > 0:
                    time.sleep(args.delay)

            if chain_aborted:
                # Mark un-submitted shard + merge entries so they don't linger as PENDING
                if not args.dry_run:
                    skip_keys = [f"{key_prefix}{model}_{dataset}{tag_suffix}_shard{i}of{num_shards}" for i, _ in shard_scripts]
                    skip_keys.append(f"{key_prefix}{model}_{dataset}{tag_suffix}_merge")
                    _mark_status_keys(status_file, skip_keys, 'FAILED', 'Skipped: pretrain submission failed')
                continue

            # 2. Submit shard scripts (with pretrain dependency if applicable)
            shard_dep = f"afterok:{pretrain_id}" if pretrain_id and pretrain_id != "DRY-RUN" else None
            shard_failed = False
            for i, sp in shard_scripts:
                success, result = submit_job(sp, args.dry_run, dependency=shard_dep)
                status_key = f"{key_prefix}{model}_{dataset}{tag_suffix}_shard{i}of{num_shards}"
                if success:
                    submitted += 1
                    shard_ids.append(result)
                    status_str = "[DRY-RUN]" if args.dry_run else f"Job {result}"
                    print(f"  {key_prefix}{model}_{dataset}{tag_suffix}_shard{i}of{num_shards}: {status_str}")
                    if not args.dry_run and result != "DRY-RUN":
                        _update_job_id(status_file, status_key, result)
                else:
                    failed += 1
                    shard_failed = True
                    print(f"  {key_prefix}{model}_{dataset}{tag_suffix}_shard{i}of{num_shards}: FAILED - {result}")
                    if not args.dry_run:
                        _mark_status_keys(status_file, [status_key], 'FAILED',
                                          f'Submission failed: {result}')
                if not args.dry_run and args.delay > 0:
                    time.sleep(args.delay)

            # 3. Submit merge (with dependency on all shards)
            # Only submit if ALL shards succeeded — partial chains produce incomplete data
            if shard_failed:
                print(f"  {key_prefix}{model}_{dataset}{tag_suffix}_merge: SKIPPED (not all shards submitted)")
                if not args.dry_run:
                    _mark_status_keys(status_file, [f"{key_prefix}{model}_{dataset}{tag_suffix}_merge"], 'FAILED',
                                      'Skipped: not all shard submissions succeeded')
            elif shard_ids:
                real_ids = [sid for sid in shard_ids if sid != "DRY-RUN"]
                merge_dep = f"afterok:{':'.join(real_ids)}" if real_ids else None
                success, result = submit_job(merge_script, args.dry_run, dependency=merge_dep)
                status_key = f"{key_prefix}{model}_{dataset}{tag_suffix}_merge"
                if success:
                    submitted += 1
                    status_str = "[DRY-RUN]" if args.dry_run else f"Job {result}"
                    print(f"  {key_prefix}{model}_{dataset}{tag_suffix}_merge: {status_str}")
                    if not args.dry_run and result != "DRY-RUN":
                        _update_job_id(status_file, status_key, result)
                else:
                    failed += 1
                    print(f"  {key_prefix}{model}_{dataset}{tag_suffix}_merge: FAILED - {result}")
                    if not args.dry_run:
                        _mark_status_keys(status_file, [status_key], 'FAILED',
                                          f'Submission failed: {result}')

    print("=" * 60)
    print(f"Submitted: {submitted}")
    if failed:
        print(f"Failed:    {failed}")

    if not args.dry_run and submitted > 0:
        print(f"\nMonitor progress with:")
        print(f"  python slurm/monitor_jobs.py")


def _discover_row_data_datasets(project_root: Path, filter_names: list[str] | None = None) -> list[str]:
    """Discover canonical dataset names from data/row_data/openml_*/dataset.json."""
    row_data_dir = project_root / 'data' / 'row_data'
    if not row_data_dir.exists():
        return []
    datasets = []
    for d in sorted(row_data_dir.iterdir()):
        if d.is_dir() and (d / 'dataset.json').exists():
            if filter_names is None or d.name in filter_names:
                datasets.append(d.name)
    return datasets


def _run_pca_pipeline(args, project_root, config_dir, resources_config, paths_config, status_file):
    """Handle --type pca: generate and submit PCA compression jobs."""
    print("Loading PCA plan configurations...")
    generator_script = project_root / 'slurm' / 'generate_pca_scripts.py'
    scripts_dir = project_root / paths_config.get('pca_scripts_dir', 'slurm/scripts/generated/pca')

    # Generate scripts if needed
    if not args.no_generate:
        # Clear existing scripts to prevent stale submissions (skip during dry-run)
        if not args.dry_run and scripts_dir.exists():
            for old_script in scripts_dir.glob('*.sbatch'):
                old_script.unlink()

        print("\nGenerating PCA sbatch scripts...")
        cmd = [sys.executable, str(generator_script)]
        if args.models:
            cmd.extend(['--models'] + args.models)
        if args.plans:
            cmd.extend(['--plans'] + args.plans)
        if args.force:
            cmd.append('--force')

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("Error generating PCA scripts:")
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
            sys.exit(1)
        if args.verbose:
            print(result.stdout)

    # Collect and submit scripts
    if not scripts_dir.exists():
        print(f"No scripts directory: {scripts_dir}")
        return

    scripts = sorted(scripts_dir.glob('*.sbatch'))
    if not scripts:
        print("No PCA scripts to submit.")
        return

    print(f"\nFound {len(scripts)} PCA scripts to submit")

    if args.dry_run:
        for s in scripts:
            print(f"  [DRY-RUN] Would submit: {s.name}")
        return

    # Build (model, dataset, script_path) tuples for _submit_scripts
    # For PCA, "dataset" is the plan name (extracted from filename: {plan}--{model}.sbatch)
    scripts_to_submit = []
    for s in scripts:
        # Parse {plan}--{model}.sbatch using '--' delimiter (avoids underscore ambiguity)
        stem = s.stem  # e.g., "ssl_from128--dae"
        parts = stem.split('--', 1)
        if len(parts) == 2:
            plan_name, model = parts
        else:
            plan_name, model = stem, 'unknown'
        scripts_to_submit.append((model, plan_name, s))

    _submit_scripts(args, scripts_to_submit, status_file, embedding_type='pca')


def _run_pca_downstream_pipeline(args, project_root, config_dir, resources_config, paths_config, status_file):
    """Handle --type pca_downstream: run downstream on all PCA embedding directories.

    Reads pca_plans.yaml to derive all (source_dim, target_dim) combinations,
    then generates and submits downstream evaluation scripts for each PCA
    output directory.
    """
    pca_config = load_yaml(project_root / 'slurm' / 'config' / 'pca_plans.yaml')
    generator_script = project_root / 'slurm' / 'generate_row_data_downstream_scripts.py'
    output_base = project_root / pca_config.get('output_base', 'embeddings')
    output_pattern = pca_config.get('output_pattern', 'row_prediction_pca{target}_from{source}')

    plans = pca_config.get('plans', {})
    if args.plans:
        plans = {k: v for k, v in plans.items() if k in args.plans}

    # Derive all unique (embedding_root, result_tag) pairs from plans
    pca_targets = []
    seen = set()
    for plan_name, plan in plans.items():
        for td in plan['target_dims']:
            dir_name = output_pattern.format(target=td, source=plan['source_dim'])
            tag = f"pca{td}_from{plan['source_dim']}"
            if tag not in seen:
                seen.add(tag)
                pca_targets.append((dir_name, tag))

    pca_targets.sort()

    head_types = getattr(args, 'head_type', ['mlp']) or ['mlp']

    print(f"PCA downstream evaluation: {len(pca_targets)} embedding directories")
    print(f"Head types: {head_types}")
    print("=" * 60)

    total_generated = 0
    total_submitted = 0

    for dir_name, tag in pca_targets:
        embedding_root = output_base / dir_name
        if not embedding_root.exists():
            print(f"\n  SKIP {tag}: {embedding_root} does not exist")
            continue

        # If --models filter is set, skip this directory if none of the
        # requested models have embeddings here (avoids noisy errors)
        if args.models:
            available_models = [m for m in args.models
                                if (embedding_root / m).is_dir()]
            if not available_models:
                if args.verbose:
                    print(f"\n  SKIP {tag}: no requested models found")
                continue

        for head_type in head_types:
            ht_suffix = f"_{head_type}" if head_type != 'mlp' else ''
            full_tag = f"{tag}{ht_suffix}"

            print(f"\n  Processing: {full_tag} ({embedding_root.name}, {head_type})")

            # Generate downstream scripts
            scripts_dir_name = f"row_data_downstream_{full_tag}"
            scripts_dir = project_root / paths_config.get(
                'row_data_downstream_scripts_dir', 'slurm/scripts/generated/row_data_downstream'
            )
            # Use the tag-specific directory
            tagged_scripts_dir = scripts_dir.parent / scripts_dir_name

            if not args.no_generate:
                cmd = [sys.executable, str(generator_script),
                       '--embedding-root', str(embedding_root),
                       '--result-tag', full_tag,
                       '--head-type', head_type]
                if args.models:
                    cmd.extend(['--models'] + args.models)
                if args.datasets:
                    cmd.extend(['--datasets'] + args.datasets)

                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"    Error generating scripts for {full_tag}:")
                    if result.stderr:
                        print(f"    {result.stderr[:200]}")
                    continue
                if args.verbose:
                    print(result.stdout)

            # Collect scripts
            if not tagged_scripts_dir.exists():
                continue

            scripts = sorted(tagged_scripts_dir.glob('*.sbatch'))
            if not scripts:
                continue

            total_generated += len(scripts)
            print(f"    {len(scripts)} scripts generated")

            if args.dry_run:
                if args.verbose:
                    for s in scripts:
                        print(f"      [DRY-RUN] {s.name}")
                continue

            # Import discover_embeddings for building valid pairs
            tools_dir = Path(__file__).resolve().parent
            if str(tools_dir) not in sys.path:
                sys.path.insert(0, str(tools_dir))
            from generate_row_data_downstream_scripts import (
                discover_embeddings,
                discover_trained_row_data_models,
            )

            discovered = discover_embeddings(
                embedding_root,
                args.models,
                args.datasets,
                strict_overlay_models=discover_trained_row_data_models(project_root),
            )
            valid_pairs = {(m, d) for m, d, _ in discovered}

            # Build submission list by matching filenames against valid pairs.
            # Cannot naively split on '_' since model names may contain underscores
            # (e.g., tabular_binning). Instead, try all valid (model, dataset) pairs
            # and check if the filename starts with "{model}_{dataset}_seed".
            scripts_to_submit = []
            for s in scripts:
                stem = s.stem  # e.g., "tabular_binning_openml_1063_seed42"
                matched = False
                for model, dataset in valid_pairs:
                    prefix = f"{model}_{dataset}_seed"
                    if stem.startswith(prefix):
                        scripts_to_submit.append((model, dataset, s))
                        matched = True
                        break

            if scripts_to_submit:
                import time
                for model, dataset, script_path in scripts_to_submit:
                    success, result = submit_job(script_path, False)
                    if success:
                        total_submitted += 1
                        if args.verbose:
                            print(f"      {model}_{dataset}: Job {result}")
                    else:
                        print(f"      {model}_{dataset}: FAILED - {result}")
                    if args.delay > 0:
                        time.sleep(args.delay)

    print("\n" + "=" * 60)
    print(f"Total: {total_generated} scripts generated, {total_submitted} submitted")
    print(f"PCA directories processed: {len(pca_targets)}")

    if args.dry_run:
        print(f"\n[DRY-RUN] Would submit {total_generated} downstream jobs across {len(pca_targets)} PCA directories")


def _run_row_data_pipeline(args, project_root, config_dir, resources_config, paths_config, status_file):
    """Handle --type row_data and row_data_downstream."""

    if args.type == 'row_data':
        print("Loading row_data configurations...")
        models_config = load_yaml(config_dir / 'row_data_models.yaml')
        all_models = list(models_config.get('models', {}).keys())
        scripts_dir = project_root / paths_config['row_data_scripts_dir']
        generator_script = project_root / 'slurm' / 'generate_row_data_scripts.py'
    else:
        # row_data_downstream
        all_models = None  # Discovered from embeddings, not config
        ds_scripts_base = paths_config['row_data_downstream_scripts_dir']
        scripts_dir = project_root / ds_scripts_base
        generator_script = project_root / 'slurm' / 'generate_row_data_downstream_scripts.py'

    models = args.models
    datasets = args.datasets

    # --dim mode: auto-filter to SSL (trained) models
    if getattr(args, 'dim', None) is not None and args.type == 'row_data':
        ssl_models = [m for m, cfg in models_config.get('models', {}).items()
                      if cfg.get('model_type') == 'trained']
        if models:
            models = [m for m in models if m in ssl_models]
        else:
            models = ssl_models
        if not models:
            print("Error: --dim requires at least one trained row_data model")
            sys.exit(1)
        all_models = ssl_models
        print(f"--dim {args.dim}: filtered to {len(models)} SSL models")

    # Validate models against config (row_data only)
    if args.type == 'row_data' and models:
        invalid_models = set(models) - set(all_models)
        if invalid_models:
            print(f"Error: Unknown models: {invalid_models}")
            print(f"Available: {all_models}")
            sys.exit(1)

    # Validate datasets against filesystem (row_data only)
    if args.type == 'row_data' and datasets:
        for ds in datasets:
            manifest = project_root / 'data' / 'row_data' / ds / 'dataset.json'
            if not manifest.exists():
                print(f"Error: Dataset not found: {ds} (no {manifest})")
                sys.exit(1)

    # Discover valid targets for status initialization and stale-script filtering
    if args.type == 'row_data':
        discovered_datasets = _discover_row_data_datasets(project_root, datasets)
        resolved_models = models or all_models
        # Valid pairs: all model x dataset combinations
        valid_pairs = {(m, d) for m in resolved_models for d in discovered_datasets}
        print(f"\nPreparing {len(resolved_models)} models x {len(discovered_datasets)} datasets = {len(valid_pairs)} jobs")
    elif args.type == 'row_data_downstream':
        # Import discover_embeddings from the downstream generator (same directory)
        tools_dir = Path(__file__).resolve().parent
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        from generate_row_data_downstream_scripts import (
            discover_embeddings,
            discover_trained_row_data_models,
            derive_result_tag,
            resolve_embedding_roots,
        )
        default_root = paths_config.get('row_data_output_dir', 'embeddings/row_prediction')
        effective_result_tag = derive_result_tag(args.embedding_root, default_root, getattr(args, 'result_tag', None))
        embeddings_roots = resolve_embedding_roots(project_root, default_root, args.embedding_root)
        discovered = discover_embeddings(
            embeddings_roots,
            models,
            datasets,
            strict_overlay_models=discover_trained_row_data_models(project_root),
        )
        valid_pairs = {(m, d) for m, d, _ in discovered}
        discovered_datasets = sorted({d for _, d in valid_pairs})
        resolved_models = sorted({m for m, _ in valid_pairs})
        print(f"\nFound {len(valid_pairs)} (model, dataset) embedding pairs")
    else:
        discovered_datasets = datasets or []
        resolved_models = models or []
        valid_pairs = None

    # Backup existing embeddings if requested (row_data only, not downstream)
    if args.backup and args.type == 'row_data':
        bk_models = models or all_models
        bk_datasets = discovered_datasets
        backup_embeddings(project_root, bk_models, bk_datasets, args.verbose, embedding_type='row_data')

    # Generate scripts if needed
    if not args.no_generate:
        print("\nGenerating sbatch scripts...")
        cmd = [sys.executable, str(generator_script)]
        if models:
            cmd.extend(['--models'] + models)
        if datasets:
            cmd.extend(['--datasets'] + datasets)
        if getattr(args, 'dim', None) is not None and args.type == 'row_data':
            cmd.extend(['--dim', str(args.dim)])
        if getattr(args, 'embedding_root', None) and args.type == 'row_data_downstream':
            cmd.extend(['--embedding-root', args.embedding_root])
        if args.type == 'row_data_downstream':
            effective_result_tag = locals().get('effective_result_tag', getattr(args, 'result_tag', None))
            head_types = getattr(args, 'head_type', ['mlp']) or ['mlp']
            if len(head_types) > 1:
                print("Error: row_data_downstream currently supports one --head-type at a time")
                sys.exit(1)
            cmd.extend(['--head-type', head_types[0]])
            if effective_result_tag:
                cmd.extend(['--result-tag', effective_result_tag])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("Error generating scripts:")
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
            sys.exit(1)
        if args.verbose:
            print(result.stdout)

    # Collect scripts to submit, filtering against valid_pairs to avoid stale scripts
    scripts_to_submit = []
    if scripts_dir.exists():
        if args.type == 'row_data_downstream':
            effective_result_tag = locals().get('effective_result_tag', getattr(args, 'result_tag', None))
            ds_scripts_base = paths_config['row_data_downstream_scripts_dir']
            if effective_result_tag:
                ds_scripts_base = f"{ds_scripts_base}_{effective_result_tag}"
            scripts_dir = project_root / ds_scripts_base
        # Determine model list for filename parsing
        if args.type == 'row_data_downstream':
            # Use dynamically discovered models, not just YAML-listed ones
            parse_models = resolved_models
            requested_head = (getattr(args, 'head_type', ['mlp']) or ['mlp'])[0]
        else:
            rd_config = load_yaml(config_dir / 'row_data_models.yaml')
            parse_models = list(rd_config.get('models', {}).keys())
            requested_head = None

        for script_path in sorted(scripts_dir.glob('*.sbatch')):
            stem = script_path.stem  # e.g. "tabpfn_openml_1486" or "tabbie_openml_3_seed42"
            matched = False
            for model in parse_models:
                if stem.startswith(f"{model}_"):
                    dataset = stem[len(f"{model}_"):]
                    if args.type == 'row_data_downstream':
                        if requested_head == 'mlp':
                            if re.search(r'_seed\d+_(?:linear|dummy)$', dataset):
                                if args.verbose:
                                    print(f"  SKIP: {script_path.name} (head mismatch)")
                                break
                        else:
                            if not dataset.endswith(f"_{requested_head}"):
                                if args.verbose:
                                    print(f"  SKIP: {script_path.name} (head mismatch)")
                                break
                    # Strip _seed{N} suffix for valid_pairs matching
                    dataset_base = re.sub(r'_seed\d+(?:_(?:linear|dummy))?$', '', dataset)
                    if valid_pairs is not None and (model, dataset_base) not in valid_pairs:
                        if args.verbose:
                            print(f"  SKIP: {script_path.name} (not in valid pairs)")
                        break
                    scripts_to_submit.append((model, dataset, script_path))
                    matched = True
                    break
            if not matched and args.verbose:
                print(f"  SKIP: {script_path.name} (could not parse or filtered out)")

    _submit_scripts(args, scripts_to_submit, status_file, embedding_type=args.type)


def _submit_scripts(args, scripts_to_submit, status_file, embedding_type=None):
    """Common submission logic for all pipeline types."""
    if embedding_type is None:
        embedding_type = args.type

    if not scripts_to_submit:
        print("No scripts to submit.")
        sys.exit(0)

    # Initialize status file
    jobs_list = [(m, d) for m, d, _ in scripts_to_submit]
    initialize_status_file(status_file, jobs_list, embedding_type=embedding_type)

    # Submit jobs
    print(f"\nSubmitting {len(scripts_to_submit)} jobs...")
    print("=" * 60)

    submitted = 0
    failed = 0
    job_ids = []

    import time

    for model, dataset, script_path in scripts_to_submit:
        success, result = submit_job(script_path, args.dry_run)

        if success:
            submitted += 1
            job_ids.append(result)
            status = "[DRY-RUN]" if args.dry_run else f"Job {result}"
            print(f"  {model}_{dataset}: {status}")
        else:
            failed += 1
            print(f"  {model}_{dataset}: FAILED - {result}")

        # Small delay to avoid overwhelming SLURM scheduler
        if not args.dry_run and args.delay > 0:
            time.sleep(args.delay)

    print("=" * 60)
    print(f"Submitted: {submitted}")
    if failed:
        print(f"Failed:    {failed}")

    if not args.dry_run and submitted > 0:
        print(f"\nMonitor progress with:")
        print(f"  python slurm/monitor_jobs.py")
        print(f"\nOr check status file:")
        print(f"  cat {status_file} | python -m json.tool")


if __name__ == '__main__':
    main()
