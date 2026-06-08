#!/usr/bin/env python
"""
Generate OpenAI embeddings for all datasets where GTE has embeddings.

Covers column, row, and table (derived) embeddings. Uses the Batch API
for column embeddings and synchronous API for row embeddings.

This script is designed to run on a machine with internet access. It reads
the existing GTE embedding assets and dataset configs to determine what
needs to be generated, then runs the appropriate OpenAI scripts.

Usage:
    # Generate all column embeddings (batch API)
    python models/openai/generate_all_embeddings.py --mode column

    # Generate all row embeddings (sync API)
    python models/openai/generate_all_embeddings.py --mode row

    # Generate table embeddings (derived from column, no API needed)
    python models/openai/generate_all_embeddings.py --mode table

    # Generate everything
    python models/openai/generate_all_embeddings.py --mode all

    # Dry run — just show what would be generated
    python models/openai/generate_all_embeddings.py --mode all --dry_run

    # Generate only specific datasets
    python models/openai/generate_all_embeddings.py --mode column --datasets santos nq_tables
"""

import os
import sys
import argparse
import subprocess
import yaml
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_datasets_config():
    config_path = PROJECT_ROOT / 'slurm' / 'config' / 'datasets.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_csv_dir(dataset_name, datasets_config):
    """Resolve the CSV directory for a dataset."""
    ds_cfg = datasets_config['datasets'].get(dataset_name, {})
    tables_source = ds_cfg.get('tables_source')
    tables_dir = ds_cfg.get('tables_dir')

    if tables_source:
        base = PROJECT_ROOT / 'datasets' / tables_source
        source_cfg = datasets_config['datasets'].get(tables_source, {})
        tables_dir = source_cfg.get('tables_dir', tables_dir)
    else:
        base = PROJECT_ROOT / 'datasets' / dataset_name

    if tables_dir:
        csv_dir = base / tables_dir
    else:
        # Auto-detect: check common subdirs
        for subdir in ['csv', 'tables', 'datalake', 'datasets', '.']:
            candidate = base / subdir
            if candidate.is_dir():
                csvs = [f for f in os.listdir(candidate) if f.endswith('.csv')]
                if csvs:
                    csv_dir = candidate
                    break
        else:
            return None

    if csv_dir.is_dir():
        csvs = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
        if csvs:
            return csv_dir
    return None


def get_gte_datasets(embedding_type):
    """Get list of datasets that GTE has embeddings for."""
    gte_dir = PROJECT_ROOT / 'assets' / 'embeddings' / embedding_type / 'gte'
    if not gte_dir.is_dir():
        return []
    if embedding_type == 'row_prediction':
        # Directories with metadata.json, not .pkl files
        return sorted([d for d in os.listdir(gte_dir)
                       if (gte_dir / d / 'metadata.json').exists()])
    return sorted([f.replace('.pkl', '') for f in os.listdir(gte_dir) if f.endswith('.pkl')])


def get_existing_openai_datasets(embedding_type, label='openai'):
    """Get list of datasets that already have embeddings for the given label."""
    out_dir = PROJECT_ROOT / 'assets' / 'embeddings' / embedding_type / label
    if not out_dir.is_dir():
        return []
    if embedding_type == 'row_prediction':
        return sorted([d for d in os.listdir(out_dir)
                       if (out_dir / d / 'metadata.json').exists()])
    return sorted([f.replace('.pkl', '') for f in os.listdir(out_dir) if f.endswith('.pkl')])


def run_command(cmd, dry_run=False):
    """Run a command, printing it first."""
    print(f"  $ {' '.join(cmd)}")
    if dry_run:
        print("    [dry run — skipped]")
        return True
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode == 0


def generate_column_embeddings(datasets, datasets_config, args):
    """Generate column embeddings.

    With batch API (default): submits all datasets first, then polls all
    together, then downloads all — maximizing parallelism on OpenAI's side.
    With sync API: processes datasets sequentially.
    """
    openai_dir = PROJECT_ROOT / 'assets' / 'embeddings' / 'column' / args.model_label
    openai_dir.mkdir(parents=True, exist_ok=True)

    if not args.use_batch:
        # Sync mode: sequential
        for ds in datasets:
            output = openai_dir / f'{ds}.pkl'
            if output.exists() and not args.force:
                print(f"  [skip] {ds} — already exists")
                continue
            csv_dir = resolve_csv_dir(ds, datasets_config)
            if csv_dir is None:
                print(f"  [skip] {ds} — CSV directory not found")
                continue
            print(f"\n  [{ds}] CSV dir: {csv_dir}")
            cmd = [
                sys.executable, str(SCRIPT_DIR / 'generate_column_embeddings.py'),
                '--input', str(csv_dir),
                '--output', str(output),
                '--model', str(args.model),
                '--max_rows', str(args.max_rows),
                '--dimensions', str(args.dimensions),
                '--workers', str(args.workers),
            ]
            run_command(cmd, args.dry_run)
        return

    # Batch mode: submit all → poll all → download all
    batch_script = str(SCRIPT_DIR / 'generate_column_embeddings_batch.py')

    # Phase 1: Submit all datasets
    pending = []
    print("\n  --- Phase 1: Submitting all datasets ---")
    for ds in datasets:
        output = openai_dir / f'{ds}.pkl'
        if output.exists() and not args.force:
            print(f"  [skip] {ds} — already exists")
            continue
        csv_dir = resolve_csv_dir(ds, datasets_config)
        if csv_dir is None:
            print(f"  [skip] {ds} — CSV directory not found")
            continue

        print(f"\n  [{ds}] Submitting...")
        cmd = [
            sys.executable, batch_script,
            '--input', str(csv_dir),
            '--output', str(output),
            '--model', str(args.model),
            '--max_rows', str(args.max_rows),
            '--dimensions', str(args.dimensions),
            '--submit_only',
        ]
        if not run_command(cmd, args.dry_run):
            print(f"  [{ds}] Submit failed, skipping")
            continue
        pending.append((ds, output, csv_dir))

    if not pending or args.dry_run:
        return

    # Fix stale failed batches: mark queue-limit failures as retryable
    print(f"\n  --- Fixing stale failed batches ---")
    import json
    import time
    from models.openai.client import create_client
    client, _ = create_client()

    for ds, output, csv_dir in pending:
        work_dir = str(output) + ".batch_work"
        state_path = os.path.join(work_dir, "batch_state.json")
        if not os.path.exists(state_path):
            continue
        with open(state_path) as f:
            state = json.load(f)
        changed = False
        for entry in state["batches"]:
            if entry["status"] == "failed" and not entry.get("retryable"):
                try:
                    batch = client.batches.retrieve(entry["batch_id"])
                    if batch.errors and any(
                        (e.code or '') in ('request_limit_exceeded', 'token_limit_exceeded')
                        for e in batch.errors.data
                    ):
                        entry["retryable"] = True
                        changed = True
                except Exception:
                    pass
        if changed:
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)

    # Phase 2: Poll all datasets together
    print(f"\n  --- Phase 2: Polling {len(pending)} datasets ---")
    last_retry_time = 0
    RETRY_INTERVAL = 300  # Only retry failed batches every 5 minutes

    while True:
        all_done = True
        total_completed = 0
        total_requests = 0
        now = time.time()
        can_retry = (now - last_retry_time) >= RETRY_INTERVAL

        for ds, output, csv_dir in pending:
            work_dir = str(output) + ".batch_work"
            state_path = os.path.join(work_dir, "batch_state.json")
            if not os.path.exists(state_path):
                continue

            with open(state_path) as f:
                state = json.load(f)

            ds_done = True
            ds_completed = 0
            ds_total = 0
            for entry in state["batches"]:
                ds_total += entry["num_requests"]
                if entry["status"] == "completed":
                    ds_completed += entry["num_requests"]
                    continue

                # For any non-completed batch, check live status
                ds_done = False

                if entry.get("retryable"):
                    if can_retry:
                        try:
                            new_batch = client.batches.create(
                                input_file_id=entry["file_id"],
                                endpoint="/v1/embeddings",
                                completion_window="24h",
                            )
                            entry["batch_id"] = new_batch.id
                            entry["status"] = new_batch.status
                            entry["retryable"] = False
                        except Exception:
                            pass
                    continue

                # Query live status from OpenAI
                try:
                    batch = client.batches.retrieve(entry["batch_id"])
                except Exception:
                    continue
                entry["status"] = batch.status
                if batch.status == "completed":
                    entry["output_file_id"] = batch.output_file_id
                    ds_completed += entry["num_requests"]
                elif batch.status in ("failed", "expired", "cancelled"):
                    try:
                        if batch.errors and any(
                            (e.code or '') in ('request_limit_exceeded', 'token_limit_exceeded')
                            for e in batch.errors.data
                        ):
                            entry["retryable"] = True
                    except Exception:
                        pass

            # Save updated state
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)

            if not ds_done:
                all_done = False

            total_completed += ds_completed
            total_requests += ds_total

        if can_retry:
            last_retry_time = now

        if total_requests > 0:
            pct = total_completed / total_requests * 100
            print(f"  Overall: {total_completed:,}/{total_requests:,} requests ({pct:.1f}%)")

        if all_done:
            break

        time.sleep(args.poll_interval)

    # Phase 3: Download and assemble all datasets
    print(f"\n  --- Phase 3: Downloading {len(pending)} datasets ---")
    for ds, output, csv_dir in pending:
        if output.exists() and not args.force:
            continue
        print(f"\n  [{ds}] Downloading and assembling...")
        cmd = [
            sys.executable, batch_script,
            '--input', str(csv_dir),
            '--output', str(output),
            '--max_rows', str(args.max_rows),
            '--dimensions', str(args.dimensions),
            '--download_only',
        ]
        run_command(cmd, args.dry_run)


def generate_row_embeddings(datasets, datasets_config, args):
    """Generate row embeddings. Uses sync API (sequential per dataset)."""
    openai_dir = PROJECT_ROOT / 'assets' / 'embeddings' / 'row' / args.model_label
    openai_dir.mkdir(parents=True, exist_ok=True)

    row_datasets_config_path = PROJECT_ROOT / 'slurm' / 'config' / 'row_datasets.yaml'
    row_datasets_cfg = {}
    if row_datasets_config_path.exists():
        with open(row_datasets_config_path) as f:
            row_datasets_cfg = yaml.safe_load(f) or {}

    sync_script = str(SCRIPT_DIR / 'generate_row_embeddings.py')

    for ds in datasets:
        output = openai_dir / f'{ds}.pkl'
        if output.exists() and not args.force:
            print(f"  [skip] {ds} — already exists")
            continue

        csv_dir = resolve_csv_dir(ds, datasets_config)
        if csv_dir is None:
            print(f"  [skip] {ds} — CSV directory not found")
            continue

        print(f"\n  [{ds}] CSV dir: {csv_dir}")

        cmd = [
            sys.executable, sync_script,
            '--input_dir', str(csv_dir),
            '--output_path', str(output),
            '--model', str(args.model),
            '--dimensions', str(args.dimensions),
            '--max_chars_per_cell', '100',
            '--row_batch_size', '256',
            '--checkpoint_interval', '50',
            '--workers', str(args.workers),
        ]

        ds_cfg = row_datasets_cfg.get('datasets', {}).get(ds, {})
        label_cols = ds_cfg.get('label_columns')
        if label_cols:
            cmd.extend(['--label_columns'] + label_cols)

        max_rows = ds_cfg.get('max_rows')
        if max_rows:
            cmd.extend(['--max_rows', str(max_rows)])

        run_command(cmd, args.dry_run)


def generate_row_prediction_embeddings(datasets, args):
    """Generate row_prediction embeddings (split-aware, for canonical datasets).

    Runs multiple datasets concurrently using a process pool.
    """
    openai_dir = PROJECT_ROOT / 'assets' / 'embeddings' / 'row_prediction' / args.model_label
    data_root = PROJECT_ROOT / 'datasets' / 'row_data'
    script = str(SCRIPT_DIR / 'generate_embeddings_train_test.py')

    # Build list of (ds, cmd) pairs
    pending = []
    for ds in datasets:
        out_dir = openai_dir / ds
        if (out_dir / 'metadata.json').exists() and not args.force:
            print(f"  [skip] {ds} — already exists")
            continue

        data_dir = data_root / ds
        if not (data_dir / 'dataset.json').exists():
            print(f"  [skip] {ds} — dataset not found at {data_dir}")
            continue

        cmd = [
            sys.executable, script,
            '--data_dir', str(data_dir),
            '--embedding_dir', str(out_dir),
            '--model', str(args.model),
            '--dimensions', str(args.dimensions),
            '--max_chars_per_cell', '100',
            '--row_batch_size', '256',
            '--label_policy', 'manifest',
        ]
        pending.append((ds, cmd))

    if not pending or args.dry_run:
        for ds, cmd in pending:
            print(f"  [{ds}] $ {' '.join(cmd)}")
            print(f"    [dry run — skipped]")
        return

    workers = min(args.workers, len(pending))
    print(f"\n  Running {len(pending)} datasets with {workers} parallel workers...")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _run_one(ds_cmd):
        ds, cmd = ds_cmd
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT),
                                capture_output=True, text=True)
        return ds, result.returncode, result.stdout, result.stderr

    completed_count = 0
    failed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one, p): p[0] for p in pending}
        for future in as_completed(futures):
            ds, rc, stdout, stderr = future.result()
            completed_count += 1
            if rc == 0:
                print(f"  [{completed_count}/{len(pending)}] {ds}: done")
            else:
                print(f"  [{completed_count}/{len(pending)}] {ds}: FAILED (rc={rc})")
                if stderr:
                    # Print last few lines of error
                    err_lines = stderr.strip().split('\n')
                    for line in err_lines[-3:]:
                        print(f"    {line}")
                failed.append(ds)

    if failed:
        print(f"\n  {len(failed)} datasets failed: {failed}")


def generate_table_embeddings(datasets, args):
    """Derive table embeddings from column embeddings (no API calls)."""
    col_dir = PROJECT_ROOT / 'assets' / 'embeddings' / 'column' / args.model_label
    derive_script = PROJECT_ROOT / 'scripts' / 'generate_table_embeddings.py'

    if not derive_script.exists():
        print("  Table derivation script not found at scripts/generate_table_embeddings.py")
        return

    # Filter to datasets that have column embeddings ready
    ready = []
    for ds in datasets:
        col_pkl = col_dir / f'{ds}.pkl'
        table_pkl = PROJECT_ROOT / 'assets' / 'embeddings' / 'table' / args.model_label / f'{ds}.pkl'
        if table_pkl.exists() and not args.force:
            print(f"  [skip] {ds} — already exists")
        elif not col_pkl.exists():
            print(f"  [skip] {ds} — column embeddings not found (run column mode first)")
        else:
            ready.append(ds)

    if not ready:
        print("  Nothing to derive.")
        return

    cmd = [
        sys.executable, str(derive_script),
        '--models', args.model_label,
        '--datasets', *ready,
    ]
    if args.force:
        cmd.append('--force')

    run_command(cmd, args.dry_run)


def main():
    parser = argparse.ArgumentParser(
        description='Generate OpenAI embeddings for all GTE-covered datasets'
    )
    parser.add_argument('--mode', type=str, required=True,
                        choices=['column', 'row', 'row_prediction', 'table', 'all'],
                        help='Which embedding types to generate')
    parser.add_argument('--datasets', nargs='*', default=None,
                        help='Specific datasets to process (default: all GTE datasets)')
    parser.add_argument('--model', type=str, default='text-embedding-3-small',
                        help='Model name (default: text-embedding-3-small)')
    parser.add_argument('--model_label', type=str, default=None,
                        help='Label for output dirs (default: auto from model, e.g. "openai", "openai_ada")')
    parser.add_argument('--max_rows', type=int, default=100,
                        help='Max rows per table (default: 100)')
    parser.add_argument('--dimensions', type=int, default=None,
                        help='Embedding dimensions (default: 768 for v3, native for ada-002)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Parallel workers for sync API (default: 8)')
    parser.add_argument('--use_batch', action='store_true', default=True,
                        help='Use batch API for column embeddings (default: true)')
    parser.add_argument('--use_sync', dest='use_batch', action='store_false',
                        help='Use sync API instead of batch for column embeddings')
    parser.add_argument('--poll_interval', type=int, default=60,
                        help='Batch API poll interval in seconds (default: 60)')
    parser.add_argument('--force', action='store_true',
                        help='Regenerate even if output exists')
    parser.add_argument('--dry_run', action='store_true',
                        help='Just show what would be generated')

    args = parser.parse_args()

    if not (os.environ.get('OPENAI_API_KEY') or os.environ.get('OPENROUTER_API_KEY')) and args.mode != 'table':
        print("ERROR: No API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY in environment or .env file.")
        sys.exit(1)

    # Resolve model label for output directories
    from models.openai.client import get_model_label, supports_dimensions, get_model_info
    if args.model_label is None:
        args.model_label = get_model_label(args.model)
    if args.dimensions is None and supports_dimensions(args.model):
        args.dimensions = 768  # default for v3 models to match GTE-base
    elif args.dimensions is None:
        _, native_dim, _ = get_model_info(args.model)
        args.dimensions = native_dim

    print(f"Model: {args.model}, Label: {args.model_label}, Dimensions: {args.dimensions}")

    datasets_config = load_datasets_config()
    modes = ['column', 'row', 'row_prediction', 'table'] if args.mode == 'all' else [args.mode]

    for mode in modes:
        gte_datasets = get_gte_datasets(mode)
        existing = get_existing_openai_datasets(mode, args.model_label)

        if args.datasets:
            datasets = [d for d in args.datasets if d in gte_datasets]
        else:
            datasets = gte_datasets

        pending = [d for d in datasets if d not in existing or args.force]

        print(f"\n{'='*60}")
        print(f"MODE: {mode.upper()} EMBEDDINGS")
        print(f"{'='*60}")
        print(f"GTE has: {len(gte_datasets)} datasets")
        print(f"OpenAI has: {len(existing)} datasets")
        print(f"To generate: {len(pending)} datasets")

        if not pending:
            print("Nothing to do.")
            continue

        if mode == 'column':
            generate_column_embeddings(pending, datasets_config, args)
        elif mode == 'row':
            generate_row_embeddings(pending, datasets_config, args)
        elif mode == 'row_prediction':
            generate_row_prediction_embeddings(pending, args)
        elif mode == 'table':
            generate_table_embeddings(pending, args)

    print(f"\n{'='*60}")
    print("Done.")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
