#!/usr/bin/env python3
"""
Generate SLURM sbatch scripts for row_prediction downstream evaluation.

Discovers available row data embeddings from
embeddings/row_prediction/{model}/{dataset}/metadata.json
and generates individual sbatch scripts for each (model, dataset) pair.

Usage:
    # Generate all scripts
    python generate_row_data_downstream_scripts.py

    # Generate for specific models
    python generate_row_data_downstream_scripts.py --models tabpfn scarf

    # Generate for specific datasets
    python generate_row_data_downstream_scripts.py --datasets openml_1486 openml_3

    # Dry run (show what would be generated)
    python generate_row_data_downstream_scripts.py --dry-run
"""

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import yaml

import local_overrides as local


def get_project_root() -> Path:
    """Get the project root directory.

    File at slurm/generate_row_data_downstream_scripts.py; one .parent
    reaches the repo root.
    """
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent


def load_yaml(path: Path) -> dict:
    """Load a YAML configuration file."""
    with open(path, 'r') as f:
        return local.apply(yaml.safe_load(f))


def resolve_embedding_roots(
    project_root: Path,
    default_root: str | Path,
    overlay_root: str | Path | None = None,
) -> list[Path]:
    """Resolve row-data embedding roots in precedence order.

    Earlier roots win when the same (model, dataset) exists in multiple roots.
    This lets us overlay a specific SSL dimension directory on top of the
    canonical fixed-model root without copying fixed embeddings repeatedly.
    """

    def _resolve(path_like: str | Path) -> Path:
        p = Path(path_like)
        return p if p.is_absolute() else project_root / p

    roots: list[Path] = []
    if overlay_root:
        roots.append(_resolve(overlay_root))

    default = _resolve(default_root)
    if default not in roots:
        roots.append(default)
    return roots


def derive_result_tag(
    overlay_root: str | Path | None,
    default_root: str | Path,
    explicit_tag: str | None = None,
) -> str | None:
    """Derive a stable result tag from an overlay root when none is provided."""
    if explicit_tag:
        return explicit_tag
    if not overlay_root:
        return None

    overlay_name = Path(overlay_root).name
    default_name = Path(default_root).name
    if overlay_name == default_name:
        return None
    if overlay_name.startswith(default_name + "_"):
        return overlay_name[len(default_name) + 1:]
    return overlay_name


def discover_trained_row_data_models(project_root: Path) -> set[str]:
    """Return trained row_data models whose alternate dims live in overlay roots."""
    cfg = load_yaml(project_root / 'slurm' / 'config' / 'row_data_models.yaml')
    return {
        model_name
        for model_name, model_cfg in (cfg.get('models', {}) or {}).items()
        if model_cfg.get('model_type') == 'trained'
    }


def resolve_model_roots(
    default_root: Path,
    overlay_root: Path | None,
    strict_overlay_models: set[str],
    model_name: str,
) -> list[Path]:
    """Resolve precedence roots for a model, keeping trained overlays strict."""
    if overlay_root and model_name in strict_overlay_models:
        return [overlay_root]
    return [default_root]


def discover_embeddings(
    embeddings_root: Path | list[Path],
    filter_models: list[str] | None = None,
    filter_datasets: list[str] | None = None,
    strict_overlay_models: set[str] | None = None,
) -> list[tuple[str, str, Path]]:
    """
    Discover available row data embeddings.

    Returns:
        list of (model_name, dataset_name, embedding_dir) tuples
    """
    roots = embeddings_root if isinstance(embeddings_root, list) else [embeddings_root]
    default_root = roots[-1]
    overlay_root = roots[0] if len(roots) > 1 else None
    strict_models = strict_overlay_models or set()

    candidate_models = set(filter_models or [])
    for root in roots:
        if not root.exists():
            continue
        for model_dir in sorted(root.iterdir()):
            if model_dir.is_dir():
                candidate_models.add(model_dir.name)

    discovered: dict[tuple[str, str], Path] = {}
    for model_name in sorted(candidate_models):
        if filter_models and model_name not in filter_models:
            continue
        for root in resolve_model_roots(default_root, overlay_root, strict_models, model_name):
            model_dir = root / model_name
            if not model_dir.is_dir():
                continue
            for dataset_dir in sorted(model_dir.iterdir()):
                if not dataset_dir.is_dir():
                    continue
                dataset_name = dataset_dir.name
                if filter_datasets and dataset_name not in filter_datasets:
                    continue

                metadata = dataset_dir / 'metadata.json'
                if not metadata.exists():
                    continue

                key = (model_name, dataset_name)
                if key not in discovered:
                    discovered[key] = dataset_dir

    return [(m, d, discovered[(m, d)]) for m, d in sorted(discovered)]


def generate_downstream_script(
    model_name: str,
    dataset_name: str,
    embedding_dir: Path,
    resources: dict,
    project_root: Path,
    template_content: str,
    seed: int = 42,
    head_type: str = 'mlp',
    result_tag: str | None = None,
    variant: str | None = None,
) -> str:
    """Generate a single sbatch script for a downstream row_prediction job."""

    # CPU-only LIGHT profile resources
    time_limit = "02:00:00"
    memory = "32G"
    cpus = 8
    partition = local.partition("cpubase_bycore_b2")

    # Resolve paths
    paths_config = resources.get('paths', {})
    logs_dir = project_root / paths_config.get('logs_dir', 'slurm/logs')
    status_file = project_root / paths_config.get('status_file', 'slurm/logs/status/job_status.json')
    if result_tag:
        output_dir = project_root / 'results' / 'evaluation' / 'row_prediction' / model_name / dataset_name / result_tag / f'seed{seed}'
    else:
        output_dir = project_root / 'results' / 'evaluation' / 'row_prediction' / model_name / dataset_name / f'seed{seed}'

    # Build SLURM directives
    extra_directives = []
    slurm_defaults = resources.get('slurm_defaults', {})
    if slurm_defaults.get('account'):
        extra_directives.append(f"#SBATCH --account={slurm_defaults['account']}")
    if slurm_defaults.get('mail_user'):
        extra_directives.append(f"#SBATCH --mail-type={slurm_defaults.get('mail_type', 'FAIL')}")
        extra_directives.append(f"#SBATCH --mail-user={slurm_defaults['mail_user']}")

    # Template substitutions
    head_suffix = f"_{head_type}" if head_type != 'mlp' else ''
    substitutions = {
        'JOB_NAME': local.job_name(f"ds_rowpred_{model_name}_{dataset_name}_seed{seed}{head_suffix}"),
        'SEED': str(seed),
        'MODEL': model_name,
        'DATASET': dataset_name,
        'TIME_LIMIT': time_limit,
        'MEMORY': memory,
        'CPUS': str(cpus),
        'PARTITION': partition,
        'EXTRA_SLURM_DIRECTIVES': '\n'.join(extra_directives) if extra_directives else '',
        'PROJECT_ROOT': str(project_root),
        'EMBEDDING_DIR': str(embedding_dir),
        'OUTPUT_DIR': str(output_dir),
        'STATUS_FILE': str(status_file),
        'LOG_DIR': str(logs_dir),
        'TIMESTAMP': datetime.now().isoformat(),
        'ENV_SETUP': local.env_setup('source load_env'),
        'HEAD_TYPE': head_type,
        'HEAD_SUFFIX': head_suffix,
        'RESULT_TAG': result_tag or '',
        'VARIANT': variant or result_tag or '',
    }

    # Adjust output path for linear probe
    if head_type in ('linear', 'dummy'):
        substitutions['OUTPUT_DIR'] = str(Path(substitutions['OUTPUT_DIR']) / head_type)

    result = template_content
    for key, value in substitutions.items():
        result = result.replace(f'${{{key}}}', str(value))

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Generate SLURM sbatch scripts for row_prediction downstream evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--seeds', nargs='+', type=int, default=[42],
                       help='Seeds to generate scripts for (default: [42])')
    parser.add_argument('--models', nargs='+', help='Generate scripts only for these models')
    parser.add_argument('--datasets', nargs='+', help='Generate scripts only for these datasets')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be generated without writing')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--head-type', type=str, default='mlp', choices=['mlp', 'linear', 'dummy'],
                       help='Probe type: mlp (default), linear, or dummy (majority/mean baseline)')
    parser.add_argument('--embedding-root', type=str, default=None,
                       help='Overlay embedding root directory (e.g., embeddings/row_prediction_dim512). '
                            'When set, discovery merges this root on top of the canonical '
                            'embeddings/row_prediction root.')
    parser.add_argument('--result-tag', type=str, default=None,
                       help='Tag for result namespacing (e.g., "pca128_from768"). '
                            'Affects output dirs and script directories.')
    parser.add_argument('--variant', type=str, default=None,
                       help='Embedding variant label passed to train_downstream.py (default: derived from --result-tag)')

    args = parser.parse_args()

    project_root = get_project_root()
    template_path = project_root / 'slurm' / 'scripts' / 'templates' / 'downstream' / 'row_prediction.sbatch.template'
    resources_config = load_yaml(project_root / 'slurm' / 'config' / 'resources.yaml')

    # Discover available embeddings
    paths_config = resources_config.get('paths', {})
    default_root = paths_config.get('row_data_output_dir', 'embeddings/row_prediction')
    embeddings_roots = resolve_embedding_roots(project_root, default_root, args.embedding_root)
    effective_result_tag = derive_result_tag(args.embedding_root, default_root, args.result_tag)
    strict_overlay_models = discover_trained_row_data_models(project_root)

    print("Discovering available row data embeddings...")
    for root in embeddings_roots:
        print(f"  root: {root}")
    available = discover_embeddings(
        embeddings_roots,
        args.models,
        args.datasets,
        strict_overlay_models=strict_overlay_models,
    )

    if not available:
        print("No embeddings found in the configured embedding roots.")
        print("Run row data embedding generation first.")
        sys.exit(1)

    # Get output directory for generated scripts
    ds_scripts_dir = resources_config['paths'].get(
        'row_data_downstream_scripts_dir', 'slurm/scripts/generated/row_data_downstream'
    )
    if effective_result_tag:
        ds_scripts_dir = f"{ds_scripts_dir}_{effective_result_tag}"
    output_dir = project_root / ds_scripts_dir

    # Load template
    with open(template_path, 'r') as f:
        template_content = f.read()

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating downstream scripts for {len(available)} (model, dataset) pairs")
    if effective_result_tag:
        print(f"Result tag: {effective_result_tag}")
    print("=" * 60)

    generated = 0

    for model_name, dataset_name, embedding_dir in available:
        for seed in args.seeds:
            head_suffix = f"_{args.head_type}" if args.head_type != 'mlp' else ''
            script_name = f"{model_name}_{dataset_name}_seed{seed}{head_suffix}.sbatch"
            script_path = output_dir / script_name

            if args.dry_run:
                print(f"  [DRY-RUN] Would generate: {script_name}")
                if args.verbose:
                    print(f"            Embeddings: {embedding_dir}")
            else:
                script_content = generate_downstream_script(
                    model_name, dataset_name, embedding_dir,
                    resources_config, project_root, template_content,
                    seed=seed,
                    head_type=args.head_type,
                    result_tag=effective_result_tag,
                    variant=args.variant,
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
    print(f"Generated: {generated} downstream scripts")
    if not args.dry_run:
        print(f"Output:    {output_dir}")


if __name__ == '__main__':
    main()
