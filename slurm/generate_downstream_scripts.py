#!/usr/bin/env python3
"""
Generate SLURM sbatch scripts for downstream task evaluation.

This script reads task and dataset configurations, discovers available embeddings,
and generates individual sbatch scripts for each (task, model, dataset) combination.

Usage:
    # Generate all scripts
    python generate_downstream_scripts.py

    # Generate for specific tasks
    python generate_downstream_scripts.py --tasks column_clustering column_type_prediction

    # Generate for specific models
    python generate_downstream_scripts.py --models starmie tapas

    # Dry run (show what would be generated)
    python generate_downstream_scripts.py --dry-run

    # Verbose output
    python generate_downstream_scripts.py -v
"""

import argparse
import json
import os
import pickle
import re
import sys
from datetime import datetime
from pathlib import Path

import yaml

import local_overrides as local

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from trl_bench.utils.pickle_compat import load_pickle

_SHARD_RE = re.compile(r'.*_shard\d+of\d+\.pkl$')

def _is_shard_or_checkpoint(pkl_name: str) -> bool:
    """Return True if the pkl filename is a shard file or checkpoint."""
    return pkl_name.endswith('.checkpoint.pkl') or bool(_SHARD_RE.match(pkl_name))


def get_project_root() -> Path:
    """Get the project root directory.

    File at slurm/generate_downstream_scripts.py; one .parent reaches
    the repo root (matches _PROJECT_ROOT above).
    """
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent


def load_yaml(path: Path) -> dict:
    """Load a YAML configuration file."""
    with open(path, 'r') as f:
        return local.apply(yaml.safe_load(f))


def discover_embeddings(project_root: Path, models: list[str] | None = None,
                        embeddings_base: Path | None = None) -> dict:
    """
    Discover available embeddings in {embeddings_base}/column/{model}/{dataset}.pkl

    Returns:
        dict: {model: [dataset1, dataset2, ...]}
    """
    embeddings_dir = (embeddings_base or project_root / 'embeddings') / 'column'
    discovered = {}

    if not embeddings_dir.exists():
        print(f"Warning: Embeddings directory not found: {embeddings_dir}")
        return discovered

    for model_dir in embeddings_dir.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        if models and model_name not in models:
            continue

        datasets = []
        for pkl_file in model_dir.glob('*.pkl'):
            if _is_shard_or_checkpoint(pkl_file.name):
                continue
            dataset_name = pkl_file.stem
            datasets.append(dataset_name)

        if datasets:
            discovered[model_name] = sorted(datasets)

    return discovered


def discover_table_embeddings(project_root: Path, models: list[str] | None = None,
                              embeddings_base: Path | None = None) -> dict:
    """
    Discover available table embeddings in {embeddings_base}/table/{model}/{dataset}.pkl

    Returns:
        dict: {model: [dataset1, dataset2, ...]}
    """
    embeddings_dir = (embeddings_base or project_root / 'embeddings') / 'table'
    discovered = {}

    if not embeddings_dir.exists():
        print(f"Warning: Table embeddings directory not found: {embeddings_dir}")
        return discovered

    for model_dir in embeddings_dir.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        # Skip pseudo-model directories (hybrids, backups, query encoder bridges)
        if model_name.endswith('_hybrid') or '_backup_' in model_name:
            continue
        if model_name in QUERY_ENCODER_MODELS:
            continue
        if models and model_name not in models:
            continue

        datasets = []
        for pkl_file in model_dir.glob('*.pkl'):
            if _is_shard_or_checkpoint(pkl_file.name):
                continue
            dataset_name = pkl_file.stem
            datasets.append(dataset_name)

        if datasets:
            discovered[model_name] = sorted(datasets)

    return discovered


def resolve_optional_root(project_root: Path, path_like: str | Path | None) -> Path | None:
    """Resolve an optional path relative to the project root."""
    if not path_like:
        return None
    p = Path(path_like)
    return p if p.is_absolute() else project_root / p


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


def discover_trained_row_models(project_root: Path) -> set[str]:
    """Return row models whose outputs should come from a dimension overlay when provided."""
    cfg = load_yaml(project_root / 'slurm' / 'config' / 'row_models.yaml')
    return {
        model_name
        for model_name, model_cfg in (cfg.get('models', {}) or {}).items()
        if model_cfg.get('model_type') == 'trained'
    }


def _row_model_roots(
    default_root: Path,
    overlay_root: Path | None,
    strict_overlay_models: set[str],
    model_name: str,
) -> list[Path]:
    """Resolve row embedding roots for a model.

    When an overlay root is provided, trained SSL row models should only be read
    from that overlay so dimension-specific runs never silently fall back to the
    canonical root. Fixed/pretrained models stay on the canonical root.
    """
    if overlay_root and model_name in strict_overlay_models:
        return [overlay_root]
    return [default_root]


def resolve_row_embedding_path(
    project_root: Path,
    model_name: str,
    dataset_name: str,
    embeddings_base: Path | None = None,
    overlay_root: str | Path | None = None,
    strict_overlay_models: set[str] | None = None,
) -> Path:
    """Resolve the concrete row embedding file for a model/dataset pair."""
    default_root = (embeddings_base or project_root / 'embeddings') / 'row'
    resolved_overlay = resolve_optional_root(project_root, overlay_root)
    strict_models = strict_overlay_models or set()

    for root in _row_model_roots(default_root, resolved_overlay, strict_models, model_name):
        candidate = root / model_name / f'{dataset_name}.pkl'
        if candidate.exists():
            return candidate

    # Fall back to the first expected location for clearer downstream errors.
    preferred_root = _row_model_roots(default_root, resolved_overlay, strict_models, model_name)[0]
    return preferred_root / model_name / f'{dataset_name}.pkl'


def discover_table_embedding_variants(
    project_root: Path, models: list[str] | None = None,
    embeddings_base: Path | None = None,
) -> dict[str, dict[str, list[str]]]:
    """
    Probe table embedding pkls to discover available variants per model.

    Loads one entry from each pkl and checks which keys in
    table_embedding dict have non-None values.

    Returns: {model_name: {dataset_name: [variant1, variant2, ...]}}
    """
    embeddings_dir = (embeddings_base or project_root / 'embeddings') / 'table'
    result = {}

    if not embeddings_dir.exists():
        return result

    for model_dir in sorted(embeddings_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        # Skip pseudo-model directories (hybrids, backups, query encoder bridges)
        if model_name.endswith('_hybrid') or '_backup_' in model_name:
            continue
        if model_name in QUERY_ENCODER_MODELS:
            continue
        if models and model_name not in models:
            continue

        model_variants = {}
        # Cache: once we discover variants for one dataset, reuse for others
        cached_variants = None

        for pkl_file in sorted(model_dir.glob('*.pkl')):
            if _is_shard_or_checkpoint(pkl_file.name):
                continue
            dataset_name = pkl_file.stem

            if cached_variants is not None:
                model_variants[dataset_name] = cached_variants
                continue

            try:
                data = load_pickle(pkl_file)

                if not isinstance(data, list) or len(data) == 0:
                    continue

                # Inspect first entry's table_embedding dict
                first_entry = data[0]
                table_emb = first_entry.get('table_embedding', {})
                if not isinstance(table_emb, dict):
                    # v1.0 format: single embedding, treat as column_mean
                    cached_variants = ['column_mean']
                    model_variants[dataset_name] = cached_variants
                    continue

                # Find variants with non-None values
                variants = [k for k, v in table_emb.items() if v is not None]
                cached_variants = sorted(variants)
                model_variants[dataset_name] = cached_variants

            except Exception as e:
                print(f"  Warning: Could not probe {pkl_file}: {e}")
                continue

        if model_variants:
            result[model_name] = model_variants

    return result


def discover_dlte_embeddings(project_root: Path, models: list[str] | None = None) -> dict:
    """
    Discover available embeddings for DLTE pipeline stages.

    Returns:
        dict with keys:
            'table_column': models with table-level embeddings (stage 1)
            'column': models with column embeddings + ckan_subset (stage 2)
            'row': models with row embeddings (stages 3-4)
    """
    result = {'table_column': [], 'column': [], 'row': []}

    # Stage 1: table-level embeddings (pkl) at embeddings/table/{model}/
    table_emb_dir = project_root / 'embeddings' / 'table'
    if table_emb_dir.exists():
        for model_dir in sorted(table_emb_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            if model_name.endswith('_hybrid') or '_backup_' in model_name:
                continue
            if models and model_name not in models:
                continue
            required = ['dlte_v1_queries.pkl', 'dlte_v1_targets.pkl', 'ckan_subset.pkl']
            if all((model_dir / f).exists() for f in required):
                result['table_column'].append(model_name)

    # Stage 2: column-level embeddings (pkl) — needs queries, targets, AND ckan_subset
    col_emb_dir = project_root / 'embeddings' / 'column'
    if col_emb_dir.exists():
        for model_dir in sorted(col_emb_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            if model_name.endswith('_hybrid') or '_backup_' in model_name:
                continue
            if models and model_name not in models:
                continue
            required = ['dlte_v1_queries.pkl', 'dlte_v1_targets.pkl', 'ckan_subset.pkl']
            if all((model_dir / f).exists() for f in required):
                result['column'].append(model_dir.name)

    # Stages 3-4: row-level embeddings (pkl)
    row_emb_dir = project_root / 'embeddings' / 'row'
    if row_emb_dir.exists():
        for model_dir in sorted(row_emb_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            required = ['dlte_v1_queries.pkl', 'dlte_v1_targets.pkl']
            if all((model_dir / f).exists() for f in required):
                result['row'].append(model_dir.name)

    return result


def get_task_datasets(task_name: str, task_datasets_config: dict) -> list[str]:
    """Get the list of supported datasets for a task."""
    task_config = task_datasets_config.get('task_datasets', {}).get(task_name, {})
    datasets = task_config.get('datasets', {})
    return list(datasets.keys())


# Required config keys per task (each must be a valid file path)
TASK_REQUIRED_FILES = {
    'column_clustering': ['labels_file'],
    'column_relation_prediction': ['train_metadata', 'test_metadata'],
    'column_type_prediction': ['train_csv', 'test_csv'],
    'join_search': ['query_list', 'ground_truth'],
    'join_search_learned': ['query_list', 'ground_truth', 'split_dir'],
    'join_classification': ['labels_file'],
    'join_containment': ['labels_file'],
    'table_subset': ['labels_file'],
    'union_search': ['groundtruth_file'],
    'union_classification': ['labels_file'],
    'union_regression': ['labels_file'],
    'table_fact_verification': ['train_labels', 'val_labels', 'test_labels', 'statement_embeddings_dir'],
    'semantic_parsing': ['dataset_dir', 'question_embeddings_base', 'test_file', 'table_file', 'config_file'],
    'table_retrieval': ['train_questions', 'dev_questions', 'tables_json', 'table_id_mapping', 'query_embeddings_base'],
    # DLTE tasks use their own discovery/validation in generate_dlte_scripts()
    'record_linkage': ['labels_file'],
    'schema_matching': ['pairs_file', 'ground_truth', 'tables_dir'],
    'dlte_retrieval': ['query_tasks'],
    'dlte_alignment': ['query_tasks', 'fragments_manifest', 'table_maps_dir'],
    'dlte_merge': ['query_tasks', 'fragments_manifest', 'parents_manifest', 'table_maps_dir'],
}

# Template file mapping
TEMPLATE_FILES = {
    'column_clustering': 'column_clustering.sbatch.template',
    'column_relation_prediction': 'column_relation_prediction.sbatch.template',
    'column_type_prediction': 'column_type_prediction.sbatch.template',
    'join_search': 'join_search.sbatch.template',
    'join_search_learned': 'join_search_learned.sbatch.template',
    'join_classification': 'join_classification.sbatch.template',
    'join_containment': 'join_containment.sbatch.template',
    'table_subset': 'table_subset.sbatch.template',
    'union_search': 'union_search.sbatch.template',
    'union_classification': 'union_classification.sbatch.template',
    'union_regression': 'union_regression.sbatch.template',
    'table_fact_verification': 'table_fact_verification.sbatch.template',
    'semantic_parsing': 'semantic_parsing.sbatch.template',
    'table_retrieval': 'table_retrieval.sbatch.template',
    'record_linkage': 'record_linkage.sbatch.template',
    'schema_matching': 'schema_matching.sbatch.template',
    'dlte_retrieval': 'dlte_retrieval.sbatch.template',
    'dlte_alignment': 'dlte_alignment.sbatch.template',
    'dlte_merge': 'dlte_merge.sbatch.template',
}

# Output subdirectory mapping
OUTPUT_SUBDIRS = {
    'column_clustering': 'clustering',
    'column_relation_prediction': 'relation',
    'column_type_prediction': 'type_prediction',
    'join_search': 'join_search',
    'join_search_learned': 'join_search_learned',
    'join_classification': 'join_classification',
    'join_containment': 'join_containment',
    'table_subset': 'table_subset',
    'union_search': 'union_search',
    'union_classification': 'union_classification',
    'union_regression': 'union_regression',
    'table_fact_verification': 'table_fact_verification',
    'semantic_parsing': 'semantic_parsing',
    'table_retrieval': 'table_retrieval',
    'record_linkage': 'record_linkage',
    'schema_matching': 'schema_matching',
    'dlte_retrieval': 'dlte_retrieval',
    'dlte_alignment': 'dlte_alignment',
    'dlte_merge': 'dlte_merge',
}

# DLTE tasks use a separate dispatch mechanism (column_only / col_x_row)
# Query encoder models used as bridge table embeddings for retrieval tasks.
# These should NOT be discovered as table embedding models in the evaluation matrix.
QUERY_ENCODER_MODELS = {'sentence_t5', 'mpnet'}

DLTE_TASKS = {'dlte_retrieval', 'dlte_alignment', 'dlte_merge'}

# Deterministic tasks: no seed dimension (evaluation is fully deterministic)
DETERMINISTIC_TASKS = {
    'column_clustering', 'join_search', 'union_search',
    'schema_matching',
    'dlte_retrieval', 'dlte_alignment', 'dlte_merge',
}

# Tasks that use table-level embeddings (read from embeddings/table/)
# rather than column-level embeddings (embeddings/column/)
TABLE_EMBEDDING_TASKS = {
    'join_classification', 'union_classification', 'union_regression',
    'table_subset', 'table_fact_verification', 'table_retrieval',
}

# Tasks that use row-level embeddings (read from embeddings/row/)
ROW_EMBEDDING_TASKS = {'record_linkage'}

# Tasks whose templates pass --head_type ${HEAD_TYPE} to their Python scripts.
# Only these tasks should have paths adjusted for non-MLP probes.
PROBE_TASKS = {
    'column_relation_prediction', 'column_type_prediction',
    'join_classification', 'join_containment',
    'record_linkage', 'row_prediction',
    'table_fact_verification', 'table_subset',
    'union_classification', 'union_regression',
}

# Tasks that support the cosine_threshold head type (pair-based tasks only).
COSINE_THRESHOLD_TASKS = {'record_linkage'}

# Tasks that support the interaction head type (two-embedding cross-modal tasks only).
INTERACTION_TASKS = {'table_fact_verification'}

# Canonical variant names (pkl keys) -> script CLI values
# For run_task.py tasks (join_classification, table_subset, union_classification, union_regression, join_containment)
RUN_TASK_VARIANT_MAP = {
    'column_mean': 'column_mean',
    'cls_embedding': 'cls',
    'table_embedding': 'table',
    'token_mean': 'token_mean',
}

# For table_fact_verification (train.py / evaluate.py)
TABFACT_VARIANT_MAP = {
    'column_mean': 'column_mean',
    'cls_embedding': 'cls_embedding',
    'table_embedding': 'table_embedding',
    'token_mean': 'token_mean',
}

# For table_retrieval (data_utils.py variant_map)
TABLE_RETRIEVAL_VARIANT_MAP = {
    'column_mean': 'column_mean',
    'cls_embedding': 'cls_embedding',
    'table_embedding': 'table_embedding',
    'token_mean': 'token_mean',
}

# DLTE Stage 1 uses canonical variant names directly (no mapping needed)
DLTE_VARIANT_MAP = {
    'column_mean': 'column_mean',
    'cls_embedding': 'cls_embedding',
    'table_embedding': 'table_embedding',
    'token_mean': 'token_mean',
}

DLTE_TABLE_PKLS = ['dlte_v1_queries.pkl', 'dlte_v1_targets.pkl', 'ckan_subset.pkl']

# Variants that are cosine-equivalent to column_mean (redundant for cross-model jobs)
DLTE_REDUNDANT_VARIANTS = set()


def _probe_pkl_variants(pkl_path: Path) -> set[str] | None:
    """Probe a single pkl and return the set of non-None variant keys, or None on error."""
    # Legacy-only; remove after asset refresh
    _LEGACY_VARIANTS = {'column_sum'}
    try:
        data = load_pickle(pkl_path)
        if not isinstance(data, list) or len(data) == 0:
            return None
        table_emb = data[0].get('table_embedding', {})
        if not isinstance(table_emb, dict):
            return {'column_mean'}
        variants = {k for k, v in table_emb.items() if v is not None}
        legacy_found = variants & _LEGACY_VARIANTS
        if legacy_found:
            print(f"  Warning: skipping legacy variant {legacy_found} from {pkl_path.name} — regenerate embeddings to remove")
            variants -= legacy_found
        return variants
    except Exception as e:
        print(f"  Warning: Could not probe {pkl_path}: {e}")
        return None


def discover_dlte_table_variants(
    project_root: Path, models: list[str] | None = None
) -> dict[str, list[str]]:
    """
    Probe DLTE-specific table embedding pkls to discover available variants.

    Inspects all three DLTE pkls (queries, targets, ckan_subset) per model
    and returns the INTERSECTION of their variants.

    Returns: {model_name: [variant1, variant2, ...]}
    """
    embeddings_dir = project_root / 'embeddings' / 'table'
    result = {}

    if not embeddings_dir.exists():
        return result

    for model_dir in sorted(embeddings_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        if model_name.endswith('_hybrid') or '_backup_' in model_name:
            continue
        if models and model_name not in models:
            continue

        # All three DLTE pkls must exist
        pkl_paths = [model_dir / name for name in DLTE_TABLE_PKLS]
        if not all(p.exists() for p in pkl_paths):
            continue

        # Probe each pkl and intersect variants
        variant_sets = []
        for pkl_path in pkl_paths:
            variants = _probe_pkl_variants(pkl_path)
            if variants is None:
                break
            variant_sets.append(variants)
        else:
            # All three probed successfully — intersect
            common = variant_sets[0]
            for vs in variant_sets[1:]:
                common &= vs
            if common:
                result[model_name] = sorted(common)

    return result


def _get_variant_map(task_name: str) -> dict[str, str]:
    """Get the variant name mapping for a given task."""
    if task_name == 'table_fact_verification':
        return TABFACT_VARIANT_MAP
    elif task_name == 'table_retrieval':
        return TABLE_RETRIEVAL_VARIANT_MAP
    else:
        return RUN_TASK_VARIANT_MAP


def discover_row_embeddings(project_root: Path, models: list[str] | None = None,
                            embeddings_base: Path | None = None,
                            overlay_root: str | Path | None = None,
                            strict_overlay_models: set[str] | None = None) -> dict:
    """Discover available row embeddings at {embeddings_base}/row/{model}/{dataset}.pkl."""
    row_dir = (embeddings_base or project_root / 'embeddings') / 'row'
    overlay_dir = resolve_optional_root(project_root, overlay_root)
    strict_models = strict_overlay_models or set()

    if not row_dir.exists() and not (overlay_dir and overlay_dir.exists()):
        return {}

    candidate_models = set(models or [])
    for root in [row_dir, overlay_dir]:
        if not root or not root.exists():
            continue
        for model_dir in sorted(root.iterdir()):
            if model_dir.is_dir():
                candidate_models.add(model_dir.name)

    available = {}
    for model_name in sorted(candidate_models):
        if models and model_name not in models:
            continue
        datasets = []
        seen = set()
        for root in _row_model_roots(row_dir, overlay_dir, strict_models, model_name):
            model_dir = root / model_name
            if not model_dir.is_dir():
                continue
            for p in sorted(model_dir.glob('*.pkl')):
                if p.name.startswith('.') or _is_shard_or_checkpoint(p.name):
                    continue
                if p.stem in seen:
                    continue
                seen.add(p.stem)
                datasets.append(p.stem)
        if datasets:
            available[model_name] = datasets
    return available


def validate_dataset_for_task(
    task_name: str,
    dataset_name: str,
    project_root: Path,
    task_datasets_config: dict,
    embeddings_base: Path | None = None,
) -> tuple[bool, str]:
    """
    Validate that required dataset files exist for a task.

    Returns:
        (is_valid, message)
    """
    task_config = task_datasets_config.get('task_datasets', {}).get(task_name, {})
    dataset_config = task_config.get('datasets', {}).get(dataset_name)

    if not dataset_config:
        return False, f"Dataset '{dataset_name}' not configured for task '{task_name}'"

    required_keys = TASK_REQUIRED_FILES.get(task_name, [])
    for key in required_keys:
        if key not in dataset_config or not dataset_config[key]:
            return False, f"Missing required key '{key}' in config for {task_name}/{dataset_name}"
        rel_path = dataset_config[key]
        # Remap embedding paths when embeddings_base is provided
        if embeddings_base and rel_path.startswith('embeddings/'):
            file_path = embeddings_base / rel_path[len('embeddings/'):]
        else:
            file_path = project_root / rel_path
        # Some keys point to directories rather than files
        if key.endswith('_dir') or key.endswith('_base'):
            if not file_path.is_dir():
                return False, f"{key} directory not found: {file_path}"
        else:
            if not file_path.exists():
                return False, f"{key} not found: {file_path}"

    return True, "OK"


def generate_script(
    task_name: str,
    model_name: str,
    dataset_name: str,
    tasks_config: dict,
    task_datasets_config: dict,
    project_root: Path,
    template_content: str,
    variant: str | None = None,
    seed: int | None = None,
    head_type_override: str | None = None,
    retrieval_mode: str | None = None,
    statement_only: bool = False,
    embeddings_base: Path | None = None,
    results_base_dir: str | None = None,
    query_encoder: str | None = None,
    row_embedding_root: str | Path | None = None,
    strict_row_overlay_models: set[str] | None = None,
    result_tag: str | None = None,
    split_protocol: str = 'legacy',
) -> str:
    """
    Generate a single sbatch script for a task-model-dataset combination.

    Args:
        task_name: Name of the downstream task
        model_name: Name of the model
        dataset_name: Name of the dataset
        tasks_config: Task configuration dict
        task_datasets_config: Task-dataset configuration dict
        project_root: Project root path
        template_content: SBATCH template content
        variant: Table embedding variant (e.g. 'column_mean', 'cls_embedding').
                 None for column-level tasks or default behavior.

    Returns:
        Generated script content
    """
    task_config = tasks_config['tasks'][task_name]
    # Allow per-dataset resource profile override
    _ds_overrides = task_config.get('dataset_overrides', {}).get(dataset_name, {})
    resource_profile = _ds_overrides.get('resource_profile', task_config.get('resource_profile', 'MEDIUM'))
    resources = dict(tasks_config['resource_profiles'][resource_profile])
    # Non-MLP probes (linear, dummy, cosine_threshold) are CPU-only — drop GPU
    # but keep the original time/memory limits.
    effective_head = head_type_override or task_config.get('defaults', {}).get('head_type', 'mlp')
    if effective_head in ('linear', 'dummy', 'cosine_threshold') and task_name in PROBE_TASKS:
        resources['gpus'] = None
        resources['partition'] = tasks_config['resource_profiles']['LIGHT']['partition']
    slurm_defaults = tasks_config.get('slurm_defaults', {})

    # Build extra SLURM directives
    extra_directives = []
    if slurm_defaults.get('account'):
        extra_directives.append(f"#SBATCH --account={slurm_defaults['account']}")
    if slurm_defaults.get('mail_user'):
        extra_directives.append(f"#SBATCH --mail-type={slurm_defaults.get('mail_type', 'FAIL')}")
        extra_directives.append(f"#SBATCH --mail-user={slurm_defaults['mail_user']}")

    # Build GPU directive — either full #SBATCH line or a comment
    gpus = resources.get('gpus')
    local.append_extra_slurm_directives(extra_directives, gpu_requested=bool(gpus))
    if gpus:
        gpu_directive = f'#SBATCH --gres=gpu:{gpus}'
    else:
        gpu_directive = '# No GPU requested (CPU-only job)'

    # Paths — select embedding directory by task type
    emb_base = embeddings_base or project_root / 'embeddings'
    effective_result_tag = result_tag if task_name in ROW_EMBEDDING_TASKS else None
    result_tag_suffix = f"_{effective_result_tag}" if effective_result_tag else ""
    if task_name in ROW_EMBEDDING_TASKS:
        embeddings_path = resolve_row_embedding_path(
            project_root,
            model_name,
            dataset_name,
            embeddings_base=embeddings_base,
            overlay_root=row_embedding_root,
            strict_overlay_models=strict_row_overlay_models,
        )
    elif task_name in TABLE_EMBEDDING_TASKS:
        embeddings_path = emb_base / 'table' / model_name / f'{dataset_name}.pkl'
    else:
        embeddings_path = emb_base / 'column' / model_name / f'{dataset_name}.pkl'
    logs_dir = project_root / 'slurm' / 'logs' / 'downstream'
    status_file = logs_dir / 'status' / 'job_status.json'
    results_base = results_base_dir or tasks_config.get('results', {}).get('base_dir', 'results/evaluation')
    results_dir = project_root / results_base / task_name / model_name
    if effective_result_tag:
        results_dir = results_dir / effective_result_tag
    seed_suffix = f"_seed{seed}" if seed is not None else ""
    results_file = results_dir / f'{model_name}_{dataset_name}{seed_suffix}.json'

    # Task-specific paths and defaults (with per-dataset overrides)
    task_defaults = dict(task_config.get('defaults', {}))
    dataset_overrides = task_config.get('dataset_overrides', {}).get(dataset_name, {})
    task_defaults.update(dataset_overrides)

    # Variant-aware naming and paths
    variant_suffix = f"_{variant}" if variant else ""

    if variant:
        results_dir = project_root / results_base / task_name / model_name / variant
        results_file = results_dir / f'{model_name}_{dataset_name}{seed_suffix}.json'

    # Base substitutions
    substitutions = {
        'JOB_NAME': local.job_name(f"ds_{OUTPUT_SUBDIRS[task_name][:8]}_{model_name}_{dataset_name}{variant_suffix}{result_tag_suffix}{seed_suffix}{'_' + effective_head if effective_head != 'mlp' else ''}"),
        'SEED': str(seed) if seed is not None else '',
        'MODEL': model_name,
        'DATASET': dataset_name,
        'TIME_LIMIT': resources['time_limit'],
        'MEMORY': resources['memory'],
        'CPUS': str(resources['cpus']),
        'PARTITION': resources['partition'],
        'GPU_DIRECTIVE': gpu_directive,
        'EXTRA_SLURM_DIRECTIVES': '\n'.join(extra_directives) if extra_directives else '',
        'PROJECT_ROOT': str(project_root),
        'EMBEDDINGS_PATH': str(embeddings_path),
        'STATUS_FILE': str(status_file),
        'RESULTS_DIR': str(results_dir),
        'RESULTS_FILE': str(results_file),
        'LOG_DIR': str(logs_dir),
        'TIMESTAMP': datetime.now().isoformat(),
        'ENV_SETUP': local.env_setup('source load_env'),
        'SCRIPT_PATH': task_config['script'],
        'VARIANT_SUFFIX': variant_suffix,
        'RESULT_TAG': effective_result_tag or '',
        'RESULT_TAG_SUFFIX': result_tag_suffix,
        'QENC_SUFFIX': '',
        'MODE_SUFFIX': '',
        'HEAD_TYPE': str(head_type_override or task_defaults.get('head_type', 'mlp')),
        'HEAD_SUFFIX': f"_{effective_head}" if effective_head != 'mlp' and task_name in PROBE_TASKS else '',
    }

    # Task-specific substitutions
    if task_name == 'column_clustering':
        task_ds_config = task_datasets_config['task_datasets'][task_name]['datasets'].get(dataset_name, {})
        labels_path = project_root / task_ds_config.get('labels_file', f'datasets/{dataset_name}/all.csv')
        substitutions.update({
            'LABELS_PATH': str(labels_path),
            'K': str(task_defaults.get('k', 20)),
            'TARGET_AVG_SIZE': str(task_defaults.get('target_avg_size', 50)),
            'BATCH_SIZE': str(task_defaults.get('batch_size', 4096)),
            'MIN_COVERAGE': str(task_defaults.get('min_coverage', 0.95)),
        })

    elif task_name == 'column_relation_prediction':
        dataset_dir = project_root / 'data' / dataset_name
        output_dir = results_dir / dataset_name / f'seed{seed}'
        substitutions.update({
            'DATASET_DIR': str(dataset_dir),
            'OUTPUT_DIR': str(output_dir),
            'EPOCHS': str(task_defaults.get('epochs', 20)),
            'BATCH_SIZE': str(task_defaults.get('batch_size', 32)),
            'LEARNING_RATE': str(task_defaults.get('lr', 0.001)),
            'HIDDEN_DIM': str(task_defaults.get('hidden_dim', 256)),
        })

    elif task_name == 'column_type_prediction':
        output_dir = results_dir / dataset_name / f'seed{seed}'
        substitutions.update({
            'DATASET_NAME': dataset_name,
            'OUTPUT_DIR': str(output_dir),
            'NUM_EPOCHS': str(task_defaults.get('num_epochs', 10)),
            'BATCH_SIZE': str(task_defaults.get('batch_size', 20)),
            'LEARNING_RATE': str(task_defaults.get('learning_rate', 0.0005)),
        })

    elif task_name == 'join_search':
        task_ds_config = task_datasets_config['task_datasets'][task_name]['datasets'][dataset_name]
        query_list = project_root / task_ds_config['query_list']
        ground_truth = project_root / task_ds_config['ground_truth']
        output_csv = results_dir / f'{model_name}_{dataset_name}_results.csv'
        substitutions.update({
            'QUERY_LIST': str(query_list),
            'GROUND_TRUTH': str(ground_truth),
            'OUTPUT_CSV': str(output_csv),
            'K': str(task_defaults.get('k', 50)),
        })

    elif task_name == 'join_search_learned':
        task_ds_config = task_datasets_config['task_datasets'][task_name]['datasets'][dataset_name]
        query_list = project_root / task_ds_config['query_list']
        ground_truth = project_root / task_ds_config['ground_truth']
        split_dir = project_root / task_ds_config['split_dir']
        output_dir = results_dir / dataset_name / f'seed{seed}'
        substitutions.update({
            'QUERY_LIST': str(query_list),
            'GROUND_TRUTH': str(ground_truth),
            'SPLIT_DIR': str(split_dir),
            'OUTPUT_DIR': str(output_dir),
            'K': str(task_defaults.get('k', 50)),
            'NUM_LAYERS': str(task_defaults.get('num_layers', 1)),
            'BATCH_SIZE': str(task_defaults.get('batch_size', 512)),
            'MAX_EPOCHS': str(task_defaults.get('max_epochs', 10)),
            'LEARNING_RATE': str(task_defaults.get('learning_rate', 1e-3)),
            'TEMPERATURE': str(task_defaults.get('temperature', 0.07)),
            'DROPOUT': str(task_defaults.get('dropout', 0.1)),
        })

    elif task_name in ('join_classification', 'join_containment', 'table_subset', 'union_classification', 'union_regression'):
        task_ds_config = task_datasets_config['task_datasets'][task_name]['datasets'][dataset_name]
        if split_protocol == 'strict' and 'labels_file_strict' in task_ds_config:
            labels_path = project_root / task_ds_config['labels_file_strict']
        else:
            labels_path = project_root / task_ds_config['labels_file']
        output_dir = results_dir / f'seed{seed}'
        embedding_type = str(task_defaults['embedding_type'])
        if variant:
            embedding_type = RUN_TASK_VARIANT_MAP[variant]
        substitutions.update({
            'LABELS_PATH': str(labels_path),
            'OUTPUT_DIR': str(output_dir),
            'TASK_TYPE': str(task_defaults['task_type']),
            'EMBEDDING_TYPE': embedding_type,
            'COMBINATION_METHOD': str(task_defaults['combination_method']),
            'HIDDEN_DIM': str(task_defaults['hidden_dim']),
            'NUM_LABELS': str(task_defaults['num_labels']),
            'BATCH_SIZE': str(task_defaults['batch_size']),
            'MAX_EPOCHS': str(task_defaults['max_epochs']),
            'LEARNING_RATE': str(task_defaults['learning_rate']),
            'DROPOUT_PROB': str(task_defaults['dropout_prob']),
            'CONFIG_PATH': str(project_root / 'configs' / 'downstream' / f'{task_name}.yaml'),
        })

    elif task_name == 'record_linkage':
        task_ds_config = task_datasets_config['task_datasets'][task_name]['datasets'][dataset_name]
        if split_protocol == 'strict' and 'labels_file_strict' in task_ds_config:
            labels_path = project_root / task_ds_config['labels_file_strict']
        else:
            labels_path = project_root / task_ds_config['labels_file']
        output_dir = results_dir / dataset_name / f'seed{seed}'
        substitutions.update({
            'LABELS_PATH': str(labels_path),
            'OUTPUT_DIR': str(output_dir),
            'COMBINATION_METHOD': str(task_defaults.get('combination_method', 'concat')),
            'HIDDEN_DIM': str(task_defaults.get('hidden_dim', 256)),
            'NUM_LABELS': str(task_defaults.get('num_labels', 2)),
            'BATCH_SIZE': str(task_defaults.get('batch_size', 64)),
            'MAX_EPOCHS': str(task_defaults.get('max_epochs', 50)),
            'LEARNING_RATE': str(task_defaults.get('learning_rate', 1e-3)),
            'DROPOUT_PROB': str(task_defaults.get('dropout_prob', 0.1)),
            'CONFIG_PATH': str(project_root / 'configs' / 'downstream' / 'record_linkage.yaml'),
        })

    elif task_name == 'schema_matching':
        task_ds_config = task_datasets_config['task_datasets'][task_name]['datasets'][dataset_name]
        pairs_path = project_root / task_ds_config['pairs_file']
        ground_truth_path = project_root / task_ds_config['ground_truth']
        tables_dir_path = project_root / task_ds_config['tables_dir']
        substitutions.update({
            'PAIRS_PATH': str(pairs_path),
            'GROUND_TRUTH_PATH': str(ground_truth_path),
            'TABLES_DIR': str(tables_dir_path),
            'MATCHING_STRATEGY': str(task_defaults.get('matching_strategy', 'hungarian')),
            'THRESHOLD': str(task_defaults.get('threshold', 0.0)),
        })

    elif task_name == 'union_search':
        task_ds_config = task_datasets_config['task_datasets'][task_name]['datasets'][dataset_name]
        groundtruth_path = project_root / task_ds_config['groundtruth_file']
        substitutions.update({
            'GROUNDTRUTH_PATH': str(groundtruth_path),
            'METHOD': str(task_defaults.get('method', 'hnsw')),
            'K': str(task_defaults.get('K', 10)),
            'THRESHOLD': str(task_defaults.get('threshold', 0.7)),
            'EF': str(task_defaults.get('ef', 100)),
            'N': str(task_defaults.get('N', 100)),
        })

    elif task_name == 'table_fact_verification':
        task_ds_config = task_datasets_config['task_datasets'][task_name]['datasets'][dataset_name]
        stmt_emb_dir = project_root / task_ds_config['statement_embeddings_dir']
        # Statement-only mode: adjust paths and suffixes
        mode_suffix = '_stmtonly' if statement_only else ''
        if statement_only:
            results_dir = results_dir / 'statement_only'
        output_dir = results_dir / f'seed{seed}'
        effective_variant_suffix = variant_suffix + mode_suffix
        table_emb_variant = str(task_defaults['table_embedding_variant'])
        if variant:
            table_emb_variant = TABFACT_VARIANT_MAP[variant]
        substitutions.update({
            'JOB_NAME': local.job_name(f"ds_{OUTPUT_SUBDIRS[task_name][:8]}_{model_name}_{dataset_name}{effective_variant_suffix}{seed_suffix}"),
            'VARIANT_SUFFIX': effective_variant_suffix,
            'STATEMENT_ONLY': 'true' if statement_only else 'false',
            'TABLE_EMBEDDINGS_PATH': str(embeddings_path),
            'TRAIN_STATEMENT_EMBEDDINGS': str(stmt_emb_dir / 'statements_train.pkl'),
            'VAL_STATEMENT_EMBEDDINGS': str(stmt_emb_dir / 'statements_validation.pkl'),
            'TEST_STATEMENT_EMBEDDINGS': str(stmt_emb_dir / 'statements_test.pkl'),
            'TRAIN_LABELS_JSON': str(project_root / task_ds_config['train_labels']),
            'VAL_LABELS_JSON': str(project_root / task_ds_config['val_labels']),
            'TEST_LABELS_JSON': str(project_root / task_ds_config['test_labels']),
            'OUTPUT_DIR': str(output_dir),
            'RESULTS_DIR': str(results_dir),
            'RESULTS_FILE': str(results_dir / f'{model_name}_{dataset_name}{seed_suffix}.json'),
            'MODEL_TYPE': str(task_defaults['model_type']),
            'TABLE_EMBEDDING_VARIANT': table_emb_variant,
            'HIDDEN_DIM': str(task_defaults['hidden_dim']),
            'DROPOUT': str(task_defaults['dropout']),
            'EPOCHS': str(task_defaults['epochs']),
            'BATCH_SIZE': str(task_defaults['batch_size']),
            'LR': str(task_defaults['lr']),
            'COMBINE_METHOD': str(task_defaults['combine_method']),
            'EVALUATE_SCRIPT_PATH': task_config['evaluate_script'],
        })

    elif task_name == 'table_retrieval':
        task_ds_config = task_datasets_config['task_datasets'][task_name]['datasets'][dataset_name]
        # Resolve query embedding dir from base + encoder name
        q_emb_base_rel = task_ds_config['query_embeddings_base']
        qenc = query_encoder or 'sentence_t5'
        qenc_suffix = f"_{qenc}"
        if embeddings_base and q_emb_base_rel.startswith('embeddings/'):
            q_emb_dir = emb_base / q_emb_base_rel[len('embeddings/'):] / qenc
        else:
            q_emb_dir = project_root / q_emb_base_rel / qenc
        # Use query encoder's table embeddings for hybrid (instead of bert)
        qenc_table_emb = emb_base / 'table' / qenc / f'{dataset_name}.pkl'
        hybrid_model_name = model_name if model_name.endswith(f'_{qenc}_hybrid') else f'{model_name}_{qenc}_hybrid'
        # Resolve retrieval mode (BERT is always model_only)
        effective_mode = retrieval_mode or 'hybrid'
        is_model_only = (effective_mode == 'model_only')
        # Variant-specific hybrid and checkpoint paths
        if variant:
            hybrid_table_emb = emb_base / 'table' / hybrid_model_name / variant / f'{dataset_name}.pkl'
            ckpt_base = project_root / results_base / 'checkpoints' / 'table_retrieval' / f'{model_name}_{dataset_name}' / variant
        else:
            hybrid_table_emb = emb_base / 'table' / hybrid_model_name / f'{dataset_name}.pkl'
            ckpt_base = project_root / results_base / 'checkpoints' / 'table_retrieval' / f'{model_name}_{dataset_name}'
        # Insert query encoder and mode subdirectories to prevent collisions
        ckpt_base = ckpt_base / qenc
        results_dir = results_dir / qenc
        if is_model_only:
            ckpt_base = ckpt_base / 'model_only'
            results_dir = results_dir / 'model_only'
        checkpoint_dir = ckpt_base / f'seed{seed}'
        output_dir = results_dir / f'seed{seed}'
        embedding_variant = TABLE_RETRIEVAL_VARIANT_MAP.get(variant, '') if variant else ''
        # Extend variant suffix and job name for model_only
        mode_suffix = '_modelonly' if is_model_only else ''
        substitutions.update({
            'JOB_NAME': local.job_name(f"ds_{OUTPUT_SUBDIRS[task_name][:8]}_{model_name}_{dataset_name}{variant_suffix}{qenc_suffix}{mode_suffix}{seed_suffix}"),
            'VARIANT_SUFFIX': variant_suffix,
            'QENC_SUFFIX': qenc_suffix,
            'MODE_SUFFIX': mode_suffix,
            'RETRIEVAL_MODE': effective_mode,
            'RESULTS_DIR': str(results_dir),
            'TRAIN_QUERY_EMBEDDINGS': str(q_emb_dir / 'queries_train.pkl'),
            'DEV_QUERY_EMBEDDINGS': str(q_emb_dir / 'queries_dev.pkl'),
            'TEST_QUERY_EMBEDDINGS': str(q_emb_dir / 'queries_test.pkl'),
            'TRAIN_QUESTIONS': str(project_root / task_ds_config['train_questions']),
            'DEV_QUESTIONS': str(project_root / task_ds_config['dev_questions']),
            'TEST_QUESTIONS': str(project_root / task_ds_config['test_questions']),
            'TABLES_JSON': str(project_root / task_ds_config['tables_json']),
            'TABLE_ID_MAPPING': str(project_root / task_ds_config['table_id_mapping']),
            'BERT_TABLE_EMBEDDINGS': str(qenc_table_emb),
            'HYBRID_TABLE_EMBEDDINGS': str(hybrid_table_emb),
            'CHECKPOINT_DIR': str(checkpoint_dir),
            'OUTPUT_DIR': str(output_dir),
            'RESULTS_FILE': str(results_dir / f'{model_name}_{dataset_name}_{qenc}{seed_suffix}.json'),
            'EPOCHS': str(task_defaults.get('epochs', 60)),
            'BATCH_SIZE': str(task_defaults.get('batch_size', 512)),
            'LEARNING_RATE': str(task_defaults.get('lr', 1.28e-3)),
            'PROJECTION_DIM': str(task_defaults.get('projection_dim', 256)),
            'HIDDEN_DIM': str(task_defaults.get('hidden_dim', 512)),
            'EVALUATE_SCRIPT_PATH': task_config['evaluate_script'],
            'EMBEDDING_VARIANT': embedding_variant,
            'BERT_VARIANT': 'column_mean',  # task default; not derived from EMBEDDING_VARIANT
            'NUM_LAYERS': str(task_defaults.get('num_layers', 1)),
            'ADAPTER_FLAG': '--no_adapter' if task_defaults.get('no_adapter', True) else '',
            'SIMILARITY_FN': task_defaults.get('similarity_fn', 'cosine'),
            'TEMPERATURE': str(task_defaults.get('temperature', 0.1)),
        })

    elif task_name == 'semantic_parsing':
        task_ds_config = task_datasets_config['task_datasets'][task_name]['datasets'][dataset_name]
        # Resolve question embedding dir from base + encoder name
        q_emb_base_rel = task_ds_config['question_embeddings_base']
        qenc = query_encoder or 'sentence_t5'
        qenc_suffix = f"_{qenc}"
        if embeddings_base and q_emb_base_rel.startswith('embeddings/'):
            q_emb_dir = emb_base / q_emb_base_rel[len('embeddings/'):] / qenc
        else:
            q_emb_dir = project_root / q_emb_base_rel / qenc
        sem_results_dir = results_dir / qenc
        output_dir = sem_results_dir / f'seed{seed}'
        substitutions.update({
            'JOB_NAME': local.job_name(f"ds_{OUTPUT_SUBDIRS[task_name][:8]}_{model_name}_{dataset_name}{qenc_suffix}{seed_suffix}"),
            'QENC_SUFFIX': qenc_suffix,
            'COLUMN_EMBEDDINGS_PATH': str(embeddings_path),
            'QUESTION_EMBEDDINGS_TRAIN': str(q_emb_dir / 'questions_train.pkl'),
            'QUESTION_EMBEDDINGS_DEV': str(q_emb_dir / 'questions_dev.pkl'),
            'QUESTION_EMBEDDINGS_TEST': str(q_emb_dir / 'questions_test.pkl'),
            'DATASET_PATH': str(project_root / task_ds_config['dataset_dir']),
            'CONFIG_PATH': str(project_root / task_ds_config['config_file']),
            'TEST_FILE': str(project_root / task_ds_config['test_file']),
            'TABLE_FILE': str(project_root / task_ds_config['table_file']),
            'OUTPUT_DIR': str(output_dir),
            'RESULTS_FILE': str(sem_results_dir / f'{model_name}_{dataset_name}_{qenc}{seed_suffix}.json'),
            'SEED': str(seed),
            'BEAM_SIZE': str(task_defaults.get('beam_size', 5)),
        })

    # Adjust output paths for non-MLP probes: append /{head_type} so
    # results.json and canonical RESULTS_FILE don't collide across head types.
    # Only apply to tasks that actually support --head_type (PROBE_TASKS).
    head_type = substitutions.get('HEAD_TYPE', 'mlp')
    if head_type in ('linear', 'dummy', 'cosine_threshold', 'interaction') and task_name in PROBE_TASKS and 'OUTPUT_DIR' in substitutions:
        substitutions['OUTPUT_DIR'] = str(Path(substitutions['OUTPUT_DIR']) / head_type)
        if 'RESULTS_DIR' in substitutions:
            substitutions['RESULTS_DIR'] = str(Path(substitutions['RESULTS_DIR']) / head_type)
        if 'RESULTS_FILE' in substitutions:
            rf = Path(substitutions['RESULTS_FILE'])
            substitutions['RESULTS_FILE'] = str(rf.parent / head_type / rf.name)

    # Adjust output paths for strict split protocol: append /strict to prevent
    # collisions between strict and legacy results
    if split_protocol == 'strict' and 'OUTPUT_DIR' in substitutions:
        task_ds = task_datasets_config.get('task_datasets', {}).get(task_name, {}).get('datasets', {}).get(dataset_name, {})
        if 'labels_file_strict' in task_ds:
            substitutions['OUTPUT_DIR'] = str(Path(substitutions['OUTPUT_DIR']) / 'strict')
            if 'RESULTS_DIR' in substitutions:
                substitutions['RESULTS_DIR'] = str(Path(substitutions['RESULTS_DIR']) / 'strict')
            if 'RESULTS_FILE' in substitutions:
                substitutions['RESULTS_FILE'] = str(
                    Path(substitutions['RESULTS_DIR']) / Path(substitutions['RESULTS_FILE']).name
                )

    # Apply substitutions
    result = template_content
    for key, value in substitutions.items():
        result = result.replace(f'${{{key}}}', str(value))

    return result


def generate_dlte_script(
    task_name: str,
    tasks_config: dict,
    project_root: Path,
    template_content: str,
    col_model: str,
    row_model: str | None = None,
    variant: str | None = None,
    table_model: str | None = None,
) -> str:
    """Generate a single sbatch script for a DLTE pipeline stage."""
    task_config = tasks_config['tasks'][task_name]
    resource_profile = task_config.get('resource_profile', 'LIGHT')
    resources = tasks_config['resource_profiles'][resource_profile]
    slurm_defaults = tasks_config.get('slurm_defaults', {})
    task_defaults = task_config.get('defaults', {})

    logs_dir = project_root / 'slurm' / 'logs' / 'downstream'
    status_file = logs_dir / 'status' / 'job_status.json'
    variant_suffix = f"_{variant}" if variant else ""
    dlte_results_base = tasks_config.get('results', {}).get('base_dir', 'results/evaluation')
    if variant and variant != 'column_mean':
        output_root = project_root / dlte_results_base / 'dlte' / variant
    else:
        output_root = project_root / dlte_results_base / 'dlte'

    table_prefix = f"{table_model}__" if table_model and table_model != col_model else ""

    extra_directives = []
    if slurm_defaults.get('account'):
        extra_directives.append(f"#SBATCH --account={slurm_defaults['account']}")
    if slurm_defaults.get('mail_user'):
        extra_directives.append(f"#SBATCH --mail-type={slurm_defaults.get('mail_type', 'FAIL')}")
        extra_directives.append(f"#SBATCH --mail-user={slurm_defaults['mail_user']}")
    local.append_extra_slurm_directives(
        extra_directives,
        gpu_requested=bool(resources.get('gpus')),
    )

    substitutions = {
        'TIME_LIMIT': resources['time_limit'],
        'MEMORY': resources['memory'],
        'CPUS': str(resources['cpus']),
        'PARTITION': resources['partition'],
        'EXTRA_SLURM_DIRECTIVES': '\n'.join(extra_directives) if extra_directives else '',
        'PROJECT_ROOT': str(project_root),
        'OUTPUT_ROOT': str(output_root),
        'STATUS_FILE': str(status_file),
        'LOG_DIR': str(logs_dir),
        'TIMESTAMP': datetime.now().isoformat(),
        'ENV_SETUP': local.env_setup('source load_env'),
        'SCRIPT_PATH': task_config['script'],
        'TABLE_VARIANT': variant or 'column_mean',
        'VARIANT_SUFFIX': variant_suffix,
        'TABLE_MODEL': table_model or '',
    }

    iteration_mode = task_config.get('iteration_mode', 'column_only')

    if iteration_mode == 'column_only':
        stage_tag = task_name.replace('dlte_', '')
        substitutions.update({
            'MODEL': col_model,
            'JOB_NAME': local.job_name(f"dlte_{stage_tag}_{table_prefix}{col_model}{variant_suffix}"),
            'TOPK': task_defaults.get('topk', '100'),
        })
    else:  # col_x_row
        stage_tag = task_name.replace('dlte_', '')
        substitutions.update({
            'COL_MODEL': col_model,
            'ROW_MODEL': row_model,
            'JOB_NAME': local.job_name(f"dlte_{stage_tag}_{table_prefix}{col_model}_{row_model}{variant_suffix}"),
            'SPLITS': task_defaults.get('splits', 'dev test train'),
        })

    result = template_content
    for key, value in substitutions.items():
        result = result.replace(f'${{{key}}}', str(value))

    return result


def _get_dlte_model_variants(col_model, table_variants):
    """Get supported table variants for a DLTE column model.

    Args:
        col_model: Model name.
        table_variants: None if discovery was skipped (legacy mode),
                        or dict {model: [variants]} from discovery.

    Returns list of variant strings, or empty list if model should be
    skipped (no discoverable variants).
    """
    if table_variants is None:
        return [None]
    model_variants = table_variants.get(col_model)
    if model_variants is None:
        return []
    return [v for v in model_variants if v in DLTE_VARIANT_MAP]


def _emit_script(task_name, tasks_config, project_root, template_content,
                  output_dir, task_subdir, script_name,
                  col_model, variant, dry_run, verbose,
                  row_model=None, table_model=None):
    """Emit a single DLTE sbatch script (or print dry-run). Returns 1 if generated."""
    script_path = output_dir / task_subdir / script_name
    if dry_run:
        print(f"    [DRY-RUN] Would generate: {task_subdir}/{script_name}")
    else:
        content = generate_dlte_script(
            task_name, tasks_config, project_root,
            template_content, col_model=col_model,
            row_model=row_model, variant=variant,
            table_model=table_model,
        )
        with open(script_path, 'w') as f:
            f.write(content)
        os.chmod(script_path, 0o755)
        if verbose:
            print(f"    Generated: {task_subdir}/{script_name}")
        else:
            print(f"    {script_name}")
    return 1


def generate_dlte_scripts(
    dlte_tasks: list[str],
    tasks_config: dict,
    project_root: Path,
    templates_dir: Path,
    output_dir: Path,
    dlte_embeddings: dict,
    dry_run: bool = False,
    verbose: bool = False,
    model_filter: list[str] | None = None,
    table_variants: dict[str, list[str]] | None = None,
    include_redundant_variants: bool = False,
) -> tuple[int, int]:
    """
    Generate scripts for DLTE pipeline stages.

    Handles two iteration modes:
      - column_only: one script per col_model × variant (+ cross-model)
      - col_x_row: one script per col_model × variant × row_model (+ cross-model)

    model_filter controls which column models get coupled scripts.
    All table models are always available for cross-model pairing.
    Stage 1 (dlte_retrieval) is never filtered by model_filter.
    """
    generated = 0
    skipped = 0
    all_table_models = list(dlte_embeddings['table_column'])

    for task_name in sorted(dlte_tasks):
        task_config = tasks_config['tasks'][task_name]
        template_file = TEMPLATE_FILES.get(task_name)

        if not template_file:
            print(f"  SKIP {task_name}: No template configured")
            continue

        template_path = templates_dir / template_file
        if not template_path.exists():
            print(f"  SKIP {task_name}: Template not found: {template_path}")
            continue

        with open(template_path, 'r') as f:
            template_content = f.read()

        iteration_mode = task_config.get('iteration_mode', 'column_only')
        task_subdir = OUTPUT_SUBDIRS[task_name]

        if not dry_run:
            (output_dir / task_subdir).mkdir(parents=True, exist_ok=True)

        # Determine available col models for this stage
        if task_name == 'dlte_retrieval':
            col_models = list(dlte_embeddings['table_column'])
        else:
            col_models = list(dlte_embeddings['column'])

        # model_filter applies to column models only, NOT retrieval (Stage 1)
        if model_filter and task_name != 'dlte_retrieval':
            col_models = [m for m in col_models if m in model_filter]

        print(f"\n{task_name} (iteration_mode={iteration_mode}):")

        if not col_models:
            print(f"  No column models available")
            continue

        # Clean stale scripts (coupled + cross-model)
        if not dry_run:
            task_dir = output_dir / task_subdir
            for col_model in col_models:
                # Legacy unsuffixed: e.g. bert.sbatch
                legacy = task_dir / f'{col_model}.sbatch'
                if legacy.exists():
                    legacy.unlink()
                # Suffixed (variant and/or row_model): e.g. bert_column_mean.sbatch, bert_dae_cls_embedding.sbatch
                for stale in task_dir.glob(f'{col_model}_*.sbatch'):
                    stale.unlink()
            # Clean cross-model scripts (pattern: {table_model}__{col_model}*.sbatch)
            for table_model in all_table_models:
                for col_model in col_models:
                    if table_model == col_model:
                        continue
                    for stale in task_dir.glob(f'{table_model}__{col_model}*.sbatch'):
                        stale.unlink()

        if iteration_mode == 'column_only':
            print(f"  Column models: {col_models}")

            # ── Coupled scripts (table_model == col_model) ──
            for col_model in col_models:
                supported = _get_dlte_model_variants(col_model, table_variants)
                for variant in supported:
                    variant_str = f"_{variant}" if variant else ""
                    script_name = f"{col_model}{variant_str}.sbatch"
                    generated += _emit_script(
                        task_name, tasks_config, project_root, template_content,
                        output_dir, task_subdir, script_name,
                        col_model, variant, dry_run, verbose)

            # ── Cross-model scripts (table_model != col_model) ──
            if task_name != 'dlte_retrieval':
                for table_model in all_table_models:
                    supported = _get_dlte_model_variants(table_model, table_variants)
                    cross_variants = supported if include_redundant_variants else [v for v in supported if v not in DLTE_REDUNDANT_VARIANTS]
                    for variant in cross_variants:
                        for col_model in col_models:
                            if table_model == col_model:
                                continue
                            variant_str = f"_{variant}" if variant else ""
                            script_name = f"{table_model}__{col_model}{variant_str}.sbatch"
                            generated += _emit_script(
                                task_name, tasks_config, project_root, template_content,
                                output_dir, task_subdir, script_name,
                                col_model, variant, dry_run, verbose,
                                table_model=table_model)

        else:  # col_x_row
            row_models = dlte_embeddings['row']
            if not row_models:
                print(f"  No row models available")
                continue

            print(f"  Column models: {col_models}")
            print(f"  Row models: {row_models}")

            # ── Coupled scripts ──
            for col_model in col_models:
                supported = _get_dlte_model_variants(col_model, table_variants)
                for variant in supported:
                    for row_model in row_models:
                        variant_str = f"_{variant}" if variant else ""
                        script_name = f"{col_model}_{row_model}{variant_str}.sbatch"
                        generated += _emit_script(
                            task_name, tasks_config, project_root, template_content,
                            output_dir, task_subdir, script_name,
                            col_model, variant, dry_run, verbose,
                            row_model=row_model)

            # ── Cross-model scripts ──
            for table_model in all_table_models:
                supported = _get_dlte_model_variants(table_model, table_variants)
                cross_variants = supported if include_redundant_variants else [v for v in supported if v not in DLTE_REDUNDANT_VARIANTS]
                for variant in cross_variants:
                    for col_model in col_models:
                        if table_model == col_model:
                            continue
                        for row_model in row_models:
                            variant_str = f"_{variant}" if variant else ""
                            script_name = f"{table_model}__{col_model}_{row_model}{variant_str}.sbatch"
                            generated += _emit_script(
                                task_name, tasks_config, project_root, template_content,
                                output_dir, task_subdir, script_name,
                                col_model, variant, dry_run, verbose,
                                row_model=row_model, table_model=table_model)

    return generated, skipped


# ── Embedding-free baseline generation ────────────────────────────────────────

def _count_surviving_baselines(baselines_config, task_filter, model_filter, dataset_filter):
    """Count baseline (baseline, task, dataset) triples surviving all filters."""
    count = 0
    for baseline_name, bl_cfg in baselines_config.get('baselines', {}).items():
        if model_filter and baseline_name not in model_filter:
            continue
        for task_name, task_cfg in bl_cfg.get('tasks', {}).items():
            if task_filter and task_name not in task_filter:
                continue
            for dataset_name in task_cfg.get('datasets', {}):
                if dataset_filter and dataset_name not in dataset_filter:
                    continue
                count += 1
    return count


def generate_baseline_scripts(
    baselines_config: dict,
    tasks_config: dict,
    project_root: Path,
    templates_dir: Path,
    output_dir: Path,
    task_filter: list[str] | None = None,
    model_filter: list[str] | None = None,
    dataset_filter: list[str] | None = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> tuple[int, list[dict]]:
    """Generate sbatch scripts for embedding-free baselines.

    Returns (generated_count, manifest_entries).
    """
    template_path = templates_dir / 'baseline.sbatch.template'
    if not template_path.exists():
        print(f"\n  WARN: Baseline template not found: {template_path}")
        return 0, []

    with open(template_path, 'r') as f:
        template_content = f.read()

    resource_profiles = tasks_config.get('resource_profiles', {})
    slurm_defaults = tasks_config.get('slurm_defaults', {})

    generated = 0
    manifest_entries = []

    for baseline_name, bl_cfg in sorted(baselines_config.get('baselines', {}).items()):
        # Respect --models filter for baselines
        if model_filter and baseline_name not in model_filter:
            continue

        resource_profile = bl_cfg.get('resource_profile', 'LIGHT')
        resources = resource_profiles.get(resource_profile, resource_profiles.get('LIGHT', {}))

        for task_name, task_cfg in sorted(bl_cfg.get('tasks', {}).items()):
            if task_filter and task_name not in task_filter:
                continue

            if task_name not in OUTPUT_SUBDIRS:
                if verbose:
                    print(f"  SKIP baseline {baseline_name}/{task_name}: unknown task")
                continue

            task_subdir = OUTPUT_SUBDIRS[task_name]

            if not dry_run:
                (output_dir / task_subdir).mkdir(parents=True, exist_ok=True)

            script_module = task_cfg.get('script', '')
            invoke_style = task_cfg.get('invoke_style', 'module')

            for dataset_name, ds_cfg in sorted(task_cfg.get('datasets', {}).items()):
                if dataset_filter and dataset_name not in dataset_filter:
                    continue

                script_args = ds_cfg.get('args', '')
                script_name = f"{baseline_name}_{dataset_name}.sbatch"
                script_path = output_dir / task_subdir / script_name

                # Build extra SLURM directives
                extra_directives = []
                if slurm_defaults.get('account'):
                    extra_directives.append(f"#SBATCH --account={slurm_defaults['account']}")

                gpus = resources.get('gpus')
                local.append_extra_slurm_directives(extra_directives, gpu_requested=bool(gpus))
                job_key = f"{baseline_name}_{dataset_name}_{task_name}"

                substitutions = {
                    'JOB_NAME': local.job_name(f"bl_{task_subdir[:8]}_{baseline_name}_{dataset_name}"),
                    'BASELINE_NAME': baseline_name,
                    'DATASET': dataset_name,
                    'TASK': task_name,
                    'TIME_LIMIT': ds_cfg.get('time_limit') or resources.get('time_limit', '02:00:00'),
                    'MEMORY': ds_cfg.get('memory') or resources.get('memory', '32G'),
                    'CPUS': str(resources.get('cpus', 8)),
                    'PARTITION': local.partition(resources.get('partition', 'cpubase_bycore_b2')),
                    'EXTRA_SLURM_DIRECTIVES': '\n'.join(extra_directives) if extra_directives else '',
                    'PROJECT_ROOT': str(project_root),
                    'STATUS_FILE': str(project_root / 'slurm' / 'logs' / 'downstream' / 'status' / 'job_status.json'),
                    'LOG_DIR': str(project_root / 'slurm' / 'logs' / 'downstream'),
                    'TIMESTAMP': datetime.now().isoformat(),
                    'ENV_SETUP': local.env_setup('source load_env'),
                    'SCRIPT_MODULE': script_module,
                    'SCRIPT_ARGS': script_args,
                }

                manifest_entries.append({
                    'task': task_name,
                    'model': baseline_name,
                    'dataset': dataset_name,
                    'variant': None,
                    'seed': None,
                    'script_path': str(script_path),
                    'is_baseline': True,
                    'job_key': job_key,
                })

                if dry_run:
                    print(f"    [DRY-RUN] Would generate: {task_subdir}/{script_name}")
                else:
                    # Perform template substitution
                    result = template_content
                    for key, value in substitutions.items():
                        result = result.replace(f'${{{key}}}', str(value))

                    with open(script_path, 'w') as f:
                        f.write(result)
                    os.chmod(script_path, 0o755)

                    if verbose:
                        print(f"    Generated: {task_subdir}/{script_name}")
                    else:
                        print(f"    {script_name}")

                generated += 1

    return generated, manifest_entries


def write_manifest(output_dir: Path, entries: list[dict], dry_run: bool = False):
    """Write downstream_manifest.json for submit_downstream.py discovery."""
    if dry_run:
        return
    manifest_path = output_dir / 'downstream_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(entries, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Generate SLURM sbatch scripts for downstream task evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--seeds', nargs='+', type=int, default=[42],
                       help='Seeds to generate scripts for (default: [42])')
    parser.add_argument('--tasks', nargs='+',
                       help='Generate scripts only for these tasks')
    parser.add_argument('--models', nargs='+',
                       help='Generate scripts only for these models '
                            '(for DLTE: filters column models only; '
                            'retrieval and cross-model table models are always emitted)')
    parser.add_argument('--datasets', nargs='+',
                       help='Generate scripts only for these datasets')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be generated without writing')
    parser.add_argument('--output-dir',
                       help='Output directory for generated scripts')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--include-redundant-variants', action='store_true',
                       help='Include cosine-equivalent variants in cross-model scripts')
    parser.add_argument('--head-type', type=str, default='mlp',
                       choices=['mlp', 'linear', 'dummy', 'cosine_threshold', 'interaction'],
                       help='Probe type: mlp (default), linear (sklearn), dummy (majority/mean), '
                            'or cosine_threshold (threshold on raw cosine similarity)')
    parser.add_argument('--embeddings-dir', type=str, default=None,
                       help='Base directory containing column/, table/, row/ subdirs '
                            '(default: embeddings)')
    parser.add_argument('--row-embedding-root', type=str, default=None,
                       help='Overlay row embedding root directory (e.g., embeddings/row_dim768). '
                            'Trained row models are read strictly from this root when set; '
                            'fixed/pretrained row models continue to use embeddings/row.')
    parser.add_argument('--result-tag', type=str, default=None,
                       help='Tag appended to row-level result paths and job identifiers '
                            '(default: derived from --row-embedding-root)')
    parser.add_argument('--results-dir', type=str, default=None,
                       help='Base directory for evaluation results '
                            '(default: from tasks.yaml results.base_dir)')
    parser.add_argument('--split-protocol', type=str, default='both',
                       choices=['legacy', 'strict', 'both'],
                       help='Split protocol: legacy (original labels.json), '
                            'strict (table-disjoint labels_strict.json), '
                            'or both (generate scripts for both protocols; default)')

    args = parser.parse_args()

    # Get paths
    project_root = get_project_root()
    config_dir = project_root / 'slurm' / 'config' / 'downstream'
    templates_dir = project_root / 'slurm' / 'scripts' / 'templates' / 'downstream'
    embeddings_base = Path(args.embeddings_dir).resolve() if args.embeddings_dir else None
    results_base_dir = args.results_dir  # relative to project_root, or absolute
    default_row_root = ((embeddings_base or project_root / 'embeddings') / 'row')
    strict_row_overlay_models = discover_trained_row_models(project_root)
    effective_result_tag = derive_result_tag(args.row_embedding_root, default_row_root, args.result_tag)

    # Load configurations
    print("Loading configurations...")
    tasks_config = load_yaml(config_dir / 'tasks.yaml')
    task_datasets_config = load_yaml(config_dir / 'task_datasets.yaml')
    resources_config = load_yaml(project_root / 'slurm' / 'config' / 'resources.yaml')

    # Get output directory from config or args
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / resources_config['paths']['downstream_scripts_dir']

    # Filter tasks
    all_tasks = list(tasks_config['tasks'].keys())
    tasks = args.tasks or all_tasks

    invalid_tasks = set(tasks) - set(all_tasks)
    if invalid_tasks:
        print(f"Error: Unknown tasks: {invalid_tasks}")
        print(f"Available tasks: {all_tasks}")
        sys.exit(1)

    # Split into standard and DLTE tasks
    standard_tasks = [t for t in tasks if t not in DLTE_TASKS]
    dlte_tasks = [t for t in tasks if t in DLTE_TASKS]

    # Discover available embeddings (column, table, and row) for standard tasks
    available_column_embeddings = {}
    available_table_embeddings = {}
    available_row_embeddings = {}
    if standard_tasks:
        # Row embeddings (for row-level tasks like record_linkage)
        row_tasks = [t for t in standard_tasks if t in ROW_EMBEDDING_TASKS]
        if row_tasks:
            print("Discovering available row embeddings...")
            if args.row_embedding_root:
                print(f"  overlay: {resolve_optional_root(project_root, args.row_embedding_root)}")
            available_row_embeddings = discover_row_embeddings(
                project_root,
                args.models,
                embeddings_base,
                overlay_root=args.row_embedding_root,
                strict_overlay_models=strict_row_overlay_models,
            )
            print(f"  Found row embeddings for {len(available_row_embeddings)} models")
            for model, datasets in available_row_embeddings.items():
                print(f"    {model}: {len(datasets)} datasets")
            if effective_result_tag:
                print(f"  Result tag: {effective_result_tag}")

        # Column embeddings (for column-level tasks like clustering, join_search, etc.)
        column_tasks = [t for t in standard_tasks if t not in TABLE_EMBEDDING_TASKS and t not in ROW_EMBEDDING_TASKS]
        if column_tasks:
            print("Discovering available column embeddings...")
            available_column_embeddings = discover_embeddings(project_root, args.models, embeddings_base)
            print(f"  Found column embeddings for {len(available_column_embeddings)} models")
            for model, datasets in available_column_embeddings.items():
                print(f"    {model}: {len(datasets)} datasets")

        # Table embeddings (for table-level tasks)
        table_tasks = [t for t in standard_tasks if t in TABLE_EMBEDDING_TASKS]
        table_variants = {}
        if table_tasks:
            print("Discovering available table embeddings...")
            available_table_embeddings = discover_table_embeddings(project_root, args.models, embeddings_base)
            if available_table_embeddings:
                print(f"  Found table embeddings for {len(available_table_embeddings)} models")
                for model, datasets in available_table_embeddings.items():
                    print(f"    {model}: {len(datasets)} datasets")

                # Discover variant availability for table embeddings
                print("Discovering table embedding variants...")
                table_variants = discover_table_embedding_variants(project_root, args.models, embeddings_base)
                for model, datasets in table_variants.items():
                    sample_variants = next(iter(datasets.values()), [])
                    print(f"    {model}: {sample_variants}")
            else:
                print("  Warning: No table embeddings found in embeddings/table/")
                print("  Run 'python scripts/generate_table_embeddings.py' first")

        # Merged view for backward compat (column embeddings used as fallback)
        available_embeddings = dict(available_column_embeddings)
        for source in [available_table_embeddings, available_row_embeddings]:
            for model, datasets in source.items():
                if model not in available_embeddings:
                    available_embeddings[model] = datasets
                else:
                    available_embeddings[model] = sorted(set(available_embeddings[model]) | set(datasets))

        if not available_embeddings:
            # Check if baselines survive all filters before exiting
            baselines_yaml = config_dir / 'baselines.yaml'
            has_baselines = False
            if baselines_yaml.exists():
                bl_config = load_yaml(baselines_yaml)
                has_baselines = _count_surviving_baselines(
                    bl_config, args.tasks, args.models, args.datasets
                ) > 0
            if not has_baselines:
                print("Error: No embeddings found")
                sys.exit(1)
            print("  No model embeddings found, but baselines are configured")

    template_files = TEMPLATE_FILES
    output_subdirs = OUTPUT_SUBDIRS

    # Create output directories for standard tasks
    if not args.dry_run:
        for task in standard_tasks:
            subdir = output_subdirs.get(task)
            if subdir:
                (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Generate scripts
    print(f"\nGenerating downstream task scripts...")
    print("=" * 60)

    generated = 0
    skipped = 0
    skipped_reasons = {}
    manifest_entries = []

    for task_name in sorted(standard_tasks):
        task_config = tasks_config['tasks'][task_name]
        template_file = template_files.get(task_name)

        if not template_file:
            print(f"  SKIP {task_name}: No template configured")
            continue

        template_path = templates_dir / template_file
        if not template_path.exists():
            print(f"  SKIP {task_name}: Template not found: {template_path}")
            continue

        with open(template_path, 'r') as f:
            template_content = f.read()

        # Determine the effective head type for this task.
        # Non-probe tasks (clustering, search, retrieval, parsing) don't use
        # a head type at all — always generate them with the default config.
        # This ensures the manifest is complete regardless of --head-type.
        if task_name not in PROBE_TASKS:
            effective_head = 'mlp'
        elif args.head_type == 'cosine_threshold' and task_name not in COSINE_THRESHOLD_TASKS:
            print(f"\n{task_name}:")
            print(f"  SKIP: cosine_threshold head type not supported for {task_name}")
            continue
        elif args.head_type == 'interaction' and task_name not in INTERACTION_TASKS:
            print(f"\n{task_name}:")
            print(f"  SKIP: interaction head type not supported for {task_name}")
            continue
        else:
            effective_head = args.head_type

        # Get supported datasets for this task
        supported_datasets = get_task_datasets(task_name, task_datasets_config)

        print(f"\n{task_name}:")
        print(f"  Supported datasets: {supported_datasets}")

        # Select the right embedding source for this task
        if task_name in ROW_EMBEDDING_TASKS:
            task_embeddings = available_row_embeddings
        elif task_name in TABLE_EMBEDDING_TASKS:
            task_embeddings = available_table_embeddings
        else:
            task_embeddings = available_column_embeddings

        # Exclude models listed in task config (e.g., non-neural baselines
        # that only cover a subset of tables and would crash on certain tasks)
        exclude_models = set(task_config.get('exclude_models', []))

        for model_name in sorted(task_embeddings.keys()):
            if model_name in exclude_models:
                if args.verbose:
                    print(f"  SKIP {model_name}: excluded for {task_name}")
                skipped += 1
                continue

            model_datasets = task_embeddings[model_name]

            # Filter by --datasets if specified
            if args.datasets:
                model_datasets = [d for d in model_datasets if d in args.datasets]

            for dataset_name in model_datasets:
                # Check if this dataset is supported for this task
                if dataset_name not in supported_datasets:
                    reason = f"not supported for {task_name}"
                    skipped_reasons[f"{model_name}/{dataset_name}"] = reason
                    skipped += 1
                    continue

                # Validate dataset files exist
                is_valid, msg = validate_dataset_for_task(
                    task_name, dataset_name, project_root, task_datasets_config,
                    embeddings_base=embeddings_base,
                )
                if not is_valid:
                    if args.verbose:
                        print(f"    SKIP {model_name}/{dataset_name}: {msg}")
                    skipped_reasons[f"{model_name}/{dataset_name}"] = msg
                    skipped += 1
                    continue

                # For table-embedding tasks, generate per-variant scripts
                if task_name in TABLE_EMBEDDING_TASKS:
                    model_variants = table_variants.get(model_name, {}).get(
                        dataset_name, ['column_mean']
                    )
                    task_variant_map = _get_variant_map(task_name)
                    supported_variants = [v for v in model_variants if v in task_variant_map]
                    if not supported_variants:
                        supported_variants = ['column_mean']
                else:
                    supported_variants = [None]  # Column tasks: no variant dimension

                # Query encoder loop: table_retrieval and semantic_parsing
                # iterate over configured query encoders
                task_td = task_datasets_config.get('task_datasets', {}).get(task_name, {})
                query_encoders = task_td.get('query_encoders', [None])

                for qenc in query_encoders:
                  qenc_suffix = f"_{qenc}" if qenc else ""

                  for variant in supported_variants:
                    # Retrieval mode loop: table_retrieval generates both hybrid and model_only
                    if task_name == 'table_retrieval':
                        retrieval_modes = ['hybrid', 'model_only']
                    else:
                        retrieval_modes = [None]

                    # Statement-only mode loop: table_fact_verification generates both standard and stmt_only
                    if task_name == 'table_fact_verification':
                        stmt_only_modes = [False, True]
                    else:
                        stmt_only_modes = [False]

                    for retrieval_mode in retrieval_modes:
                      for is_stmt_only in stmt_only_modes:
                        mode_suffix = '_modelonly' if retrieval_mode == 'model_only' else ''
                        mode_suffix += '_stmtonly' if is_stmt_only else ''

                        # Seed loop: deterministic tasks get a single None seed
                        seeds = [None] if task_name in DETERMINISTIC_TASKS else args.seeds
                        for seed in seeds:
                            seed_suffix = f"_seed{seed}" if seed is not None else ""
                            head_suffix = f"_{effective_head}" if effective_head and effective_head != 'mlp' else ""

                            # Split protocol loop: when --split-protocol both,
                            # generate one legacy and one strict script for tasks
                            # that have labels_file_strict; otherwise just legacy.
                            if args.split_protocol == 'both':
                                ds_cfg = task_datasets_config['task_datasets'][task_name]['datasets'][dataset_name]
                                has_strict = 'labels_file_strict' in ds_cfg
                                if has_strict:
                                    strict_path = project_root / ds_cfg['labels_file_strict']
                                    has_strict = strict_path.exists()
                                protocols = ['legacy', 'strict'] if has_strict else ['legacy']
                            else:
                                protocols = [args.split_protocol]

                            for protocol in protocols:
                                proto_suffix = '_strict' if protocol == 'strict' else ''

                                # Generate script — mode_suffix + head_suffix + qenc + proto in filename prevents collisions
                                result_tag_sfx = f"_{effective_result_tag}" if effective_result_tag and task_name in ROW_EMBEDDING_TASKS else ""
                                if variant:
                                    script_name = f"{model_name}_{dataset_name}_{variant}{result_tag_sfx}{qenc_suffix}{mode_suffix}{seed_suffix}{head_suffix}{proto_suffix}.sbatch"
                                else:
                                    script_name = f"{model_name}_{dataset_name}{result_tag_sfx}{qenc_suffix}{mode_suffix}{seed_suffix}{head_suffix}{proto_suffix}.sbatch"
                                script_path = output_dir / output_subdirs[task_name] / script_name

                                if args.dry_run:
                                    print(f"    [DRY-RUN] Would generate: {output_subdirs[task_name]}/{script_name}")
                                else:
                                    script_content = generate_script(
                                        task_name, model_name, dataset_name,
                                        tasks_config, task_datasets_config,
                                        project_root, template_content,
                                        variant=variant,
                                        seed=seed,
                                        head_type_override=effective_head if effective_head != 'mlp' else None,
                                        retrieval_mode=retrieval_mode,
                                        statement_only=is_stmt_only,
                                        embeddings_base=embeddings_base,
                                        results_base_dir=results_base_dir,
                                        query_encoder=qenc,
                                        row_embedding_root=args.row_embedding_root,
                                        strict_row_overlay_models=strict_row_overlay_models,
                                        result_tag=effective_result_tag,
                                        split_protocol=protocol,
                                    )

                                    with open(script_path, 'w') as f:
                                        f.write(script_content)

                                    os.chmod(script_path, 0o755)

                                    if args.verbose:
                                        print(f"    Generated: {output_subdirs[task_name]}/{script_name}")
                                    else:
                                        print(f"    {script_name}")

                                # Compute the concrete job_key for the manifest.
                                # Must match the JOB_KEY in the rendered template.
                                variant_sfx = f"_{variant}" if variant else ""
                                m_suffix = '_modelonly' if retrieval_mode == 'model_only' else ''
                                m_suffix += '_stmtonly' if is_stmt_only else ''
                                row_tag_sfx = f"_{effective_result_tag}" if effective_result_tag and task_name in ROW_EMBEDDING_TASKS else ""
                                head_sfx = f"_{effective_head}" if effective_head != 'mlp' and task_name in PROBE_TASKS else ""
                                if seed is not None:
                                    job_key = f"{model_name}_{dataset_name}_{task_name}{variant_sfx}{row_tag_sfx}{qenc_suffix}{m_suffix}_seed{seed}{head_sfx}{proto_suffix}"
                                else:
                                    job_key = f"{model_name}_{dataset_name}_{task_name}{variant_sfx}{row_tag_sfx}{qenc_suffix}{m_suffix}{head_sfx}{proto_suffix}"

                                manifest_entries.append({
                                    'task': task_name,
                                    'model': model_name,
                                    'dataset': dataset_name,
                                    'variant': variant,
                                    'seed': seed,
                                    'split_protocol': protocol,
                                    'script_path': str(script_path),
                                    'is_baseline': False,
                                    'job_key': job_key,
                                })

                                generated += 1

    # ── DLTE pipeline stages ──────────────────────────────────────
    if dlte_tasks:
        print("\nDiscovering DLTE embeddings...")
        dlte_embeddings = discover_dlte_embeddings(project_root)
        print(f"  Table-level column models (stage 1): {dlte_embeddings['table_column']}")
        print(f"  Column models (stage 2):             {dlte_embeddings['column']}")
        print(f"  Row models (stages 3-4):             {dlte_embeddings['row']}")

        # Discover DLTE-specific table variants (intersects across all 3 DLTE pkls)
        dlte_table_variants = None
        if dlte_embeddings['table_column']:
            print("Discovering DLTE table embedding variants...")
            dlte_table_variants = discover_dlte_table_variants(
                project_root, dlte_embeddings['table_column']
            )
            for model, variants in dlte_table_variants.items():
                print(f"    {model}: {variants}")
            if not dlte_table_variants:
                print("  Warning: No DLTE table variants discovered")

        dlte_gen, dlte_skip = generate_dlte_scripts(
            dlte_tasks, tasks_config, project_root,
            templates_dir, output_dir, dlte_embeddings,
            dry_run=args.dry_run, verbose=args.verbose,
            model_filter=args.models,
            table_variants=dlte_table_variants,
            include_redundant_variants=args.include_redundant_variants,
        )
        generated += dlte_gen
        skipped += dlte_skip

    # ── Embedding-free baselines ──────────────────────────────────
    baselines_yaml = config_dir / 'baselines.yaml'
    if baselines_yaml.exists():
        baselines_config = load_yaml(baselines_yaml)
        if baselines_config.get('baselines'):
            print(f"\nGenerating baseline scripts...")
            baseline_gen, baseline_manifest = generate_baseline_scripts(
                baselines_config, tasks_config,
                project_root, templates_dir, output_dir,
                task_filter=args.tasks,
                model_filter=args.models,
                dataset_filter=args.datasets,
                dry_run=args.dry_run,
                verbose=args.verbose,
            )
            generated += baseline_gen
            manifest_entries.extend(baseline_manifest)
            if baseline_gen:
                print(f"  Generated {baseline_gen} baseline scripts")

    write_manifest(output_dir, manifest_entries, dry_run=args.dry_run)

    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Generated: {generated} scripts")
    if skipped:
        print(f"Skipped:   {skipped} combinations")
        if args.verbose and skipped_reasons:
            print("Skipped reasons:")
            for combo, reason in sorted(skipped_reasons.items()):
                print(f"  {combo}: {reason}")
    if not args.dry_run:
        print(f"Output:    {output_dir}")


if __name__ == '__main__':
    main()
