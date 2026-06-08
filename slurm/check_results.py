#!/usr/bin/env python3
"""
Check completeness of embeddings and downstream evaluation results.

Derives expected outputs from the same YAML configs used by the generators,
scans assets/ on disk, and reports what's missing.

Usage:
    # Check downstream results
    python slurm/check_results.py                  # full report
    python slurm/check_results.py --models gte     # GTE only
    python slurm/check_results.py --missing-only   # only show gaps
    python slurm/check_results.py --head-types mlp linear dummy

    # Check embeddings
    python slurm/check_results.py --embeddings                 # all types
    python slurm/check_results.py --embeddings --types column  # column only
    python slurm/check_results.py --embeddings --models gte bert
    python slurm/check_results.py --embeddings --missing-only

    # Check both
    python slurm/check_results.py --all
"""

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Active external query encoders for retrieval-style downstream tasks.
# Keep this aligned with slurm/config/downstream/task_datasets.yaml.
QUERY_ENCODER_MODELS = {'sentence_t5', 'mpnet'}


def discover_embedding_model_dirs(base_dir: Path, *,
                                  exclude_query_encoders: bool = False,
                                  exclude_hybrids: bool = False) -> list[str]:
    """Discover model directory names from an embeddings subtree."""
    if not base_dir.exists():
        return []

    names = []
    for child in base_dir.iterdir():
        if not child.is_dir() or child.name.startswith('.'):
            continue
        name = child.name
        if exclude_query_encoders and name in QUERY_ENCODER_MODELS:
            continue
        if exclude_hybrids and (name.endswith('_hybrid') or '_backup_' in name):
            continue
        names.append(name)
    return sorted(names)


def resolve_row_data_roots(
    project_root: Path,
    default_root: str | Path,
    overlay_root: str | Path | None = None,
) -> list[Path]:
    """Resolve row-data embedding roots in precedence order."""

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


def resolve_optional_path(project_root: Path, path_like: str | Path | None) -> Path | None:
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


def discover_trained_models(models_cfg: dict) -> set[str]:
    """Return model names whose alternate dimensions should come from an overlay root."""
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

# ── Config loading ────────────────────────────────────────────────────────────

def load_yaml(path):
    with open(PROJECT_ROOT / path) as f:
        return yaml.safe_load(f)


# ── Embedding checking ───────────────────────────────────────────────────────

def check_embeddings(args):
    """Check completeness of embedding generation."""
    cfg_datasets = load_yaml('slurm/config/datasets.yaml')
    cfg_models = load_yaml('slurm/config/models.yaml')
    cfg_row_models = load_yaml('slurm/config/row_models.yaml')
    cfg_row_data_models = load_yaml('slurm/config/row_data_models.yaml')

    datasets = cfg_datasets['datasets']
    datasets_dir = PROJECT_ROOT / 'datasets'

    embeddings_base = PROJECT_ROOT / 'embeddings'
    default_row_root = embeddings_base / 'row'
    row_overlay_root = resolve_optional_path(PROJECT_ROOT, getattr(args, 'row_overlay_root', None))
    row_data_overlay_root = resolve_optional_path(PROJECT_ROOT, getattr(args, 'row_data_overlay_root', None))
    row_trained_models = discover_trained_models(cfg_row_models)
    row_data_trained_models = discover_trained_models(cfg_row_data_models)

    column_models = sorted(
        set(cfg_models['models'].keys()) |
        set(discover_embedding_model_dirs(embeddings_base / 'column'))
    )
    row_models = sorted(
        set(cfg_row_models['models'].keys()) |
        set(discover_embedding_model_dirs(default_row_root)) |
        (set(discover_embedding_model_dirs(row_overlay_root)) if row_overlay_root else set())
    )
    row_data_models = sorted(
        set(cfg_row_data_models['models'].keys()) |
        set(discover_embedding_model_dirs(embeddings_base / 'row_prediction')) |
        (set(discover_embedding_model_dirs(row_data_overlay_root)) if row_data_overlay_root else set())
    )

    # Datasets to skip in embedding completeness checks (experimental,
    # deprecated, or not part of the core evaluation pipeline)
    SKIP_DATASETS = {
        # Shuffled variants (sensitivity analysis, not core)
        'wiki_tables_shuffled_column', 'wiki_tables_shuffled_row',
        # Observatory datasets (external benchmark, generated separately)
        'observatory_wiki_tables_original', 'observatory_wiki_tables_0.25',
        'observatory_wiki_tables_0.5', 'observatory_wiki_tables_0.75',
        # Testbed / debugging
        'testbedXS',
        # DB perturbation variants
        'DB_DBcontent_equivalence_pre', 'DB_DBcontent_equivalence_post',
        'DB_schema_abbreviation_pre', 'DB_schema_abbreviation_post',
        'DB_schema_synonym_pre', 'DB_schema_synonym_post',
        # Deprecated / not in active pipeline
        'adult', 'tabfact',
        # Source-only dataset for wiki_containment embedding generation;
        # not used as a standalone downstream target.
        'wiki-join-search-deepjoin',
    }

    # Build column dataset list: respect per-dataset 'models:' restriction
    # and skip datasets whose tables_source doesn't exist on disk
    def is_column_dataset_valid(ds_name, ds_cfg, model_name):
        if ds_name in SKIP_DATASETS:
            return False
        if ds_cfg is None:
            ds_cfg = {}
        allowed = ds_cfg.get('models')
        if allowed and model_name not in allowed:
            return False
        tables_source = ds_cfg.get('tables_source')
        if tables_source and not (datasets_dir / tables_source).exists():
            return False
        return True

    # Row embedding datasets: only those with row_embedding: true
    # Also respect tables_source existence
    def is_row_dataset_valid(ds_name, ds_cfg):
        if ds_name in SKIP_DATASETS:
            return False
        if ds_cfg is None or not ds_cfg.get('row_embedding'):
            return False
        tables_source = ds_cfg.get('tables_source')
        if tables_source and not (datasets_dir / tables_source).exists():
            return False
        return True

    row_datasets = sorted(k for k, v in datasets.items() if is_row_dataset_valid(k, v))

    # Discover row_data datasets from filesystem
    row_data_dir = PROJECT_ROOT / 'datasets' / 'row_data'
    row_data_datasets = sorted(d.name for d in row_data_dir.iterdir()
                               if d.is_dir() and d.name.startswith('openml_')) \
                        if row_data_dir.exists() else []

    # Models with known dataset incompatibilities (e.g., tabtransformer
    # can't handle numeric-only tables — its architecture requires at least
    # one categorical feature for the transformer encoder)
    ROW_DATA_SKIP = {
        'tabtransformer': {
            'openml_1063', 'openml_1486', 'openml_40668', 'openml_40978',
            'openml_44958', 'openml_44975', 'openml_4534', 'openml_458',
            'openml_46908', 'openml_46912', 'openml_46915', 'openml_46919',
            'openml_46929', 'openml_46930', 'openml_46932', 'openml_46933',
            'openml_46934', 'openml_46950', 'openml_46952', 'openml_46955',
            'openml_46961', 'openml_46964', 'openml_46969', 'openml_46979',
            'openml_46980',
        },
    }

    # Table embeddings: derived from column embeddings for specific datasets
    table_datasets = ['ckan_subset', 'dlte_v1_queries', 'dlte_v1_targets',
                      'ecb_union', 'nq_tables', 'spider_join', 'wiki_union']

    # Text embedding jobs
    text_jobs = {k: v for k, v in cfg_models.get('text_embedding_jobs', {}).items()}

    # Filter models if requested
    filter_models = set(args.models) if args.models else None

    # Determine which types to check
    check_types = set(args.types) if args.types else {'column', 'row', 'row_data', 'table', 'text'}

    total_expected = 0
    total_present = 0
    all_missing = []

    # ── Column embeddings ─────────────────────────────────────────────────
    if 'column' in check_types:
        models = [m for m in column_models if not filter_models or m in filter_models]
        expected = 0
        present = 0
        missing = []

        for model in models:
            required_datasets = set()
            for task_name in model_expected_tasks(model, COLUMN_EMBEDDING_TASKS):
                if model in TASK_EXCLUDE_MODELS.get(task_name, set()):
                    continue
                for ds_name in TASK_DATASETS.get(task_name, []):
                    ds_cfg = datasets.get(ds_name)
                    if is_column_dataset_valid(ds_name, ds_cfg, model):
                        required_datasets.add(ds_name)

            for ds_name in sorted(required_datasets):
                expected += 1
                pkl = embeddings_base / 'column' / model / f'{ds_name}.pkl'
                if pkl.exists():
                    present += 1
                else:
                    missing.append((model, ds_name))

        total_expected += expected
        total_present += present
        all_missing.extend(('column', m, d) for m, d in missing)

        if not args.missing_only or missing:
            pct = (present / expected * 100) if expected > 0 else 100
            status = "COMPLETE" if not missing else f"{len(missing)} MISSING"
            print(f"\nColumn embeddings: {present}/{expected} ({pct:.0f}%) — {status}")
            if missing:
                _print_missing_by_model(missing, args.missing_only)

    # ── Row embeddings ────────────────────────────────────────────────────
    if 'row' in check_types:
        models = [m for m in row_models if not filter_models or m in filter_models]
        expected = 0
        present = 0
        missing = []

        for model in models:
            row_roots = resolve_model_overlay_roots(
                default_row_root,
                row_overlay_root,
                row_trained_models,
                model,
            )
            for ds in row_datasets:
                expected += 1
                if any((root / model / f'{ds}.pkl').exists() for root in row_roots):
                    present += 1
                else:
                    missing.append((model, ds))

        total_expected += expected
        total_present += present
        all_missing.extend(('row', m, d) for m, d in missing)

        if not args.missing_only or missing:
            pct = (present / expected * 100) if expected > 0 else 100
            status = "COMPLETE" if not missing else f"{len(missing)} MISSING"
            print(f"\nRow embeddings: {present}/{expected} ({pct:.0f}%) — {status}")
            if missing:
                _print_missing_by_model(missing, args.missing_only)

    # ── Row data embeddings ───────────────────────────────────────────────
    if 'row_data' in check_types:
        models = [m for m in row_data_models if not filter_models or m in filter_models]
        expected = 0
        present = 0
        missing = []
        default_row_data_root = embeddings_base / 'row_prediction'

        for model in models:
            row_data_roots = resolve_model_overlay_roots(
                default_row_data_root,
                row_data_overlay_root,
                row_data_trained_models,
                model,
            )
            skip_ds = ROW_DATA_SKIP.get(model, set())
            for ds in row_data_datasets:
                if ds in skip_ds:
                    continue
                expected += 1
                meta_exists = any((root / model / ds / 'metadata.json').exists() for root in row_data_roots)
                if meta_exists:
                    present += 1
                else:
                    missing.append((model, ds))

        total_expected += expected
        total_present += present
        all_missing.extend(('row_data', m, d) for m, d in missing)

        if not args.missing_only or missing:
            pct = (present / expected * 100) if expected > 0 else 100
            status = "COMPLETE" if not missing else f"{len(missing)} MISSING"
            print(f"\nRow data embeddings: {present}/{expected} ({pct:.0f}%) — {status}")
            if missing:
                _print_missing_by_model(missing, args.missing_only)

    # ── Table embeddings ──────────────────────────────────────────────────
    if 'table' in check_types:
        table_models = sorted(
            set(column_models) |
            set(discover_embedding_model_dirs(
                embeddings_base / 'table',
                exclude_query_encoders=True,
                exclude_hybrids=True,
            ))
        )
        models = [m for m in table_models if not filter_models or m in filter_models]
        expected = 0
        present = 0
        missing = []

        for model in models:
            required_datasets = set()
            for task_name in model_expected_tasks(model, set(TABLE_VARIANT_TASKS)):
                if model in TASK_EXCLUDE_MODELS.get(task_name, set()):
                    continue
                for ds in TASK_DATASETS.get(task_name, []):
                    ds_cfg = datasets.get(ds)
                    if is_column_dataset_valid(ds, ds_cfg, model):
                        required_datasets.add(ds)

            for ds in sorted(required_datasets):
                expected += 1
                pkl = embeddings_base / 'table' / model / f'{ds}.pkl'
                if pkl.exists():
                    present += 1
                else:
                    missing.append((model, ds))

        total_expected += expected
        total_present += present
        all_missing.extend(('table', m, d) for m, d in missing)

        if not args.missing_only or missing:
            pct = (present / expected * 100) if expected > 0 else 100
            status = "COMPLETE" if not missing else f"{len(missing)} MISSING"
            print(f"\nTable embeddings: {present}/{expected} ({pct:.0f}%) — {status}")
            if missing:
                _print_missing_by_model(missing, args.missing_only)

    # ── Text embeddings ───────────────────────────────────────────────────
    if 'text' in check_types:
        expected = 0
        present = 0
        missing = []

        for job_name, job_cfg in text_jobs.items():
            if filter_models:
                # Extract model from job name prefix (e.g., gte_nq_tables -> gte)
                job_model = job_name.split('_')[0]
                if job_model not in filter_models:
                    continue
            expected += 1
            output = PROJECT_ROOT / job_cfg['output']
            if output.exists():
                present += 1
            else:
                missing.append((job_name, str(job_cfg['output'])))

        total_expected += expected
        total_present += present
        all_missing.extend(('text', name, path) for name, path in missing)

        if not args.missing_only or missing:
            pct = (present / expected * 100) if expected > 0 else 100
            status = "COMPLETE" if not missing else f"{len(missing)} MISSING"
            print(f"\nText embeddings: {present}/{expected} ({pct:.0f}%) — {status}")
            if missing:
                for name, path in missing:
                    print(f"  {name}: {path}")

    # ── Round variants (gte_r1, gte_r2, ...) ────────────────────────────
    if args.rounds:
        # Determine expected GTE datasets for column/row/row_data
        gte_col_datasets = sorted(
            ds for ds, cfg in datasets.items()
            if is_column_dataset_valid(ds, cfg, 'gte')
        )
        gte_row_datasets = [ds for ds in row_datasets]  # gte is a valid row model

        for r in args.rounds:
            tag = f'gte_r{r}'
            round_expected = 0
            round_present = 0
            round_missing = []

            # Column
            if not check_types or 'column' in check_types:
                for ds in gte_col_datasets:
                    round_expected += 1
                    if (embeddings_base / 'column' / tag / f'{ds}.pkl').exists():
                        round_present += 1
                    else:
                        round_missing.append(('column', ds))

            # Table
            if not check_types or 'table' in check_types:
                for ds in table_datasets:
                    round_expected += 1
                    if (embeddings_base / 'table' / tag / f'{ds}.pkl').exists():
                        round_present += 1
                    else:
                        round_missing.append(('table', ds))

            # Row
            if not check_types or 'row' in check_types:
                for ds in gte_row_datasets:
                    round_expected += 1
                    if (embeddings_base / 'row' / tag / f'{ds}.pkl').exists():
                        round_present += 1
                    else:
                        round_missing.append(('row', ds))

            # Row data
            if not check_types or 'row_data' in check_types:
                for ds in row_data_datasets:
                    round_expected += 1
                    if (embeddings_base / 'row_prediction' / tag / ds / 'metadata.json').exists():
                        round_present += 1
                    else:
                        round_missing.append(('row_data', ds))

            total_expected += round_expected
            total_present += round_present
            all_missing.extend((f'round_{r}', t, d) for t, d in round_missing)

            if not args.missing_only or round_missing:
                pct = (round_present / round_expected * 100) if round_expected > 0 else 100
                status = "COMPLETE" if not round_missing else f"{len(round_missing)} MISSING"
                print(f"\nRound {r} ({tag}): {round_present}/{round_expected} ({pct:.0f}%) — {status}")
                if round_missing:
                    by_type = {}
                    for emb_type, ds in round_missing:
                        by_type.setdefault(emb_type, []).append(ds)
                    for emb_type, items in sorted(by_type.items()):
                        if len(items) <= 5 or args.missing_only:
                            print(f"  {emb_type}: {', '.join(items)}")
                        else:
                            print(f"  {emb_type}: {len(items)} missing")

    # Summary
    total_missing = len(all_missing)
    pct = (total_present / total_expected * 100) if total_expected > 0 else 100
    print(f"\n{'=' * 60}")
    print(f"EMBEDDINGS TOTAL: {total_present}/{total_expected} ({pct:.0f}%)")
    print(f"  Present: {total_present}")
    print(f"  Missing: {total_missing}")
    print(f"{'=' * 60}")

    return total_missing


def _print_missing_by_model(missing, missing_only):
    """Group and print missing items by model."""
    by_model = {}
    for model, ds in missing:
        by_model.setdefault(model, []).append(ds)
    for model, items in sorted(by_model.items()):
        if len(items) <= 5 or missing_only:
            print(f"  {model}: {', '.join(items)}")
        else:
            print(f"  {model}: {len(items)} missing")


# ── Downstream result checking ───────────────────────────────────────────────

# Tasks that support linear/dummy/cosine_threshold via --head_type
PROBE_TASKS = {
    'column_relation_prediction', 'column_type_prediction',
    'join_classification', 'join_containment',
    'record_linkage', 'row_prediction',
    'table_fact_verification', 'table_subset',
    'union_classification', 'union_regression',
}

COSINE_THRESHOLD_TASKS = {'record_linkage'}
INTERACTION_TASKS = {'table_fact_verification'}

REMOVED_TASKS = {'table_fact_verification'}

TABLE_VARIANT_TASKS = {
    'join_classification', 'table_subset',
    'union_classification', 'union_regression',
    'table_retrieval',
}
TABLE_VARIANTS = ['cls_embedding', 'column_mean', 'token_mean']

# Table embedding variants each model actually produces.
# Discovered from the table_embedding dict keys in the pkl files.
# Models not listed here are assumed to support all TABLE_VARIANTS.
MODEL_TABLE_VARIANTS = {
    'bert':        ['cls_embedding', 'column_mean', 'token_mean'],
    'gte':         ['cls_embedding', 'column_mean', 'token_mean'],
    'openai':      ['cls_embedding', 'column_mean'],
    'openai_3l':   ['cls_embedding', 'column_mean'],
    'openai_ada':  ['cls_embedding', 'column_mean'],
    'random':      ['cls_embedding', 'column_mean', 'token_mean'],
    'starmie':     ['column_mean'],
    'tabbie':      ['cls_embedding', 'column_mean'],
    'tabert':      ['column_mean'],
    'tabsketchfm': ['cls_embedding', 'column_mean', 'token_mean'],
    'tapas':       ['cls_embedding', 'column_mean', 'token_mean'],
    'tapex':       ['cls_embedding', 'table_embedding', 'token_mean'],
    'turl':        ['column_mean'],
    'tuta':        ['cls_embedding'],
}

TASK_DATASETS = {
    'column_clustering':            ['sato', 'sotab'],
    'column_relation_prediction':   ['sotab', 'WikiCT_relation'],
    'column_type_prediction':       ['sato', 'sotab'],
    'join_search':                  ['opendata_main', 'opendata_can', 'opendata_usa', 'opendata_uk_sg'],
    'join_classification':          ['spider_join'],
    'join_containment':             ['wiki_containment'],
    'table_retrieval':              ['nq_tables'],
    'table_subset':                 ['ckan_subset'],
    'union_search':                 ['santos', 'ugen_v1', 'ugen_v2', 'tus', 'tus_hard'],
    'union_classification':         ['wiki_union'],
    'union_regression':             ['ecb_union'],
    'schema_matching':              ['valentine'],
    'semantic_parsing':             ['semantic_parsing'],
    'record_linkage': [
        'deepmatcher_abt_buy', 'deepmatcher_amazon_google', 'deepmatcher_beer',
        'deepmatcher_dblp_acm', 'deepmatcher_dblp_acm_dirty',
        'deepmatcher_dblp_scholar', 'deepmatcher_dblp_scholar_dirty',
        'deepmatcher_fodors_zagats',
        'deepmatcher_itunes_amazon', 'deepmatcher_itunes_amazon_dirty',
        'deepmatcher_walmart_amazon', 'deepmatcher_walmart_amazon_dirty',
        'wdc_products_small', 'wdc_products_medium',
        'wdc_products_large', 'wdc_products_xlarge',
    ],
}

TASK_EXCLUDE_MODELS = {
    'column_relation_prediction': {'tfidf'},
}

# Some embedding families are intentionally only used in a subset of downstream
# tasks. Embedding completeness should not require assets for tasks that never
# consume that model family in practice.
MODEL_TASK_ALLOWLIST = {
    'tfidf': {'column_clustering', 'column_type_prediction'},
}

# Downstream tasks that consume column embeddings directly.
COLUMN_EMBEDDING_TASKS = {
    'column_clustering',
    'column_relation_prediction',
    'column_type_prediction',
    'join_search',
    'join_containment',
    'schema_matching',
    'semantic_parsing',
    'union_search',
}

# Tasks that iterate over external query encoders.  Results are stored in
# per-encoder subdirectories, so the checker must multiply the expected
# count by the number of encoders and look in the right path.
QUERY_ENCODER_TASKS = {
    'table_retrieval':  ['sentence_t5', 'mpnet'],
    'semantic_parsing': ['sentence_t5', 'mpnet'],
}

# Note: table_retrieval generates both hybrid and model_only retrieval modes
# in the generator, but they currently write to the same RESULTS_FILE path.
# Until the generator produces distinct result paths per mode, the checker
# does not multiply by retrieval mode to avoid false-missing reports.


# Query encoder models produce bridge table embeddings for retrieval tasks
# but should not be evaluated as standalone table embedding models.
def discover_models(results_base, embeddings_base, row_overlay_root: Path | None = None):
    models = set()
    for sub in ('column', 'table'):
        d = embeddings_base / sub
        if d.exists():
            models |= {x.name for x in d.iterdir() if x.is_dir() and not x.name.startswith('.')}
    for root in filter(None, [embeddings_base / 'row', row_overlay_root]):
        if root.exists():
            models |= {x.name for x in root.iterdir() if x.is_dir() and not x.name.startswith('.')}
    for task_dir in results_base.iterdir():
        if task_dir.is_dir() and task_dir.name not in ('summary', 'dlte'):
            models |= {x.name for x in task_dir.iterdir() if x.is_dir()}
    models -= QUERY_ENCODER_MODELS
    return sorted(models)


def model_expected_tasks(model: str, candidate_tasks: set[str]) -> set[str]:
    """Restrict task scope for models that are only used in select tasks."""
    allowed = MODEL_TASK_ALLOWLIST.get(model)
    if allowed is None:
        return set(candidate_tasks)
    return set(candidate_tasks) & allowed


def has_result(results_base, task, model, head_type, dataset, variant=None, seed=42, query_encoder=None,
               result_tag: str | None = None):
    model_dir = results_base / task / model
    if result_tag and task in ROW_EMBEDDING_TASKS:
        model_dir = model_dir / result_tag
    if variant:
        model_dir = model_dir / variant

    seed_str = f'seed{seed}'

    if head_type in ('linear', 'dummy', 'cosine_threshold'):
        head_dir = model_dir / head_type
        if head_dir.exists() and any(head_dir.rglob('*.json')):
            return True
        ds_head = model_dir / dataset / seed_str / head_type
        if ds_head.exists() and any(ds_head.rglob('*.json')):
            return True
        return False
    else:
        # For query-encoder tasks, check encoder-specific paths
        qenc_suffix = f'_{query_encoder}' if query_encoder else ''
        candidates = [
            model_dir / f'{model}_{dataset}{qenc_suffix}.json',
            model_dir / f'{model}_{dataset}{qenc_suffix}_{seed_str}.json',
            model_dir / f'{model}_{dataset}_eval.json',
            model_dir / dataset / seed_str / 'results.json',
            model_dir / seed_str / 'results.json',
            model_dir / 'results.json',
        ]
        if query_encoder:
            # Encoder-specific output dirs (e.g. semantic_parsing/bert/sentence_t5/seed72/)
            candidates.append(model_dir / query_encoder / seed_str / 'results.json')
            candidates.append(model_dir / dataset / query_encoder / seed_str / 'results.json')
            # table_retrieval stores results under variant/qenc/ subdirectory
            candidates.append(model_dir / query_encoder / f'{model}_{dataset}_{query_encoder}_{seed_str}.json')
        for pattern in candidates:
            if pattern.exists():
                return True
        return False


def has_embeddings(embeddings_base, emb_type, model, dataset=None, *,
                   row_overlay_root: Path | None = None,
                   strict_row_overlay_models: set[str] | None = None):
    if emb_type == 'row':
        roots = resolve_model_overlay_roots(
            embeddings_base / 'row',
            row_overlay_root,
            strict_row_overlay_models or set(),
            model,
        )
        for root in roots:
            emb_dir = root / model
            if not emb_dir.exists():
                continue
            if dataset:
                if (emb_dir / f'{dataset}.pkl').exists():
                    return True
            elif list(emb_dir.glob('*.pkl')):
                return True
        return False

    emb_dir = embeddings_base / emb_type / model
    if not emb_dir.exists():
        return False
    if dataset:
        return (emb_dir / f'{dataset}.pkl').exists()
    return bool(list(emb_dir.glob('*.pkl')))


def get_valid_head_types(task, requested_heads):
    if task in REMOVED_TASKS:
        return []
    valid = {'mlp'}
    if task in PROBE_TASKS:
        valid |= {'linear', 'dummy'}
    if task in COSINE_THRESHOLD_TASKS:
        valid.add('cosine_threshold')
    if task in INTERACTION_TASKS:
        valid.add('interaction')
    if requested_heads:
        valid &= set(requested_heads)
    return sorted(valid)


def check_downstream(args):
    """Check completeness of downstream evaluation results.

    Default results_base is PROJECT_ROOT/results/evaluation/ (TRL-Bench's
    documented public layout per README and src/trl_bench/run.py).
    """
    results_base = Path(args.results_dir) if args.results_dir else PROJECT_ROOT / 'results' / 'evaluation'
    embeddings_base = Path(args.embeddings_dir) if args.embeddings_dir else PROJECT_ROOT / 'embeddings'
    row_overlay_root = resolve_optional_path(PROJECT_ROOT, getattr(args, 'row_overlay_root', None))
    strict_row_overlay_models = discover_trained_models(load_yaml('slurm/config/row_models.yaml'))
    effective_result_tag = derive_result_tag(
        getattr(args, 'row_overlay_root', None),
        embeddings_base / 'row',
        getattr(args, 'result_tag', None),
    )

    if args.models:
        models = args.models
    else:
        models = discover_models(results_base, embeddings_base, row_overlay_root=row_overlay_root)
        models = [m for m in models if not m.startswith('gte_r')]

    tasks = list(TASK_DATASETS.keys())
    if not args.include_dlte:
        tasks = [t for t in tasks if not t.startswith('dlte')]
    if args.tasks:
        tasks = [t for t in tasks if t in args.tasks]

    total_expected = 0
    total_present = 0
    total_missing = 0

    for task in tasks:
        if task in REMOVED_TASKS:
            continue

        datasets = TASK_DATASETS.get(task, [])
        head_types = get_valid_head_types(task, args.head_types)
        excluded = TASK_EXCLUDE_MODELS.get(task, set())

        if not head_types:
            continue

        task_expected = 0
        task_present = 0
        task_missing = []

        for model in models:
            if model in excluded:
                continue

            for head in head_types:
                for dataset in datasets:
                    # Check if model has the required embeddings for this specific dataset
                    if task == 'record_linkage':
                        if not has_embeddings(
                            embeddings_base,
                            'row',
                            model,
                            dataset,
                            row_overlay_root=row_overlay_root,
                            strict_row_overlay_models=strict_row_overlay_models,
                        ):
                            continue
                    elif task in TABLE_VARIANT_TASKS:
                        if not has_embeddings(embeddings_base, 'table', model, dataset):
                            continue
                    elif task in ('column_clustering', 'column_relation_prediction',
                                  'column_type_prediction', 'join_search',
                                  'join_containment', 'schema_matching',
                                  'semantic_parsing', 'union_search'):
                        if not has_embeddings(embeddings_base, 'column', model, dataset):
                            continue

                    # Query encoder loop: tasks with query_encoders
                    # multiply expected results by the number of encoders
                    query_encoders = QUERY_ENCODER_TASKS.get(task, [None])

                    if task in TABLE_VARIANT_TASKS:
                        # Only check variants this model actually produces
                        model_variants = MODEL_TABLE_VARIANTS.get(model, TABLE_VARIANTS)
                        valid_variants = [v for v in model_variants if v in TABLE_VARIANTS]
                        for variant in valid_variants:
                            for qenc in query_encoders:
                                task_expected += 1
                                if has_result(results_base, task, model, head, dataset, variant, seed=args.seed,
                                              query_encoder=qenc, result_tag=effective_result_tag):
                                    task_present += 1
                                else:
                                    label = f"{dataset}/{variant}" + (f"/{qenc}" if qenc else "")
                                    task_missing.append((model, head, label, None))
                    else:
                        for qenc in query_encoders:
                            task_expected += 1
                            if has_result(results_base, task, model, head, dataset, seed=args.seed,
                                          query_encoder=qenc, result_tag=effective_result_tag):
                                task_present += 1
                            else:
                                label = dataset + (f"/{qenc}" if qenc else "")
                                task_missing.append((model, head, label, None))

        total_expected += task_expected
        total_present += task_present
        total_missing += len(task_missing)

        if not args.missing_only or task_missing:
            pct = (task_present / task_expected * 100) if task_expected > 0 else 100
            status = "COMPLETE" if not task_missing else f"{len(task_missing)} MISSING"
            print(f"\n{task}: {task_present}/{task_expected} ({pct:.0f}%) — {status}")

            if task_missing:
                by_model = {}
                for model, head, dataset, variant in task_missing:
                    by_model.setdefault(model, []).append(
                        f"{dataset}" + (f"/{variant}" if variant else "") + f" [{head}]"
                    )
                for model, items in sorted(by_model.items()):
                    if len(items) <= 5 or args.missing_only:
                        print(f"  {model}: {', '.join(items)}")
                    else:
                        print(f"  {model}: {len(items)} missing")

    pct = (total_present / total_expected * 100) if total_expected > 0 else 100
    print(f"\n{'=' * 60}")
    print(f"DOWNSTREAM TOTAL: {total_present}/{total_expected} ({pct:.0f}%)")
    print(f"  Present: {total_present}")
    print(f"  Missing: {total_missing}")
    print(f"{'=' * 60}")

    return total_missing


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Check completeness of embeddings and downstream results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--models', nargs='+', default=None,
                        help='Only check these models')
    parser.add_argument('--missing-only', action='store_true',
                        help='Only print missing combinations')

    # Mode selection
    mode = parser.add_argument_group('mode')
    mode.add_argument('--embeddings', action='store_true',
                      help='Check embedding completeness')
    mode.add_argument('--downstream', action='store_true',
                      help='Check downstream result completeness (default if no mode given)')
    mode.add_argument('--all', action='store_true',
                      help='Check both embeddings and downstream')

    # Embedding options
    emb_opts = parser.add_argument_group('embedding options')
    emb_opts.add_argument('--types', nargs='+', default=None,
                          choices=['column', 'row', 'row_data', 'table', 'text'],
                          help='Embedding types to check (default: all)')
    emb_opts.add_argument('--rounds', nargs='+', default=None,
                          help='Check GTE round variants (e.g., --rounds 1 2 3 4)')
    emb_opts.add_argument('--row-overlay-root', '--row-embedding-root', dest='row_overlay_root', type=str, default=None,
                          help='Overlay row embedding root to merge on top of embeddings/row '
                               '(e.g., embeddings/row_dim768). Trained row models are '
                               'read strictly from this root when set. Also applies to row-level '
                               'downstream completeness checks.')
    emb_opts.add_argument('--row-data-overlay-root', type=str, default=None,
                          help='Overlay row_data embedding root to merge on top of '
                               'embeddings/row_prediction (e.g., '
                               'embeddings/row_prediction_dim512)')

    # Downstream options
    ds_opts = parser.add_argument_group('downstream options')
    ds_opts.add_argument('--tasks', nargs='+', default=None,
                         help='Only check these tasks')
    ds_opts.add_argument('--head-types', nargs='+', default=None,
                         help='Only check these head types')
    ds_opts.add_argument('--seed', type=int, default=42,
                         help='Seed to check results for (default: 42)')
    ds_opts.add_argument('--include-dlte', action='store_true',
                         help='Include DLTE multi-stage tasks')
    ds_opts.add_argument('--result-tag', type=str, default=None,
                         help='Row-level result tag (default: derived from --row-embedding-root)')
    ds_opts.add_argument('--results-dir', type=str, default=None)
    ds_opts.add_argument('--embeddings-dir', type=str, default=None)

    args = parser.parse_args()

    # Default to downstream if no mode specified
    if not args.embeddings and not args.downstream and not args.all:
        args.downstream = True

    total_missing = 0

    if args.embeddings or args.all:
        total_missing += check_embeddings(args)

    if args.downstream or args.all:
        if args.embeddings or args.all:
            print()  # separator
        total_missing += check_downstream(args)

    return 0 if total_missing == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
