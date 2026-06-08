"""Single-experiment entry point for trl-bench.

Dispatches one (model, task, dataset, setting, probe, seed) cell, with
auto-staging of the HF dataset to the on-disk layout the wrappers expect.

Pipeline (per cell):

    Stage 0  HF dataset -> <data_root>/<dataset>/<inner>/{tables_all/, labels.json}
             (handled by ``trl_bench.data.stage.stage_dataset``; auto unless
             --labels-path is passed)
    Stage 1  Per-model column extraction. Auto-dispatched via the registry's
             ``_MODEL_EXTRACTORS`` table for wired models (BERT, GTE today).
             Wrappers not in the table still require manual Stage-1 invocation;
             see ``src/trl_bench/models/<m>/USAGE.md``.
    Stage 2  python -m trl_bench.scripts.generate_table_embeddings (auto when
             Stage-1 is auto-dispatched; idempotent — skips if the table pickle
             already exists).
    Stage 3  python -m trl_bench.utils.downstream.run_task + envelope wrap
             (dispatched by ``trl_bench.registry.build_command``; runs here)

Output JSON is written to::

    <results-dir>/evaluation/<task>/<model>/<setting>/<probe>/<model>_<dataset>_seed<S>.json

in the same flat-key envelope format the reference uses
(``test_results_accuracy``, ``test_results_weighted_f1``, ...).
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from trl_bench.registry import (
    build_command, build_extractor_command, build_query_extractor_command,
    build_question_extractor_command,
    build_table_encoder_command,
    build_starmie_pretrain_command, starmie_checkpoint_path,
    build_row_data_commands,
    SettingError, _MODEL_EXTRACTORS, _MODEL_GRANULARITIES,
    _QUERY_ENCODER_EXTRACTORS, _SEMPARSE_QUESTION_ENCODERS,
    _ROW_RUNNERS, _TABLE_ENCODERS, _TABLE_NATIVE_RUNNERS,
    DETERMINISTIC_TASKS,
    union_search_params,
)


# Tasks whose runners only emit metrics to stdout (no native JSON / log
# file). For these, run.py captures stdout to ``stage_run.log`` inside the
# per-cell output directory and the envelope wrapper parses it with the same
# regex set that the canonical .sbatch ``extract_metrics`` shell helpers use.
_STDOUT_CAPTURE_TASKS = frozenset({
    "column_clustering",
    "union_search",
    "join_search",
    # join_search_learned: run_learned_search.py prints the same COL-metric
    # lines to stdout (no aggregate JSON); the join_search_flat wrapper parses
    # them. (results.json holds per-pair retrieval, not the COL aggregates.)
    "join_search_learned",
})


# task -> suite. Used by auto-stage when --labels-path is not provided.
_TASK_TO_SUITE: dict[str, str] = {
    # ctbench (col / table)
    "column_type_prediction":     "ctbench",
    "column_clustering":          "ctbench",
    "column_relation_prediction": "ctbench",
    "join_search":                "ctbench",
    "join_search_learned":        "ctbench",
    "schema_matching":            "ctbench",
    "join_containment":           "ctbench",
    "join_classification":        "ctbench",
    "union_search":               "ctbench",
    "union_classification":       "ctbench",
    "union_regression":           "ctbench",
    "table_subset":               "ctbench",
    "table_retrieval":            "ctbench",
    "semantic_parsing":           "ctbench",
    # rbench
    "row_prediction":             "rbench",
    "record_linkage":             "rbench",
    # dlte
    "dlte_retrieval":             "dlte",
    "dlte_alignment":             "dlte",
    "dlte_merge":                 "dlte",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="trl-bench-run",
        description="Run one (model, task, dataset, setting, probe, seed) experiment.",
    )
    p.add_argument("--model",   required=True)
    p.add_argument("--task",    required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--setting", required=True,
                   help="Aggregation: cls_embedding | column_mean | token_mean | "
                        "table_embedding | row_embedding")
    p.add_argument("--probe", default=None,
                   help="Probe head: linear | mlp | cosine_threshold | dummy. "
                        "Required for supervised PROBE_TASKS.")
    p.add_argument("--seed", type=int, required=True)

    p.add_argument("--embeddings-path", default=None,
                   help="Path to pre-extracted table-embedding pickle (Stage-1/2 "
                        "output). If omitted, defaults to "
                        "<embeddings-dir>/table/<model>/<dataset>.pkl, which must "
                        "exist; Stage 1+2 are not yet auto-orchestrated.")
    p.add_argument("--labels-path", default=None,
                   help="Path to task labels.json. If omitted, auto-stages from "
                        "HF (Stage 0) into <data-root> and uses the produced "
                        "labels.json. Pass --no-auto-stage to disable.")
    p.add_argument("--no-auto-stage", action="store_true",
                   help="Disable auto-staging of HF dataset to disk. Requires "
                        "--labels-path to be passed explicitly.")
    p.add_argument("--embeddings-dir", default="./embeddings",
                   help="Directory holding column/<model>/<dataset>.pkl and "
                        "table/<model>/<dataset>.pkl (default: ./embeddings).")
    p.add_argument("--extract-device", default=None,
                   help="Device for auto-orchestrated Stage-1 column extraction "
                        "(cuda/cpu). Default: cuda (override per registry's "
                        "ExtractorConfig.device_value). Only honored when "
                        "--embeddings-path is omitted and the model has an "
                        "extractor wired.")
    p.add_argument("--checkpoint-root", default=None,
                   help="On-disk root for licensed-checkpoint wrappers "
                        "(TaBERT, TabSketchFM, TURL, TUTA, TABBIE, Starmie). "
                        "Falls back to $TRL_BENCH_CKPT_ROOT and then to "
                        "./checkpoints. The dispatcher resolves "
                        "<checkpoint-root>/<template> per "
                        "ExtractorConfig.checkpoint_template. See "
                        "scripts/download_checkpoints.sh and "
                        "docs/CHECKPOINT_LICENSES.md.")
    p.add_argument("--results-dir", default="./results",
                   help="Directory to write per-job result JSONs.")
    p.add_argument("--data-root", default="./data",
                   help="Directory where Stage-0 staged datasets live.")
    p.add_argument("--configs-root", default="configs/downstream",
                   help="Directory holding downstream-task YAML configs.")
    return p.parse_args(argv)


def _envelope_meta(
    *, model: str, dataset: str, task: str, seed: int, head_type: str,
    cfg_hyperparams: dict,
) -> dict:
    """Build the metadata block shared across all envelope variants."""
    return {
        "model":        model,
        "dataset":      dataset,
        "task":         task,
        "seed":         seed,
        "head_type":    head_type,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "hyperparameters": cfg_hyperparams,
    }


def _wrap_envelope_test_results_flat(
    raw_results_path: Path,
    *, model: str, dataset: str, task: str, seed: int, head_type: str,
    cfg_hyperparams: dict,
) -> dict:
    """Convert run_task.py's ``results.json`` into the reference envelope.

    The reference JSONs (pair tasks + record_linkage) use flat
    ``test_results_<metric>`` keys, plus a top-level metadata block.
    ``run_task.py`` emits a nested ``test_results`` sub-dict, which this
    function flattens.
    """
    raw = json.loads(raw_results_path.read_text())
    envelope = _envelope_meta(
        model=model, dataset=dataset, task=task, seed=seed,
        head_type=head_type, cfg_hyperparams=cfg_hyperparams,
    )
    for key, value in raw.get("test_results", {}).items():
        clean = key[5:] if key.startswith("test_") else key
        envelope[f"test_results_{clean}"] = value
    if "data_stats" in raw:
        envelope["data_stats"] = raw["data_stats"]
    envelope["status"] = "completed"
    return envelope


# Metadata keys that runners stamp into their ``results.json`` for bookkeeping
# but that the reference envelope keeps under the top-level meta block (not
# duplicated as flat metric keys).
_META_PASSTHROUGH_KEYS = frozenset({
    "task_name", "task", "head_type", "seed", "model", "dataset",
    "task_type", "variant", "label_column",
})


def _wrap_envelope_cta_flat(
    raw_results_path: Path,
    *, model: str, dataset: str, task: str, seed: int, head_type: str,
    cfg_hyperparams: dict,
) -> dict:
    """Wrap CTA's ``results.json`` into the reference flat envelope.

    The CTA reference envelope (see
    ``round5/evaluation/column_type_prediction/bert/bert_sato_seed42.json``)
    has flat keys ``test_micro_f1``, ``micro_f1``, ``MAP``, ``best_MAP``,
    ``final_train_MAP``, ``final_test_MAP``. ``train_ct_mode4.py`` already
    writes these at the top level of ``results.json`` (after stripping the
    ``test_`` prefix from some), so we pass them through verbatim.
    """
    raw = json.loads(raw_results_path.read_text())
    envelope = _envelope_meta(
        model=model, dataset=dataset, task=task, seed=seed,
        head_type=head_type, cfg_hyperparams=cfg_hyperparams,
    )
    for key, value in raw.items():
        if key in _META_PASSTHROUGH_KEYS:
            continue
        if key == "data_stats":
            envelope["data_stats"] = value
            continue
        if key == "class_to_idx":
            # Large bookkeeping dict; preserved out-of-band as references do.
            continue
        envelope[key] = value
    envelope["status"] = "completed"
    return envelope


def _wrap_envelope_cra_flat(
    raw_results_path: Path,
    *, model: str, dataset: str, task: str, seed: int, head_type: str,
    cfg_hyperparams: dict,
) -> dict:
    """Wrap CRA's ``results.json`` into the reference flat envelope.

    The CRA reference envelope (see
    ``round5/evaluation/column_relation_prediction/bert/bert_SOTAB_seed42.json``)
    has flat keys ``best_micro_f1``, ``micro_f1``, ``macro_f1``,
    ``subset_accuracy``, ``hamming_accuracy``. ``csv_relation_pipeline.py``
    writes these at the top level (some after ``test_``-prefix stripping), so
    we pass them through verbatim.
    """
    raw = json.loads(raw_results_path.read_text())
    envelope = _envelope_meta(
        model=model, dataset=dataset, task=task, seed=seed,
        head_type=head_type, cfg_hyperparams=cfg_hyperparams,
    )
    for key, value in raw.items():
        if key in _META_PASSTHROUGH_KEYS:
            continue
        if key == "data_stats":
            envelope["data_stats"] = value
            continue
        envelope[key] = value
    envelope["status"] = "completed"
    return envelope


def _wrap_envelope_retrieval_flat(
    raw_results_path: Path,
    *, model: str, dataset: str, task: str, seed: int, head_type: str,
    cfg_hyperparams: dict,
) -> dict:
    """Wrap table_retrieval's ``results.json`` into the reference envelope.

    The retrieval reference envelope (see
    ``round5/evaluation/table_retrieval/bert/<setting>/bert_nq_tables_seed42.json``)
    has flat keys ``Recall@1/5/10/20/50/100``, ``MRR``, ``NDCG@10/20``, plus
    ``num_queries``, ``num_tables``, ``retrieval_mode``. ``evaluate.py`` writes
    metrics under a nested ``metrics`` sub-dict; this wrapper flattens them.
    Note the retrieval envelope does NOT include ``head_type`` — embedding
    variant is the per-setting axis instead.
    """
    raw = json.loads(raw_results_path.read_text())
    # ``retrieval_mode`` is "hybrid" for every canonical .sbatch in the
    # reference (the table embeddings are always the concat of base + bert
    # column_mean). ``evaluate.py`` does not emit it, so we hardcode it here
    # to match the reference shape. When non-hybrid retrieval modes are wired,
    # plumb the mode through the registry instead of defaulting it here.
    envelope: dict = {
        "model":        model,
        "dataset":      dataset,
        "task":         task,
        "seed":         seed,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "retrieval_mode": raw.get("retrieval_mode", "hybrid"),
        "hyperparameters": cfg_hyperparams,
    }
    # evaluate.py nests metrics under "metrics"; flatten them.
    for k, v in (raw.get("metrics") or {}).items():
        envelope[k] = v
    # num_queries / num_tables are top-level in the runner's output.
    for k in ("num_queries", "num_tables"):
        if k in raw:
            envelope[k] = raw[k]
    envelope["status"] = "completed"
    return envelope


def _wrap_envelope_row_prediction_flat(
    raw_results_path: Path,
    *, model: str, dataset: str, task: str, seed: int, head_type: str,
    cfg_hyperparams: dict,
) -> dict:
    """Wrap row_prediction's ``results.json`` into the reference envelope.

    The row_prediction reference envelope is structured (not flat): it preserves the runner's nested ``test_results``,
    ``training``, ``data_stats``, ``train_results``, ``scaled_test_results``,
    ``target_zscore``, ``target_zscore_split_mode``, ``target_scaler`` blocks
    intact. The runner already stamps ``task``, ``task_type``, ``head_type``,
    ``seed``, ``model``, ``dataset``, ``variant``, ``label_column`` at the
    top level; we pass them through and add ``slurm_job_id`` + ``status``.

    Unlike pair-task envelopes, there is no ``hyperparameters`` block (the
    YAML drives everything); we add a minimal one keyed to ``head_type`` to
    document the configured probe head, matching the spec.
    """
    raw = json.loads(raw_results_path.read_text())
    envelope: dict = {}
    # The runner stamps task/seed/model/dataset/etc at top level; pass-through
    # in canonical reference order. If a key is missing, use the dispatch
    # values (so the envelope is robust even if the runner doesn't echo them).
    runner_seed = raw.get("seed", seed)
    runner_head = raw.get("head_type", head_type)
    runner_model = raw.get("model") or model
    runner_dataset = raw.get("dataset") or dataset
    envelope["task_name"]    = raw.get("task_name")
    envelope["task"]         = raw.get("task", task)
    envelope["task_type"]    = raw.get("task_type")
    envelope["head_type"]    = runner_head
    envelope["seed"]         = runner_seed
    envelope["model"]        = runner_model
    envelope["dataset"]      = runner_dataset
    envelope["variant"]      = raw.get("variant")
    envelope["label_column"] = raw.get("label_column")
    # Structured metric / training blocks (preserved as-is).
    for k in (
        "test_results", "training", "data_stats",
        "train_results", "scaled_test_results",
        "target_zscore", "target_zscore_split_mode", "target_scaler",
        "label_map",
    ):
        if k in raw:
            envelope[k] = raw[k]
    # Release-only bookkeeping.
    envelope["slurm_job_id"] = os.environ.get("SLURM_JOB_ID", "local")
    if cfg_hyperparams:
        envelope["hyperparameters"] = cfg_hyperparams
    envelope["status"] = "completed"
    return envelope


def _wrap_envelope_semparse_flat(
    raw_results_path: Path,
    *, model: str, dataset: str, task: str, seed: int, head_type: str,
    cfg_hyperparams: dict,
) -> dict:
    """Wrap semantic_parsing's ``test.log`` into the reference envelope.

    The semantic_parsing reference envelope (see
    ``evaluation_results/semantic_parsing/bert/mpnet/
    bert_semantic_parsing_mpnet_seed72.json``) is flat:

        {model, dataset, task, seed, slurm_job_id,
         hyperparameters={seed, beam_size},
         accuracy, oracle_accuracy, status}

    Note the envelope has NO ``head_type`` — semantic_parsing has no probe
    head axis. The raw input is ``test.log``, which ``run_test.py`` writes as
    a JSON dump of ``{"accuracy": ..., "oracle_accuracy": ...}`` next to
    ``model.best.bin`` in the training output directory.
    """
    raw = json.loads(raw_results_path.read_text())
    envelope: dict = {
        "model":        model,
        "dataset":      dataset,
        "task":         task,
        "seed":         seed,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "hyperparameters": cfg_hyperparams,
    }
    for k in ("accuracy", "oracle_accuracy"):
        if k in raw:
            envelope[k] = raw[k]
    envelope["status"] = "completed"
    return envelope


def _wrap_envelope_dlte_retrieval_flat(
    raw_results_path: Path,
    *, model: str, dataset: str, task: str, seed: int, head_type: str,
    cfg_hyperparams: dict,
) -> dict:
    """Wrap DLTE Stage-1 (retrieval) outputs into a reference envelope.

    The DLTE Stage-1 runner writes per-(split, K) metric JSONs:
        <results>/dlte/stage1/<model>/metrics_<split>_topk_<K>.json
    each carrying flat keys ``k``, ``n_queries``, ``n_evaluated``, ``n_skipped``,
    ``recall_any``, ``recall_union``, ``recall_join``, ``mrr_any``, ``per_tier``.

    The reference aggregated envelope groups them under
    ``metrics.test_topk_<K>`` (see ``metrics/<col>__<col>/stage1.json``). This
    wrapper reads ``raw_results_path`` (which the dispatcher resolves to the
    test/topk_100 file by convention) and packs the discovered siblings into
    that structured envelope.
    """
    raw = json.loads(raw_results_path.read_text())
    envelope: dict = {
        "model":        model,
        "dataset":      dataset,
        "task":         task,
        "seed":         seed,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "table_model":  model,
        "col_model":    model,
    }
    if cfg_hyperparams:
        envelope["hyperparameters"] = cfg_hyperparams
    # Discover sibling metric files: metrics_test_topk_<K>.json.
    metrics: dict = {}
    raw_dir = raw_results_path.parent
    for sibling in sorted(raw_dir.glob("metrics_test_topk_*.json")):
        try:
            k_str = sibling.stem.split("metrics_test_topk_")[-1]
            k_int = int(k_str)
        except ValueError:
            continue
        try:
            block = json.loads(sibling.read_text())
        except Exception:
            continue
        metrics[f"test_topk_{k_int}"] = block
    if not metrics:
        # Fallback: treat raw_results_path itself as the sole entry.
        k_val = raw.get("k") or "default"
        metrics[f"test_topk_{k_val}"] = raw
    envelope["metrics"] = metrics
    envelope["status"] = "completed"
    return envelope


def _wrap_envelope_dlte_alignment_flat(
    raw_results_path: Path,
    *, model: str, dataset: str, task: str, seed: int, head_type: str,
    cfg_hyperparams: dict,
) -> dict:
    """Wrap DLTE Stage-2 (alignment) output into a reference envelope.

    The DLTE Stage-2 runner produces ``calibration_dev.json`` and
    ``metrics_<split>_topk_100.json`` under
    ``<results>/dlte/stage2/<stage2_key>/``. The reference's aggregated
    Stage-2 envelope (verified at
    ``metrics/<col>__<col>/stage2.json``) carries::

        {col_model, calibration: {...}, metrics: {test: {...}}}

    The ``test`` block holds ``split``, ``n_queries``, ``n_pairs``,
    ``relation_acc``, ``relation_per_class``, ``key_col_acc``,
    ``key_col_total``, ``col_align_f1_union``, ``col_align_f1_count``, plus
    ``per_tier``. This wrapper passes those values through verbatim.
    """
    raw = json.loads(raw_results_path.read_text())
    envelope: dict = {
        "col_model": model,
        "task":      task,
        "dataset":   dataset,
        "seed":      seed,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
    }
    if cfg_hyperparams:
        envelope["hyperparameters"] = cfg_hyperparams
    # Discover calibration sibling.
    raw_dir = raw_results_path.parent
    calib_path = raw_dir / "calibration_dev.json"
    if calib_path.exists():
        envelope["calibration"] = json.loads(calib_path.read_text())
    # The raw_results_path itself is the test metrics block.
    envelope["metrics"] = {"test": raw}
    envelope["status"] = "completed"
    return envelope


def _wrap_envelope_dlte_merge_flat(
    raw_results_path: Path,
    *, model: str, dataset: str, task: str, seed: int, head_type: str,
    cfg_hyperparams: dict,
) -> dict:
    """Wrap DLTE Stage-3 (merge) output into a reference envelope.

    The DLTE Stage-3 runner writes ``metrics/<col>__<row>__<table>/end_to_end.json``
    with shape::

        {col_model, row_model, table_model,
         splits: {test: {n_queries, n_evaluated, cell_f1, cell_precision,
                         cell_recall, parent_row_recall, parent_col_recall,
                         region_recall, uj_h, per_tier}, ...}}

    This wrapper passes that structure through verbatim and adds the
    envelope-wide metadata block.
    """
    raw = json.loads(raw_results_path.read_text())
    envelope: dict = {
        "col_model":   raw.get("col_model", model),
        "row_model":   raw.get("row_model", model),
        "table_model": raw.get("table_model", model),
        "task":        task,
        "dataset":     dataset,
        "seed":        seed,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
    }
    if cfg_hyperparams:
        envelope["hyperparameters"] = cfg_hyperparams
    # Pass through the structured splits block.
    if "splits" in raw:
        envelope["splits"] = raw["splits"]
    envelope["status"] = "completed"
    return envelope


def _wrap_envelope_column_clustering_flat(
    raw_results_path: Path,
    *, model: str, dataset: str, task: str, seed: int, head_type: str,
    cfg_hyperparams: dict,
) -> dict:
    """Wrap column_clustering's captured stdout into the reference envelope.

    The column_clustering reference envelope (verified at
    ``round5/evaluation_results/column_clustering/bert/bert_sato.json``)
    carries flat keys::

        {model, dataset, task, slurm_job_id,
         hyperparameters: {k, target_avg_size},
         purity, num_clusters, avg_cluster_size, nmi, ari,
         total_columns, matched_columns, expected_columns, coverage_pct,
         missing_tables_count, col_mismatches, min_coverage, status}

    The runner (``evaluate_clustering.py``) prints these to stdout; we parse
    them with the same regex pattern the canonical .sbatch's extract_metrics
    helper uses.
    """
    import re
    text = raw_results_path.read_text() if raw_results_path.exists() else ""
    envelope: dict = {
        "model":   model,
        "dataset": dataset,
        "task":    task,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
    }
    if cfg_hyperparams:
        envelope["hyperparameters"] = cfg_hyperparams

    # Core scalar metrics (printed by evaluate_clustering.py's banner). The
    # reference's purity field is a 4-decimal truncation of the runner's print
    # ("Purity: %.4f"). We keep full float precision here; consumers comparing
    # against the reference should round before equality checks.
    patterns_float = {
        "purity": r"Purity:\s+([0-9.]+)",
        "avg_cluster_size": r"Avg cluster size:\s+([0-9.]+)",
        "nmi": r"NMI:\s+([0-9.]+)",
        "ari": r"ARI:\s+([0-9.]+)",
        "coverage_pct": r"Coverage:\s+\d+/\d+\s+\(([0-9.]+)\)",
    }
    patterns_int = {
        "num_clusters": r"Number of clusters:\s+([0-9,]+)",
        "total_columns": r"Total columns:\s+([0-9,]+)",
        "missing_tables_count": r"Missing tables:\s+([0-9]+)",
        "col_mismatches": r"Column mismatches:\s+([0-9]+)",
    }
    for key, pat in patterns_float.items():
        m = re.search(pat, text)
        if m:
            envelope[key] = float(m.group(1))
    for key, pat in patterns_int.items():
        m = re.search(pat, text)
        if m:
            envelope[key] = int(m.group(1).replace(",", ""))
    # Coverage's secondary numerator/denominator: "Coverage: matched/expected (frac)".
    m = re.search(r"Coverage:\s+(\d+)/(\d+)\s+\(", text)
    if m:
        envelope["matched_columns"] = int(m.group(1))
        envelope["expected_columns"] = int(m.group(2))
    envelope["status"] = "completed" if text else "error"
    return envelope


def _wrap_envelope_schema_matching_flat(
    raw_results_path: Path,
    *, model: str, dataset: str, task: str, seed: int, head_type: str,
    cfg_hyperparams: dict,
) -> dict:
    """Wrap schema_matching's native ``results.json`` into the reference envelope.

    The schema_matching reference envelope (verified at
    ``round5/evaluation_results/schema_matching/turl/turl_valentine.json``)
    carries flat keys::

        {model, dataset, task, slurm_job_id,
         hyperparameters: {matching_strategy, threshold},
         micro_precision, micro_recall, micro_f1,
         macro_precision, macro_recall, macro_f1,
         pairs_evaluated, pairs_skipped, recall_at_gt, gt_coverage, status}

    The runner writes a richer ``results.json`` (with nested per-source /
    per-noise breakdowns); this wrapper extracts only the flat keys that the
    reference preserves at the top level.
    """
    raw = json.loads(raw_results_path.read_text())
    envelope: dict = {
        "model":   model,
        "dataset": dataset,
        "task":    task,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
    }
    if cfg_hyperparams:
        envelope["hyperparameters"] = cfg_hyperparams
    for key in (
        "micro_precision", "micro_recall", "micro_f1",
        "macro_precision", "macro_recall", "macro_f1",
    ):
        if key in raw:
            envelope[key] = raw[key]
    # The reference uses ``pairs_evaluated`` / ``pairs_skipped``; the runner
    # writes ``n_pairs`` / ``n_skipped``. Map the runner's keys to the
    # reference's names.
    if "n_pairs" in raw:
        envelope["pairs_evaluated"] = raw["n_pairs"]
    if "n_skipped" in raw:
        envelope["pairs_skipped"] = raw["n_skipped"]
    for key in ("recall_at_gt", "gt_coverage"):
        if key in raw:
            envelope[key] = raw[key]
    envelope["status"] = "completed"
    return envelope


def _wrap_envelope_union_search_flat(
    raw_results_path: Path,
    *, model: str, dataset: str, task: str, seed: int, head_type: str,
    cfg_hyperparams: dict,
) -> dict:
    """Wrap union_search's captured stdout into the reference envelope.

    The union_search reference envelope (verified at
    ``round5/evaluation_results/union_search/gte/gte_santos.json``) carries::

        {model, dataset, task, slurm_job_id,
         hyperparameters: {method, k, threshold, ef, N},
         map_at_k, precision_at_k, recall_at_k, status}

    The runner (``run_search.py``) prints ``MAP@K = ...``, ``P@K = ...``,
    ``R@K = ...`` to stdout; we parse them with the same regex the canonical
    .sbatch extract_metrics helper uses.
    """
    import re
    text = raw_results_path.read_text() if raw_results_path.exists() else ""
    envelope: dict = {
        "model":   model,
        "dataset": dataset,
        "task":    task,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
    }
    if cfg_hyperparams:
        envelope["hyperparameters"] = cfg_hyperparams
    # ``MAP@<k> = <float>`` -> ``map_at_k``; same for P@k / R@k.
    m = re.search(r"MAP@\s*\d+\s*=\s*([0-9.]+)", text)
    if m:
        envelope["map_at_k"] = float(m.group(1))
    m = re.search(r"P@\s*\d+\s*=\s*([0-9.]+)", text)
    if m:
        envelope["precision_at_k"] = float(m.group(1))
    m = re.search(r"R@\s*\d+\s*=\s*([0-9.]+)", text)
    if m:
        envelope["recall_at_k"] = float(m.group(1))
    envelope["status"] = "completed" if text else "error"
    return envelope


def _wrap_envelope_join_search_flat(
    raw_results_path: Path,
    *, model: str, dataset: str, task: str, seed: int, head_type: str,
    cfg_hyperparams: dict,
) -> dict:
    """Wrap join_search's captured stdout into the reference envelope.

    The join_search reference envelope (verified at
    ``round5/evaluation_results/join_search/tabbie/tabbie_opendata.json``)
    carries::

        {model, dataset, task, slurm_job_id,
         hyperparameters: {k},
         col_recall_at_10/20/50, col_precision_at_10/20/50,
         col_f1_at_10/20/50, col_map, status}

    The runner (``run_search_and_evaluate.py``) emits column-level metrics
    prefixed with ``COL`` (e.g. ``COL Precision@10: 0.2400``) to stdout; we
    parse them with the same regex the canonical .sbatch's extract_metrics
    helper uses. Table-level mode (``TBL`` prefix) is not the reference
    convention — we extract only the COL block.
    """
    import re
    text = raw_results_path.read_text() if raw_results_path.exists() else ""
    envelope: dict = {
        "model":   model,
        "dataset": dataset,
        "task":    task,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
    }
    if cfg_hyperparams:
        envelope["hyperparameters"] = cfg_hyperparams
    # Per-K column-level metrics: COL Recall@<k>: <float>, etc.
    for k_val in (10, 20, 50):
        m = re.search(rf"COL Recall@{k_val}:\s+([0-9.]+)", text)
        if m:
            envelope[f"col_recall_at_{k_val}"] = float(m.group(1))
        m = re.search(rf"COL Precision@{k_val}:\s+([0-9.]+)", text)
        if m:
            envelope[f"col_precision_at_{k_val}"] = float(m.group(1))
        m = re.search(rf"COL F1@{k_val}:\s+([0-9.]+)", text)
        if m:
            envelope[f"col_f1_at_{k_val}"] = float(m.group(1))
    # K-independent MAP.
    m = re.search(r"COL MAP:\s+([0-9.]+)", text)
    if m:
        envelope["col_map"] = float(m.group(1))
    envelope["status"] = "completed" if text else "error"
    return envelope


_ENVELOPE_WRAPPERS = {
    "test_results_flat":     _wrap_envelope_test_results_flat,
    "cta_flat":              _wrap_envelope_cta_flat,
    "cra_flat":              _wrap_envelope_cra_flat,
    "retrieval_flat":        _wrap_envelope_retrieval_flat,
    "row_prediction_flat":   _wrap_envelope_row_prediction_flat,
    "semparse_flat":         _wrap_envelope_semparse_flat,
    "dlte_retrieval_flat":   _wrap_envelope_dlte_retrieval_flat,
    "dlte_alignment_flat":   _wrap_envelope_dlte_alignment_flat,
    "dlte_merge_flat":       _wrap_envelope_dlte_merge_flat,
    "column_clustering_flat": _wrap_envelope_column_clustering_flat,
    "schema_matching_flat":   _wrap_envelope_schema_matching_flat,
    "union_search_flat":      _wrap_envelope_union_search_flat,
    "join_search_flat":       _wrap_envelope_join_search_flat,
}


def _wrap_envelope(
    raw_results_path: Path,
    *, model: str, dataset: str, task: str, seed: int, head_type: str,
    cfg_hyperparams: dict,
    envelope_kind: str = "test_results_flat",
) -> dict:
    """Dispatch to the per-``envelope_kind`` wrapper.

    Defaults to ``test_results_flat`` to preserve pair-task behavior for
    callers that don't pass ``envelope_kind``.
    """
    wrapper = _ENVELOPE_WRAPPERS.get(envelope_kind, _wrap_envelope_test_results_flat)
    return wrapper(
        raw_results_path,
        model=model, dataset=dataset, task=task, seed=seed,
        head_type=head_type, cfg_hyperparams=cfg_hyperparams,
    )


def _resolve_labels_path(args: argparse.Namespace) -> Optional[Path]:
    """Auto-stage if --labels-path is missing and --no-auto-stage is not set.

    Returns the resolved labels.json path, or None if the caller should let
    build_command raise a clear error.
    """
    if args.labels_path:
        return Path(args.labels_path)
    if args.no_auto_stage:
        return None

    suite = _TASK_TO_SUITE.get(args.task)
    if suite is None:
        return None
    from trl_bench.data.stage import stage_dataset
    try:
        base = stage_dataset(
            suite=suite, task=args.task, dataset=args.dataset,
            data_root=args.data_root,
        )
    except NotImplementedError as exc:
        print(f"auto-stage unavailable: {exc}", file=sys.stderr)
        return None
    # Most stagers write labels.json at the returned ``base``. The DLTE stager
    # writes it one level deeper (under ``base/datasets/dlte_v1/``) so the
    # dispatcher's 3-parent walk-up from labels_path lands on ``base`` (= the
    # runner's ``--project_root``). We probe the deeper path first when the
    # suite is dlte to honor that convention.
    if suite == "dlte":
        labels = base / "datasets" / "dlte_v1" / "labels.json"
        if labels.exists():
            print(f"auto-staged {suite}/{args.dataset} -> {base}", file=sys.stderr)
            return labels
    labels = base / "labels.json"
    print(f"auto-staged {suite}/{args.dataset} -> {base}", file=sys.stderr)
    return labels if labels.exists() else None


def _embeddings_ready(derived: Path, task: str) -> bool:
    """Whether the Stage-1 output at ``derived`` is complete enough to skip
    auto-extract.

    For ``row_prediction`` the Stage-1 output is a DIRECTORY (the
    ``unified_row_embedding`` format: ``metadata.json`` + per-split/-label
    ``.npy``). A bare ``.exists()`` check is wrong here: a prior *failed* run
    can leave an EMPTY directory, whose existence would otherwise short-circuit
    generation and make the probe fail later with a misleading ``Available: []``
    (masking the real Stage-1 error). So require the dir to be populated —
    ``metadata.json`` (generated) or ``not_applicable.json`` (legitimately
    skipped, e.g. TabTransformer on an all-continuous dataset).

    For every other task the Stage-1 output is a single pickle file, where
    existence is sufficient.
    """
    if task == "row_prediction":
        return derived.is_dir() and (
            (derived / "metadata.json").exists()
            or (derived / "not_applicable.json").exists()
        )
    return derived.exists()


def _auto_extract_queries(
    args: argparse.Namespace, *, labels_path: Path, table_pkl: Path,
) -> bool:
    """Auto-run the query-side text encoder for table_retrieval.

    Idempotent: skips per-split if the queries pickle already exists.
    Returns True iff both ``queries_train.pkl`` and ``queries_dev.pkl`` exist
    alongside ``table_pkl`` after this call. Returns False on dispatch
    failure; the caller surfaces a clear error.

    The retrieval dispatcher in ``registry.py::_build_retrieval_cmd`` derives
    these paths as ``<embeddings_path.parent>/queries_{train,dev}.pkl`` so we
    write to the same sibling-of-table-pkl location.
    """
    # Cross-encoder: the QUERY side uses the encoder named in args.probe
    # ("<enc>" or "<enc>_modelonly"), NOT the table model under test.
    pr = args.probe or "mpnet"
    encoder = pr[: -len("_modelonly")] if pr.endswith("_modelonly") else pr
    if encoder not in _QUERY_ENCODER_EXTRACTORS:
        wired = sorted(_QUERY_ENCODER_EXTRACTORS)
        print(
            f"auto-extract: query encoder {encoder!r} (from --probe {args.probe!r}) "
            f"is not wired in `_QUERY_ENCODER_EXTRACTORS` (registry.py). "
            f"Wired: {wired}. For table_retrieval, --probe must be one of "
            f"mpnet/sentence_t5 (+ optional _modelonly suffix).",
            file=sys.stderr,
        )
        return False
    labels_dir = labels_path.parent
    questions_train = labels_dir / "train.json"
    questions_dev   = labels_dir / "dev.json"
    if not questions_train.exists() or not questions_dev.exists():
        print(
            f"auto-extract: expected train.json + dev.json next to "
            f"labels.json at {labels_dir}, not found; the table_retrieval "
            f"stager (nq_tables) should produce these. Pass --embeddings-path "
            f"and pre-extract the queries manually otherwise.",
            file=sys.stderr,
        )
        return False
    # write to the encoder's query dir, matching _build_retrieval_cmd's
    # derivation: <emb_root>/table_retrieval/<encoder>/queries_{train,dev}.pkl
    emb_root = table_pkl.parent.parent.parent
    queries_dir = emb_root / "table_retrieval" / encoder
    queries_dir.mkdir(parents=True, exist_ok=True)
    for split_name, q_json, q_pkl_name in (
        ("train", questions_train, "queries_train.pkl"),
        ("dev",   questions_dev,   "queries_dev.pkl"),
    ):
        q_pkl = queries_dir / q_pkl_name
        if q_pkl.exists():
            continue
        cmd = build_query_extractor_command(
            model=encoder, input_json=q_json, output_path=q_pkl,
            device=args.extract_device,
        )
        print(f"auto-extract: running query encoder "
              f"({encoder}, {args.dataset}, {split_name}) -> {q_pkl}",
              file=sys.stderr)
        rc = subprocess.run(cmd, check=False).returncode
        if rc != 0:
            print(f"auto-extract: query encoder ({split_name}) failed with "
                  f"rc={rc}", file=sys.stderr)
            return False
    return (queries_dir / "queries_train.pkl").exists() and \
           (queries_dir / "queries_dev.pkl").exists()


def _auto_extract_questions(
    args: argparse.Namespace, *, labels_path: Path, column_pkl: Path,
) -> bool:
    """Auto-run the token-mode QUESTION encoder for semantic_parsing.

    Idempotent: skips per-split if the questions pickle already exists. Returns
    True iff ``questions_{train,dev,test}.pkl`` all exist under
    ``<column_pkl.parent>/<setting>/`` after this call -- the exact location the
    MAPO decoder (``registry._build_semparse_cmd``) reads. The WikiTableQuestions
    split JSONLs live next to labels.json at
    ``<labels_path.parent>/data_split_1/<split>_split.jsonl`` (the same files
    ``run_test`` consumes).

    Returns False (with a clear message) when ``setting`` has no token-level
    encoder (openai's API is single-vector cls; table models have none -> use
    ``--setting sentence_t5``/``mpnet``) or a split's source JSONL is missing.
    """
    questions_dir = column_pkl.parent / args.setting
    splits_dir = labels_path.parent / "data_split_1"
    questions_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        q_pkl = questions_dir / f"questions_{split}.pkl"
        if q_pkl.exists():
            continue
        in_json = splits_dir / f"{split}_split.jsonl"
        if not in_json.exists():
            print(
                f"auto-extract: semantic_parsing question source {in_json} "
                f"not found; the wiki_table_questions stager should produce "
                f"data_split_1/<split>_split.jsonl. Pre-extract the question "
                f"pickles manually otherwise.",
                file=sys.stderr,
            )
            return False
        try:
            cmd = build_question_extractor_command(
                model=args.setting, input_json=in_json, output_path=q_pkl,
                device=args.extract_device,
            )
        except SettingError as exc:
            print(f"auto-extract: {exc}", file=sys.stderr)
            return False
        print(f"auto-extract: encoding semantic_parsing questions "
              f"({args.setting}, {split}) -> {q_pkl}", file=sys.stderr)
        rc = subprocess.run(cmd, check=False).returncode
        if rc != 0:
            print(f"auto-extract: question encoder ({split}) failed with "
                  f"rc={rc}", file=sys.stderr)
            return False
    return all((questions_dir / f"questions_{s}.pkl").exists()
               for s in ("train", "dev", "test"))


def _resolve_question_encoder(task: str, setting: str) -> str:
    """Normalize the semantic_parsing question encoder.

    For semantic_parsing the ``setting`` axis IS the question encoder, which the
    benchmark restricts to ``_SEMPARSE_QUESTION_ENCODERS`` ({sentence_t5, mpnet},
    default sentence_t5) -- the model under test supplies columns, not questions.
    Any other value (the gen_grid_cells ``setting=model`` diagonal, or a
    table-aggregation mode like ``cls_embedding``) normalizes to the default so
    the decoder and the question auto-extract agree. Non-semparse tasks are
    returned unchanged.
    """
    if task != "semantic_parsing":
        return setting
    return setting if setting in _SEMPARSE_QUESTION_ENCODERS else "sentence_t5"


# Column extractors all drop columns on some tables: TABBIE's 2-D grid caps at
# MAX_COLS=20, TAPAS/TaBERT truncate wide tables at the 512-token limit, and even
# per-column models (BERT/GTE/starmie) occasionally fail a column. The paper's
# round-generation pipeline ran ``utils/embedding_repair`` (scan + repair) after
# EVERY column extraction to backfill them (re-embedding the missing column
# subsets through the same extractor -- recursively, one column at a time in the
# tail). We mirror that for every model with a repair adapter. No-op when nothing
# is missing (the scan finds no gaps).
# (max_rows, chunk_size) for the repair pass, per model -- these mirror the
# paper's round-generation repair invocations exactly. ``max_rows`` matches each
# extractor's own --max_rows (run.py's extraction uses the extractor defaults,
# which equal these: bert/gte/tapas/tabert=100, starmie=1000, tabbie=30); ``None``
# means the extractor takes no --max_rows (tabsketchfm). ``chunk_size`` is the
# paper's repair chunk size (TaBERT used 16; everything else 64). Membership ==
# "this model's extractor can drop columns AND has a repair adapter".
_REPAIR_PARAMS: dict = {
    "bert":        (100, 64),
    "gte":         (100, 64),
    "tapas":       (100, 64),
    "tabert":      (100, 16),
    "starmie":     (1000, 64),
    "tabsketchfm": (None, 64),
    "tabbie":      (30, 64),
}


def _repair_checkpoint(model: str, dataset: str, ckpt_root) -> Optional[str]:
    """Checkpoint to pass to the repair CLI -- the same one the extraction used.

    Returns None for HuggingFace models (BERT/GTE/TAPAS have no local checkpoint;
    the CLI then resolves the model name from slurm/config/models.yaml, exactly as
    the round-generation scripts did). starmie's checkpoint is per-dataset.
    """
    if model == "starmie":
        return str(starmie_checkpoint_path(dataset, ckpt_root))
    template = _MODEL_EXTRACTORS[model].checkpoint_template
    if template and "{" not in template:
        return str(Path(ckpt_root) / template)
    return None


def _build_column_repair_command(
    model: str, dataset: str, embeddings_pkl, checkpoint=None,
    *, device: Optional[str] = None, chunk_size: int = 64,
    max_rows: Optional[int] = None,
) -> list[str]:
    """Build the embedding-repair CLI command for a column extraction.

    ``--action repair`` scans the pickle for columns the extractor dropped (a
    grid cap, a token-limit truncation, or a per-column failure) and re-embeds
    them via the model's repair adapter. ``chunk_size`` and ``max_rows`` must
    match the model's extraction (see ``_REPAIR_PARAMS``). ``max_rows=None`` omits
    ``--max_rows`` (the extractor takes none); ``checkpoint=None`` omits
    ``--checkpoint`` so the CLI resolves the default (HuggingFace models).
    """
    cmd = [
        sys.executable, "-m", "trl_bench.utils.embedding_repair.cli",
        "--model", model, "--dataset", dataset,
        "--action", "repair",
        "--embeddings", str(embeddings_pkl),
        "--chunk_size", str(chunk_size),
    ]
    if max_rows is not None:
        cmd += ["--max_rows", str(max_rows)]
    if checkpoint is not None:
        cmd += ["--checkpoint", str(checkpoint)]
    if device:
        cmd += ["--device", device]
    return cmd


def _maybe_repair_columns(
    args: argparse.Namespace, column_pkl, ckpt_root,
) -> bool:
    """Run the embedding-repair pass after a column extraction.

    Returns True on success (incl. the no-op case for models without a repair
    adapter). Backfills columns the extractor dropped (cap / token-limit /
    failure).
    """
    if args.model not in _REPAIR_PARAMS:
        return True
    max_rows, chunk_size = _REPAIR_PARAMS[args.model]
    checkpoint = _repair_checkpoint(args.model, args.dataset, ckpt_root)
    cmd = _build_column_repair_command(
        args.model, args.dataset, column_pkl, checkpoint,
        device=args.extract_device, chunk_size=chunk_size, max_rows=max_rows,
    )
    print(f"auto-extract: scanning/repairing dropped columns "
          f"({args.model}, {args.dataset}) -> {column_pkl}", file=sys.stderr)
    return subprocess.run(cmd, check=False).returncode == 0


def _resolve_embeddings_path(
    args: argparse.Namespace, *, labels_path: Optional[Path],
) -> Optional[Path]:
    """Resolve --embeddings-path; auto-orchestrate Stage 1+2 when possible.

    Resolution order:

      1. ``--embeddings-path`` explicit -> use it.
      2. ``<embeddings-dir>/table/<model>/<dataset>.pkl`` exists -> use it.
      3. ``args.model`` has a ``TableEncoderConfig`` wired (see registry's
         ``_TABLE_ENCODERS``) -> auto-run table-DIRECT Stage-1 (mpnet,
         sentence_t5, tapex). The runner writes the table-level pickle
         in a single pass; Stage-2 is skipped (no column pickle exists).
      4. ``args.model`` has an ``ExtractorConfig`` wired (see registry's
         ``_MODEL_EXTRACTORS``) -> auto-run Stage-1 column extraction
         (skipped if the column pickle is already present) and Stage-2
         table aggregation (idempotent — skips if the table pickle is
         present), then return the table pickle path.
      5. Otherwise, return ``None`` and let ``main`` print a clear error.

    For ``task == "table_retrieval"``: after the table pickle resolves,
    additionally auto-orchestrate the query-side encoder (mpnet / sentence_t5
    / bert / gte / openai) over ``train.json`` + ``dev.json`` (siblings of
    labels.json) to produce ``queries_{train,dev}.pkl`` alongside the table
    pickle. The retrieval dispatcher reads from
    ``<embeddings_path.parent>/queries_{train,dev}.pkl`` by convention; auto-
    extracting here ensures the dispatcher's lookup succeeds.

    Stage-1 reads ``<labels.json>``'s sibling ``tables_all/`` directory; this
    relies on the Stage-0 staged layout (see ``trl_bench.data.stage``).
    """
    # The table-pickle resolution is unchanged from the pre-2026-05-20 logic
    # except that for ``task == "table_retrieval"`` we additionally
    # orchestrate the query-side encoder before returning the table pickle.
    if args.embeddings_path:
        table_pkl: Optional[Path] = Path(args.embeddings_path)
    else:
        embeddings_dir = Path(args.embeddings_dir)
        # DLTE convention: the runner reads <dataset>_queries.pkl and
        # <dataset>_targets.pkl from <embeddings_dir>/table/<model>/, not a
        # single <dataset>.pkl. Probe for the queries pickle and use it as
        # the sentinel ``embeddings_path``; the dispatcher derives the rest
        # from its grandparent (the table-embedding root) via _build_dlte_*.
        #
        # Col-granularity tasks consume the column pickle directly — Stage-2
        # (aggregator) is not the right level of abstraction for them.
        # We route those to ``column/<model>/<ds>.pkl``. Two families here:
        #   - deterministic col tasks (column_clustering / schema_matching /
        #     union_search / join_search) — no probe head.
        #   - learned col probes whose runners load 'column_embeddings':
        #     column_type_prediction (train_ct_mode4.py:55),
        #     column_relation_prediction (csv_relation_pipeline.py:17),
        #     semantic_parsing (run_training.py --column-pkl),
        #     join_containment (run_task.py --embedding_type=column; reads the
        #       per-column 'column_embeddings'/'column_names' that the Stage-2
        #       aggregator drops — see registry _SETTING_TO_EMBEDDING_TYPE).
        #       Must route to the column pickle, NOT
        #       the table pickle, or run_task raises "column embeddings None".
        if args.task in ("dlte_retrieval", "dlte_alignment", "dlte_merge"):
            derived_pkl = (embeddings_dir / "table" / args.model
                           / f"{args.dataset}_queries.pkl")
        elif args.task in (
            "column_clustering", "schema_matching",
            "union_search", "join_search", "join_search_learned",
            "column_type_prediction", "column_relation_prediction",
            "semantic_parsing", "join_containment",
        ):
            derived_pkl = embeddings_dir / "column" / args.model / f"{args.dataset}.pkl"
        elif args.task == "row_prediction":
            # row_prediction uses the unified_row_embedding format (a DIRECTORY,
            # not a pickle): metadata.json + per-split .npy + per-label .npy.
            derived_pkl = embeddings_dir / "row_prediction" / args.model / args.dataset
        elif args.task == "record_linkage":
            derived_pkl = embeddings_dir / "row" / args.model / f"{args.dataset}.pkl"
        else:
            derived_pkl = embeddings_dir / "table" / args.model / f"{args.dataset}.pkl"
        if _embeddings_ready(derived_pkl, args.task):
            table_pkl = derived_pkl
        elif args.model in _TABLE_ENCODERS and labels_path is not None:
            # Table-direct Stage-1: the runner writes the table-level pickle
            # in one pass. No column pickle, no Stage-2 aggregator. Used for
            # mpnet, sentence_t5 (text encoders that linearize the table) and
            # tapex (BART encoder, table-only output by design).
            tables_all = labels_path.parent / "tables_all"
            if not tables_all.exists():
                # Fallback: some datasets stage their CSVs under tables/ rather
                # than tables_all/ (e.g. valentine, wtq). Use tables/ when the
                # canonical tables_all/ is absent; tables_all/ still wins when
                # both exist, so this never overrides the full set.
                _alt = labels_path.parent / "tables"
                if _alt.exists():
                    tables_all = _alt
            if not tables_all.exists():
                print(
                    f"auto-extract: expected tables_all/ next to labels.json at "
                    f"{tables_all}, not found; pass --embeddings-path or pre-stage "
                    f"the data manually.",
                    file=sys.stderr,
                )
                table_pkl = None
            else:
                derived_pkl.parent.mkdir(parents=True, exist_ok=True)
                cmd = build_table_encoder_command(
                    model=args.model, dataset=args.dataset,
                    input_dir=tables_all, output_path=derived_pkl,
                )
                print(f"auto-extract: running table-direct Stage-1 "
                      f"({args.model}, {args.dataset}) -> {derived_pkl}",
                      file=sys.stderr)
                rc = subprocess.run(cmd, check=False).returncode
                if rc != 0:
                    print(f"auto-extract: table-direct Stage-1 failed with "
                          f"rc={rc}", file=sys.stderr)
                    return None
                table_pkl = derived_pkl if derived_pkl.exists() else None
        elif args.task == "row_prediction" and labels_path is not None:
            # row_prediction has its own Stage-1 pipeline, driven entirely from
            # `slurm/config/row_data_models.yaml` via `build_row_data_commands`
            # -- the faithful port of `slurm/generate_row_data_scripts.py`. Every
            # row-data model resolves through it (NOT a hardcoded per-runner CLI),
            # so per-model arg-name quirks and licensed checkpoints come from the
            # one source of truth the slurm path used:
            #   - pretrained text encoders (bert/gte): one generate pass, HF
            #     model id + tokenization flags from the YAML defaults (which
            #     equal the runners' argparse defaults -> identical output);
            #   - pretrained licensed models (tabbie/tuta): one generate pass
            #     PLUS `--model_path <ckpt>` (their runners REQUIRE it; tuta also
            #     uses native `--dataset_dir`/`--output_dir` arg names);
            #   - pretrained checkpoint-less (tabicl/tabpfn): one generate pass;
            #   - trained SSL (scarf/dae/subtab/vime/saint/transtab/
            #     tabular_binning/tabtransformer): [train, generate] -- a train
            #     pass writes a per-dataset checkpoint, then generate loads it.
            # All write the `unified_row_embedding` DIRECTORY (metadata.json +
            # per-split .npy + per-label .npy) the probe trainer consumes.
            # `--label_policy manifest` (in every model's YAML defaults) makes the
            # extractor read label_columns from the staged labels.json; without
            # it the probe later rejects the cell with `Available: []`.
            derived_pkl.mkdir(parents=True, exist_ok=True)
            ckpt_root = args.checkpoint_root or os.environ.get(
                "TRL_BENCH_CKPT_ROOT", "./checkpoints")
            # Per-dataset TRAINING-checkpoint dir (used only by trained SSL
            # models' --checkpoint_dir); the licensed PRETRAINED checkpoint
            # (--model_path for tabbie/tuta) resolves from `checkpoint_root`.
            ckpt_dir = Path(ckpt_root) / "row_data" / args.model / args.dataset
            try:
                cmds = build_row_data_commands(
                    args.model, labels_path.parent, ckpt_dir, derived_pkl,
                    checkpoint_root=ckpt_root)
            except SettingError as exc:
                print(f"auto-extract: row_prediction for {args.model!r} not "
                      f"available: {exc}", file=sys.stderr)
                return None
            for i, stage_cmd in enumerate(cmds, 1):
                print(f"auto-extract: row_prediction Stage-1 step {i}/{len(cmds)} "
                      f"({args.model}, {args.dataset}) -> {derived_pkl}",
                      file=sys.stderr)
                rc = subprocess.run(stage_cmd, check=False).returncode
                if rc != 0:
                    print(f"auto-extract: row_prediction Stage-1 step {i} failed "
                          f"with rc={rc}", file=sys.stderr)
                    return None
            # Probe expects a directory (not a pickle) -- return it as-is.
            table_pkl = derived_pkl if derived_pkl.exists() else None
        elif args.task == "record_linkage" and labels_path is not None:
            # record_linkage: keeps the DLTE-style raw row pickle (one per-table
            # row-embedding pickle). Input is the staged dataset dir's tables/
            # subdir (Rbench record-linkage layout: data/<ds>/tables/{tableA,
            # tableB}.csv), NOT the dataset root. Pure-row models'
            # `_MODEL_EXTRACTORS` runner IS the row extractor; dual-granularity
            # models (bert/gte/openai/tabbie/tuta) override via `_ROW_RUNNERS`
            # to use `generate_row_embeddings.py`.
            if "row" not in _MODEL_GRANULARITIES.get(args.model, set()):
                print(
                    f"auto-extract: {args.model} is not row-capable; pass "
                    f"--embeddings-path with a pre-extracted row pickle.",
                    file=sys.stderr,
                )
                table_pkl = None
            elif args.model not in _MODEL_EXTRACTORS:
                table_pkl = None
            else:
                row_input = labels_path.parent / "tables"
                if not row_input.exists():
                    print(
                        f"auto-extract: expected tables/ next to labels.json "
                        f"at {row_input}, not found; pre-stage the data "
                        f"(record_linkage staging puts CSVs in tables/).",
                        file=sys.stderr,
                    )
                    return None
                derived_pkl.parent.mkdir(parents=True, exist_ok=True)
                ckpt_root = args.checkpoint_root or os.environ.get(
                    "TRL_BENCH_CKPT_ROOT", "./checkpoints"
                )
                cmd = build_extractor_command(
                    model=args.model, dataset=args.dataset,
                    input_dir=row_input, output_path=derived_pkl,
                    device=args.extract_device, checkpoint_root=ckpt_root,
                    runner_override=_ROW_RUNNERS.get(args.model),
                )
                print(f"auto-extract: running record_linkage Stage-1 "
                      f"({args.model}, {args.dataset}) -> {derived_pkl}",
                      file=sys.stderr)
                rc = subprocess.run(cmd, check=False).returncode
                if rc != 0:
                    print(f"auto-extract: record_linkage Stage-1 failed with "
                          f"rc={rc}", file=sys.stderr)
                    return None
                table_pkl = derived_pkl if derived_pkl.exists() else None
        elif args.model not in _MODEL_EXTRACTORS or labels_path is None:
            table_pkl = None
        else:
            tables_all = labels_path.parent / "tables_all"
            if not tables_all.exists():
                # Fallback: some datasets stage their CSVs under tables/ rather
                # than tables_all/ (e.g. valentine, wtq). Use tables/ when the
                # canonical tables_all/ is absent; tables_all/ still wins when
                # both exist, so this never overrides the full set.
                _alt = labels_path.parent / "tables"
                if _alt.exists():
                    tables_all = _alt
            if not tables_all.exists():
                print(
                    f"auto-extract: expected tables_all/ next to labels.json at "
                    f"{tables_all}, not found; pass --embeddings-path or pre-stage "
                    f"the data manually.",
                    file=sys.stderr,
                )
                table_pkl = None
            elif args.model in _TABLE_NATIVE_RUNNERS:
                # TABLE-native models (tuta) emit the table pickle DIRECTLY via a
                # single native forward pass that populates cls_embedding. Skip
                # the column-extract + Stage-2-aggregator path entirely: that
                # path derives the table embedding from the column pickle, but
                # tuta's column pickle holds only ROW embeddings, so the
                # aggregator produced an all-None table_embedding dict (every
                # cls cell then crashed). The native runner reuses tuta's
                # ExtractorConfig (--model_path/--device_id) via runner_override.
                derived_pkl.parent.mkdir(parents=True, exist_ok=True)
                ckpt_root = args.checkpoint_root or os.environ.get(
                    "TRL_BENCH_CKPT_ROOT", "./checkpoints"
                )
                cmd = build_extractor_command(
                    model=args.model, dataset=args.dataset,
                    input_dir=tables_all, output_path=derived_pkl,
                    device=args.extract_device, checkpoint_root=ckpt_root,
                    runner_override=_TABLE_NATIVE_RUNNERS[args.model],
                )
                print(f"auto-extract: running TABLE-native Stage-1 "
                      f"({args.model}, {args.dataset}) -> {derived_pkl}",
                      file=sys.stderr)
                rc = subprocess.run(cmd, check=False).returncode
                if rc != 0:
                    print(f"auto-extract: TABLE-native Stage-1 failed with "
                          f"rc={rc}", file=sys.stderr)
                    return None
                table_pkl = derived_pkl if derived_pkl.exists() else None
            else:
                column_pkl = embeddings_dir / "column" / args.model / f"{args.dataset}.pkl"
                column_pkl.parent.mkdir(parents=True, exist_ok=True)
                if not column_pkl.exists():
                    ckpt_root = args.checkpoint_root or os.environ.get(
                        "TRL_BENCH_CKPT_ROOT", "./checkpoints"
                    )
                    # starmie ships no released checkpoint -- each (model,dataset)
                    # cell trains its own contrastive encoder first. Auto-pretrain
                    # on a cache miss so `run.py --model starmie ...` is one-command
                    # runnable (else build_extractor_command raises SettingError:
                    # checkpoint not found). The .pt lands exactly where the
                    # extractor template resolves it (shared checkpoint-path helper).
                    if args.model == "starmie" and not starmie_checkpoint_path(
                            args.dataset, ckpt_root).exists():
                        pre = build_starmie_pretrain_command(
                            args.dataset, tables_all, ckpt_root)
                        print(f"auto-extract: no starmie checkpoint for "
                              f"{args.dataset!r}; pretraining first (one-time "
                              f"per-dataset cost) -> "
                              f"{starmie_checkpoint_path(args.dataset, ckpt_root)}",
                              file=sys.stderr)
                        prc = subprocess.run(pre, check=False).returncode
                        if prc != 0:
                            print(f"auto-extract: starmie pretrain failed "
                                  f"rc={prc}", file=sys.stderr)
                            return None
                    cmd = build_extractor_command(
                        model=args.model, dataset=args.dataset,
                        input_dir=tables_all, output_path=column_pkl,
                        device=args.extract_device,
                        checkpoint_root=ckpt_root,
                    )
                    print(f"auto-extract: running Stage-1 column extraction "
                          f"({args.model}, {args.dataset}) -> {column_pkl}",
                          file=sys.stderr)
                    rc = subprocess.run(cmd, check=False).returncode
                    if rc != 0:
                        print(f"auto-extract: Stage-1 failed with rc={rc}",
                              file=sys.stderr)
                        return None
                    # tabbie's 20-col grid caps wide tables; fill the columns
                    # beyond the cap with the embedding-repair pass that the
                    # paper's round-generation pipeline ran right after this
                    # extraction (no-op for models that don't cap).
                    if not _maybe_repair_columns(args, column_pkl, ckpt_root):
                        print("auto-extract: column repair failed",
                              file=sys.stderr)
                        return None

                # Stage-2 aggregator (idempotent: process_model_dataset skips
                # when the output pickle already exists unless --force is set).
                derived_pkl.parent.mkdir(parents=True, exist_ok=True)
                if not derived_pkl.exists():
                    agg_cmd = [
                        sys.executable, "-m", "trl_bench.scripts.generate_table_embeddings",
                        "--models", args.model,
                        "--datasets", args.dataset,
                        "--column-embeddings-dir", str(embeddings_dir / "column"),
                        "--output-dir",            str(embeddings_dir / "table"),
                    ]
                    print(f"auto-extract: running Stage-2 aggregator -> {derived_pkl}",
                          file=sys.stderr)
                    rc = subprocess.run(agg_cmd, check=False).returncode
                    if rc != 0:
                        print(f"auto-extract: Stage-2 failed with rc={rc}",
                              file=sys.stderr)
                        return None
                table_pkl = derived_pkl if derived_pkl.exists() else None

    if table_pkl is None:
        return None

    # Hybrid table_retrieval also needs the QUERY ENCODER's table embeddings
    # (combined with the model's via create_hybrid in the dispatcher). Auto-
    # extract them too (idempotent). model_only mode skips this (raw model emb).
    if (args.task == "table_retrieval" and labels_path is not None
            and args.probe and not args.probe.endswith("_modelonly")):
        enc = args.probe
        enc_table = table_pkl.parent.parent / enc / "nq_tables.pkl"
        if not enc_table.exists() and enc in _TABLE_ENCODERS:
            tables_all = labels_path.parent / "tables_all"
            if tables_all.exists():
                enc_table.parent.mkdir(parents=True, exist_ok=True)
                enc_cmd = build_table_encoder_command(
                    model=enc, dataset=args.dataset,
                    input_dir=str(tables_all), output_path=str(enc_table),
                )
                print(f"auto-extract: hybrid query-encoder table embeddings "
                      f"({enc}, {args.dataset}) -> {enc_table}", file=sys.stderr)
                rc = subprocess.run(enc_cmd, check=False).returncode
                if rc != 0:
                    print(f"auto-extract: encoder table embeddings failed rc={rc}",
                          file=sys.stderr)

    # Auto-orchestrate query-side encoding for table_retrieval. Idempotent;
    # skipped per-split when the queries pickle already exists. Only attempt
    # when the table pickle ACTUALLY exists on disk (callers may pass an
    # explicit non-existent path in tests / for staged dispatches; we honor
    # that by short-circuiting query extraction).
    if (args.task == "table_retrieval" and labels_path is not None
            and table_pkl.exists()):
        if not _auto_extract_queries(
            args, labels_path=labels_path, table_pkl=table_pkl,
        ):
            # If query auto-extract failed but the queries already exist
            # from a prior run (e.g. user pre-extracted manually) we still
            # let the caller proceed; the dispatcher will fail later with a
            # clear missing-file error if they don't.
            _pr = args.probe or "mpnet"
            _enc = _pr[: -len("_modelonly")] if _pr.endswith("_modelonly") else _pr
            _qdir = table_pkl.parent.parent.parent / "table_retrieval" / _enc
            q_train = _qdir / "queries_train.pkl"
            q_dev   = _qdir / "queries_dev.pkl"
            if not (q_train.exists() and q_dev.exists()):
                print(
                    f"auto-extract: table pickle resolved but query-side "
                    f"encoding failed; the retrieval dispatcher requires "
                    f"{q_train} + {q_dev} to exist. Pre-extract manually or "
                    f"resolve the upstream error above.",
                    file=sys.stderr,
                )

    # Auto-orchestrate QUESTION-side encoding for semantic_parsing. The MAPO
    # decoder loads per-token question embeddings from
    # <column_pkl.parent>/<setting>/questions_{train,dev,test}.pkl; generate them
    # (idempotent) from the WTQ split JSONLs next to labels.json. Only attempt
    # when the column pickle actually exists (mirrors table_retrieval).
    if (args.task == "semantic_parsing" and labels_path is not None
            and table_pkl.exists()):
        if not _auto_extract_questions(
            args, labels_path=labels_path, column_pkl=table_pkl,
        ):
            q_dir = table_pkl.parent / args.setting
            needed = [q_dir / f"questions_{s}.pkl"
                      for s in ("train", "dev", "test")]
            if not all(p.exists() for p in needed):
                print(
                    f"auto-extract: column pickle resolved but question-side "
                    f"encoding failed; the MAPO decoder requires "
                    f"questions_{{train,dev,test}}.pkl under {q_dir} to exist. "
                    f"Pre-extract manually or resolve the upstream error above.",
                    file=sys.stderr,
                )
    return table_pkl


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # semantic_parsing's setting axis is the question encoder (sentence_t5/mpnet,
    # default sentence_t5); the model under test supplies columns. Normalize so
    # the decoder + question auto-extract use the same encoder.
    _qenc = _resolve_question_encoder(args.task, args.setting)
    if _qenc != args.setting:
        print(f"semantic_parsing: '{args.setting}' is not a designated question "
              f"encoder; using '{_qenc}' (the model supplies columns; "
              f"sentence_t5/mpnet supply questions).", file=sys.stderr)
        args.setting = _qenc

    labels_path = _resolve_labels_path(args)
    embeddings_path = _resolve_embeddings_path(args, labels_path=labels_path)

    if labels_path is None or embeddings_path is None:
        missing = []
        if labels_path is None:
            missing.append("labels (Stage-0 output): pass --labels-path or let "
                           "--auto-stage materialize it")
        if embeddings_path is None:
            derived = (Path(args.embeddings_dir) / "table" / args.model
                       / f"{args.dataset}.pkl")
            wired = sorted(_MODEL_EXTRACTORS)
            missing.append(
                f"table-embeddings (Stage-1+2 output): expected at {derived} "
                f"or pass --embeddings-path. Models with auto-extract wired: "
                f"{wired}; other models must pre-extract per "
                f"src/trl_bench/models/<m>/USAGE.md."
            )
        for m in missing:
            print(f"error: missing {m}", file=sys.stderr)
        return 2

    try:
        stages = build_command(
            model=args.model, task=args.task, dataset=args.dataset,
            setting=args.setting, probe=args.probe, seed=args.seed,
            embeddings_path=embeddings_path,
            labels_path=labels_path,
            results_dir=args.results_dir,
            embeddings_dir=args.embeddings_dir,
            configs_root=args.configs_root,
            data_root=args.data_root,
        )
    except SettingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Deterministic stdout-emitting tasks (column_clustering / union_search /
    # join_search) need their stdout teed to ``stage_run.log`` inside the
    # per-cell output directory so the envelope wrapper can regex-extract the
    # metrics. schema_matching writes its own ``results.json`` and needs no
    # stdout capture (the runner takes ``--output_dir`` directly).
    stdout_capture_path: Optional[Path] = None
    if args.task in _STDOUT_CAPTURE_TASKS:
        from trl_bench.registry import _TASK_PROBE_CONFIG as _det_probe_config
        _det_cfg = _det_probe_config.get(args.task)
        if _det_cfg is not None and _det_cfg.path_layout == "deterministic":
            _det_out_dir = (Path(args.results_dir) / "evaluation"
                            / args.task / args.model)
            _det_out_dir.mkdir(parents=True, exist_ok=True)
            stdout_capture_path = _det_out_dir / "stage_run.log"

    for stage in stages:
        if stdout_capture_path is not None:
            with open(stdout_capture_path, "wb") as _log_fd:
                completed = subprocess.run(
                    stage, check=False,
                    stdout=_log_fd, stderr=subprocess.STDOUT,
                )
        else:
            completed = subprocess.run(stage, check=False)
        if completed.returncode != 0:
            return completed.returncode

    if (args.probe is not None or args.task == "semantic_parsing"
            or args.task == "join_search_learned"
            or args.task in ("dlte_retrieval", "dlte_alignment", "dlte_merge")
            or args.task in DETERMINISTIC_TASKS):
        from trl_bench.registry import _TASK_PROBE_CONFIG, _SETTING_TO_EMBEDDING_TYPE
        results_dir = Path(args.results_dir)
        cfg = _TASK_PROBE_CONFIG.get(args.task)
        if cfg is not None:
            # Per-task results.json + envelope path layout (mirrors registry).
            if cfg.path_layout == "without_setting":
                stage3_dir = (results_dir / "evaluation" / args.task / args.model
                              / f"seed{args.seed}" / args.probe)
                envelope_dir = (results_dir / "evaluation" / args.task / args.model
                                / args.probe)
            elif cfg.path_layout == "with_dataset":
                stage3_dir = (results_dir / "evaluation" / args.task / args.model
                              / args.dataset / f"seed{args.seed}" / args.probe)
                envelope_dir = (results_dir / "evaluation" / args.task / args.model
                                / args.probe)
            elif cfg.path_layout == "with_dataset_label":
                # row_prediction: <task>/<model>/<dataset>/seed<S>/<label_col>/.
                # No probe-axis in the directory; envelope sits under <label_col>.
                stage3_dir = (results_dir / "evaluation" / args.task / args.model
                              / args.dataset / f"seed{args.seed}" / args.setting)
                envelope_dir = (results_dir / "evaluation" / args.task / args.model
                                / args.dataset / args.setting)
            elif cfg.path_layout == "semparse":
                # semantic_parsing: <task>/<model>/<setting>/seed<S>/ (no probe
                # axis; setting is the query-encoder model). Raw input is
                # ``test.log`` (JSON), not ``results.json``.
                stage3_dir = (results_dir / "evaluation" / args.task / args.model
                              / args.setting / f"seed{args.seed}")
                envelope_dir = (results_dir / "evaluation" / args.task / args.model
                                / args.setting)
            elif cfg.path_layout == "deterministic":
                # Deterministic training-free tasks (column_clustering /
                # schema_matching / union_search / join_search): the
                # reference layout is FLAT —
                #   <results>/evaluation/<task>/<model>/<model>_<dataset>.json
                # with no seed-subdir and no probe-subdir. The raw input is
                # either ``stage_run.log`` (stdout-only runners) or
                # ``<output_dir>/results.json`` (schema_matching), both
                # written to the same directory the envelope is placed in.
                stage3_dir = (results_dir / "evaluation" / args.task / args.model)
                envelope_dir = stage3_dir
            elif cfg.path_layout == "dlte":
                # DLTE: the runners write outputs under
                # ``<results>/evaluation/dlte/{stage1,stage2,metrics}/<key>/``
                # (shared across all three tasks). The envelope is written
                # under ``<results>/evaluation/<task>/<model>/<setting>/``.
                # ``stage3_dir`` here points at the runner's per-task raw
                # output location, derived from (task, model, setting).
                dlte_root = results_dir / "evaluation" / "dlte"
                # Stage-2 / Stage-3 use composite directory keys.
                if args.task == "dlte_retrieval":
                    # Single-axis: <model> (table_model).
                    stage3_dir = dlte_root / "stage1" / args.model
                elif args.task == "dlte_alignment":
                    # Composite key: ``<col>`` (diagonal) or ``<table>__<col>``.
                    if args.setting and args.setting != "diagonal":
                        # Setting carries the table_model.
                        key = (f"{args.setting}__{args.model}"
                               if args.setting != args.model else args.model)
                    else:
                        key = args.model
                    stage3_dir = dlte_root / "stage2" / key
                else:   # dlte_merge
                    # Composite key: ``<col>__<row>__<table>``. Diagonal
                    # default = model fills every axis.
                    if args.setting and args.setting != "diagonal":
                        tokens = args.setting.split("__")
                        if len(tokens) == 2:
                            col_m, table_m = tokens
                        else:
                            col_m, table_m = args.model, args.model
                    else:
                        col_m, table_m = args.model, args.model
                    # step10's ``derive_stage2_key`` collapses table==col to
                    # a single name, then ``combo_name = stage2_key__row_m``.
                    # Mirror that collapse rule so the dispatcher reads from
                    # the same path the runner wrote to.
                    if table_m == col_m:
                        stage2_key = col_m
                    else:
                        stage2_key = f"{table_m}__{col_m}"
                    key = f"{stage2_key}__{args.model}"
                    stage3_dir = dlte_root / "metrics" / key
                envelope_dir = (results_dir / "evaluation" / args.task / args.model
                                / args.setting)
            else:
                stage3_dir = (results_dir / "evaluation" / args.task / args.model
                              / args.setting / f"seed{args.seed}" / args.probe)
                envelope_dir = (results_dir / "evaluation" / args.task / args.model
                                / args.setting / args.probe)
            # The raw-results filename is family-dependent:
            #   * ``test.log``      — semantic_parsing (JSON dump).
            #   * ``end_to_end.json`` — dlte_merge (Stage-3 evaluation output).
            #   * ``stage2.json``     — dlte_alignment (but its raw is actually
            #                           ``metrics_test_topk_100.json``; the
            #                           reference aggregator collates them
            #                           into stage2.json elsewhere).
            #   * ``metrics_test_topk_100.json`` — dlte_retrieval / dlte_alignment.
            #   * ``results.json``    — every other family (default).
            if cfg.envelope_kind == "semparse_flat":
                raw_filename = "test.log"
            elif cfg.envelope_kind == "dlte_merge_flat":
                raw_filename = "end_to_end.json"
            elif cfg.envelope_kind in ("dlte_retrieval_flat", "dlte_alignment_flat"):
                raw_filename = "metrics_test_topk_100.json"
            elif cfg.envelope_kind in (
                "column_clustering_flat", "union_search_flat", "join_search_flat",
            ):
                # Stdout-only runners; main() teed their stdout to stage_run.log.
                raw_filename = "stage_run.log"
            else:
                raw_filename = "results.json"
            raw = stage3_dir / raw_filename
            if raw.exists():
                emb_type_for_hyper = _SETTING_TO_EMBEDDING_TYPE.get(
                    args.setting,
                    args.setting[:-10] if args.setting.endswith("_embedding") else args.setting,
                )
                # Per-family envelope hyperparameter dict, matching the keys
                # observed in each reference envelope:
                #   pair_task -> {task_type, embedding_type, combination_method,
                #                 hidden_dim, max_epochs, learning_rate}
                #   cta       -> {num_epochs, batch_size, learning_rate}
                #   cra       -> {epochs, batch_size, learning_rate, hidden_dim}
                #   retrieval -> {projection_dim, hidden_dim}
                if cfg.family == "cta":
                    hp = {
                        "num_epochs":    cfg.max_epochs,
                        "batch_size":    cfg.batch_size,
                        "learning_rate": cfg.learning_rate,
                    }
                elif cfg.family == "cra":
                    hp = {
                        "epochs":        cfg.max_epochs,
                        "batch_size":    cfg.batch_size,
                        "learning_rate": cfg.learning_rate,
                        "hidden_dim":    cfg.hidden_dim,
                    }
                elif cfg.family == "retrieval":
                    hp = {
                        "projection_dim": cfg.hidden_dim,
                        "hidden_dim":     cfg.hidden_dim,
                    }
                elif cfg.family == "row_prediction":
                    # The runner is YAML-driven; the reference omits a
                    # ``hyperparameters`` block entirely. We pass an empty
                    # dict, and the row_prediction envelope wrapper skips
                    # writing the key when it's falsy.
                    hp = {}
                elif cfg.family == "semparse":
                    # semantic_parsing reference envelope:
                    #   hyperparameters = {seed, beam_size=5}.
                    # The runner's --beam-size default is 5; we mirror it here.
                    hp = {
                        "seed":      args.seed,
                        "beam_size": 5,
                    }
                elif cfg.family in (
                    "dlte_retrieval", "dlte_alignment", "dlte_merge",
                ):
                    # DLTE reference envelopes carry no ``hyperparameters`` block
                    # of their own (the runners are deterministic + path-driven).
                    # We emit an empty dict; the envelope wrappers drop the key
                    # when it's empty.
                    hp = {}
                elif cfg.family == "column_clustering":
                    # Reference envelope: hyperparameters = {k, target_avg_size}.
                    hp = {
                        "k":               cfg.max_epochs,
                        "target_avg_size": cfg.hidden_dim,
                    }
                elif cfg.family == "schema_matching":
                    # Reference envelope:
                    #   hyperparameters = {matching_strategy, threshold}.
                    hp = {
                        "matching_strategy": cfg.combination_method,
                        "threshold":         cfg.learning_rate,
                    }
                elif cfg.family == "union_search":
                    # Reference envelope:
                    #   hyperparameters = {method, k, threshold, ef, N}.
                    # Per-dataset (tus/tus_hard override to HNSW); resolved by
                    # the single source of truth in registry so the recorded
                    # block always matches the CLI the dispatcher ran.
                    usp = union_search_params(args.dataset, cfg)
                    hp = {
                        "method":    usp["method"],
                        "k":         usp["K"],
                        "threshold": usp["threshold"],
                        "ef":        usp["ef"],
                        "N":         usp["N"],
                    }
                elif cfg.family == "join_search":
                    # Reference envelope: hyperparameters = {k}.
                    hp = {
                        "k": cfg.max_epochs,
                    }
                elif cfg.family == "join_search_learned":
                    # Learned-projection variant: training hyperparameters.
                    hp = {
                        "num_layers":    cfg.num_labels,
                        "batch_size":    cfg.batch_size,
                        "max_epochs":    cfg.max_epochs,
                        "learning_rate": cfg.learning_rate,
                        "k":             50,
                    }
                else:
                    # Pair-task / record_linkage / other.
                    hp = {
                        "combination_method": cfg.combination_method,
                        "hidden_dim":         cfg.hidden_dim,
                        "max_epochs":         cfg.max_epochs,
                        "learning_rate":      cfg.learning_rate,
                    }
                    if cfg.pass_task_type:
                        hp = {"task_type": cfg.task_type, **hp}
                    if cfg.pass_embedding_type:
                        # Insert embedding_type just after task_type (or at start).
                        new_hp: dict = {}
                        inserted = False
                        for k, v in hp.items():
                            new_hp[k] = v
                            if k == "task_type" and not inserted:
                                new_hp["embedding_type"] = emb_type_for_hyper
                                inserted = True
                        if not inserted:
                            new_hp = {"embedding_type": emb_type_for_hyper, **new_hp}
                        hp = new_hp

                envelope = _wrap_envelope(
                    raw, model=args.model, dataset=args.dataset, task=args.task,
                    seed=args.seed, head_type=args.probe,
                    cfg_hyperparams=hp,
                    envelope_kind=cfg.envelope_kind,
                )
                envelope_dir.mkdir(parents=True, exist_ok=True)
                # Deterministic tasks: reference filename is
                # ``<model>_<dataset>.json`` (no seed suffix; the runners are
                # deterministic per (model, dataset)).
                if args.task in DETERMINISTIC_TASKS:
                    envelope_path = (envelope_dir
                                     / f"{args.model}_{args.dataset}.json")
                else:
                    envelope_path = (envelope_dir
                                     / f"{args.model}_{args.dataset}_seed{args.seed}.json")
                envelope_path.write_text(json.dumps(envelope, indent=2))
                print(f"wrote reference envelope: {envelope_path}")

    return 0


if __name__ == "__main__":   # pragma: no cover
    raise SystemExit(main())
