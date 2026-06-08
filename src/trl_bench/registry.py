"""(model, task) -> subprocess command dispatch registry.

This module encodes the dispatch knowledge that ties together:
  Stage 0  HF dataset -> on-disk staged layout         (src/trl_bench/data/stage.py)
  Stage 1  per-model embedding extraction              (src/trl_bench/models/<m>/*.py)
  Stage 2  table-embedding aggregator                  (src/trl_bench/scripts/generate_table_embeddings.py)
  Stage 3  per-task probe head training/inference      (src/trl_bench/utils/downstream/run_task.py)

The probe stage (Stage 3) is fully encoded below. Stage 1 dispatch is wired via
``ExtractorConfig`` and the ``build_extractor_command`` factory for the
column-extraction family (BERT, GTE, TAPAS, OpenAI), the pretrained row
family (TabICL, TabPFN), the trained-per-table row family (TransTab, DAE,
SCARF, VIME, SubTab, SAINT, TabularBinning, TabTransformer), and the
licensed-checkpoint family (TaBERT, TabSketchFM, TURL, TUTA, TABBIE,
Starmie — all auto-dispatch once the checkpoint is on-disk under
``$TRL_BENCH_CKPT_ROOT`` / ``--checkpoint-root``). The table-direct
encoders (MPNet, Sentence-T5, TAPEX) emit a table-level embedding in a
single pass instead of following the column->table (Stage-1 extract ->
Stage-2 aggregate) contract, so they auto-dispatch via ``_TABLE_ENCODERS``
/ ``TableEncoderConfig`` with Stage-2 skipped; see each
``src/trl_bench/models/<m>/USAGE.md`` for details.

When ``embeddings_path`` is supplied to ``build_command`` the registry assumes
Stage-1 and Stage-2 have already produced the table-embedding pickle; when it
is omitted, ``trl_bench.run.main`` looks up the model in
``_MODEL_EXTRACTORS`` and orchestrates Stage-1 + Stage-2 automatically.
"""
from __future__ import annotations
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


class SettingError(ValueError):
    """Raised when (model, task, setting) is not a supported combination."""


# == Task categories (mirror slurm/tools/generate_downstream_scripts.py) ======

DLTE_TASKS = frozenset({"dlte_retrieval", "dlte_alignment", "dlte_merge"})

DETERMINISTIC_TASKS = frozenset({
    "column_clustering",
    "schema_matching",
    "union_search",
    "join_search",
})

TABLE_EMBEDDING_TASKS = frozenset({
    "table_retrieval", "table_subset",
    "union_search", "join_search",
})

ROW_EMBEDDING_TASKS = frozenset({"record_linkage", "row_prediction"})

PROBE_TASKS = frozenset({
    "column_type_prediction", "column_relation_prediction",
    "join_classification", "union_classification", "union_regression",
    "join_containment", "table_retrieval", "table_subset",
    "record_linkage", "row_prediction", "semantic_parsing",
    # join_search_learned: learned-projection (InfoNCE) variant of join_search.
    # Custom runner (run_learned_search.py), no probe-head axis, but SEEDED
    # (trained) -> NOT in DETERMINISTIC_TASKS, so its envelope keeps the seed
    # suffix. Routed via build_command's PROBE_TASKS branch -> _probe_command ->
    # the join_search_learned family builder.
    "join_search_learned",
    # DLTE: although the runners are not traditional probes (no head_type
    # axis), their dispatch surface matches PROBE_TASKS' contract:
    # caller provides --embeddings-path and --labels-path, dispatcher returns
    # a command sequence + writes a reference envelope. Membership here
    # routes build_command's PROBE_TASKS branch to _probe_command for DLTE too.
    "dlte_retrieval", "dlte_alignment", "dlte_merge",
})

COSINE_THRESHOLD_TASKS = frozenset({"record_linkage"})
INTERACTION_TASKS = frozenset()

# == Model capability table ===================================================

_MODEL_GRANULARITIES: dict[str, frozenset[str]] = {
    "bert":  frozenset({"col", "row", "table"}),
    "gte":   frozenset({"col", "row", "table"}),
    "mpnet":       frozenset({"col", "table"}),
    "sentence_t5": frozenset({"col", "table"}),
    "openai":      frozenset({"col", "table", "row"}),
    "tapas": frozenset({"col", "table"}),
    "tapex": frozenset({"col", "table"}),
    "tabert": frozenset({"col", "table"}),
    "turl":  frozenset({"col", "table"}),
    # TABBIE + TUTA also expose row embeddings (synthesized via per-row
    # mini-tables; see models/<m>/generate_row_embeddings.py), produced by the
    # slurm row-script pipeline. run.py has no row auto-extract branch for any
    # model, so row cells are probed from a pre-extracted --embeddings-path.
    "tuta":  frozenset({"col", "row", "table"}),
    "tabbie": frozenset({"col", "row", "table"}),
    "tabsketchfm": frozenset({"col", "table"}),
    "starmie": frozenset({"col", "table"}),  # paper reports table_subset + table_retrieval
    "tabicl": frozenset({"row"}),
    "tabpfn": frozenset({"row"}),
    "transtab": frozenset({"row"}),
    "dae":    frozenset({"row"}),
    "scarf":  frozenset({"row"}),
    "switchtab": frozenset({"row"}),
    "vime":   frozenset({"row"}),
    "subtab": frozenset({"row"}),
    "saint":  frozenset({"row"}),
    "tabular_binning": frozenset({"row"}),
    "tabtransformer":  frozenset({"row"}),
}

_TASK_GRANULARITIES: dict[str, str] = {
    # CTbench
    "column_type_prediction":     "col",
    "column_clustering":          "col",
    "column_relation_prediction": "col",
    "join_search":                "col",
    "join_search_learned":        "col",
    "schema_matching":            "col",
    "join_containment":           "col",
    "join_classification":        "col",
    "union_search":               "col",
    "union_classification":       "col",
    "union_regression":           "col",
    "table_subset":               "table",
    "table_retrieval":            "table",
    "semantic_parsing":           "col",
    # RBench
    "row_prediction":             "row",
    "record_linkage":             "row",
    # DLTE
    "dlte_retrieval":             "table",
    "dlte_alignment":             "col",
    "dlte_merge":                 "row",
}


# Paper-derived (model, task) exclusions: cells the granularity heuristic marks
# valid but the paper (.paper_reference/ct_table.csv) does NOT report -- so they
# are NOT benchmark cells. Two rationales, both -> exclude (full audit 2026-06-02):
#   * tapex -- table-only (linearizes whole tables -> only a table_embedding);
#     the per-column tasks read 'column_embeddings' and crash with KeyError.
#   * tuta  -- column-capable but its batch-1 row-by-row encoder is impractically
#     slow on the large per-column datasets (sato 157k, ...), so the paper ran it
#     on table-level tasks only (see reference_tuta_slowness).
# Both ARE reported on the 5 table-level tasks (join/union classification,
# union_regression, table_subset, table_retrieval), which stay valid. tapex/tuta
# keep 'col' granularity (needed for join/union classification) -- the per-column
# exclusion is expressed here, not by dropping a granularity.
_PER_COLUMN_TASKS_PAPER_SKIPS: tuple = (
    "column_type_prediction", "column_clustering", "column_relation_prediction",
    "join_search", "join_containment", "schema_matching", "union_search",
    "semantic_parsing",
)
_UNSUPPORTED_CELLS: frozenset = frozenset(
    (m, t) for m in ("tapex", "tuta") for t in _PER_COLUMN_TASKS_PAPER_SKIPS
)


def is_valid_cell(model: str, task: str) -> bool:
    """True iff `model` exports the granularity `task` requires AND the
    (model, task) cell is reported in the paper. Some granularity-valid cells are
    paper-excluded (``_UNSUPPORTED_CELLS``) -- e.g. tapex/tuta on per-column tasks
    -- so the valid-cell set matches ct_table.csv, not just the coarse heuristic."""
    if model not in _MODEL_GRANULARITIES or task not in _TASK_GRANULARITIES:
        return False
    if (model, task) in _UNSUPPORTED_CELLS:
        return False
    return _TASK_GRANULARITIES[task] in _MODEL_GRANULARITIES[model]


def list_cells() -> Iterable[tuple[str, str]]:
    """Yield every (model, task) cell supported by the registry."""
    for model in _MODEL_GRANULARITIES:
        for task in _TASK_GRANULARITIES:
            if is_valid_cell(model, task):
                yield (model, task)


# == Stage-3 (probe head) dispatch configuration ==============================
# Source-of-truth: the per-cell .sbatch files from the reference runs at
#   results/round{4,5}/downstream/<task>/<model>_<dataset>_<setting>_seed<S>_<probe>.sbatch
# Each .sbatch invokes ``python utils/downstream/run_task.py`` with the args
# below. Defaults match what the paper-producing runs used; YAML values are
# loaded by run_task.py and overridden by these CLI flags.

@dataclass(frozen=True)
class ProbeConfig:
    task_name_template: str      # e.g. "join_{dataset}" -> "join_spider_join"
    task_type: str               # "classification" | "regression"
    yaml: str                    # path relative to repo root (or cwd)
    combination_method: str
    hidden_dim: int
    num_labels: int
    batch_size: int
    max_epochs: int
    learning_rate: float
    dropout_prob: float
    # Output-directory layout for Stage-3 results + envelope. Five variants
    # observed in the reference:
    #   "with_setting"    -> <task>/<model>/<setting>/seed<S>/<probe>/ (default,
    #                        used by join/union_classification, table_subset, ...)
    #   "without_setting" -> <task>/<model>/seed<S>/<probe>/         (join_containment)
    #   "with_dataset"    -> <task>/<model>/<dataset>/seed<S>/<probe>/ (record_linkage)
    #   "with_dataset_label" -> <task>/<model>/<dataset>/seed<S>/<setting>/
    #                        (row_prediction, where setting carries the label column)
    #   "semparse"        -> <task>/<model>/<setting>/seed<S>/        (no probe axis;
    #                        semantic_parsing setting is the query-encoder model)
    path_layout: str = "with_setting"
    # The Python module to invoke (sys.executable -m <runner>). Default is
    # the shared pair-task runner; record_linkage and others use task-family-
    # specific runners that accept a subset of these CLI flags.
    runner: str = "trl_bench.utils.downstream.run_task"
    # Whether the runner accepts --task_type / --embedding_type. Pair tasks do;
    # record_linkage does not (it's row-level only, no aggregation choice).
    pass_task_type: bool = True
    pass_embedding_type: bool = True
    # Probe-task family. Selects the CLI builder in ``_probe_command``:
    #   "pair_task"        — default. trl_bench.utils.downstream.run_task with
    #                        --embeddings/--labels/--task_name/--task_type/...
    #                        (used by join/union/table_subset/record_linkage).
    #   "cta"              — column_type_prediction: train_ct_mode4.py with
    #                        --embeddings/--dataset/--num_epochs/--learning_rate/
    #                        --output_dir/--seed/--head_type.
    #   "cra"              — column_relation_prediction: csv_relation_pipeline.py
    #                        with --embeddings_file/--dataset_dir/--epochs/--lr/
    #                        --hidden_dim/--output_dir/--seed/--head_type.
    #   "retrieval"        — table_retrieval: two-stage (train.py then
    #                        evaluate.py) with --table_embeddings/--train_query_*
    #                        /--dev_query_*/--projection_head/...
    #   "semparse"         — semantic_parsing: two-stage (run_training.py then
    #                        run_test.py) with --column-pkl/--question-pkls/
    #                        --dataset-path/--output-dir/--config <json>/--seed.
    family: str = "pair_task"
    # Output-envelope schema variant; consumed by run.py::_wrap_envelope:
    #   "test_results_flat" — default pair-task format with flat
    #                         test_results_<metric> keys.
    #   "cta_flat"          — CTA's flat {test_micro_f1, micro_f1, MAP, ...}.
    #   "cra_flat"          — CRA's flat {best_micro_f1, subset_accuracy, ...}.
    #   "retrieval_flat"    — retrieval's flat {Recall@k, MRR, NDCG@k, ...}.
    #   "semparse_flat"     — semantic_parsing's flat {accuracy, oracle_accuracy};
    #                         raw is ``test.log`` (a JSON dump), not ``results.json``.
    envelope_kind: str = "test_results_flat"


_TASK_PROBE_CONFIG: dict[str, ProbeConfig] = {
    "join_classification": ProbeConfig(
        task_name_template="join_{dataset}",
        task_type="classification",
        yaml="configs/downstream/join_classification.yaml",
        combination_method="concat",
        hidden_dim=256, num_labels=2,
        batch_size=32, max_epochs=50,
        learning_rate=1e-3, dropout_prob=0.1,
    ),
    "join_containment": ProbeConfig(
        task_name_template="containment_{dataset}",
        task_type="regression",
        yaml="configs/downstream/join_containment.yaml",
        combination_method="concat",
        hidden_dim=256, num_labels=1,
        batch_size=2048,                               # column-pickle input -> large batch
        max_epochs=50,
        learning_rate=1e-3, dropout_prob=0.1,
        path_layout="without_setting",                 # reference: <task>/<model>/seed<S>/<probe>/
    ),
    "union_classification": ProbeConfig(
        task_name_template="union_classification_{dataset}",
        task_type="classification",
        yaml="configs/downstream/union_classification.yaml",
        combination_method="concat",
        hidden_dim=256, num_labels=2,
        batch_size=32, max_epochs=50,
        learning_rate=1e-3, dropout_prob=0.1,
    ),
    "union_regression": ProbeConfig(
        task_name_template="union_regression_{dataset}",
        task_type="regression",
        yaml="configs/downstream/union_regression.yaml",
        combination_method="concat",
        hidden_dim=256, num_labels=1,
        batch_size=32, max_epochs=50,
        learning_rate=1e-3, dropout_prob=0.1,
    ),
    # NOTE: semantic_parsing IS wired below (see entry near the end). It uses a
    # train-then-test pair invocation; the runner's ``test.log`` (a JSON dump
    # of {accuracy, oracle_accuracy}) is the raw consumed by the
    # ``semparse_flat`` envelope wrapper.
    #
    # row_prediction IS wired below: its YAML-driven per-label-column loop is
    # handled by invoking ``--label_column <setting>`` from the dispatcher.
    # The probe-task setting axis carries the label column name for row_prediction
    # (e.g. setting=class for openml_3's `class` label), so each (model, dataset,
    # label) triple maps to a distinct settings-keyed output directory matching
    # the reference layout ``<task>/<model>/<dataset>/seed<S>/<label_col>/``.
    #
    # column_type_prediction (CTA), column_relation_prediction (CRA), and
    # table_retrieval ARE wired below — each with a distinct ``family`` and
    # ``envelope_kind``. Stage-0 staging for these (sato/SOTAB/nq_tables) is
    # implemented in ``trl_bench.data.stage``.
    # The "record_linkage" entry lives below (task-specific runner +
    # path_layout). Do NOT add a generic run_task.py-style entry for it here.
    "column_type_prediction": ProbeConfig(
        # hyperparameters={num_epochs=10, batch_size=20,
        # learning_rate=0.001}. Runner: train_ct_mode4.py.
        task_name_template="cta_{dataset}",              # documentary only — runner does not consume task_name
        task_type="classification",                      # documentary only
        yaml="configs/downstream/column_type_prediction.yaml",
        combination_method="none",                       # documentary only
        hidden_dim=256, num_labels=0,                    # runner reads class vocab from labels
        batch_size=20,                                   # reference default
        max_epochs=10,                                   # reference default (--num_epochs)
        learning_rate=1e-3, dropout_prob=0.1,
        path_layout="with_dataset",                      # output: <task>/<model>/<dataset>/seed<S>/<probe>/
        runner="trl_bench.tasks.column_type_prediction.train_ct_mode4",
        pass_task_type=False, pass_embedding_type=False,
        family="cta",
        envelope_kind="cta_flat",
    ),
    "column_relation_prediction": ProbeConfig(
        # hyperparameters={epochs=20, batch_size=32,
        # learning_rate=0.001, hidden_dim=256}. Runner: csv_relation_pipeline.py.
        # CLI uses --epochs (not --num_epochs), --lr (not --learning_rate),
        # --embeddings_file (not --embeddings), --dataset_dir (not --dataset).
        task_name_template="cra_{dataset}",              # documentary only
        task_type="classification",                      # documentary only
        yaml="configs/downstream/column_relation_prediction.yaml",
        combination_method="concat",                     # documentary only
        hidden_dim=256, num_labels=0,
        batch_size=32, max_epochs=20,
        learning_rate=1e-3, dropout_prob=0.1,
        path_layout="with_dataset",
        runner="trl_bench.tasks.column_relation_prediction.csv_relation_pipeline",
        pass_task_type=False, pass_embedding_type=False,
        family="cra",
        envelope_kind="cra_flat",
    ),
    "table_retrieval": ProbeConfig(
        # hyperparameters={projection_dim=256, hidden_dim=256}. Two-stage:
        #   1) train.py  -> writes best_model.pt to <output_dir>/
        #   2) evaluate.py --projection_head <output_dir>/best_model.pt
        #      --output_path <output_dir>/results.json
        # Setting axis (cls_embedding / column_mean / token_mean) is passed to
        # train.py as --embedding_variant.
        task_name_template="table_retrieval_{dataset}",  # documentary only
        task_type="retrieval",                           # documentary only
        yaml="configs/downstream/table_retrieval.yaml",
        combination_method="none",
        hidden_dim=256, num_labels=0,
        batch_size=512, max_epochs=60,                   # reference defaults
        learning_rate=1e-3, dropout_prob=0.1,
        path_layout="with_setting",                      # output: <task>/<model>/<setting>/seed<S>/<probe>/
        runner="trl_bench.tasks.table_retrieval.train",  # train stage; eval stage hardcoded
        pass_task_type=False, pass_embedding_type=False,
        family="retrieval",
        envelope_kind="retrieval_flat",
    ),
    "table_subset": ProbeConfig(
        task_name_template="table_subset_{dataset}",
        task_type="classification",
        yaml="configs/downstream/table_subset.yaml",
        combination_method="concat",
        hidden_dim=256, num_labels=2,
        batch_size=32, max_epochs=50,
        learning_rate=1e-3, dropout_prob=0.1,
    ),
    "record_linkage": ProbeConfig(
        # Runner uses its own script (not run_task.py). CLI:
        #   python downstream_tasks/record_linkage/run_record_linkage.py \
        #     --embeddings <row_pkl> --labels <labels> --task_name record_linkage_<ds> ...
        task_name_template="record_linkage_{dataset}",
        task_type="classification",          # documentary; not passed (no --task_type)
        yaml="configs/downstream/record_linkage.yaml",
        combination_method="concat",
        hidden_dim=256, num_labels=2,
        batch_size=64, max_epochs=50,
        learning_rate=1e-3, dropout_prob=0.1,
        path_layout="with_dataset",          # <task>/<model>/<dataset>/seed<S>/<probe>/
        runner="trl_bench.tasks.record_linkage.run_record_linkage",
        pass_task_type=False, pass_embedding_type=False,
    ),
    "row_prediction": ProbeConfig(
        # Runner: train_downstream.py (Trainer pipeline). Its results.json
        # carries top-level: task, task_type, head_type, seed, model,
        # dataset, variant, label_column, test_results, training, data_stats,
        # train_results, scaled_test_results, target_zscore, target_scaler.
        #
        # The CLI is YAML-driven (configs/downstream/row_prediction.yaml drives
        # hyperparameters); the dispatcher only passes the per-cell knobs
        # (--seed/--model/--dataset/--head_type/--label_column). The setting
        # axis carries the label column name -- reference layout is
        # ``<task>/<model>/<dataset>/seed<S>/<label_col>/results.json``.
        task_name_template="row_prediction_{dataset}",   # documentary only
        task_type="regression",                          # documentary only; task
                                                         # type is auto-detected from data
        yaml="configs/downstream/row_prediction.yaml",
        combination_method="none",                       # documentary only
        hidden_dim=256, num_labels=0,
        batch_size=32, max_epochs=100,                   # YAML-driven; documentary
        learning_rate=1e-3, dropout_prob=0.1,
        path_layout="with_dataset_label",                # <task>/<model>/<dataset>/seed<S>/<label_col>/
        runner="trl_bench.tasks.row_prediction.train_downstream",
        pass_task_type=False, pass_embedding_type=False,
        family="row_prediction",
        envelope_kind="row_prediction_flat",
    ),
    # == DLTE (Data-Lake Table Enrichment) — 3-stage pipeline =================
    # The DLTE benchmark consists of 3 stages, each dispatched as its own task:
    #
    #   dlte_retrieval  -> step8_faiss_retrieval.py  (Stage 1)
    #                        Input:  table-embedding pickles for {queries,
    #                                targets, ckan_subset} sharing a model.
    #                        Output: <results_dir>/dlte/stage1/<model>/
    #                                  topk_<K>.jsonl + metrics_<split>_topk_<K>.json
    #   dlte_alignment  -> step9_column_alignment.py  (Stage 2)
    #                        Input:  Stage-1 topk + column-embedding pickle for
    #                                col_model. Uses --table_model to point at
    #                                the upstream Stage-1 retrieval.
    #                        Output: <results_dir>/dlte/stage2/<col_model>/
    #                                  or <results_dir>/dlte/stage2/<table>__<col>/
    #   dlte_merge      -> step10_row_matching.py + step11_evaluation.py (Stage 3)
    #                        Input:  Stage-2 alignment + row-embedding pickle for
    #                                row_model. Uses --col_models / --row_models /
    #                                --table_model.
    #                        Output: <results_dir>/dlte/{stage3,metrics}/
    #                                  <col>__<row>__<table>/{end_to_end.json,
    #                                  stage1.json, stage2.json, summary.csv}
    #
    # Dispatch convention (per-cell axis-mapping):
    #
    #   For each task the caller supplies a single ``--model`` plus a ``--setting``
    #   that may carry up to two auxiliary model names, packed as
    #   ``"<aux1>__<aux2>"`` (e.g. for ``dlte_merge`` ``setting="bert__bert"`` means
    #   ``col_model=bert`` and ``table_model=bert``; ``model`` is the row_model).
    #   When ``setting`` is omitted or is a single token it defaults to the diagonal
    #   case (model fills every axis the task needs). The diagonal default mirrors
    #   the most-frequently-observed reference cell (e.g. ``bert__bert__bert``).
    #
    #   * ``dlte_retrieval`` -> ``model`` IS the table_model. Setting axis is
    #     unused by the runner; we accept ``setting`` as documentation only.
    #   * ``dlte_alignment`` -> ``model`` IS the col_model. ``setting`` may carry
    #     the table_model; defaults to the diagonal (table_model = col_model).
    #   * ``dlte_merge``     -> ``model`` IS the row_model. ``setting`` may carry
    #     ``"<col_model>__<table_model>"``; defaults to the full diagonal.
    #
    # Probe-axis: DLTE has no probe-head axis (the runners do not consume
    # ``--head_type``). ``--probe`` is OPTIONAL for these tasks — mirroring the
    # semantic_parsing exemption. The path_layout ``dlte`` writes directly to
    # ``<results>/dlte/<stage>/<model_key>/`` with no probe sub-directory.
    "dlte_retrieval": ProbeConfig(
        task_name_template="dlte_retrieval",          # documentary only
        task_type="retrieval",                        # documentary only
        yaml="configs/downstream/dlte.yaml",          # documentary; the runner
                                                      # is path/CLI driven (no YAML)
        combination_method="none",
        hidden_dim=0, num_labels=0,
        batch_size=0, max_epochs=0,
        learning_rate=0.0, dropout_prob=0.0,
        path_layout="dlte",
        runner="trl_bench.tasks.dlte.scripts.step8_faiss_retrieval",
        pass_task_type=False, pass_embedding_type=False,
        family="dlte_retrieval",
        envelope_kind="dlte_retrieval_flat",
    ),
    "dlte_alignment": ProbeConfig(
        task_name_template="dlte_alignment",
        task_type="classification",
        yaml="configs/downstream/dlte.yaml",
        combination_method="none",
        hidden_dim=0, num_labels=0,
        batch_size=0, max_epochs=0,
        learning_rate=0.0, dropout_prob=0.0,
        path_layout="dlte",
        runner="trl_bench.tasks.dlte.scripts.step9_column_alignment",
        pass_task_type=False, pass_embedding_type=False,
        family="dlte_alignment",
        envelope_kind="dlte_alignment_flat",
    ),
    "dlte_merge": ProbeConfig(
        task_name_template="dlte_merge",
        task_type="merge",
        yaml="configs/downstream/dlte.yaml",
        combination_method="none",
        hidden_dim=0, num_labels=0,
        batch_size=0, max_epochs=0,
        learning_rate=0.0, dropout_prob=0.0,
        path_layout="dlte",
        runner="trl_bench.tasks.dlte.scripts.step10_row_matching",
        pass_task_type=False, pass_embedding_type=False,
        family="dlte_merge",
        envelope_kind="dlte_merge_flat",
    ),
    # == Deterministic (training-free) tasks ===================================
    # These tasks have no probe head and no seed dimension: their runners are
    # deterministic per (model, dataset). The reference output layout is
    # FLAT — ``<task>/<model>/<model>_<dataset>.json`` with no
    # ``seed<S>`` / ``<probe>`` subdirectories. ``path_layout="deterministic"``
    # encodes that flat layout.
    #
    # ``--probe`` is ignored for these tasks (mirroring the semantic_parsing /
    # DLTE exemption); the dispatcher accepts ``probe=None`` and the family
    # builders do not emit any ``--head_type`` CLI flag.
    "column_clustering": ProbeConfig(
        # hyperparameters = {k=20, target_avg_size=50}.
        # Runner: evaluate_clustering.py. CLI:
        #   --embeddings <col_pkl> --dataset <ds_dir>
        #   --k 20 --target_avg_size 50 --batch_size 4096
        # Output: prints metrics to stdout (Number of clusters, Avg cluster
        # size, Purity, NMI, ARI, Total columns, Coverage, ...). The dispatcher
        # captures stdout to ``stage_run.log`` in the envelope directory.
        task_name_template="column_clustering_{dataset}",   # documentary only
        task_type="clustering",                             # documentary only
        yaml="configs/downstream/column_clustering.yaml",   # documentary
        combination_method="none",                          # documentary only
        # We re-purpose three ProbeConfig fields to carry the reference's
        # ``hyperparameters`` block: max_epochs=k, hidden_dim=target_avg_size,
        # batch_size=runner's --batch_size argument. The hp-construction
        # branch in run.py reads them back as {"k": cfg.max_epochs,
        # "target_avg_size": cfg.hidden_dim}.
        hidden_dim=50,                                       # = target_avg_size
        num_labels=0,
        batch_size=4096,                                     # runner --batch_size
        max_epochs=20,                                       # = k (nearest neighbors)
        learning_rate=0.0, dropout_prob=0.0,
        path_layout="deterministic",
        runner="trl_bench.tasks.column_clustering.evaluate_clustering",
        pass_task_type=False, pass_embedding_type=False,
        family="column_clustering",
        envelope_kind="column_clustering_flat",
    ),
    "schema_matching": ProbeConfig(
        # hyperparameters = {matching_strategy="hungarian", threshold=0.0}.
        # Runner: run_schema_matching.py. CLI:
        #   --embeddings <col_pkl> --pairs <pairs.json>
        #   --ground_truth <gt.csv> --tables_dir <tables/>
        #   --output_dir <out> --matching_strategy hungarian --threshold 0.0
        # Output: writes ``<output_dir>/results.json`` natively + per_pair.csv.
        task_name_template="schema_matching_{dataset}",
        task_type="matching",                                # documentary only
        yaml="configs/downstream/schema_matching.yaml",      # documentary
        # Mirror the reference hyperparameters:
        #   combination_method=matching_strategy,
        #   learning_rate=threshold (re-purposed for the float threshold).
        combination_method="hungarian",
        hidden_dim=0, num_labels=0, batch_size=0, max_epochs=0,
        learning_rate=0.0,
        dropout_prob=0.0,
        path_layout="deterministic",
        runner="trl_bench.tasks.schema_matching.run_schema_matching",
        pass_task_type=False, pass_embedding_type=False,
        family="schema_matching",
        envelope_kind="schema_matching_flat",
    ),
    "union_search": ProbeConfig(
        # hyperparameters = {method="linear", k=10, threshold=0.7, ef=100, N=100}.
        # Runner: run_search.py. CLI:
        #   --query_embeddings <pkl> --datalake_embeddings <pkl>
        #   --groundtruth <pickle>
        #   --method linear --K 10 --threshold 0.7 --ef 100 --N 100
        # Output: prints MAP@K, P@K, R@K to stdout (no JSON written by the
        # runner). The dispatcher captures stdout to ``stage_run.log``.
        task_name_template="union_search_{dataset}",
        task_type="retrieval",
        yaml="configs/downstream/union_search.yaml",
        # Re-purposed fields to carry the reference hp keys:
        #   combination_method=method, num_labels=k, learning_rate=threshold,
        #   batch_size=ef, max_epochs=N.
        combination_method="linear",
        hidden_dim=0,
        num_labels=10,                                       # = K
        batch_size=100,                                      # = ef
        max_epochs=100,                                      # = N
        learning_rate=0.7,                                   # = threshold
        dropout_prob=0.0,
        path_layout="deterministic",
        runner="trl_bench.tasks.union_search.run_search",
        pass_task_type=False, pass_embedding_type=False,
        family="union_search",
        envelope_kind="union_search_flat",
    ),
    "join_search": ProbeConfig(
        # hyperparameters = {k=50}.
        # Runner: run_search_and_evaluate.py (combined search+eval). CLI:
        #   --query_emb <pkl> --datalake_emb <pkl>
        #   --query_list <query.csv> --ground_truth <gt.csv>
        #   --k 50 --output <results.csv>
        # Output: writes ``<output>.csv`` (per-pair retrieval) and prints
        # column-level aggregate metrics (COL Precision@10/20/50,
        # COL Recall@10/20/50, COL F1@10/20/50, COL MAP) to stdout. The
        # dispatcher captures stdout to ``stage_run.log``.
        task_name_template="join_search_{dataset}",
        task_type="retrieval",
        yaml="configs/downstream/join_search.yaml",
        combination_method="none",
        hidden_dim=0, num_labels=0, batch_size=0,
        max_epochs=50,                                       # = k
        learning_rate=0.0, dropout_prob=0.0,
        path_layout="deterministic",
        runner="trl_bench.tasks.join_search.run_search_and_evaluate",
        pass_task_type=False, pass_embedding_type=False,
        family="join_search",
        envelope_kind="join_search_flat",
    ),
    "join_search_learned": ProbeConfig(
        # Learned-projection (InfoNCE) variant of join_search. Runner:
        # run_learned_search.py (train projection head -> search -> eval).
        # Canonical CLI (slurm/config tasks.yaml defaults): --num_layers 1
        # --batch_size 512 --max_epochs 10 --learning_rate 1e-3 --k 50
        # (--temperature 0.07, --dropout 0.1 are runner defaults). SEEDED
        # training task -> NOT deterministic, envelope keeps the seed suffix.
        # Re-purposed fields: num_labels=num_layers, max_epochs=epochs,
        # hidden_dim=projection hidden_dim. Reuses join_search's COL-metric
        # stdout parsing (join_search_flat) since the runner prints the same
        # ``COL MAP``/``COL Recall@k`` lines.
        task_name_template="join_search_learned_{dataset}",
        task_type="retrieval",
        yaml="configs/downstream/join_search.yaml",
        combination_method="none",
        hidden_dim=256,                                  # projection hidden_dim
        num_labels=1,                                    # = num_layers
        batch_size=512,
        max_epochs=10,                                   # = epochs
        learning_rate=1e-3, dropout_prob=0.1,
        path_layout="deterministic",
        runner="trl_bench.tasks.join_search.run_learned_search",
        pass_task_type=False, pass_embedding_type=False,
        family="join_search_learned",
        envelope_kind="join_search_flat",
    ),
    "semantic_parsing": ProbeConfig(
        # Two-stage runner: ``run_training.py`` trains a MAPO decoder; then
        # ``run_test.py`` loads ``model.best.bin`` and writes ``test.log``
        # (a JSON dump of {accuracy, oracle_accuracy}) next to it.
        # The envelope carries: model, dataset, task, seed, slurm_job_id,
        # hyperparameters={seed, beam_size=5}, accuracy, oracle_accuracy, status.
        # The setting axis carries the QUERY ENCODER model (mpnet/sentence_t5)
        # used to embed natural-language questions; the layout matches
        # ``with_setting`` but WITHOUT a probe sub-directory.
        task_name_template="semantic_parsing_{dataset}",     # documentary only
        task_type="classification",                          # documentary only
        yaml="configs/downstream/semantic_parsing.yaml",     # documentary; runner
                                                             # consumes a JSON config
        combination_method="none",                           # documentary only
        hidden_dim=256, num_labels=0,
        batch_size=32, max_epochs=50,                        # documentary only
        learning_rate=1e-3, dropout_prob=0.1,
        path_layout="semparse",                              # <task>/<model>/<setting>/seed<S>/
        runner="trl_bench.tasks.semantic_parsing.run_training",
        pass_task_type=False, pass_embedding_type=False,
        family="semparse",
        envelope_kind="semparse_flat",
    ),
}


_SETTING_TO_EMBEDDING_TYPE: dict[str, str] = {
    "cls_embedding":   "cls",
    # The pair-task runner's argparse has choices={cls,table,column_mean,
    # token_mean,column}; ``table_embedding`` setting -> ``table`` flag value.
    "table_embedding": "table",
    "column_mean":     "column_mean",
    "token_mean":      "token_mean",
    "row_embedding":   "row",
    # join_containment uses the column pickle directly with --embedding_type=column.
    "column":          "column",
}


# == Stage-1 (column / row extraction) dispatch configuration =================
# Source-of-truth: each wrapper's CLI surface, documented in
#   src/trl_bench/models/<m>/USAGE.md (or its argparse if no USAGE.md).
# Wrappers are preserved as-is, so the registry only encodes *which* runner to
# invoke and what model-specific extras to pass. Three CLI shapes cover almost
# every wrapper:
#
#   A) BERT-shape  -> ``--input <csv-dir> --output <pkl> [--model <hf-id>]
#                       --device <cuda|cpu>``        (bert, gte, tapas, openai*)
#   B) "_dir"-shape -> ``--input_dir <csv-dir> --output_path <pkl>
#                       [--device <cuda|cpu>]``       (tabicl, tabpfn, ...)
#   C) "_dir+ckpt-dir"-shape -> shape B plus a REQUIRED ``--checkpoint_base_dir``
#                       output directory for per-table trained checkpoints
#                       (transtab, dae, scarf, vime, subtab, saint,
#                       tabular_binning, tabtransformer).
#
# Distinct flag names are encoded via ``ExtractorConfig.input_flag`` /
# ``output_flag``. Wrappers without a ``--device`` flag (e.g. openai which
# routes through the OpenAI HTTP API) set ``device_flag=None`` so the device
# pair is omitted entirely. Templated values like the per-cell checkpoint
# directory (used as a write-only training artefacts dir, cleaned up by the
# wrapper unless ``--keep_checkpoints``) are emitted via ``derived_args``:
# the template ``{output_parent}`` resolves to ``Path(output_path).parent``.
#
# Wrappers requiring a host-specific licensed checkpoint file (TaBERT,
# TabSketchFM, TURL, TUTA, TABBIE, Starmie) are wired through the
# ``checkpoint_template`` / ``checkpoint_arg`` fields below: the dispatcher
# resolves ``<checkpoint_root>/<template>`` at command-build time and appends
# ``<cfg.checkpoint_arg> <resolved>``. Users obtain the checkpoint files via
# ``scripts/download_checkpoints.sh`` (see ``docs/CHECKPOINT_LICENSES.md``)
# and point ``--checkpoint-root`` at the on-disk directory.

@dataclass(frozen=True)
class ExtractorConfig:
    """Stage-1 column-/row-extraction subprocess dispatch for one model.

    The runner is invoked as::

        python -m <runner> <input_flag> <csv-dir> <output_flag> <pkl> \
            [extra-args...] [derived-args...] \
            [<checkpoint_arg> <checkpoint_root>/<checkpoint_template>] \
            [<device_flag> <device_value>]

    where ``input_flag`` / ``output_flag`` default to the BERT-shape
    ``--input`` / ``--output`` but may be overridden for wrappers using
    ``--input_dir`` / ``--output_path``. ``extra_args`` lets each model pin
    a static flag (e.g. ``--model thenlper/gte-base`` for GTE).
    ``derived_args`` carries values templated against the per-call arguments
    — supported template tokens:

      ``{output_parent}``  -> ``Path(output_path).parent``
      ``{output_stem}``    -> ``Path(output_path).stem`` (e.g.
                              ``spider_join`` for ``spider_join.pkl``)
      ``{dataset}``        -> the ``dataset`` argument (e.g. ``spider_join``);
                              used by wrappers whose checkpoint path is
                              per-dataset (Starmie).

    The combination ``{output_parent}/_ckpts_{output_stem}`` is the canonical
    per-cell scratch dir for trained-per-table wrappers' checkpoints —
    isolated from the output pickle AND from sibling datasets' scratch dirs
    in the same model's column-pickle directory. The wrappers' shared
    ``cleanup_checkpoints`` helper does ``shutil.rmtree(<base>/<table_id>)``
    per processed CSV; scoping the base to a dedicated subdir prevents
    accidental rmtree of any file or directory that happens to share a stem
    with a CSV in the input dir.

    ``checkpoint_template`` (if set) is a *relative* path resolved against
    ``checkpoint_root`` (``./checkpoints`` by default, overrideable via
    ``--checkpoint-root`` / ``$TRL_BENCH_CKPT_ROOT``). It supports the same
    template tokens as ``derived_args``. When ``checkpoint_required=True``
    the dispatcher raises ``SettingError`` if the resolved file/directory
    does not exist on disk; when ``False`` the path is passed through
    verbatim and missing-file errors are deferred to the runner.

    Set ``device_flag=None`` for wrappers that have no ``--device`` flag
    (e.g. OpenAI, which routes API calls through the OpenAI HTTP endpoint).
    ``device_value_map`` (optional) translates the canonical ``cuda``/``cpu``
    overrides into wrapper-specific values for ``--device_id <int>``-style
    flags (TUTA, TABBIE), e.g. ``{"cuda": "0", "cpu": "-1"}``.
    """
    runner: str
    # Model-specific flags appended after the input/output pair and before
    # the device pair. Use a tuple-of-tuples so the dataclass stays frozen-
    # clean (immutable defaults).
    extra_args: tuple[tuple[str, str], ...] = ()
    # Derived-value flags whose VALUES are resolved per-call against a small
    # template set (see docstring). The FLAG names are still static. We keep
    # this distinct from extra_args so the resolver does not silently mangle
    # static values that happen to contain ``{`` characters.
    derived_args: tuple[tuple[str, str], ...] = ()
    # The CLI flag names for the input directory and output pickle paths.
    # BERT-shape wrappers use ``--input`` / ``--output``; row-level wrappers
    # use ``--input_dir`` / ``--output_path``.
    input_flag: str = "--input"
    output_flag: str = "--output"
    # ``None`` => the wrapper has no ``--device`` flag; do not emit one.
    device_flag: str | None = "--device"
    device_value: str = "cuda"
    # Optional cuda/cpu -> wrapper-value translation (e.g. TUTA / TABBIE's
    # ``--device_id <int>``). Empty default => identity.
    device_value_map: tuple[tuple[str, str], ...] = ()
    # Path to a host-specific licensed checkpoint, relative to
    # ``checkpoint_root``. Supports {dataset}, {output_parent}, {output_stem}.
    checkpoint_template: str | None = None
    # The CLI flag name for the resolved checkpoint path. Varies across
    # wrappers: --checkpoint (tabert/tabsketchfm/turl), --model_path
    # (tuta/tabbie/starmie).
    checkpoint_arg: str = "--checkpoint"
    # If True, ``build_extractor_command`` raises ``SettingError`` when the
    # resolved checkpoint file/directory does not exist on disk.
    checkpoint_required: bool = False


_MODEL_EXTRACTORS: dict[str, ExtractorConfig] = {
    # == Shape A: BERT-shape --input / --output / --device ====================
    "bert": ExtractorConfig(
        runner="trl_bench.models.bert.generate_column_embeddings",
    ),
    "gte": ExtractorConfig(
        runner="trl_bench.models.gte.generate_column_embeddings",
        extra_args=(("--model", "thenlper/gte-base"),),
    ),
    # TAPAS shares the BERT CLI shape (--input <csv-dir>, --output <pkl>,
    # --model <hf-id>, --device <cuda|cpu>). Its argparse default for --model
    # is already ``google/tapas-base``, the paper checkpoint, so no extras are
    # needed. See ``models/tapas/generate_column_embeddings.py`` and
    # ``models/tapas/USAGE.md``.
    "tapas": ExtractorConfig(
        runner="trl_bench.models.tapas.generate_column_embeddings",
    ),
    # OpenAI: BERT-shape (--input / --output / --model <openai-model-id>)
    # but NO --device flag — the wrapper routes through the OpenAI HTTP API
    # (model client lives in ``models/openai/client.py``, picks up
    # ``OPENAI_API_KEY`` from env). The argparse default for --model is
    # ``text-embedding-3-small``, the paper checkpoint, so no extras needed
    # (mirrors the BERT/TAPAS decision to leave argparse defaults in place).
    "openai": ExtractorConfig(
        runner="trl_bench.models.openai.generate_column_embeddings",
        device_flag=None,
    ),
    # == Shape B: --input_dir / --output_path / --device ======================
    # Pretrained row-level models whose checkpoints are auto-fetched at first
    # use (TabICL / TabPFN both ship their .ckpt via PyPI; the constructor
    # triggers the download — see ``scripts/download_checkpoints.sh``). We
    # pin --device cuda explicitly (instead of the wrapper's "auto" default)
    # to match the BERT/GTE/TAPAS convention; pass device="cpu" to override.
    "tabicl": ExtractorConfig(
        runner="trl_bench.models.tabicl.generate_embeddings_directory",
        input_flag="--input_dir", output_flag="--output_path",
    ),
    "tabpfn": ExtractorConfig(
        runner="trl_bench.models.tabpfn.generate_embeddings_directory",
        input_flag="--input_dir", output_flag="--output_path",
    ),
    # == Shape C: --input_dir / --output_path + per-cell --checkpoint_base_dir
    # Trained-per-table self-supervised row models. The wrapper TRAINS a fresh
    # model per CSV and writes per-table checkpoints under
    # ``--checkpoint_base_dir/<table_id>/`` (cleaned up via
    # ``shutil.rmtree(<base>/<table_id>)`` after each table is embedded
    # successfully, unless ``--keep_checkpoints``). We derive the base dir
    # as ``{output_parent}/_ckpts_{output_stem}`` -- a dedicated subdir
    # scoped to this (model, dataset) pair so the rmtree-per-table cannot
    # touch the output pickle or sibling datasets' artefacts.
    "transtab": ExtractorConfig(
        runner="trl_bench.models.transtab.generate_embeddings_directory",
        input_flag="--input_dir", output_flag="--output_path",
        derived_args=(("--checkpoint_base_dir", "{output_parent}/_ckpts_{output_stem}"),),
        device_flag=None,
    ),
    "dae": ExtractorConfig(
        runner="trl_bench.models.dae.generate_embeddings_directory",
        input_flag="--input_dir", output_flag="--output_path",
        derived_args=(("--checkpoint_base_dir", "{output_parent}/_ckpts_{output_stem}"),),
        device_flag=None,
    ),
    "scarf": ExtractorConfig(
        runner="trl_bench.models.scarf.generate_embeddings_directory",
        input_flag="--input_dir", output_flag="--output_path",
        derived_args=(("--checkpoint_base_dir", "{output_parent}/_ckpts_{output_stem}"),),
        device_flag=None,
    ),
    "switchtab": ExtractorConfig(
        runner="trl_bench.models.switchtab.generate_embeddings_directory",
        input_flag="--input_dir", output_flag="--output_path",
        derived_args=(("--checkpoint_base_dir", "{output_parent}/_ckpts_{output_stem}"),),
        device_flag=None,
    ),
    "vime": ExtractorConfig(
        runner="trl_bench.models.vime.generate_embeddings_directory",
        input_flag="--input_dir", output_flag="--output_path",
        derived_args=(("--checkpoint_base_dir", "{output_parent}/_ckpts_{output_stem}"),),
        device_flag=None,
    ),
    "subtab": ExtractorConfig(
        runner="trl_bench.models.subtab.generate_embeddings_directory",
        input_flag="--input_dir", output_flag="--output_path",
        derived_args=(("--checkpoint_base_dir", "{output_parent}/_ckpts_{output_stem}"),),
        device_flag=None,
    ),
    "saint": ExtractorConfig(
        runner="trl_bench.models.saint.generate_embeddings_directory",
        input_flag="--input_dir", output_flag="--output_path",
        derived_args=(("--checkpoint_base_dir", "{output_parent}/_ckpts_{output_stem}"),),
        device_flag=None,
    ),
    "tabular_binning": ExtractorConfig(
        runner="trl_bench.models.tabular_binning.generate_embeddings_directory",
        input_flag="--input_dir", output_flag="--output_path",
        derived_args=(("--checkpoint_base_dir", "{output_parent}/_ckpts_{output_stem}"),),
        device_flag=None,
    ),
    "tabtransformer": ExtractorConfig(
        runner="trl_bench.models.tabtransformer.generate_embeddings_directory",
        input_flag="--input_dir", output_flag="--output_path",
        derived_args=(("--checkpoint_base_dir", "{output_parent}/_ckpts_{output_stem}"),),
        device_flag=None,
    ),
    # == Shape D: licensed local checkpoint via ``checkpoint_template`` =======
    # These wrappers consume a checkpoint file/dir the user obtained via
    # ``scripts/download_checkpoints.sh`` (or trained locally, for Starmie).
    # The dispatcher resolves ``<checkpoint_root>/<template>`` and appends
    # ``<checkpoint_arg> <resolved>``. ``checkpoint_root`` is overrideable
    # via the ``--checkpoint-root`` CLI flag, the ``$TRL_BENCH_CKPT_ROOT``
    # env var, or the function kwarg; default is ``./checkpoints``.
    # ``checkpoint_required=True`` makes the dispatcher fail fast (clear
    # error pointing at ``scripts/download_checkpoints.sh``) instead of
    # silently emitting a path the runner will then fail on.
    #
    # tabert (CC BY-NC 4.0): BERT-shape CLI (--input / --output / --device)
    #   + --checkpoint <model.bin>. The release mirrors tabert_base_k3 on
    #   logo-lab/trl-arena-ckpts under the upstream NC-only terms; users on
    #   commercial deployments must fetch from the upstream Google Drive
    #   instead (see docs/CHECKPOINT_LICENSES.md). The wrapper additionally
    #   needs its own venv ('source models/tabert/load_env' in the working
    #   repo); the dispatcher only emits the python command — the venv
    #   activation is the caller's responsibility for slurm submission, but
    #   the in-repo wrapper imports work in the local venv we test in.
    "tabert": ExtractorConfig(
        runner="trl_bench.models.tabert.generate_column_embeddings",
        checkpoint_template="tabert/tabert_base_k3/model.bin",
        checkpoint_arg="--checkpoint",
        checkpoint_required=True,
    ),
    # tabsketchfm (CC BY-NC-ND 4.0): BERT-shape CLI + --checkpoint <.ckpt>.
    #   No --max_rows / --batch_size on this wrapper. Cannot be mirrored
    #   (ND clause); users fetch from the IBM/LakeBench Zenodo record
    #   (https://doi.org/10.5281/zenodo.8014642). Default path mirrors the
    #   working repo's ``checkpoints/tabsketchfm/epoch=10-step=27786.ckpt``.
    "tabsketchfm": ExtractorConfig(
        runner="trl_bench.models.tabsketchfm.generate_column_embeddings",
        checkpoint_template="tabsketchfm/epoch=10-step=27786.ckpt",
        checkpoint_arg="--checkpoint",
        checkpoint_required=True,
    ),
    # turl (Apache-2.0): --input_dir / --output_file (NOTE: --output_file,
    #   not --output_path) + --checkpoint <dir> (a directory containing
    #   pytorch_model.bin + config.json). The wrapper requires
    #   ``--mode table_directory`` as an extra arg. Mirrored on
    #   logo-lab/trl-arena-ckpts (Apache-2.0) under
    #   ``turl/pretrained/{pytorch_model.bin,config.json}``.
    "turl": ExtractorConfig(
        runner="trl_bench.models.turl.generate_column_embeddings_dataset",
        # Memory bounds for the single-cell quickstart path. Two distinct OOMs
        # surfaced without these:
        #   1. host-RAM SIGKILL (rc=-9): the wrapper default of max_entities=None
        #      lets collate_fn build a ``(batch, N, N)`` entity mask sized to the
        #      largest table -- unbounded. ``max_entities=12000`` (paper cap,
        #      slurm/config/models.yaml:158) bounds it.
        #   2. GPU OOM: self-attention is ``(batch, heads, E, E)``. At batch=16
        #      a 12000-entity table needs 16*12*12000^2*4B ~= 110 GB -- larger
        #      than any single GPU. ``batch_size=1`` drops that to ~6.9 GB,
        #      which fits a 24 GB card. Per-table column embeddings are
        #      batch-invariant (padding is masked), so batch_size only trades
        #      throughput, not results. The slurm paper grid keeps batch_size=16
        #      for speed on 80 GB GPUs via its own config; this default favors
        #      memory-safety for the documented one-experiment quickstart.
        extra_args=(
            ("--mode", "table_directory"),
            ("--max_rows", "100"),
            ("--max_entities", "12000"),
            ("--batch_size", "1"),
            ("--num_workers", "4"),
            ("--max_cell_chars", "512"),
            ("--checkpoint_interval", "100"),
        ),
        input_flag="--input_dir", output_flag="--output_file",
        checkpoint_template="turl/pretrained",
        checkpoint_arg="--checkpoint",
        checkpoint_required=True,
    ),
    # tuta (MIT): --input_dir / --output_path + --model_path <.bin> and
    #   --device_id <int> (NOT --device cuda|cpu). The wrapper script lives
    #   at models/tuta/generate_embeddings_directory.py (note: directory.py,
    #   not column_embeddings.py — TUTA emits per-column embeddings via a
    #   directory-scanning runner that does NOT match the BERT --input shape).
    #   ``device_value_map`` translates the canonical cuda/cpu override into
    #   the wrapper's int device id; default cuda -> "0", explicit cpu -> "-1".
    "tuta": ExtractorConfig(
        runner="trl_bench.models.tuta.generate_embeddings_directory",
        input_flag="--input_dir", output_flag="--output_path",
        checkpoint_template="tuta/tuta.bin",
        checkpoint_arg="--model_path",
        checkpoint_required=True,
        device_flag="--device_id",
        device_value="0",
        device_value_map=(("cuda", "0"), ("cpu", "-1")),
    ),
    # tabbie (MIT — see docs/CHECKPOINT_LICENSES.md):
    #   BERT-shape --input / --output but --model_path <weights.pt> and
    #   --device_id <int>. Same device-id translation as TUTA. Not mirrored
    #   on logo-lab/trl-arena-ckpts; users
    #   fetch from the SFIG611 Google Drive folder (see
    #   docs/CHECKPOINT_LICENSES.md).
    "tabbie": ExtractorConfig(
        runner="trl_bench.models.tabbie.generate_column_embeddings",
        checkpoint_template="tabbie/weights.pt",
        checkpoint_arg="--model_path",
        checkpoint_required=True,
        device_flag="--device_id",
        device_value="0",
        device_value_map=(("cuda", "0"), ("cpu", "-1")),
    ),
    # starmie: NO upstream checkpoint distributed — users must retrain via
    #   ``python -m trl_bench.models.starmie.run_pretrain --data_path
    #   <dataset> --checkpoint_dir <ckpt_root>/starmie/<dataset>``. The
    #   pretrain script writes outputs under ``<checkpoint_dir>/datalake/``
    #   (the ``datalake`` segment is the data variant name) — final
    #   per-dataset .pt sits at
    #   ``starmie/<dataset>/datalake/model_drop_col,sample_row_head_column_0.pt``.
    #   The dispatcher's template uses {dataset} so each cell auto-resolves
    #   to its own retrained binary. Stage-1 CLI: --input_dir /
    #   --output_path + --model_path; no --device flag (the wrapper picks
    #   device internally via cuda.is_available()).
    "starmie": ExtractorConfig(
        runner="trl_bench.models.starmie.generate_column_embeddings",
        input_flag="--input_dir", output_flag="--output_path",
        checkpoint_template=(
            "starmie/{dataset}/datalake/model_drop_col,sample_row_head_column_0.pt"
        ),
        checkpoint_arg="--model_path",
        checkpoint_required=True,
        device_flag=None,
    ),
    # mpnet, sentence_t5, tapex: routed via ``_TABLE_ENCODERS`` /
    #   ``build_table_encoder_command`` below — they take the *table-direct*
    #   Stage-1 path (one pass produces table-level embeddings including
    #   ``column_mean``; Stage-2 is skipped). mpnet + sentence_t5 also have
    #   query-side wiring in ``_QUERY_ENCODER_EXTRACTORS`` for table_retrieval
    #   / semantic_parsing.
}


# == Row-extraction runner overrides =========================================
# Models that expose BOTH column/table AND row embeddings use a separate
# script for row extraction; the script's CLI mirrors the col/table runner
# (same --input_dir / --output_path flags), only the runner module changes.
# Pure-row models (tabicl, tabpfn, transtab, dae, scarf, vime, subtab, saint,
# tabular_binning, tabtransformer) are NOT in this dict -- their
# `_MODEL_EXTRACTORS` runner IS their row extractor.
_ROW_RUNNERS: dict[str, str] = {
    "bert":   "trl_bench.models.bert.generate_row_embeddings",
    "gte":    "trl_bench.models.gte.generate_row_embeddings",
    "openai": "trl_bench.models.openai.generate_row_embeddings",
    "tabbie": "trl_bench.models.tabbie.generate_row_embeddings",
    # NB: tuta's generate_row_embeddings.py is split-aware row_PREDICTION tooling
    # (--dataset_dir with train/test -> unified_row_embedding directory), NOT the
    # --input_dir/--output_path directory->row-pickle contract record_linkage needs.
    # tuta's generate_embeddings_directory already extracts per-row [CLS]
    # (aggregate="row") into the build_table_result list that run_record_linkage
    # consumes -- byte-compatible with bert's row runner (same shared util).
    "tuta":   "trl_bench.models.tuta.generate_embeddings_directory",
}

# == TABLE-native runner overrides ===========================================
# Models whose TABLE-level embedding is produced by a single NATIVE forward
# pass (emitting a populated ``table_embedding.cls_embedding``), NOT by the
# generic column-extraction + Stage-2-aggregator path.
#
# tuta is the case: its column/row runner (``generate_embeddings_directory``)
# emits per-ROW [CLS] embeddings only (key ``row_embeddings``). The Stage-2
# aggregator (``scripts.generate_table_embeddings``) derives the table pickle's
# CLS/column_mean from the column pickle's ``column_embeddings``/``cls_embedding``
# -- tuta's column pickle has neither, so every field came out ``None`` and all
# table-level cls cells (join/union classification, union_regression,
# table_subset, table_retrieval) crashed with "embedding_type='cls' ...
# 'cls_embedding' is None". TUTA's native table representation is the [CLS]
# token over the multi-sequence-aggregated whole table -- exactly what the
# paper used (canonical_ref round5 shows ``embedding_type: "cls"`` for all 5
# tuta cells). ``generate_table_embeddings_native`` runs that pass and writes
# the {table_id, table_embedding: {cls_embedding, ...}, ...} schema directly.
#
# The override is consumed via ``build_extractor_command(..., runner_override=
# ...)`` -- the same mechanism record_linkage uses to swap the table extractor
# for a row runner. It reuses tuta's ExtractorConfig (so ``--model_path`` +
# ``--device_id`` are appended) and forces the ``--input_dir``/``--output_path``
# convention the native runner declares.
_TABLE_NATIVE_RUNNERS: dict[str, str] = {
    "tuta": "trl_bench.models.tuta.generate_table_embeddings_native",
}

# == row_prediction Stage-1 pipeline =========================================
# row_prediction has its OWN Stage-1 extractor family (separate from the DLTE
# row pipeline above). Each runner produces a `unified_row_embedding` DIRECTORY
# (metadata.json + per-split .npy + per-label .npy) which the probe trainer
# (trl_bench.tasks.row_prediction.train_downstream) consumes via
# `get_available_labels(embedding_dir)`.
#
# This dict is the INVENTORY of "row-native pretrained models with their own
# generate-only row_prediction runner" -- it is NOT the dispatch table. Dispatch
# for EVERY row-data model (these + the trained SSL family: transtab, dae, scarf,
# vime, subtab, saint, tabular_binning, tabtransformer) flows through
# `build_row_data_commands` (driven by slurm/config/row_data_models.yaml), so the
# per-model CLI/checkpoint quirks live in one place. tuta is listed here because
# it IS a row-native pretrained model (its `generate_row_embeddings.py` emits the
# unified_row_embedding directory natively); it uses native arg names
# (--dataset_dir/--output_dir) and REQUIRES --model_path, both expressed in the
# YAML. The membership/granularity invariant (test_row_runner_models_are_declared
# _row_capable) consumes this dict, so it must stay a complete inventory.
_ROW_DATA_RUNNERS: dict[str, str] = {
    "bert":   "trl_bench.models.bert.generate_embeddings_train_test",
    "gte":    "trl_bench.models.gte.generate_embeddings_train_test",
    "openai": "trl_bench.models.openai.generate_embeddings_train_test",
    "tabbie": "trl_bench.models.tabbie.generate_embeddings_train_test",
    "tabicl": "trl_bench.models.tabicl.generate_embeddings_train_test",
    "tabpfn": "trl_bench.models.tabpfn.generate_embeddings_train_test",
    "tuta":   "trl_bench.models.tuta.generate_row_embeddings",
}


# == Query-encoder dispatch (text_retrieval / semantic_parsing) =============
# Question-side embedding Stage-1: ``python -m
# trl_bench.models.<m>.generate_text_embeddings --mode cls --input_json <json>
# --text_field <question> --id_field <question_id> --output <pkl> [--device <d>]``.
#
# All wired wrappers share the same CLI surface (bert, gte, mpnet, sentence_t5
# verified 2026-05-20); only the runner module path and the default device-flag
# behavior vary. OpenAI has no ``--device`` flag (the client routes via the
# OpenAI HTTP API). The ``mode`` is always ``cls`` for retrieval (reference
# convention; ``token`` mode is reserved for semantic_parsing token-level
# encoding, which is a separate code path).
#
# The auto-orchestrator in ``run.py::_resolve_embeddings_path`` calls
# ``build_query_extractor_command`` once per (train, dev) split when the task
# is ``table_retrieval`` and the model has a wired query encoder. It writes
# the pickles alongside the table pickle (sibling files in
# ``<embeddings-dir>/table/<model>/``) — matching the convention the
# retrieval dispatcher reads from (``embeddings_path.parent`` lookup).

@dataclass(frozen=True)
class QueryEncoderConfig:
    """Stage-1 query-/question-text encoder subprocess dispatch for one model.

    Mirrors ``ExtractorConfig`` but for the text-encoder pipeline. The runner
    is invoked as::

        python -m <runner> --mode cls \
            --input_json <questions.json> --text_field <field> \
            --id_field <id_field> --output <pkl> [--device <d>]

    ``text_field`` and ``id_field`` default to ``question`` / ``question_id``
    (the table_retrieval convention). Set ``device_flag=None`` for wrappers
    without a device flag (OpenAI).
    """
    runner: str
    mode: str | None = "cls"  # None -> wrapper has no pooling mode (OpenAI API) -> omit --mode
    device_flag: str | None = "--device"
    device_value: str = "cuda"


_QUERY_ENCODER_EXTRACTORS: dict[str, QueryEncoderConfig] = {
    "bert":        QueryEncoderConfig(
        runner="trl_bench.models.bert.generate_text_embeddings",
    ),
    "gte":         QueryEncoderConfig(
        runner="trl_bench.models.gte.generate_text_embeddings",
    ),
    "mpnet":       QueryEncoderConfig(
        runner="trl_bench.models.mpnet.generate_text_embeddings",
    ),
    "sentence_t5": QueryEncoderConfig(
        runner="trl_bench.models.sentence_t5.generate_text_embeddings",
    ),
    "openai":      QueryEncoderConfig(
        runner="trl_bench.models.openai.generate_text_embeddings",
        mode=None,  # OpenAI embeddings have no cls/mean pooling -> runner has no --mode
        device_flag=None,
    ),
}


# semantic_parsing question embeddings come ONLY from these encoders: the model
# under test supplies COLUMN embeddings, the question side is always one of these
# two (default sentence_t5). Mirrors the benchmark's hard-coded set in
# slurm/generate_downstream_scripts.py::QUERY_ENCODER_MODELS. bert/gte are
# token-capable but are NOT question encoders for this task; openai is cls-only.
_SEMPARSE_QUESTION_ENCODERS: frozenset = frozenset({"sentence_t5", "mpnet"})


def build_query_extractor_command(
    *, model: str,
    input_json: str | Path,
    output_path: str | Path,
    text_field: str = "question",
    id_field: str = "question_id",
    device: str | None = None,
) -> list[str]:
    """Build the Stage-1 query-text-encoder subprocess command.

    Used by run.py to auto-orchestrate query-side encoding for
    ``table_retrieval``. ``input_json`` is the ``train.json`` or ``dev.json``
    produced by the nq_tables stager (sibling of ``labels.json``).
    ``output_path`` is the per-split queries pickle (e.g. ``queries_train.pkl``).

    Raises ``SettingError`` if the model has no wired query encoder.
    """
    if model not in _QUERY_ENCODER_EXTRACTORS:
        wired = sorted(_QUERY_ENCODER_EXTRACTORS)
        raise SettingError(
            f"model {model!r} has no query encoder wired in "
            f"`_QUERY_ENCODER_EXTRACTORS` (registry.py). Wired: {wired}. "
            f"Add a `QueryEncoderConfig` entry or pre-extract the queries pickle "
            f"and pass it via convention "
            f"(<embeddings_path.parent>/queries_train.pkl + queries_dev.pkl)."
        )
    cfg = _QUERY_ENCODER_EXTRACTORS[model]
    cmd: list[str] = [sys.executable, "-m", cfg.runner]
    # OpenAI's text-embedding runner has no pooling mode (cfg.mode is None); its
    # argparse rejects --mode, so omit it. All sentence/text encoders keep it.
    if cfg.mode is not None:
        cmd += ["--mode", cfg.mode]
    cmd += [
        "--input_json",  str(input_json),
        "--text_field",  text_field,
        "--id_field",    id_field,
        "--output",      str(output_path),
    ]
    if cfg.device_flag is not None:
        cmd += [cfg.device_flag, device or cfg.device_value]
    return cmd


def build_question_extractor_command(
    *, model: str,
    input_json: str | Path,
    output_path: str | Path,
    device: str | None = None,
) -> list[str]:
    """Build the semantic_parsing QUESTION-embedding command (token mode).

    MAPO's decoder needs per-token ``(seq_len, dim)`` question embeddings keyed
    by the WikiTableQuestions example ``id`` -> ``--mode token``,
    ``--tokens_field tokens``, ``--id_field id``. This is distinct from
    ``build_query_extractor_command`` (table_retrieval), which uses cls/single-
    vector pooling with ``question``/``question_id`` fields and would desync the
    per-token sequence length the MAPO environment indexes against.

    Raises ``SettingError`` unless ``model`` is one of the benchmark's
    designated question encoders (``_SEMPARSE_QUESTION_ENCODERS`` =
    {sentence_t5, mpnet}, default sentence_t5). The model under test supplies
    COLUMN embeddings, not questions; the question side is always one of those
    two. bert/gte (token-capable but not designated), openai (single-vector
    cls), and table models all redirect to the designated set.
    """
    if model not in _SEMPARSE_QUESTION_ENCODERS:
        raise SettingError(
            f"semantic_parsing question embeddings come only from the "
            f"benchmark's designated encoders "
            f"{sorted(_SEMPARSE_QUESTION_ENCODERS)} (the model under test "
            f"supplies column embeddings, not questions). --setting {model!r} "
            f"is not a valid question encoder; re-run with --setting sentence_t5 "
            f"(default) or mpnet."
        )
    cfg = _QUERY_ENCODER_EXTRACTORS[model]
    cmd: list[str] = [
        sys.executable, "-m", cfg.runner,
        "--mode",         "token",
        "--input_json",   str(input_json),
        "--tokens_field", "tokens",
        "--id_field",     "id",
        "--output",       str(output_path),
    ]
    if cfg.device_flag is not None:
        cmd += [cfg.device_flag, device or cfg.device_value]
    return cmd


# == Table-direct Stage-1 dispatch (no column step + Stage-2 skipped) ========
# Some wrappers don't emit per-column embeddings: text encoders that linearize
# the whole table to a single string (mpnet, sentence_t5) and TAPEX (which is
# table-only by design). Their Stage-1 script writes a TABLE-LEVEL pickle
# directly to ``<embeddings-dir>/table/<model>/<dataset>.pkl``; Stage-2
# aggregation is unnecessary and is skipped by the auto-orchestrator.
#
# The shared text-encoder runner is
# ``trl_bench.utils.generate_table_embeddings_text_encoder`` (one model per
# call -- the runner is generic over any HuggingFace AutoModel that exposes
# ``last_hidden_state``). TAPEX has its own runner because it uses a BART
# encoder + custom linearization. All three share an ``--input_dir`` /
# ``--output_path`` CLI surface and produce the same {table_id,
# table_embedding: {cls_embedding, column_mean, token_mean, ...}, ...} schema.

@dataclass(frozen=True)
class TableEncoderConfig:
    """Stage-1 TABLE-direct extraction dispatch for one model.

    Used for wrappers that produce a table-level pickle in a single pass and
    therefore skip the column step + Stage-2 aggregation. The runner is
    invoked as::

        python -m <runner> <input_flag> <csv-dir> <output_flag> <pkl> \\
            --model <hf-id> [extras...]

    where ``input_flag`` / ``output_flag`` default to ``--input_dir`` /
    ``--output_path`` (matching the table-direct convention). The shared
    text-encoder runner expects ``--model`` + ``--pooling``; TAPEX's runner
    accepts ``--model`` (default ``microsoft/tapex-base``, the paper
    checkpoint) and has no ``--pooling`` flag.
    """
    runner: str
    pooling: str           # "cls" or "mean" -- documented for parity even
                           # when the runner ignores the flag (TAPEX)
    model_id: str          # HF model id (always emitted via --model)
    extra_args: tuple[tuple[str, str], ...] = ()
    input_flag: str = "--input_dir"
    output_flag: str = "--output_path"


_TABLE_ENCODERS: dict[str, TableEncoderConfig] = {
    "mpnet": TableEncoderConfig(
        runner="trl_bench.utils.generate_table_embeddings_text_encoder",
        pooling="mean",
        model_id="sentence-transformers/all-mpnet-base-v2",
        extra_args=(("--pooling", "mean"),),
    ),
    "sentence_t5": TableEncoderConfig(
        runner="trl_bench.utils.generate_table_embeddings_text_encoder",
        pooling="mean",
        model_id="sentence-transformers/sentence-t5-base",
        extra_args=(("--pooling", "mean"),),
    ),
    "tapex": TableEncoderConfig(
        # TAPEX uses its own runner (BART encoder + TAPEX linearization).
        # No --pooling flag on the wrapper; pooling is hardcoded to
        # mean-pool of non-padding encoder tokens.
        runner="trl_bench.models.tapex.generate_table_embeddings",
        pooling="mean",
        model_id="microsoft/tapex-base",
        extra_args=(),
    ),
}


def build_table_encoder_command(
    *, model: str, dataset: str,
    input_dir: str | Path,
    output_path: str | Path,
) -> list[str]:
    """Build the Stage-1 table-DIRECT extraction command for one model.

    Used for wrappers that emit table-level embeddings in a single pass
    (no column step). The auto-orchestrator in ``run.py`` calls this
    *instead of* ``build_extractor_command`` + the Stage-2 aggregator
    when ``model`` appears in ``_TABLE_ENCODERS``.

    ``input_dir`` is the ``tables_all/`` directory produced by Stage-0
    staging; ``output_path`` is the table pickle the runner writes
    (typically ``<embeddings-dir>/table/<model>/<dataset>.pkl``).

    Raises ``SettingError`` if the model has no table-encoder entry.
    """
    if model not in _TABLE_ENCODERS:
        wired = sorted(_TABLE_ENCODERS)
        raise SettingError(
            f"model {model!r} has no table-direct extractor wired in "
            f"`_TABLE_ENCODERS` (registry.py). Wired: {wired}. "
            f"Add a `TableEncoderConfig` entry or pass --embeddings-path."
        )
    cfg = _TABLE_ENCODERS[model]
    cmd: list[str] = [
        sys.executable, "-m", cfg.runner,
        cfg.input_flag,  str(input_dir),
        cfg.output_flag, str(output_path),
        "--model", cfg.model_id,
    ]
    for flag, value in cfg.extra_args:
        cmd += [flag, value]
    return cmd


def build_extractor_command(
    *, model: str, dataset: str,
    input_dir: str | Path,
    output_path: str | Path,
    device: str | None = None,
    checkpoint_root: str | Path | None = None,
    runner_override: str | None = None,
) -> list[str]:
    """Build the Stage-1 column-/row-extraction subprocess command for one model.

    ``input_dir`` should be the ``tables_all/`` directory produced by Stage-0
    staging; ``output_path`` is the embedding pickle the script writes.
    ``device`` overrides ``ExtractorConfig.device_value`` (e.g. "cpu" on
    GPU-less hosts); pass ``None`` to use the configured default. For
    wrappers with ``device_flag=None`` (e.g. ``openai``, the trained-row
    family), the ``device`` argument is silently ignored — those wrappers
    have no ``--device`` flag. Wrappers with a ``device_value_map`` (e.g.
    TUTA / TABBIE ``--device_id <int>``) translate the canonical "cuda" /
    "cpu" override into the wrapper-specific value.

    ``checkpoint_root`` is the on-disk root for licensed-checkpoint
    wrappers (TaBERT, TabSketchFM, TURL, TUTA, TABBIE, Starmie). When the
    model's ``ExtractorConfig.checkpoint_template`` is set, the dispatcher
    appends ``<cfg.checkpoint_arg> <checkpoint_root>/<resolved-template>``
    to the command. ``checkpoint_root`` defaults to ``./checkpoints``
    (relative to CWD) -- override via the function kwarg, the
    ``--checkpoint-root`` CLI flag in ``trl_bench.run``, or the
    ``$TRL_BENCH_CKPT_ROOT`` env var. Wrappers without a checkpoint
    template ignore this argument.

    Raises ``SettingError`` if:
      - the model is not yet wired into ``_MODEL_EXTRACTORS`` (the exception
        message names the registry constant so callers know where to add
        the entry); or
      - the resolved checkpoint path does not exist AND the wrapper has
        ``checkpoint_required=True``.
    """
    if model not in _MODEL_EXTRACTORS:
        wired = sorted(_MODEL_EXTRACTORS)
        raise SettingError(
            f"model {model!r} has no Stage-1 extractor wired in "
            f"`_MODEL_EXTRACTORS` (registry.py). Wired: {wired}. "
            f"Add an `ExtractorConfig` entry or pass --embeddings-path."
        )
    cfg = _MODEL_EXTRACTORS[model]
    output_path_p = Path(str(output_path))
    template_env = {
        "output_parent": str(output_path_p.parent),
        "output_stem":   output_path_p.stem,
        "dataset":       dataset,
    }
    runner = runner_override or cfg.runner
    # record_linkage Stage-1 swaps the table extractor for a row-embedding
    # runner (``runner_override``). Those runners uniformly use the directory
    # convention ``--input_dir``/``--output_path`` -- even for models whose
    # TABLE extractor uses the BERT-shape ``--input``/``--output`` (bert, gte,
    # openai, tapas). Emitting cfg's table flags to a row runner argparse-fails
    # (the record_linkage bug found for bert/gte/openai/tabbie). So a row
    # override forces the directory convention; normal extraction keeps cfg's.
    in_flag = "--input_dir" if runner_override else cfg.input_flag
    out_flag = "--output_path" if runner_override else cfg.output_flag
    cmd: list[str] = [
        sys.executable, "-m", runner,
        in_flag,  str(input_dir),
        out_flag, str(output_path),
    ]
    for flag, value in cfg.extra_args:
        cmd += [flag, value]
    for flag, template in cfg.derived_args:
        try:
            cmd += [flag, template.format(**template_env)]
        except KeyError as exc:  # pragma: no cover - guard for future templates
            raise SettingError(
                f"model {model!r} derived_args references unknown template "
                f"token {exc.args[0]!r}; supported: {sorted(template_env)}."
            ) from exc
    if cfg.checkpoint_template is not None:
        # Resolve <checkpoint_root>/<template-with-{dataset}-etc>.
        ckpt_root = Path(str(checkpoint_root)) if checkpoint_root is not None \
            else Path("checkpoints")
        try:
            resolved_rel = cfg.checkpoint_template.format(**template_env)
        except KeyError as exc:
            raise SettingError(
                f"model {model!r} checkpoint_template references unknown "
                f"template token {exc.args[0]!r}; supported: "
                f"{sorted(template_env)}."
            ) from exc
        resolved = ckpt_root / resolved_rel
        if cfg.checkpoint_required and not resolved.exists():
            raise SettingError(
                f"model {model!r}: required checkpoint not found at "
                f"{resolved}. Download via "
                f"`bash scripts/download_checkpoints.sh {model}` or place "
                f"the file manually (see docs/CHECKPOINT_LICENSES.md for "
                f"the upstream URL + license)."
            )
        cmd += [cfg.checkpoint_arg, str(resolved)]
    if cfg.device_flag is not None:
        raw_device = device or cfg.device_value
        if cfg.device_value_map:
            value_map = dict(cfg.device_value_map)
            mapped = value_map.get(raw_device, raw_device)
            cmd += [cfg.device_flag, mapped]
        else:
            cmd += [cfg.device_flag, raw_device]
    return cmd


def starmie_checkpoint_path(dataset: str, checkpoint_root: str | Path) -> Path:
    """Resolve the per-dataset starmie pretrained checkpoint ``build_extractor_command``
    expects, from ``ExtractorConfig.checkpoint_template``. Single source of truth
    so the auto-pretrain hook and the extractor agree on the path."""
    cfg = _MODEL_EXTRACTORS["starmie"]
    return Path(str(checkpoint_root)) / cfg.checkpoint_template.format(dataset=dataset)


def build_starmie_pretrain_command(
    dataset: str, tables_dir: str | Path, checkpoint_root: str | Path,
) -> list[str]:
    """Build the starmie self-supervised pre-pretraining command.

    starmie has no released checkpoint -- each cell trains a contrastive encoder
    on its own dataset first. ``run_pretrain`` writes
    ``<checkpoint_dir>/datalake/<model>.pt``, so ``--checkpoint_dir`` is
    ``<ckpt_root>/starmie/<dataset>`` to land the .pt exactly where
    ``starmie_checkpoint_path`` (and thus the extractor) resolves it.
    """
    ckpt_dir = Path(str(checkpoint_root)) / "starmie" / dataset
    cfg = _MODEL_EXTRACTORS["starmie"]
    # checkpoint_template == starmie/<dataset>/<subdir>/<file>; take <subdir> so
    # the pinned pretrain subdir can never drift from what the extractor reads.
    # run_pretrain otherwise defaults the subdir to basename(--data_path), which
    # is 'datalake' only for union_search inputs -- pinning it makes the .pt land
    # where build_extractor_command resolves it for EVERY dataset.
    subdir = Path(cfg.checkpoint_template.format(dataset=dataset)).parts[-2]
    return [
        sys.executable, "-m", "trl_bench.models.starmie.run_pretrain",
        "--data_path", str(tables_dir),
        "--checkpoint_dir", str(ckpt_dir),
        "--checkpoint_subdir", subdir,
        # Match the paper checkpoints' baked-in hp (torch.load(ckpt)['hp']):
        # max_len=256, fp16, max_rows=1000. run_pretrain DEFAULTS to max_len=128,
        # which truncates each serialized table to HALF -> the contrastive loss
        # stalls ~4.5 and recall@gt collapses 0.76->0.09 (verified e2e). mlflow
        # never logged these, so the checkpoint hp is the source of truth.
        "--max_len", "256",
        "--max_rows", "1000",
        "--fp16",
        "--save_model",
    ]


def build_row_data_commands(
    model: str,
    dataset_dir: str | Path,
    checkpoint_dir: str | Path,
    embedding_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    checkpoint_root: str | Path | None = None,
) -> list[list[str]]:
    """Build the row_prediction Stage-1 subprocess command(s) for a row-data
    model, from ``slurm/config/row_data_models.yaml``.

    Trained (self-supervised) models -- scarf/dae/subtab/vime/saint/transtab/
    tabular_binning/tabtransformer -- emit ``[train_cmd, generate_cmd]``: a train
    pass writes a per-dataset checkpoint, then a generate pass loads it and writes
    the unified_row_embedding directory the probe consumes. Pretrained models
    (tabicl/tabpfn/...) emit ``[generate_cmd]`` only. Mirrors
    slurm/generate_row_data_scripts.py so ``run.py --task row_prediction`` runs
    the same pipeline as the slurm path -- the previously "not wired" branch.

    ``checkpoint_root`` is the on-disk root for licensed pretrained checkpoints
    (TUTA, TABBIE). When the YAML model config carries a ``checkpoint:`` value AND
    its args map declares ``model_path``, the generate command appends
    ``<model_path-flag> <resolved-checkpoint>`` -- mirroring the canonical slurm
    generator (slurm/generate_row_data_scripts.py:115-118), which TUTA's and
    TABBIE's row runners REQUIRE (argparse ``required=True``). The path resolves
    as ``<checkpoint_root>/<yaml-checkpoint with a single leading 'checkpoints/'
    stripped>`` (so a passed ``--checkpoint-root`` lands the same
    ``<root>/<template>`` shape the column extractors use via
    ``ExtractorConfig.checkpoint_template``); when ``checkpoint_root`` is None the
    value falls back to ``repo_root/<yaml-checkpoint>`` (the slurm default, i.e.
    ``./checkpoints``). Models without a ``checkpoint:`` key (HF bert/gte;
    checkpoint-less pretrained tabicl/tabpfn; all trained SSL models) never emit
    ``--model_path``.
    """
    import yaml
    repo_root = Path(__file__).resolve().parents[2]
    path = (Path(str(config_path)) if config_path
            else repo_root / "slurm" / "config" / "row_data_models.yaml")
    models = (yaml.safe_load(path.read_text()) or {})
    models = models.get("models", models)
    if model not in models:
        raise SettingError(
            f"model {model!r} has no row-data config in {path}. "
            f"Wired: {sorted(models)}."
        )
    cfg = models[model]
    args_map = cfg.get("args", {})
    defaults = {**(cfg.get("defaults") or {}), **(cfg.get("model_defaults") or {})}
    skip = {"data_dir", "checkpoint_dir", "embedding_dir", "model_path"}

    # Resolve the licensed-checkpoint path for ``--model_path`` (TUTA/TABBIE).
    # Only models with BOTH a ``checkpoint:`` key and ``model_path`` in their args
    # map get one; everything else resolves to None (no flag emitted). This
    # mirrors slurm/generate_row_data_scripts.py:build_path_args exactly.
    model_checkpoint = cfg.get("checkpoint")
    resolved_model_path: str | None = None
    if model_checkpoint and "model_path" in args_map:
        if checkpoint_root is not None:
            # Strip a single leading "checkpoints/" so the YAML's repo-relative
            # value (e.g. "checkpoints/tuta/tuta.bin") rebinds under the passed
            # root as "<root>/tuta/tuta.bin" -- the same <root>/<template> shape
            # the column extractors resolve via ExtractorConfig.checkpoint_template.
            rel = str(model_checkpoint)
            for prefix in ("checkpoints/", "./checkpoints/"):
                if rel.startswith(prefix):
                    rel = rel[len(prefix):]
                    break
            resolved_model_path = str(Path(checkpoint_root) / rel)
        else:
            # Slurm default: project_root / <yaml checkpoint> (i.e. ./checkpoints).
            resolved_model_path = str(repo_root / model_checkpoint)

    def paths(include_embedding: bool) -> list[str]:
        parts: list[str] = []
        if "data_dir" in args_map:
            parts += [args_map["data_dir"], str(dataset_dir)]
        if "checkpoint_dir" in args_map:
            parts += [args_map["checkpoint_dir"], str(checkpoint_dir)]
        if include_embedding and "embedding_dir" in args_map:
            parts += [args_map["embedding_dir"], str(embedding_dir)]
        # The pretrained checkpoint (--model_path) goes on the GENERATE command
        # (include_embedding=True). Pretrained models (tuta/tabbie) have only a
        # generate pass; trained models never reach here (no checkpoint: key).
        if include_embedding and resolved_model_path is not None:
            parts += [args_map["model_path"], resolved_model_path]
        return parts

    def default_flags() -> list[str]:
        flags: list[str] = []
        for param, value in defaults.items():
            if param in skip or value is None:
                continue
            if isinstance(value, (list, tuple)):
                flags += [f"--{param}", *[str(v) for v in value]]
            else:
                flags += [f"--{param}", str(value)]
        return flags

    def script(rel: str) -> str:
        return str(repo_root / rel)

    if cfg.get("model_type", "pretrained") == "trained":
        # train (all defaults: hyperparams + label_policy) then generate (paths;
        # the generator loads training artifacts, so no defaults on generate).
        return [
            [sys.executable, script(cfg["train_script"])]
            + paths(include_embedding=False) + default_flags(),
            [sys.executable, script(cfg["generate_script"])]
            + paths(include_embedding=True),
        ]
    # pretrained: one generate pass with all defaults.
    return [[sys.executable, script(cfg["generate_script"])]
            + paths(include_embedding=True) + default_flags()]


def _resolve_output_dir(
    *, task: str, model: str, dataset: str, setting: str, probe: str,
    seed: int, path_layout: str, results_dir: Path,
) -> Path:
    """Compute Stage-3 output directory per the configured path layout.

    Three variants mirror the reference layout (see ``ProbeConfig`` docs).
    """
    if path_layout == "without_setting":
        return (results_dir / "evaluation" / task / model
                / f"seed{seed}" / probe)
    if path_layout == "with_dataset":
        return (results_dir / "evaluation" / task / model / dataset
                / f"seed{seed}" / probe)
    if path_layout == "with_dataset_label":
        # Setting carries the label column name (row_prediction). The runner
        # APPENDS ``<label_col>/`` itself when --label_column is passed, so
        # the dispatcher's ``--output_dir`` must NOT include the label suffix
        # (else the runner would write
        # ``.../seed<S>/<label_col>/<label_col>/results.json``, breaking the
        # envelope-write path).
        # The runner ends up writing ``.../seed<S>/<label_col>/results.json``,
        # matching the reference's
        # ``<task>/<model>/<dataset>/seed<S>/<label_col>/`` layout.
        return (results_dir / "evaluation" / task / model / dataset
                / f"seed{seed}")
    if path_layout == "semparse":
        # semantic_parsing: <task>/<model>/<setting>/seed<S>/ (no probe axis;
        # the setting is the query-encoder model — mpnet / sentence_t5).
        return (results_dir / "evaluation" / task / model / setting
                / f"seed{seed}")
    if path_layout == "dlte":
        # DLTE: <results>/dlte/<stage-key>/<model-key>/ — no probe axis, no
        # seed dimension (the runners are deterministic per (model_key, data)).
        # The model_key is the diagonal triple ``<col>__<row>__<table>`` for
        # stage3/metrics; for retrieval and alignment the upstream runners
        # write under their own keys (handled by the family builders).
        return (results_dir / "evaluation" / task / model / setting)
    if path_layout == "deterministic":
        # Deterministic training-free tasks (column_clustering /
        # schema_matching / union_search / join_search): the reference
        # layout is FLAT — ``<task>/<model>/`` with no seed/probe subdirs.
        # The envelope file is named ``<model>_<dataset>.json`` in this dir.
        return (results_dir / "evaluation" / task / model)
    return (results_dir / "evaluation" / task / model / setting
            / f"seed{seed}" / probe)


def _resolve_yaml(cfg_yaml: str, configs_root: Path) -> Path:
    """Resolve a YAML config path: prefer ``configs_root / <basename>`` if it
    exists; otherwise fall back to the configured (typically relative) path.
    """
    yaml_path = configs_root / Path(cfg_yaml).name
    if not yaml_path.exists():
        yaml_path = Path(cfg_yaml)
    return yaml_path


def _build_pair_task_cmd(
    *, cfg: ProbeConfig, model: str, task: str, dataset: str, setting: str,
    probe: str, seed: int, embeddings_path: Path, labels_path: Path,
    output_dir: Path, configs_root: Path,
) -> list[list[str]]:
    """Build the canonical pair-task / record_linkage Stage-3 command.

    Mirrors the canonical .sbatch's call to ``python utils/downstream/run_task.py``
    used by join_classification, union_classification, union_regression,
    join_containment, table_subset, record_linkage.
    """
    if setting not in _SETTING_TO_EMBEDDING_TYPE:
        raise SettingError(
            f"unknown setting {setting!r}. Valid: {sorted(_SETTING_TO_EMBEDDING_TYPE)}"
        )
    embedding_type = _SETTING_TO_EMBEDDING_TYPE[setting]
    task_name = cfg.task_name_template.format(dataset=dataset)
    yaml_path = _resolve_yaml(cfg.yaml, configs_root)

    cmd: list[str] = [
        sys.executable, "-m", cfg.runner,
        "--embeddings",         str(embeddings_path),
        "--labels",             str(labels_path),
        "--task_name",          task_name,
        "--output_dir",         str(output_dir),
        "--config",             str(yaml_path),
        "--combination_method", cfg.combination_method,
        "--hidden_dim",         str(cfg.hidden_dim),
        "--batch_size",         str(cfg.batch_size),
        "--max_epochs",         str(cfg.max_epochs),
        "--learning_rate",      str(cfg.learning_rate),
        "--dropout_prob",       str(cfg.dropout_prob),
        "--seed",               str(seed),
        "--head_type",          probe,
    ]
    if cfg.pass_task_type:
        cmd += ["--task_type", cfg.task_type]
    if cfg.pass_embedding_type:
        cmd += ["--embedding_type", embedding_type]
    if cfg.num_labels > 0:
        cmd += ["--num_labels", str(cfg.num_labels)]
    return [cmd]


def _build_cta_cmd(
    *, cfg: ProbeConfig, model: str, task: str, dataset: str, setting: str,
    probe: str, seed: int, embeddings_path: Path, labels_path: Path,
    output_dir: Path, configs_root: Path,
) -> list[list[str]]:
    """Build the column_type_prediction (CTA) Stage-3 command.

    Mirrors the canonical .sbatch's call::

        python -m trl_bench.tasks.column_type_prediction.train_ct_mode4 \\
            --embeddings <pkl> --dataset <ds_dir> \\
            --num_epochs 10 --batch_size 20 --learning_rate 0.001 \\
            --output_dir <out> --seed <S> --head_type <probe>

    CTA's ``--dataset`` is the on-disk directory containing ``train.csv`` /
    ``test.csv`` (Stage-0 staging output); for unit-test purposes here we
    take the parent of ``labels_path`` as that directory by convention.
    """
    dataset_dir = labels_path.parent
    cmd: list[str] = [
        sys.executable, "-m", cfg.runner,
        "--embeddings",    str(embeddings_path),
        "--dataset",       str(dataset_dir),
        "--num_epochs",    str(cfg.max_epochs),
        "--batch_size",    str(cfg.batch_size),
        "--learning_rate", str(cfg.learning_rate),
        "--output_dir",    str(output_dir),
        "--seed",          str(seed),
        "--head_type",     probe,
    ]
    return [cmd]


def _build_cra_cmd(
    *, cfg: ProbeConfig, model: str, task: str, dataset: str, setting: str,
    probe: str, seed: int, embeddings_path: Path, labels_path: Path,
    output_dir: Path, configs_root: Path,
) -> list[list[str]]:
    """Build the column_relation_prediction (CRA) Stage-3 command.

    Mirrors the canonical .sbatch::

        python -m trl_bench.tasks.column_relation_prediction.csv_relation_pipeline \\
            --embeddings_file <pkl> --dataset_dir <ds_dir> \\
            --epochs 20 --batch_size 32 --lr 0.001 --hidden_dim 256 \\
            --output_dir <out> --seed <S> --head_type <probe>

    CRA's ``--dataset_dir`` is the on-disk dataset root; for unit-test purposes
    we take the parent of ``labels_path`` by convention.
    """
    dataset_dir = labels_path.parent
    cmd: list[str] = [
        sys.executable, "-m", cfg.runner,
        "--embeddings_file", str(embeddings_path),
        "--dataset_dir",     str(dataset_dir),
        "--epochs",          str(cfg.max_epochs),
        "--batch_size",      str(cfg.batch_size),
        "--lr",              str(cfg.learning_rate),
        "--hidden_dim",      str(cfg.hidden_dim),
        "--output_dir",      str(output_dir),
        "--seed",            str(seed),
        "--head_type",       probe,
    ]
    return [cmd]


def _build_retrieval_cmd(
    *, cfg: ProbeConfig, model: str, task: str, dataset: str, setting: str,
    probe: str, seed: int, embeddings_path: Path, labels_path: Path,
    output_dir: Path, configs_root: Path,
) -> list[list[str]]:
    """Build the table_retrieval Stage-3 command(s).

    Two-stage: ``train.py`` produces ``best_model.pt`` under ``--output_dir``,
    then ``evaluate.py --projection_head <output_dir>/best_model.pt`` writes
    the metrics JSON to ``<output_dir>/results.json``.

    The retrieval CLI needs paths not present in ``build_command``'s public
    signature (query-side embeddings, train/dev questions, table_id_mapping).
    We derive them from ``embeddings_path`` and ``labels_path`` by convention:

        train_query_embeddings  := <embeddings_path.parent>/queries_train.pkl
        dev_query_embeddings    := <embeddings_path.parent>/queries_dev.pkl
        train_questions         := <labels_path.parent>/train.json
        dev_questions           := <labels_path.parent>/dev.json
        table_id_mapping        := <labels_path.parent>/table_id_to_csv.json

    Stage-0 staging materializes these files for nq_tables (see
    ``trl_bench.data.stage``); the dispatcher auto-stages them on first run.
    """
    # Cross-encoder cell encoding (table_retrieval is NOT a single-axis probe):
    #   setting = table-side POOLING (cls_embedding / column_mean / token_mean)
    #   probe   = QUERY ENCODER + retrieval MODE:
    #               "<enc>"            -> hybrid    (model+encoder table emb concat)
    #               "<enc>_modelonly"  -> model_only(raw model table emb)
    #             with enc in {mpnet, sentence_t5}.
    # The query side is ALWAYS the encoder (not the table model) — the canonical
    # table_retrieval is a cross-encoder pipeline. Training uses the canonical
    # config (cosine + temp 0.1 + 1-layer + no-adapter + 2-round hard-negative
    # mining).
    pooling = setting
    pr = probe or "mpnet"
    model_only = pr.endswith("_modelonly")
    encoder = pr[: -len("_modelonly")] if model_only else pr

    emb_table_dir = embeddings_path.parent.parent      # <emb_root>/table
    emb_root      = emb_table_dir.parent               # <emb_root>
    labels_dir = labels_path.parent
    train_q  = labels_dir / "train.json"
    dev_q    = labels_dir / "dev.json"
    table_id_mapping = labels_dir / "table_id_to_csv.json"
    # query embeddings come from the QUERY ENCODER, not the table model
    qdir = emb_root / "table_retrieval" / encoder
    train_q_emb = qdir / "queries_train.pkl"
    dev_q_emb   = qdir / "queries_dev.pkl"

    best_model = output_dir / "best_model.pt"
    results_json = output_dir / "results.json"

    stages: list[list[str]] = []
    if model_only:
        table_emb = embeddings_path                    # raw model table embeddings
    else:
        # hybrid: concat the table model + query-encoder table embeddings
        table_emb = emb_table_dir / f"{model}_{encoder}_hybrid" / pooling / "nq_tables.pkl"
        stages.append([
            sys.executable, "-m",
            "trl_bench.tasks.table_retrieval.create_hybrid_embeddings",
            "--base_embeddings", str(embeddings_path),
            "--bert_embeddings", str(emb_table_dir / encoder / "nq_tables.pkl"),
            "--base_variant",    pooling,
            "--bert_variant",    "column_mean",
            "--combination_method", "concat",
            "--table_id_mapping", str(table_id_mapping),
            "--output_path",     str(table_emb),
        ])

    stages.append([
        sys.executable, "-m", cfg.runner,
        "--table_embeddings",       str(table_emb),
        "--table_id_mapping",       str(table_id_mapping),
        "--train_query_embeddings", str(train_q_emb),
        "--train_questions",        str(train_q),
        "--dev_query_embeddings",   str(dev_q_emb),
        "--dev_questions",          str(dev_q),
        "--output_dir",             str(output_dir),
        "--epochs",                 str(cfg.max_epochs),
        "--batch_size",             str(cfg.batch_size),
        "--learning_rate",          str(cfg.learning_rate),
        "--projection_dim",         str(cfg.hidden_dim),
        "--hidden_dim",             str(cfg.hidden_dim),
        "--num_layers",             "1",
        "--no_adapter",
        "--similarity_fn",          "cosine",
        "--temperature",            "0.1",
        "--refinement_rounds",      "2",
        "--embedding_variant",      pooling,
        "--seed",                   str(seed),
    ])
    stages.append([
        sys.executable, "-m", "trl_bench.tasks.table_retrieval.evaluate",
        "--table_embeddings", str(table_emb),
        "--table_id_mapping", str(table_id_mapping),
        "--query_embeddings", str(dev_q_emb),
        "--questions_path",   str(dev_q),
        "--projection_head",  str(best_model),
        "--output_path",      str(results_json),
        "--embedding_type",   pooling,
    ])
    return stages


def _build_row_prediction_cmd(
    *, cfg: ProbeConfig, model: str, task: str, dataset: str, setting: str,
    probe: str, seed: int, embeddings_path: Path, labels_path: Path,
    output_dir: Path, configs_root: Path,
) -> list[list[str]]:
    """Build the row_prediction Stage-3 command.

    Mirrors the canonical sbatch::

        python -m trl_bench.tasks.row_prediction.train_downstream \\
            --embedding_dir <emb_dir> --output_dir <out> --config <yaml> \\
            --seed <S> --model <m> --dataset <ds> \\
            --head_type <probe> --label_column <setting>

    For row_prediction the runner consumes a DIRECTORY of .npy files
    (``--embedding_dir`` not ``--embeddings``), so ``embeddings_path`` is
    passed through as-is. The setting axis carries the label column name
    (matching the reference's per-label subdir layout). YAML-driven
    hyperparameters override anything not explicitly passed here.
    """
    yaml_path = _resolve_yaml(cfg.yaml, configs_root)
    cmd: list[str] = [
        sys.executable, "-m", cfg.runner,
        "--embedding_dir", str(embeddings_path),
        "--output_dir",    str(output_dir),
        "--config",        str(yaml_path),
        "--seed",          str(seed),
        "--model",         model,
        "--dataset",       dataset,
        "--head_type",     probe,
        "--label_column",  setting,
    ]
    return [cmd]


def _build_semparse_cmd(
    *, cfg: ProbeConfig, model: str, task: str, dataset: str, setting: str,
    probe: str, seed: int, embeddings_path: Path, labels_path: Path,
    output_dir: Path, configs_root: Path,
) -> list[list[str]]:
    """Build the semantic_parsing Stage-3 command(s).

    Two-stage: ``run_training.py`` trains a MAPO decoder and writes
    ``model.best.bin`` under ``--output-dir``; then ``run_test.py`` loads it
    and writes ``test.log`` (a JSON dump of {accuracy, oracle_accuracy}) into
    the same directory.

    Mirrors the canonical invocation::

        python -m trl_bench.tasks.semantic_parsing.run_training \\
            --task wiki_table_questions --decoder mapo \\
            --column-pkl <col_pkl> \\
            --question-pkls <q_train_pkl> <q_dev_pkl> \\
            --dataset-path <ds_dir> --output-dir <out> \\
            --config <config.json> --seed <S> --cuda

        python -m trl_bench.tasks.semantic_parsing.run_test \\
            --model <out>/model.best.bin \\
            --column-pkl <col_pkl> \\
            --question-pkls <q_test_pkl> \\
            --test-file <ds_dir>/data_split_1/test_split.jsonl \\
            --table-file <ds_dir>/tables.jsonl --cuda

    The dispatcher derives query-side embedding and dataset-file paths from
    ``embeddings_path`` (the column pickle) and ``labels_path`` (any file
    inside the WikiTableQuestions staged directory; we take its parent):

        questions_dir       := <embeddings_path.parent>/<setting>/
        questions_train_pkl := <questions_dir>/questions_train.pkl
        questions_dev_pkl   := <questions_dir>/questions_dev.pkl
        questions_test_pkl  := <questions_dir>/questions_test.pkl
        dataset_dir         := <labels_path.parent>
        test_file           := <dataset_dir>/data_split_1/test_split.jsonl
        table_file          := <dataset_dir>/tables.jsonl

    The dispatcher assumes the WikiTableQuestions directory layout above is
    pre-materialized; ``semantic_parsing`` runs against that staged WTQ tree.

    The MAPO JSON config ships inside the task package at
    ``trl_bench/tasks/semantic_parsing/config/mapo.json``; the dispatcher
    resolves it relative to that package so callers don't need to pass a
    ``--config`` path explicitly.
    """
    from importlib.resources import files as _resource_files

    # The runner ships its own JSON config (NOT a YAML in configs_root).
    try:
        mapo_cfg = _resource_files(
            "trl_bench.tasks.semantic_parsing.config"
        ).joinpath("mapo.json")
        config_path = Path(str(mapo_cfg))
    except (ModuleNotFoundError, FileNotFoundError):
        # Test-time / loose-checkout fallback: walk up from this file.
        here = Path(__file__).resolve().parent
        config_path = here / "tasks" / "semantic_parsing" / "config" / "mapo.json"

    # The setting axis is the query-encoder model (mpnet / sentence_t5). The
    # per-encoder question pickles live as siblings of the column pickle,
    # under a per-encoder subdirectory keyed by the encoder name.
    questions_dir = embeddings_path.parent / setting
    questions_train_pkl = questions_dir / "questions_train.pkl"
    questions_dev_pkl   = questions_dir / "questions_dev.pkl"
    questions_test_pkl  = questions_dir / "questions_test.pkl"

    dataset_dir = labels_path.parent
    test_file   = dataset_dir / "data_split_1" / "test_split.jsonl"
    table_file  = dataset_dir / "tables.jsonl"

    train_cmd: list[str] = [
        sys.executable, "-m", cfg.runner,
        "--task",          dataset,
        "--decoder",       "mapo",
        "--column-pkl",    str(embeddings_path),
        "--question-pkls", str(questions_train_pkl), str(questions_dev_pkl),
        "--dataset-path",  str(dataset_dir),
        "--output-dir",    str(output_dir),
        "--config",        str(config_path),
        "--seed",          str(seed),
        "--cuda",
    ]

    eval_runner = "trl_bench.tasks.semantic_parsing.run_test"
    eval_cmd: list[str] = [
        sys.executable, "-m", eval_runner,
        "--model",         str(output_dir / "model.best.bin"),
        "--column-pkl",    str(embeddings_path),
        "--question-pkls", str(questions_test_pkl),
        "--test-file",     str(test_file),
        "--table-file",    str(table_file),
        "--cuda",
    ]
    return [train_cmd, eval_cmd]


# DLTE setting axis: pack auxiliary model names (col_model / table_model) into
# a single string. The diagonal default (model fills every axis) is the most-
# common reference cell.
_DLTE_DIAGONAL_SETTING = "diagonal"


def _parse_dlte_setting(setting: str, *, expect: int) -> list[str]:
    """Parse a DLTE setting axis into ``expect`` auxiliary model names.

    Examples::

        >>> _parse_dlte_setting("bert", expect=1)
        ['bert']
        >>> _parse_dlte_setting("bert__gte", expect=2)
        ['bert', 'gte']
        >>> _parse_dlte_setting("diagonal", expect=2)
        []

    Raises ``SettingError`` when the number of ``__``-separated tokens does not
    match ``expect`` (and the value is not the diagonal sentinel).
    """
    if setting in ("", _DLTE_DIAGONAL_SETTING):
        return []
    tokens = setting.split("__")
    if len(tokens) != expect:
        raise SettingError(
            f"dlte setting {setting!r} should have {expect} `__`-separated "
            f"model name(s); got {len(tokens)}. Use {_DLTE_DIAGONAL_SETTING!r} "
            f"for the diagonal default (model fills every axis)."
        )
    return tokens


def _resolve_dlte_output_root(output_dir: Path) -> Path:
    """Resolve the DLTE output root from ``_resolve_output_dir``'s output.

    For the ``dlte`` path_layout the dispatcher computes
    ``<results>/evaluation/<task>/<model>/<setting>/``; the DLTE runners take
    a single ``--output_root`` that the dispatcher should resolve to
    ``<results>/evaluation/dlte/`` so all three stages share state (Stage 2
    reads Stage 1's outputs under ``<output_root>/stage1/``, etc.). The
    runners DO NOT prepend ``dlte/`` to ``--output_root`` when one is supplied
    — they only do so via their default-path computation when no
    ``--output_root`` is given. (See step8/9/10 ``resolve_paths``: ``output_root
    = Path(args.output_root) if args.output_root else <project>/assets/
    evaluation_results/dlte``.)
    """
    # output_dir == <results>/evaluation/<task>/<model>/<setting>/
    # Walk up 4 parents -> <results>/, then we append "evaluation/dlte" so the
    # final value matches the runners' default ``.../evaluation_results/dlte``
    # layout naming convention. (Earlier .parent.parent.parent gave only 3
    # levels which left ``evaluation/`` in the path and produced a duplicated
    # ``evaluation/evaluation/`` directory.)
    results_root = output_dir.parents[3]
    return results_root / "evaluation" / "dlte"


def _build_dlte_retrieval_cmd(
    *, cfg: ProbeConfig, model: str, task: str, dataset: str, setting: str,
    probe: str, seed: int, embeddings_path: Path, labels_path: Path,
    output_dir: Path, configs_root: Path,
) -> list[list[str]]:
    """Build the DLTE Stage-1 (FAISS retrieval) command.

    Mirrors the canonical invocation::

        python -m trl_bench.tasks.dlte.scripts.step8_faiss_retrieval \\
            --models <table_model> \\
            --table_variant <variant> \\
            --output_root <out>/dlte \\
            --project_root <project_root>     [optional]
            --embeddings_root <emb_root>      [optional]

    The ``model`` argument IS the table_model — Stage-1 retrieval is run with
    one table-model pickle at a time. The ``setting`` axis is unused by the
    runner; if non-default, we accept it as the table-embedding variant
    (``cls_embedding`` / ``column_mean`` / ``token_mean`` / ``table_embedding``).

    ``embeddings_path`` is expected to point at a table-embedding pickle (the
    Stage-1+2 output for ``model``). The runner reads from
    ``<emb_root>/table/<model>/{dlte_v1_queries,dlte_v1_targets,ckan_subset}.pkl``
    so we derive ``--embeddings_root`` from ``embeddings_path``'s grandparent.
    The DLTE ground-truth files (``query_tasks.jsonl``, ``lake_manifest.jsonl``)
    must live under ``--project_root/datasets/dlte_v1/``; we derive
    ``--project_root`` from ``labels_path``'s parent (the staged DLTE root).
    """
    # Derive --embeddings_root from embeddings_path:
    #   embeddings_path = <emb_root>/table/<model>/dlte_v1_queries.pkl
    #   -> <emb_root> = embeddings_path.parent.parent.parent
    emb_root = embeddings_path.parent.parent.parent

    # Derive --project_root from labels_path:
    #   labels_path = <project_root>/datasets/dlte_v1/labels.json (manifest)
    #   -> <project_root> = labels_path.parent.parent.parent
    project_root = labels_path.parent.parent.parent

    output_root = _resolve_dlte_output_root(output_dir)

    # The setting axis carries the table-embedding variant. The diagonal
    # default uses the runner's own default (column_mean for column models,
    # table_embedding for native table models like TAPEX — encoded in the
    # runner's NATIVE_TABLE_MODELS dict, so we don't need to mirror it here).
    extra_args: list[str] = []
    if setting and setting != _DLTE_DIAGONAL_SETTING:
        # Accept any of the canonical variants; the runner validates.
        extra_args += ["--table_variant", setting]

    cmd: list[str] = [
        sys.executable, "-m", cfg.runner,
        "--models", model,
        "--output_root", str(output_root),
        "--project_root", str(project_root),
        "--embeddings_root", str(emb_root),
        *extra_args,
    ]
    return [cmd]


def _build_dlte_alignment_cmd(
    *, cfg: ProbeConfig, model: str, task: str, dataset: str, setting: str,
    probe: str, seed: int, embeddings_path: Path, labels_path: Path,
    output_dir: Path, configs_root: Path,
) -> list[list[str]]:
    """Build the DLTE Stage-2 (column alignment) command.

    Mirrors the canonical invocation::

        python -m trl_bench.tasks.dlte.scripts.step9_column_alignment \\
            --models <col_model> \\
            --table_model <table_model>   [optional]
            --topk 100 \\
            --output_root <out>/dlte \\
            --project_root <project_root> \\
            --embeddings_root <emb_root>

    ``model`` IS the col_model. ``setting`` may carry the table_model (the
    Stage-1 upstream key); defaults to diagonal (table_model = col_model so
    the Stage-2 directory is named ``<col_model>``, not ``<table>__<col>``).
    """
    tokens = _parse_dlte_setting(setting, expect=1)
    table_model = tokens[0] if tokens else model   # diagonal default

    # Derive --embeddings_root from embeddings_path:
    #   embeddings_path = <emb_root>/column/<model>/<dataset>.pkl
    #   -> <emb_root> = embeddings_path.parent.parent.parent
    emb_root = embeddings_path.parent.parent.parent
    project_root = labels_path.parent.parent.parent
    output_root = _resolve_dlte_output_root(output_dir)

    cmd: list[str] = [
        sys.executable, "-m", cfg.runner,
        "--models", model,
        "--topk", "100",
        "--output_root", str(output_root),
        "--project_root", str(project_root),
        "--embeddings_root", str(emb_root),
    ]
    # Only pass --table_model when it differs from --models. The runner's
    # ``derive_stage2_key`` produces ``<table>__<col>`` only in that case;
    # otherwise the Stage-2 directory key is just ``<col_model>``.
    if table_model != model:
        cmd += ["--table_model", table_model]
    return [cmd]


def _build_dlte_merge_cmd(
    *, cfg: ProbeConfig, model: str, task: str, dataset: str, setting: str,
    probe: str, seed: int, embeddings_path: Path, labels_path: Path,
    output_dir: Path, configs_root: Path,
) -> list[list[str]]:
    """Build the DLTE Stage-3 (row matching + merge + evaluation) command.

    Mirrors the canonical invocation::

        python -m trl_bench.tasks.dlte.scripts.step10_row_matching \\
            --col_models <col> --row_models <row> \\
            --table_model <table>     [optional, only if non-diagonal]
            --output_root <out>/dlte \\
            --project_root <project_root> \\
            --embeddings_root <emb_root>

    ``model`` IS the row_model. ``setting`` may carry
    ``"<col_model>__<table_model>"``; defaults to diagonal (model fills every
    axis). step10 produces the merge log + (when ``--skip-evaluation`` is not
    passed) the per-cell ``end_to_end.json`` + ``summary.csv`` under
    ``metrics/<col>__<row>__<table>/``.
    """
    tokens = _parse_dlte_setting(setting, expect=2)
    if tokens:
        col_model, table_model = tokens
    else:
        col_model, table_model = model, model     # diagonal default

    emb_root = embeddings_path.parent.parent.parent
    project_root = labels_path.parent.parent.parent
    output_root = _resolve_dlte_output_root(output_dir)

    cmd: list[str] = [
        sys.executable, "-m", cfg.runner,
        "--col_models", col_model,
        "--row_models", model,
        "--output_root", str(output_root),
        "--project_root", str(project_root),
        "--embeddings_root", str(emb_root),
    ]
    # ``derive_stage2_key`` requires --table_model only when it differs from
    # --col_models.
    if table_model != col_model:
        cmd += ["--table_model", table_model]
    return [cmd]


# == Deterministic training-free task command builders ========================
# These builders mirror the canonical .sbatch invocations preserved in the
# working repo at ``results/round{4,5}/downstream/<task>/<cell>.sbatch``.
# The runners are training-free (no probe head, no seed), so ``probe`` is
# accepted but ignored — the dispatcher does NOT pass ``--head_type``.


def _build_column_clustering_cmd(
    *, cfg: ProbeConfig, model: str, task: str, dataset: str, setting: str,
    probe: str, seed: int, embeddings_path: Path, labels_path: Path,
    output_dir: Path, configs_root: Path,
) -> list[list[str]]:
    """Build the column_clustering Stage-3 command.

    Mirrors the canonical .sbatch::

        python -m trl_bench.tasks.column_clustering.evaluate_clustering \\
            --embeddings <col_pkl> --dataset <ds_dir> \\
            --k 20 --target_avg_size 50 --batch_size 4096

    The runner's ``--dataset`` is a DIRECTORY name (it loads
    ``<dataset>/all.csv``); by convention we take the parent of
    ``labels_path`` as that directory (Stage-0 staging materializes
    ``all.csv`` alongside ``labels.json``). The runner prints all metrics to
    stdout; ``run.py`` captures it to ``stage_run.log``.
    """
    dataset_dir = labels_path.parent
    cmd: list[str] = [
        sys.executable, "-m", cfg.runner,
        "--embeddings",      str(embeddings_path),
        "--dataset",         str(dataset_dir),
        "--k",               str(cfg.max_epochs),      # 20
        "--target_avg_size", str(cfg.hidden_dim),      # 50
        "--batch_size",      str(cfg.batch_size),      # 4096
    ]
    return [cmd]


def _build_schema_matching_cmd(
    *, cfg: ProbeConfig, model: str, task: str, dataset: str, setting: str,
    probe: str, seed: int, embeddings_path: Path, labels_path: Path,
    output_dir: Path, configs_root: Path,
) -> list[list[str]]:
    """Build the schema_matching Stage-3 command.

    Mirrors the canonical .sbatch::

        python -m trl_bench.tasks.schema_matching.run_schema_matching \\
            --embeddings <col_pkl> --pairs <pairs.json> \\
            --ground_truth <gt.csv> --tables_dir <tables/> \\
            --output_dir <out> --matching_strategy hungarian --threshold 0.0

    By convention the dispatcher derives ``--pairs``, ``--ground_truth``, and
    ``--tables_dir`` from ``labels_path``'s parent (the staged Valentine
    directory): pairs.json + ground_truth.csv + tables/. The runner writes
    ``<output_dir>/results.json`` natively; the envelope wrapper reads it.
    """
    ds_dir = labels_path.parent
    pairs_path        = ds_dir / "pairs.json"
    ground_truth_path = ds_dir / "ground_truth.csv"
    tables_dir        = ds_dir / "tables"

    cmd: list[str] = [
        sys.executable, "-m", cfg.runner,
        "--embeddings",        str(embeddings_path),
        "--pairs",             str(pairs_path),
        "--ground_truth",      str(ground_truth_path),
        "--tables_dir",        str(tables_dir),
        "--output_dir",        str(output_dir),
        "--matching_strategy", cfg.combination_method,        # "hungarian"
        "--threshold",         str(cfg.learning_rate),        # 0.0
    ]
    return [cmd]


# Per-dataset union_search search config.
#
# Most union_search benchmarks (santos, ugen_v1, ugen_v2) use the
# linear/K=10/threshold=0.7/ef=100/N=100 default carried by the ProbeConfig.
# The two TUS benchmarks are large and low-similarity, so the canonical paper
# runs them with an HNSW config and a much lower acceptance threshold. This is
# not recorded in any yaml — it lived only in the generated .sbatch scripts
# and is preserved in the reference envelopes' ``hyperparameters`` block
# (method=hnsw, k=60, threshold=0.1, ef=100, N=500). tus / tus_hard are large
# + low-similarity, so the linear/K=10 default has a low recall ceiling
# (~0.06 = 10 / ~181 relevant-per-query); they need the HNSW operating point.
_UNION_SEARCH_DATASET_OVERRIDES: dict[str, dict[str, object]] = {
    "tus":      {"method": "hnsw", "K": 60, "threshold": 0.1, "ef": 100, "N": 500},
    "tus_hard": {"method": "hnsw", "K": 60, "threshold": 0.1, "ef": 100, "N": 500},
}


def union_search_params(dataset: str, cfg: ProbeConfig) -> dict:
    """Resolve the per-dataset union_search search config.

    Returns a dict with keys ``method`` / ``K`` / ``threshold`` / ``ef`` / ``N``.
    Defaults come from the ProbeConfig (the re-purposed fields carry the
    reference hyperparameters); ``_UNION_SEARCH_DATASET_OVERRIDES`` supplies the
    per-dataset deviations (currently tus / tus_hard -> HNSW). Single source of
    truth shared by the CLI builder (below) and the output-envelope metadata in
    ``run.py`` so the recorded ``hyperparameters`` block always matches the run.
    """
    params: dict[str, object] = {
        "method":    cfg.combination_method,   # "linear"
        "K":         cfg.num_labels,           # 10
        "threshold": cfg.learning_rate,        # 0.7
        "ef":        cfg.batch_size,           # 100
        "N":         cfg.max_epochs,           # 100
    }
    params.update(_UNION_SEARCH_DATASET_OVERRIDES.get(dataset, {}))
    return params


def _build_union_search_cmd(
    *, cfg: ProbeConfig, model: str, task: str, dataset: str, setting: str,
    probe: str, seed: int, embeddings_path: Path, labels_path: Path,
    output_dir: Path, configs_root: Path,
) -> list[list[str]]:
    """Build the union_search Stage-3 command.

    Mirrors the canonical .sbatch::

        python -m trl_bench.tasks.union_search.run_search \\
            --query_embeddings <col_pkl> --datalake_embeddings <col_pkl> \\
            --groundtruth <gt.pickle> \\
            --method linear --K 10 --threshold 0.7 --ef 100 --N 100

    The search config is per-dataset (see ``union_search_params`` /
    ``_UNION_SEARCH_DATASET_OVERRIDES``): santos/ugen use the linear/K=10
    default above; tus/tus_hard use ``--method hnsw --K 60 --threshold 0.1
    --ef 100 --N 500``.

    Both ``--query_embeddings`` and ``--datalake_embeddings`` point at the
    same column pickle, because query tables are a subset of the datalake;
    the runner's compute_metrics filters by groundtruth keys.

    By convention the dispatcher derives ``--groundtruth`` from
    ``labels_path``'s parent: ``<dataset_root>/groundtruth.pickle`` (the
    Stage-0 stagers normalize the variety of source-benchmark groundtruth
    filenames — santosUnionBenchmark.pickle, ugen_v1Benchmark.pickle, ...
    to that single filename inside the staged dataset directory).
    """
    ds_dir = labels_path.parent
    groundtruth_path = ds_dir / "groundtruth.pickle"

    p = union_search_params(dataset, cfg)
    cmd: list[str] = [
        sys.executable, "-m", cfg.runner,
        "--query_embeddings",    str(embeddings_path),
        "--datalake_embeddings", str(embeddings_path),
        "--groundtruth",         str(groundtruth_path),
        "--method",              str(p["method"]),
        "--K",                   str(p["K"]),
        "--threshold",           str(p["threshold"]),
        "--ef",                  str(p["ef"]),
        "--N",                   str(p["N"]),
    ]
    return [cmd]


def _build_join_search_cmd(
    *, cfg: ProbeConfig, model: str, task: str, dataset: str, setting: str,
    probe: str, seed: int, embeddings_path: Path, labels_path: Path,
    output_dir: Path, configs_root: Path,
) -> list[list[str]]:
    """Build the join_search Stage-3 command.

    Mirrors the canonical .sbatch::

        python -m trl_bench.tasks.join_search.run_search_and_evaluate \\
            --query_emb <col_pkl> --datalake_emb <col_pkl> \\
            --query_list <queries.csv> --ground_truth <gt.csv> \\
            --k 50 --output <out>/results.csv

    By convention the dispatcher derives ``--query_list`` and
    ``--ground_truth`` from ``labels_path``'s parent: ``<ds>/queries.csv``
    + ``<ds>/ground_truth.csv``. The Stage-0 stager is responsible for
    normalizing the opendata variants (opendata_join_query.csv etc.) to
    those names. The runner prints metrics to stdout (the CSV at
    ``--output`` is per-pair retrieval, not aggregate); ``run.py`` captures
    stdout to ``stage_run.log``.
    """
    ds_dir = labels_path.parent
    query_list_path   = ds_dir / "queries.csv"
    ground_truth_path = ds_dir / "ground_truth.csv"
    output_csv        = output_dir / "results.csv"

    cmd: list[str] = [
        sys.executable, "-m", cfg.runner,
        "--query_emb",    str(embeddings_path),
        "--datalake_emb", str(embeddings_path),
        "--query_list",   str(query_list_path),
        "--ground_truth", str(ground_truth_path),
        "--k",            str(cfg.max_epochs),     # 50
        "--output",       str(output_csv),
    ]
    return [cmd]


def _build_join_search_learned_cmd(
    *, cfg: ProbeConfig, model: str, task: str, dataset: str, setting: str,
    probe: str, seed: int, embeddings_path: Path, labels_path: Path,
    output_dir: Path, configs_root: Path,
) -> list[list[str]]:
    """Build the join_search_learned Stage-3 command (learned InfoNCE projection).

    Mirrors the canonical .sbatch::

        python -m trl_bench.tasks.join_search.run_learned_search \\
            --query_emb <col_pkl> --datalake_emb <col_pkl> \\
            --query_list <queries.csv> --ground_truth <ground_truth.csv> \\
            --split_dir <splits/join_search> --output_dir <out> \\
            --num_layers 1 --batch_size 512 --max_epochs 10 \\
            --learning_rate 1e-3 --dropout 0.1 --k 50 --seed <S>

    Both ``--query_emb`` and ``--datalake_emb`` point at the same column pickle
    (query tables are a subset of the datalake). ``--query_list`` /
    ``--ground_truth`` / ``--split_dir`` are derived from ``labels_path``'s
    parent (the staged dataset dir); the split dir holds the canonical
    train/test query partition (train_queries.csv, test_queries.csv, test_gt.csv).
    """
    ds_dir = labels_path.parent
    cmd: list[str] = [
        sys.executable, "-m", cfg.runner,
        "--query_emb",     str(embeddings_path),
        "--datalake_emb",  str(embeddings_path),
        "--query_list",    str(ds_dir / "queries.csv"),
        "--ground_truth",  str(ds_dir / "ground_truth.csv"),
        "--split_dir",     str(ds_dir / "splits" / "join_search"),
        "--output_dir",    str(output_dir),
        "--num_layers",    str(cfg.num_labels),       # 1
        "--batch_size",    str(cfg.batch_size),       # 512
        "--max_epochs",    str(cfg.max_epochs),       # 10
        "--learning_rate", str(cfg.learning_rate),    # 1e-3
        "--dropout",       str(cfg.dropout_prob),     # 0.1
        "--k",             "50",
        "--seed",          str(seed),
    ]
    return [cmd]


_FAMILY_BUILDERS = {
    "pair_task":         _build_pair_task_cmd,
    "cta":               _build_cta_cmd,
    "cra":               _build_cra_cmd,
    "retrieval":         _build_retrieval_cmd,
    "row_prediction":    _build_row_prediction_cmd,
    "semparse":          _build_semparse_cmd,
    "dlte_retrieval":    _build_dlte_retrieval_cmd,
    "dlte_alignment":    _build_dlte_alignment_cmd,
    "dlte_merge":        _build_dlte_merge_cmd,
    "column_clustering": _build_column_clustering_cmd,
    "schema_matching":   _build_schema_matching_cmd,
    "union_search":      _build_union_search_cmd,
    "join_search":       _build_join_search_cmd,
    "join_search_learned": _build_join_search_learned_cmd,
}


def _probe_command(
    *, model: str, task: str, dataset: str, setting: str, probe: str, seed: int,
    embeddings_path: Path, labels_path: Path, results_dir: Path,
    configs_root: Path,
) -> list[list[str]]:
    """Build the Stage-3 probe command(s) for one (model, task, dataset, ...) cell.

    Returns a list of stages (most families return one stage; table_retrieval
    returns two: train then evaluate). Dispatches on ``ProbeConfig.family``.
    """
    if task not in _TASK_PROBE_CONFIG:
        raise SettingError(
            f"probe-task config for {task!r} is not supported in this release. "
            f"See docs/USAGE.md for status."
        )
    cfg = _TASK_PROBE_CONFIG[task]
    output_dir = _resolve_output_dir(
        task=task, model=model, dataset=dataset, setting=setting,
        probe=probe, seed=seed, path_layout=cfg.path_layout,
        results_dir=results_dir,
    )

    builder = _FAMILY_BUILDERS.get(cfg.family)
    if builder is None:
        raise SettingError(
            f"unknown probe-task family {cfg.family!r} for task {task!r}. "
            f"Wired families: {sorted(_FAMILY_BUILDERS)}."
        )
    return builder(
        cfg=cfg, model=model, task=task, dataset=dataset, setting=setting,
        probe=probe, seed=seed,
        embeddings_path=embeddings_path, labels_path=labels_path,
        output_dir=output_dir, configs_root=configs_root,
    )


def build_command(
    *, model: str, task: str, dataset: str, setting: str,
    probe: str | None, seed: int,
    results_dir: str | Path,
    embeddings_path: str | Path | None = None,
    labels_path: str | Path | None = None,
    configs_root: str | Path = "configs/downstream",
    embeddings_dir: str | Path = "./embeddings",
    data_root: str | Path = "./data",
) -> list[list[str]]:
    """Build the subprocess command sequence for one (model, task, dataset, ...) cell.

    Returns a list of stages, each a list of args for ``subprocess.run``. The
    caller (``trl_bench.run.main``) runs them sequentially, treating any
    non-zero exit as a job failure.

    For PROBE_TASKS, callers must provide ``embeddings_path`` (a
    pre-extracted table-level embedding pickle) and ``labels_path``. Stage-1
    (per-model column extraction) and Stage-2 (table aggregation) wiring is
    tracked in docs/USAGE.md.
    """
    if not is_valid_cell(model, task):
        raise SettingError(
            f"({model}, {task}) is not supported: model exports "
            f"{_MODEL_GRANULARITIES.get(model)} but task requires "
            f"{_TASK_GRANULARITIES.get(task)}"
        )

    results_dir = Path(results_dir)
    configs_root = Path(configs_root)

    # Tasks without a probe-head axis (their envelopes omit ``head_type``):
    #   * semantic_parsing — MAPO decoder is the head, no head_type CLI knob.
    #   * dlte_{retrieval,alignment,merge} — pipeline runners with no head
    #     dimension; the per-cell axis is encoded in (model, setting).
    #   * deterministic tasks (column_clustering / schema_matching /
    #     union_search / join_search) — training-free runners with no probe
    #     head at all; ``--probe`` is accepted but ignored.
    _PROBE_EXEMPT_TASKS = (
        "semantic_parsing",
        "dlte_retrieval", "dlte_alignment", "dlte_merge",
        # join_search_learned: the learned projection IS the model; no head_type.
        "join_search_learned",
    )

    if task in PROBE_TASKS or task in DETERMINISTIC_TASKS:
        if probe is None and task not in _PROBE_EXEMPT_TASKS \
                and task not in DETERMINISTIC_TASKS:
            raise SettingError(f"task {task!r} requires --probe (linear|mlp|...)")
        if embeddings_path is None or labels_path is None:
            raise SettingError(
                f"task {task!r} could not auto-resolve embeddings/labels; "
                f"pass --embeddings-path and --labels-path explicitly "
                f"(see docs/USAGE.md)."
            )
        stages = _probe_command(
            model=model, task=task, dataset=dataset, setting=setting,
            probe=probe, seed=seed,
            embeddings_path=Path(embeddings_path),
            labels_path=Path(labels_path),
            results_dir=results_dir, configs_root=configs_root,
        )
        return stages

    # No remaining unported categories: every task in _TASK_GRANULARITIES is
    # now either a PROBE_TASK (above) or a DETERMINISTIC_TASK (also above).
    raise SettingError(
        f"task {task!r} dispatch is not supported in this release. "
        f"See docs/USAGE.md."
    )


__all__ = [
    "DLTE_TASKS", "DETERMINISTIC_TASKS", "TABLE_EMBEDDING_TASKS",
    "ROW_EMBEDDING_TASKS", "PROBE_TASKS", "COSINE_THRESHOLD_TASKS",
    "INTERACTION_TASKS",
    "SettingError", "is_valid_cell", "list_cells", "build_command",
    "ProbeConfig", "ExtractorConfig", "build_extractor_command",
    "QueryEncoderConfig", "build_query_extractor_command",
    "TableEncoderConfig", "build_table_encoder_command",
]
