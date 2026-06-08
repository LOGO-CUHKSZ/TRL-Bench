#!/usr/bin/env python3
"""Generate the col/table benchmark grid cell list (one line per cell).

Scope: the col + table granularity tasks × the 10 col/table
models, one dataset + setting + probe per (model, task). Single seed.
Row tasks (row_prediction 123 datasets, record_linkage) are generated
separately (much larger batch).

Output line: MODEL|TASK|DATASET|SETTING|PROBE
"""
from __future__ import annotations
import sys
sys.path.insert(0, "src")
from trl_bench.registry import is_valid_cell

# 10 col/table models reported in ct_table.csv
COL_MODELS = ["bert", "gte", "tabbie", "tapas", "tapex", "tabert",
              "turl", "tuta", "tabsketchfm", "starmie"]

# Per-model table/column-aggregation setting (from slurm check_results.py map).
# column_mean-only models vs cls_embedding models.
MODEL_SETTING = {
    "bert": "cls_embedding", "gte": "cls_embedding", "tabbie": "cls_embedding",
    "tapas": "cls_embedding", "tapex": "cls_embedding", "tuta": "cls_embedding",
    "tabsketchfm": "cls_embedding",
    "turl": "column_mean", "tabert": "column_mean", "starmie": "column_mean",
}

# ct_table tasks -> (canonical dataset, probe). starmie is union_search-only in
# practice; is_valid_cell still gates per granularity.
TASK_DATASET_PROBE = {
    "column_type_prediction":     ("sato",             "mlp"),
    "column_clustering":          ("sato",             "linear"),  # deterministic; probe ignored
    "column_relation_prediction": ("sotab",            "mlp"),
    "join_search":                ("opendata_can",     "linear"),  # cosine; probe ignored
    "join_containment":           ("wiki_containment", "mlp"),     # KNOWN GAP (col-name fidelity)
    "join_classification":        ("spider_join",      "linear"),
    "union_search":               ("santos",           "linear"),  # cosine; probe ignored
    "schema_matching":            ("valentine",        "linear"),  # deterministic; probe ignored
    "union_classification":       ("wiki_union",       "linear"),
    "union_regression":           ("ecb_union",        "linear"),
    "table_subset":               ("ckan_subset",      "linear"),
    "table_retrieval":            ("nq_tables",        "linear"),
    "semantic_parsing":           ("wiki_table_questions", "linear"),
}


def main() -> int:
    cells = []
    for model in COL_MODELS:
        setting = MODEL_SETTING[model]
        for task, (dataset, probe) in TASK_DATASET_PROBE.items():
            if not is_valid_cell(model, task):
                continue
            # semantic_parsing's setting axis IS the question encoder, which the
            # benchmark restricts to {sentence_t5, mpnet} (default sentence_t5);
            # the model under test supplies COLUMN embeddings, not questions -- so
            # every model is valid here, paired with sentence_t5 questions.
            s = "sentence_t5" if task == "semantic_parsing" else setting
            cells.append(f"{model}|{task}|{dataset}|{s}|{probe}")
    for c in cells:
        print(c)
    print(f"# total col/table cells: {len(cells)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
