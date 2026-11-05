# Running TRL-Bench

This document describes how to install TRL-Bench, fetch the data + model checkpoints, and run the benchmark across the paper's **20 benchmark models** (24 wrappers in total: the 20 plus the MPNet and Sentence-T5 query encoders, the OpenAI embedding ablation, and SwitchTab, which is outside the paper's pool) and all tasks.

## Pipeline stages

`trl-bench-run` composes four stages end-to-end (HF → on-disk → extract → aggregate → probe → JSON envelope):

| Stage | What it does |
|---|---|
| 0 — Data staging | HF dataset → on-disk `<data_root>/<dataset>/<inner>/{tables_all/, labels.json}` |
| 1 — Column / table extraction | Per-model wrappers in `src/trl_bench/models/<m>/` (or the shared text-encoder runner) produce per-column or per-table embedding pickles. All 24 of 24 wrappers auto-dispatch: 21 via `_MODEL_EXTRACTORS` (column extractors) and 3 via `_TABLE_ENCODERS` (mpnet, sentence_t5, tapex — table-direct, Stage-2 skipped). |
| 2 — Table aggregation | `python -m trl_bench.scripts.generate_table_embeddings` produces the table-level pickle Stage 3 reads (cls / column-mean / token-mean). |
| 3 — Probe + envelope | `trl-bench-run` dispatches the probe and wraps the result in the flat-key result envelope. |

## Run one cell

```bash
trl-bench-run \
    --model bert --task join_classification --dataset spider_join \
    --setting cls_embedding --probe linear --seed 42 \
    --data-root      ./data \
    --embeddings-dir ./embeddings \
    --results-dir    ./results
```

With `--data-root` empty, the first run auto-stages the HF dataset (Stage 0), then auto-extracts column embeddings (Stage 1) and aggregates to the table level (Stage 2) before probing. To run Stage 1+2 manually — to inspect intermediate artifacts, or to pass pre-computed embeddings via `--embeddings-path` — use:

```bash
# Stage 1: column extraction (BERT-specific args; each wrapper varies).
python -m trl_bench.models.bert.generate_column_embeddings \
    --input  ./data/spider_join/spider-join/tables_all \
    --output ./embeddings/column/bert/spider_join.pkl \
    --device cuda

# Stage 2: aggregate columns to table-level pickle.
python -m trl_bench.scripts.generate_table_embeddings \
    --models bert --datasets spider_join \
    --column-embeddings-dir ./embeddings/column \
    --output-dir            ./embeddings/table

# Stage 3: probe + envelope (same trl-bench-run as above; auto-derives paths).
```

Result lands at `results/evaluation/join_classification/bert/cls_embedding/linear/bert_spider_join_seed42.json`.

## Per-cell flag notes

`trl-bench-run` accepts `--probe linear` or `--probe mlp` (also `cosine_threshold` and `dummy`) for supervised probe tasks, and the family builders in `src/trl_bench/registry.py` apply each task's defaults, including the per-dataset search settings (Union Search on `tus` / `tus_hard` uses HNSW with K = 60), so no cell needs extra flags. The paper reads every encoder out with both the linear and the MLP head: run each probe cell with both.

## Extending to other (model, task, dataset) cells

- **Other models on the same task**: add the model's column-extractor CLI invocation (each wrapper has its own args; see `src/trl_bench/models/<m>/USAGE.md`). Stages 2/3 work unchanged.
- **Other PROBE_TASKS**: wired in the registry for `join_classification`, `join_containment`, `union_classification`, `union_regression`, `column_type_prediction`, `column_relation_prediction`, `record_linkage`, `row_prediction`, `table_subset`. Add a new task via a `ProbeConfig` entry in `src/trl_bench/registry.py` — see [`docs/ADDING_A_TASK.md`](ADDING_A_TASK.md).
- **Other datasets in ctbench**: the Stage-0 stager auto-handles them once a `_CTBENCH_INNER_SUBDIR` entry is added if the inner subdir name diverges from `<dataset>`.
- **Other suites (rbench, dlte)**: a per-suite stager in `src/trl_bench/data/stage.py` must be added; the task→suite map in `src/trl_bench/run.py::_TASK_TO_SUITE` already maps each task to the right suite.

## Setup

### 1. Install

```bash
git clone https://github.com/LOGO-CUHKSZ/TRL-Bench.git
cd TRL-Bench
pip install -e .[all]          # everything — may hit dep conflicts, see "Dependency conflicts" below
# OR
pip install -e .[bert]         # one model family at a time — safer
```

### 2. Download checkpoints

```bash
bash scripts/download_checkpoints.sh
```

This:
- Pre-pulls HF-native models into `~/.cache/huggingface/` (avoids a thundering herd when parallel jobs all fetch BERT at once).
- Pulls licensed upstream weights from `logo-lab/trl-bench-ckpts`.
- Prints the manual steps for TABBIE (upstream download) and Starmie (trained per dataset).
- Verifies all binaries against `scripts/checksums.sha256`.

Per the license audit (`docs/CHECKPOINT_LICENSES.md`):
- TUTA (MIT), TURL (Apache-2.0) and TaBERT (CC BY-NC) are mirrored on `logo-lab/trl-bench-ckpts`.
- TabSketchFM has no upstream checkpoint; the weights the paper uses were trained by the TRL-Bench authors and are mirrored on `logo-lab/trl-bench-ckpts`.
- Starmie ships no upstream checkpoint — users retrain per-dataset via `src/trl_bench/models/starmie/run_pretrain.py`.
- TABBIE weights are obtained from the upstream source.

### 3. Run the benchmark

Follow the [per-suite walkthroughs in the README](../README.md#per-suite-walkthroughs): for every task they give the `--task`, `--dataset`, `--setting` and `--probe` values the paper uses, and where each result lands.

No cluster is required: `trl-bench-run` runs any single cell standalone on one GPU (tens of minutes), and you can loop it over cells in a shell script.

## Dependency conflicts (known)

The 24 wrappers have heterogeneous deps:

| Family | Pin / source | Conflicts |
|---|---|---|
| BERT/GTE/TAPAS/TAPEX/MPNet/Sentence-T5/OpenAI | `transformers>=4.30` | TURL needs `transformers<4.20` |
| TaBERT | `fairseq>=0.10` | fairseq pulls old `omegaconf` |
| TURL | `transformers>=4.0,<4.20` | conflicts with modern transformers in the same env |
| TabSketchFM, SAINT | `pytorch-lightning` | depends on torch version |
| TabICL, TabPFN | PyPI auto-fetch | tabicl pins old `numpy<1.25` |

**Recommendation:** create separate virtual environments per model family. Single-env `pip install -e .[all]` will hit conflicts.

## Hardware

- **GPU**: 1× A100 (40GB / 80GB) or 1× V100 (32GB recommended; 16GB OOMs on TURL/TaBERT for large tables).
- **CPU memory**: 32GB+ per job. TabSketchFM ETL needs ~64GB transient memory.
- **Storage**: ~150GB for HF dataset caches + embeddings + per-job result JSONs.
