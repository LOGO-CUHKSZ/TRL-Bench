# TRL-Bench

Official code release for **TRL-Bench: Standardizing Cross-Paradigm Representation-Level Evaluation of Tabular Encoders**.

**Paper:** NeurIPS 2026, Track on Evaluations and Datasets · [arXiv:2606.09323](https://arxiv.org/abs/2606.09323)

TRL-Bench standardizes cross-paradigm representation-level evaluation of tabular encoders. Each encoder exports row-, column-, or table-embeddings through its supported wrapper, and shared lightweight heads probe them across **16 tasks** grouped into three suites: **TRL-CTbench** (column / table), **TRL-Rbench** (row), and **TRL-DLTE** (compositional Data-Lake Table Enrichment).

## TL;DR — run a cell

One command runs a (model, task, dataset) cell end-to-end (HF dataset → on-disk staging → extraction → table aggregation → probe → JSON envelope):

```bash
pip install -e .[bert]
trl-bench-run \
    --model bert --task join_classification --dataset spider_join \
    --setting cls_embedding --probe linear --seed 42 \
    --data-root ./data --embeddings-dir ./embeddings --results-dir ./results
```

The [per-suite walkthroughs](#per-suite-walkthroughs) below show how to run each suite and where its results land. Setup and dependency notes live in [`docs/USAGE.md`](docs/USAGE.md).

> **Citation**
>
> ```bibtex
> @inproceedings{pang2026trl,
>   title={TRL-Bench: Standardizing Cross-Paradigm Representation-Level Evaluation of Tabular Encoders},
>   author={Pang, Wei and Jian, Xiangru and Li, Hehan and Yu, Zhixuan and Xue, Alex and Li, Jinyang and Dong, Zhengyuan and Zhao, Xinjian and Xu, Hao and Zhang, Chao and Cheng, Reynold and {\"O}zsu, M. Tamer and Yu, Tianshu},
>   booktitle={40th Conference on Neural Information Processing Systems (NeurIPS 2026), Track on Evaluations and Datasets},
>   year={2026}
> }
> ```

## Install

```bash
git clone https://github.com/LOGO-CUHKSZ/TRL-Bench.git
cd TRL-Bench
pip install -e ".[bert]"             # base + BERT wrapper; add more extras per model family below
# pip install -e ".[bert,dev]"       # also install the test runner (pytest) for running tests/
```

Tested on Python 3.10 (the paper-time interpreter). Most extras install from
pip wheels on 3.10–3.12; the one exception is **`[tabert]`**, whose
`torch-scatter` dependency has no universal wheel and builds from source
against your installed torch. If `pip install -e ".[tabert]"` fails building
`torch-scatter`, install it with the matching find-links index *after* torch
is present, then retry the extra:

```bash
pip install -e ".[bert]"            # gets torch first
python - <<'PY'                     # discover your torch + CUDA build tag
import torch; print(f"torch-{torch.__version__.split('+')[0]}+{ 'cpu' if not torch.cuda.is_available() else 'cu'+torch.version.cuda.replace('.','') }")
PY
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu122.html   # substitute your tag
pip install -e ".[tabert]" --no-build-isolation
```

### Models that require external auth (not just pip)

Only the OpenAI ablation needs a credential beyond pip; *every other model
(including `tabpfn`) runs without credentials* (TABBIE's weights are a manual
download, see [Checkpoints](#checkpoints)).

| Model | Env var | Where to get it |
|---|---|---|
| `openai` (embedding ablation, paper Appendix O) | `OPENAI_API_KEY` | <https://platform.openai.com/api-keys> |

`export OPENAI_API_KEY=...` before invoking `python -m trl_bench.run` for that cell.

### Versions used in the paper

The paper's TabPFN and TabICL numbers come from `tabpfn==6.4.1` with
`tabpfn-extensions==0.2.2` (TabPFN-2.5 weights) and `tabicl==0.1.4`
(`tabicl-classifier-v1.1-0506.ckpt`). The `[tabpfn]` extra pins these versions.
The `[tabicl]` extra does not, and newer `tabicl` releases load a different
model, so install it with:

```bash
pip install -e ".[tabicl]" "tabicl==0.1.4"
```

Newer `tabpfn` (>=8) adds a license gate that requires a `TABPFN_TOKEN`; 6.4.1
downloads its weights with no credentials.

## One-experiment quickstart

```bash
python -m trl_bench.run \
    --model bert \
    --task join_classification \
    --dataset spider_join \
    --setting cls_embedding \
    --probe linear \
    --seed 42
```

On a cell's first run the runner stages the HF dataset (Stage 0), extracts embeddings (Stage 1), aggregates them to a table-level pickle when the task needs one (Stage 2), and runs the probe (Stage 3); later cells reuse the staged data and embeddings. This works for all 24 wrappers; only the DLTE embeddings are built separately (see [TRL-DLTE](#trl-dlte)).

For training-free tasks (`column_clustering`, `schema_matching`, `union_search`, `join_search` in cosine mode), the `--probe` argument is ignored.

Result JSON lands at:
```
results/evaluation/join_classification/bert/cls_embedding/linear/bert_spider_join_seed42.json
```


## Datasets

| Suite | HuggingFace dataset | License |
|---|---|---|
| TRL-CTbench | [`logo-lab/trl-ctbench`](https://huggingface.co/datasets/logo-lab/trl-ctbench) | CC-BY-SA-4.0 |
| TRL-Rbench  | [`logo-lab/trl-rbench`](https://huggingface.co/datasets/logo-lab/trl-rbench) | CC-BY-4.0 |
| TRL-DLTE    | [`logo-lab/trl-dlte`](https://huggingface.co/datasets/logo-lab/trl-dlte) | CC-BY-SA-4.0 |

Loaders in `src/trl_bench/data/` pull these via `datasets.load_dataset`. No raw data is bundled in this repo.

## Model coverage

The paper's pool has **20 benchmark models** (Table 5 of the paper). Wrappers live under `src/trl_bench/models/<model>/`, and `--model` takes the directory name (e.g. `tabular_binning`). Install only the extras you need:

| Model | Granularity | Checkpoint source | Install |
|---|---|---|---|
| BERT | col / row / table | HF Hub | `[bert]` |
| GTE  | col / row / table | HF Hub | `[gte]` |
| TAPAS | col / table | HF Hub | `[tapas]` |
| TAPEX | table | HF Hub | `[tapex]` |
| TaBERT | col / table | logo-lab mirror (CC BY-NC 4.0) | `[tabert]` |
| TURL | col / table | logo-lab mirror (Apache-2.0) | `[turl]` |
| TUTA | row / table | logo-lab mirror (MIT) | `[tuta]` |
| TABBIE | col / row / table | upstream source (manual download) | `[tabbie]` |
| TabSketchFM | col / table | logo-lab mirror (our run of the upstream pretraining code; no upstream checkpoint) | `[tabsketchfm]` |
| Starmie | col / table | trained per dataset (on first run) | `[starmie]` |
| TabICL | row | package download (`tabicl==0.1.4`) | `[tabicl]` |
| TabPFN | row | package download (`tabpfn==6.4.1`) | `[tabpfn]` |
| TransTab | row | trained per dataset | `[transtab]` |
| DAE | row | trained per dataset | `[dae]` |
| SCARF | row | trained per dataset | `[scarf]` |
| VIME | row | trained per dataset | `[vime]` |
| SubTab | row | trained per dataset | `[subtab]` |
| SAINT | row | trained per dataset | `[saint]` |
| TabBinning | row | trained per dataset | `[tabular_binning]` |
| TabTransformer | row | trained per dataset | `[tabtransformer]` |

Four more wrappers are outside the pool: the MPNet and Sentence-T5 query encoders for Table Retrieval and Table QA (`[gte]`), the OpenAI embedding ablation of Appendix O (`[openai]`), and SwitchTab (`[switchtab]`).

### Checkpoints

Local checkpoints are read from `--checkpoint-root` (default `./checkpoints`). `bash scripts/download_checkpoints.sh` fetches TaBERT, TURL, and TUTA from the [`logo-lab/trl-bench-ckpts`](https://huggingface.co/logo-lab/trl-bench-ckpts) mirror. TabSketchFM and TABBIE take one step each:

```bash
# TabSketchFM: the paper's weights, on the same mirror
python -c "from huggingface_hub import snapshot_download; snapshot_download('logo-lab/trl-bench-ckpts', allow_patterns='tabsketchfm/*', local_dir='./checkpoints')"

# TABBIE: save mix.tar.gz from the upstream SFIG611/tabbie Google Drive folder
# (https://drive.google.com/drive/folders/1vAMv09j-VlWHKd5djiRGuC16yb-lhJO0) to ./checkpoints/tabbie/, then
python -m trl_bench.models.tabbie.probe_checkpoint --archive ./checkpoints/tabbie/mix.tar.gz
```

## Per-suite walkthroughs

Run commands from the repository root. Each run prints `wrote reference envelope: <path>` with its result file. The paper runs every probe cell with both heads (`--probe linear` and `--probe mlp`) and five seeds (42, 52, 62, 72, 82) and reports the mean; training-free tasks are deterministic and run once. `trl-bench-run` stages data on first use; to stage ahead (needed before passing `--labels-path`):

```bash
python -c "from trl_bench.data.stage import stage_dataset; print(stage_dataset(suite='ctbench', task='join_classification', dataset='spider_join', data_root='./data'))"
```

### TRL-CTbench

```bash
trl-bench-run --model bert --task column_type_prediction --dataset sato \
    --setting cls_embedding --probe linear --seed 42
```

This writes `results/evaluation/column_type_prediction/bert/linear/bert_sato_seed42.json`, whose `macro_f1` is the paper's F1. The full suite (`trl_bench.registry.is_valid_cell(model, task)` tells which model–task pairs exist; the search and Table QA tasks need `pip install -e ".[join_search,union_search,semantic_parsing]"`):

| Paper task (metric) | `--task` | `--dataset` | `--setting` | `--probe` | Result field |
|---|---|---|---|---|---|
| Column Type (F1) | `column_type_prediction` | `sato`, `sotab` | `cls_embedding`¹ | `linear`, `mlp` | `macro_f1` |
| Column Clustering (NMI) | `column_clustering` | `sato`, `sotab` | `cls_embedding`¹ | omit² | `nmi` |
| Column Relation (F1) | `column_relation_prediction` | `wikict_relation` | `cls_embedding`¹ | `linear`, `mlp` | `macro_f1` |
| Join Search (MAP) | `join_search_learned`³ | `opendata_main`, `opendata_can`, `opendata_usa`, `opendata_uk_sg` | `cls_embedding`¹ | omit | `col_map` |
| Column Overlap (nRMSE)† | `join_containment` | `wiki_containment` | `column` | `linear`, `mlp` | `test_results_r2` |
| Union Search (MAP) | `union_search` | `santos`, `ugen_v1`, `ugen_v2`, `tus`, `tus_hard` | `cls_embedding`¹ | omit² | `map_at_k` |
| Schema Matching (R@GT) | `schema_matching` | `valentine` | `cls_embedding`¹ | omit² | `recall_at_gt` |
| Table QA (accuracy) | `semantic_parsing` | `wiki_table_questions` | `sentence_t5`, `mpnet`⁴ | omit | `accuracy` |
| Join Classification (F1)† | `join_classification` | `spider_join` | aggregation⁵ | `linear`, `mlp` | `test_results_macro_f1` |
| Union Classification (F1)† | `union_classification` | `wiki_union` | aggregation⁵ | `linear`, `mlp` | `test_results_macro_f1` |
| Union Regression (nRMSE)† | `union_regression` | `ecb_union` | aggregation⁵ | `linear`, `mlp` | `test_results_r2` |
| Table Subset (F1) | `table_subset` | `ckan_subset` | aggregation⁵ | `linear`, `mlp` | `test_results_macro_f1` |
| Table Retrieval (MRR) | `table_retrieval` | `nq_tables` | aggregation⁵ | `mpnet_modelonly`, `sentence_t5_modelonly`⁶ | `MRR` |

1. Column-level runs: `--setting` is required but unused.
2. Training-free and deterministic: one run per model and dataset.
3. Stage each `opendata_*` dataset with `task='join_search'` first.
4. The question encoder; the paper averages the two.
5. `cls_embedding`, `column_mean`, or `token_mean`; the paper picks one per model on development data.
6. Model-only retrieval, the paper's main setting; the paper picks the query encoder per model on development data.

† The paper uses table-disjoint splits for these tasks: stage the dataset, then add `--labels-path <staged dataset dir>/labels_strict.json` (e.g. `./data/spider_join/spider-join/labels_strict.json`) and a separate `--results-dir`. For the regression rows, nRMSE = √(1 − R²) from `test_results_r2`.

### TRL-Rbench

```bash
pip install -e ".[tabicl]" "tabicl==0.1.4"
trl-bench-run --model tabicl --task row_prediction --dataset openml_3 \
    --setting class --probe linear --seed 42 --results-dir ./results/linear
trl-bench-run --model bert --task record_linkage --dataset deepmatcher_beer \
    --setting row_embedding --probe linear --seed 42
```

For row prediction, `--setting` is the target column, and the result path has no head component, so use one `--results-dir` per head. The result `results/linear/evaluation/row_prediction/tabicl/openml_3/class/tabicl_openml_3_seed42.json` holds `test_results.auroc` and `test_results.macro_f1` for classification targets and `test_results.r2` for regression targets (the paper reports nRMSE = √(1 − R²), summarized across targets by the shifted geometric mean). Record linkage writes `results/evaluation/record_linkage/bert/linear/bert_deepmatcher_beer_seed42.json`, whose `test_results_f1` is the paper's F1 on the match class.

The full suite covers the 50 row-prediction tables, each with the targets listed under `label_columns` in its staged `labels.json`, and the 16 record-linkage datasets:

```text
openml_<id> for id in 3 38 458 1063 1486 4534 6332 40668 40966 40978 44958 44967 44975 44984 44992
  46906 46907 46908 46910 46911 46912 46915 46916 46918 46919 46920 46922 46923 46927 46929 46930
  46932 46933 46934 46935 46937 46939 46940 46950 46952 46955 46956 46958 46960 46961 46963 46964
  46969 46979 46980
deepmatcher_{abt_buy,amazon_google,beer,dblp_acm,dblp_scholar,fodors_zagats,itunes_amazon,walmart_amazon}
deepmatcher_{dblp_acm,dblp_scholar,itunes_amazon,walmart_amazon}_dirty
wdc_products_{small,medium,large,xlarge}
```

### TRL-DLTE

A pipeline chains a table encoder (Stage 1 retrieval), a column encoder (Stage 2 alignment), and a row encoder (Stage 3 merge). Stage 1 needs FAISS, which no extra installs:

```bash
pip install faiss-cpu
python -c "from trl_bench.data.stage import stage_dataset; print(stage_dataset(suite='dlte', task='dlte_merge', dataset='dlte_v1', data_root='./data'))"
```

`trl-bench-run` does not extract DLTE embeddings. Each encoder needs `dlte_v1_queries.pkl` and `dlte_v1_targets.pkl` under `embeddings/{table,column,row}/<model>/`, plus the distractors' `ckan_subset.pkl` under `embeddings/{table,column}/<model>/`. For BERT:

```bash
# distractors: the CTbench Table Subset cell writes embeddings/{column,table}/bert/ckan_subset.pkl
trl-bench-run --model bert --task table_subset --dataset ckan_subset \
    --setting cls_embedding --probe linear --seed 42

# the 11,032 target fragments (the lake directory also holds the distractors)
python - <<'PY'
import json, pathlib
d = pathlib.Path("data/dlte_v1/datasets/dlte_v1")
ids = {json.loads(line)["table_id"] for line in open(d / "manifests/fragments_manifest.jsonl")}
names = sorted(p.name for p in (d / "lake/targets/tables").glob("*.csv") if p.stem in ids)
pathlib.Path("dlte_targets.txt").write_text("\n".join(names) + "\n")
PY

D=./data/dlte_v1/datasets/dlte_v1
mkdir -p ./embeddings/column/bert
python -m trl_bench.models.bert.generate_column_embeddings --input $D/queries/tables \
    --output ./embeddings/column/bert/dlte_v1_queries.pkl
python -m trl_bench.models.bert.generate_column_embeddings --input $D/lake/targets/tables \
    --table_list dlte_targets.txt --output ./embeddings/column/bert/dlte_v1_targets.pkl
python -m trl_bench.scripts.generate_table_embeddings --models bert --datasets dlte_v1_queries dlte_v1_targets
python -m trl_bench.models.bert.generate_row_embeddings --input_dir $D/queries/tables \
    --output_path ./embeddings/row/bert/dlte_v1_queries.pkl
python -m trl_bench.models.bert.generate_row_embeddings --input_dir $D/lake/targets/tables \
    --table_list dlte_targets.txt --output_path ./embeddings/row/bert/dlte_v1_targets.pkl
```

The other encoders' extraction settings are in `slurm/config/models.yaml` (column) and `slurm/config/row_models.yaml` (row). Then run one pipeline (BERT at all three stages) or the full grid:

```bash
for task in dlte_retrieval dlte_alignment dlte_merge; do
  trl-bench-run --model bert --task $task --dataset dlte_v1 --setting diagonal --seed 42
done
python -m trl_bench.tasks.dlte.run_all --embeddings-dir ./embeddings \
    --results-dir ./results/evaluation/dlte --data-dir ./data/dlte_v1     # --dry-run lists the jobs first
```

The pipeline's result is `results/evaluation/dlte/metrics/bert__bert/end_to_end.json`, where `splits.test.uj_h` is the paper's UJ-H and `splits.test.cell_f1` its Cell F1. `run_all` runs every table × column × row combination whose embeddings exist; the paper's grid is 10 × 8 × 14 = 1,120 pipelines.

## Extending the benchmark

- Add a new **encoder**: [`docs/ADDING_A_MODEL.md`](docs/ADDING_A_MODEL.md) — wire the wrapper's CLI + declare granularities in the registry. Stage-1 dispatch is auto-orchestrated for all 24 of 24 wrappers today: 21 route through `_MODEL_EXTRACTORS` + `ExtractorConfig` (BERT, GTE, TAPAS, OpenAI, TabICL, TabPFN, TransTab, DAE, SCARF, SwitchTab, VIME, SubTab, SAINT, TabularBinning, TabTransformer, TaBERT, TabSketchFM, TURL, TUTA, TABBIE, Starmie) and 3 route through `_TABLE_ENCODERS` + `TableEncoderConfig` (mpnet, sentence_t5, tapex — table-direct one-pass extraction; Stage-2 skipped); add an entry to the appropriate dict to extend the auto-path to your encoder.
- Add a new **probe task**: [`docs/ADDING_A_TASK.md`](docs/ADDING_A_TASK.md) — write a `ProbeConfig` entry in `src/trl_bench/registry.py` and add the task's YAML under `configs/downstream/`.

## License

Code: Apache-2.0 (see `LICENSE`). Datasets retain their upstream licenses listed above. Model checkpoints retain the licenses of their original authors; see `docs/CHECKPOINT_LICENSES.md`.

## Acknowledgements

This release ships wrappers and orchestration code authored by the TRL-Bench team. Pretrained model checkpoints are the work of their original authors, except the TabSketchFM weights, which come from our run of the upstream TabSketchFM pretraining code; attribution is recorded per-model in `docs/CHECKPOINT_LICENSES.md`.
