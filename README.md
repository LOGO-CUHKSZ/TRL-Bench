# TRL-Bench

Official code release for **TRL-Bench: Standardizing Cross-Paradigm Representation-Level Evaluation of Tabular Encoders**.

**Paper:** [arXiv:2606.09323](https://arxiv.org/abs/2606.09323)

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

Setup, dependency notes, per-cell flags, and how to run the full grid live in [`docs/USAGE.md`](docs/USAGE.md).

> **Citation**
>
> ```bibtex
> @article{pang2026trl,
>   title={TRL-Bench: Standardizing Cross-Paradigm Representation-Level Evaluation of Tabular Encoders},
>   author={Pang, Wei and Jian, Xiangru and Li, Hehan and Yu, Zhixuan and Xue, Alex and Li, Jinyang and Dong, Zhengyuan and Zhao, Xinjian and Xu, Hao and Zhang, Chao and Cheng, Reynold and {\"O}zsu, M. Tamer and Yu, Tianshu},
>   journal={arXiv preprint arXiv:2606.09323},
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
(including `tabpfn`) is reproducible from pip + HuggingFace/GCS alone.*

| Model | Env var | Where to get it |
|---|---|---|
| `openai` (Table 30 ablation) | `OPENAI_API_KEY` | <https://platform.openai.com/api-keys> |

`export OPENAI_API_KEY=...` before invoking `python -m trl_bench.run` for that cell.

> **TabPFN note.** The `[tabpfn]` extra pins `tabpfn==6.4.1`, which downloads
> its weights from a public Google Cloud bucket with no credentials. Newer
> tabpfn (>=8) added a Prior Labs *license gate* that raises
> `TabPFNLicenseError` and demands a `TABPFN_TOKEN` in non-interactive runs —
> the pin keeps the benchmark token-free and matches the paper-time weights.

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

For models with Stage-1 wired (`bert`, `gte`), the runner auto-stages the HF dataset (Stage 0), auto-extracts column embeddings (Stage 1), auto-aggregates to a table-level pickle (Stage 2), and runs the probe (Stage 3). Models whose Stage-1 is not wired in the registry's `_MODEL_EXTRACTORS` table still require manual pre-extraction; see each `src/trl_bench/models/<m>/USAGE.md`.

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

21 benchmark models, plus 2 query encoders (MPNet, Sentence-T5) and the OpenAI API ablation — 24 wrappers total, under `src/trl_bench/models/<model>/`. Install only the extras you need:

| Model | Granularity | Checkpoint source | Install |
|---|---|---|---|
| BERT | col / row / table | HF Hub | `[bert]` |
| GTE  | col / row / table | HF Hub | `[gte]` |
| MPNet | col / table | HF Hub (default query encoder) | `[gte]` |
| Sentence-T5 | col / table | HF Hub (QE ablation) | `[gte]` |
| TAPAS | col / table | HF Hub | `[tapas]` |
| TAPEX | col / table | HF Hub | `[tapex]` |
| TaBERT | col / table | upstream URL (CC BY-NC-4.0) | `[tabert]` |
| TURL | col / table | logo-lab mirror (Apache-2.0) | `[turl]` |
| TUTA | col / row / table | logo-lab mirror (MIT) | `[tuta]` |
| TABBIE | col / row / table | upstream source | `[tabbie]` |
| TabSketchFM | col / table | upstream URL (CC BY-NC-ND-4.0) | `[tabsketchfm]` |
| Starmie | col | retrain via `models/starmie/run_pretrain.py` | `[starmie]` |
| OpenAI | col / table | API (Table 30 ablation) | `[openai]` |
| TabICL | row | PyPI auto-fetch | `[tabicl]` |
| TabPFN | row | PyPI auto-fetch | `[tabpfn]` |
| TransTab | row | trained per dataset | `[transtab]` |
| DAE | row | trained per dataset | `[dae]` |
| SCARF | row | trained per dataset | `[scarf]` |
| SwitchTab | row | trained per dataset | `[switchtab]` |
| VIME | row | trained per dataset | `[vime]` |
| SubTab | row | trained per dataset | `[subtab]` |
| SAINT | row | trained per dataset | `[saint]` |
| TabBinning | row | trained per dataset | `[tabular_binning]` |
| TabTransformer | row | trained per dataset | `[tabtransformer]` |

See [`docs/CHECKPOINT_LICENSES.md`](docs/CHECKPOINT_LICENSES.md) for the license audit of the 6 upstream-pretrained models, and `scripts/download_checkpoints.sh` to fetch them.

## Extending the benchmark

- Add a new **encoder**: [`docs/ADDING_A_MODEL.md`](docs/ADDING_A_MODEL.md) — wire the wrapper's CLI + declare granularities in the registry. Stage-1 dispatch is auto-orchestrated for all 24 of 24 wrappers today: 21 route through `_MODEL_EXTRACTORS` + `ExtractorConfig` (BERT, GTE, TAPAS, OpenAI, TabICL, TabPFN, TransTab, DAE, SCARF, SwitchTab, VIME, SubTab, SAINT, TabularBinning, TabTransformer, TaBERT, TabSketchFM, TURL, TUTA, TABBIE, Starmie) and 3 route through `_TABLE_ENCODERS` + `TableEncoderConfig` (mpnet, sentence_t5, tapex — table-direct one-pass extraction; Stage-2 skipped); add an entry to the appropriate dict to extend the auto-path to your encoder.
- Add a new **probe task**: [`docs/ADDING_A_TASK.md`](docs/ADDING_A_TASK.md) — write a `ProbeConfig` entry, drop in the YAML, anchor it against the canonical `.sbatch` from the working repo.

## License

Code: Apache-2.0 (see `LICENSE`). Datasets retain their upstream licenses listed above. Model checkpoints retain the licenses of their original authors; see `docs/CHECKPOINT_LICENSES.md`.

## Acknowledgements

This release ships wrappers and orchestration code authored by the TRL-Bench team. Pretrained model checkpoints are the work of their original authors; attribution is recorded per-model in `docs/CHECKPOINT_LICENSES.md`.
