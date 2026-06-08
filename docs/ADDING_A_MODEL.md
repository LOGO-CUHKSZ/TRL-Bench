# Adding a New Encoder to TRL-Bench

TRL-Bench evaluates encoders via subprocess dispatch over each wrapper's existing CLI — there's no uniform Python API to implement. To add a new encoder, you write the wrapper your way, then register it.

## Step 1: Add the wrapper directory

Under `src/trl_bench/models/<your_model>/`, ship whatever scripts you need for embedding extraction. The dispatcher (`src/trl_bench/registry.py`) looks for canonical script names. The first matching name wins:

| Granularity | Script names tried (in order) |
|---|---|
| col   | `generate_column_embeddings.py`, `csv_to_embeddings.py`, `extract_batch_embeddings.py`, `generate_table_embeddings_native.py`, `generate_text_embeddings.py` |
| row   | `generate_row_embeddings.py`, `generate_embeddings_train_test.py`, `generate_embeddings_single_file.py`, `generate_embeddings.py` |
| table | `generate_table_embeddings.py`, `generate_column_embeddings.py`, `extract_batch_embeddings.py`, `generate_text_embeddings.py` |

Each script must accept at minimum:
- `--input <hf_dataset_ref_or_path>` — input data spec; the dispatcher passes `<data_root>/<dataset>/tables/` after Task 14b staging materializes the HF dataset onto disk.
- `--output <path>` — where to write the extracted embeddings (pickle / npy / whatever your downstream task probe expects).
- `--seed <int>` — for any stochastic step in extraction.
- (col/table encoders) Transformer wrappers typically emit all three table-level aggregations (`cls_embedding | column_mean | token_mean`) in their output pickle; the aggregation is selected **downstream** via `--setting` at probe time — extractors do not take an `--aggregation` flag.

## Step 2: Declare model capabilities

Edit `src/trl_bench/registry.py` and add an entry to `_MODEL_GRANULARITIES`:

```python
_MODEL_GRANULARITIES["your_model"] = frozenset({"col", "row", "table"})  # subset of these
```

The registry will then yield `(your_model, task)` cells for any task whose required granularity is in your set.

## Step 3: Add an extras group to `pyproject.toml`

```toml
[project.optional-dependencies]
your_model = ["whatever-pypi-deps"]
```

Append `your_model` to the `all` extras group.

## Step 4: Smoke-test

```bash
pip install -e .[your_model]
python -m trl_bench.run \
    --model your_model \
    --task join_classification \
    --dataset spider_join \
    --setting cls_embedding \
    --probe linear \
    --seed 42
```

You should see `results/evaluation/join_classification/your_model/cls_embedding/linear/your_model_spider_join_seed42.json` appear.

## Step 5: Run the full grid

```bash
python slurm/generate_jobs.py --config configs/grid.yaml --models-config <(printf 'models:\n  your_model: {}\n')
bash slurm/submit.sh
```

## Reference: how BERT does it

`src/trl_bench/models/bert/` is the simplest example — it has `generate_column_embeddings.py` (with `--input`, `--output`, `--model`, `--device`) and `generate_row_embeddings.py`. No class API, no Python registration step — just scripts with conventional names and arguments.
