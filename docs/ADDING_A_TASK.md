# Adding a New Probe Task to TRL-Bench

TRL-Bench's Stage-3 layer is a registry-driven subprocess dispatch: each probe task is one entry in `_TASK_PROBE_CONFIG` (`src/trl_bench/registry.py`). Adding a task means populating that entry, dropping the YAML the runner reads, and matching the paper's reported configuration for that task.

## Step 1: Mirror an existing task's config

Use the existing `_TASK_PROBE_CONFIG` entries in `src/trl_bench/registry.py` as templates — each encodes the exact CLI its runner expects (`python -m trl_bench.utils.downstream.run_task …` or a task-specific script). The keys you must encode in `ProbeConfig` are:

- `--task_name` (e.g. `join_spider_join`, `record_linkage_wdc_products_small`) → `task_name_template`
- `--task_type` (`classification` | `regression`)
- `--combination_method` (`concat` | `none` | other)
- `--hidden_dim`, `--num_labels`, `--batch_size`, `--max_epochs`, `--learning_rate`, `--dropout_prob`
- The YAML the runner reads (`--config …`) — copy it into `configs/downstream/<task>.yaml`
- The output directory shape — see Step 3

## Step 2: Add the `ProbeConfig` entry

Edit `_TASK_PROBE_CONFIG` in `src/trl_bench/registry.py`. The simplest template is `join_classification` — `task_name_template="join_{dataset}"`, `task_type="classification"`, `combination_method="concat"`, defaults for everything else.

For tasks whose labels.json key for the task name differs from the dataset slug, use the same template style: `task_name_template="containment_{dataset}"`, `task_name_template="cta_{dataset}"`. The dataset substring in `--task_name` must match the runner's internal task lookup; cross-check against the `.sbatch`.

## Step 3: Pick a `path_layout`

The release supports three layouts for Stage-3 output dirs (see `_probe_command` and `run.py::main`):

- `with_setting` (default) — `<task>/<model>/<setting>/seed<S>/<probe>/`. Used by `join_classification`, `union_classification`, `union_regression`, `table_subset`, … Pair-classification / pair-regression on table-level embeddings.
- `without_setting` — `<task>/<model>/seed<S>/<probe>/`. Used by `join_containment`, which reads the *column* pickle directly with `--embedding_type column` and `batch_size 2048`, so there is no setting choice to encode.
- `with_dataset` — `<task>/<model>/<dataset>/seed<S>/<probe>/`. Used by `record_linkage`, which runs `trl_bench.tasks.record_linkage.run_record_linkage` (its own script, not `run_task.py`) with `pass_task_type=False`, `pass_embedding_type=False` because row-level tasks have no aggregation choice.

The envelope hyperparameter dict ordering — `task_type` first, then `embedding_type` inserted right after, then the rest — is implemented in `run.py::main` and determines the result envelope's key ordering. Tasks with `pass_task_type=False` (record_linkage) drop the `task_type` / `embedding_type` keys entirely.

## Step 4: Drop the YAML

Add the task's `configs/downstream/<task>.yaml` (the config its runner loads via `--config`) to the release. `run_task.py` loads it; CLI flags from `_probe_command` override its values. The CLI flags are authoritative — YAML values are only fallbacks for keys the registry does not pass.

## Step 5: Test the dispatch

Add a unit test in `tests/test_registry.py` that asserts on the *content* of the command (not just shape). Mirror the existing `test_build_command_probe_task_dispatch_join_classification`:

```python
def test_build_command_my_task_dispatch(tmp_path):
    embeddings = tmp_path / "model_dataset.pkl"; embeddings.write_bytes(b"")
    labels = tmp_path / "labels.json"; labels.write_text("{}")
    stages = build_command(
        model="bert", task="my_task", dataset="my_dataset",
        setting="cls_embedding", probe="linear", seed=42,
        results_dir=tmp_path,
        embeddings_path=embeddings, labels_path=labels,
        configs_root=tmp_path / "configs",
    )
    args = dict(zip(stages[0][3::2], stages[0][4::2]))
    assert args["--task_name"] == "my_task_my_dataset"
    # ... assert on the rest of the CLI surface ...
```

For end-to-end coverage, run one (model, task, dataset, setting, probe, seed) cell via `trl-bench-run` and confirm it produces a result envelope at the documented path.

## Reference entries

- `join_classification` — simplest pair-classification template, `with_setting` layout, `--embedding_type` derived from `--setting`.
- `join_containment` — pair-regression with `without_setting` layout and column-pickle input (`batch_size 2048`, `--embedding_type column`).
- `record_linkage` — different runner script (`trl_bench.tasks.record_linkage.run_record_linkage`), `with_dataset` layout, no `--task_type` / `--embedding_type`. Documents what a non-`run_task.py` task entry looks like.

If your task has no `.sbatch` reference in the working repo, write its runner first.
