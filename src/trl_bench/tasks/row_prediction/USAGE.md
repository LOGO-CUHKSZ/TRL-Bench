# Row Prediction

## Overview

Downstream task for evaluating row-level embeddings on classification or regression tasks. Trains an MLP head (PyTorch Trainer) or linear probe (sklearn) on frozen embeddings.

This is a generic evaluation pipeline that works with any row-level embedding model.

## Embeddings Consumed

> **Embedding Level:** Row
> **Primary Embedding:** NumPy arrays (`.npy` files)
> **Pair Input:** No

| Embedding Type | Required | Shape | Description | Compatible Models |
|----------------|----------|-------|-------------|-------------------|
| `train_embeddings.npy` | **Yes** | `(N, dim)` | Training row embeddings | DAE, SCARF, SubTab, VIME, TabPFN, TabICL |
| `train_labels.npy` | **Yes** | `(N,)` | Training labels | - |
| `val_embeddings.npy` | No | `(V, dim)` | Validation row embeddings (canonical split) | Same as above |
| `val_labels.npy` | No | `(V,)` | Validation labels | - |
| `test_embeddings.npy` | **Yes** | `(M, dim)` | Test row embeddings | DAE, SCARF, SubTab, VIME, TabPFN, TabICL |
| `test_labels.npy` | **Yes** | `(M,)` | Test labels | - |

**Input format:** Directory containing NumPy files (v2 split-aware):
```
embedding_dir/
├── train_embeddings.npy       # shape: (num_train, embedding_dim)
├── train_labels_<col>.npy     # per-label column
├── val_embeddings.npy         # shape: (num_val, embedding_dim) [optional]
├── val_labels_<col>.npy       # per-label column [optional]
├── test_embeddings.npy        # shape: (num_test, embedding_dim)
├── test_labels_<col>.npy      # per-label column
└── metadata.json
```

When no val split is present (v1 data), a 10% ad-hoc val split is carved from training data.

## Task Configuration

| Property | Value |
|----------|-------|
| **Task Type** | Classification or Regression (from metadata or auto-detected) |
| **Embedding Level** | Row |
| **Pair Input** | No |
| **Config** | `configs/downstream/row_prediction.yaml` |

## Evaluation Metrics

### Classification
| Metric | Primary | Description |
|--------|---------|-------------|
| Accuracy | Yes | Test accuracy |
| Weighted F1 | No | Weighted F1 score |
| Macro F1 | No | Macro-averaged F1 score |

### Regression
| Metric | Primary | Description |
|--------|---------|-------------|
| R² | Yes | Coefficient of determination |
| MSE | No | Mean squared error |
| MAE | No | Mean absolute error |

## Input Data

**Embeddings directory:** `embeddings/row_prediction/<model>/<dataset>/`

Expected files:
```
embeddings/row_prediction/scarf/adult/
├── train_embeddings.npy
├── train_labels_income.npy
├── val_embeddings.npy          # optional
├── val_labels_income.npy       # optional
├── test_embeddings.npy
├── test_labels_income.npy
└── metadata.json
```

## Example Commands

### Basic Usage (Auto-detect task type)

```bash
python downstream_tasks/row_prediction/train_downstream.py \
    --embedding_dir embeddings/row_prediction/scarf/adult
```

### With Custom Config

```bash
python downstream_tasks/row_prediction/train_downstream.py \
    --embedding_dir embeddings/row_prediction/dae/openml_3 \
    --config configs/downstream/row_prediction.yaml
```

### Linear Probe

```bash
python downstream_tasks/row_prediction/train_downstream.py \
    --embedding_dir embeddings/row_prediction/bert/openml_1486 \
    --head_type linear --seed 42
```

### Force Classification Task

```bash
python downstream_tasks/row_prediction/train_downstream.py \
    --embedding_dir embeddings/row_prediction/vime/adult \
    --task classification
```

### Specific Label Column

```bash
python downstream_tasks/row_prediction/train_downstream.py \
    --embedding_dir embeddings/row_prediction/dae/openml_1063 \
    --label_column problems
```

## Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--embedding_dir` | No | embeddings | Directory containing embeddings |
| `--output_dir` | No | results/evaluation/row_prediction | Directory to save results |
| `--config` | No | configs/downstream/row_prediction.yaml | Path to YAML config file |
| `--label_column` | No | None | Specific label to predict (default: all) |
| `--task` | No | auto | Task type: auto, classification, regression |
| `--head_type` | No | mlp | Probe type: mlp (PyTorch MLP) or linear (sklearn linear probe) |
| `--seed` | No | None | Random seed (overrides config training.seed) |
| `--model` | No | None | Model name (stored in result JSON for aggregation) |
| `--dataset` | No | None | Dataset name (stored in result JSON for aggregation) |
| `--quiet` | No | False | Suppress detailed output |

## Output

Results saved to `{output_dir}/` (per-label subdirectories for multi-label datasets):
- `results.json`: Test metrics and training summary
- `best_model.pt`: Best model checkpoint (MLP only, by val_loss)

### Example results.json (MLP)
```json
{
  "task_name": "row_prediction_Class",
  "task": "row_prediction",
  "task_type": "classification",
  "head_type": "mlp",
  "seed": 42,
  "model": "bert",
  "dataset": "openml_1486",
  "label_column": "Class",
  "test_results": {
    "loss": 0.3421,
    "accuracy": 0.8745,
    "weighted_f1": 0.8712,
    "macro_f1": 0.8701
  },
  "training": {
    "best_epoch": 42,
    "best_value": 0.3198,
    "total_epochs": 52
  },
  "data_stats": {
    "train": 27572,
    "val": 3446,
    "test": 3447,
    "input_dim": 768,
    "n_classes": 2
  }
}
```

## Task Type Detection

Task type is resolved in order:
1. **Metadata** (`label_task_types` from `metadata.json`) — preferred
2. **CLI override** (`--task classification` or `--task regression`)
3. **Heuristic** (fallback): integers with <20 unique values → classification; floats with >10% uniqueness or >20 unique values → regression

When metadata specifies `label_task_types`, `--task` is ignored. Use `--task` to override only when metadata lacks task type.

---

## Troubleshooting

### Embeddings not found
```
Error: Could not load embeddings from embeddings/
```
**Solution:** Ensure embeddings are generated first. Check expected files:
- `train_embeddings.npy`
- `train_labels.npy`
- `test_embeddings.npy`
- `test_labels.npy`

### CUDA out of memory
**Solution:** Edit `configs/downstream/row_prediction.yaml` and reduce `training.batch_size`, or set `training.device: cpu`.

### Poor accuracy
- Check embedding quality with embedding visualization
- Verify label correctness
- Try adjusting learning rate or hidden dims in the config

### Wrong task type detected
```
Auto-detected task type: regression (expected classification)
```
**Solution:** Override with `--task classification`.

## Related

- Embedding generation: `models/scarf/USAGE.md`, `models/vime/USAGE.md`, `models/subtab/USAGE.md`
- Similar evaluation: `downstream_tasks/column_type_prediction/`
- Config: `configs/downstream/row_prediction.yaml`
- Trainer: `utils/downstream/trainer.py`
