# Semantic Parsing

## Overview

Translates natural language questions about tables into executable programs (logical forms). Uses table and question embeddings to generate programs that answer questions via table operations.

Reference: "MAPO: Coupling Maximum a Posteriori Inference with Policy Optimization" (Liang et al., ICLR 2018)

## Embeddings Consumed

> **Embedding Level:** Column + Question
> **Primary Embedding:** NPZ file with column and question embeddings per example
> **Pair Input:** Yes (table-question pairs)

| Embedding Type | Required | Shape | Description | Compatible Models |
|----------------|----------|-------|-------------|-------------------|
| `{id}_column` | **Yes** | `(num_cols, dim)` | Column embeddings per example | TaBERT, TAPAS |
| `{id}_question` | **Yes** | `(dim,)` | Question embedding per example | TaBERT, TAPAS |

**Input format:** NPZ file (`.npz`) containing:
```python
{
    '{example_id}_column': np.array(num_cols, dim),   # Column embeddings
    '{example_id}_question': np.array(dim,),          # Question embedding
    ...
}
```

Example with `load()`:
```python
import numpy as np
data = np.load('embeddings.npz')
# Access example 'train-1':
column_emb = data['train-1_column']    # shape: (num_cols, 768)
question_emb = data['train-1_question']  # shape: (768,)
```

## Task Configuration

| Property | Value |
|----------|-------|
| **Task Type** | Program Generation |
| **Embedding Level** | Column + Question |
| **Pair Input** | Yes |
| **Dataset** | WikiTableQuestions (WTQ) |

## Evaluation Metrics

| Metric | Primary | Description |
|--------|---------|-------------|
| Denotation Accuracy | Yes | Fraction of correct answers |
| Program Accuracy | No | Exact program match |

## Input Data

**Embeddings:** `embeddings/semantic_parsing/<task>/<model>/embeddings.npz`

**Dataset:**
```
datasets/semantic_parsing/wiki_table_questions/
├── tables.jsonl
├── data_split_1/
│   ├── train_split_shard_90-*.jsonl
│   └── dev_split.jsonl
└── saved_programs.json
```

## Example Commands

### Basic Training

```bash
python -m downstream_tasks.semantic_parsing.run_training \
    --task wiki_table_questions \
    --decoder mapo \
    --embedding-path embeddings/semantic_parsing/wiki_table_questions/tabert_large_k3/embeddings.npz \
    --dataset-path datasets/semantic_parsing/wiki_table_questions \
    --output-dir checkpoints/semantic_parsing/wiki_table_questions/mapo/tabert_large_k3 \
    --config downstream_tasks/semantic_parsing/config/mapo.json \
    --cuda
```

### With Custom Configuration

```bash
python -m downstream_tasks.semantic_parsing.run_training \
    --task wiki_table_questions \
    --decoder mapo \
    --embedding-path embeddings/semantic_parsing/wtq/tapas/embeddings.npz \
    --dataset-path datasets/semantic_parsing/wiki_table_questions \
    --output-dir checkpoints/semantic_parsing/wtq/tapas \
    --config config/mapo_custom.json \
    --cuda \
    --seed 42
```

### Resume Training

```bash
python -m downstream_tasks.semantic_parsing.run_training \
    --task wiki_table_questions \
    --decoder mapo \
    --embedding-path embeddings.npz \
    --dataset-path datasets/semantic_parsing/wiki_table_questions \
    --output-dir checkpoints/existing \
    --config config/mapo.json \
    --cuda \
    --resume
```

## Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--task` | No | wiki_table_questions | Task name |
| `--decoder` | No | mapo | Decoder name (mapo) |
| `--embedding-path` | Yes | - | Path to embeddings NPZ file |
| `--dataset-path` | Yes | - | Path to dataset directory |
| `--output-dir` | Yes | - | Output directory for checkpoints |
| `--log-dir` | No | None | Directory for training logs |
| `--config` | Yes | - | Path to configuration JSON |
| `--cuda` | No | False | Use CUDA |
| `--seed` | No | 0 | Random seed |
| `--resume` | No | False | Resume from checkpoint |

## Configuration File

Example `config/mapo.json`:
```json
{
    "max_n_mem": 10,
    "max_n_exp": 10,
    "use_replay_prob": 0.5,
    "batch_size": 50,
    "n_actors": 4,
    "n_explore_samples": 10,
    "use_nonreplay_prob": 0.1,
    "learning_rate": 0.001,
    "entropy_coef": 0.01,
    "eval_every_n": 100,
    "save_every_n": 1000
}
```

## MAPO Decoder

The MAPO (Maximum a Posteriori Policy Optimization) decoder:

1. **Actors:** Parallel exploration for program candidates
2. **Replay Buffer:** Stores high-reward programs
3. **Policy Gradient:** Updates based on execution rewards
4. **Memory:** Caches successful programs for weak supervision

## Dataset Structure

WikiTableQuestions format:
```
datasets/semantic_parsing/wiki_table_questions/
├── tables.jsonl              # Table definitions
├── data_split_1/
│   ├── train_split_shard_90-0.jsonl
│   ├── train_split_shard_90-1.jsonl
│   └── dev_split.jsonl
└── saved_programs.json       # Pre-searched programs
```

**Example entry:**
```json
{
    "id": "train-1",
    "question": "What is the total population?",
    "table_id": "csv/123.csv",
    "answer": ["45000"]
}
```

## Output

Results saved to `{output_dir}/`:
- `model.pt`: Final model checkpoint
- `config.json`: Training configuration
- `metrics.json`: Evaluation metrics

### Training output
```
[run_training] Loading dataset from datasets/semantic_parsing/wiki_table_questions...
[run_training] Loaded 11321 train examples, 2831 dev examples
[run_training] Starting training with mapo decoder...
...
[run_training] Training complete. Results saved to checkpoints/semantic_parsing/wtq/tabert
```

---

## Troubleshooting

### Embedding dimension mismatch
```
RuntimeError: size mismatch
```
**Solution:** Ensure embeddings were generated with compatible model (TaBERT large: 1024-dim, base: 768-dim).

### Missing programs
```
Warning: No saved programs found
```
**Solution:** Run program search first or ensure `saved_programs.json` exists.

### CUDA out of memory
- Reduce `batch_size` in config
- Reduce `n_actors`
- Use smaller embedding model

### Low accuracy
- Use saved programs for weak supervision
- Increase training iterations
- Try different random seeds

## Related

- Embedding generation: `models/tabert/USAGE.md`, `models/tapas/USAGE.md`
- Similar tasks: `downstream_tasks/table_fact_verification/`
