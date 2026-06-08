# Semantic Parsing Task

This module provides a modular framework for semantic parsing tasks with swappable decoders.

## Architecture

```
downstream_tasks/semantic_parsing/
├── tasks/                      # Task definitions
│   ├── base.py                 # TaskBase interface
│   └── wiki_table_questions/   # WTQ task implementation
│
├── decoders/                   # Decoder implementations
│   ├── base.py                 # DecoderBase interface
│   └── mapo/                   # MAPO decoder (actor-learner architecture)
│
├── execution/                  # Program execution layer
│   ├── env_factory.py          # Environment creation
│   ├── computer_factory.py     # Lisp interpreter
│   └── executor_factory.py     # Table executors
│
├── evaluation/                 # Evaluation utilities
│   └── metrics.py              # Accuracy metrics
│
├── data/                       # Data utilities
│   └── data_utils.py           # Data loading helpers
│
├── config/                     # Configuration files
│   └── mapo.json               # MAPO hyperparameters
│
├── run_training.py             # Training entry point
└── run_evaluation.py           # Evaluation entry point
```

## Usage

### Training

```bash
cd /path/to/TRL-Bench

python -m downstream_tasks.semantic_parsing.run_training \
    --task wiki_table_questions \
    --decoder mapo \
    --embedding-path embeddings/semantic_parsing/wiki_table_questions/tabert_large_k3/embeddings.npz \
    --dataset-path datasets/semantic_parsing/wiki_table_questions \
    --output-dir assets/checkpoints/semantic_parsing/wiki_table_questions/mapo/tabert_large_k3 \
    --config downstream_tasks/semantic_parsing/config/mapo.json \
    --cuda
```

### Evaluation

```bash
python -m downstream_tasks.semantic_parsing.run_evaluation \
    --task wiki_table_questions \
    --decoder mapo \
    --model-path assets/checkpoints/semantic_parsing/wiki_table_questions/mapo/tabert_large_k3/model.best.bin \
    --embedding-path embeddings/semantic_parsing/wiki_table_questions/tabert_large_k3/embeddings.npz \
    --dataset-path datasets/semantic_parsing/wiki_table_questions \
    --output-dir results/evaluation/semantic_parsing/wiki_table_questions/mapo/tabert_large_k3 \
    --beam-size 10 \
    --cuda
```

## Supported Tasks

| Task | Description | Status |
|------|-------------|--------|
| wiki_table_questions | WikiTableQuestions semantic parsing | Implemented |
| spider | Text-to-SQL on Spider | Planned |
| hybridqa | Hybrid table+text QA | Planned |

## Supported Decoders

| Decoder | Description | Status |
|---------|-------------|--------|
| mapo | Memory Augmented Policy Optimization | Implemented |
| transformer | Transformer-based decoder | Planned |
| seq2seq | Sequence-to-sequence decoder | Planned |

## Embeddings

The framework uses pre-computed embeddings, making it model-agnostic. Embeddings should be generated separately and stored as `.npz` files with the format:

```
{example_id}_question: numpy array of shape (seq_len, embed_dim)
{example_id}_column: numpy array of shape (num_cols, embed_dim)
```

## Expected Results

With TaBERT Large K=3 embeddings:
- Dev Accuracy: ~30.84%
- Dev Oracle Accuracy: ~51.64%

Note: Lower than end-to-end training (~50.23%) because embeddings are not fine-tuned.

## Development Status

### Completed
- Directory structure
- Dataset and embedding migration
- Task interface (TaskBase, WTQTask)
- Decoder interface (DecoderBase, MAPODecoder)
- Execution layer (interpreter, executor, environments)
- Configuration files
- Training entry point

### In Progress
- Full independence from TaBERT repo (load_environments function)
- Evaluation entry point

### Planned
- Additional tasks (Spider, HybridQA)
- Additional decoders (Transformer, Seq2seq)
