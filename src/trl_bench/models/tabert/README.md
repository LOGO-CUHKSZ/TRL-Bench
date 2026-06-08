# TaBERT Integration

TaBERT is a pretrained language model for joint understanding of natural language utterances and (semi-)structured tables. It extends BERT to learn contextual representations for both text and tabular data.

## Directory Structure

```
models/tabert/
├── generate_column_embeddings.py  # Main embedding API
├── README.md                      # This file
└── tabert/                        # Core TaBERT modules
    ├── table_bert.py              # Base model class
    ├── vanilla_table_bert.py      # K=1 model
    ├── table.py                   # Table/Column classes
    ├── config.py                  # Configuration
    ├── input_formatter.py         # Input tokenization
    ├── utils.py                   # Utilities
    ├── dataset.py                 # Dataset utilities
    └── vertical/                  # K>1 models with vertical attention
```

## Checkpoints

Available at `/path/to/TRL-Bench/checkpoints/tabert/`:

| Checkpoint | Hidden Size | K (rows) | Description |
|------------|-------------|----------|-------------|
| `tabert_base_k1` | 768 | 1 | Base model, single synthetic row |
| `tabert_base_k3` | 768 | 3 | Base model, vertical attention |
| `tabert_large_k1` | 1024 | 1 | Large model, single synthetic row |
| `tabert_large_k3` | 1024 | 3 | Large model, vertical attention (best) |

## Usage

### Python API

```python
from models.tabert.generate_column_embeddings import TaBERTEmbedder

# Load model (auto-detects K=1 vs K=3)
embedder = TaBERTEmbedder(
    checkpoint_path='checkpoints/tabert/tabert_large_k3/model.bin',
    device='cuda'
)

# Mode 1: Column-only embeddings (no context)
result = embedder.encode_csv(
    'path/to/table.csv',
    context_mode='column'
)

# Mode 2: Header-aware embeddings (column names as context)
result = embedder.encode_csv(
    'path/to/table.csv',
    context_mode='header'
)

# Mode 3: Context-aware embeddings (with question)
result = embedder.encode_csv(
    'path/to/table.csv',
    context="What is the population of Tokyo?",
    context_mode='context'
)

# Access embeddings
print(result['table_embedding'].shape)      # (1024,)
print(result['column_embeddings'][0].shape) # (1024,)
print(result['column_names'])               # ['col1', 'col2', ...]
```

### Command Line

```bash
# Single file - column embeddings
python models/tabert/generate_column_embeddings.py \
    --input path/to/table.csv \
    --checkpoint checkpoints/tabert/tabert_large_k3/model.bin \
    --context_mode column \
    --output embeddings.pkl

# With question context
python models/tabert/generate_column_embeddings.py \
    --input path/to/table.csv \
    --checkpoint checkpoints/tabert/tabert_large_k3/model.bin \
    --context "What is the population?" \
    --context_mode context \
    --output embeddings.pkl

# Batch mode (directory)
python models/tabert/generate_column_embeddings.py \
    --input path/to/csv_directory/ \
    --checkpoint checkpoints/tabert/tabert_large_k3/model.bin \
    --output all_embeddings.pkl
```

## Context Modes

1. **column** (default): Pure column embeddings without any context. Best for table similarity/search tasks.

2. **header**: Uses column names as pseudo-context. Provides column-aware representations without requiring a question.

3. **context**: Full context-aware embeddings using a provided question. Best for semantic parsing and QA tasks where embeddings should be conditioned on the question.

## Output Format

```python
{
    'table_embedding': np.ndarray,     # (hidden_size,) - mean-pooled table
    'column_embeddings': {             # Per-column embeddings
        0: np.ndarray,                 # (hidden_size,)
        1: np.ndarray,
        ...
    },
    'context_embedding': np.ndarray,   # (seq_len, hidden_size) - if context provided
    'column_names': ['col1', 'col2'],  # Column name strings
    'table_name': 'table_id',          # Table identifier
    'model_type': 'VerticalAttentionTableBert',  # or 'VanillaTableBert'
    'embedding_dim': 1024              # 768 for base, 1024 for large
}
```

## Batch Processing with Questions

For semantic parsing tasks where multiple questions use the same table:

```python
results = embedder.encode_with_questions(
    csv_path='path/to/table.csv',
    questions=[
        "What is the population?",
        "Which city has the highest GDP?",
        "How many countries are listed?"
    ]
)
# Returns list of embedding dicts, one per question
```

## Downstream Task Scripts

Available at `scripts/tabert/`:

| Script | Task | Description |
|--------|------|-------------|
| `semantic_parsing_wtq.sh` | Semantic Parsing | WikiTableQuestions with MAPO decoder |
| `join_search_wiki_join.sh` | Join Search | Wiki join search with FAISS |
| `union_search_classification_wiki_union.sh` | Union Search | Wiki union search |

### Running Downstream Tasks

```bash
cd /path/to/TRL-Bench

# Semantic Parsing (WTQ)
bash scripts/tabert/semantic_parsing_wtq.sh --cuda

# Join Search
bash scripts/tabert/join_search_wiki_join.sh --use_gpu

# Skip extraction if embeddings exist
bash scripts/tabert/join_search_wiki_join.sh --use_gpu --skip_extraction
```

## Configuration

Task-specific configs are stored in `configs/` (separate from checkpoints):

```
configs/
└── semantic_parsing/
    └── wtq_mapo.json    # MAPO decoder config for WTQ task
```

The config contains MAPO decoder hyperparameters (learning rate, hidden size, etc.) and is **shared across embedding models** (TaBERT, TabSketchFM). The embedding path is passed via command line, not stored in the config.

## Embedding Format Conversion

TaBERT outputs embeddings in dict format. Some downstream scripts (e.g., `run_tabsketchfm_search.py`) expect tuple format. Use the conversion utility:

```bash
# Convert TaBERT embeddings to tuple format
python utils/embedding_conversion/tabert_to_tuple_format.py \
    --input embeddings.pkl \
    --inplace

# Or output to new file
python utils/embedding_conversion/tabert_to_tuple_format.py \
    --input embeddings.pkl \
    --output embeddings_tuple.pkl
```

The conversion is automatically integrated into `join_search_wiki_join.sh` and `union_search_classification_wiki_union.sh`.

## Environment

Requires the TaBERT environment:

```bash
source /path/to/TaBERT/load_env
# Or from trl project:
# Ensure PYTHONPATH includes TaBERT dependencies
```

## Dependencies

- PyTorch with CUDA support
- transformers (HuggingFace)
- torch_scatter
- pandas, numpy, tqdm

## References

- TaBERT Paper: [TaBERT: Pretraining for Joint Understanding of Textual and Tabular Data](https://arxiv.org/abs/2005.08314)
- Original Repository: https://github.com/facebookresearch/TaBERT
