# TAPAS: Table Parser for Table Understanding

TAPAS (Table Parser) is a BERT-based model for answering questions about tabular data, introduced in:

**"TAPAS: Weakly Supervised Table Parsing via Pre-training"**
Jonathan Herzig, Paweł Krzysztof Nowak, Thomas Müller, Francesco Piccinno, Julian Martin Eisenschlos
ACL 2020
Paper: https://arxiv.org/abs/2004.02349

## Overview

TAPAS extends BERT to encode tables along with natural language questions. Unlike traditional semantic parsing approaches that generate SQL-like logical forms, TAPAS directly predicts answers by:
1. **Selecting relevant cells** from the table
2. **Applying aggregation operations** (COUNT, SUM, AVERAGE) when needed

### Key Features

- **Joint text-table encoding**: Encodes questions and tables in a single sequence
- **Weak supervision**: Trained on question-answer pairs without intermediate logical forms
- **Aggregation prediction**: Automatically detects when aggregation is needed
- **Pre-trained models**: Multiple variants available (tiny to large)

## Installation

```bash
# Install HuggingFace transformers (TAPAS is included)
pip install transformers torch pandas numpy
```

## Directory Structure

```
models/tapas/
├── __init__.py                    # Package exports
├── generate_column_embeddings.py  # Table/column embedding generation
├── tapas_qa.py                    # Table question answering
└── README.md                      # This file
```

## Usage

### 1. Generate Table Embeddings

Generate embeddings for downstream tasks like table retrieval or similarity.

```bash
# Single table
python models/tapas/generate_column_embeddings.py \
    --input data/tables/sample.csv \
    --output embeddings.pkl

# With question context (question-aware embeddings)
python models/tapas/generate_column_embeddings.py \
    --input data/tables/sample.csv \
    --question "What is the total revenue?" \
    --output embeddings.pkl

# Batch mode (directory of CSVs)
python models/tapas/generate_column_embeddings.py \
    --input data/tables/ \
    --output all_embeddings.pkl

# Use larger model
python models/tapas/generate_column_embeddings.py \
    --input data/tables/sample.csv \
    --model google/tapas-large \
    --output embeddings.pkl
```

#### Python API

```python
from models.tapas import TAPASEmbedder

# Initialize embedder
embedder = TAPASEmbedder(model_name='google/tapas-base')

# Generate embeddings for a CSV file
result = embedder.encode_csv('table.csv')
print(result['table_embedding'].shape)      # (768,)
print(result['column_embeddings'])          # {0: array, 1: array, ...}
print(result['column_names'])               # ['col1', 'col2', ...]

# With question context
result = embedder.encode_csv('table.csv', question='What is the total?')

# Batch processing
results = embedder.encode_directory('csv_directory/')
```

### 2. Table Question Answering

Answer natural language questions about tables.

```bash
# Single question
python models/tapas/tapas_qa.py \
    --table data/tables/sales.csv \
    --question "What is the total revenue?"

# Multiple questions
python models/tapas/tapas_qa.py \
    --table data/tables/sales.csv \
    --questions "What is the total?" "How many products?" "What is the average price?"

# Show detailed output (cell coordinates)
python models/tapas/tapas_qa.py \
    --table data/tables/sales.csv \
    --question "Which product has the highest sales?" \
    --details

# Save results to JSON
python models/tapas/tapas_qa.py \
    --table data/tables/sales.csv \
    --questions_file questions.txt \
    --output answers.json
```

#### Python API

```python
from models.tapas import TAPASQuestionAnswering

# Initialize QA model
qa = TAPASQuestionAnswering(model_name='google/tapas-base-finetuned-wtq')

# Answer a question
answer = qa.answer('sales.csv', 'What is the total revenue?')
print(answer)  # "1500000"

# Get detailed output
result = qa.answer('sales.csv', 'What is the total?', return_details=True)
print(result['answer'])        # "1500000"
print(result['aggregation'])   # "SUM"
print(result['coordinates'])   # [(0, 2), (1, 2), (2, 2)]
print(result['selected_cells'])  # List of selected cell info

# Batch QA
answers = qa.answer_batch('sales.csv', [
    'What is the total revenue?',
    'How many products are there?',
    'What is the average price?'
])

# Conversational QA (for SQA model)
qa_sqa = TAPASQuestionAnswering(model_name='google/tapas-base-finetuned-sqa')
results = qa_sqa.answer_conversational('table.csv', [
    'What products are listed?',
    'What is the price of the first one?',
    'And the second?'
])
```

## Available Models

### Pre-trained Models (embeddings only)

| Model | Hidden Size | Parameters |
|-------|-------------|------------|
| `google/tapas-tiny` | 128 | ~5M |
| `google/tapas-mini` | 256 | ~11M |
| `google/tapas-small` | 512 | ~22M |
| `google/tapas-base` | 768 | ~110M |
| `google/tapas-large` | 1024 | ~340M |

### Fine-tuned Models (for QA)

| Model | Dataset | Best For |
|-------|---------|----------|
| `google/tapas-base-finetuned-wtq` | WikiTableQuestions | General table QA |
| `google/tapas-base-finetuned-sqa` | SQA | Conversational/sequential QA |
| `google/tapas-base-finetuned-wikisql-supervised` | WikiSQL | SQL-style queries |

## Output Format

### Embedding Output

```python
{
    'table_embedding': np.ndarray,      # Shape: (hidden_size,)
    'column_embeddings': {              # Dict: col_idx -> embedding
        0: np.ndarray,                  # Shape: (hidden_size,)
        1: np.ndarray,
        ...
    },
    'cls_embedding': np.ndarray,        # CLS token embedding
    'column_names': ['col1', 'col2'],   # List of column names
    'table_name': 'filename',           # Table identifier
    'model_name': 'google/tapas-base',  # Model used
    'embedding_dim': 768,               # Embedding dimension
    'question': 'optional question'     # If provided
}
```

### QA Output (with details)

```python
{
    'answer': '1500000',                # Final answer string
    'aggregation': 'SUM',               # NONE, SUM, AVERAGE, or COUNT
    'coordinates': [(0, 2), (1, 2)],    # (row, col) of selected cells
    'selected_cells': [                 # Detailed cell info
        {'row': 0, 'column': 2, 'column_name': 'revenue', 'value': '500000'},
        {'row': 1, 'column': 2, 'column_name': 'revenue', 'value': '1000000'}
    ],
    'question': 'What is the total revenue?'
}
```

## Architecture Details

TAPAS extends BERT with:
- **7 token type embeddings**: segment, column, row, previous_label, column_rank, inverse_column_rank, numeric_relations
- **Relative position embeddings**: Instead of absolute positions
- **Cell selection head**: Predicts probability of each cell being part of the answer
- **Aggregation head**: Predicts aggregation operation (NONE/SUM/AVERAGE/COUNT)

The model flattens tables row-by-row:
```
[CLS] question tokens [SEP] header1 [SEP] header2 [SEP] ... [SEP] cell11 cell12 ... [SEP] cell21 ...
```

## Usage notes

- **Sequence length**: up to 512 tokens (the TAPAS standard); larger tables are truncated to fit the budget.
- **Rows**: ~50-100 rows fit within the token budget, depending on cell content.

## Citation

```bibtex
@inproceedings{herzig-etal-2020-tapas,
    title = "{T}a{P}as: Weakly Supervised Table Parsing via Pre-training",
    author = "Herzig, Jonathan and Nowak, Pawe{\l} Krzysztof and M{\"u}ller, Thomas and Piccinno, Francesco and Eisenschlos, Julian Martin",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    year = "2020",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.acl-main.398",
    doi = "10.18653/v1/2020.acl-main.398",
    pages = "4320--4333",
}
```

## References

- [Original Paper](https://arxiv.org/abs/2004.02349)
- [HuggingFace Documentation](https://huggingface.co/docs/transformers/model_doc/tapas)
- [Google Research Blog](https://ai.googleblog.com/2020/04/using-neural-networks-to-find-answers.html)
- [Papers With Code](https://paperswithcode.com/paper/tapas-weakly-supervised-table-parsing-via)
