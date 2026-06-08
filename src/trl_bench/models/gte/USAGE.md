# GTE Embedding Generation

**Model:** [thenlper/gte-base](https://huggingface.co/thenlper/gte-base)
**Architecture:** BERT-like encoder, 768-dim, CLS pooling
**Max tokens:** 512

## Column Embeddings (Table Retrieval)

Generate table and per-column embeddings from CSV files using pretrained GTE.

```bash
# Single CSV
python models/gte/generate_column_embeddings.py \
    --input table.csv --output embeddings.pkl

# Directory of CSVs
python models/gte/generate_column_embeddings.py \
    --input /path/to/csvs/ --output embeddings.pkl \
    --device cuda --max_rows 50 --checkpoint_interval 200
```

Output: v2.0 pickle (list of dicts) with `table_embedding.cls_embedding`, `column_mean`, `token_mean`, and per-column embeddings.

## Text Embeddings (Queries / Questions)

Encode arbitrary text into GTE embeddings. Two modes:

```bash
# CLS embeddings from JSON (for table retrieval queries)
python models/gte/generate_text_embeddings.py --mode cls \
    --input_json train.json --text_field question \
    --output queries.pkl

# Token-level embeddings from text file (for semantic parsing)
python models/gte/generate_text_embeddings.py --mode token \
    --input_text sentences.txt \
    --output token_embs.pkl

# With batching and GPU
python models/gte/generate_text_embeddings.py --mode cls \
    --input_json data.json --text_field question \
    --batch_size 64 --device cuda --output queries.pkl
```

Output: pickle (list of dicts) with `text_id`, `text`, `embedding`, `model_name`, `mode`.
- `cls` mode: embedding shape `(768,)`
- `token` mode: embedding shape `(seq_len, 768)`

## Row Embeddings (Record Linkage / Row Prediction)

Generate per-row embeddings by serializing each data row as `"col: val | col: val | ..."` and encoding through GTE's [CLS] token.

### Pipeline 1: Directory Mode

Processes a flat directory of CSVs into an aggregate pickle.

```bash
python models/gte/generate_row_embeddings.py \
    --input_dir /path/to/csvs/ \
    --output_path embeddings.pkl \
    --device cuda \
    --label_columns target_col
```

Key options:
- `--max_chars_per_cell 100`: Truncate cell values (default 100)
- `--row_batch_size 64`: Rows per forward pass
- `--checkpoint_interval 50`: Save progress every N tables
- `--max_rows N`: **Debug only.** Truncates each table to N rows, producing pickles that no longer match source CSV row counts. Not compatible with row-position-based consumers like `run_record_linkage.py`.

Output: aggregate pickle with `row_embeddings` shape `(n_rows, 768)` per table.

### Pipeline 2: Split-Aware Mode

Generates embeddings for canonical datasets with train/test splits.

```bash
python models/gte/generate_embeddings_train_test.py \
    --data_dir datasets/row_data/openml_1486 \
    --embedding_dir embeddings/row_prediction/gte/openml_1486 \
    --label_policy manifest \
    --device cuda
```

Output: v2.0 split-aware format (`metadata.json` + per-split `.npy` files).
