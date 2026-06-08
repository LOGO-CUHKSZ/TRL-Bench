#!/usr/bin/env python
"""
Generate TaBERT embeddings for NQ-Tables dataset.

This script generates table embeddings using TaBERT and query embeddings using BERT.
TaBERT is designed for joint table-query encoding, but for retrieval we need:
- Tables: TaBERT table embeddings (context_mode='column' for query-independent)
- Queries: BERT embeddings (same as other table encoders like TAPAS, TabSketchFM)

Usage:
    # Generate both table and query embeddings
    python -m downstream_tasks.table_retrieval.generate_tabert_embeddings \
        --tables_json datasets/nq_tables/json/tables.json \
        --train_json datasets/nq_tables/json/train.json \
        --dev_json datasets/nq_tables/json/dev.json \
        --tabert_checkpoint checkpoints/tabert/tabert_base_k1/model.bin \
        --bert_model bert-base-uncased \
        --output_dir embeddings/table_retrieval/tabert \
        --batch_size 32

    # Generate only table embeddings (queries will be symlinked from BERT)
    python -m downstream_tasks.table_retrieval.generate_tabert_embeddings \
        --tables_json datasets/nq_tables/json/tables.json \
        --tabert_checkpoint checkpoints/tabert/tabert_base_k1/model.bin \
        --output_dir embeddings/table_retrieval/tabert \
        --tables_only \
        --link_bert_queries embeddings/table_retrieval/bert
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from tqdm import tqdm

# Add TaBERT to path
TABERT_PATH = os.environ.get("TABERT_PATH", "models/tabert")
sys.path.insert(0, TABERT_PATH)

from table_bert.table_bert import TableBertModel
from table_bert.table import Table, Column


def nq_table_to_tabert_table(nq_table: Dict[str, Any]) -> Table:
    """
    Convert NQ-Tables format to TaBERT Table format.

    NQ-Tables format:
    {
        "table_id": "...",
        "title": "...",
        "header": ["col1", "col2", ...],
        "rows": [["val1", "val2", ...], ...]
    }

    TaBERT Table format:
    Table(
        id=table_id,
        header=[Column(name=col_name, type='text', sample_value=val), ...],
        data=[{"col1": "val1", "col2": "val2", ...}, ...]
    )
    """
    table_id = nq_table["table_id"]
    header_names = nq_table["header"]
    rows = nq_table["rows"]

    # Create columns with sample values from first row
    columns = []
    for i, col_name in enumerate(header_names):
        # Get sample value from first row if available
        sample_value = ""
        if rows and i < len(rows[0]):
            sample_value = str(rows[0][i]) if rows[0][i] is not None else ""

        # Infer type (all text for simplicity, TaBERT handles this well)
        col_type = "text"

        columns.append(Column(
            name=str(col_name) if col_name else f"col_{i}",
            type=col_type,
            sample_value=sample_value
        ))

    # Convert rows to list of dicts
    data = []
    for row in rows:
        row_dict = {}
        for i, col in enumerate(columns):
            if i < len(row):
                row_dict[col.name] = str(row[i]) if row[i] is not None else ""
            else:
                row_dict[col.name] = ""
        data.append(row_dict)

    return Table(
        id=table_id,
        header=columns,
        data=data,
        name=nq_table.get("title", "")
    )


def generate_tabert_table_embeddings(
    tables_json_path: str,
    checkpoint_path: str,
    output_dir: str,
    batch_size: int = 32,
    max_rows: int = 50,
    device: str = None,
    embedding_mode: str = "mean_column"
) -> None:
    """
    Generate TaBERT table embeddings for all tables in NQ-Tables.

    Args:
        tables_json_path: Path to NQ-Tables tables.json
        checkpoint_path: Path to TaBERT checkpoint
        output_dir: Output directory for embeddings
        batch_size: Batch size for encoding
        max_rows: Maximum rows per table to encode
        device: Device to use
        embedding_mode: How to create table embedding:
            - "mean_column": Mean-pool column embeddings (default)
            - "cls": Use CLS token from TaBERT's BERT encoding
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading TaBERT model from {checkpoint_path}")
    model = TableBertModel.from_pretrained(checkpoint_path)
    model = model.to(device)
    model.eval()

    embedding_dim = model.output_size
    print(f"TaBERT embedding dimension: {embedding_dim}")
    print(f"Model type: {type(model).__name__}")
    print(f"Embedding mode: {embedding_mode}")

    # Get sample_row_num from config
    sample_row_num = getattr(model.config, 'sample_row_num', 1)
    print(f"Sample row num (K): {sample_row_num}")

    print(f"\nLoading tables from {tables_json_path}")
    with open(tables_json_path, 'r') as f:
        tables_data = json.load(f)
    print(f"Total tables: {len(tables_data)}")

    # Generate embeddings
    all_embeddings = []
    id2table = {}
    failed_tables = []

    print("\nGenerating TaBERT table embeddings...")
    for idx, nq_table in enumerate(tqdm(tables_data)):
        try:
            # Convert to TaBERT format
            table = nq_table_to_tabert_table(nq_table)

            # Limit rows for encoding
            if len(table.data) > max_rows:
                table = table.with_rows(table.data[:max_rows])

            # Further limit to sample_row_num for K>1 models
            if sample_row_num > 1 and len(table.data) > sample_row_num:
                table = table.with_rows(table.data[:sample_row_num])

            # Tokenize
            table.tokenize(model.tokenizer)

            # Encode with empty context
            # Request bert_encoding if using CLS mode
            with torch.no_grad():
                context_encoding, column_encoding, info = model.encode(
                    contexts=[[]],  # Empty context
                    tables=[table],
                    return_bert_encoding=(embedding_mode == "cls")
                )

            # Extract table embedding based on mode
            if embedding_mode == "cls":
                # Use CLS token from TaBERT's BERT encoding
                # CLS is the first token [0] in the sequence
                bert_encoding = info['bert_encoding']  # (1, seq_len, hidden_size)
                table_emb = bert_encoding[0, 0, :].cpu().numpy()  # CLS token
            elif embedding_mode == "mean_column":
                # Mean-pool column embeddings
                column_emb = column_encoding[0].cpu().numpy()  # (num_cols, hidden_size)
                if column_emb.shape[0] > 0:
                    table_emb = np.mean(column_emb, axis=0)
                else:
                    table_emb = np.zeros(embedding_dim, dtype=np.float32)
            else:
                raise ValueError(f"Unknown embedding_mode: {embedding_mode}")

            all_embeddings.append(table_emb.astype(np.float32))
            id2table[str(len(id2table))] = nq_table["table_id"]

        except Exception as e:
            failed_tables.append((nq_table["table_id"], str(e)))
            # Add zero embedding as placeholder
            all_embeddings.append(np.zeros(embedding_dim, dtype=np.float32))
            id2table[str(len(id2table))] = nq_table["table_id"]

    # Convert to numpy array
    embeddings_array = np.stack(all_embeddings, axis=0)
    print(f"\nTable embeddings shape: {embeddings_array.shape}")

    if failed_tables:
        print(f"\nWarning: {len(failed_tables)} tables failed to encode:")
        for tid, err in failed_tables[:5]:
            print(f"  - {tid}: {err}")
        if len(failed_tables) > 5:
            print(f"  ... and {len(failed_tables) - 5} more")

    # Save embeddings
    os.makedirs(output_dir, exist_ok=True)

    tables_path = os.path.join(output_dir, "tables.npy")
    np.save(tables_path, embeddings_array)
    print(f"Saved table embeddings to {tables_path}")

    id2table_path = os.path.join(output_dir, "id2table.json")
    with open(id2table_path, 'w') as f:
        json.dump(id2table, f)
    print(f"Saved id2table mapping to {id2table_path}")


def generate_bert_query_embeddings(
    questions_json_path: str,
    bert_model: str,
    output_path: str,
    batch_size: int = 32,
    max_length: int = 128,
    device: str = None
) -> None:
    """
    Generate BERT query embeddings for questions.

    Args:
        questions_json_path: Path to questions JSON (train.json or dev.json)
        bert_model: BERT model name or path
        output_path: Output path for embeddings (.npy)
        batch_size: Batch size
        max_length: Max sequence length
        device: Device to use
    """
    from transformers import BertModel, BertTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nLoading BERT model: {bert_model}")
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    model = BertModel.from_pretrained(bert_model)
    model = model.to(device)
    model.eval()

    print(f"Loading questions from {questions_json_path}")
    with open(questions_json_path, 'r') as f:
        questions_data = json.load(f)
    print(f"Total questions: {len(questions_data)}")

    # Extract question texts
    questions = [q["question"] for q in questions_data]

    # Generate embeddings in batches
    all_embeddings = []

    print("Generating BERT query embeddings...")
    for i in tqdm(range(0, len(questions), batch_size)):
        batch_questions = questions[i:i + batch_size]

        # Tokenize
        inputs = tokenizer(
            batch_questions,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        # Encode
        with torch.no_grad():
            outputs = model(**inputs)
            # Use CLS token embedding
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        all_embeddings.append(cls_embeddings)

    # Concatenate all embeddings
    embeddings_array = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    print(f"Query embeddings shape: {embeddings_array.shape}")

    # Save
    np.save(output_path, embeddings_array)
    print(f"Saved query embeddings to {output_path}")


def link_bert_queries(bert_dir: str, output_dir: str) -> None:
    """
    Create symlinks to BERT query embeddings.

    Args:
        bert_dir: Directory containing BERT embeddings
        output_dir: Output directory for TaBERT embeddings
    """
    bert_path = Path(bert_dir)
    output_path = Path(output_dir)

    for query_file in ["queries_train.npy", "queries_dev.npy"]:
        src = bert_path / query_file
        dst = output_path / query_file

        if src.exists() and not dst.exists():
            os.symlink(src.resolve(), dst)
            print(f"Created symlink: {dst} -> {src}")
        elif dst.exists():
            print(f"Skipping {query_file}: already exists")
        else:
            print(f"Warning: {src} not found")


def main():
    parser = argparse.ArgumentParser(
        description="Generate TaBERT embeddings for NQ-Tables"
    )

    # Required arguments
    parser.add_argument("--tables_json", type=str, required=True,
                        help="Path to NQ-Tables tables.json")
    parser.add_argument("--tabert_checkpoint", type=str, required=True,
                        help="Path to TaBERT checkpoint (model.bin)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for embeddings")

    # Optional arguments for query generation
    parser.add_argument("--train_json", type=str, default=None,
                        help="Path to train.json for query embeddings")
    parser.add_argument("--dev_json", type=str, default=None,
                        help="Path to dev.json for query embeddings")
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased",
                        help="BERT model for query embeddings")

    # Options
    parser.add_argument("--tables_only", action="store_true",
                        help="Only generate table embeddings")
    parser.add_argument("--link_bert_queries", type=str, default=None,
                        help="Link query embeddings from existing BERT dir instead of generating")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for encoding")
    parser.add_argument("--max_rows", type=int, default=50,
                        help="Maximum rows per table")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--embedding_mode", type=str, default="mean_column",
                        choices=["mean_column", "cls"],
                        help="How to create table embedding: mean_column (mean-pool columns, recommended) or cls (CLS token)")

    args = parser.parse_args()

    # Generate table embeddings
    print("=" * 60)
    print("GENERATING TABERT TABLE EMBEDDINGS")
    print("=" * 60)

    generate_tabert_table_embeddings(
        tables_json_path=args.tables_json,
        checkpoint_path=args.tabert_checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
        device=args.device,
        embedding_mode=args.embedding_mode
    )

    # Handle query embeddings
    if args.link_bert_queries:
        print("\n" + "=" * 60)
        print("LINKING BERT QUERY EMBEDDINGS")
        print("=" * 60)
        link_bert_queries(args.link_bert_queries, args.output_dir)

    elif not args.tables_only:
        print("\n" + "=" * 60)
        print("GENERATING BERT QUERY EMBEDDINGS")
        print("=" * 60)

        if args.train_json:
            train_output = os.path.join(args.output_dir, "queries_train.npy")
            generate_bert_query_embeddings(
                questions_json_path=args.train_json,
                bert_model=args.bert_model,
                output_path=train_output,
                batch_size=args.batch_size,
                device=args.device
            )

        if args.dev_json:
            dev_output = os.path.join(args.output_dir, "queries_dev.npy")
            generate_bert_query_embeddings(
                questions_json_path=args.dev_json,
                bert_model=args.bert_model,
                output_path=dev_output,
                batch_size=args.batch_size,
                device=args.device
            )

    print("\n" + "=" * 60)
    print("TABERT EMBEDDING GENERATION COMPLETE")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print("Files created:")
    for f in os.listdir(args.output_dir):
        fpath = os.path.join(args.output_dir, f)
        if os.path.islink(fpath):
            print(f"  {f} -> {os.readlink(fpath)}")
        else:
            size = os.path.getsize(fpath)
            print(f"  {f} ({size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
