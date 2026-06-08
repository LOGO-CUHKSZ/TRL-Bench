#!/usr/bin/env python
"""
OpenAI Row Embedding Generation via Batch API (50% cheaper).

Same serialization and output format as generate_row_embeddings.py, but
uses the asynchronous Batch API. Each row is serialized as
"col1: val1 | col2: val2 | ..." and submitted as a batch request.

Custom ID format: "{table_name}::row::{row_idx}"

Workflow:
    1. Scan CSV directory, serialize all rows → JSONL request files
    2. Upload files, create batches (50K requests / 150 MB per batch)
    3. Poll until all batches complete
    4. Download results, reassemble into per-table row embeddings

Usage:
    python generate_row_embeddings_batch.py \
        --input_dir /path/to/csvs/ --output_path emb.pkl

    # Submit only (don't poll)
    python generate_row_embeddings_batch.py \
        --input_dir /path/to/csvs/ --output_path emb.pkl --submit_only

    # Download completed results
    python generate_row_embeddings_batch.py \
        --input_dir /path/to/csvs/ --output_path emb.pkl --download_only
"""

import os
import sys
import json
import pickle
import argparse
import time
from pathlib import Path
from typing import List, Optional
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()

import tiktoken
import numpy as np
import pandas as pd
from tqdm import tqdm

OPENAI_MAX_TOKENS = 8191
_enc = None

def _get_encoder():
    global _enc
    if _enc is None:
        _enc = tiktoken.get_encoding('cl100k_base')
    return _enc

def truncate_text(text: str, max_tokens: int = OPENAI_MAX_TOKENS) -> str:
    enc = _get_encoder()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../" * 2))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trl_bench.utils.row_embedding.directory import (
    discover_csv_files,
    build_table_result,
    save_aggregate_pickle,
)

# Batch API limits
MAX_REQUESTS_PER_BATCH = 50_000
MAX_FILE_SIZE_BYTES = 150 * 1024 * 1024


def serialize_row(columns, row_values, max_chars_per_cell=100):
    pairs = []
    for col, val in zip(columns, row_values):
        if pd.isna(val):
            val_str = ""
        else:
            val_str = str(val)[:max_chars_per_cell]
        pairs.append(f"{col}: {val_str}")
    return " | ".join(pairs)


# =============================================================================
# State management
# =============================================================================

def load_state(state_path):
    if os.path.exists(state_path):
        with open(state_path, 'r') as f:
            return json.load(f)
    return {"batches": [], "phase": "init", "table_meta": {}}


def save_state(state, state_path):
    tmp = state_path + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, state_path)


# =============================================================================
# Phase 1: Prepare JSONL request files
# =============================================================================

def prepare_requests(
    csv_dir, model, dimensions, max_rows, max_chars_per_cell,
    output_dir, label_columns=None, table_list_path=None,
):
    csv_files = discover_csv_files(csv_dir, table_list_path=table_list_path)
    if not csv_files:
        raise ValueError(f"No CSV files found in {csv_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Track per-table metadata for reassembly
    table_meta = {}  # table_name -> {num_rows, column_names}

    chunks = []
    current_file = None
    current_path = None
    current_size = 0
    current_count = 0
    chunk_idx = 0
    label_set = set(label_columns) if label_columns else set()

    def _close_chunk():
        nonlocal current_file, current_path, current_size, current_count, chunk_idx
        if current_file:
            current_file.close()
            chunks.append({
                "jsonl_path": current_path,
                "num_requests": current_count,
            })
            chunk_idx += 1
            current_file = None
            current_path = None
            current_size = 0
            current_count = 0

    def _ensure_file():
        nonlocal current_file, current_path, current_size, current_count
        if current_file is None:
            current_path = os.path.join(output_dir, f"batch_requests_{chunk_idx:04d}.jsonl")
            current_file = open(current_path, 'w')
            current_size = 0
            current_count = 0

    total_rows = 0
    for csv_path in tqdm(csv_files, desc="Preparing row requests"):
        table_name = csv_path.stem

        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        except Exception:
            continue

        if len(df) < 1:
            continue

        feature_cols = [c for c in df.columns if c not in label_set]
        df_features = df[feature_cols]

        if max_rows is not None and len(df_features) > max_rows:
            df_features = df_features.iloc[:max_rows]

        column_names = list(df_features.columns)
        num_rows = len(df_features)
        table_meta[table_name] = {
            "num_rows": num_rows,
            "column_names": column_names,
        }

        for row_idx, (_, row) in enumerate(df_features.iterrows()):
            text = serialize_row(column_names, row, max_chars_per_cell)
            custom_id = f"{table_name}::row::{row_idx}"

            line = json.dumps({
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {
                    "model": model,
                    "input": truncate_text(text),
                    "dimensions": dimensions,
                },
            }) + "\n"
            line_bytes = len(line.encode('utf-8'))

            if current_file and (
                current_count >= MAX_REQUESTS_PER_BATCH
                or current_size + line_bytes > MAX_FILE_SIZE_BYTES
            ):
                _close_chunk()

            _ensure_file()
            current_file.write(line)
            current_size += line_bytes
            current_count += 1
            total_rows += 1

    _close_chunk()

    print(f"  {len(table_meta)} tables, {total_rows:,} rows -> {len(chunks)} JSONL files")
    return chunks, table_meta


# =============================================================================
# Phase 2: Upload and create batches
# =============================================================================

def submit_batches(chunks, state, state_path):
    from models.openai.client import create_client
    client, _ = create_client()

    existing_paths = {b.get("jsonl_path", "") for b in state["batches"]}

    for chunk in chunks:
        if chunk["jsonl_path"] in existing_paths:
            continue

        print(f"  Uploading {chunk['jsonl_path']} ({chunk['num_requests']} requests)...")
        file_obj = client.files.create(
            file=open(chunk["jsonl_path"], "rb"),
            purpose="batch",
        )

        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/embeddings",
            completion_window="24h",
        )

        state["batches"].append({
            "jsonl_path": chunk["jsonl_path"],
            "file_id": file_obj.id,
            "batch_id": batch.id,
            "status": batch.status,
            "num_requests": chunk["num_requests"],
            "output_file_id": None,
        })
        print(f"    Batch {batch.id} created ({batch.status})")
        save_state(state, state_path)

    state["phase"] = "polling"
    save_state(state, state_path)


# =============================================================================
# Phase 3: Poll for completion
# =============================================================================

def poll_batches(state, state_path, poll_interval=60):
    from models.openai.client import create_client
    client, _ = create_client()

    while True:
        all_done = True
        for entry in state["batches"]:
            if entry["status"] == "completed":
                continue
            if entry["status"] in ("failed", "expired", "cancelled"):
                if entry.get("retryable"):
                    all_done = False
                    try:
                        new_batch = client.batches.create(
                            input_file_id=entry["file_id"],
                            endpoint="/v1/embeddings",
                            completion_window="24h",
                        )
                        entry["batch_id"] = new_batch.id
                        entry["status"] = new_batch.status
                        entry["retryable"] = False
                    except Exception:
                        pass
                continue

            all_done = False
            batch = client.batches.retrieve(entry["batch_id"])
            entry["status"] = batch.status
            counts = batch.request_counts

            if batch.status == "completed":
                entry["output_file_id"] = batch.output_file_id
            elif batch.status == "failed":
                try:
                    if any((e.code or '') in ('request_limit_exceeded', 'token_limit_exceeded')
                           for e in batch.errors.data):
                        entry["retryable"] = True
                except Exception:
                    pass

        save_state(state, state_path)

        completed = sum(1 for b in state["batches"] if b["status"] == "completed")
        total = len(state["batches"])
        print(f"  Batches: {completed}/{total} completed")

        if all_done:
            break
        time.sleep(poll_interval)

    state["phase"] = "downloading"
    save_state(state, state_path)


# =============================================================================
# Phase 4: Download results and assemble pickle
# =============================================================================

def download_and_assemble(state, csv_dir, output_path, dimensions):
    from models.openai.client import create_client
    client, _ = create_client()

    # Download all embeddings keyed by custom_id
    all_embeddings = {}
    for entry in state["batches"]:
        if entry["status"] != "completed" or not entry["output_file_id"]:
            continue
        print(f"  Downloading batch {entry['batch_id'][:25]}...")
        content = client.files.content(entry["output_file_id"]).text
        for line in content.strip().split("\n"):
            r = json.loads(line)
            if r["response"]["status_code"] == 200:
                all_embeddings[r["custom_id"]] = np.array(
                    r["response"]["body"]["data"][0]["embedding"], dtype=np.float32)
        del content

    print(f"  Downloaded {len(all_embeddings):,} row embeddings")

    # Recover table names and row counts from custom_ids
    table_rows = defaultdict(dict)  # table_name -> {row_idx: embedding}
    for cid, emb in all_embeddings.items():
        parts = cid.split("::")
        table_name = parts[0]
        row_idx = int(parts[2])
        table_rows[table_name][row_idx] = emb

    # Get table metadata from state or reconstruct
    table_meta = state.get("table_meta", {})

    # Assemble per-table results
    results = []
    for table_name in tqdm(sorted(table_rows.keys()), desc="Assembling"):
        row_dict = table_rows[table_name]
        num_rows = max(row_dict.keys()) + 1

        # Stack row embeddings in order
        embeddings_list = []
        for i in range(num_rows):
            if i in row_dict:
                embeddings_list.append(row_dict[i])
            else:
                embeddings_list.append(np.zeros(dimensions, dtype=np.float32))

        row_embeddings = np.vstack(embeddings_list).astype(np.float32)

        # Get column names from metadata or CSV
        meta = table_meta.get(table_name, {})
        column_names = meta.get("column_names")
        if not column_names:
            # Try reading CSV header
            csv_files = discover_csv_files(csv_dir)
            for cp in csv_files:
                if cp.stem == table_name:
                    try:
                        df = pd.read_csv(cp, nrows=0, dtype=str)
                        column_names = [str(c) for c in df.columns]
                    except Exception:
                        column_names = []
                    break

        result = build_table_result(
            table_path=table_name,
            row_embeddings=row_embeddings,
            column_names=column_names or [],
            model_name="openai",
        )
        results.append(result)

    save_aggregate_pickle(results, output_path)
    print(f"  Assembled {len(results)} tables into {output_path}")
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate row embeddings via OpenAI Batch API"
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model", type=str, default="text-embedding-3-small")
    parser.add_argument("--dimensions", type=int, default=768)
    parser.add_argument("--max_chars_per_cell", type=int, default=100)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--label_columns", type=str, nargs="*", default=None)
    parser.add_argument("--table_list", default=None)
    parser.add_argument("--poll_interval", type=int, default=60)
    parser.add_argument("--submit_only", action="store_true")
    parser.add_argument("--download_only", action="store_true")
    parser.add_argument("--status_only", action="store_true")
    parser.add_argument("--work_dir", type=str, default=None)

    args = parser.parse_args()

    if not (os.environ.get('OPENAI_API_KEY') or os.environ.get('OPENROUTER_API_KEY')):
        raise RuntimeError("No API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY.")

    work_dir = args.work_dir or (args.output_path + ".batch_work")
    os.makedirs(work_dir, exist_ok=True)
    state_path = os.path.join(work_dir, "batch_state.json")
    state = load_state(state_path)

    if args.status_only:
        by_status = {}
        for b in state["batches"]:
            by_status[b["status"]] = by_status.get(b["status"], 0) + 1
        total = sum(b["num_requests"] for b in state["batches"])
        print(f"Phase: {state['phase']}, Batches: {len(state['batches'])}, "
              f"Requests: {total:,}, Status: {by_status}")
        return

    # Resolve model name for provider
    from models.openai.client import create_client, resolve_model_name
    _, provider = create_client()
    resolved_model = resolve_model_name(args.model, provider)

    # Phase 1: Prepare
    if state["phase"] == "init" and not args.download_only:
        print("Phase 1: Preparing JSONL request files")
        chunks, table_meta = prepare_requests(
            csv_dir=args.input_dir,
            model=resolved_model,
            dimensions=args.dimensions,
            max_rows=args.max_rows,
            max_chars_per_cell=args.max_chars_per_cell,
            output_dir=os.path.join(work_dir, "requests"),
            label_columns=args.label_columns,
            table_list_path=args.table_list,
        )
        state["phase"] = "submitting"
        state["table_meta"] = table_meta
        save_state(state, state_path)
    else:
        chunks = [{"jsonl_path": b.get("jsonl_path", ""), "num_requests": b["num_requests"]}
                  for b in state["batches"]]

    # Phase 2: Submit
    if state["phase"] == "submitting" and not args.download_only:
        print("Phase 2: Uploading and creating batches")
        submit_batches(chunks, state, state_path)

    if args.submit_only:
        total = sum(b["num_requests"] for b in state["batches"])
        print(f"\nSubmitted {len(state['batches'])} batches ({total:,} requests). Exiting.")
        return

    # Phase 3: Poll
    if state["phase"] == "polling" and not args.download_only:
        print("Phase 3: Polling for completion")
        poll_batches(state, state_path, poll_interval=args.poll_interval)

    # Phase 4: Download and assemble
    if state["phase"] in ("downloading", "done") or args.download_only:
        print("Phase 4: Downloading and assembling")
        results = download_and_assemble(
            state, args.input_dir, args.output_path, args.dimensions)
        state["phase"] = "done"
        save_state(state, state_path)


if __name__ == "__main__":
    main()
