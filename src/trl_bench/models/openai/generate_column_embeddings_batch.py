#!/usr/bin/env python
"""
Generate column embeddings using OpenAI Batch API (50% cheaper).

Same serialization and output format as generate_column_embeddings.py, but
uses the asynchronous Batch API instead of synchronous calls. Requests are
bundled into JSONL files, uploaded to OpenAI, and results are polled and
downloaded automatically.

Workflow:
    1. Scan CSV directory, serialize all texts → JSONL request files
    2. Upload files, create batches (50K requests / 150 MB per batch)
    3. Poll until all batches complete (up to 24h)
    4. Download results, assemble into unified v2.0 pickle

Resume support: the script writes a state file (.batch_state.json) tracking
which batches have been created/completed. Re-running the same command
resumes from where it left off.

Usage:
    python generate_column_embeddings_batch.py \
        --input /path/to/csvs/ --output emb.pkl --max_rows 100

    # Check status of a previous run without resubmitting
    python generate_column_embeddings_batch.py \
        --input /path/to/csvs/ --output emb.pkl --status_only
"""

import os
import sys
import json
import pickle
import argparse
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional

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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from trl_bench.utils.aggregation import aggregate_embeddings

# Batch API limits
MAX_REQUESTS_PER_BATCH = 50_000
MAX_FILE_SIZE_BYTES = 150 * 1024 * 1024  # 150 MB (conservative, limit is 200 MB)


# =============================================================================
# Table serialization (same as GTE)
# =============================================================================

def linearize_table(df: pd.DataFrame) -> str:
    rows = [' | '.join(str(c) for c in df.columns)]
    for _, row in df.iterrows():
        rows.append(' | '.join(str(v) for v in row))
    return ' . '.join(rows)


def serialize_column(col_name: str, values: pd.Series) -> str:
    vals = ', '.join(str(v) for v in values)
    return f"{col_name}: {vals}"


# =============================================================================
# State management
# =============================================================================

def load_state(state_path: str) -> dict:
    if os.path.exists(state_path):
        with open(state_path, 'r') as f:
            return json.load(f)
    return {"batches": [], "phase": "init"}


def save_state(state: dict, state_path: str):
    tmp = state_path + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, state_path)


# =============================================================================
# Phase 1: Prepare JSONL request files
# =============================================================================

def prepare_requests(
    csv_dir: str,
    model: str,
    dimensions: int,
    max_rows: int,
    output_dir: str,
    table_list: Optional[set] = None,
) -> List[dict]:
    """Serialize all tables into JSONL batch request files.

    Returns (chunks, table_meta) where chunks is a list of
    {"jsonl_path": ..., "num_requests": ..., "table_names": [...]}
    and table_meta maps table_name -> column_names list.
    """
    csv_files = sorted([f for f in os.listdir(csv_dir) if f.endswith('.csv')])
    if table_list is not None:
        csv_files = [f for f in csv_files if f in table_list]
    if not csv_files:
        raise ValueError(f"No CSV files found in {csv_dir}")

    os.makedirs(output_dir, exist_ok=True)
    table_meta = {}  # table_name -> column_names

    chunks = []
    current_file = None
    current_path = None
    current_size = 0
    current_count = 0
    current_tables = []
    chunk_idx = 0

    def _close_chunk():
        nonlocal current_file, current_path, current_size, current_count, current_tables, chunk_idx
        if current_file:
            current_file.close()
            chunks.append({
                "jsonl_path": current_path,
                "num_requests": current_count,
                "table_names": current_tables,
            })
            chunk_idx += 1
            current_file = None
            current_path = None
            current_size = 0
            current_count = 0
            current_tables = []

    def _ensure_file():
        nonlocal current_file, current_path, current_size, current_count, current_tables
        if current_file is None:
            current_path = os.path.join(output_dir, f"batch_requests_{chunk_idx:04d}.jsonl")
            current_file = open(current_path, 'w')
            current_size = 0
            current_count = 0
            current_tables = []

    for csv_file in tqdm(csv_files, desc="Preparing requests"):
        csv_path = os.path.join(csv_dir, csv_file)
        table_name = os.path.splitext(csv_file)[0]

        try:
            df = pd.read_csv(csv_path, nrows=max_rows, dtype=str)
        except Exception:
            try:
                df = pd.read_csv(csv_path, nrows=max_rows, engine='python', dtype=str)
            except Exception:
                print(f"  SKIP {csv_file}: cannot parse CSV")
                continue

        df = df.head(max_rows).fillna('').astype(str)
        df.columns = [str(c) for c in df.columns]
        column_names = list(df.columns)

        table_meta[table_name] = column_names

        # Build requests for this table:
        # 1 table-level + N column-level
        texts = {
            f"{table_name}::table_cls": linearize_table(df),
        }
        for i, col in enumerate(column_names):
            texts[f"{table_name}::col::{i}"] = serialize_column(col, df[col])

        # Write as JSONL lines, splitting if limits exceeded
        for custom_id, text in texts.items():
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

            # Check if we need a new chunk
            if current_file and (
                current_count >= MAX_REQUESTS_PER_BATCH
                or current_size + line_bytes > MAX_FILE_SIZE_BYTES
            ):
                _close_chunk()

            _ensure_file()
            current_file.write(line)
            current_size += line_bytes
            current_count += 1

        current_tables.append(table_name)

    _close_chunk()
    return chunks, table_meta


# =============================================================================
# Phase 2: Upload and create batches
# =============================================================================

def submit_batches(chunks: List[dict], state: dict, state_path: str):
    """Upload JSONL files and create batches. Updates state in-place."""
    from models.openai.client import create_client
    client, _ = create_client()

    existing_paths = {b["jsonl_path"] for b in state["batches"]}

    for chunk in chunks:
        if chunk["jsonl_path"] in existing_paths:
            continue

        print(f"Uploading {chunk['jsonl_path']} ({chunk['num_requests']} requests)...")
        file_obj = client.files.create(
            file=open(chunk["jsonl_path"], "rb"),
            purpose="batch",
        )

        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/embeddings",
            completion_window="24h",
            metadata={"source": chunk["jsonl_path"]},
        )

        state["batches"].append({
            "jsonl_path": chunk["jsonl_path"],
            "file_id": file_obj.id,
            "batch_id": batch.id,
            "status": batch.status,
            "num_requests": chunk["num_requests"],
            "table_names": chunk["table_names"],
            "output_file_id": None,
        })
        print(f"  Batch {batch.id} created (status: {batch.status})")

        save_state(state, state_path)

    state["phase"] = "polling"
    save_state(state, state_path)


# =============================================================================
# Phase 3: Poll for completion
# =============================================================================

def poll_batches(state: dict, state_path: str, poll_interval: int = 30):
    """Poll all batches until completion. Updates state in-place.

    Batches that failed with ``request_limit_exceeded`` are automatically
    resubmitted once queue capacity frees up.
    """
    from models.openai.client import create_client
    client, _ = create_client()

    while True:
        all_done = True
        any_completed_this_round = False

        for entry in state["batches"]:
            if entry["status"] == "completed":
                continue

            # Resubmit failed-due-to-queue-limit batches when capacity frees up
            if entry["status"] == "failed" and entry.get("retryable"):
                if any_completed_this_round:
                    print(f"  Resubmitting {entry['batch_id']} (queue capacity freed)...")
                    try:
                        new_batch = client.batches.create(
                            input_file_id=entry["file_id"],
                            endpoint="/v1/embeddings",
                            completion_window="24h",
                        )
                        entry["batch_id"] = new_batch.id
                        entry["status"] = new_batch.status
                        entry["retryable"] = False
                        print(f"    New batch: {new_batch.id} ({new_batch.status})")
                    except Exception as e:
                        print(f"    Resubmit failed: {e}")
                all_done = False
                continue

            if entry["status"] in ("expired", "cancelled"):
                continue

            all_done = False
            batch = client.batches.retrieve(entry["batch_id"])
            entry["status"] = batch.status
            counts = batch.request_counts

            print(
                f"  Batch {entry['batch_id']}: {batch.status} "
                f"({counts.completed}/{counts.total} done, {counts.failed} failed)"
            )

            if batch.status == "completed":
                entry["output_file_id"] = batch.output_file_id
                any_completed_this_round = True
            elif batch.status == "failed":
                # Check if it's a retryable queue-limit failure
                try:
                    errors = batch.errors
                    if errors and any(
                                      (e.code or '') in ('request_limit_exceeded', 'token_limit_exceeded')
                                      for e in errors.data):
                        entry["retryable"] = True
                        print(f"    (will retry when queue capacity frees up)")
                    else:
                        print(f"  WARNING: Batch {entry['batch_id']} failed permanently")
                except Exception:
                    print(f"  WARNING: Batch {entry['batch_id']} failed")

        save_state(state, state_path)

        if all_done:
            break

        time.sleep(poll_interval)

    state["phase"] = "downloading"
    save_state(state, state_path)


# =============================================================================
# Phase 4: Download results and assemble pickle
# =============================================================================

def download_and_assemble(
    state: dict,
    csv_dir: str,
    max_rows: int,
    model_name: str,
    dimensions: int,
    output_path: str,
):
    """Download batch results and assemble into unified v2.0 pickle."""
    from models.openai.client import create_client
    client, _ = create_client()

    # Collect all embeddings keyed by custom_id
    all_embeddings = {}
    total_failed = 0

    for entry in state["batches"]:
        if entry["status"] != "completed" or not entry["output_file_id"]:
            print(f"  Skipping batch {entry['batch_id']} (status: {entry['status']})")
            continue

        print(f"Downloading results from batch {entry['batch_id']}...")
        content = client.files.content(entry["output_file_id"]).text

        for line in content.strip().split("\n"):
            result = json.loads(line)
            custom_id = result["custom_id"]

            if result["response"]["status_code"] == 200:
                vector = result["response"]["body"]["data"][0]["embedding"]
                all_embeddings[custom_id] = np.array(vector, dtype=np.float32)
            else:
                total_failed += 1

    if total_failed > 0:
        print(f"WARNING: {total_failed} requests failed")

    print(f"Downloaded {len(all_embeddings)} embeddings total")

    # Reconstruct per-table results
    # Gather table names — prefer state, fall back to custom_id keys
    table_names = []
    for entry in state["batches"]:
        table_names.extend(entry.get("table_names", []))
    if not table_names:
        # Recover from custom_ids: "{table_name}::table_cls"
        table_names = sorted({
            k.split("::")[0] for k in all_embeddings if "::table_cls" in k
        })

    # Use cached table_meta from state (avoids re-reading CSVs)
    table_meta = state.get("table_meta", {})

    results = []
    for table_name in tqdm(table_names, desc="Assembling results"):
        table_cls_key = f"{table_name}::table_cls"
        if table_cls_key not in all_embeddings:
            print(f"  SKIP {table_name}: missing table embedding")
            continue

        # Get column names from cached metadata, fall back to CSV
        column_names = table_meta.get(table_name)
        if not column_names:
            csv_path = os.path.join(csv_dir, f"{table_name}.csv")
            try:
                df = pd.read_csv(csv_path, nrows=0, dtype=str)
                column_names = [str(c) for c in df.columns]
            except Exception:
                print(f"  SKIP {table_name}: cannot resolve column names")
                continue

        table_cls = all_embeddings[table_cls_key]

        col_embeddings = {}
        missing_cols = False
        for i in range(len(column_names)):
            col_key = f"{table_name}::col::{i}"
            if col_key not in all_embeddings:
                missing_cols = True
                break
            col_embeddings[i] = all_embeddings[col_key]

        if missing_cols:
            print(f"  SKIP {table_name}: missing column embeddings")
            continue

        table_embedding = {
            'cls_embedding': table_cls,
            'table_embedding': None,
            'column_mean': aggregate_embeddings(col_embeddings, 'mean'),
            'token_mean': None,
        }

        csv_path = os.path.join(csv_dir, f"{table_name}.csv")
        results.append({
            'version': '2.0',
            'format': 'unified_table_embedding',
            'table_id': table_name,
            'table': os.path.abspath(csv_path),
            'table_embedding': table_embedding,
            'column_embeddings': col_embeddings,
            'column_names': column_names,
            'table_name': table_name,
            'model_name': model_name,
            'embedding_dim': dimensions,
        })

    with open(output_path, 'wb') as f:
        pickle.dump(results, f, protocol=4)

    print(f"\nAssembled {len(results)} tables into {output_path}")
    return results


# =============================================================================
# Status display
# =============================================================================

def show_status(state: dict):
    """Print current batch status."""
    if not state["batches"]:
        print("No batches found.")
        return

    total_requests = sum(b["num_requests"] for b in state["batches"])
    by_status = {}
    for b in state["batches"]:
        by_status.setdefault(b["status"], []).append(b)

    print(f"\nPhase: {state['phase']}")
    print(f"Batches: {len(state['batches'])}  ({total_requests:,} total requests)")
    for status, batches in sorted(by_status.items()):
        reqs = sum(b["num_requests"] for b in batches)
        print(f"  {status}: {len(batches)} batches ({reqs:,} requests)")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate column embeddings using OpenAI Batch API (50% cheaper)'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Directory of CSV files')
    parser.add_argument('--output', type=str, required=True,
                        help='Output pickle file')
    parser.add_argument('--model', type=str, default='text-embedding-3-small',
                        help='OpenAI model name (default: text-embedding-3-small)')
    parser.add_argument('--max_rows', type=int, default=100,
                        help='Maximum rows to load from CSV (default: 100)')
    parser.add_argument('--dimensions', type=int, default=768,
                        help='Embedding dimensions (default: 768, matching GTE-base)')
    parser.add_argument('--poll_interval', type=int, default=60,
                        help='Seconds between status polls (default: 60)')
    parser.add_argument('--table_list', type=str, default=None,
                        help='Path to file listing CSV basenames to process')
    parser.add_argument('--status_only', action='store_true',
                        help='Just check batch status, do not submit or download')
    parser.add_argument('--submit_only', action='store_true',
                        help='Prepare and submit batches, then exit without polling')
    parser.add_argument('--download_only', action='store_true',
                        help='Skip preparation/submission, just download completed results')
    parser.add_argument('--work_dir', type=str, default=None,
                        help='Working directory for JSONL files and state '
                             '(default: <output>.batch_work/)')

    args = parser.parse_args()

    if not (os.environ.get('OPENAI_API_KEY') or os.environ.get('OPENROUTER_API_KEY')):
        raise RuntimeError(
            "No API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY in environment or .env file."
        )

    if not os.path.isdir(args.input):
        raise ValueError(f"--input must be a directory, got: {args.input}")

    work_dir = args.work_dir or (args.output + ".batch_work")
    os.makedirs(work_dir, exist_ok=True)
    state_path = os.path.join(work_dir, "batch_state.json")
    state = load_state(state_path)

    table_list = None
    if args.table_list:
        from trl_bench.utils.table_list import load_table_list
        table_list = load_table_list(args.table_list)

    # Status only
    if args.status_only:
        show_status(state)
        return

    # Resolve model name for provider
    from models.openai.client import resolve_model_name
    _, provider = create_client()
    resolved_model = resolve_model_name(args.model, provider)

    # Phase 1: Prepare
    if state["phase"] in ("init",) and not args.download_only:
        print(f"\n{'=' * 60}")
        print("PHASE 1: Preparing JSONL request files")
        print(f"{'=' * 60}")
        chunks, table_meta = prepare_requests(
            csv_dir=args.input,
            model=resolved_model,
            dimensions=args.dimensions,
            max_rows=args.max_rows,
            output_dir=os.path.join(work_dir, "requests"),
            table_list=table_list,
        )
        total_reqs = sum(c["num_requests"] for c in chunks)
        print(f"Prepared {len(chunks)} JSONL files ({total_reqs:,} requests)")

        state["phase"] = "submitting"
        state["chunks"] = [c["jsonl_path"] for c in chunks]
        state["table_meta"] = table_meta
        save_state(state, state_path)
    else:
        # Reconstruct chunks from state for submission
        chunks = []
        for b in state["batches"]:
            chunks.append({
                "jsonl_path": b.get("jsonl_path", ""),
                "num_requests": b["num_requests"],
                "table_names": b.get("table_names", []),
            })
        # Also check for un-submitted chunks
        req_dir = os.path.join(work_dir, "requests")
        if os.path.isdir(req_dir):
            submitted = {b.get("jsonl_path", "") for b in state["batches"]}
            for f in sorted(os.listdir(req_dir)):
                fp = os.path.join(req_dir, f)
                if fp not in submitted and f.endswith('.jsonl'):
                    # Count lines
                    with open(fp) as fh:
                        lines = fh.readlines()
                    table_names = list({
                        json.loads(l)["custom_id"].split("::")[0]
                        for l in lines
                    })
                    chunks.append({
                        "jsonl_path": fp,
                        "num_requests": len(lines),
                        "table_names": table_names,
                    })

    # Phase 2: Submit
    if state["phase"] in ("submitting",) and not args.download_only:
        print(f"\n{'=' * 60}")
        print("PHASE 2: Uploading and creating batches")
        print(f"{'=' * 60}")
        submit_batches(chunks, state, state_path)

    if args.submit_only:
        total_reqs = sum(b["num_requests"] for b in state["batches"])
        print(f"\nSubmitted {len(state['batches'])} batches ({total_reqs:,} requests). Exiting.")
        return

    # Phase 3: Poll
    if state["phase"] in ("polling",) and not args.download_only:
        print(f"\n{'=' * 60}")
        print("PHASE 3: Polling for completion")
        print(f"{'=' * 60}")
        poll_batches(state, state_path, poll_interval=args.poll_interval)

    # Phase 4: Download and assemble
    if state["phase"] in ("downloading", "done") or args.download_only:
        print(f"\n{'=' * 60}")
        print("PHASE 4: Downloading results and assembling pickle")
        print(f"{'=' * 60}")
        results = download_and_assemble(
            state=state,
            csv_dir=args.input,
            max_rows=args.max_rows,
            model_name=args.model,
            dimensions=args.dimensions,
            output_path=args.output,
        )

        state["phase"] = "done"
        save_state(state, state_path)

        print(f"\n{'=' * 60}")
        print("BATCH EMBEDDING EXTRACTION COMPLETE")
        print(f"{'=' * 60}")
        print(f"Model: {args.model}")
        print(f"Dimensions: {args.dimensions}")
        print(f"Tables: {len(results)}")
        print(f"Output: {args.output}")
        print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
