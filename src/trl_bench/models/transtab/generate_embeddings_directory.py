"""
TransTab Directory-Mode Row Embedding Generation

Processes a directory of CSV files: trains a TransTab contrastive learner
per table (if no checkpoint exists), then extracts row embeddings. Produces
an aggregate pickle with one entry per table.

TransTab uses its own API (not ts3l) — handles raw DataFrames directly
with Vertical-Partition Contrastive Learning (VPCL).

Output format: List[dict] pickle at --output_path.
"""

import gc
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../" * 2))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import pickle

import numpy as np
import pandas as pd
import torch
import transtab
from sklearn.preprocessing import MinMaxScaler

from trl_bench.utils.row_embedding.directory import (
    discover_csv_files,
    clean_partial_checkpoint,
    build_table_result,
    save_aggregate_pickle,
    register_save_on_signal,
    load_existing_results,
    get_completed_table_ids,
    cleanup_checkpoints,
)

MODEL_NAME = "transtab"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate row embeddings for a directory of CSV files using TransTab"
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--checkpoint_base_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layer", type=int, default=2)
    parser.add_argument("--num_attention_head", type=int, default=8)
    parser.add_argument("--num_partition", type=int, default=3)
    parser.add_argument("--overlap_ratio", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--checkpoint_interval", type=int, default=50)
    parser.add_argument("--label_columns", type=str, nargs='*', default=None,
                        help="Label columns to exclude from features")
    parser.add_argument("--keep_checkpoints", action="store_true",
                        help="Keep per-table checkpoints after embedding (default: delete)")
    parser.add_argument("--table_list", default=None, help="Path to table list file for shard filtering")
    return parser.parse_args()


def detect_column_types(df):
    """Split DataFrame columns into categorical, numerical, and binary columns.

    All-NaN numeric columns are dropped — they carry no information and crash
    TransTab's binary tokenizer (``' '.join`` on float values).
    """
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if df[c].notna().any()]
    bin_cols = [c for c in num_cols if df[c].dropna().nunique() <= 2]
    num_cols = [c for c in num_cols if c not in bin_cols]
    return cat_cols, num_cols, bin_cols


def check_checkpoint_complete(checkpoint_dir):
    """Check whether a TransTab checkpoint is complete."""
    from pathlib import Path
    ckpt_dir = Path(checkpoint_dir)
    config_path = ckpt_dir / "training_config.pkl"
    ckpt_path = ckpt_dir / "ckpt_best.pt"

    if not config_path.exists() or not ckpt_path.exists():
        return False

    try:
        with open(config_path, "rb") as f:
            config = pickle.load(f)
        # Require num_scaler key — checkpoints without it are stale (pre-fix)
        if 'num_scaler' not in config:
            return False
        return True
    except Exception:
        return False


class _RetryingCollateFn:
    """Wrap TransTab's collate_fn to retry when a random vertical partition
    draws zero categorical columns, which crashes the BERT tokenizer.

    Delegates attribute access to the wrapped collate_fn so that TransTab's
    trainer can call methods like ``.save()`` on it.
    """

    def __init__(self, base_collate_fn, max_retries=64):
        self._base = base_collate_fn
        self.max_retries = max_retries

    def __call__(self, batch):
        last_exc = None
        for _ in range(self.max_retries):
            try:
                return self._base(batch)
            except IndexError as e:
                if "list index out of range" not in str(e):
                    raise
                last_exc = e
        raise RuntimeError(
            f"TransTab collate_fn failed after {self.max_retries} retries; "
            "could not sample a valid vertical partition."
        ) from last_exc

    def __getattr__(self, name):
        return getattr(self._base, name)


def _partition_schedule(requested, n_cat):
    """Return a list of num_partition values to try, largest first."""
    start = min(requested, n_cat) if n_cat > 0 else 1
    return list(range(start, 0, -1))


def train_on_table(df, checkpoint_dir, args):
    os.makedirs(checkpoint_dir, exist_ok=True)
    transtab.random_seed(args.random_seed)

    # Drop label columns to get feature-only DataFrame
    X = df.copy()
    if args.label_columns:
        cols_to_drop = [c for c in args.label_columns if c in X.columns]
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)

    cat_cols, num_cols, bin_cols = detect_column_types(X)

    # NaN/inf guard: replace inf with NaN, then fill with median
    numeric_fill_medians = {}
    for c in num_cols:
        X[c] = X[c].replace([np.inf, -np.inf], np.nan)
        median_val = X[c].median()
        numeric_fill_medians[c] = median_val
        X[c] = X[c].fillna(median_val).fillna(0)

    # MinMaxScaler on numerical columns (matches TransTab's load_data behavior)
    num_scaler = None
    if num_cols:
        num_scaler = MinMaxScaler()
        X[num_cols] = num_scaler.fit_transform(X[num_cols])

    for c in bin_cols:
        median_val = X[c].median()
        numeric_fill_medians[c] = median_val
        # Cast to int after fill — TransTab's tokenizer multiplies bin values
        # by an embedding matrix and requires integer indices, not floats.
        X[c] = X[c].fillna(median_val).fillna(0).astype(int)
    for c in cat_cols:
        X[c] = X[c].fillna("__MISSING__")

    # TransTab's CL collator concatenates labels via pd.concat, which crashes
    # on None.  Pass a dummy zero-filled Series so the collator is happy;
    # the contrastive loss ignores labels when supervised=False.
    dummy_y = pd.Series(0, index=X.index)

    # TransTab's contrastive collator subsamples rows within each mini-batch.
    # Tiny last batches (e.g., 1 row from 65 rows / batch_size=64) produce
    # empty DataFrames that crash the BERT tokenizer.  Use full-table batches
    # for small tables, cap at batch_size for large tables to avoid OOM.
    # Decrement in a loop until the last batch has more than 1 row.
    # If decrementing reaches 1, fall back to a single full-table batch.
    train_bs = min(len(X), args.batch_size)
    while train_bs > 1 and len(X) > train_bs and len(X) % train_bs == 1:
        train_bs -= 1
    if train_bs <= 1:
        train_bs = len(X)

    # Cap num_partition by n_cat to avoid partitions with zero categorical
    # columns. Wrap the collator to retry stochastic bad draws.
    effective_k = min(args.num_partition, len(cat_cols)) if cat_cols else 1

    used_num_partition = None
    for k in _partition_schedule(effective_k, len(cat_cols)):
        transtab.random_seed(args.random_seed)
        model, collate_fn = transtab.build_contrastive_learner(
            cat_cols, num_cols, bin_cols,
            supervised=False,
            num_partition=k,
            overlap_ratio=args.overlap_ratio,
            hidden_dim=args.hidden_dim,
            num_layer=args.num_layer,
            num_attention_head=args.num_attention_head,
        )
        safe_collate = _RetryingCollateFn(collate_fn)
        try:
            transtab.train(
                model,
                trainset=(X, dummy_y),
                collate_fn=safe_collate,
                num_epoch=args.num_epoch,
                batch_size=train_bs,
                lr=args.lr,
                output_dir=checkpoint_dir,
            )
            used_num_partition = k
            break
        except RuntimeError as e:
            if "could not sample a valid vertical partition" in str(e) and k > 1:
                print(f"  num_partition={k} exhausted retries, trying {k - 1}")
                continue
            raise

    if used_num_partition is None:
        raise RuntimeError("TransTab training failed at all partition counts")

    if used_num_partition != args.num_partition:
        print(f"  Used num_partition={used_num_partition} (requested {args.num_partition})")

    # Save training config (written last as completeness marker)
    config_dict = {
        "model_name": "TransTab",
        "training_mode": "contrastive",
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "bin_cols": bin_cols,
        "hidden_dim": args.hidden_dim,
        "num_layer": args.num_layer,
        "num_attention_head": args.num_attention_head,
        "num_partition": used_num_partition,
        "requested_num_partition": args.num_partition,
        "overlap_ratio": args.overlap_ratio,
        "num_epoch": args.num_epoch,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "feature_columns": list(X.columns),
        "numeric_fill_medians": numeric_fill_medians,
        "cat_fill_token": "__MISSING__",
        "num_scaler": num_scaler,
        "label_columns": args.label_columns,
    }
    with open(os.path.join(checkpoint_dir, "training_config.pkl"), "wb") as f:
        pickle.dump(config_dict, f)

    del model
    torch.cuda.empty_cache()
    gc.collect()


def embed_from_checkpoint(df, checkpoint_dir, batch_size, label_columns=None):
    config_path = os.path.join(checkpoint_dir, "training_config.pkl")
    with open(config_path, "rb") as f:
        config = pickle.load(f)

    cat_cols = config["cat_cols"]
    num_cols = config["num_cols"]
    bin_cols = config["bin_cols"]
    numeric_fill_medians = config.get("numeric_fill_medians", {})
    cat_fill_token = config.get("cat_fill_token", "__MISSING__")
    num_scaler = config.get("num_scaler", None)

    X = df.copy()
    if label_columns:
        cols_to_drop = [c for c in label_columns if c in X.columns]
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)

    # Enforce saved feature order (strict — same as generate_embeddings.py)
    saved_features = config.get("feature_columns")
    if saved_features:
        missing = [c for c in saved_features if c not in X.columns]
        if missing:
            raise ValueError(
                f"Missing {len(missing)} training feature column(s): {missing[:10]}"
            )
        X = X[saved_features]

    # Apply same NaN/inf guard as training
    for c in num_cols:
        if c in X.columns:
            X[c] = X[c].replace([np.inf, -np.inf], np.nan)
            fill_val = numeric_fill_medians.get(c, X[c].median())
            X[c] = X[c].fillna(fill_val).fillna(0)
    if num_scaler is not None and num_cols:
        cols_present = [c for c in num_cols if c in X.columns]
        if cols_present:
            X[cols_present] = num_scaler.transform(X[cols_present])
    for c in bin_cols:
        if c in X.columns:
            fill_val = numeric_fill_medians.get(c, X[c].median())
            X[c] = X[c].fillna(fill_val).fillna(0).astype(int)
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].fillna(cat_fill_token)

    # Build encoder from checkpoint
    enc = transtab.build_encoder(
        categorical_columns=cat_cols,
        numerical_columns=num_cols,
        binary_columns=bin_cols,
        hidden_dim=config["hidden_dim"],
        num_layer=config["num_layer"],
        checkpoint=checkpoint_dir,
    )

    # Batch inference
    embeddings = []
    n = len(X)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_df = X.iloc[start:end]
        with torch.no_grad():
            batch_emb = enc(batch_df)
        embeddings.append(batch_emb.cpu().numpy())

    del enc
    torch.cuda.empty_cache()
    return np.vstack(embeddings)


def process_table(csv_path, args):
    table_stem = csv_path.stem
    checkpoint_dir = os.path.join(args.checkpoint_base_dir, table_stem)

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  SKIP {csv_path.name}: cannot read CSV: {e}")
        return None

    if len(df) < 4:
        print(f"  SKIP {csv_path.name}: too few rows ({len(df)})")
        return None

    column_names = list(df.columns)
    label_set = set(args.label_columns) if args.label_columns else set()
    feature_cols = [c for c in column_names if c not in label_set] if label_set else column_names

    if not check_checkpoint_complete(checkpoint_dir):
        clean_partial_checkpoint(checkpoint_dir)
        print(f"  Training TransTab on {len(df)} rows...")
        try:
            train_on_table(df, checkpoint_dir, args)
            print(f"  Training complete -> {checkpoint_dir}")
        except Exception as e:
            print(f"  SKIP {csv_path.name}: training failed: {e}")
            clean_partial_checkpoint(checkpoint_dir)
            return None
    else:
        print(f"  Checkpoint exists, skipping training")

    try:
        embeddings = embed_from_checkpoint(df, checkpoint_dir, args.batch_size, label_columns=args.label_columns)
    except Exception as e:
        print(f"  SKIP {csv_path.name}: embedding failed: {e}")
        return None

    # Clean up per-table checkpoint immediately after embedding
    if not args.keep_checkpoints:
        clean_partial_checkpoint(checkpoint_dir)

    return build_table_result(str(csv_path), embeddings, feature_cols, "TransTab")


def main():
    import psutil
    _proc = psutil.Process()
    def _mem():
        return f"RSS={_proc.memory_info().rss/1024**2:.0f}MB"

    args = parse_args()
    print("=" * 80)
    print(f"TransTab Directory-Mode Row Embedding Generation  PID={os.getpid()} {_mem()}")
    print("=" * 80)

    csv_files = discover_csv_files(args.input_dir, table_list_path=args.table_list)
    print(f"Found {len(csv_files)} CSV files in {args.input_dir}  {_mem()}")
    if not csv_files:
        sys.exit(0)

    results = load_existing_results(args.output_path)
    completed = get_completed_table_ids(results)
    register_save_on_signal(results, args.output_path)
    if completed:
        print(f"Resuming: {len(completed)} tables already processed  {_mem()}")

    newly_processed = 0
    for i, csv_path in enumerate(csv_files):
        if csv_path.stem in completed:
            continue
        print(f"\n[{i + 1}/{len(csv_files)}] Processing {csv_path.name}...  {_mem()}")
        result = process_table(csv_path, args)
        if result is not None:
            results.append(result)
            newly_processed += 1
            print(f"  Embedded: {result['num_rows']} rows x {result['embedding_dim']} dim  {_mem()}")
        if newly_processed > 0 and newly_processed % args.checkpoint_interval == 0:
            save_aggregate_pickle(results, args.output_path)
            print(f"  Checkpoint saved ({len(results)} tables total)")

    if newly_processed > 0:
        save_aggregate_pickle(results, args.output_path)

    if not args.keep_checkpoints and args.checkpoint_base_dir:
        processed_ids = [r['table_id'] for r in results]
        cleanup_checkpoints(args.checkpoint_base_dir, table_ids=processed_ids)

    print(f"\n{'=' * 80}")
    print(f"Done. {len(results)} tables in {args.output_path}")
    print(f"  Newly processed: {newly_processed}, Previously completed: {len(completed)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
