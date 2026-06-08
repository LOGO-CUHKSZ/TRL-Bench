"""
VIME Directory-Mode Row Embedding Generation

Processes a directory of CSV files: trains a VIME model per table
(if no checkpoint exists), then extracts row embeddings. Produces
an aggregate pickle with one entry per table.

Output format: List[dict] pickle at --output_path.
"""

import gc
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../" * 2))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, SequentialSampler

from trl_bench.utils.ts3l.pl_modules import VIMELightning
from trl_bench.utils.ts3l.utils.vime_utils import VIMEDataset, VIMEConfig
from trl_bench.utils.ts3l.utils.embedding_utils import IdentityEmbeddingConfig
from trl_bench.utils.ts3l.utils.backbone_utils import MLPBackboneConfig
from trl_bench.utils.row_embedding.directory import (
    discover_csv_files,
    check_checkpoint_complete,
    clean_partial_checkpoint,
    build_table_result,
    save_aggregate_pickle,
    register_save_on_signal,
    load_existing_results,
    get_completed_table_ids,
    cleanup_checkpoints,
    preprocess_table,
    train_raw_loop,
    save_model_checkpoint,
    load_model_from_checkpoint,
)

MODEL_NAME = "vime"
CKPT_FILENAME = "vime_self_supervised.ckpt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate row embeddings for a directory of CSV files using VIME"
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--checkpoint_base_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--phase1_epochs", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--n_hidden_layers", type=int, default=3)
    parser.add_argument("--p_m", type=float, default=0.3, help="Mask probability (default: 0.3)")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--checkpoint_interval", type=int, default=50)
    parser.add_argument("--label_columns", type=str, nargs='*', default=None,
                        help="Label columns to exclude from features")
    parser.add_argument("--table_list", default=None, help="Path to table list file for shard filtering")
    parser.add_argument("--keep_checkpoints", action="store_true",
                        help="Keep per-table checkpoints after embedding (default: delete)")
    return parser.parse_args()


def make_config(input_dim, cat_cardinalities, n_continuous, args):
    embedding_config = IdentityEmbeddingConfig(input_dim=input_dim)
    backbone_config = MLPBackboneConfig(
        input_dim=embedding_config.output_dim,
        hidden_dims=args.hidden_dim,
        n_hiddens=args.n_hidden_layers,
    )
    return VIMEConfig(
        task="classification",
        loss_fn="CrossEntropyLoss",
        metric="accuracy_score",
        metric_hparams={},
        embedding_config=embedding_config,
        backbone_config=backbone_config,
        output_dim=2,
        p_m=args.p_m,
        cat_cardinality=cat_cardinalities,
        num_continuous=n_continuous,
    )


def train_on_table(df, checkpoint_dir, args):
    os.makedirs(checkpoint_dir, exist_ok=True)

    X_encoded, encoders, scaler, category_cols, continuous_cols = preprocess_table(
        df, label_columns=args.label_columns
    )
    cat_cardinalities = [len(encoders[c].classes_) for c in category_cols]
    input_dim = X_encoded.shape[1]

    config = make_config(input_dim, cat_cardinalities, len(continuous_cols), args)

    pl_model = VIMELightning(config)
    pl_model.set_first_phase()

    dataset = VIMEDataset(
        X=X_encoded, unlabeled_data=None, config=config,
        continuous_cols=continuous_cols, category_cols=category_cols,
    )
    # Cap training batch size to avoid OOM on large tables
    train_bs = min(len(dataset), args.batch_size)
    # Avoid last batch of size 1 (BatchNorm requires ≥2 samples)
    if len(dataset) > train_bs and len(dataset) % train_bs == 1:
        train_bs -= 1
    dataloader = DataLoader(
        dataset, batch_size=train_bs, shuffle=False, num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_raw_loop(pl_model, dataloader, max_epochs=args.phase1_epochs, device=device)

    save_model_checkpoint(pl_model, {
        "model_name": "VIME", "training_mode": "self_supervised",
        "category_cols": category_cols, "continuous_cols": continuous_cols,
        "cat_cardinalities": cat_cardinalities, "input_dim": input_dim,
        "hidden_dim": args.hidden_dim, "n_hidden_layers": args.n_hidden_layers,
        "categorical_encoders": encoders, "scaler": scaler,
        "label_columns": args.label_columns,
        "model_config": config,
    }, checkpoint_dir, CKPT_FILENAME)

    del pl_model
    torch.cuda.empty_cache()
    gc.collect()


def embed_from_checkpoint(df, checkpoint_dir, batch_size, label_columns=None):
    import pickle
    config_path = os.path.join(checkpoint_dir, "training_config.pkl")
    with open(config_path, "rb") as f:
        train_config = pickle.load(f)

    category_cols = train_config["category_cols"]
    continuous_cols = train_config["continuous_cols"]
    encoders = train_config["categorical_encoders"]
    scaler_obj = train_config["scaler"]

    X = df.copy()
    if label_columns:
        cols_to_drop = [c for c in label_columns if c in X.columns]
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)
    for col in category_cols:
        X[col] = encoders[col].transform(X[col].astype(str))
    if continuous_cols:
        X[continuous_cols] = X[continuous_cols].replace([np.inf, -np.inf], np.nan)
        medians = X[continuous_cols].median()
        X[continuous_cols] = X[continuous_cols].fillna(medians).fillna(0)
        X[continuous_cols] = scaler_obj.transform(X[continuous_cols])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pl_model = load_model_from_checkpoint(
        VIMELightning, train_config["model_config"],
        checkpoint_dir, CKPT_FILENAME, device=device,
    )

    dataset = VIMEDataset(
        X=X, continuous_cols=continuous_cols, category_cols=category_cols, is_second_phase=True,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        sampler=SequentialSampler(dataset), num_workers=0,
    )

    embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)
            x_emb = pl_model.model.embedding_module(x)
            x_enc = pl_model.model.encoder(x_emb)
            embeddings.append(x_enc.cpu().numpy())

    del pl_model
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

    if not check_checkpoint_complete(checkpoint_dir, MODEL_NAME):
        clean_partial_checkpoint(checkpoint_dir)
        print(f"  Training VIME on {len(df)} rows...")
        try:
            train_on_table(df, checkpoint_dir, args)
            print(f"  Training complete → {checkpoint_dir}")
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

    # Clean up per-table checkpoint immediately after embedding to avoid
    # large checkpoint accumulation on long-running shards.
    if not args.keep_checkpoints:
        clean_partial_checkpoint(checkpoint_dir)

    return build_table_result(str(csv_path), embeddings, feature_cols, "VIME")


def main():
    args = parse_args()
    print("=" * 80)
    print("VIME Directory-Mode Row Embedding Generation")
    print("=" * 80)

    csv_files = discover_csv_files(args.input_dir, table_list_path=args.table_list)
    print(f"Found {len(csv_files)} CSV files in {args.input_dir}")
    if not csv_files:
        sys.exit(0)

    results = load_existing_results(args.output_path)
    completed = get_completed_table_ids(results)
    register_save_on_signal(results, args.output_path)
    if completed:
        print(f"Resuming: {len(completed)} tables already processed")

    newly_processed = 0
    for i, csv_path in enumerate(csv_files):
        if csv_path.stem in completed:
            continue
        print(f"\n[{i + 1}/{len(csv_files)}] Processing {csv_path.name}...")
        result = process_table(csv_path, args)
        if result is not None:
            results.append(result)
            newly_processed += 1
            print(f"  Embedded: {result['num_rows']} rows x {result['embedding_dim']} dim")
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
