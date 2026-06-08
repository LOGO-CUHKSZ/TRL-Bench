"""
SAINT Directory-Mode Row Embedding Generation

Processes a directory of CSV files: trains a SAINT model per table
(if no checkpoint exists), then extracts row embeddings. Produces
an aggregate pickle with one entry per table.

Output format: List[dict] pickle at --output_path.

NOTE: SAINT (full) and SAINT-i variants use intersample attention,
making row embeddings batch-dependent. Embeddings are reproducible
for the same batch_size and row ordering, but will differ if either
changes. Use --saint_variant saint_s for batch-independent embeddings.
"""

import gc
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../" * 2))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, SequentialSampler

from trl_bench.utils.ts3l.pl_modules import SAINTLightning
from trl_bench.utils.ts3l.utils.saint_utils import SAINTDataset, SAINTConfig
from trl_bench.utils.ts3l.utils.embedding_utils import FTEmbeddingConfig
from trl_bench.utils.ts3l.utils.backbone_utils import SAINTBackboneConfig
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

logger = logging.getLogger(__name__)

MODEL_NAME = "saint"
CKPT_FILENAME = "saint_self_supervised.ckpt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate row embeddings for a directory of CSV files using SAINT"
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--checkpoint_base_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--phase1_epochs", type=int, default=20)
    parser.add_argument("--emb_dim", type=int, default=512,
                        help="Per-feature embedding dimension (default: 512)")
    parser.add_argument("--encoder_depth", type=int, default=6,
                        help="Number of SAINTBlocks (default: 6)")
    parser.add_argument("--n_head", type=int, default=8,
                        help="Number of attention heads (default: 8)")
    parser.add_argument("--ffn_factor", type=float, default=4.0,
                        help="FFN hidden dimension multiplier (default: 4.0)")
    parser.add_argument("--saint_variant", type=str, default="saint",
                        choices=["saint", "saint_s", "saint_i"],
                        help="SAINT variant (default: saint)")
    parser.add_argument("--cutmix_probability", type=float, default=0.3,
                        help="CutMix feature swap probability (default: 0.3)")
    parser.add_argument("--mixup_alpha", type=float, default=0.2,
                        help="Mixup Beta distribution alpha (default: 0.2)")
    parser.add_argument("--tau", type=float, default=0.7,
                        help="NTXent temperature (default: 0.7)")
    parser.add_argument("--lambda_denoise", type=float, default=10.0,
                        help="Denoising loss weight (default: 10.0)")
    parser.add_argument("--pretraining_head_dim", type=int, default=256,
                        help="Projection head dimension (default: 256)")
    parser.add_argument("--dropout_rate", type=float, default=0.0,
                        help="Dropout rate (default: 0.0)")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--checkpoint_interval", type=int, default=50)
    parser.add_argument("--label_columns", type=str, nargs='*', default=None,
                        help="Label columns to exclude from features")
    parser.add_argument("--keep_checkpoints", action="store_true",
                        help="Keep per-table checkpoints after embedding (default: delete)")
    parser.add_argument("--table_list", default=None, help="Path to table list file for shard filtering")
    return parser.parse_args()


def make_config(input_dim, cat_cardinalities, n_continuous, args):
    embedding_config = FTEmbeddingConfig(
        input_dim=input_dim,
        emb_dim=args.emb_dim,
        cont_nums=n_continuous,
        cat_cardinality=cat_cardinalities,
        required_token_dim=2,
    )
    backbone_config = SAINTBackboneConfig(
        d_model=args.emb_dim,
        encoder_depth=args.encoder_depth,
        n_head=args.n_head,
        ffn_factor=args.ffn_factor,
        dropout_rate=args.dropout_rate,
        saint_variant=args.saint_variant,
    )
    return SAINTConfig(
        task="classification",
        loss_fn="CrossEntropyLoss",
        metric="accuracy_score",
        metric_hparams={},
        embedding_config=embedding_config,
        backbone_config=backbone_config,
        output_dim=2,
        num_continuous=n_continuous,
        cat_cardinality=cat_cardinalities,
        cutmix_probability=args.cutmix_probability,
        mixup_alpha=args.mixup_alpha,
        tau=args.tau,
        lambda_denoise=args.lambda_denoise,
        pretraining_head_dim=args.pretraining_head_dim,
    )


def train_on_table(df, checkpoint_dir, args):
    os.makedirs(checkpoint_dir, exist_ok=True)

    X_encoded, encoders, scaler, category_cols, continuous_cols = preprocess_table(
        df, label_columns=args.label_columns
    )
    cat_cardinalities = [len(encoders[c].classes_) for c in category_cols]
    n_continuous = len(continuous_cols)
    input_dim = X_encoded.shape[1]

    config = make_config(input_dim, cat_cardinalities, n_continuous, args)

    pl_model = SAINTLightning(config)
    pl_model.set_first_phase()

    dataset = SAINTDataset(
        X=X_encoded, Y=None, unlabeled_data=None, config=config,
        continuous_cols=continuous_cols, category_cols=category_cols,
        is_second_phase=False,
    )
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
        "model_name": "SAINT", "training_mode": "self_supervised",
        "category_cols": category_cols, "continuous_cols": continuous_cols,
        "cat_cardinalities": cat_cardinalities, "input_dim": input_dim,
        "n_continuous": n_continuous,
        "emb_dim": args.emb_dim, "encoder_depth": args.encoder_depth,
        "n_head": args.n_head, "ffn_factor": args.ffn_factor,
        "saint_variant": args.saint_variant,
        "cutmix_probability": args.cutmix_probability,
        "mixup_alpha": args.mixup_alpha, "tau": args.tau,
        "lambda_denoise": args.lambda_denoise,
        "pretraining_head_dim": args.pretraining_head_dim,
        "dropout_rate": args.dropout_rate,
        "categorical_encoders": encoders, "scaler": scaler,
        "label_columns": args.label_columns,
        "model_config": config,
    }, checkpoint_dir, CKPT_FILENAME)

    del pl_model
    torch.cuda.empty_cache()
    gc.collect()


def embed_from_checkpoint(df, checkpoint_dir, batch_size, label_columns=None,
                          saint_variant="saint"):
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
        SAINTLightning, train_config["model_config"],
        checkpoint_dir, CKPT_FILENAME, device=device,
    )

    variant = train_config.get("saint_variant", "saint")
    if variant != "saint_s":
        logger.warning(
            "SAINT variant '%s' uses intersample attention: row embeddings "
            "are batch-dependent. Embeddings are reproducible for the same "
            "batch_size=%d and row ordering, but will differ if either changes.",
            variant, batch_size)

    # Inference dataset (no augmentation)
    inference_config = SAINTConfig(
        task="classification", loss_fn="CrossEntropyLoss", metric="accuracy_score",
        metric_hparams={},
        embedding_config=train_config["model_config"].embedding_config,
        backbone_config=train_config["model_config"].backbone_config,
        output_dim=2,
        num_continuous=train_config.get("n_continuous", len(continuous_cols)),
        cat_cardinality=train_config.get("cat_cardinalities", []),
        cutmix_probability=train_config.get("cutmix_probability", 0.3),
    )
    dataset = SAINTDataset(
        X=X, Y=None, config=inference_config,
        continuous_cols=continuous_cols, category_cols=category_cols,
        is_second_phase=True,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        sampler=SequentialSampler(dataset), num_workers=0,
    )

    embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)
            z = pl_model.model.embedding_module(x)
            emb = pl_model.model.encoder(z)  # SAINTEncoder returns CLS token directly
            embeddings.append(emb.cpu().numpy())

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
        print(f"  Training SAINT on {len(df)} rows...")
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
        embeddings = embed_from_checkpoint(
            df, checkpoint_dir, args.batch_size,
            label_columns=args.label_columns,
            saint_variant=args.saint_variant)
    except Exception as e:
        print(f"  SKIP {csv_path.name}: embedding failed: {e}")
        return None

    if not args.keep_checkpoints:
        clean_partial_checkpoint(checkpoint_dir)

    return build_table_result(str(csv_path), embeddings, feature_cols, "SAINT")


def main():
    import psutil
    _proc = psutil.Process()
    def _mem():
        return f"RSS={_proc.memory_info().rss/1024**2:.0f}MB"

    args = parse_args()
    print("=" * 80)
    print(f"SAINT Directory-Mode Row Embedding Generation  PID={os.getpid()} {_mem()}")
    print(f"  variant={args.saint_variant}  emb_dim={args.emb_dim}  depth={args.encoder_depth}")
    print("=" * 80)

    if args.saint_variant != "saint_s":
        print(f"WARNING: SAINT variant '{args.saint_variant}' uses intersample attention.")
        print(f"  Row embeddings are batch-dependent (reproducible for same batch_size={args.batch_size}")
        print(f"  and row ordering, but will differ if either changes).")
        print(f"  Use --saint_variant saint_s for batch-independent embeddings.")

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
