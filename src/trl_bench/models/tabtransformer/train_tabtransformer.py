"""
TabTransformer-SSL Self-Supervised Training (MLM + RTD)
Trains embeddings using masked language modeling and replaced token detection.

Two modes of operation:
1. Without label column (--label_column not specified):
   - Trains on ALL columns as features
   - Pure unsupervised embeddings

2. With label column (--label_column <name>):
   - Trains on all columns EXCEPT the label column
   - Label saved separately for downstream evaluation
   - Embeddings are still self-supervised (label not used in training)

Requires at least 1 categorical column.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../' * 2))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
import pickle
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy

from trl_bench.utils.ts3l.pl_modules import TabTransformerSSLLightning
from trl_bench.utils.ts3l.utils.tabtransformer_utils import (
    TabTransformerSSLDataset, TabTransformerSSLCollateFN, TabTransformerSSLConfig,
)
from trl_bench.utils.ts3l.utils import TS3LDataModule
from trl_bench.utils.ts3l.utils.embedding_utils import IdentityEmbeddingConfig
from trl_bench.utils.ts3l.utils.backbone_utils import MLPBackboneConfig
from trl_bench.utils.table_dataset import load_table_dataset, resolve_label_columns_cli, SSLPreprocessor, is_regression_label

print("="*80)
print("TabTransformer-SSL Pipeline - Self-Supervised Training (MLM + RTD)")
print("="*80)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train TabTransformer-SSL')

# Data configuration
parser.add_argument('--data_dir', type=str, default='data/adult')
parser.add_argument('--checkpoint_dir', type=str, default='models/tabtransformer/checkpoints')
parser.add_argument('--label_column', type=str, default=None)
parser.add_argument('--label_policy', type=str, default='auto',
                    choices=['auto', 'none', 'manifest', 'cli'])

# Training configuration
parser.add_argument('--val_ratio', type=float, default=0.2)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--phase1_epochs', type=int, default=20)
parser.add_argument('--random_seed', type=int, default=42)

# TabTransformer hyperparameters
parser.add_argument('--hidden_dim', type=int, default=512)
parser.add_argument('--n_hidden_layers', type=int, default=3)
parser.add_argument('--emb_dim', type=int, default=32)
parser.add_argument('--n_transformer_layers', type=int, default=6)
parser.add_argument('--n_heads', type=int, default=8)
parser.add_argument('--mlm_probability', type=float, default=0.15)
parser.add_argument('--rtd_probability', type=float, default=0.15)
parser.add_argument('--dropout_rate', type=float, default=0.04)

# Training options
parser.add_argument('--early_stopping_patience', type=int, default=5)

# Split-aware options
parser.add_argument('--preprocess_fit_scope', type=str, default='all',
                    help='TabTransformer uses nn.Embedding for categoricals, so unseen categories '
                         '(encoded as -1) cause index-out-of-range errors. Default "all" avoids this.')
parser.add_argument('--context_split', type=str, default='train')
parser.add_argument('--ignore_fingerprint', action='store_true')

args = parser.parse_args()

os.makedirs(args.checkpoint_dir, exist_ok=True)

# ============================================================================
# 1. Load data
# ============================================================================
print(f"\n1. Loading data from {args.data_dir}...")

label_columns_cli = resolve_label_columns_cli(args.label_column, args.label_policy)
dataset = load_table_dataset(
    args.data_dir,
    label_columns_cli=label_columns_cli,
    ignore_fingerprint=args.ignore_fingerprint,
)

print(f"   Dataset: {dataset}")
if args.label_policy == 'manifest':
    label_cols = list(dataset.label_columns)
elif args.label_column:
    if args.label_column in dataset.label_columns:
        label_cols = [args.label_column]
    elif args.label_policy == 'cli':
        raise ValueError(f"Label column '{args.label_column}' not found.")
    else:
        print(f"   Warning: label column '{args.label_column}' not found; continuing without labels")
        label_cols = []
else:
    label_cols = []

has_labels = bool(label_cols)
if label_cols:
    print(f"   Label columns: {label_cols}")

# Determine training split
if args.context_split in dataset.split_names:
    train_split_name = args.context_split
elif "train" in dataset.split_names:
    train_split_name = "train"
else:
    train_split_name = dataset.split_names[0]

train_view = dataset.get_split(train_split_name)
print(f"   Training on split '{train_split_name}': {len(train_view)} samples")

# ============================================================================
# 2. Train/validation split
# ============================================================================
print(f"\n2. Creating train/validation splits...")

X_all = train_view.X
y_all = train_view.y

if has_labels and y_all is not None:
    stratify_col = next(
        (c for c in label_cols
         if not is_regression_label(
             y_all[c] if isinstance(y_all, pd.DataFrame) else y_all,
             dataset.label_task_types, c)),
        None,
    )
    if stratify_col is not None:
        stratify_values = y_all[stratify_col] if isinstance(y_all, pd.DataFrame) else y_all
        try:
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_all, y_all, test_size=args.val_ratio, random_state=args.random_seed, stratify=stratify_values
            )
        except ValueError:
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_all, y_all, test_size=args.val_ratio, random_state=args.random_seed
            )
    else:
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_all, y_all, test_size=args.val_ratio, random_state=args.random_seed
        )
else:
    X_train, X_valid = train_test_split(
        X_all, test_size=args.val_ratio, random_state=args.random_seed
    )
    y_train, y_valid = None, None

print(f"   Training samples: {len(X_train)}")
print(f"   Validation samples: {len(X_valid)}")

# ============================================================================
# 3. Fit preprocessor
# ============================================================================
print(f"\n3. Fitting SSLPreprocessor...")

preprocessor = SSLPreprocessor()
if args.preprocess_fit_scope == "all":
    preprocessor.fit(dataset.get_full().X)
else:
    preprocessor.fit(X_train)

category_cols = preprocessor.category_cols
continuous_cols = preprocessor.continuous_cols

print(f"   Categorical columns ({len(category_cols)}): {category_cols[:5]}{'...' if len(category_cols) > 5 else ''}")
print(f"   Continuous columns ({len(continuous_cols)}): {continuous_cols[:5]}{'...' if len(continuous_cols) > 5 else ''}")

if len(category_cols) == 0:
    print("NOT APPLICABLE: TabTransformer requires at least 1 categorical column.")
    print("   This dataset has no categorical features — skipping with clean exit.")
    # Write N/A marker so the generate script knows training was intentionally skipped
    import json
    na_marker = os.path.join(args.checkpoint_dir, "not_applicable.json")
    with open(na_marker, 'w') as f:
        json.dump({
            "reason": "No categorical columns found",
            "model": "TabTransformerSSL",
            "dataset": args.data_dir,
            "continuous_cols": len(continuous_cols),
            "category_cols": 0,
        }, f, indent=2)
    print(f"   Wrote N/A marker: {na_marker}")
    sys.exit(0)

X_train_enc = preprocessor.transform(X_train)
X_valid_enc = preprocessor.transform(X_valid)

cat_cardinalities = preprocessor.cat_cardinalities
print(f"   Category cardinalities: {cat_cardinalities}")

# Encode target labels
label_encoders = {}
if has_labels and y_all is not None:
    full_view = dataset.get_full()
    for col in label_cols:
        y_col = full_view.y[col] if isinstance(full_view.y, pd.DataFrame) else full_view.y
        if is_regression_label(y_col, dataset.label_task_types, col):
            label_encoders[col] = None
        else:
            le = LabelEncoder()
            le.fit(y_col)
            label_encoders[col] = le
            print(f"   Label '{col}': classification ({le.classes_.tolist()})")

output_dim = 2  # Dummy for SSL
input_dim = preprocessor.input_dim
n_cat = len(category_cols)
n_continuous = len(continuous_cols)
emb_dim = args.emb_dim

print(f"\n   Input dimension: {input_dim}")
print(f"   TabTransformer embedding output: {n_cat * emb_dim + n_continuous}")

# ============================================================================
# 4. Configure and train TabTransformer-SSL
# ============================================================================
print(f"\n4. Configuring TabTransformer-SSL model...")

tabtransformer_output = n_cat * emb_dim + n_continuous
embedding_config = IdentityEmbeddingConfig(input_dim=tabtransformer_output)
backbone_config = MLPBackboneConfig(
    input_dim=tabtransformer_output,
    hidden_dims=args.hidden_dim,
    n_hiddens=args.n_hidden_layers,
)

config = TabTransformerSSLConfig(
    task="classification",
    loss_fn="CrossEntropyLoss",
    metric="accuracy_score",
    metric_hparams={},
    embedding_config=embedding_config,
    backbone_config=backbone_config,
    output_dim=output_dim,
    emb_dim=emb_dim,
    n_transformer_layers=args.n_transformer_layers,
    n_heads=args.n_heads,
    cat_cardinality=cat_cardinalities,
    num_continuous=n_continuous,
    mlm_probability=args.mlm_probability,
    rtd_probability=args.rtd_probability,
    dropout_rate=args.dropout_rate,
)

pl_model = TabTransformerSSLLightning(config)

# ============================================================================
# Self-Supervised Training
# ============================================================================
print(f"\n" + "="*80)
print("Self-Supervised Training (MLM + RTD Pretraining)")
print("="*80)

pl_model.set_first_phase()

train_ds = TabTransformerSSLDataset(
    X=X_train_enc, Y=None, unlabeled_data=None,
    continuous_cols=continuous_cols, category_cols=category_cols,
)
valid_ds = TabTransformerSSLDataset(
    X=X_valid_enc, Y=None,
    continuous_cols=continuous_cols, category_cols=category_cols,
)

collate_fn = TabTransformerSSLCollateFN(config)
effective_batch_size = min(len(train_ds), args.batch_size)
datamodule = TS3LDataModule(
    train_ds, valid_ds, effective_batch_size,
    train_sampler='random',
    train_collate_fn=collate_fn,
    valid_collate_fn=collate_fn,
    drop_last=True,  # avoid batch-size-1 crash in BatchNorm1d
)

callbacks = [
    ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename='tabtransformer-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        verbose=True,
    ),
    EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=args.early_stopping_patience,
        verbose=True,
    ),
]

trainer = Trainer(
    accelerator='auto',
    max_epochs=args.phase1_epochs,
    callbacks=callbacks,
    enable_progress_bar=True,
    logger=False,
    default_root_dir=args.checkpoint_dir,
    strategy=DDPStrategy(find_unused_parameters=True) if torch.cuda.device_count() > 1 else 'auto',
)

print(f"\nTraining for up to {args.phase1_epochs} epochs...")
trainer.fit(pl_model, datamodule)

best_ckpt = callbacks[0].best_model_path
print(f"\nTraining completed!")
print(f"  Best checkpoint: {best_ckpt}")

final_ckpt = os.path.join(args.checkpoint_dir, "tabtransformer_self_supervised.ckpt")
os.system(f"cp '{best_ckpt}' '{final_ckpt}'")

config_dict = {
    'val_ratio': args.val_ratio,
    'batch_size': args.batch_size,
    'epochs': args.phase1_epochs,
    'hidden_dim': args.hidden_dim,
    'n_hidden_layers': args.n_hidden_layers,
    'emb_dim': args.emb_dim,
    'n_transformer_layers': args.n_transformer_layers,
    'n_heads': args.n_heads,
    'mlm_probability': args.mlm_probability,
    'rtd_probability': args.rtd_probability,
    'dropout_rate': args.dropout_rate,

    'model_name': 'TabTransformerSSL',
    'training_mode': 'self_supervised',
    'note': 'Self-supervised embeddings using MLM + RTD',

    'best_checkpoint': best_ckpt,
    'final_checkpoint': final_ckpt,

    'label_columns': label_cols,
    'label_encoders': label_encoders,
    'label_task_types': {c: dataset.label_task_types.get(c, '') for c in label_cols},
    'label_policy': args.label_policy,
    'has_label': has_labels,
    'category_cols': category_cols,
    'continuous_cols': continuous_cols,
    'cat_cardinalities': cat_cardinalities,
    'input_dim': input_dim,
    'output_dim': output_dim,

    'label_column': label_cols[0] if label_cols else None,
    'label_encoder': label_encoders.get(label_cols[0]) if label_cols else None,

    'preprocessor': preprocessor,

    'preprocess_fit_scope': args.preprocess_fit_scope,
    'context_split': train_split_name,
    'dataset_source_path': dataset.source_path,
    'dataset_sha256': dataset.fingerprint,

    'self_sufficient': True,
    'detected_by': 'TabTransformerSSL',
}

config_file = os.path.join(args.checkpoint_dir, "training_config.pkl")
with open(config_file, 'wb') as f:
    pickle.dump(config_dict, f)

print("\n" + "="*80)
print("Self-supervised training completed successfully!")
print("="*80)
print(f"\nSaved files:")
print(f"  - Best checkpoint: {best_ckpt}")
print(f"  - Final checkpoint: {final_ckpt}")
print(f"  - Training config: {config_file}")
print("="*80)
