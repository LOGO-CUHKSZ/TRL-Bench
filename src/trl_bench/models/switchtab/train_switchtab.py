"""
SwitchTab Self-Supervised Training (Reconstruction And Contrastive Learning)
Trains embeddings using reconstruction and contrastive learning - always self-supervised

Two modes of operation:
1. Without label column (--label_column not specified):
   - Trains on ALL columns as features
   - Pure unsupervised embeddings

2. With label column (--label_column <name>):
   - Trains on all columns EXCEPT the label column
   - Label saved separately for downstream evaluation
   - Embeddings are still self-supervised (label not used in training)

Use this when:
- You want generic embeddings for multiple downstream tasks
- You'll train a separate downstream classifier (train_downstream.py)
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

from trl_bench.utils.ts3l.pl_modules import SwitchTabLightning
from trl_bench.utils.ts3l.utils.switchtab_utils import SwitchTabDataset, SwitchTabFirstPhaseCollateFN, SwitchTabConfig
from trl_bench.utils.ts3l.utils import TS3LDataModule
from trl_bench.utils.ts3l.utils.embedding_utils import IdentityEmbeddingConfig
from trl_bench.utils.ts3l.utils.backbone_utils import MLPBackboneConfig
from trl_bench.utils.table_dataset import load_table_dataset, resolve_label_columns_cli, SSLPreprocessor, is_regression_label

print("="*80)
print("SwitchTab Pipeline - Phase 1 Only (Self-Supervised Pretraining)")
print("="*80)

parser = argparse.ArgumentParser(description='Train SwitchTab using self-supervised reconstruction and contrastive learning')

# Data configuration
parser.add_argument('--data_dir', type=str, default='data/adult',
                    help='Directory containing data (supports canonical, legacy, or single CSV layouts)')
parser.add_argument('--checkpoint_dir', type=str, default='models/switchtab/checkpoints')
parser.add_argument('--label_column', type=str, default=None)
parser.add_argument(
    '--label_policy',
    type=str,
    default='auto',
    choices=['auto', 'none', 'manifest', 'cli'],
    help=(
        "Label resolution policy: auto (legacy behavior), none (force unlabeled), "
        "manifest (use dataset.json labels), cli (require --label_column)."
    ),
)

# Training configuration
parser.add_argument('--val_ratio', type=float, default=0.2)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--phase1_epochs', type=int, default=20)
parser.add_argument('--random_seed', type=int, default=42)

# SwitchTab hyperparameters
parser.add_argument('--hidden_dim', type=int, default=512)
parser.add_argument('--n_hidden_layers', type=int, default=3)
parser.add_argument('--corruption_rate', type=float, default=0.3)
parser.add_argument('--alpha', type=float, default=1.0)

# Training options
parser.add_argument('--early_stopping_patience', type=int, default=5)

# Split-aware options
parser.add_argument('--preprocess_fit_scope', type=str, default='train',
                    help='Which split to fit preprocessor on: "train" or "all" (default: train)')
parser.add_argument('--context_split', type=str, default='train',
                    help='Which split to use for training (default: train)')
parser.add_argument('--ignore_fingerprint', action='store_true',
                    help='Skip SHA256 fingerprint verification for canonical datasets')

args = parser.parse_args()

os.makedirs(args.checkpoint_dir, exist_ok=True)

# ============================================================================
# 1. Load data via TableDataset
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
        raise ValueError(
            f"Label column '{args.label_column}' not found in dataset. "
            f"Available labels: {dataset.label_columns}"
        )
    else:
        print(f"   Warning: label column '{args.label_column}' was not found; continuing without labels")
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
# 2. Split training data into train/validation
# ============================================================================
print(f"\n2. Creating data splits...")

X_all = train_view.X
y_all = train_view.y

if has_labels and y_all is not None:
    # Use first classification label for stratification
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
            print(f"   Using stratified split based on label column '{stratify_col}'")
        except ValueError:
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_all, y_all, test_size=args.val_ratio, random_state=args.random_seed
            )
            print(f"   Using random split (stratification failed for label column)")
    else:
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_all, y_all, test_size=args.val_ratio, random_state=args.random_seed
        )
        print(f"   Using random split (all labels are regression)")
else:
    print(f"   Using random split (no labels)")
    X_train, X_valid = train_test_split(
        X_all, test_size=args.val_ratio, random_state=args.random_seed
    )
    y_train, y_valid = None, None

print(f"   Training samples: {len(X_train)}, Validation samples: {len(X_valid)}")

# ============================================================================
# 3. Fit preprocessor and transform data
# ============================================================================
print(f"\n3. Fitting SSLPreprocessor (scope: {args.preprocess_fit_scope})...")

preprocessor = SSLPreprocessor()
if args.preprocess_fit_scope == "all":
    preprocessor.fit(dataset.get_full().X)
else:
    preprocessor.fit(X_train)

category_cols = preprocessor.category_cols
continuous_cols = preprocessor.continuous_cols

print(f"   Categorical columns ({len(category_cols)}), Continuous columns ({len(continuous_cols)})")

X_train_enc = preprocessor.transform(X_train)
X_valid_enc = preprocessor.transform(X_valid)

cat_cardinalities = preprocessor.cat_cardinalities

# Encode target labels (per-column encoders)
label_encoders = {}
if has_labels and y_all is not None:
    full_view = dataset.get_full()
    for col in label_cols:
        y_col = full_view.y[col] if isinstance(full_view.y, pd.DataFrame) else full_view.y
        if is_regression_label(y_col, dataset.label_task_types, col):
            label_encoders[col] = None
            print(f"   Label '{col}': regression (raw values will be saved)")
        else:
            le = LabelEncoder()
            le.fit(y_col)
            label_encoders[col] = le
            print(f"   Label '{col}': classification ({le.classes_.tolist()})")

output_dim = 2  # Dummy value for SwitchTab config (SSL never uses labels)

input_dim = preprocessor.input_dim

print(f"   Input dimension: {input_dim}, Output dimension: {output_dim}")

# ============================================================================
# 5. Configure and train SwitchTab
# ============================================================================
print(f"\n5. Configuring SwitchTab model...")
embedding_config = IdentityEmbeddingConfig(input_dim=input_dim)
backbone_config = MLPBackboneConfig(
    input_dim=embedding_config.output_dim,
    hidden_dims=args.hidden_dim,
    n_hiddens=args.n_hidden_layers
)

config = SwitchTabConfig(
    task="classification",
    loss_fn="CrossEntropyLoss",
    metric="accuracy_score",
    metric_hparams={},
    embedding_config=embedding_config,
    backbone_config=backbone_config,
    output_dim=output_dim,
    corruption_rate=args.corruption_rate,
    alpha=args.alpha
)

pl_switchtab = SwitchTabLightning(config)

print(f"\n" + "="*80)
print("PHASE 1: Self-Supervised Learning (Corrupted Feature Recovery)")
print("="*80)

pl_switchtab.set_first_phase()

train_ds = SwitchTabDataset(
    X=X_train_enc, Y=None, config=config, unlabeled_data=None,
    continuous_cols=continuous_cols, category_cols=category_cols
)
valid_ds = SwitchTabDataset(
    X=X_valid_enc, Y=None, config=config,
    continuous_cols=continuous_cols, category_cols=category_cols
)

collate_fn = SwitchTabFirstPhaseCollateFN()
datamodule = TS3LDataModule(train_ds, valid_ds, args.batch_size, train_sampler='random', train_collate_fn=collate_fn, valid_collate_fn=collate_fn)

callbacks = [
    ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename='switchtab-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss', mode='min', save_top_k=1, verbose=True
    ),
    EarlyStopping(
        monitor='val_loss', mode='min',
        patience=args.early_stopping_patience, verbose=True
    )
]

trainer = Trainer(
    accelerator='auto',
    max_epochs=args.phase1_epochs,
    callbacks=callbacks,
    enable_progress_bar=True,
    logger=False,
    default_root_dir=args.checkpoint_dir
)

print(f"\nTraining for up to {args.phase1_epochs} epochs...")
trainer.fit(pl_switchtab, datamodule)

best_ckpt = callbacks[0].best_model_path
print(f"\nPhase 1 completed!")

final_ckpt = os.path.join(args.checkpoint_dir, "switchtab_self_supervised.ckpt")
os.system(f"cp '{best_ckpt}' '{final_ckpt}'")

config_dict = {
    # Training configuration
    'val_ratio': args.val_ratio, 'batch_size': args.batch_size, 'epochs': args.phase1_epochs,
    'hidden_dim': args.hidden_dim, 'n_hidden_layers': args.n_hidden_layers,
    'corruption_rate': args.corruption_rate, 'alpha': args.alpha,

    # Model info
    'training_mode': 'self_supervised', 'model_name': 'SwitchTab',

    # Checkpoints
    'best_checkpoint': best_ckpt, 'final_checkpoint': final_ckpt,

    # Data configuration
    'label_columns': label_cols,
    'label_encoders': label_encoders,
    'label_task_types': {c: dataset.label_task_types.get(c, '') for c in label_cols},
    'label_policy': args.label_policy,
    'has_label': has_labels,
    'category_cols': category_cols, 'continuous_cols': continuous_cols,
    'cat_cardinalities': cat_cardinalities, 'input_dim': input_dim, 'output_dim': output_dim,

    # Backward compat keys (singular)
    'label_column': label_cols[0] if label_cols else None,
    'label_encoder': label_encoders.get(label_cols[0]) if label_cols else None,

    # New: unified preprocessor
    'preprocessor': preprocessor,

    # Split-aware metadata
    'preprocess_fit_scope': args.preprocess_fit_scope,
    'context_split': train_split_name,
    'dataset_source_path': dataset.source_path,
    'dataset_sha256': dataset.fingerprint,

    'self_sufficient': True, 'detected_by': 'SwitchTab'
}

config_file = os.path.join(args.checkpoint_dir, "training_config.pkl")
with open(config_file, 'wb') as f:
    pickle.dump(config_dict, f)

print("\n" + "="*80)
print("Self-supervised training completed successfully!")
print("="*80)
print(f"\nTraining mode:")
if has_labels:
    print(f"  - Trained on {len(category_cols) + len(continuous_cols)} features (excluding {len(label_cols)} label(s))")
    print(f"  - Label columns {label_cols} saved for downstream evaluation")
else:
    print(f"  - Trained on ALL {len(category_cols) + len(continuous_cols)} columns")
    print(f"  - No label column (pure unsupervised)")
print(f"\nSaved files:")
print(f"  - Best checkpoint: {best_ckpt}")
print(f"  - Final checkpoint: {final_ckpt}")
print(f"  - Training config: {config_file}")
print(f"\nIMPORTANT: training_config.pkl includes:")
print(f"  - SSLPreprocessor (OrdinalEncoder + MinMaxScaler)")
if has_labels:
    print(f"  - Label encoders for {label_cols}")
else:
    print(f"  - No label encoder (no labels)")
print(f"\nNext steps:")
print(f"  1. Generate embeddings:")
print(f"     python models/switchtab/generate_embeddings.py --checkpoint {final_ckpt}")
print(f"  2. Train downstream classifier:")
print(f"     python train_downstream.py --embedding_dir models/switchtab/embeddings")
print("="*80)
