"""
SCARF Self-Supervised Training (Contrastive Learning)
Trains embeddings using contrastive learning - always self-supervised

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

from trl_bench.utils.ts3l.pl_modules import SCARFLightning
from trl_bench.utils.ts3l.utils.scarf_utils import SCARFDataset, SCARFConfig
from trl_bench.utils.ts3l.utils import TS3LDataModule
from trl_bench.utils.ts3l.utils.embedding_utils import IdentityEmbeddingConfig
from trl_bench.utils.ts3l.utils.backbone_utils import MLPBackboneConfig
from trl_bench.utils.table_dataset import load_table_dataset, resolve_label_columns_cli, SSLPreprocessor, is_regression_label

print("="*80)
print("SCARF Pipeline - Self-Supervised Training (Contrastive Learning)")
print("="*80)
print("Trains embeddings using contrastive learning - always self-supervised")
print("Labels (if provided) are saved but NOT used in training")
print("="*80)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train SCARF using self-supervised contrastive learning')

# Data configuration
parser.add_argument('--data_dir', type=str, default='data/adult',
                    help='Directory containing data (supports canonical, legacy, or single CSV layouts)')
parser.add_argument('--checkpoint_dir', type=str, default='models/scarf/checkpoints',
                    help='Directory to save model checkpoints (default: models/scarf/checkpoints)')
parser.add_argument('--label_column', type=str, default=None,
                    help='Name of the label column to exclude from training and save separately (optional)')
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
parser.add_argument('--val_ratio', type=float, default=0.2,
                    help='Ratio of training data for validation (default: 0.2)')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size for training (default: 128)')
parser.add_argument('--phase1_epochs', type=int, default=20,
                    help='Number of epochs for phase 1 training (default: 20)')
parser.add_argument('--random_seed', type=int, default=42,
                    help='Random seed for reproducibility (default: 42)')

# SCARF hyperparameters
parser.add_argument('--hidden_dim', type=int, default=512,
                    help='Hidden dimension of MLP backbone (default: 512)')
parser.add_argument('--n_hidden_layers', type=int, default=3,
                    help='Number of hidden layers in MLP (default: 3)')
parser.add_argument('--pretraining_head_dim', type=int, default=256,
                    help='Dimension of pretraining head (default: 256)')
parser.add_argument('--head_depth', type=int, default=2,
                    help='Depth of head (default: 2)')
parser.add_argument('--corruption_rate', type=float, default=0.6,
                    help='Feature corruption rate (default: 0.6)')
parser.add_argument('--tau', type=float, default=0.1,
                    help='Temperature parameter for contrastive loss (default: 0.1)')
parser.add_argument('--dropout_rate', type=float, default=0.04,
                    help='Dropout rate (default: 0.04)')

# Training options
parser.add_argument('--early_stopping_patience', type=int, default=5,
                    help='Early stopping patience (default: 5)')

# Split-aware options
parser.add_argument('--preprocess_fit_scope', type=str, default='train',
                    help='Which split to fit preprocessor on: "train" or "all" (default: train)')
parser.add_argument('--context_split', type=str, default='train',
                    help='Which split to use for training (default: train)')
parser.add_argument('--ignore_fingerprint', action='store_true',
                    help='Skip SHA256 fingerprint verification for canonical datasets')

args = parser.parse_args()

# Create checkpoint directory
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
    # Single CSV or "all" split
    train_split_name = dataset.split_names[0]

train_view = dataset.get_split(train_split_name)
print(f"   Training on split '{train_split_name}': {len(train_view)} samples")

# ============================================================================
# 2. Split training data into train/validation
# ============================================================================
print(f"\n2. Creating train/validation splits...")
print(f"   Validation ratio: {args.val_ratio}")

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

print(f"   Training samples: {len(X_train)}")
print(f"   Validation samples: {len(X_valid)}")

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

print(f"   Categorical columns ({len(category_cols)}): {category_cols[:5]}{'...' if len(category_cols) > 5 else ''}")
print(f"   Continuous columns ({len(continuous_cols)}): {continuous_cols[:5]}{'...' if len(continuous_cols) > 5 else ''}")

X_train_enc = preprocessor.transform(X_train)
X_valid_enc = preprocessor.transform(X_valid)

cat_cardinalities = preprocessor.cat_cardinalities
print(f"   Category cardinalities: {cat_cardinalities}")
print(f"   Data encoded and normalized")

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

output_dim = 2  # Dummy value for SCARF config (SSL never uses labels)

# Get dimensions
input_dim = preprocessor.input_dim

print(f"\n   Input dimension: {input_dim}")
print(f"   Output dimension: {output_dim}")

# ============================================================================
# 5. Configure and train SCARF
# ============================================================================
print(f"\n5. Configuring SCARF model (Phase 1 only)...")
embedding_config = IdentityEmbeddingConfig(input_dim=input_dim)
backbone_config = MLPBackboneConfig(
    input_dim=embedding_config.output_dim,
    hidden_dims=args.hidden_dim,
    n_hiddens=args.n_hidden_layers
)

config = SCARFConfig(
    task="classification",
    loss_fn="CrossEntropyLoss",
    metric="accuracy_score",
    metric_hparams={},
    embedding_config=embedding_config,
    backbone_config=backbone_config,
    output_dim=output_dim,
    pretraining_head_dim=args.pretraining_head_dim,
    head_depth=args.head_depth,
    corruption_rate=args.corruption_rate,
    tau=args.tau,
    dropout_rate=args.dropout_rate
)

print(f"   Embedding: Identity")
print(f"   Backbone: MLP (hidden_dim={args.hidden_dim}, n_layers={args.n_hidden_layers})")
print(f"   Hyperparameters: corruption_rate={args.corruption_rate}, tau={args.tau}, dropout={args.dropout_rate}")

# Initialize SCARF
pl_scarf = SCARFLightning(config)

# ============================================================================
# Self-Supervised Training (Contrastive Learning)
# ============================================================================
print(f"\n" + "="*80)
print("Self-Supervised Training (Contrastive Pretraining)")
print("="*80)

pl_scarf.set_first_phase()

# Prepare datasets (self-supervised - no labels used)
train_ds = SCARFDataset(
    X=X_train_enc,
    Y=None,  # Always None - self-supervised training
    unlabeled_data=None,
    config=config,
    continuous_cols=continuous_cols,
    category_cols=category_cols,
    is_second_phase=False
)
valid_ds = SCARFDataset(
    X=X_valid_enc,
    Y=None,  # Always None - self-supervised training
    config=config,
    continuous_cols=continuous_cols,
    category_cols=category_cols,
    is_second_phase=False
)

datamodule = TS3LDataModule(train_ds, valid_ds, args.batch_size, train_sampler='random')

# Callbacks
callbacks = [
    ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename='scarf-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        verbose=True
    ),
    EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=args.early_stopping_patience,
        verbose=True
    )
]

# Train SCARF
trainer = Trainer(
    accelerator='auto',
    max_epochs=args.phase1_epochs,
    callbacks=callbacks,
    enable_progress_bar=True,
    logger=False,
    default_root_dir=args.checkpoint_dir
)

print(f"\nTraining for up to {args.phase1_epochs} epochs...")
trainer.fit(pl_scarf, datamodule)

# Get best checkpoint
best_ckpt = callbacks[0].best_model_path
print(f"\nTraining completed!")
print(f"  Best checkpoint: {best_ckpt}")

# Save final checkpoint with a standardized name
final_ckpt = os.path.join(args.checkpoint_dir, "scarf_self_supervised.ckpt")
os.system(f"cp '{best_ckpt}' '{final_ckpt}'")

# Save training configuration INCLUDING preprocessor
config_dict = {
    # Training configuration
    'val_ratio': args.val_ratio,
    'batch_size': args.batch_size,
    'epochs': args.phase1_epochs,
    'hidden_dim': args.hidden_dim,
    'n_hidden_layers': args.n_hidden_layers,
    'pretraining_head_dim': args.pretraining_head_dim,
    'head_depth': args.head_depth,
    'corruption_rate': args.corruption_rate,
    'tau': args.tau,
    'dropout_rate': args.dropout_rate,

    # Model info
    'model_name': 'SCARF',
    'training_mode': 'self_supervised',
    'note': 'Self-supervised embeddings using contrastive learning',

    # Checkpoints
    'best_checkpoint': best_ckpt,
    'final_checkpoint': final_ckpt,

    # Data configuration
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

    # Metadata
    'self_sufficient': True,
    'detected_by': 'SCARF'
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
print(f"     python models/scarf/generate_embeddings.py --checkpoint {final_ckpt}")
print(f"  2. Train downstream classifier:")
print(f"     python train_downstream.py --embedding_dir models/scarf/embeddings")
print("="*80)
