"""
SAINT Self-Supervised Training (Contrastive + Denoising)
Trains embeddings using contrastive learning + feature reconstruction.

Two modes of operation:
1. Without label column (--label_column not specified):
   - Trains on ALL columns as features
   - Pure unsupervised embeddings

2. With label column (--label_column <name>):
   - Trains on all columns EXCEPT the label column
   - Label saved separately for downstream evaluation
   - Embeddings are still self-supervised (label not used in training)

NOTE: SAINT (full) and SAINT-i variants use intersample attention,
making row embeddings batch-dependent. Use --saint_variant saint_s
for batch-independent embeddings.
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

from trl_bench.utils.ts3l.pl_modules import SAINTLightning
from trl_bench.utils.ts3l.utils.saint_utils import SAINTDataset, SAINTConfig
from trl_bench.utils.ts3l.utils import TS3LDataModule
from trl_bench.utils.ts3l.utils.embedding_utils import FTEmbeddingConfig
from trl_bench.utils.ts3l.utils.backbone_utils import SAINTBackboneConfig
from trl_bench.utils.table_dataset import load_table_dataset, resolve_label_columns_cli, SSLPreprocessor, is_regression_label

print("="*80)
print("SAINT Pipeline - Self-Supervised Training (Contrastive + Denoising)")
print("="*80)
print("Trains embeddings using contrastive learning + feature reconstruction")
print("Labels (if provided) are saved but NOT used in training")
print("="*80)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train SAINT using self-supervised learning')

# Data configuration
parser.add_argument('--data_dir', type=str, default='data/adult',
                    help='Directory containing data (supports canonical, legacy, or single CSV layouts)')
parser.add_argument('--checkpoint_dir', type=str, default='models/saint/checkpoints',
                    help='Directory to save model checkpoints (default: models/saint/checkpoints)')
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

# SAINT hyperparameters
parser.add_argument('--emb_dim', type=int, default=512,
                    help='Per-feature embedding dimension (default: 512)')
parser.add_argument('--encoder_depth', type=int, default=6,
                    help='Number of SAINTBlocks (default: 6)')
parser.add_argument('--n_head', type=int, default=8,
                    help='Number of attention heads (default: 8)')
parser.add_argument('--ffn_factor', type=float, default=4.0,
                    help='FFN hidden dimension multiplier (default: 4.0)')
parser.add_argument('--saint_variant', type=str, default='saint',
                    choices=['saint', 'saint_s', 'saint_i'],
                    help='SAINT variant (default: saint)')
parser.add_argument('--pretraining_head_dim', type=int, default=256,
                    help='Dimension of pretraining head (default: 256)')
parser.add_argument('--head_depth', type=int, default=2,
                    help='Depth of head (default: 2)')
parser.add_argument('--cutmix_probability', type=float, default=0.3,
                    help='CutMix feature swap probability (default: 0.3)')
parser.add_argument('--mixup_alpha', type=float, default=0.2,
                    help='Mixup Beta distribution alpha (default: 0.2)')
parser.add_argument('--tau', type=float, default=0.7,
                    help='Temperature parameter for contrastive loss (default: 0.7)')
parser.add_argument('--lambda_denoise', type=float, default=10.0,
                    help='Denoising loss weight (default: 10.0)')
parser.add_argument('--dropout_rate', type=float, default=0.0,
                    help='Dropout rate (default: 0.0)')

# Training options
parser.add_argument('--early_stopping_patience', type=int, default=5,
                    help='Early stopping patience (default: 5)')

# Split-aware options
parser.add_argument('--preprocess_fit_scope', type=str, default='all',
                    help='Which split to fit preprocessor on: "train" or "all" (default: all). '
                         'SAINT uses nn.Embedding for categoricals, so unseen categories (encoded '
                         'as -1) cause index-out-of-range errors. Default "all" avoids this.')
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
n_continuous = len(continuous_cols)
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

output_dim = 2  # Dummy value for SAINT config (SSL never uses labels)

# Get dimensions
input_dim = preprocessor.input_dim

print(f"\n   Input dimension: {input_dim}")
print(f"   Output dimension: {output_dim}")

# ============================================================================
# 4. Configure and train SAINT
# ============================================================================
print(f"\n4. Configuring SAINT model (Phase 1 only)...")

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

config = SAINTConfig(
    task="classification",
    loss_fn="CrossEntropyLoss",
    metric="accuracy_score",
    metric_hparams={},
    embedding_config=embedding_config,
    backbone_config=backbone_config,
    output_dim=output_dim,
    num_continuous=n_continuous,
    cat_cardinality=cat_cardinalities,
    pretraining_head_dim=args.pretraining_head_dim,
    head_depth=args.head_depth,
    cutmix_probability=args.cutmix_probability,
    mixup_alpha=args.mixup_alpha,
    tau=args.tau,
    lambda_denoise=args.lambda_denoise,
    dropout_rate=args.dropout_rate,
)

print(f"   Embedding: FeatureTokenizer (emb_dim={args.emb_dim})")
print(f"   Backbone: SAINTEncoder (depth={args.encoder_depth}, heads={args.n_head}, variant={args.saint_variant})")
print(f"   Hyperparameters: cutmix={args.cutmix_probability}, mixup_alpha={args.mixup_alpha}, tau={args.tau}, lambda_denoise={args.lambda_denoise}")

if args.saint_variant != "saint_s":
    print(f"   WARNING: variant '{args.saint_variant}' uses intersample attention — embeddings are batch-dependent")

# Initialize SAINT
pl_saint = SAINTLightning(config)

# ============================================================================
# Self-Supervised Training (Contrastive + Denoising)
# ============================================================================
print(f"\n" + "="*80)
print("Self-Supervised Training (Contrastive + Denoising Pretraining)")
print("="*80)

pl_saint.set_first_phase()

# Prepare datasets (self-supervised - no labels used)
train_ds = SAINTDataset(
    X=X_train_enc,
    Y=None,
    unlabeled_data=None,
    config=config,
    continuous_cols=continuous_cols,
    category_cols=category_cols,
    is_second_phase=False
)
valid_ds = SAINTDataset(
    X=X_valid_enc,
    Y=None,
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
        filename='saint-{epoch:02d}-{val_loss:.4f}',
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

# Train SAINT
trainer = Trainer(
    accelerator='auto',
    max_epochs=args.phase1_epochs,
    callbacks=callbacks,
    enable_progress_bar=True,
    logger=False,
    default_root_dir=args.checkpoint_dir
)

print(f"\nTraining for up to {args.phase1_epochs} epochs...")
trainer.fit(pl_saint, datamodule)

# Get best checkpoint
best_ckpt = callbacks[0].best_model_path
print(f"\nTraining completed!")
print(f"  Best checkpoint: {best_ckpt}")

# Save final checkpoint with a standardized name
final_ckpt = os.path.join(args.checkpoint_dir, "saint_self_supervised.ckpt")
os.system(f"cp '{best_ckpt}' '{final_ckpt}'")

# Save training configuration INCLUDING preprocessor
config_dict = {
    # Training configuration
    'val_ratio': args.val_ratio,
    'batch_size': args.batch_size,
    'epochs': args.phase1_epochs,
    'emb_dim': args.emb_dim,
    'encoder_depth': args.encoder_depth,
    'n_head': args.n_head,
    'ffn_factor': args.ffn_factor,
    'saint_variant': args.saint_variant,
    'pretraining_head_dim': args.pretraining_head_dim,
    'head_depth': args.head_depth,
    'cutmix_probability': args.cutmix_probability,
    'mixup_alpha': args.mixup_alpha,
    'tau': args.tau,
    'lambda_denoise': args.lambda_denoise,
    'dropout_rate': args.dropout_rate,

    # Model info
    'model_name': 'SAINT',
    'training_mode': 'self_supervised',
    'note': 'Self-supervised embeddings using contrastive + denoising learning',

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
    'n_continuous': n_continuous,

    # Backward compat keys (singular)
    'label_column': label_cols[0] if label_cols else None,
    'label_encoder': label_encoders.get(label_cols[0]) if label_cols else None,

    # New: unified preprocessor
    'preprocessor': preprocessor,

    # Model config (for load_model_from_checkpoint)
    'model_config': config,

    # Split-aware metadata
    'preprocess_fit_scope': args.preprocess_fit_scope,
    'context_split': train_split_name,
    'dataset_source_path': dataset.source_path,
    'dataset_sha256': dataset.fingerprint,

    # Metadata
    'self_sufficient': True,
    'detected_by': 'SAINT'
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
if args.saint_variant != "saint_s":
    print(f"\nWARNING: SAINT variant '{args.saint_variant}' uses intersample attention.")
    print(f"  Row embeddings are batch-dependent.")
print(f"\nNext steps:")
print(f"  1. Generate embeddings:")
print(f"     python models/saint/generate_embeddings.py --data_dir {args.data_dir} --checkpoint_dir {args.checkpoint_dir}")
print(f"  2. Train downstream classifier:")
print(f"     python train_downstream.py --embedding_dir models/saint/embeddings")
print("="*80)
