"""
TabPFN Embedding Generation
Generates embeddings using TabPFN for both supervised and self-supervised learning

Supports multiple input modes:
- Pre-split: --data_dir with train.csv and test.csv (or canonical dataset.json)
- Single CSV: --input with a single file to be split

Three Modes of Operation:
1. Supervised mode (--mode supervised --label_column <name>):
   - Uses specified label column as target
   - Model learns label-aware embeddings

2. Self-supervised mode without label (--mode self-supervised):
   - Uses ALL columns as features
   - Model trained with dummy target (no label information)
   - Generates purely unsupervised embeddings

3. Self-supervised mode with label (--mode self-supervised --label_column <name>):
   - Label column is saved separately
   - Embeddings computed on remaining columns
   - Model trained with dummy target (label-agnostic embeddings)
   - Useful for evaluation while maintaining unsupervised training

Output: JSON metadata (metadata.json) with v2.0 split-aware format
"""

import pandas as pd
import numpy as np
import os
import argparse
import logging
import sys
from sklearn.preprocessing import LabelEncoder

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../' * 2))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trl_bench.utils.unified_embedding_format import RowEmbeddingMetadataV2, save_split_embeddings, encode_label_column
from trl_bench.utils.table_dataset import (
    load_table_dataset,
    resolve_label_columns_cli,
    PretrainedPreprocessor,
    is_regression_label,
)

logger = logging.getLogger(__name__)

print("="*80)
print("TabPFN Pipeline - Generating Embeddings")
print("="*80)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate embeddings using TabPFN')

# Data input (mutually exclusive)
input_group = parser.add_mutually_exclusive_group(required=True)
input_group.add_argument('--data_dir', type=str, default=None,
                    help='Directory containing data (canonical, legacy, or single CSV)')
input_group.add_argument('--input', type=str, default=None,
                    help='Single CSV file (will be split internally)')

# Output configuration
parser.add_argument('--embedding_dir', type=str, default='embeddings/row_prediction/TabPFN',
                    help='Directory to save embeddings (default: embeddings/row_prediction/TabPFN)')

# Mode configuration
parser.add_argument('--label_column', type=str, default=None,
                    help='Name of the label column. Supervised mode: used as target. Self-supervised mode: saved separately but not used in training.')
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
parser.add_argument('--mode', type=str, default='auto', choices=['auto', 'supervised', 'self-supervised'],
                    help='Mode: auto (detect from resolved labels), supervised, or self-supervised (default: auto)')

# Processing configuration
parser.add_argument('--batch_size', type=int, default=256,
                    help='Batch size for embedding generation (default: 256)')

# Split configuration (only for --input mode)
parser.add_argument('--split_ratio', type=float, default=0.8,
                    help='Train/test split ratio for --input mode (default: 0.8)')
parser.add_argument('--random_seed', type=int, default=42,
                    help='Random seed for reproducibility (default: 42)')

# TabPFN configuration
parser.add_argument('--n_estimators', type=int, default=8,
                    help='Number of TabPFN estimators (default: 8)')
parser.add_argument('--device', type=str, default='auto',
                    help='Device to use: auto, cuda, cpu (default: auto)')

# Split-aware options
parser.add_argument('--preprocess_fit_scope', type=str, default='all',
                    help='Which split to fit preprocessor on: "train" or "all" (default: all)')
parser.add_argument('--context_split', type=str, default='train',
                    help='Which split to use as context for model.fit() (default: train)')
parser.add_argument('--ignore_fingerprint', action='store_true',
                    help='Skip SHA256 fingerprint verification for canonical datasets')

args = parser.parse_args()

# Create embedding directory
os.makedirs(args.embedding_dir, exist_ok=True)

# ============================================================================
# 1. Load data via TableDataset
# ============================================================================
print(f"\n1. Loading data...")
data_path = args.data_dir or args.input
label_columns_cli = resolve_label_columns_cli(args.label_column, args.label_policy)

dataset = load_table_dataset(
    data_path,
    label_columns_cli=label_columns_cli,
    ignore_fingerprint=args.ignore_fingerprint,
)

# Resolve effective label columns (multi-label support)
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

# Determine mode after label resolution
if args.mode == 'auto':
    if len(label_cols) == 1:
        mode = 'supervised'
    elif label_cols:
        mode = 'self-supervised'
        print(f"   Multi-label ({len(label_cols)} labels): auto-selecting self-supervised mode")
    else:
        mode = 'self-supervised'
else:
    mode = args.mode

print(f"\nMode: {mode.upper()}")
if mode == 'supervised' and not label_cols:
    raise ValueError(
        "Supervised mode requires a resolved label column."
    )

# --input single-CSV mode: split into train/test (matches old behavior)
if args.input is not None and dataset.split_names == ["all"]:
    dataset.apply_train_test_split(
        train_ratio=args.split_ratio,
        random_seed=args.random_seed,
        stratify_on_label=has_labels,
    )
    print(f"   Split single CSV into train/test (ratio={args.split_ratio}, seed={args.random_seed})")

print(f"   Dataset: {dataset}")
data_source = data_path

# ============================================================================
# 2. Fit PretrainedPreprocessor
# ============================================================================
print(f"\n2. Fitting PretrainedPreprocessor (scope: {args.preprocess_fit_scope})...")

preprocessor = PretrainedPreprocessor()
if args.preprocess_fit_scope == "all":
    fit_view = dataset.get_full()
elif args.preprocess_fit_scope in dataset.split_names:
    fit_view = dataset.get_split(args.preprocess_fit_scope)
else:
    fallback = "train" if "train" in dataset.split_names else dataset.split_names[0]
    print(f"   Warning: --preprocess_fit_scope '{args.preprocess_fit_scope}' not found "
          f"in {dataset.split_names}, falling back to '{fallback}'")
    fit_view = dataset.get_split(fallback)

preprocessor.fit(fit_view.X)

category_cols = preprocessor.category_cols
continuous_cols = preprocessor.continuous_cols
categorical_indices = preprocessor.categorical_indices

print(f"   Categorical columns ({len(category_cols)}): {category_cols[:5]}{'...' if len(category_cols) > 5 else ''}")
print(f"   Continuous columns ({len(continuous_cols)}): {continuous_cols[:5]}{'...' if len(continuous_cols) > 5 else ''}")
print(f"   Categorical indices: {categorical_indices[:10]}{'...' if len(categorical_indices) > 10 else ''}")

# ============================================================================
# 3. Prepare context (training) data and labels
# ============================================================================
print(f"\n3. Preparing context data (split: {args.context_split})...")

# Determine context split
if args.context_split in dataset.split_names:
    ctx_split_name = args.context_split
elif "train" in dataset.split_names:
    ctx_split_name = "train"
else:
    ctx_split_name = dataset.split_names[0]

ctx_view = dataset.get_split(ctx_split_name)
X_ctx = preprocessor.transform(ctx_view.X)

# Build per-column label encoders
label_encoders = {}
if has_labels:
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

# Prepare context labels for model.fit()
if (mode == 'supervised'
    and len(label_cols) == 1
    and label_encoders.get(label_cols[0]) is not None):
    # Single classification label + supervised mode: real labels
    y_col = ctx_view.y[label_cols[0]] if isinstance(ctx_view.y, pd.DataFrame) else ctx_view.y
    y_ctx = label_encoders[label_cols[0]].transform(y_col)
    print(f"   Supervised: using label column '{label_cols[0]}'")
else:
    # Multi-label, self-supervised, or regression: dummy labels
    y_ctx = np.zeros(len(ctx_view), dtype=int)
    y_ctx[len(ctx_view) // 2:] = 1
    if len(label_cols) > 1:
        print(f"   Multi-label ({len(label_cols)} labels): using dummy context labels (effective self-supervised)")
    elif not label_cols:
        print(f"   Self-supervised: using all columns, dummy target")
    else:
        print(f"   Self-supervised with label column '{label_cols[0]}' (saved but not used in training)")

# ============================================================================
# 4. Initialize and fit TabPFN
# ============================================================================
print(f"\n4. Initializing TabPFN model...")
print(f"   Number of estimators: {args.n_estimators}")
print(f"   Device: {args.device}")

from tabpfn_extensions import TabPFNClassifier

model = TabPFNClassifier(
    n_estimators=args.n_estimators,
    categorical_features_indices=categorical_indices if categorical_indices else None,
    device=args.device,
    random_state=args.random_seed,
    ignore_pretraining_limits=True,
    memory_saving_mode=True,
    fit_mode='low_memory'
)

print(f"   Fitting on context split '{ctx_split_name}' ({len(X_ctx)} samples)...")
model.fit(X_ctx, y_ctx)
print(f"   Model fitted")

# ============================================================================
# 5. Generate embeddings per split
# ============================================================================
print(f"\n5. Generating embeddings per split...")

emb_dict = {}
lbl_dict = {}
idx_dict = {}

for split_name in dataset.split_names:
    view = dataset.get_split(split_name)
    print(f"\n   Processing split '{split_name}' ({len(view)} samples)...")

    X_enc = preprocessor.transform(view.X)
    embeddings = model.get_embeddings(X_enc, data_source='test')

    # If n_estimators > 1, average over estimators
    if embeddings.ndim == 3:  # (n_estimators, n_samples, embedding_dim)
        embeddings = embeddings.mean(axis=0)

    emb_dict[split_name] = embeddings
    print(f"   Embeddings shape: {embeddings.shape}")

    # Encode labels if available (per-column)
    if has_labels and view.y is not None:
        per_col = {}
        for col in label_cols:
            y_col = view.y[col] if isinstance(view.y, pd.DataFrame) else view.y
            le = label_encoders.get(col)
            per_col[col] = encode_label_column(y_col, le, split_name, col, logger)
        lbl_dict[split_name] = per_col

    # Row indices if available
    if view.row_indices is not None:
        idx_dict[split_name] = view.row_indices

# ============================================================================
# 6. Save with v2.0 split-aware format
# ============================================================================
print(f"\n6. Saving embeddings...")

first_split = next(iter(emb_dict.values()))
feature_columns = list(dataset.feature_columns)

generation_config = {
    'batch_size': args.batch_size,
    'data_source': data_source,
    'mode': mode,
    'label_policy': args.label_policy,
    'n_estimators': args.n_estimators,
    'device': args.device,
    'preprocess_fit_scope': args.preprocess_fit_scope,
    'context_split': ctx_split_name,
}
if args.input is not None:
    generation_config['split_ratio'] = args.split_ratio
    generation_config['random_seed'] = args.random_seed

dataset_info = {
    'source_path': dataset.source_path,
    'layout': dataset.layout,
}
if dataset.fingerprint:
    dataset_info['fingerprint_sha256'] = dataset.fingerprint

metadata = RowEmbeddingMetadataV2(
    model_name='TabPFN',
    embedding_dim=first_split.shape[1],
    label_columns=label_cols,
    label_task_types={c: dataset.label_task_types.get(c, '') for c in label_cols},
    feature_columns=feature_columns,
    generation_config=generation_config,
    dataset=dataset_info,
    checkpoint_path=None,  # TabPFN doesn't use checkpoints
)

output_files = save_split_embeddings(
    embeddings=emb_dict,
    metadata=metadata,
    output_dir=args.embedding_dir,
    labels=lbl_dict if lbl_dict else None,
    row_indices=idx_dict if idx_dict else None,
)

print(f"   Saved:")
for file_type, file_path in output_files.items():
    print(f"      {file_path}")

print("\n" + "="*80)
print("Embedding generation completed successfully!")
print("="*80)
print(f"\nSummary:")
print(f"  Mode: {mode}")
print(f"  Embedding dimension: {first_split.shape[1]}")
for name, emb in emb_dict.items():
    print(f"  {name} embeddings: {emb.shape}")
if has_labels:
    print(f"  Label columns: {label_cols}")
    for col in label_cols:
        le = label_encoders.get(col)
        if le is not None:
            print(f"    '{col}': classification ({le.classes_.tolist()})")
        else:
            print(f"    '{col}': regression")
else:
    print(f"  No labels (pure unsupervised)")

print(f"\nAll files in: {args.embedding_dir}/")
if has_labels:
    print(f"\nNext step:")
    print(f"  python downstream_tasks/row_prediction/train_downstream.py --embedding_dir {args.embedding_dir}")
print("="*80)
