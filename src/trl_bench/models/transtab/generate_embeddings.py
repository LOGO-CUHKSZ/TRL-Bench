"""
TransTab Embedding Generation
Loads raw data, applies saved NaN fill values and MinMaxScaler, generates
embeddings using trained TransTab encoder.

Supports multiple input modes:
- Pre-split: --data_dir with train.csv and test.csv (or canonical dataset.json)
- Single CSV: --input with a single file to be split

Output: JSON metadata (metadata.json) with v2.0 split-aware format
"""

import pandas as pd
import numpy as np
import pickle
import os
import argparse
import logging
import torch
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../' * 2))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import transtab
from trl_bench.utils.unified_embedding_format import RowEmbeddingMetadataV2, save_split_embeddings, encode_label_column
from trl_bench.utils.table_dataset import load_table_dataset

logger = logging.getLogger(__name__)

print("=" * 80)
print("TransTab Pipeline - Generating Embeddings")
print("=" * 80)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate embeddings using trained TransTab model')

# Data input (mutually exclusive)
input_group = parser.add_mutually_exclusive_group(required=True)
input_group.add_argument('--data_dir', type=str, default=None,
                    help='Directory containing data (canonical, legacy, or single CSV)')
input_group.add_argument('--input', type=str, default=None,
                    help='Single CSV file (will be split internally)')

# Model configuration
parser.add_argument('--checkpoint_dir', type=str, default='models/transtab/checkpoints',
                    help='Directory containing model checkpoints')
parser.add_argument('--embedding_dir', type=str, default='models/transtab/embeddings',
                    help='Directory to save embeddings')

# Processing configuration
parser.add_argument('--batch_size', type=int, default=256,
                    help='Batch size for embedding generation (default: 256)')

# Split configuration (only for --input mode)
parser.add_argument('--split_ratio', type=float, default=0.8,
                    help='Train/test split ratio for --input mode (default: 0.8)')
parser.add_argument('--random_seed', type=int, default=42,
                    help='Random seed for splitting (default: 42)')

# Split-aware options
parser.add_argument('--ignore_fingerprint', action='store_true',
                    help='Skip SHA256 fingerprint verification for canonical datasets')

args = parser.parse_args()

# Create embedding directory
os.makedirs(args.embedding_dir, exist_ok=True)

# ============================================================================
# 1. Load training configuration
# ============================================================================
print(f"\n1. Loading training configuration...")
config_file = os.path.join(args.checkpoint_dir, "training_config.pkl")
with open(config_file, 'rb') as f:
    train_config = pickle.load(f)

# Load multi-label config with backward compat
label_cols = train_config.get('label_columns')
if label_cols is None:
    old = train_config.get('label_column')
    label_cols = [old] if old else []

label_encoders = train_config.get('label_encoders')
if label_encoders is None:
    old_enc = train_config.get('label_encoder')
    label_encoders = {label_cols[0]: old_enc} if label_cols else {}

has_labels = bool(label_cols)

cat_cols = train_config['cat_cols']
num_cols = train_config['num_cols']
bin_cols = train_config['bin_cols']
feature_columns = train_config.get('feature_columns', cat_cols + num_cols + bin_cols)

# Extract saved NaN fill values
numeric_fill_medians = train_config.get('numeric_fill_medians', {})
cat_fill_token = train_config.get('cat_fill_token', '__MISSING__')
num_scaler = train_config.get('num_scaler', None)

print(f"   Categorical columns: {len(cat_cols)}")
print(f"   Numerical columns: {len(num_cols)}")
print(f"   Binary columns: {len(bin_cols)}")
if has_labels:
    print(f"   Label columns: {label_cols}")
else:
    print(f"   No label column (pure unsupervised)")

# Verify dataset fingerprint if available
dataset_sha256 = train_config.get('dataset_sha256', '')

# ============================================================================
# 2. Load data via TableDataset
# ============================================================================
print(f"\n2. Loading data...")
data_path = args.data_dir or args.input
label_columns_cli = label_cols

dataset = load_table_dataset(
    data_path,
    label_columns_cli=label_columns_cli,
    ignore_fingerprint=args.ignore_fingerprint,
)

# Recompute label info from what TableDataset actually resolved
label_cols = list(dataset.label_columns)
has_labels = bool(label_cols)

# --input single-CSV mode: split into train/test
if args.input is not None and dataset.split_names == ["all"]:
    dataset.apply_train_test_split(
        train_ratio=args.split_ratio,
        random_seed=args.random_seed,
        stratify_on_label=has_labels,
    )
    print(f"   Split single CSV into train/test (ratio={args.split_ratio}, seed={args.random_seed})")

# Verify fingerprint
if dataset_sha256 and dataset.fingerprint and dataset_sha256 != dataset.fingerprint:
    logger.warning(
        "Dataset fingerprint mismatch: training used %s, current dataset is %s",
        dataset_sha256[:16] + "...",
        dataset.fingerprint[:16] + "...",
    )

print(f"   Dataset: {dataset}")
data_source = data_path

# ============================================================================
# 3. Build TransTab encoder
# ============================================================================
print(f"\n3. Loading TransTab encoder from checkpoint...")
enc = transtab.build_encoder(
    categorical_columns=cat_cols,
    numerical_columns=num_cols,
    binary_columns=bin_cols,
    hidden_dim=train_config['hidden_dim'],
    num_layer=train_config['num_layer'],
    checkpoint=args.checkpoint_dir,
)
print(f"   Encoder loaded")

# ============================================================================
# 4. Generate embeddings per split
# ============================================================================
print(f"\n4. Generating embeddings per split...")

emb_dict = {}
lbl_dict = {}
idx_dict = {}

for split_name in dataset.split_names:
    view = dataset.get_split(split_name)
    print(f"\n   Processing split '{split_name}' ({len(view)} samples)...")

    X = view.X.copy()

    # Enforce training feature order: all saved columns must be present
    missing = [c for c in feature_columns if c not in X.columns]
    if missing:
        raise ValueError(
            f"Split '{split_name}' is missing {len(missing)} training feature column(s): "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
        )
    X = X[feature_columns]

    # Apply NaN fill using saved medians/token from training
    for c in num_cols:
        if c in X.columns:
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

    # Batch inference
    embeddings = []
    n = len(X)
    for start in range(0, n, args.batch_size):
        end = min(start + args.batch_size, n)
        batch_df = X.iloc[start:end]
        with torch.no_grad():
            batch_emb = enc(batch_df)
        embeddings.append(batch_emb.cpu().numpy())

        if (start // args.batch_size + 1) % 10 == 0:
            print(f"      Processed {end} samples...", end='\r')

    emb = np.vstack(embeddings)
    emb_dict[split_name] = emb
    print(f"   Embeddings shape: {emb.shape}")

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
# 5. Save with v2.0 split-aware format
# ============================================================================
print(f"\n5. Saving embeddings...")

first_split = next(iter(emb_dict.values()))
# Use the saved training column order (already enforced during inference above)
saved_feature_columns = feature_columns

generation_config = {
    'batch_size': args.batch_size,
    'data_source': data_source,
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
    model_name='TransTab',
    embedding_dim=first_split.shape[1],
    label_columns=label_cols,
    label_task_types=train_config.get('label_task_types', {}),
    feature_columns=saved_feature_columns,
    generation_config=generation_config,
    dataset=dataset_info,
    checkpoint_path=args.checkpoint_dir,
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

print("\n" + "=" * 80)
print("Embedding generation completed successfully!")
print("=" * 80)
print(f"\nSummary:")
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
print("=" * 80)
