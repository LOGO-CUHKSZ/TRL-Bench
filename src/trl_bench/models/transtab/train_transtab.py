"""
TransTab Self-Supervised Training (Contrastive Learning)
Trains row embeddings using Vertical-Partition Contrastive Learning (VPCL).

Always self-supervised — labels are saved for downstream evaluation but
NOT used in training.

TransTab uses its own API (not ts3l) — handles raw DataFrames directly.
Preprocessing: median NaN fill + MinMaxScaler for numerical columns,
"__MISSING__" sentinel for categorical (TransTab tokenizes internally).
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../' * 2))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pickle
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import transtab

from trl_bench.utils.table_dataset import load_table_dataset, resolve_label_columns_cli, is_regression_label

print("=" * 80)
print("TransTab Pipeline - Self-Supervised Training (Contrastive Learning)")
print("=" * 80)
print("Trains embeddings using VPCL contrastive learning - always self-supervised")
print("Labels (if provided) are saved but NOT used in training")
print("=" * 80)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train TransTab using self-supervised contrastive learning')

# Data configuration
parser.add_argument('--data_dir', type=str, default='data/adult',
                    help='Directory containing data (supports canonical, legacy, or single CSV layouts)')
parser.add_argument('--checkpoint_dir', type=str, default='models/transtab/checkpoints',
                    help='Directory to save model checkpoints')
parser.add_argument('--label_column', type=str, default=None,
                    help='Name of the label column to exclude from training and save separately')
parser.add_argument('--label_policy', type=str, default='auto',
                    choices=['auto', 'none', 'manifest', 'cli'],
                    help="Label resolution policy")

# Training configuration
parser.add_argument('--val_ratio', type=float, default=0.2,
                    help='Ratio of training data for validation (default: 0.2)')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Batch size for training (default: 64)')
parser.add_argument('--num_epoch', type=int, default=50,
                    help='Number of epochs for training (default: 50)')
parser.add_argument('--random_seed', type=int, default=42,
                    help='Random seed for reproducibility (default: 42)')

# TransTab hyperparameters
parser.add_argument('--hidden_dim', type=int, default=512,
                    help='Hidden dimension (default: 512)')
parser.add_argument('--num_layer', type=int, default=2,
                    help='Number of transformer layers (default: 2)')
parser.add_argument('--num_attention_head', type=int, default=8,
                    help='Number of attention heads (default: 8)')
parser.add_argument('--num_partition', type=int, default=3,
                    help='Number of vertical partitions for VPCL (default: 3)')
parser.add_argument('--overlap_ratio', type=float, default=0.5,
                    help='Overlap ratio between partitions (default: 0.5)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate (default: 1e-4)')

# Split-aware options
parser.add_argument('--context_split', type=str, default='train',
                    help='Which split to use for training (default: train)')
parser.add_argument('--ignore_fingerprint', action='store_true',
                    help='Skip SHA256 fingerprint verification for canonical datasets')

args = parser.parse_args()

# Create checkpoint directory
os.makedirs(args.checkpoint_dir, exist_ok=True)


def detect_column_types(df):
    """Split DataFrame columns into categorical, numerical, and binary columns."""
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    bin_cols = [c for c in num_cols if df[c].dropna().nunique() <= 2]
    num_cols = [c for c in num_cols if c not in bin_cols]
    return cat_cols, num_cols, bin_cols


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
# 3. Detect column types and apply NaN guard
# ============================================================================
print(f"\n3. Detecting column types and filling NaN...")

feature_columns = list(X_train.columns)
cat_cols, num_cols, bin_cols = detect_column_types(X_train)

print(f"   Categorical columns ({len(cat_cols)}): {cat_cols[:5]}{'...' if len(cat_cols) > 5 else ''}")
print(f"   Numerical columns ({len(num_cols)}): {num_cols[:5]}{'...' if len(num_cols) > 5 else ''}")
print(f"   Binary columns ({len(bin_cols)}): {bin_cols[:5]}{'...' if len(bin_cols) > 5 else ''}")

# Compute medians on training data for reproducibility across splits
numeric_fill_medians = {}
for c in num_cols:
    median_val = X_train[c].median()
    numeric_fill_medians[c] = median_val
    X_train[c] = X_train[c].fillna(median_val).fillna(0)
    X_valid[c] = X_valid[c].fillna(median_val).fillna(0)

# MinMaxScaler on numerical columns (matches TransTab's load_data behavior)
num_scaler = None
if num_cols:
    num_scaler = MinMaxScaler()
    X_train[num_cols] = num_scaler.fit_transform(X_train[num_cols])
    X_valid[num_cols] = num_scaler.transform(X_valid[num_cols])

for c in bin_cols:
    median_val = X_train[c].median()
    numeric_fill_medians[c] = median_val
    # Cast to int after fill — TransTab's tokenizer multiplies bin values
    # by an embedding matrix and requires integer indices, not floats.
    X_train[c] = X_train[c].fillna(median_val).fillna(0).astype(int)
    X_valid[c] = X_valid[c].fillna(median_val).fillna(0).astype(int)

cat_fill_token = "__MISSING__"
for c in cat_cols:
    X_train[c] = X_train[c].fillna(cat_fill_token)
    X_valid[c] = X_valid[c].fillna(cat_fill_token)

print(f"   NaN guard applied (median fill + MinMaxScaler for numeric, '{cat_fill_token}' for categorical)")

# ============================================================================
# 4. Encode labels
# ============================================================================
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

# ============================================================================
# 5. Train TransTab contrastive learner
# ============================================================================
print(f"\n5. Training TransTab contrastive learner...")
print(f"   hidden_dim={args.hidden_dim}, num_layer={args.num_layer}, "
      f"num_attention_head={args.num_attention_head}")
print(f"   num_partition={args.num_partition}, overlap_ratio={args.overlap_ratio}, lr={args.lr}")

model, collate_fn = transtab.build_contrastive_learner(
    cat_cols, num_cols, bin_cols,
    supervised=False,
    num_partition=args.num_partition,
    overlap_ratio=args.overlap_ratio,
    hidden_dim=args.hidden_dim,
    num_layer=args.num_layer,
    num_attention_head=args.num_attention_head,
)

# TransTab's CL collator concatenates labels via pd.concat, which crashes
# on None.  Pass a dummy zero-filled Series so the collator is happy;
# the contrastive loss ignores labels when supervised=False.
dummy_y_train = pd.Series(0, index=X_train.index)
dummy_y_valid = pd.Series(0, index=X_valid.index)

# ---------------------------------------------------------------------------
# Pre-tokenize to avoid redundant BERT tokenization every epoch.
# The column partitions are deterministic, so we tokenize each partition
# view once and cache the tensors.  During training the collator just
# indexes into the cache — no more BertTokenizerFast calls.
# ---------------------------------------------------------------------------
import math
import time
from torch.utils.data import Dataset, DataLoader

def _build_partition_columns(all_cols, num_partition, overlap_ratio):
    """Reproduce TransTab's deterministic column partitioning."""
    sub_col_list = np.array_split(np.array(all_cols), num_partition)
    len_cols = len(sub_col_list[0])
    overlap = int(math.ceil(len_cols * overlap_ratio))
    views = []
    for i, sub_col in enumerate(sub_col_list):
        if overlap > 0 and i < num_partition - 1:
            sub_col = np.concatenate([sub_col, sub_col_list[i + 1][:overlap]])
        elif overlap > 0 and i == num_partition - 1:
            sub_col = np.concatenate([sub_col, sub_col_list[i - 1][-overlap:]])
        views.append(list(sub_col))
    return views

def _pretokenize_view(df, feature_extractor, batch_size=512):
    """Run feature_extractor on all rows of df in chunks, return per-row dicts."""
    n = len(df)
    all_rows = []
    for start in range(0, n, batch_size):
        chunk = df.iloc[start:start + batch_size]
        tokenized = feature_extractor(chunk)
        # Split batch dimension into individual rows
        bs = len(chunk)
        for i in range(bs):
            row = {}
            for key, val in tokenized.items():
                if val is None:
                    row[key] = None
                elif val.dim() == 0:
                    row[key] = val
                elif val.shape[0] == bs:
                    row[key] = val[i]  # per-row tensor
                else:
                    row[key] = val  # shared tensor (e.g. num_col_input_ids)
            all_rows.append(row)

    # Normalize keys across chunks: TransTab's feature extractor conditionally
    # adds keys (e.g. bin_att_mask only when binary features are non-zero).
    # Rows from different chunks can have different key sets and mixed
    # None/tensor values for the same key.  Replace None with zero tensors
    # so the collation function can stack consistently.
    all_keys = set()
    for row in all_rows:
        all_keys.update(row.keys())
    for key in all_keys:
        # Find a reference non-None tensor for this key
        ref = None
        for row in all_rows:
            v = row.get(key)
            if v is not None:
                ref = v
                break
        if ref is None:
            continue  # all rows have None for this key — fine
        for row in all_rows:
            if key not in row or row[key] is None:
                row[key] = torch.zeros_like(ref)

    return all_rows

class PreTokenizedCLDataset(Dataset):
    """Dataset that holds pre-tokenized partition views."""
    def __init__(self, views_data):
        # views_data: list of lists, views_data[view_idx][row_idx] = dict of tensors
        self.views_data = views_data
        self.n_rows = len(views_data[0])

    def __len__(self):
        return self.n_rows

    def __getitem__(self, idx):
        return idx

def _pretokenized_collate_fn(views_data, shared_keys):
    """Build a collator that batches pre-tokenized rows by index."""
    def collate(indices):
        input_sub_x = []
        for view_rows in views_data:
            batch = {}
            for key in view_rows[0]:
                vals = [view_rows[i][key] for i in indices]
                if vals[0] is None:
                    batch[key] = None
                elif key in shared_keys:
                    batch[key] = vals[0]  # same for all rows
                elif vals[0].dim() == 0:
                    batch[key] = torch.stack(vals, dim=0)
                else:
                    # Pad variable-length tensors (e.g. input_ids from
                    # different tokenization chunks have different lengths)
                    max_len = max(v.shape[-1] for v in vals)
                    if all(v.shape[-1] == max_len for v in vals):
                        batch[key] = torch.stack(vals, dim=0)
                    else:
                        padded = []
                        for v in vals:
                            pad_size = max_len - v.shape[-1]
                            if pad_size > 0:
                                v = torch.nn.functional.pad(v, (0, pad_size))
                            padded.append(v)
                        batch[key] = torch.stack(padded, dim=0)
            input_sub_x.append(batch)
        x = {'input_sub_x': input_sub_x}
        # dummy y (contrastive loss ignores it when unsupervised)
        y = pd.Series(0, index=range(len(indices)))
        return x, y
    return collate

feature_extractor = model.input_encoder.feature_extractor
all_cols = list(X_train.columns)
partition_views = _build_partition_columns(all_cols, args.num_partition, args.overlap_ratio)

print(f"\n   Pre-tokenizing {len(X_train)} rows x {len(partition_views)} views...")
t_tok = time.time()
views_data = []
# Keys that are shared across all rows (e.g. column name token IDs)
shared_keys = {'num_col_input_ids', 'num_att_mask'}
for vi, view_cols in enumerate(partition_views):
    sub_df = X_train[view_cols]
    rows = _pretokenize_view(sub_df, feature_extractor, batch_size=512)
    views_data.append(rows)
    print(f"     View {vi}: {len(view_cols)} cols -> {len(rows)} rows tokenized")
print(f"   Pre-tokenization done in {time.time() - t_tok:.1f}s")

pt_dataset = PreTokenizedCLDataset(views_data)
pt_collate = _pretokenized_collate_fn(views_data, shared_keys)

# Custom training loop using pre-tokenized data
print(f"\nTraining for {args.num_epoch} epochs (pre-tokenized)...")
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
train_loader = DataLoader(
    pt_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=pt_collate
)

start_time = time.time()
best_loss = float('inf')
best_epoch = -1
for epoch in range(args.num_epoch):
    epoch_loss = 0.0
    n_batches = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        _, loss = model(batch_x, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1
    avg_loss = epoch_loss / max(n_batches, 1)
    elapsed = time.time() - start_time
    if not math.isnan(avg_loss) and avg_loss < best_loss:
        best_loss = avg_loss
        best_epoch = epoch + 1
        model.save(args.checkpoint_dir)
    print(f"   Epoch {epoch+1}/{args.num_epoch}  loss={avg_loss:.4f}  best={best_loss:.4f}@{best_epoch}  elapsed={elapsed:.0f}s")

# Reload best checkpoint (in case later epochs degraded or went NaN)
if best_epoch > 0 and best_epoch < args.num_epoch:
    best_state = torch.load(os.path.join(args.checkpoint_dir, 'pytorch_model.bin'), map_location='cpu')
    model.load_state_dict(best_state)
    print(f"   Loaded best checkpoint from epoch {best_epoch} (loss={best_loss:.4f})")
elif best_epoch == args.num_epoch:
    pass  # already saved at last epoch
else:
    print("   WARNING: no valid checkpoint saved (all epochs produced NaN loss)")
    model.save(args.checkpoint_dir)

# ============================================================================
# 6. Save training configuration
# ============================================================================
config_dict = {
    # Training configuration
    'val_ratio': args.val_ratio,
    'batch_size': args.batch_size,
    'num_epoch': args.num_epoch,
    'hidden_dim': args.hidden_dim,
    'num_layer': args.num_layer,
    'num_attention_head': args.num_attention_head,
    'num_partition': args.num_partition,
    'overlap_ratio': args.overlap_ratio,
    'lr': args.lr,

    # Model info
    'model_name': 'TransTab',
    'training_mode': 'contrastive',
    'note': 'Self-supervised embeddings using VPCL contrastive learning',

    # Column types
    'cat_cols': cat_cols,
    'num_cols': num_cols,
    'bin_cols': bin_cols,
    'feature_columns': feature_columns,

    # NaN fill values (computed on training data, reused at inference)
    'numeric_fill_medians': numeric_fill_medians,
    'cat_fill_token': cat_fill_token,
    'num_scaler': num_scaler,

    # Label configuration
    'label_columns': label_cols,
    'label_encoders': label_encoders,
    'label_task_types': {c: dataset.label_task_types.get(c, '') for c in label_cols},
    'label_policy': args.label_policy,
    'has_label': has_labels,

    # Backward compat keys (singular)
    'label_column': label_cols[0] if label_cols else None,
    'label_encoder': label_encoders.get(label_cols[0]) if label_cols else None,

    # Split-aware metadata
    'context_split': train_split_name,
    'dataset_source_path': dataset.source_path,
    'dataset_sha256': dataset.fingerprint,

    # Metadata
    'self_sufficient': True,
    'detected_by': 'TransTab',
}

config_file = os.path.join(args.checkpoint_dir, "training_config.pkl")
with open(config_file, 'wb') as f:
    pickle.dump(config_dict, f)

print("\n" + "=" * 80)
print("Self-supervised training completed successfully!")
print("=" * 80)
print(f"\nTraining mode:")
if has_labels:
    print(f"  - Trained on {len(cat_cols) + len(num_cols) + len(bin_cols)} features (excluding {len(label_cols)} label(s))")
    print(f"  - Label columns {label_cols} saved for downstream evaluation")
else:
    print(f"  - Trained on ALL {len(cat_cols) + len(num_cols) + len(bin_cols)} columns")
    print(f"  - No label column (pure unsupervised)")
print(f"\nSaved files:")
print(f"  - Checkpoint: {args.checkpoint_dir}/ckpt_best.pt")
print(f"  - Training config: {config_file}")
print(f"\nNext steps:")
print(f"  1. Generate embeddings:")
print(f"     python models/transtab/generate_embeddings.py --data_dir {args.data_dir} --checkpoint_dir {args.checkpoint_dir}")
print("=" * 80)
