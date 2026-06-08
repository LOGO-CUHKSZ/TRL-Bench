"""
TabICL Embedding Generation
Generates row-level embeddings using TabICL's internal Stage 2 representations.

TabICL is a tabular foundation model (ICML 2025) that uses in-context learning.
Unlike TabPFN which exposes get_embeddings(), TabICL requires accessing its
internal col_embedder + row_interactor stages to extract 512-dim row embeddings.

Embeddings are extracted after Stage 2 (row_interactor), which is label-free:
  Stage 1: col_embedder  -- maps raw features into column-level representations
  Stage 2: row_interactor -- SetTransformer capturing row interactions -> (N, 512)
  Stage 3: predictor      -- adds label information (NOT used for embeddings)

Supports multiple input modes:
- Pre-split: --data_dir with train.csv and test.csv (or canonical dataset.json)
- Single CSV: --input with a single file to be split

Three Modes of Operation:
1. Supervised mode (--mode supervised --label_column <name>):
   - Label column excluded from features, saved with embeddings
   - Note: embeddings are identical to self-supervised (labels enter at Stage 3)

2. Self-supervised mode without label (--mode self-supervised):
   - Uses ALL columns as features
   - Generates purely unsupervised embeddings

3. Self-supervised mode with label (--mode self-supervised --label_column <name>):
   - Label column excluded from features and saved separately
   - Embeddings computed on remaining columns
   - Useful for evaluation while maintaining unsupervised training

Output: JSON metadata (metadata.json) with v2.0 split-aware format
"""

import pandas as pd
import numpy as np
import os
import argparse
import logging
import sys
import torch
from sklearn.preprocessing import LabelEncoder

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../' * 2))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trl_bench.utils.unified_embedding_format import RowEmbeddingMetadataV2, save_split_embeddings, encode_label_column
from trl_bench.utils.table_dataset import load_table_dataset, resolve_label_columns_cli, is_regression_label

logger = logging.getLogger(__name__)

print("="*80)
print("TabICL Pipeline - Generating Embeddings")
print("="*80)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate embeddings using TabICL')

# Data input (mutually exclusive)
input_group = parser.add_mutually_exclusive_group(required=True)
input_group.add_argument('--data_dir', type=str, default=None,
                    help='Directory containing data (canonical, legacy, or single CSV)')
input_group.add_argument('--input', type=str, default=None,
                    help='Single CSV file (will be split internally)')

# Output configuration
parser.add_argument('--embedding_dir', type=str, default='embeddings/row_prediction/TabICL',
                    help='Directory to save embeddings (default: embeddings/row_prediction/TabICL)')

# Mode configuration
parser.add_argument('--label_column', type=str, default=None,
                    help='Name of the label column. Excluded from features; saved with embeddings.')
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

# Split configuration (only for --input mode)
parser.add_argument('--split_ratio', type=float, default=0.8,
                    help='Train/test split ratio for --input mode (default: 0.8)')
parser.add_argument('--random_seed', type=int, default=42,
                    help='Random seed for reproducibility (default: 42)')

# TabICL configuration
parser.add_argument('--n_estimators', type=int, default=1,
                    help='Number of TabICL estimators (default: 1). '
                         'Values >1 use feature shuffles with RoPE, making embedding '
                         'averaging across shuffles semantically incoherent.')
parser.add_argument('--checkpoint_version', type=str, default='tabicl-classifier-v1.1-0506.ckpt',
                    help='TabICL checkpoint version (default: tabicl-classifier-v1.1-0506.ckpt)')
parser.add_argument('--device', type=str, default='auto',
                    help='Device to use: auto, cuda, cpu (default: auto)')

# Split-aware options
parser.add_argument('--context_split', type=str, default='train',
                    help='Which split to use as context for model.fit() (default: train)')
parser.add_argument('--ignore_fingerprint', action='store_true',
                    help='Skip SHA256 fingerprint verification for canonical datasets')

args = parser.parse_args()

# Create embedding directory
os.makedirs(args.embedding_dir, exist_ok=True)

# Resolve device
if args.device == 'auto':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
    device = args.device

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
    raise ValueError("Supervised mode requires a resolved label column.")

# --input single-CSV mode: split into train/test
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
# 2. Prepare context (training) data and labels
# ============================================================================
print(f"\n2. Preparing context data (split: {args.context_split})...")

# Determine context split
if args.context_split in dataset.split_names:
    ctx_split_name = args.context_split
elif "train" in dataset.split_names:
    ctx_split_name = "train"
else:
    ctx_split_name = dataset.split_names[0]

ctx_view = dataset.get_split(ctx_split_name)
X_ctx = ctx_view.X

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
else:
    print(f"   Self-supervised: using all columns, no labels")

# ============================================================================
# 3. Initialize and fit TabICL
# ============================================================================
print(f"\n3. Initializing TabICL model...")
print(f"   Number of estimators: {args.n_estimators}")
print(f"   Checkpoint: {args.checkpoint_version}")
print(f"   Device: {device}")

from tabicl import TabICLClassifier

clf = TabICLClassifier(
    n_estimators=args.n_estimators,
    device=device,
)

# Prepare context labels for fit()
# Note: labels do NOT affect Stage 2 embeddings (labels enter at Stage 3 only),
# but we pass real labels in supervised mode for consistency with TabPFN.
n_ctx = len(X_ctx)
if (mode == 'supervised'
    and len(label_cols) == 1
    and label_encoders.get(label_cols[0]) is not None):
    # Single classification label + supervised mode: real labels
    y_col = ctx_view.y[label_cols[0]] if isinstance(ctx_view.y, pd.DataFrame) else ctx_view.y
    y_ctx = label_encoders[label_cols[0]].transform(y_col)
    print(f"   Supervised: using label column '{label_cols[0]}'")
else:
    # Multi-label, self-supervised, or regression: dummy labels
    # TabICLClassifier requires >=2 classes
    y_ctx = np.zeros(n_ctx, dtype=int)
    y_ctx[n_ctx // 2:] = 1  # Balanced 2-class dummy labels
    if len(label_cols) > 1:
        print(f"   Multi-label ({len(label_cols)} labels): using dummy context labels (effective self-supervised)")
    elif not label_cols:
        print(f"   Self-supervised: using all columns, dummy target")
    else:
        print(f"   Self-supervised with label column '{label_cols[0]}' (saved but not used in training)")

print(f"   Fitting on context split '{ctx_split_name}' ({n_ctx} samples)...")
clf.fit(X_ctx, y_ctx)
print(f"   Model fitted")

# ============================================================================
# 4. Extract Stage 2 embeddings via forward hook
# ============================================================================
print(f"\n4. Extracting Stage 2 embeddings via forward hook...")

model = clf.model_
train_size = n_ctx

captured_reps = []

def capture_row_reps(module, input, output):
    """Capture row_interactor output (Stage 2 embeddings)."""
    captured_reps.append(output.detach().cpu())

MAX_TEST_CHUNK = 5000

emb_dict = {}
lbl_dict = {}
idx_dict = {}

hook_handle = model.row_interactor.register_forward_hook(capture_row_reps)
try:
    for split_name in dataset.split_names:
        view = dataset.get_split(split_name)
        X_split = view.X
        split_size = len(X_split)
        print(f"\n   Processing split '{split_name}' ({split_size} samples)...")

        if split_name == ctx_split_name:
            # For the context split, we need a minimal test call to extract train embeddings
            # Use 1 row from another split if available, otherwise use first context row
            other_splits = [s for s in dataset.split_names if s != ctx_split_name]
            if other_splits:
                probe_view = dataset.get_split(other_splits[0])
                probe_X = probe_view.X.iloc[:1]
            else:
                probe_X = X_split.iloc[:1]

            captured_reps.clear()
            with torch.no_grad():
                clf.predict_proba(probe_X)
            ctx_embeddings = captured_reps[0][0, :train_size].numpy()
            emb_dict[split_name] = ctx_embeddings
            print(f"   Embeddings shape: {ctx_embeddings.shape}")

        elif split_size <= MAX_TEST_CHUNK:
            # Small enough to process in one pass
            captured_reps.clear()
            with torch.no_grad():
                clf.predict_proba(X_split)
            reps = captured_reps[0][0].numpy()
            split_embeddings = reps[train_size:]
            emb_dict[split_name] = split_embeddings
            print(f"   Embeddings shape: {split_embeddings.shape}")

        else:
            # Process in chunks
            chunks = []
            n_chunks = (split_size + MAX_TEST_CHUNK - 1) // MAX_TEST_CHUNK

            for i in range(n_chunks):
                start = i * MAX_TEST_CHUNK
                end = min(start + MAX_TEST_CHUNK, split_size)

                captured_reps.clear()
                with torch.no_grad():
                    clf.predict_proba(X_split.iloc[start:end])

                chunk_emb = captured_reps[0][0, train_size:].numpy()
                chunks.append(chunk_emb)
                print(f"   Processed chunk {i+1}/{n_chunks}: rows {start}-{end-1}")

            split_embeddings = np.concatenate(chunks, axis=0)
            emb_dict[split_name] = split_embeddings
            print(f"   Embeddings shape: {split_embeddings.shape}")

        # Encode labels if available (multi-column)
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

finally:
    hook_handle.remove()

# ============================================================================
# 5. Save with v2.0 split-aware format
# ============================================================================
print(f"\n5. Saving embeddings...")

first_split = next(iter(emb_dict.values()))
feature_columns = list(dataset.feature_columns)

generation_config = {
    'data_source': data_source,
    'mode': mode,
    'label_policy': args.label_policy,
    'n_estimators': args.n_estimators,
    'checkpoint_version': args.checkpoint_version,
    'device': device,
    'extraction_point': 'Stage 2 (row_interactor)',
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
    model_name='TabICL',
    embedding_dim=first_split.shape[1],
    label_columns=label_cols,
    label_task_types={c: dataset.label_task_types.get(c, '') for c in label_cols},
    feature_columns=feature_columns,
    generation_config=generation_config,
    dataset=dataset_info,
    checkpoint_path=args.checkpoint_version,
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
print(f"  Embedding dimension: {first_split.shape[1]} (4 CLS tokens x 128)")
for name, emb in emb_dict.items():
    print(f"  {name} embeddings: {emb.shape}")
print(f"  Label columns: {label_cols if label_cols else 'None'}")

if mode == 'self-supervised':
    if not label_cols:
        print(f"  Note: All columns used as features (no label info)")
    else:
        print(f"  Note: Labels {label_cols} saved but NOT used for embeddings")
        print(f"        Embeddings are label-agnostic (labels enter at Stage 3 only)")
        for col in label_cols:
            le = label_encoders.get(col)
            if le is not None:
                print(f"  '{col}': {len(le.classes_)} classes")
            else:
                print(f"  '{col}': regression (raw values saved)")
else:
    print(f"  Note: Supervised/self-supervised produce identical embeddings")
    print(f"        (labels only enter at Stage 3, which is not used)")
    for col in label_cols:
        le = label_encoders.get(col)
        if le is not None:
            print(f"  '{col}': {len(le.classes_)} classes")
        else:
            print(f"  '{col}': regression (raw values saved)")

print(f"\nAll files in: {args.embedding_dir}/")
if has_labels:
    print(f"\nNext step:")
    print(f"  python downstream_tasks/row_prediction/train_downstream.py --embedding_dir {args.embedding_dir}")
print("="*80)
