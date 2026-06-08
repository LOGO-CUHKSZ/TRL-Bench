"""
SCARF Embedding Generation
Loads RAW data, applies saved preprocessor, generates embeddings using trained SCARF

Supports multiple input modes:
- Pre-split: --data_dir with train.csv and test.csv (or canonical dataset.json)
- Single CSV: --input with a single file to be split

Supports both label modes:
- With label column: Generates embeddings for features only, saves labels separately
- Without label column: Generates embeddings for all columns (pure unsupervised)

Output: JSON metadata (metadata.json) with v2.0 split-aware format
"""

import pandas as pd
import numpy as np
import pickle
import os
import argparse
import logging
import torch
from torch.utils.data import DataLoader, SequentialSampler
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../' * 2))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trl_bench.utils.ts3l.pl_modules import SCARFLightning
from trl_bench.utils.ts3l.utils.scarf_utils import SCARFDataset, SCARFConfig
from trl_bench.utils.unified_embedding_format import RowEmbeddingMetadataV2, save_split_embeddings, encode_label_column
from trl_bench.utils.table_dataset import load_table_dataset, SSLPreprocessor

logger = logging.getLogger(__name__)


def _load_lightning_checkpoint_compat(lightning_cls, checkpoint_path):
    """Load Lightning checkpoint with PyTorch 2.6+ weights_only compatibility."""
    try:
        return lightning_cls.load_from_checkpoint(checkpoint_path)
    except Exception as exc:
        if "Weights only load failed" not in str(exc):
            raise
        logger.warning(
            "Checkpoint load hit PyTorch weights_only guard; retrying with "
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1"
        )
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        return lightning_cls.load_from_checkpoint(checkpoint_path)


class _LegacyPreprocessorAdapter:
    """Adapter that wraps old-style per-column LabelEncoders + MinMaxScaler.

    Provides the same ``transform(X_df) -> pd.DataFrame`` interface as
    SSLPreprocessor without trying to reconstruct sklearn internals.
    """

    def __init__(self, categorical_encoders, scaler, category_cols, continuous_cols):
        self._categorical_encoders = categorical_encoders  # dict[col_name -> LabelEncoder]
        self._scaler = scaler                              # MinMaxScaler (already fitted)
        self.category_cols = list(category_cols)
        self.continuous_cols = list(continuous_cols)

    def transform(self, X_df):
        result = X_df.copy()
        for col in self.category_cols:
            le = self._categorical_encoders[col]
            result[col] = le.transform(result[col].astype(str))
        if self.continuous_cols and self._scaler is not None:
            result[self.continuous_cols] = self._scaler.transform(result[self.continuous_cols])
        return result


print("="*80)
print("SCARF Pipeline - Generating Embeddings")
print("="*80)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate embeddings using trained SCARF model')

# Data input (mutually exclusive)
input_group = parser.add_mutually_exclusive_group(required=True)
input_group.add_argument('--data_dir', type=str, default=None,
                    help='Directory containing data (canonical, legacy, or single CSV)')
input_group.add_argument('--input', type=str, default=None,
                    help='Single CSV file (will be split internally)')

# Model configuration
parser.add_argument('--checkpoint_dir', type=str, default='models/scarf/checkpoints',
                    help='Directory containing model checkpoints (default: models/scarf/checkpoints)')
parser.add_argument('--embedding_dir', type=str, default='models/scarf/embeddings',
                    help='Directory to save embeddings (default: models/scarf/embeddings)')

# Processing configuration
parser.add_argument('--batch_size', type=int, default=256,
                    help='Batch size for embedding generation (default: 256)')
parser.add_argument('--num_workers', type=int, default=4,
                    help='Number of data loading workers (default: 4)')

# Checkpoint selection
parser.add_argument('--checkpoint', type=str, default=None,
                    help='Path to specific checkpoint file (default: use scarf_self_supervised.ckpt from training_config.pkl)')

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
print(f"\n1. Loading training configuration and encoders...")
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

category_cols = train_config['category_cols']
continuous_cols = train_config['continuous_cols']

# Use specified checkpoint or fall back to the one from training config
if args.checkpoint is not None:
    final_ckpt = args.checkpoint
else:
    final_ckpt = train_config['final_checkpoint']

# Load preprocessor: new-style (SSLPreprocessor) or reconstruct from old-style
preprocessor = train_config.get('preprocessor')
if preprocessor is not None:
    print(f"   Using saved SSLPreprocessor")
else:
    # Backward compat: wrap old per-column LabelEncoders + MinMaxScaler directly
    print(f"   Reconstructing preprocessor from legacy encoders/scaler")
    categorical_encoders = train_config['categorical_encoders']
    scaler = train_config['scaler']
    preprocessor = _LegacyPreprocessorAdapter(
        categorical_encoders, scaler, category_cols, continuous_cols,
    )

print(f"   Checkpoint: {final_ckpt}")
print(f"   Categorical columns: {len(category_cols)}")
print(f"   Continuous columns: {len(continuous_cols)}")
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
label_columns_cli = label_cols  # ALL label columns so features exclude all labels

dataset = load_table_dataset(
    data_path,
    label_columns_cli=label_columns_cli,
    ignore_fingerprint=args.ignore_fingerprint,
)

# Recompute label info from what TableDataset actually resolved
label_cols = list(dataset.label_columns)
has_labels = bool(label_cols)

# --input single-CSV mode: split into train/test (matches old behavior)
if args.input is not None and dataset.split_names == ["all"]:
    dataset.apply_train_test_split(
        train_ratio=args.split_ratio,
        random_seed=args.random_seed,
        stratify_on_label=has_labels,
    )
    print(f"   Split single CSV into train/test (ratio={args.split_ratio}, seed={args.random_seed})")

# Verify fingerprint if we have one from training
if dataset_sha256 and dataset.fingerprint and dataset_sha256 != dataset.fingerprint:
    logger.warning(
        "Dataset fingerprint mismatch: training used %s, current dataset is %s",
        dataset_sha256[:16] + "...",
        dataset.fingerprint[:16] + "...",
    )

print(f"   Dataset: {dataset}")
data_source = data_path

# ============================================================================
# 3. Load SCARF model
# ============================================================================
print(f"\n3. Loading trained SCARF model...")
pl_scarf = _load_lightning_checkpoint_compat(SCARFLightning, final_ckpt)
pl_scarf.eval()  # Set to evaluation mode
pl_scarf.set_second_phase()  # Ensure we're in second phase for embedding extraction

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pl_scarf = pl_scarf.to(device)
print(f"   Model loaded on: {device}")
print(f"   Encoder output dimension: {pl_scarf.model.encoder.output_dim}")


def generate_embeddings_batch(X_encoded, model, device, batch_size, num_workers=4):
    """Generate embeddings for pre-encoded data using SCARF model."""
    # Create dummy config for dataset (corruption_rate=0 for inference)
    dummy_config = SCARFConfig(
        task="classification",
        loss_fn="CrossEntropyLoss",
        metric="accuracy_score",
        metric_hparams={},
        embedding_config=model.model.embedding_module.config,
        backbone_config=model.model.encoder.config,
        output_dim=train_config['output_dim'],
        corruption_rate=0.0  # No corruption for embedding generation
    )

    ds = SCARFDataset(
        X=X_encoded,
        Y=None,
        config=dummy_config,
        continuous_cols=continuous_cols,
        category_cols=category_cols,
        is_second_phase=True
    )

    dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=SequentialSampler(ds),
        num_workers=num_workers
    )

    embeddings = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            x = batch[0].to(device)
            x_emb = model.model.embedding_module(x)
            x_enc = model.model.encoder(x_emb)
            embeddings.append(x_enc.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"      Processed {(batch_idx + 1) * batch_size} samples...", end='\r')

    return np.vstack(embeddings)


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

    # Preprocess features
    X_enc = preprocessor.transform(view.X)
    emb = generate_embeddings_batch(X_enc, pl_scarf, device, args.batch_size, args.num_workers)
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

# Get feature columns and embedding dim
first_split = next(iter(emb_dict.values()))
feature_columns = list(dataset.feature_columns)

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
    model_name='SCARF',
    embedding_dim=first_split.shape[1],
    label_columns=label_cols,
    label_task_types=train_config.get('label_task_types', {}),
    feature_columns=feature_columns,
    generation_config=generation_config,
    dataset=dataset_info,
    checkpoint_path=final_ckpt,
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
print("="*80)
