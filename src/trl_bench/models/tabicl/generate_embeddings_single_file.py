"""
TabICL Embedding Generation (Single File)
Generates embeddings using TabICL for a single CSV file.

Uses TabICL's internal Stage 2 representations (col_embedder → row_interactor)
to produce 512-dim row embeddings. Labels are NOT used in embedding extraction
(they enter at Stage 3 only).

Three Modes of Operation:
1. Supervised mode (--mode supervised --label_column <name>):
   - Uses specified label column as target
   - Note: embeddings are identical to self-supervised

2. Self-supervised mode without label (--mode self-supervised):
   - Uses ALL columns as features
   - Generates purely unsupervised embeddings

3. Self-supervised mode with label (--mode self-supervised --label_column <name>):
   - Label column is saved separately
   - Embeddings computed on remaining columns
   - Useful for evaluation while maintaining unsupervised training
"""

import pandas as pd
import numpy as np
import pickle
import os
import argparse
import torch
from sklearn.preprocessing import LabelEncoder

print("="*80)
print("TabICL Pipeline - Generating Embeddings (Single File)")
print("="*80)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate embeddings using TabICL (single file)')

# Data configuration
parser.add_argument('--csv_file', type=str, required=True,
                    help='Path to the CSV file to generate embeddings for')
parser.add_argument('--output_dir', type=str, default='embeddings/row_prediction/TabICL',
                    help='Directory to save embeddings (default: embeddings/row_prediction/TabICL)')

# Mode configuration
parser.add_argument('--label_column', type=str, default=None,
                    help='Name of the label column. Excluded from features; saved with embeddings.')
parser.add_argument('--mode', type=str, default='auto', choices=['auto', 'supervised', 'self-supervised'],
                    help='Mode: auto (detect from label_column), supervised, or self-supervised (default: auto)')

# TabICL configuration
parser.add_argument('--n_estimators', type=int, default=1,
                    help='Number of TabICL estimators (default: 1)')
parser.add_argument('--checkpoint_version', type=str, default='tabicl-classifier-v1.1-0506.ckpt',
                    help='TabICL checkpoint version (default: tabicl-classifier-v1.1-0506.ckpt)')
parser.add_argument('--device', type=str, default='auto',
                    help='Device to use: auto, cuda, cpu (default: auto)')

args = parser.parse_args()

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Resolve device
if args.device == 'auto':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
    device = args.device

# Determine mode
if args.mode == 'auto':
    mode = 'supervised' if args.label_column is not None else 'self-supervised'
else:
    mode = args.mode

print(f"\nMode: {mode.upper()}")
if mode == 'supervised' and args.label_column is None:
    raise ValueError("Supervised mode requires --label_column to be specified")

print(f"\n1. Loading data from {args.csv_file}...")

# Load the CSV file
if not os.path.exists(args.csv_file):
    raise FileNotFoundError(f"CSV file not found: {args.csv_file}")

data = pd.read_csv(args.csv_file)

print(f"   Loaded {len(data)} samples from {os.path.basename(args.csv_file)}")

# Prepare data based on mode
if mode == 'supervised':
    label_col = args.label_column

    # Verify label column exists
    if label_col not in data.columns:
        raise ValueError(
            f"Label column '{label_col}' not found in {args.csv_file}\n"
            f"Available columns: {list(data.columns)}"
        )

    print(f"   Label column: '{label_col}'")

    # Separate features and labels
    X = data.drop(label_col, axis=1)
    y_raw = data[label_col]

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    print(f"   Label classes: {label_encoder.classes_.tolist()}")
    print(f"   Number of classes: {len(label_encoder.classes_)}")

else:  # self-supervised
    if args.label_column is None:
        # Case 2: Self-supervised mode WITHOUT label column
        print(f"   Self-supervised mode: using all {len(data.columns)} columns")

        X = data

        label_col = None
        label_encoder = LabelEncoder()
        label_encoder.fit(['dummy'])

        print(f"   No label column specified")

    else:
        # Case 3: Self-supervised mode WITH label column specified
        label_col = args.label_column

        # Verify label column exists
        if label_col not in data.columns:
            raise ValueError(
                f"Label column '{label_col}' not found in {args.csv_file}\n"
                f"Available columns: {list(data.columns)}"
            )

        print(f"   Self-supervised mode with label column: '{label_col}'")
        print(f"   Label will be saved separately but NOT used for embedding generation")

        # Separate label column for saving
        X = data.drop(label_col, axis=1)
        y_raw = data[label_col]

        # Encode label for saving purposes
        label_encoder = LabelEncoder()
        y_for_saving = label_encoder.fit_transform(y_raw.astype(str))

        print(f"   Label classes: {label_encoder.classes_.tolist()}")
        print(f"   Number of classes: {len(label_encoder.classes_)}")
        print(f"   Embeddings are label-agnostic (labels enter at Stage 3 only)")

# Initialize TabICL model
print(f"\n2. Initializing TabICL model...")
print(f"   Number of estimators: {args.n_estimators}")
print(f"   Checkpoint: {args.checkpoint_version}")
print(f"   Device: {device}")

from tabicl import TabICLClassifier

clf = TabICLClassifier(
    n_estimators=args.n_estimators,
    device=device,
)

print(f"   Model: TabICLClassifier")

# Create dummy labels for fit() - TabICLClassifier requires >=2 classes
print(f"\n3. Fitting TabICL model (initializes preprocessing + loads weights)...")
n_samples = len(X)
dummy_y = np.zeros(n_samples, dtype=int)
dummy_y[n_samples // 2:] = 1  # Balanced 2-class dummy labels

clf.fit(X, dummy_y)
print(f"   Model fitted on {n_samples} samples (dummy 2-class labels for initialization)")

# Extract Stage 2 embeddings using a forward hook on row_interactor.
# We use predict_proba() to ensure the full preprocessing pipeline runs:
#   X_encoder_ (categorical→numerical) → EnsembleGenerator (StandardScaler +
#   PowerTransform + OutlierRemover) → model forward.
print(f"\n4. Extracting Stage 2 embeddings via forward hook...")

model = clf.model_

captured_reps = []

def capture_row_reps(module, input, output):
    """Capture row_interactor output (Stage 2 embeddings)."""
    captured_reps.append(output.detach().cpu())

hook_handle = model.row_interactor.register_forward_hook(capture_row_reps)
try:
    # For single-file mode, all data was used for fit (as "train").
    # Pass X as "test" to predict_proba — the model sees the fitted train
    # data as context and X as the test data.
    print(f"   Running pipeline on {n_samples} samples...")
    with torch.no_grad():
        clf.predict_proba(X)

    # First hook call, first batch: (n_samples_context + n_samples, 512)
    # The context rows (from fit) are the same as X, so take all rows
    reps = captured_reps[0][0].numpy()
    # In single-file mode, context = fit data = X itself
    # Take the "test" portion (rows after train_size = n_samples)
    embeddings = reps[n_samples:].numpy() if reps.shape[0] > n_samples else reps
    # Since we fit on X and predict on X, the output has 2*n_samples rows:
    # first n_samples = context (train), second n_samples = predictions (test=X)
    embeddings = reps[n_samples:]
finally:
    hook_handle.remove()

print(f"   Embeddings shape: {embeddings.shape}")

# Save embeddings
print(f"\n5. Saving embeddings...")
embedding_file = os.path.join(args.output_dir, "embeddings.npy")

np.save(embedding_file, embeddings)

print(f"   Saved:")
print(f"      {embedding_file}")

# Save labels
label_file = os.path.join(args.output_dir, "labels.npy")

# For self-supervised mode with label column, save the actual labels
if mode == 'self-supervised' and args.label_column is not None:
    np.save(label_file, y_for_saving)
elif mode == 'supervised':
    np.save(label_file, y)
else:
    # Self-supervised without label - save dummy
    np.save(label_file, np.zeros(n_samples, dtype=int))

print(f"      {label_file}")

# Save metadata
embedding_metadata = {
    'model_name': 'TabICL',
    'mode': mode,
    'embedding_dim': embeddings.shape[1],
    'num_samples': len(embeddings),
    'csv_file': args.csv_file,
    'n_estimators': args.n_estimators,
    'checkpoint_version': args.checkpoint_version,
    'device': device,
    'extraction_point': 'Stage 2 (row_interactor)',
    'label_column': label_col,
    'num_classes': len(label_encoder.classes_),
    'label_classes': label_encoder.classes_.tolist()
}

if mode == 'self-supervised':
    if args.label_column is None:
        embedding_metadata['training_note'] = "Self-supervised: Used all columns as features (no label info)"
        embedding_metadata['label_used_in_training'] = False
    else:
        embedding_metadata['training_note'] = f"Self-supervised: Label column '{label_col}' saved but NOT used for embeddings"
        embedding_metadata['label_used_in_training'] = False
        embedding_metadata['label_saved'] = True
else:
    embedding_metadata['training_note'] = "Supervised: Labels saved but NOT used in embedding extraction (labels enter at Stage 3 only)"
    embedding_metadata['label_used_in_training'] = False

metadata_file = os.path.join(args.output_dir, "embedding_metadata.pkl")
with open(metadata_file, 'wb') as f:
    pickle.dump(embedding_metadata, f)

print(f"      {metadata_file}")

print("\n" + "="*80)
print("Embedding generation completed successfully!")
print("="*80)
print(f"\nSummary:")
print(f"  Mode: {mode}")
print(f"  Input file: {os.path.basename(args.csv_file)}")
print(f"  Embedding dimension: {embeddings.shape[1]} (4 CLS tokens x 128)")
print(f"  Embeddings shape: {embeddings.shape}")
print(f"  Label column: {label_col if label_col else 'None'}")

if mode == 'self-supervised':
    if args.label_column is None:
        print(f"  Note: All columns used as features (no label info)")
    else:
        print(f"  Note: Label '{label_col}' saved but NOT used for embeddings")
        print(f"        Embeddings are label-agnostic (labels enter at Stage 3 only)")
        print(f"  Number of classes: {embedding_metadata['num_classes']}")
else:
    print(f"  Note: Labels saved; embeddings are still label-agnostic")
    print(f"  Number of classes: {embedding_metadata['num_classes']}")

print(f"\nSaved files:")
print(f"  - embeddings.npy")
print(f"  - labels.npy")
print(f"  - embedding_metadata.pkl")
print(f"\nAll files in: {args.output_dir}/")
print("="*80)
