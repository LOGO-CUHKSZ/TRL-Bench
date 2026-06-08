"""
TabPFN Embedding Generation
Generates embeddings using TabPFN for both supervised and self-supervised learning

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
"""

import pandas as pd
import numpy as np
import pickle
import os
import argparse
from sklearn.preprocessing import LabelEncoder

print("="*80)
print("TabPFN Pipeline - Generating Embeddings")
print("="*80)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate embeddings using TabPFN')

# Data configuration
parser.add_argument('--csv_file', type=str, required=True,
                    help='Path to the CSV file to generate embeddings for')
parser.add_argument('--output_dir', type=str, default='embeddings/row_prediction/TabPFN',
                    help='Directory to save embeddings (default: embeddings/row_prediction/TabPFN)')

# Mode configuration
parser.add_argument('--label_column', type=str, default=None,
                    help='Name of the label column. Supervised mode: used as target. Self-supervised mode: saved separately but not used in training.')
parser.add_argument('--mode', type=str, default='auto', choices=['auto', 'supervised', 'self-supervised'],
                    help='Mode: auto (detect from label_column), supervised, or self-supervised (default: auto)')

# Processing configuration
parser.add_argument('--batch_size', type=int, default=256,
                    help='Batch size for embedding generation (default: 256)')

# TabPFN configuration
parser.add_argument('--n_estimators', type=int, default=8,
                    help='Number of TabPFN estimators (default: 8)')
parser.add_argument('--device', type=str, default='auto',
                    help='Device to use: auto, cuda, cpu (default: auto)')
parser.add_argument('--random_state', type=int, default=42,
                    help='Random seed for reproducibility (default: 42)')

args = parser.parse_args()

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

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
        # Use ALL columns as features and create dummy target
        print(f"   Self-supervised mode: using all {len(data.columns)} columns")

        X = data

        # Create dummy target (all same class) - model won't learn from this
        y = np.zeros(len(data), dtype=int)

        label_col = None
        label_encoder = LabelEncoder()
        label_encoder.fit(['dummy'])  # Just for consistency in metadata

        print(f"   Using dummy target for self-supervised learning (no label information)")

    else:
        # Case 3: Self-supervised mode WITH label column specified
        # Save label column but compute embeddings on rest WITHOUT using label info
        label_col = args.label_column

        # Verify label column exists
        if label_col not in data.columns:
            raise ValueError(
                f"Label column '{label_col}' not found in {args.csv_file}\n"
                f"Available columns: {list(data.columns)}"
            )

        print(f"   Self-supervised mode with label column: '{label_col}'")
        print(f"   Label will be saved separately but NOT used for embedding training")

        # Separate label column for saving, but don't use it for training
        X = data.drop(label_col, axis=1)
        y_raw = data[label_col]

        # Encode label for saving purposes
        label_encoder = LabelEncoder()
        y_for_saving = label_encoder.fit_transform(y_raw.astype(str))

        # Create dummy target for training (model won't use label info)
        y = np.zeros(len(data), dtype=int)

        print(f"   Label classes: {label_encoder.classes_.tolist()}")
        print(f"   Number of classes: {len(label_encoder.classes_)}")
        print(f"   Using dummy target for training (embeddings are label-agnostic)")

# Detect categorical and continuous columns
print(f"\n2. Detecting feature types...")
category_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
continuous_cols = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"   Categorical columns ({len(category_cols)}): {category_cols[:5]}{'...' if len(category_cols) > 5 else ''}")
print(f"   Continuous columns ({len(continuous_cols)}): {continuous_cols[:5]}{'...' if len(continuous_cols) > 5 else ''}")

# Get categorical feature indices
categorical_indices = [i for i, col in enumerate(X.columns) if col in category_cols]

# Initialize TabPFN model
print(f"\n3. Initializing TabPFN model...")
print(f"   Number of estimators: {args.n_estimators}")
print(f"   Device: {args.device}")
print(f"   Random state: {args.random_state}")

from tabpfn_extensions import TabPFNClassifier

model = TabPFNClassifier(
    n_estimators=args.n_estimators,
    categorical_features_indices=categorical_indices if categorical_indices else None,
    device=args.device,
    random_state=args.random_state,
    ignore_pretraining_limits=True,  # Allow larger datasets
    memory_saving_mode=True,  # Enable memory saving to prevent GPU issues
    fit_mode='low_memory'  # Use low memory mode for large datasets
)

print(f"   Model: TabPFNClassifier")
print(f"   Memory saving mode: Enabled")

# Fit the model
print(f"\n4. Fitting TabPFN model on data...")
model.fit(X, y)
print(f"   Model fitted on {len(X)} samples")

# Generate embeddings
print(f"\n5. Generating embeddings...")

def get_embeddings_from_model(X_data):
    """Generate embeddings for given data"""
    print(f"   Generating embeddings for {len(X_data)} samples...")

    # Use get_embeddings with data_source='test'
    embeddings = model.get_embeddings(X_data, data_source='test')

    # If n_estimators > 1, average over estimators
    if embeddings.ndim == 3:  # (n_estimators, n_samples, embedding_dim)
        embeddings = embeddings.mean(axis=0)  # Average over estimators

    print(f"   Embeddings shape: {embeddings.shape}")
    return embeddings

# Generate embeddings
embeddings = get_embeddings_from_model(X)

# Save embeddings
print(f"\n6. Saving embeddings...")
embedding_file = os.path.join(args.output_dir, "embeddings.npy")

np.save(embedding_file, embeddings)

print(f"   Saved:")
print(f"      {embedding_file}")

# Save labels
label_file = os.path.join(args.output_dir, "labels.npy")

# For self-supervised mode with label column, save the actual labels
if mode == 'self-supervised' and args.label_column is not None:
    np.save(label_file, y_for_saving)
else:
    # For supervised or self-supervised without label, save y
    np.save(label_file, y)

print(f"      {label_file}")

# Save metadata
embedding_metadata = {
    'model_name': 'TabPFN',
    'mode': mode,
    'embedding_dim': embeddings.shape[1],
    'num_samples': len(embeddings),
    'csv_file': args.csv_file,
    'category_cols': category_cols,
    'continuous_cols': continuous_cols,
    'n_estimators': args.n_estimators,
    'device': args.device,
    'random_state': args.random_state,
    'label_column': label_col,
    'num_classes': len(label_encoder.classes_),
    'label_classes': label_encoder.classes_.tolist()
}

if mode == 'self-supervised':
    if args.label_column is None:
        embedding_metadata['training_note'] = "Self-supervised: Used all columns as features with dummy target (no label info)"
        embedding_metadata['label_used_in_training'] = False
    else:
        embedding_metadata['training_note'] = f"Self-supervised: Label column '{label_col}' saved but NOT used in training (label-agnostic embeddings)"
        embedding_metadata['label_used_in_training'] = False
        embedding_metadata['label_saved'] = True
else:
    embedding_metadata['label_used_in_training'] = True

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
print(f"  Embedding dimension: {embeddings.shape[1]}")
print(f"  Embeddings shape: {embeddings.shape}")
print(f"  Label column: {label_col if label_col else 'None'}")

if mode == 'self-supervised':
    if args.label_column is None:
        print(f"  Note: All columns used as features (no label info in training)")
    else:
        print(f"  Note: Label '{label_col}' saved but NOT used in training")
        print(f"        Embeddings are label-agnostic (unsupervised)")
        print(f"  Number of classes: {embedding_metadata['num_classes']}")
else:
    print(f"  Label used in training: Yes")
    print(f"  Number of classes: {embedding_metadata['num_classes']}")

print(f"\nSaved files:")
print(f"  - embeddings.npy")
print(f"  - labels.npy")
print(f"  - embedding_metadata.pkl")
print(f"\nAll files in: {args.output_dir}/")
print("="*80)
