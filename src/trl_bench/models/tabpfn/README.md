# TabPFN Embedding Generator

This script generates embeddings using TabPFN (Tabular Prior-Data Fitted Network) for both supervised and self-supervised learning scenarios.

## Features

- **Supervised Mode**: Specify a label column to generate embeddings with supervised learning
- **Self-Supervised Mode**: Automatically uses a column as pseudo-target for self-supervised embedding generation
- **GPU Support**: Optimized for GPU usage with memory-saving mode enabled
- **Compatible Output**: Generates the same output format as DAE embeddings

## Installation

Make sure `tabpfn-extensions` is installed in your conda environment:

```bash
conda activate trl_tasks
```

## Usage

### Supervised Mode

Generate embeddings with a specific label column:

```bash
python models/TabPFN/generate_embeddings.py \
    --data_dir datasets/adult \
    --label_column income \
    --embedding_dir embeddings/row_prediction/TabPFN \
    --n_estimators 4
```

### Self-Supervised Mode

Generate embeddings without specifying a label column (uses a categorical column as pseudo-target):

```bash
python models/TabPFN/generate_embeddings.py \
    --data_dir datasets/adult \
    --embedding_dir embeddings/row_prediction/TabPFN_unsupervised \
    --n_estimators 4
```

## Arguments

- `--data_dir`: Directory containing `train.csv` and `test.csv` (default: `datasets/adult`)
- `--embedding_dir`: Output directory for embeddings (default: `embeddings/row_prediction/TabPFN`)
- `--label_column`: Name of the label column for supervised mode (if None, uses self-supervised mode)
- `--mode`: Mode selection: `auto`, `supervised`, or `self-supervised` (default: `auto`)
- `--n_estimators`: Number of TabPFN estimators (default: 8, recommended: 4 for faster processing)
- `--device`: Device to use: `auto`, `cuda`, or `cpu` (default: `auto`)
- `--random_state`: Random seed for reproducibility (default: 42)

## Output Files

The script generates the following files:

1. `train_embeddings.npy` - Training set embeddings (numpy array)
2. `test_embeddings.npy` - Test set embeddings (numpy array)
3. `train_labels.npy` - Training labels (numpy array)
4. `test_labels.npy` - Test labels (numpy array)
5. `embedding_metadata.pkl` - Metadata including:
   - Model name
   - Mode (supervised/self-supervised)
   - Embedding dimension
   - Number of samples
   - Feature column names
   - Label information

## How It Works

### Supervised Mode

- Uses the specified label column as the target
- Generates embeddings based on the supervised task
- Output embeddings capture discriminative features for the task

### Self-Supervised Mode

- Automatically selects a categorical column as pseudo-target (prefers categorical to avoid unseen values)
- Creates a pretext task similar to autoencoders
- Generates embeddings that capture general data structure
- Note: The pseudo-target labels are saved but are mainly for internal use

## Technical Details

- **Memory Management**: Uses `memory_saving_mode=True` and `fit_mode='low_memory'` to handle large datasets on GPU
- **Feature Handling**: Automatically detects categorical and continuous features
- **Ensemble Averaging**: When `n_estimators > 1`, embeddings from multiple estimators are averaged
- **GPU Optimization**: Configured to work efficiently on NVIDIA GPUs with CUDA support

## Example Output

```
Summary:
  Mode: supervised
  Embedding dimension: 192
  Train embeddings: (39073, 192)
  Test embeddings: (9769, 192)
  Label column: income
  Number of classes: 2
```

## Notes

- For self-supervised mode, the script automatically handles unseen categorical values by mapping them to the first class
- The embedding dimension is 192 (48 features × 4 estimators, averaged)
- GPU memory saving is automatically enabled to prevent CUDA errors
