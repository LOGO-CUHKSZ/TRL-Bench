"""
Task-agnostic classifier/regressor training and evaluation.

This script takes pre-extracted embeddings and trains a model for any task
defined by a labels file. Supports both classification and regression tasks.

Usage:
    # Classification task (table-level embeddings)
    python run_task.py \
        --embeddings datalake_embeddings.pkl \
        --labels spider_join/labels.json \
        --task_name spider_join \
        --task_type classification \
        --num_labels 2 \
        --output_dir results/spider_join

    # Regression task with COLUMN-LEVEL embeddings (for containment/join tasks)
    # This uses the specific column embeddings specified in labels
    python run_task.py \
        --embeddings datalake_embeddings.pkl \
        --labels wiki_containment/labels.json \
        --task_name wiki_containment \
        --task_type regression \
        --embedding_type column \
        --num_labels 1 \
        --output_dir results/wiki_containment

    # Compare different embedding sources on same task
    python run_task.py --embeddings embeddings_pretrained.pkl --labels task.json
    python run_task.py --embeddings embeddings_raw_bert.pkl --labels task.json

Embedding Types:
    - cls: Use the CLS token embedding (table-level representation)
    - table: Use mean-pooled table embedding (table-level representation)
    - column_mean: Use mean of all column embeddings (table-level representation)
    - column: Use specific column embeddings from join_col_table1/join_col_table2 in labels
              (column-level representation - recommended for join/containment tasks)
"""

import os
import sys
import pickle
import json
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, r2_score, mean_squared_error
from torchmetrics.classification import Accuracy
import random
from pathlib import Path

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)


def combine_embeddings(emb1, emb2, method='concat'):
    """Combine two embeddings."""
    emb1 = np.array(emb1)
    emb2 = np.array(emb2)

    if method == 'concat':
        return np.concatenate([emb1, emb2])
    elif method == 'add':
        return emb1 + emb2
    elif method == 'multiply':
        return emb1 * emb2
    elif method == 'diff':
        return np.abs(emb1 - emb2)
    else:
        raise ValueError(f"Unknown combination method: {method}")


def prepare_task_data(table_embeddings, labels, embedding_type='cls', combination_method='concat'):
    """
    Prepare training data for a task from individual table embeddings and labels.

    Args:
        table_embeddings: List of dicts with table embeddings
        labels: Labels dict with train/valid/test splits
        embedding_type: Which embedding to use:
            - 'cls': CLS token embedding (table-level)
            - 'table': Mean-pooled table embedding (table-level)
            - 'column_mean': Mean of all column embeddings (table-level)
            - 'column': Specific column embeddings from labels (column-level)
        combination_method: How to combine pairs ('concat', 'add', 'multiply', 'diff')

    Returns:
        Dict with train/valid/test datasets ready for training
    """
    # For column-level embeddings, we need to store the full column_embedding dict
    use_column_embeddings = (embedding_type == 'column')

    # Create lookup: table filename -> embeddings
    table_to_emb = {}
    for item in table_embeddings:
        table_name = item['table']
        table_key = table_name.split('/')[-1]

        # Handle both 'column_embeddings' (unified) and 'column_embedding' (legacy)
        col_emb_data = item.get('column_embeddings') or item.get('column_embedding', {})

        if use_column_embeddings:
            # Store the entire column_embedding dict for column-level lookups
            table_to_emb[table_key] = {
                'column_embedding': col_emb_data,  # Keep internal key name for compatibility
                'cls_embedding': item['cls_embedding']  # Fallback
            }
        elif embedding_type == 'cls':
            table_to_emb[table_key] = item['cls_embedding']
        elif embedding_type == 'table':
            table_to_emb[table_key] = item['table_embedding']
        elif embedding_type == 'column_mean':
            col_embs = list(col_emb_data.values())
            table_to_emb[table_key] = np.mean(col_embs, axis=0).tolist()
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")

    print(f"\n📊 Loaded embeddings for {len(table_to_emb)} unique tables")
    if use_column_embeddings:
        print(f"   Using COLUMN-LEVEL embeddings (from join_col_table1/join_col_table2 in labels)")

    # Create paired datasets
    task_data = {}
    stats = {'train': {'total': 0, 'skipped': 0, 'missing_col': 0},
             'valid': {'total': 0, 'skipped': 0, 'missing_col': 0},
             'test': {'total': 0, 'skipped': 0, 'missing_col': 0}}

    for split_name in ['train', 'valid', 'test']:
        if split_name not in labels:
            continue

        task_data[split_name] = []

        for item in labels[split_name]:
            stats[split_name]['total'] += 1

            table1_name = item['table1']['filename'].split('/')[-1]
            table2_name = item['table2']['filename'].split('/')[-1]
            label = item['label']

            if table1_name not in table_to_emb or table2_name not in table_to_emb:
                stats[split_name]['skipped'] += 1
                continue

            if use_column_embeddings:
                # Extract specific column embeddings based on labels
                col1_id = item.get('join_col_table1', item.get('col1', '0'))
                col2_id = item.get('join_col_table2', item.get('col2', '0'))

                # Column IDs in embeddings are integers, labels may have strings
                # Try both string and int versions
                col1_key = _find_column_key(table_to_emb[table1_name]['column_embedding'], col1_id)
                col2_key = _find_column_key(table_to_emb[table2_name]['column_embedding'], col2_id)

                if col1_key is None or col2_key is None:
                    stats[split_name]['missing_col'] += 1
                    # Fallback to CLS embedding if column not found
                    emb1 = table_to_emb[table1_name]['cls_embedding']
                    emb2 = table_to_emb[table2_name]['cls_embedding']
                else:
                    emb1 = table_to_emb[table1_name]['column_embedding'][col1_key]
                    emb2 = table_to_emb[table2_name]['column_embedding'][col2_key]
            else:
                # Table-level embeddings
                emb1 = table_to_emb[table1_name]
                emb2 = table_to_emb[table2_name]

            combined = combine_embeddings(emb1, emb2, combination_method)

            task_data[split_name].append({
                'embedding': combined.tolist() if isinstance(combined, np.ndarray) else combined,
                'label': label,
                'split': split_name
            })

    # Print statistics
    print("\n" + "="*60)
    print("TASK DATA PREPARATION")
    print("="*60)
    print(f"Embedding type: {embedding_type}")
    for split in ['train', 'valid', 'test']:
        if split in stats:
            total = stats[split]['total']
            skipped = stats[split]['skipped']
            missing_col = stats[split].get('missing_col', 0)
            kept = total - skipped
            msg = f"{split.upper()}: {kept}/{total} samples (skipped {skipped} due to missing tables)"
            if use_column_embeddings and missing_col > 0:
                msg += f", {missing_col} used CLS fallback (missing column)"
            print(msg)
    print("="*60)

    return task_data


def _find_column_key(column_embedding_dict, col_id):
    """
    Find the column key in the embedding dict, handling int/string mismatches.

    Args:
        column_embedding_dict: Dict mapping column IDs to embeddings
        col_id: Column ID from labels (could be int or string)

    Returns:
        The matching key in the dict, or None if not found
    """
    # Try direct lookup
    if col_id in column_embedding_dict:
        return col_id

    # Try as integer
    try:
        int_key = int(col_id)
        if int_key in column_embedding_dict:
            return int_key
    except (ValueError, TypeError):
        pass

    # Try as string
    str_key = str(col_id)
    if str_key in column_embedding_dict:
        return str_key

    return None


class EmbeddingDataset(Dataset):
    """Dataset for pre-extracted embeddings."""
    def __init__(self, embeddings_list, task_type='classification', num_labels=2):
        self.embeddings = [item['embedding'] for item in embeddings_list]
        self.labels = [item['label'] for item in embeddings_list]
        self.task_type = task_type
        self.num_labels = num_labels

        # Auto-detect multi-label classification
        self.is_multi_label = False
        if task_type == 'classification':
            # Check if any label is a list with multiple elements
            for label in self.labels:
                if isinstance(label, list) and len(label) > 1:
                    self.is_multi_label = True
                    break

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        embedding = torch.tensor(self.embeddings[idx], dtype=torch.float32)

        if self.task_type == 'regression':
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
        elif self.is_multi_label:
            # Convert list of label indices to multi-hot encoding
            label_indices = self.labels[idx] if isinstance(self.labels[idx], list) else [self.labels[idx]]
            multi_hot = torch.zeros(self.num_labels, dtype=torch.float32)
            for label_idx in label_indices:
                if 0 <= label_idx < self.num_labels:
                    multi_hot[label_idx] = 1.0
            label = multi_hot
        else:
            # Single-label classification
            label_val = self.labels[idx] if not isinstance(self.labels[idx], list) else self.labels[idx][0]
            label = torch.tensor(label_val, dtype=torch.long)

        return embedding, label


class SimpleClassifier(pl.LightningModule):
    """2-layer MLP for classification or regression."""
    def __init__(self, input_dim=768, hidden_dim=256, num_labels=2, learning_rate=2e-5,
                 adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8, dropout_prob=0.1,
                 task_type='classification', is_multi_label=False):
        super().__init__()
        self.save_hyperparameters()

        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        self.task_type = task_type
        self.is_multi_label = is_multi_label

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, num_labels)
        )

        self.val_step_outputs = []
        self.val_step_targets = []
        self.test_step_outputs = []
        self.test_step_targets = []

    def forward(self, embeddings):
        return self.classifier(embeddings)

    def training_step(self, batch, batch_idx):
        embeddings, labels = batch
        logits = self(embeddings)

        if self.task_type == 'regression':
            loss = nn.MSELoss()(logits.squeeze(), labels)
        elif self.is_multi_label:
            loss = nn.BCEWithLogitsLoss()(logits, labels)
        else:
            loss = nn.CrossEntropyLoss()(logits, labels)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        embeddings, labels = batch
        logits = self(embeddings)

        if self.task_type == 'regression':
            loss = nn.MSELoss()(logits.squeeze(), labels)
        elif self.is_multi_label:
            loss = nn.BCEWithLogitsLoss()(logits, labels)
        else:
            loss = nn.CrossEntropyLoss()(logits, labels)

        self.log('valid_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_step_outputs.extend(logits.cpu().numpy())
        self.val_step_targets.extend(labels.cpu().numpy())
        return loss

    def on_validation_epoch_end(self):
        if len(self.val_step_outputs) == 0:
            return

        outputs = np.array(self.val_step_outputs)
        targets = np.array(self.val_step_targets)

        if self.task_type == 'regression':
            # Regression metrics
            predictions = outputs.squeeze()
            mse = mean_squared_error(targets, predictions)
            r2 = r2_score(targets, predictions)

            self.log('val_mse', mse, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('val_r2', r2, on_epoch=True, prog_bar=True, sync_dist=True)
        elif self.is_multi_label:
            # Multi-label classification metrics
            logits = torch.tensor(outputs)
            predictions = (torch.sigmoid(logits) > 0.5).float().numpy()

            # Subset accuracy (exact match)
            subset_acc = np.mean(np.all(predictions == targets, axis=1))

            # Hamming score (per-label accuracy)
            hamming_acc = np.mean(predictions == targets)

            # F1 score (micro and macro)
            f1_micro = f1_score(targets, predictions, average='micro', zero_division=0)
            f1_macro = f1_score(targets, predictions, average='macro', zero_division=0)

            self.log('total_val_accuracy', subset_acc, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('hamming_accuracy', hamming_acc, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('f1', f1_micro, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('f1_macro', f1_macro, on_epoch=True, prog_bar=True, sync_dist=True)
        else:
            # Single-label classification metrics
            logits = torch.tensor(outputs)
            targets_tensor = torch.tensor(targets)
            predicted = logits.argmax(dim=1)

            # Use appropriate task type based on number of labels
            if self.num_labels == 2:
                acc = Accuracy(task='binary').to(self.device)(predicted, targets_tensor)
            else:
                acc = Accuracy(task='multiclass', num_classes=self.num_labels).to(self.device)(predicted, targets_tensor)
            f1 = f1_score(targets, predicted.cpu().numpy(), average='weighted', zero_division=1)

            self.log('total_val_accuracy', acc, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('f1', f1, on_epoch=True, prog_bar=True, sync_dist=True)

        self.val_step_outputs.clear()
        self.val_step_targets.clear()

    def test_step(self, batch, batch_idx):
        embeddings, labels = batch
        logits = self(embeddings)

        if self.task_type == 'regression':
            loss = nn.MSELoss()(logits.squeeze(), labels)
        elif self.is_multi_label:
            loss = nn.BCEWithLogitsLoss()(logits, labels)
        else:
            loss = nn.CrossEntropyLoss()(logits, labels)

        self.log('test_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.test_step_outputs.extend(logits.cpu().numpy())
        self.test_step_targets.extend(labels.cpu().numpy())
        return loss

    def on_test_epoch_end(self):
        if len(self.test_step_outputs) == 0:
            return

        outputs = np.array(self.test_step_outputs)
        targets = np.array(self.test_step_targets)

        if self.task_type == 'regression':
            # Regression metrics
            predictions = outputs.squeeze()
            mse = mean_squared_error(targets, predictions)
            r2 = r2_score(targets, predictions)

            self.log('test_mse', mse, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('test_r2', r2, on_epoch=True, prog_bar=True, sync_dist=True)
        elif self.is_multi_label:
            # Multi-label classification metrics
            logits = torch.tensor(outputs)
            predictions = (torch.sigmoid(logits) > 0.5).float().numpy()

            # Subset accuracy (exact match)
            subset_acc = np.mean(np.all(predictions == targets, axis=1))

            # Hamming score (per-label accuracy)
            hamming_acc = np.mean(predictions == targets)

            # F1 score (micro and macro)
            f1_micro = f1_score(targets, predictions, average='micro', zero_division=0)
            f1_macro = f1_score(targets, predictions, average='macro', zero_division=0)

            self.log('total_test_accuracy', subset_acc, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('hamming_accuracy', hamming_acc, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('f1', f1_micro, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('f1_macro', f1_macro, on_epoch=True, prog_bar=True, sync_dist=True)
        else:
            # Single-label classification metrics
            logits = torch.tensor(outputs)
            targets_tensor = torch.tensor(targets)
            predicted = logits.argmax(dim=1)

            # Use appropriate task type based on number of labels
            if self.num_labels == 2:
                acc = Accuracy(task='binary').to(self.device)(predicted, targets_tensor)
            else:
                acc = Accuracy(task='multiclass', num_classes=self.num_labels).to(self.device)(predicted, targets_tensor)
            f1 = f1_score(targets, predicted.cpu().numpy(), average='weighted', zero_division=1)

            self.log('total_test_accuracy', acc, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('f1', f1, on_epoch=True, prog_bar=True, sync_dist=True)

        self.test_step_outputs.clear()
        self.test_step_targets.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=(self.adam_beta1, self.adam_beta2),
            eps=self.adam_epsilon
        )


def main():
    parser = ArgumentParser(description="Train classifier on task-specific paired embeddings")
    parser.add_argument('--embeddings', type=str, required=True,
                        help='Pickle file with individual table embeddings')
    parser.add_argument('--labels', type=str, required=True,
                        help='Labels JSON file defining the task')
    parser.add_argument('--task_name', type=str, required=True,
                        help='Name of the task (for logging)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')

    # Embedding combination
    parser.add_argument('--embedding_type', type=str, default='cls',
                        choices=['cls', 'table', 'column_mean', 'column'],
                        help='Which embedding type to use: cls/table/column_mean (table-level), '
                             'column (uses specific columns from join_col_table1/join_col_table2 in labels)')
    parser.add_argument('--combination_method', type=str, default='concat',
                        choices=['concat', 'add', 'multiply', 'diff'],
                        help='How to combine table pair embeddings')

    # Task configuration
    parser.add_argument('--task_type', type=str, default='classification',
                        choices=['classification', 'regression'],
                        help='Task type: classification or regression')

    # Classifier architecture
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension for 2-layer MLP')
    parser.add_argument('--num_labels', type=int, default=2,
                        help='Number of output labels (2 for binary classification, 1 for regression)')

    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--dropout_prob', type=float, default=0.1)
    parser.add_argument('--random_seed', type=int, default=0)

    # Hardware
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)

    args = parser.parse_args()

    # Set seeds
    print(f"\n🎲 Setting random seed: {args.random_seed}")
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    pl.seed_everything(args.random_seed, workers=True)

    # Load embeddings
    print(f"\n📂 Loading table embeddings from: {args.embeddings}")
    with open(args.embeddings, 'rb') as f:
        table_embeddings = pickle.load(f)
    print(f"   Loaded {len(table_embeddings)} table embeddings")

    # Load labels
    print(f"\n📂 Loading task labels from: {args.labels}")
    with open(args.labels, 'r') as f:
        labels = json.load(f)

    # Prepare task data (pair embeddings according to labels)
    print(f"\n🔄 Preparing task data...")
    print(f"   Task: {args.task_name}")
    print(f"   Embedding type: {args.embedding_type}")
    print(f"   Combination method: {args.combination_method}")

    task_data = prepare_task_data(
        table_embeddings,
        labels,
        embedding_type=args.embedding_type,
        combination_method=args.combination_method
    )

    # Auto-detect input dimension
    if len(task_data['train']) > 0:
        input_dim = len(task_data['train'][0]['embedding'])
        print(f"\n✅ Detected input dimension: {input_dim}")
    else:
        raise ValueError("No training data available!")

    # Create datasets
    train_dataset = EmbeddingDataset(task_data.get('train', []), task_type=args.task_type, num_labels=args.num_labels)
    valid_dataset = EmbeddingDataset(task_data.get('valid', []), task_type=args.task_type, num_labels=args.num_labels)
    test_dataset = EmbeddingDataset(task_data.get('test', []), task_type=args.task_type, num_labels=args.num_labels)

    # Check if multi-label classification is detected
    is_multi_label = train_dataset.is_multi_label
    if is_multi_label:
        print(f"\n⚠️  Detected multi-label classification (samples have variable-length labels)")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Initialize model
    print(f"\n🔧 Initializing model...")
    print(f"   Task type: {args.task_type}")
    if is_multi_label:
        print(f"   Multi-label: Yes (BCEWithLogitsLoss)")
    print(f"   Architecture: {input_dim} → {args.hidden_dim} → {args.num_labels}")
    model = SimpleClassifier(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_labels=args.num_labels,
        learning_rate=args.learning_rate,
        dropout_prob=args.dropout_prob,
        task_type=args.task_type,
        is_multi_label=is_multi_label
    )
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup callbacks
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(args.output_dir) / 'checkpoints',
        filename='best',
        monitor='valid_loss',
        mode='min',
        save_top_k=1,
        verbose=True
    )

    early_stop_callback = EarlyStopping(
        monitor='valid_loss',
        patience=5,
        mode='min',
        verbose=True
    )

    # Train
    print(f"\n🚀 Training on task: {args.task_name}")
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        default_root_dir=args.output_dir,
        callbacks=[checkpoint_callback, early_stop_callback],
        enable_progress_bar=True,
        log_every_n_steps=10
    )

    trainer.fit(model, train_loader, valid_loader)

    # Test
    print(f"\n🧪 Evaluating on test set...")
    test_results = trainer.test(model, test_loader, ckpt_path='best')

    # Save results summary
    results_file = Path(args.output_dir) / 'results.json'
    results = {
        'task_name': args.task_name,
        'task_type': args.task_type,
        'embedding_type': args.embedding_type,
        'combination_method': args.combination_method,
        'input_dim': input_dim,
        'hidden_dim': args.hidden_dim,
        'num_labels': args.num_labels,
        'test_results': test_results[0] if test_results else {},
        'data_stats': {
            'train': len(train_dataset),
            'valid': len(valid_dataset),
            'test': len(test_dataset)
        }
    }

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Task complete!")
    print(f"   Results saved to: {args.output_dir}")
    print(f"   Summary: {results_file}")


if __name__ == '__main__':
    main()
