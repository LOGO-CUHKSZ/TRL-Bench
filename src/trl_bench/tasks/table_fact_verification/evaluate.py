#!/usr/bin/env python3
"""
Evaluate a trained TabFact classifier on test set.

Standard embedding format:
{
    'table_embeddings': {example_id: np.array(768,), ...},
    'labels': {example_id: int, ...},
    'statement_embeddings': {example_id: np.array(768,), ...}  # Optional
}

Usage:
    # Single-embedding mode (e.g., TAPAS joint or table-only)
    python evaluate.py \
        --model_checkpoint checkpoints/tabfact/tapas/best_model.pt \
        --test_embeddings embeddings/tabfact/tapas/test.pkl \
        --output_file results/evaluation/tabfact/tapas_results.json \
        --device cuda

    # Two-embedding mode
    python evaluate.py \
        --model_checkpoint checkpoints/tabfact/doduo_concat/best_model.pt \
        --test_embeddings embeddings/tabfact/doduo/test.pkl \
        --combine_method concat \
        --output_file results/evaluation/tabfact/doduo_concat_results.json \
        --device cuda
"""

import os
import sys
import pickle
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from downstream_tasks.table_fact_verification.train import (
    TabFactClassifier, LinearClassifier, load_embeddings, load_from_precomputed
)
from trl_bench.utils.downstream.heads import ProjectedInteractionHead


def evaluate_model(
    model_checkpoint: str,
    test_embeddings: str = None,
    config_file: str = None,
    device: str = 'cuda',
    output_file: str = None,
    combine_method: str = None,
    table_embeddings: str = None,
    statement_embeddings: str = None,
    labels_json: str = None,
    table_embedding_variant: str = 'column_mean',
    statement_only: bool = False,
):
    """
    Evaluate trained model on test set.

    Args:
        model_checkpoint: Path to model checkpoint
        test_embeddings: Path to test embeddings pickle (legacy monolithic format)
        config_file: Path to training config (to get model architecture)
        device: Device to use
        output_file: Path to save results
        combine_method: How to combine embeddings (overrides config if provided)
        table_embeddings: Path to column embeddings pickle (pre-computed mode)
        statement_embeddings: Path to statement embeddings pickle (pre-computed mode)
        labels_json: Path to TabFact JSONL (pre-computed mode)
        table_embedding_variant: Which table embedding variant to use
    """
    print("TabFact Evaluation")
    print("="*60)

    use_precomputed = table_embeddings is not None

    # Load config to determine model architecture and combine method
    model_type = 'mlp'
    hidden_dim = 256
    dropout = 0.1

    # Current convention: dropout_first=False (YAML config default).
    # Old checkpoints with dropout_first=True are safe — their config.json records the True value.
    dropout_first = False

    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        model_type = config.get('model_type', 'mlp')
        hidden_dim = config.get('hidden_dim', 256)
        dropout = config.get('dropout', 0.1)
        dropout_first = config.get('dropout_first', False)
        # Get combine_method from config if not overridden
        if combine_method is None:
            combine_method = config.get('combine_method')
        # Get table_embedding_variant from config if not overridden via CLI
        if use_precomputed and table_embedding_variant == 'column_mean':
            table_embedding_variant = config.get('table_embedding_variant', 'column_mean')
        # Recover statement_only from config when not passed via CLI
        if not statement_only:
            statement_only = config.get('statement_only', False)
        print(f"\nLoaded config: {model_type}, hidden_dim={hidden_dim}, combine_method={combine_method}")
    else:
        config = None

    # Guard: statement_only requires pre-computed mode
    if statement_only and not use_precomputed:
        raise ValueError(
            "--statement_only / config statement_only=true requires pre-computed mode "
            "(--table_embeddings + --statement_embeddings), not --test_embeddings"
        )

    # Load test embeddings
    print("Loading test embeddings...")
    if use_precomputed:
        combine = combine_method or 'concat'
        test_emb, test_labels, test_ids = load_from_precomputed(
            table_embeddings, statement_embeddings, labels_json,
            variant=table_embedding_variant, combine_method=combine,
            statement_only=statement_only,
        )
    else:
        test_emb, test_labels, test_ids = load_embeddings(test_embeddings, combine_method=combine_method)
    print(f"Test: {len(test_emb)} examples, embedding dim: {test_emb.shape[1]}")

    # Check label distribution
    entailed = sum(test_labels == 1)
    refuted = sum(test_labels == 0)
    print(f"\nLabel distribution:")
    print(f"  Entailed: {entailed} ({100*entailed/len(test_labels):.1f}%)")
    print(f"  Refuted: {refuted} ({100*refuted/len(test_labels):.1f}%)")

    input_dim = test_emb.shape[1]

    # Create model
    if model_type == 'interaction':
        # Reconstruct interaction head from persisted config
        table_dim = config.get('table_input_dim') if config else None
        stmt_dim = config.get('stmt_input_dim') if config else None
        if table_dim is None or stmt_dim is None:
            raise ValueError(
                "config.json missing table_input_dim/stmt_input_dim; "
                "cannot reconstruct interaction head"
            )
        model = ProjectedInteractionHead(
            table_input_dim=table_dim,
            stmt_input_dim=stmt_dim,
            projection_dim=config.get('projection_dim', 256),
            classifier_hidden_dim=config.get('classifier_hidden_dim', 256),
            num_classes=2,
            dropout=dropout,
            interaction_type=config.get('interaction_type', 'full'),
            normalize_projection=config.get('normalize_projection', True),
        )
    elif model_type == 'linear':
        model = LinearClassifier(input_dim=input_dim, dropout=dropout,
                                 dropout_first=dropout_first)
    else:
        model = TabFactClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            dropout_first=dropout_first,
        )

    # Load weights
    print(f"Loading model from {model_checkpoint}...")
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    model = model.to(device)
    model.eval()

    # Create data loader
    test_dataset = TensorDataset(
        torch.tensor(test_emb, dtype=torch.float32),
        torch.tensor(test_labels, dtype=torch.long)
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Evaluate
    print("\nEvaluating...")
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_emb, batch_labels in test_loader:
            batch_emb = batch_emb.to(device)

            outputs = model(batch_emb)
            probs = torch.softmax(outputs, dim=1)

            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_labels.numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')

    # AUROC (using shared implementation for consistent degenerate handling)
    from trl_bench.utils.downstream.metrics import auroc as compute_auroc
    auroc_value = compute_auroc(all_probs, all_labels)

    # Per-class metrics
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    precision_per_class = precision_score(all_labels, all_preds, average=None)
    recall_per_class = recall_score(all_labels, all_preds, average=None)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Print results
    print("\n" + "="*60)
    print("Results")
    print("="*60)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1 (macro): {f1_macro:.4f}")
    print(f"F1 (weighted): {f1_weighted:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"AUROC: {auroc_value:.4f}")

    print("\nPer-class metrics:")
    print(f"  Refuted:  F1={f1_per_class[0]:.4f}, P={precision_per_class[0]:.4f}, R={recall_per_class[0]:.4f}")
    print(f"  Entailed: F1={f1_per_class[1]:.4f}, P={precision_per_class[1]:.4f}, R={recall_per_class[1]:.4f}")

    print("\nConfusion Matrix:")
    print(f"              Pred Refuted  Pred Entailed")
    print(f"True Refuted      {cm[0,0]:5d}         {cm[0,1]:5d}")
    print(f"True Entailed     {cm[1,0]:5d}         {cm[1,1]:5d}")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Refuted', 'Entailed']))

    # Comparison with baselines
    print("\nComparison:")
    print(f"  Random baseline: 50.00%")
    print(f"  Our model: {accuracy*100:.2f}%")
    print(f"  TAPAS (finetuned): 81.00%")
    print(f"  Human: 92.10%")

    # Save results
    if output_file:
        results = {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'precision_macro': float(precision),
            'recall_macro': float(recall),
            'auroc': float(auroc_value),
            'f1_per_class': {
                'refuted': float(f1_per_class[0]),
                'entailed': float(f1_per_class[1]),
            },
            'precision_per_class': {
                'refuted': float(precision_per_class[0]),
                'entailed': float(precision_per_class[1]),
            },
            'recall_per_class': {
                'refuted': float(recall_per_class[0]),
                'entailed': float(recall_per_class[1]),
            },
            'confusion_matrix': cm.tolist(),
            'num_test_examples': len(test_labels),
            'head_type': model_type,
            'statement_only': statement_only,
            'model_checkpoint': model_checkpoint,
            'test_embeddings': test_embeddings,
        }

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {output_file}")

    return accuracy, f1_macro


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate TabFact classifier on test set"
    )
    parser.add_argument(
        '--model_checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--test_embeddings',
        type=str,
        default=None,
        help='Path to test embeddings pickle file (legacy monolithic format)'
    )
    parser.add_argument(
        '--config_file',
        type=str,
        default=None,
        help='Path to training config file'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Path to save results JSON'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    parser.add_argument(
        '--combine_method',
        type=str,
        default=None,
        choices=['concat', 'add'],
        help='How to combine embeddings (overrides config). Ignored for TAPAS format.'
    )
    # Pre-computed embedding arguments
    parser.add_argument(
        '--table_embeddings',
        type=str,
        default=None,
        help='Path to column embeddings pickle (unified v2 format)'
    )
    parser.add_argument(
        '--statement_embeddings',
        type=str,
        default=None,
        help='Path to statement embeddings pickle (text embedding format)'
    )
    parser.add_argument(
        '--labels_json',
        type=str,
        default=None,
        help='Path to TabFact JSONL (for labels + table_id mapping)'
    )
    parser.add_argument(
        '--table_embedding_variant',
        type=str,
        default='column_mean',
        choices=['column_mean', 'cls_embedding', 'table_embedding', 'token_mean'],
        help='Which table embedding variant to use (default: column_mean)'
    )
    parser.add_argument(
        '--statement_only',
        action='store_true',
        help='Zero out table embeddings (modality ablation)'
    )

    args = parser.parse_args()

    # Validate: need either --test_embeddings or --table_embeddings
    if not args.test_embeddings and not args.table_embeddings:
        parser.error("Either --test_embeddings or --table_embeddings is required")
    if args.table_embeddings and (not args.statement_embeddings or not args.labels_json):
        parser.error("--table_embeddings requires --statement_embeddings and --labels_json")

    # Auto-detect config file
    if args.config_file is None:
        checkpoint_dir = Path(args.model_checkpoint).parent
        config_path = checkpoint_dir / "config.json"
        if config_path.exists():
            args.config_file = str(config_path)

    evaluate_model(
        model_checkpoint=args.model_checkpoint,
        test_embeddings=args.test_embeddings,
        config_file=args.config_file,
        device=args.device,
        output_file=args.output_file,
        combine_method=args.combine_method,
        table_embeddings=args.table_embeddings,
        statement_embeddings=args.statement_embeddings,
        labels_json=args.labels_json,
        statement_only=args.statement_only,
        table_embedding_variant=args.table_embedding_variant,
    )


if __name__ == '__main__':
    main()
