#!/usr/bin/env python3
"""
Evaluate Column Type Prediction Classifier

Computes MAP, micro-F1, and macro-F1 using argmax (single-label) prediction.

Usage:
    python evaluate_ct_mode4.py --classifier_path classifier_mode4/best_model.pt
"""

import argparse
import os
import pickle
import sys

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.metrics import f1_score
from train_ct_mode4 import EmbeddingDataset, collate_fn, ct_forward_with_loss, compute_map, load_unified_embeddings
from trl_bench.utils.downstream.heads import MLPHead


def evaluate(model, dataloader, device):
    """
    Evaluate column type prediction using argmax (single-label) prediction.

    Matches the evaluation semantics of CTTaskSpec.compute_metrics() in the
    training script. Per TURL, argmax is correct for single-label datasets
    (SATO, SOTAB); threshold-based binarization is for multi-label datasets.

    Args:
        model: Trained classifier
        dataloader: Test dataloader
        device: Device to use

    Returns:
        dict with metrics (map, micro_f1, macro_f1)
    """
    model.eval()

    # Collect per-column results across batches.  Each batch may have a
    # different max_cols (from collate_fn padding), so we flatten to
    # per-column vectors within each batch to avoid shape mismatches.
    all_col_logits = []   # list of (num_types,) tensors, one per valid column
    all_col_labels = []   # list of (num_types,) tensors, one per valid column

    print("Generating predictions...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            embeddings, labels, labels_mask, _ = batch

            embeddings = embeddings.to(device)
            labels = labels.to(device)
            labels_mask = labels_mask.to(device)

            logits = ct_forward_with_loss(model, embeddings)[0]

            # Flatten to per-column and keep only valid (unpadded) columns
            logits_flat = logits.view(-1, logits.shape[-1])        # (B*C, T)
            labels_flat = labels.view(-1, labels.shape[-1])        # (B*C, T)
            mask_flat = labels_mask.view(-1)                       # (B*C,)
            valid = mask_flat == 1

            all_col_logits.append(logits_flat[valid].cpu())
            all_col_labels.append(labels_flat[valid].cpu())

    # Concatenate along the column axis (all tensors are 2-D with same dim=-1)
    col_logits = torch.cat(all_col_logits, dim=0)   # (N_valid, num_types)
    col_labels = torch.cat(all_col_labels, dim=0)    # (N_valid, num_types)

    # MAP (per-column average precision)
    map_score = compute_map(
        col_logits.unsqueeze(0), col_labels.unsqueeze(0),
        torch.ones(1, col_logits.shape[0]),
    )

    # F1 (argmax, matching training eval)
    num_columns = col_logits.shape[0]
    if num_columns > 0:
        y_pred = torch.sigmoid(col_logits).argmax(dim=1).numpy()
        y_true = col_labels.argmax(dim=1).numpy()
        micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    else:
        micro_f1 = 0.0
        macro_f1 = 0.0

    print(f"Valid columns evaluated: {num_columns}")

    return {
        'map': map_score,
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'num_columns': num_columns,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate CT Mode 4 classifier')
    parser.add_argument('--classifier_path', type=str,
                        required=True,
                        help='Path to trained classifier checkpoint')
    parser.add_argument('--embeddings', type=str,
                        required=True,
                        help='Path to unified column embeddings .pkl file')
    parser.add_argument('--test_csv', type=str,
                        required=True,
                        help='Path to test labels CSV file')
    parser.add_argument('--batch_size', type=int,
                        default=20,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')

    args = parser.parse_args()

    # Resolve relative paths against CWD (not script_dir)
    args.embeddings = os.path.abspath(args.embeddings)
    args.test_csv = os.path.abspath(args.test_csv)
    args.classifier_path = os.path.abspath(args.classifier_path)

    print("="*80)
    print("Column Type Prediction Evaluation")
    print("="*80)
    print(f"Classifier: {args.classifier_path}")
    print(f"Embeddings: {args.embeddings}")
    print(f"Device: {args.device}")
    print()

    # Load checkpoint first (need class_to_idx and model metadata)
    print(f"Loading model from {args.classifier_path}")
    checkpoint = torch.load(args.classifier_path, map_location=args.device, weights_only=False)

    hidden_size = checkpoint['hidden_size']
    num_types = checkpoint['num_types']
    dropout = checkpoint.get('dropout', 0.1)
    # Old (pre-unified) checkpoints used num_layers=1 and had no hidden_dim field.
    # New checkpoints save these explicitly. Fallback to old geometry for compat.
    hidden_dim = checkpoint.get('hidden_dim', 256)
    num_layers = checkpoint.get('num_layers', 1)

    # Load test dataset
    if 'class_to_idx' not in checkpoint:
        raise ValueError(
            "Checkpoint is missing 'class_to_idx'. "
            "Re-train with the latest train_ct_mode4.py to produce a compatible checkpoint."
        )
    class_to_idx = checkpoint['class_to_idx']
    emb, lab, mask, ids, num_classes = load_unified_embeddings(
        args.embeddings, args.test_csv, class_to_idx=class_to_idx)

    # Apply z-score normalization if checkpoint was trained with it
    if 'emb_scaler_mean' in checkpoint:
        from sklearn.preprocessing import StandardScaler
        emb_scaler = StandardScaler()
        emb_scaler.mean_ = np.array(checkpoint['emb_scaler_mean'])
        emb_scaler.scale_ = np.array(checkpoint['emb_scaler_scale'])
        emb_scaler.var_ = emb_scaler.scale_ ** 2
        emb_scaler.n_features_in_ = len(emb_scaler.mean_)
        for j in range(len(emb)):
            emb[j] = emb_scaler.transform(emb[j].astype(np.float32))
        print(f"  Applied embedding z-score from checkpoint")

    test_dataset = EmbeddingDataset(emb, lab, mask, ids)

    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=args.batch_size,
        collate_fn=collate_fn
    )

    print(f"Test samples: {len(test_dataset)}")
    print()

    # Build model (read full head geometry from checkpoint; old checkpoints default to 256/2)
    # Backward compat: old checkpoints were trained with dropout_first=True hardcoded
    dropout_first = checkpoint.get('dropout_first', True)
    model = MLPHead(input_dim=hidden_size, output_dim=num_types,
                    hidden_dim=hidden_dim, num_layers=num_layers,
                    dropout=dropout, dropout_first=dropout_first)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()

    print(f"  Hidden size: {hidden_size} (hidden_dim={hidden_dim}, num_layers={num_layers})")
    print(f"  Num types: {num_types}")
    print(f"  Trained epoch: {checkpoint.get('epoch', 'N/A')}")
    best_map = checkpoint.get('best_map', checkpoint.get('val_map', checkpoint.get('test_map', None)))
    print(f"  Best MAP: {best_map:.4f}" if best_map is not None else "  Best MAP: N/A")
    print()

    # Evaluate (argmax-based single-label prediction, matching training eval)
    results = evaluate(model, test_dataloader, args.device)

    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"  MAP:      {results['map']:.4f} ({results['map']*100:.2f}%)")
    print(f"  Micro F1: {results['micro_f1']:.4f} ({results['micro_f1']*100:.2f}%)")
    print(f"  Macro F1: {results['macro_f1']:.4f} ({results['macro_f1']*100:.2f}%)")
    print(f"\n  Columns evaluated: {results['num_columns']:,}")
    print("="*80)
    print()

    # Save results
    output_dir = os.path.dirname(args.classifier_path)
    results_file = os.path.join(output_dir, 'evaluation_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to: {results_file}")


if __name__ == '__main__':
    main()
