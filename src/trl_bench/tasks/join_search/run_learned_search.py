#!/usr/bin/env python3
"""
Learned projection for join search.

Trains a linear projection head with multi-positive InfoNCE loss so that
cosine similarity in the projected space reflects joinability.  At inference
the trained projection is applied to all column embeddings and the existing
dot-product search pipeline is reused unchanged.

Usage:
    # Full pipeline (train + search + eval)
    python downstream_tasks/join_search/run_learned_search.py \
        --query_emb embeddings/column/bert/opendata.pkl \
        --datalake_emb embeddings/column/bert/opendata.pkl \
        --output_dir results/evaluation/join_search/bert_learned

    # Train only
    python downstream_tasks/join_search/run_learned_search.py \
        --query_emb ... --datalake_emb ... --output_dir ... --train_only

    # Eval only (reuse saved projection + split)
    python downstream_tasks/join_search/run_learned_search.py \
        --query_emb ... --datalake_emb ... --output_dir ... \
        --eval_only --projection_weights <output_dir>/projection_weights.pt
"""
import os
import sys
import pickle
import json
import random
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from trl_bench.tasks.join_search.run_search_and_evaluate import (
    convert_to_tuples,
    run_search,
    run_evaluation,
)


# =============================================================================
# GT Preprocessing
# =============================================================================

def preprocess_gt(gt_path, query_list_path):
    """Load and filter GT to match evaluation semantics in run_search_and_evaluate.py."""
    gt_df = pd.read_csv(gt_path, dtype=str, keep_default_na=False)
    gt_df['query_table'] = gt_df['query_table'].apply(os.path.basename)
    gt_df['candidate_table'] = gt_df['candidate_table'].apply(os.path.basename)
    raw_count = len(gt_df)

    # Remove self-table pairs
    self_mask = gt_df['query_table'] == gt_df['candidate_table']
    self_removed = int(self_mask.sum())
    gt_df = gt_df[~self_mask].reset_index(drop=True)

    # Deduplicate
    pre_dedup = len(gt_df)
    gt_df = gt_df.drop_duplicates().reset_index(drop=True)
    dups_removed = pre_dedup - len(gt_df)

    # Intersect with query list
    query_list = pd.read_csv(query_list_path, dtype=str, keep_default_na=False)
    query_list['query_table'] = query_list['query_table'].apply(os.path.basename)
    valid_query_keys = set(zip(query_list['query_table'], query_list['query_column']))
    gt_query_keys = set(zip(gt_df['query_table'], gt_df['query_column']))
    excluded_queries = gt_query_keys - valid_query_keys
    gt_df = gt_df[
        gt_df.apply(lambda r: (r['query_table'], r['query_column']) in valid_query_keys, axis=1)
    ].reset_index(drop=True)

    n_queries = len(set(zip(gt_df['query_table'], gt_df['query_column'])))
    print("GT Preprocessing:")
    print(f"  Raw:                {raw_count}")
    print(f"  Self-pairs removed: {self_removed}")
    print(f"  Duplicates removed: {dups_removed}")
    print(f"  Queries excluded:   {len(excluded_queries)} (not in query list)")
    print(f"  Final:              {len(gt_df)} rows, {n_queries} queries")
    return gt_df


# =============================================================================
# Query-Role-Disjoint Split
# =============================================================================

def split_by_query(gt_df, train_ratio, val_ratio, seed):
    """Split GT rows by query key so no query appears in multiple splits."""
    query_keys = sorted(set(zip(gt_df['query_table'], gt_df['query_column'])))
    rng = random.Random(seed)
    rng.shuffle(query_keys)

    n = len(query_keys)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_keys = set(query_keys[:n_train])
    val_keys = set(query_keys[n_train:n_train + n_val])
    test_keys = set(query_keys[n_train + n_val:])

    def _filter(keys):
        mask = gt_df.apply(lambda r: (r['query_table'], r['query_column']) in keys, axis=1)
        return gt_df[mask].reset_index(drop=True)

    return (
        _filter(train_keys), _filter(val_keys), _filter(test_keys),
        sorted(train_keys), sorted(val_keys), sorted(test_keys),
    )


# =============================================================================
# Model
# =============================================================================

class ProjectionHead(nn.Module):
    """Projection head aligned with MLPHead conventions in utils/downstream/heads.py.

    Default (num_layers=2, hidden_dim=256):
        Linear(d, 256) -> ReLU -> Dropout -> Linear(256, d) -> L2-normalize
    Linear-only (num_layers=1):
        Dropout -> Linear(d, d) -> L2-normalize
    """

    def __init__(self, dim, hidden_dim=256, num_layers=2, dropout=0.1):
        super().__init__()
        layers = []
        if num_layers == 1:
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(dim, dim))
        else:
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return F.normalize(self.mlp(x), p=2, dim=-1)


# =============================================================================
# Dataset
# =============================================================================

class JoinPairDataset(Dataset):
    """Dataset of (query_emb, candidate_emb) positive pairs from GT."""

    def __init__(self, gt_df, emb_lookup):
        self.query_embs = []
        self.candidate_embs = []
        self.query_keys = []
        self.candidate_keys = []

        skipped = 0
        for _, row in gt_df.iterrows():
            q_key = (row['query_table'], row['query_column'])
            c_key = (row['candidate_table'], row['candidate_column'])
            if q_key not in emb_lookup or c_key not in emb_lookup:
                skipped += 1
                continue
            self.query_embs.append(emb_lookup[q_key])
            self.candidate_embs.append(emb_lookup[c_key])
            self.query_keys.append(q_key)
            self.candidate_keys.append(c_key)

        if skipped > 0:
            print(f"  Skipped {skipped} pairs (missing embeddings)")

        # Positive lookup for multi-positive masking
        self.query_to_positives = defaultdict(set)
        for q, c in zip(self.query_keys, self.candidate_keys):
            self.query_to_positives[q].add(c)

    def __len__(self):
        return len(self.query_embs)

    def __getitem__(self, idx):
        return (
            torch.as_tensor(self.query_embs[idx], dtype=torch.float32),
            torch.as_tensor(self.candidate_embs[idx], dtype=torch.float32),
            idx,
        )


# =============================================================================
# Multi-Positive InfoNCE Loss
# =============================================================================

def _build_positive_mask(batch_indices, dataset):
    """Build a boolean mask where mask[i,j]=True means candidate j is a known
    positive for query i (and j != i), so it should NOT be treated as negative."""
    B = len(batch_indices)
    mask = torch.zeros(B, B, dtype=torch.bool)

    # Inverted index: candidate_key -> list of batch positions
    c_key_to_positions = defaultdict(list)
    for j, idx in enumerate(batch_indices):
        c_key_to_positions[dataset.candidate_keys[idx]].append(j)

    for i, idx in enumerate(batch_indices):
        q_key = dataset.query_keys[idx]
        for c_key in dataset.query_to_positives[q_key]:
            for j in c_key_to_positions.get(c_key, ()):
                if j != i:
                    mask[i, j] = True
    return mask


def multi_positive_infonce(q_proj, c_proj, batch_indices, dataset, temperature):
    """Multi-positive InfoNCE with false-negative masking.

    For each anchor q_i the sampled positive is c_i (diagonal).  Any other
    candidate c_j that is a *known* positive for q_i is masked out of the
    denominator to avoid penalizing true positives as negatives.
    """
    sim = torch.mm(q_proj, c_proj.t()) / temperature
    mask = _build_positive_mask(batch_indices, dataset).to(sim.device)
    sim = sim.masked_fill(mask, float('-inf'))
    labels = torch.arange(sim.size(0), device=sim.device)
    return F.cross_entropy(sim, labels)


# =============================================================================
# Training
# =============================================================================

def train_projection(train_dataset, val_dataset, dim, args):
    """Train projection head and return (model, training_log)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ProjectionHead(dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                           dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, drop_last=True,
    )
    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=0, drop_last=False,
        )

    best_metric = float('inf')
    best_state = None
    training_log = []

    print(f"\nTraining on {device}")
    print(f"  Train pairs: {len(train_dataset)}")
    if val_loader:
        print(f"  Val pairs:   {len(val_dataset)}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Batches/epoch: {len(train_loader)}")
    print(f"  Max epochs:  {args.max_epochs}")
    print()

    for epoch in range(1, args.max_epochs + 1):
        # --- train ---
        model.train()
        train_losses = []
        for q_emb, c_emb, idx in train_loader:
            q_proj = model(q_emb.to(device))
            c_proj = model(c_emb.to(device))
            loss = multi_positive_infonce(q_proj, c_proj, idx.tolist(), train_dataset, args.temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        avg_train = float(np.mean(train_losses))

        # --- val ---
        avg_val = None
        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for q_emb, c_emb, idx in val_loader:
                    q_proj = model(q_emb.to(device))
                    c_proj = model(c_emb.to(device))
                    loss = multi_positive_infonce(
                        q_proj, c_proj, idx.tolist(), val_dataset, args.temperature,
                    )
                    val_losses.append(loss.item())
            avg_val = float(np.mean(val_losses))

        # --- checkpoint (best val_loss, or train_loss if no val) ---
        metric = avg_val if avg_val is not None else avg_train
        if metric < best_metric:
            best_metric = metric
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        entry = {'epoch': epoch, 'train_loss': avg_train}
        if avg_val is not None:
            entry['val_loss'] = avg_val
        training_log.append(entry)

        val_str = f", val_loss={avg_val:.4f}" if avg_val is not None else ""
        print(f"  Epoch {epoch}/{args.max_epochs}: train_loss={avg_train:.4f}{val_str}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n  Restored best checkpoint (metric={best_metric:.4f})")

    return model, training_log


# =============================================================================
# Projection
# =============================================================================

def project_embeddings(model, tuples, device, batch_size=4096):
    """Apply trained projection to all embeddings.

    Returns list of (table, col, projected_emb_np) tuples.
    """
    model.eval()
    embs = np.array([emb for _, _, emb in tuples], dtype=np.float32)
    parts = []
    with torch.no_grad():
        for i in range(0, len(embs), batch_size):
            batch = torch.from_numpy(embs[i:i + batch_size]).to(device)
            parts.append(model(batch).cpu().numpy())
    projected = np.concatenate(parts, axis=0)
    return [(table, col, projected[i]) for i, (table, col, _) in enumerate(tuples)]


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Learned projection for join search",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    p.add_argument("--query_emb", type=str, required=True)
    p.add_argument("--datalake_emb", type=str, required=True)
    p.add_argument("--query_list", type=str,
                   default=os.path.join(_PROJECT_ROOT, "datasets/opendata/queries/opendata_join/opendata_join_query.csv"))
    p.add_argument("--ground_truth", type=str,
                   default=os.path.join(_PROJECT_ROOT, "datasets/opendata/gt/opendata_join_ground_truth.csv"))
    p.add_argument("--output_dir", type=str, required=True)

    # Training hyperparameters (aligned with utils/downstream/config.py defaults)
    p.add_argument("--hidden_dim", type=int, default=256,
                   help="Hidden dimension for projection MLP (matches MLPHead default)")
    p.add_argument("--num_layers", type=int, default=2,
                   help="Number of layers in projection head (1=linear only, 2+=MLP)")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_epochs", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)

    # Search / evaluation
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--k_values", type=int, nargs='+', default=[10, 20, 50])

    # Split (fixed canonical split by default)
    p.add_argument("--split_dir", type=str,
                   default=os.path.join(_PROJECT_ROOT, "datasets/opendata/splits/join_search"),
                   help="Directory containing train_queries.csv, test_queries.csv, test_gt.csv")
    p.add_argument("--train_ratio", type=float, default=None,
                   help="Override fixed split: generate a random split with this train ratio")
    p.add_argument("--val_ratio", type=float, default=0.0)

    # Silver-pairs mode: train on external pairs (e.g., value overlap), eval on full GT
    p.add_argument("--training_pairs", type=str, default=None,
                   help="CSV with (query_table, query_column, candidate_table, candidate_column, similarity) "
                        "for silver-label training. Evaluates on full GT (no split).")
    p.add_argument("--min_containment", type=float, default=0.3,
                   help="Minimum similarity/containment to keep from --training_pairs")

    # Pipeline control
    p.add_argument("--train_only", action="store_true")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--projection_weights", type=str, default=None,
                   help="Path to saved projection weights (for --eval_only)")

    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    if args.eval_only and args.train_only:
        print("Error: cannot specify both --eval_only and --train_only")
        sys.exit(1)
    if args.eval_only and args.projection_weights is None:
        print("Error: --eval_only requires --projection_weights")
        sys.exit(1)
    if not args.eval_only and not args.train_only:
        max_k_eval = max(args.k_values)
        if args.k < max_k_eval:
            print(f"Error: --k ({args.k}) must be >= max(--k_values) ({max_k_eval})")
            sys.exit(1)

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load embeddings
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Learned Projection for Join Search")
    print("=" * 60)

    print("\n[1/6] Loading embeddings...")
    with open(args.query_emb, 'rb') as f:
        query_tuples = convert_to_tuples(pickle.load(f))
    print(f"  Query columns: {len(query_tuples)}")

    with open(args.datalake_emb, 'rb') as f:
        datalake_tuples = convert_to_tuples(pickle.load(f))
    print(f"  Datalake columns: {len(datalake_tuples)}")

    dim = datalake_tuples[0][2].shape[0]
    print(f"  Embedding dim: {dim}")

    # Build lookup: (table, col) -> embedding (float32)
    emb_lookup = {}
    for table, col, emb in datalake_tuples:
        emb_lookup[(table, col)] = emb.astype(np.float32)
    for table, col, emb in query_tuples:
        emb_lookup[(table, col)] = emb.astype(np.float32)

    # ------------------------------------------------------------------
    # Determine training mode
    # ------------------------------------------------------------------
    silver_mode = args.training_pairs is not None and not args.eval_only

    # ------------------------------------------------------------------
    # Training path
    # ------------------------------------------------------------------
    if not args.eval_only:

        if silver_mode:
            # === Silver-pairs mode: train on external pairs, eval on full GT ===
            print(f"\n[2/6] Loading silver training pairs from: {args.training_pairs}")
            silver_df = pd.read_csv(args.training_pairs, dtype=str, keep_default_na=False)
            silver_df['query_table'] = silver_df['query_table'].apply(os.path.basename)
            silver_df['candidate_table'] = silver_df['candidate_table'].apply(os.path.basename)
            # Filter by containment threshold (similarity column)
            if 'similarity' in silver_df.columns:
                silver_df['similarity'] = silver_df['similarity'].astype(float)
                pre_filter = len(silver_df)
                silver_df = silver_df[silver_df['similarity'] >= args.min_containment].reset_index(drop=True)
                print(f"  Loaded: {pre_filter} pairs, after threshold >= {args.min_containment}: {len(silver_df)}")
            else:
                print(f"  Loaded: {len(silver_df)} pairs (no similarity column, using all)")
            # Remove self-table pairs
            self_mask = silver_df['query_table'] == silver_df['candidate_table']
            if self_mask.any():
                silver_df = silver_df[~self_mask].reset_index(drop=True)
            # Dedup
            pair_cols = ['query_table', 'query_column', 'candidate_table', 'candidate_column']
            silver_df = silver_df.drop_duplicates(subset=pair_cols).reset_index(drop=True)
            print(f"  Final silver pairs: {len(silver_df)}")

            # 90/10 random split for train/val monitoring
            print(f"\n[3/6] Splitting silver pairs (90/10 random)...")
            indices = list(range(len(silver_df)))
            rng = random.Random(args.seed)
            rng.shuffle(indices)
            n_train = int(len(indices) * 0.9)
            train_idx = set(indices[:n_train])
            train_df = silver_df.iloc[sorted(train_idx)].reset_index(drop=True)
            val_df = silver_df.iloc[sorted(set(indices) - train_idx)].reset_index(drop=True)
            print(f"  Train: {len(train_df)} pairs")
            print(f"  Val:   {len(val_df)} pairs")

            # Full GT evaluation — use the original query list and ground truth
            eval_query_path = args.query_list
            eval_gt_path = args.ground_truth

        else:
            # === GT-split mode: train on GT split, eval on held-out queries ===
            print(f"\n[2/6] Preprocessing ground truth...")
            gt_df = preprocess_gt(args.ground_truth, args.query_list)

            use_fixed_split = (args.train_ratio is None and args.split_dir
                               and os.path.isdir(args.split_dir))

            if use_fixed_split:
                # --- Load canonical fixed split ---
                print(f"\n[3/6] Loading fixed split from: {args.split_dir}")
                train_q_df = pd.read_csv(
                    os.path.join(args.split_dir, 'train_queries.csv'),
                    dtype=str, keep_default_na=False,
                )
                train_keys = set(zip(train_q_df['query_table'], train_q_df['query_column']))
                test_q_df = pd.read_csv(
                    os.path.join(args.split_dir, 'test_queries.csv'),
                    dtype=str, keep_default_na=False,
                )
                test_keys = set(zip(test_q_df['query_table'], test_q_df['query_column']))

                train_mask = gt_df.apply(
                    lambda r: (r['query_table'], r['query_column']) in train_keys, axis=1)
                test_mask = gt_df.apply(
                    lambda r: (r['query_table'], r['query_column']) in test_keys, axis=1)
                train_df = gt_df[train_mask].reset_index(drop=True)
                val_df = pd.DataFrame(columns=gt_df.columns)  # no val split
                test_df = gt_df[test_mask].reset_index(drop=True)

                print(f"  Train: {len(train_keys)} queries, {len(train_df)} pairs")
                print(f"  Test:  {len(test_keys)} queries, {len(test_df)} pairs")

                eval_gt_path = os.path.join(args.split_dir, 'test_gt.csv')
                eval_query_path = os.path.join(args.split_dir, 'test_queries.csv')
            else:
                # --- Generate random split ---
                train_ratio = args.train_ratio if args.train_ratio is not None else 0.2
                print(f"\n[3/6] Splitting by query (query-role-disjoint, {train_ratio}/{args.val_ratio}/{1 - train_ratio - args.val_ratio})...")
                train_df, val_df, test_df, train_keys_list, val_keys_list, test_keys_list = split_by_query(
                    gt_df, train_ratio, args.val_ratio, args.seed,
                )
                train_keys = set(train_keys_list)
                test_keys = set(test_keys_list)
                print(f"  Train: {len(train_keys)} queries, {len(train_df)} pairs")
                if len(val_df) > 0:
                    print(f"  Val:   {len(val_keys_list)} queries, {len(val_df)} pairs")
                print(f"  Test:  {len(test_keys_list)} queries, {len(test_df)} pairs")

                # Save split + test files for evaluation
                split_info = {
                    'seed': args.seed,
                    'train_ratio': train_ratio,
                    'val_ratio': args.val_ratio,
                    'train_queries': [list(k) for k in train_keys_list],
                    'val_queries': [list(k) for k in val_keys_list],
                    'test_queries': [list(k) for k in test_keys_list],
                    'train_pairs': len(train_df),
                    'val_pairs': len(val_df),
                    'test_pairs': len(test_df),
                }
                with open(os.path.join(args.output_dir, 'split_info.json'), 'w') as f:
                    json.dump(split_info, f, indent=2)

                eval_gt_path = os.path.join(args.output_dir, 'test_gt.csv')
                eval_query_path = os.path.join(args.output_dir, 'test_queries.csv')
                test_df.to_csv(eval_gt_path, index=False)
                pd.DataFrame(test_keys_list, columns=['query_table', 'query_column']).to_csv(
                    eval_query_path, index=False,
                )

        # Create datasets
        print(f"\n[4/6] Creating datasets...")
        train_dataset = JoinPairDataset(train_df, emb_lookup)
        val_dataset = JoinPairDataset(val_df, emb_lookup)
        print(f"  Train: {len(train_dataset)} usable pairs")
        print(f"  Val:   {len(val_dataset)} usable pairs")
        emb_match_rate = (len(train_dataset) + len(val_dataset)) / max(len(train_df) + len(val_df), 1)
        print(f"  Embedding match rate: {emb_match_rate:.1%}")

        # Train
        print(f"\n[5/6] Training projection head...")
        model, training_log = train_projection(train_dataset, val_dataset, dim, args)

        # Save weights + log
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'projection_weights.pt'))
        with open(os.path.join(args.output_dir, 'training_log.json'), 'w') as f:
            json.dump(training_log, f, indent=2)

    else:
        # --eval_only: load saved projection
        print(f"\n[2-5/6] Skipped (eval-only mode)")
        print(f"  Loading projection from: {args.projection_weights}")
        model = ProjectionHead(dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                               dropout=0.0)
        model.load_state_dict(torch.load(args.projection_weights, map_location='cpu', weights_only=True))
        if silver_mode or args.training_pairs:
            eval_query_path = args.query_list
            eval_gt_path = args.ground_truth
        elif args.split_dir and os.path.isdir(args.split_dir):
            eval_gt_path = os.path.join(args.split_dir, 'test_gt.csv')
            eval_query_path = os.path.join(args.split_dir, 'test_queries.csv')
        else:
            eval_gt_path = os.path.join(args.output_dir, 'test_gt.csv')
            eval_query_path = os.path.join(args.output_dir, 'test_queries.csv')
            if not os.path.exists(eval_gt_path) or not os.path.exists(eval_query_path):
                print(f"Error: test split files not found in --split_dir or --output_dir.")
                sys.exit(1)

    if args.train_only:
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE (--train_only)")
        print(f"  Projection weights: {os.path.join(args.output_dir, 'projection_weights.pt')}")
        print("=" * 60)
        return

    # ------------------------------------------------------------------
    # Inference: project + search + evaluate
    # ------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print(f"\n[6/6] Projecting embeddings and running search...")
    proj_datalake = project_embeddings(model, datalake_tuples, device)
    proj_query = project_embeddings(model, query_tuples, device)
    print(f"  Projected {len(proj_datalake)} datalake + {len(proj_query)} query columns")

    # Save projected embeddings as temp pickles for run_search()
    proj_dl_path = os.path.join(args.output_dir, '_proj_datalake.pkl')
    proj_q_path = os.path.join(args.output_dir, '_proj_query.pkl')
    with open(proj_dl_path, 'wb') as f:
        pickle.dump(proj_datalake, f)
    with open(proj_q_path, 'wb') as f:
        pickle.dump(proj_query, f)

    # Run search
    results_df, _ = run_search(
        query_emb_path=proj_q_path,
        datalake_emb_path=proj_dl_path,
        query_list_path=eval_query_path,
        k=args.k,
        threshold=0,
    )

    # Save results
    results_csv_path = os.path.join(args.output_dir, 'results.csv')
    results_df.to_csv(results_csv_path, index=False)

    # Evaluate
    metrics = run_evaluation(
        results_df=results_df,
        ground_truth_path=eval_gt_path,
        k_values=args.k_values,
    )

    # Save metrics
    results_json = {
        'task': 'join_search_learned',
        'training_mode': 'silver' if silver_mode else 'gt_split',
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'seed': args.seed,
        'temperature': args.temperature,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'max_epochs': args.max_epochs,
        'dropout': args.dropout,
    }
    if silver_mode:
        results_json['training_pairs'] = args.training_pairs
        results_json['min_containment'] = args.min_containment
    for k_val, m in metrics.items():
        if k_val == 'map':
            results_json['col_map'] = m
        else:
            results_json[f'col_precision_at_{k_val}'] = m['precision']
            results_json[f'col_recall_at_{k_val}'] = m['recall']
            results_json[f'col_f1_at_{k_val}'] = m['f1']

    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results_json, f, indent=2)

    # Clean up temp projected embeddings
    for p in (proj_dl_path, proj_q_path):
        if os.path.exists(p):
            os.remove(p)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"  Results: {results_csv_path}")
    print(f"  Metrics: {os.path.join(args.output_dir, 'results.json')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
