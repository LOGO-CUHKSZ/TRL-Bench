"""
Step 7: Aggregate column embeddings to table-level vectors for FAISS retrieval.

.. deprecated::
    This script is deprecated. Use ``scripts/generate_table_embeddings.py`` instead,
    which produces table embedding pkls at ``embeddings/table/{model}/{dataset}.pkl``
    with all standard variants (cls_embedding, table_embedding, column_mean).
    Step 8 now reads ``column_mean`` from those table embedding pkls directly.

Aggregation method (from PLAN.md):
  1. L2-normalize each column vector
  2. Mean pool across columns
  3. L2-normalize the result

Produces per column model M:
  - embeddings/table/column/{M}/dlte_v1_lake_embeddings.npy
  - embeddings/table/column/{M}/dlte_v1_lake_table_ids.txt
  - embeddings/table/column/{M}/dlte_v1_query_embeddings.npy
  - embeddings/table/column/{M}/dlte_v1_query_table_ids.txt
  - embeddings/table/column/{M}/dlte_v1_aggregation.json

Usage:
    python downstream_tasks/dlte/scripts/step7_aggregate_table_embeddings.py
    python downstream_tasks/dlte/scripts/step7_aggregate_table_embeddings.py --models bert tabbie
"""

import argparse
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.warn(
    "step7_aggregate_table_embeddings.py is deprecated. "
    "Use scripts/generate_table_embeddings.py instead.",
    DeprecationWarning,
    stacklevel=2,
)

PROJECT_ROOT = COL_EMB_ROOT = TABLE_EMB_ROOT = DATASET_ROOT = None


def resolve_paths(args):
    global PROJECT_ROOT, COL_EMB_ROOT, TABLE_EMB_ROOT, DATASET_ROOT
    PROJECT_ROOT = Path(args.project_root) if args.project_root else Path(__file__).resolve().parents[3]
    COL_EMB_ROOT = PROJECT_ROOT / "assets" / "embeddings" / "column"
    TABLE_EMB_ROOT = PROJECT_ROOT / "assets" / "embeddings" / "table" / "column"
    DATASET_ROOT = PROJECT_ROOT / "datasets" / "dlte_v1"

COLUMN_MODELS = ["bert", "gte", "starmie", "tabbie", "tabert", "tabsketchfm", "tapas", "turl"]


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def l2_normalize(v):
    """L2-normalize a vector. Returns zero vector if norm is 0."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def aggregate_table_embedding(column_embeddings):
    """Aggregate column embeddings to a single table vector.

    Method: L2-normalize each column vector, mean pool, L2-normalize result.
    """
    if not column_embeddings:
        return None

    vecs = []
    for col_idx in sorted(column_embeddings.keys()):
        v = np.asarray(column_embeddings[col_idx], dtype=np.float32)
        vecs.append(l2_normalize(v))

    if not vecs:
        return None

    mean_vec = np.mean(vecs, axis=0).astype(np.float32)
    return l2_normalize(mean_vec).astype(np.float32)


def build_embedding_lookup(entries):
    """Build table_id -> table_embedding dict from pickle entries."""
    lookup = {}
    for entry in entries:
        col_embs = entry.get("column_embeddings", {})
        table_vec = aggregate_table_embedding(col_embs)
        if table_vec is not None:
            lookup[entry["table_id"]] = table_vec
    return lookup


def process_model(model_name):
    """Process one column model: build query + lake table embeddings."""
    print(f"\n  Model: {model_name}")
    t0 = time.time()

    # Load pickles
    queries_path = COL_EMB_ROOT / model_name / "dlte_v1_queries.pkl"
    targets_path = COL_EMB_ROOT / model_name / "dlte_v1_targets.pkl"
    ckan_path = COL_EMB_ROOT / model_name / "ckan_subset.pkl"

    for p in [queries_path, targets_path, ckan_path]:
        if not p.exists():
            print(f"    SKIP: {p.name} not found")
            return False

    queries = load_pkl(queries_path)
    targets = load_pkl(targets_path)
    ckan = load_pkl(ckan_path)

    # Detect embedding dim from first entry
    first_entry = queries[0]
    first_col = list(first_entry["column_embeddings"].values())[0]
    dim = np.asarray(first_col).shape[0]
    print(f"    dim={dim}, queries={len(queries)}, targets={len(targets)}, ckan={len(ckan)}")

    # Build lookups
    print("    Aggregating query embeddings...")
    q_lookup = build_embedding_lookup(queries)
    print(f"    -> {len(q_lookup)} query table vectors")

    print("    Aggregating target embeddings...")
    t_lookup = build_embedding_lookup(targets)
    print(f"    -> {len(t_lookup)} target table vectors")

    print("    Aggregating CKAN embeddings...")
    c_lookup = build_embedding_lookup(ckan)
    print(f"    -> {len(c_lookup)} CKAN table vectors")

    # Load lake manifest for ordering
    lake_manifest_path = DATASET_ROOT / "manifests" / "lake_manifest.jsonl"
    lake_entries = []
    with open(lake_manifest_path) as f:
        for line in f:
            lake_entries.append(json.loads(line.strip()))

    # Build lake arrays (targets + CKAN, ordered by manifest)
    lake_ids = []
    lake_vecs = []
    missing_lake = 0
    for entry in lake_entries:
        tid = entry["table_id"]
        vec = t_lookup.get(tid)
        if vec is None:
            vec = c_lookup.get(tid)
        if vec is not None:
            lake_ids.append(tid)
            lake_vecs.append(vec)
        else:
            missing_lake += 1

    if missing_lake > 0:
        print(f"    WARN: {missing_lake} lake entries missing embeddings")

    # Build query arrays (ordered by pickle order)
    query_ids = []
    query_vecs = []
    for entry in queries:
        tid = entry["table_id"]
        if tid in q_lookup:
            query_ids.append(tid)
            query_vecs.append(q_lookup[tid])

    # Stack into contiguous float32 arrays
    lake_embeddings = np.ascontiguousarray(np.stack(lake_vecs), dtype=np.float32)
    query_embeddings = np.ascontiguousarray(np.stack(query_vecs), dtype=np.float32)

    print(f"    lake: {lake_embeddings.shape}, queries: {query_embeddings.shape}")

    # Save outputs
    out_dir = TABLE_EMB_ROOT / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "dlte_v1_lake_embeddings.npy", lake_embeddings)
    np.save(out_dir / "dlte_v1_query_embeddings.npy", query_embeddings)

    with open(out_dir / "dlte_v1_lake_table_ids.txt", "w") as f:
        f.write("\n".join(lake_ids) + "\n")

    with open(out_dir / "dlte_v1_query_table_ids.txt", "w") as f:
        f.write("\n".join(query_ids) + "\n")

    config = {
        "model": model_name,
        "embedding_dim": dim,
        "aggregation_method": "l2_norm_columns_then_mean_then_l2_norm",
        "n_lake": len(lake_ids),
        "n_queries": len(query_ids),
        "n_lake_missing": missing_lake,
        "lake_sources": {
            "dlte_targets": len(t_lookup),
            "ckan_distractors": len(c_lookup),
        },
    }
    with open(out_dir / "dlte_v1_aggregation.json", "w") as f:
        json.dump(config, f, indent=2)

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate column embeddings to table-level vectors for FAISS")
    parser.add_argument("--models", nargs="+", default=COLUMN_MODELS,
                        help="Column models to process")
    parser.add_argument("--project_root", type=str, default=None,
                        help="Project root directory (default: auto-detect)")
    args = parser.parse_args()
    resolve_paths(args)

    print("Step 7: Table-level Embeddings via Aggregation")
    print("=" * 60)

    succeeded = 0
    for model in args.models:
        if process_model(model):
            succeeded += 1

    print(f"\n{'='*60}")
    print(f"Processed {succeeded}/{len(args.models)} models")
    print(f"Output: {TABLE_EMB_ROOT}")
    print(f"{'='*60}")

    return 0 if succeeded == len(args.models) else 1


if __name__ == "__main__":
    sys.exit(main())
