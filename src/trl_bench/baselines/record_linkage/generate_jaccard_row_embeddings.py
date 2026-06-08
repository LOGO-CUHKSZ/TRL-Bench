#!/usr/bin/env python3
"""
Generate Jaccard-style token-overlap row embeddings for record linkage.

Each row is serialized as `col: val | col: val | ...` (matching the GTE / BERT
row template), tokenized into word unigrams, and represented as an L2-normalized
binary token-presence vector. Under cosine similarity these vectors yield
Tanimoto similarity ( |A∩B| / sqrt(|A|·|B|) ), a near-monotone variant of token
Jaccard that the existing cosine-threshold linkage head consumes directly.
With a learned MLP / linear head over concatenated row vectors the token-level
match signal remains accessible to the probe, so all four standard linkage
heads work without modification.

Mirrors the structure of utils/baselines/record_linkage/generate_tfidf_row_embeddings.py
so the two row-level string baselines (char n-gram TF-IDF vs. token-Jaccard)
can be compared head-to-head on the same downstream pipeline.

Usage:
    python utils/baselines/record_linkage/generate_jaccard_row_embeddings.py --all
    python utils/baselines/record_linkage/generate_jaccard_row_embeddings.py \\
        --datasets wdc_products_small --max_features 1024
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trl_bench.utils.row_embedding.directory import build_table_result, save_aggregate_pickle


LINKAGE_DATASETS = [
    "deepmatcher_abt_buy",
    "deepmatcher_amazon_google",
    "deepmatcher_beer",
    "deepmatcher_dblp_acm",
    "deepmatcher_dblp_acm_dirty",
    "deepmatcher_dblp_scholar",
    "deepmatcher_dblp_scholar_dirty",
    "deepmatcher_fodors_zagats",
    "deepmatcher_itunes_amazon",
    "deepmatcher_itunes_amazon_dirty",
    "deepmatcher_walmart_amazon",
    "deepmatcher_walmart_amazon_dirty",
    "wdc_products_small",
    "wdc_products_medium",
    "wdc_products_large",
    "wdc_products_xlarge",
]


def serialize_row(columns, values, max_chars_per_cell: int = 100) -> str:
    parts = []
    for col, val in zip(columns, values):
        if pd.isna(val) or val is None:
            val_str = ""
        else:
            val_str = str(val)[:max_chars_per_cell]
        parts.append(f"{col}: {val_str}")
    return " | ".join(parts)


def embed_dataset(dataset_dir: Path, model_name: str,
                  max_features: int, ngram_range: tuple[int, int]) -> list[dict]:
    tables_dir = dataset_dir / "tables"
    table_paths = [tables_dir / "tableA.csv", tables_dir / "tableB.csv"]
    for tp in table_paths:
        if not tp.exists():
            raise FileNotFoundError(f"Missing table: {tp}")

    serialized_per_table: list[list[str]] = []
    column_names_per_table: list[list[str]] = []
    for tp in table_paths:
        df = pd.read_csv(tp, dtype=str, keep_default_na=False)
        if len(df) == 0:
            raise ValueError(f"Empty table: {tp}")
        cols = list(df.columns)
        rows = [serialize_row(cols, df.iloc[i].values) for i in range(len(df))]
        serialized_per_table.append(rows)
        column_names_per_table.append(cols)

    all_rows = serialized_per_table[0] + serialized_per_table[1]
    # Binary token-presence vectors with L2 normalization → cosine = Tanimoto coefficient
    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=ngram_range,
        max_features=max_features,
        lowercase=True,
        binary=True,        # 0/1 token presence (no count weighting)
        use_idf=False,      # raw token presence
        norm="l2",          # so cosine ∈ [0, 1] = |A∩B|/sqrt(|A|·|B|)
        token_pattern=r"(?u)\b\w+\b",
    )
    matrix = vec.fit_transform(all_rows)
    matrix = matrix.toarray().astype(np.float32)

    n_a = len(serialized_per_table[0])
    embeddings_per_table = [matrix[:n_a], matrix[n_a:]]

    results: list[dict] = []
    for tp, embs, cols in zip(table_paths, embeddings_per_table, column_names_per_table):
        results.append(build_table_result(
            table_path=str(tp),
            row_embeddings=embs,
            column_names=cols,
            model_name=model_name,
        ))
    return results


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--datasets", nargs="+", help="Linkage datasets to process")
    g.add_argument("--all", action="store_true",
                   help=f"Process all {len(LINKAGE_DATASETS)} linkage datasets")
    p.add_argument("--model_name", default="jaccard_row",
                   help="Output directory under embeddings/row/ (default: jaccard_row)")
    p.add_argument("--max_features", type=int, default=512,
                   help="Vocabulary size = embedding dimension (default: 512)")
    p.add_argument("--ngram_range", nargs=2, type=int, default=[1, 1],
                   help="Word n-gram range (default: 1 1, i.e. unigrams)")
    p.add_argument("--datasets_root", default=None)
    p.add_argument("--output_root", default=None)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    datasets = LINKAGE_DATASETS if args.all else args.datasets

    datasets_root = Path(args.datasets_root) if args.datasets_root \
        else PROJECT_ROOT / "datasets" / "record_linkage"
    output_root = Path(args.output_root) if args.output_root \
        else PROJECT_ROOT / "assets" / "embeddings" / "row"

    out_dir = output_root / args.model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    ngram = (args.ngram_range[0], args.ngram_range[1])
    print(f"Jaccard row baseline | max_features={args.max_features} | ngram={ngram} | "
          f"output={out_dir}")

    for ds in datasets:
        ds_dir = datasets_root / ds
        out_path = out_dir / f"{ds}.pkl"
        if out_path.exists() and not args.force:
            print(f"  {ds}: already exists at {out_path} (use --force)")
            continue
        if not ds_dir.is_dir():
            print(f"  SKIP {ds}: dataset dir not found at {ds_dir}")
            continue
        try:
            results = embed_dataset(ds_dir, args.model_name, args.max_features, ngram)
        except Exception as e:
            print(f"  FAIL {ds}: {e}")
            continue
        save_aggregate_pickle(results, str(out_path))
        n_rows = sum(r["num_rows"] for r in results)
        d = results[0]["embedding_dim"]
        print(f"  {ds}: 2 tables, {n_rows} rows, dim={d} -> {out_path}")


if __name__ == "__main__":
    main()
