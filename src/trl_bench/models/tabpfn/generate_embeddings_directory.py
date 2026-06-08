"""
TabPFN Directory-Mode Row Embedding Generation

Processes a directory of CSV files and produces an aggregate pickle
containing row embeddings for each table. No train/test split — each
CSV is embedded as a whole.

Output format: List[dict] pickle at --output_path, one entry per table:
  {table, table_id, row_embeddings, column_names, model_name, embedding_dim, num_rows}
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../" * 2))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from tabpfn_extensions import TabPFNClassifier
from tabpfn.base import ClassifierModelSpecs

from trl_bench.utils.row_embedding.directory import (
    discover_csv_files,
    build_table_result,
    save_aggregate_pickle,
    register_save_on_signal,
    load_existing_results,
    get_completed_table_ids,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate row embeddings for a directory of CSV files using TabPFN"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory containing CSV files"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for aggregate pickle",
    )
    parser.add_argument(
        "--n_estimators", type=int, default=8, help="Number of TabPFN estimators (default: 8)"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device: auto, cuda, cpu (default: auto)"
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=50,
        help="Save intermediate results every N tables (default: 50)",
    )
    parser.add_argument(
        "--label_columns", type=str, nargs='*', default=None,
        help="Label columns to exclude from features",
    )
    parser.add_argument("--table_list", default=None, help="Path to table list file for shard filtering")
    return parser.parse_args()


def embed_table(model, csv_path, label_columns=None):
    """Embed a single CSV file using TabPFN.

    Returns a table result dict, or None if the table cannot be processed.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  SKIP {csv_path.name}: cannot read CSV: {e}")
        return None

    if len(df) < 2:
        print(f"  SKIP {csv_path.name}: too few rows ({len(df)})")
        return None

    X = df.copy()
    if label_columns:
        cols_to_drop = [c for c in label_columns if c in X.columns]
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)
    column_names = list(X.columns)

    # Replace inf with NaN — some web tables have division-by-zero values
    num_cols = X.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan)

    # Detect categorical feature indices for TabPFN
    category_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    categorical_indices = [i for i, col in enumerate(X.columns) if col in category_cols]

    # LabelEncode categorical columns to numeric — TabPFN's internal preprocessing
    # calls np.isnan() which fails on object-dtype arrays with TypeError.
    for col in category_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Update categorical indices per table (no model re-creation needed)
    model.categorical_features_indices = categorical_indices if categorical_indices else None

    # Dummy labels — TabPFN's .fit() requires labels but they don't affect embeddings
    y_dummy = np.zeros(len(X), dtype=int)

    try:
        model.fit(X, y_dummy)

        # After first fit, cache the loaded model to skip disk I/O on subsequent fits.
        # ClassifierModelSpecs causes initialize_tabpfn_model() to return the
        # in-memory model immediately instead of calling torch.load() from disk.
        if not isinstance(model.model_path, ClassifierModelSpecs):
            model.model_path = ClassifierModelSpecs(
                model=model.models_[0],
                architecture_config=model.configs_[0],
                inference_config=model.inference_config_,
            )

        # Sync categorical indices to TabPFN's actual inferred subset after fit.
        # TabPFN may remove constant columns or reclassify high-cardinality
        # categoricals as numerical during fit, causing a fit/transform mismatch
        # if get_embeddings() reuses the original stale list.
        from tabpfn.preprocessing.pipeline_interface import FeatureModality
        inferred = model.inferred_feature_schema_.indices_for(FeatureModality.CATEGORICAL)
        model.categorical_features_indices = inferred if inferred else None

        embeddings = model.get_embeddings(X, data_source="test")
    except Exception as e:
        print(f"  SKIP {csv_path.name}: embedding failed: {e}")
        return None

    # Average over estimators if present
    if embeddings.ndim == 3:
        embeddings = embeddings.mean(axis=0)

    return build_table_result(
        table_path=str(csv_path),
        row_embeddings=embeddings,
        column_names=column_names,
        model_name="TabPFN",
    )


def main():
    args = parse_args()

    print("=" * 80)
    print("TabPFN Directory-Mode Row Embedding Generation")
    print("=" * 80)

    # Discover tables
    csv_files = discover_csv_files(args.input_dir, table_list_path=args.table_list)
    print(f"Found {len(csv_files)} CSV files in {args.input_dir}")

    if not csv_files:
        print("No CSV files found. Exiting.")
        sys.exit(0)

    # Resume support
    results = load_existing_results(args.output_path)
    completed = get_completed_table_ids(results)
    register_save_on_signal(results, args.output_path)
    if completed:
        print(f"Resuming: {len(completed)} tables already processed")

    # Single model instance — reused across all tables.
    # After first fit(), model.model_path is set to ClassifierModelSpecs
    # to cache the loaded weights and skip disk I/O on subsequent fits.
    model = TabPFNClassifier(
        n_estimators=args.n_estimators,
        device=args.device,
        random_state=42,
        ignore_pretraining_limits=True,
        memory_saving_mode=True,
        fit_mode="low_memory",
    )

    # Process tables
    newly_processed = 0
    for i, csv_path in enumerate(csv_files):
        table_id = csv_path.stem
        if table_id in completed:
            continue

        print(f"\n[{i + 1}/{len(csv_files)}] Processing {csv_path.name}...")
        result = embed_table(model, csv_path, label_columns=args.label_columns)

        if result is not None:
            results.append(result)
            newly_processed += 1
            print(
                f"  Embedded: {result['num_rows']} rows x {result['embedding_dim']} dim"
            )

        # Periodic checkpoint
        if newly_processed > 0 and newly_processed % args.checkpoint_interval == 0:
            save_aggregate_pickle(results, args.output_path)
            print(f"  Checkpoint saved ({len(results)} tables total)")

    # Final save
    if newly_processed > 0:
        save_aggregate_pickle(results, args.output_path)

    print(f"\n{'=' * 80}")
    print(f"Done. {len(results)} tables in {args.output_path}")
    print(f"  Newly processed: {newly_processed}")
    print(f"  Previously completed: {len(completed)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
