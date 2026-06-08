#!/usr/bin/env python3
"""
Extract column-level vectors from tables using Starmie model.

This script produces column-level embeddings for join search tasks where
individual columns need to be matched.

OUTPUT FORMAT
=============
The output is a pickled list of 3-tuples:

    [(table_name, column_name, embedding), ...]

Where:
    - table_name (str): The CSV filename (e.g., "ABC123.csv")
    - column_name (str): The column header name (e.g., "country")
    - embedding (np.ndarray): The column embedding vector (shape: [dim,])

Example output structure:
    [
        ("table1.csv", "id", np.array([0.1, 0.2, ...])),
        ("table1.csv", "name", np.array([0.3, 0.4, ...])),
        ("table2.csv", "city", np.array([0.5, 0.6, ...])),
        ...
    ]

COMPARISON WITH extractVectors.py
=================================
extractVectors.py outputs a different format (2-tuples with bundled columns):

    [(table_name, np.array([[col1_vec], [col2_vec], ...])), ...]

This script "flattens" that structure by:
1. Expanding each table's column vectors into separate entries
2. Including the column name (from DataFrame headers) in each tuple

The underlying embeddings are IDENTICAL - only the storage format differs.

USAGE
=====
    python extractColumnVectors.py \
        --model_path path/to/model.pt \
        --input_dir path/to/tables/ \
        --output_path path/to/output.pkl
"""
from sdd.pretrain import load_checkpoint, inference_on_tables
import torch
import pandas as pd
import numpy as np
import glob
import os
import pickle
import time
import os
import argparse
from tqdm import tqdm


def extractColumnVectors(dfs, model_path):
    """Get model inference on tables, returning column-level vectors.

    Args:
        dfs (list of DataFrames): tables to get model inference on
        model_path (str): path to the trained model checkpoint

    Returns:
        list of lists: For each table, a list of column vectors.
                       Each column vector is a numpy array of shape [dim,].
    """
    ckpt = torch.load(model_path, map_location=torch.device('cuda'), weights_only=False)
    model, trainset = load_checkpoint(ckpt)
    model.eval()  # Set to evaluation mode for deterministic inference
    return inference_on_tables(dfs, model, trainset, batch_size=1024)


def get_df(dataFolder):
    """Get the DataFrames of each table in a folder.

    Args:
        dataFolder: filepath to the folder with all tables

    Returns:
        dataDfs (dict): key is the filename, value is the dataframe of that table
    """
    dataFiles = sorted(
        entry.path
        for entry in os.scandir(dataFolder)
        if entry.is_file() and entry.name.endswith(".csv")
    )
    dataDFs = {}
    for file in dataFiles:
        try:
            # Don't use lineterminator='\n' - it causes \r to be preserved
            # in column names when CSV has Windows line endings (\r\n)
            df = pd.read_csv(file, encoding='utf-8', on_bad_lines='skip')
            # Strip whitespace and carriage returns from column names
            df.columns = df.columns.str.strip().str.rstrip('\r\n')
            # Handle potential duplicate column names after stripping
            if df.columns.duplicated().any():
                # Make duplicates unique by adding suffixes
                cols = pd.Series(df.columns)
                for dup in cols[cols.duplicated()].unique():
                    cols[cols[cols == dup].index.values.tolist()] = [dup + '.' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
                df.columns = cols
            if len(df) > 1000:
                df = df.head(1000)
            filename = file.split("/")[-1]
            dataDFs[filename] = df
        except Exception as e:
            print(f"Warning: Failed to read {file}: {e}")
            continue
    return dataDFs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract column-level vectors for join search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output format: List of (table_name, column_name, embedding) tuples.
See module docstring for detailed format specification.
        """
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model checkpoint")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to the input tables directory")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Full path to save the output vectors (including filename)")

    hp = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(hp.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Process the input directory
    print(f"//==== Processing: {hp.input_dir}")
    dfs = get_df(hp.input_dir)
    print(f"Number of tables: {len(dfs)}")

    # Extract model vectors
    start_time = time.time()
    cl_features = extractColumnVectors(list(dfs.values()), hp.model_path)
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time:.2f} seconds")

    # Build column-level embeddings list: (table_name, column_name, embedding)
    columnEmbeds = []

    for i, (filename, df) in enumerate(dfs.items()):
        column_vectors = cl_features[i]
        column_names = list(df.columns)

        # Ensure we have vectors for all columns
        if len(column_vectors) != len(column_names):
            print(f"Warning: {filename} has {len(column_names)} columns but {len(column_vectors)} vectors")
            num_cols = min(len(column_vectors), len(column_names))
        else:
            num_cols = len(column_names)

        for col_idx in range(num_cols):
            col_name = column_names[col_idx]
            col_emb = np.array(column_vectors[col_idx])
            columnEmbeds.append((filename, col_name, col_emb))

    # Save to output path
    pickle.dump(columnEmbeds, open(hp.output_path, "wb"))

    # Summary
    print("=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"  Tables processed: {len(dfs)}")
    print(f"  Column embeddings: {len(columnEmbeds)}")
    print(f"  Embedding dimension: {columnEmbeds[0][2].shape[0] if columnEmbeds else 'N/A'}")
    print(f"  Output file: {hp.output_path}")
    print(f"  Inference time: {inference_time:.2f} seconds")
    print("=" * 60)
