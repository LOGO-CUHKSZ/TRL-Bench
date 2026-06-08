#!/usr/bin/env python3
"""
Extract table-level vectors from tables using Starmie model.

This script produces table-level embeddings where all column vectors
for a table are bundled together.

OUTPUT FORMAT
=============
The output is a pickled list of 2-tuples:

    [(table_name, column_vectors_array), ...]

Where:
    - table_name (str): The CSV filename (e.g., "ABC123.csv")
    - column_vectors_array (np.ndarray): Array of all column vectors for that table
      Shape: [num_columns, embedding_dim] or list of [embedding_dim,] vectors

Example output structure:
    [
        ("table1.csv", np.array([[0.1, 0.2, ...],   # col1 vector
                                 [0.3, 0.4, ...]])), # col2 vector
        ("table2.csv", np.array([[0.5, 0.6, ...]])), # single column
        ...
    ]

COMPARISON WITH extractColumnVectors.py
=======================================
extractColumnVectors.py outputs a different format (3-tuples with individual columns):

    [(table_name, column_name, embedding), ...]

This script bundles all columns together without preserving column names.
For join search tasks that need individual column matching, use extractColumnVectors.py.

The underlying embeddings are IDENTICAL - only the storage format differs.

USAGE
=====
    python extractVectors.py \
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
import sys
import argparse
from tqdm import tqdm


def extractVectors(dfs, model_path):
    """Get model inference on tables.

    Args:
        dfs (list of DataFrames): tables to get model inference on
        model_path (str): path to the trained model checkpoint

    Returns:
        list of lists: For each table, a list of column vectors.
                       Each column vector is a numpy array of shape [dim,].
    """
    ckpt = torch.load(model_path, map_location=torch.device('cuda'), weights_only=False)
    # load_checkpoint from sdd/pretain
    model, trainset = load_checkpoint(ckpt)
    model.eval()  # Set to evaluation mode for deterministic inference
    return inference_on_tables(dfs, model, trainset, batch_size=1024)

def get_df(dataFolder, max_rows=1000):
    """Get the DataFrames of each table in a folder.

    Args:
        dataFolder: filepath to the folder with all tables
        max_rows: maximum number of rows to read per table (default 1000)

    Returns:
        dataDfs (dict): key is the filename, value is the dataframe of that table
    """
    dataFiles = sorted(
        entry.path
        for entry in os.scandir(dataFolder)
        if entry.is_file() and entry.name.endswith(".csv")
    )
    dataDFs = {}
    for file in tqdm(dataFiles, desc="Loading tables"):
        try:
            # Don't use lineterminator='\n' - it causes \r to be preserved
            # in column names when CSV has Windows line endings (\r\n)
            # Use Python engine for more robust parsing (handles malformed files)
            # Use nrows to limit rows at read time (much faster for large files)
            df = pd.read_csv(file, encoding='utf-8', on_bad_lines='skip',
                           engine='python', encoding_errors='ignore', nrows=max_rows)
            # Strip whitespace and carriage returns from column names
            df.columns = df.columns.str.strip().str.rstrip('\r\n')
            filename = file.split("/")[-1]
            dataDFs[filename] = df
        except Exception as e:
            print(f"Warning: Failed to read {file}: {e}")
            continue
    return dataDFs


if __name__ == '__main__':
    ''' Get the model features by calling model inference from sdd/pretrain
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input tables directory")
    parser.add_argument("--output_path", type=str, required=True, help="Full path to save the output vectors (including filename)")
    parser.add_argument("--max_rows", type=int, default=1000, help="Maximum rows to read per table (default: 1000)")

    hp = parser.parse_args()

    # Create output directory if it doesn't exist
    import os
    output_dir = os.path.dirname(hp.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Process the input directory
    print(f"//==== Processing: {hp.input_dir}")
    print(f"Max rows per table: {hp.max_rows}")
    dfs = get_df(hp.input_dir, max_rows=hp.max_rows)
    print("num dfs:", len(dfs))

    dataEmbeds = []

    # Extract model vectors, and measure model inference time
    start_time = time.time()
    cl_features = extractVectors(list(dfs.values()), hp.model_path)
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time} seconds")

    for i, file in enumerate(dfs):
        # get features for this file / dataset
        cl_features_file = np.array(cl_features[i])
        dataEmbeds.append((file, cl_features_file))

    # Save to output path
    pickle.dump(dataEmbeds, open(hp.output_path, "wb"))
    print(f"Saved vectors to: {hp.output_path}")

    print("--- Total Inference Time: %s seconds ---" % (inference_time))
