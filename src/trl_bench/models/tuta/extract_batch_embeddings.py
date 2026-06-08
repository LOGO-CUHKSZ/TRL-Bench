#!/usr/bin/env python3
"""
Batch extract TUTA table embeddings from a directory of CSV files.

Supports multi-GPU parallel processing for faster extraction.

Usage:
    # Single GPU
    python models/tuta/extract_batch_embeddings.py \
        --model_path checkpoints/tuta/tuta.bin \
        --input_dir datasets/wiki_union/tables \
        --output_path embeddings/union_search/tuta/wiki_union_embeddings.pkl

    # Multi-GPU (auto-detect all GPUs)
    python models/tuta/extract_batch_embeddings.py \
        --model_path checkpoints/tuta/tuta.bin \
        --input_dir datasets/wiki_union/tables \
        --output_path embeddings.pkl \
        --use_multi_gpu

    # Multi-GPU (specify GPUs)
    python models/tuta/extract_batch_embeddings.py \
        --model_path checkpoints/tuta/tuta.bin \
        --input_dir datasets/wiki_union/tables \
        --output_path embeddings.pkl \
        --gpu_ids 0 1 2 3
"""

import os
import sys
import pickle
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# Add tuta directory to path
sys.path.insert(0, os.path.dirname(__file__))

from csv_to_embeddings import TUTAEmbedder


def process_files_on_gpu(gpu_id, csv_files, model_path, model_type, progress_queue):
    """
    Worker function to process a batch of CSV files on a specific GPU.

    Args:
        gpu_id: GPU device ID to use
        csv_files: List of Path objects for CSV files to process
        model_path: Path to TUTA checkpoint
        model_type: TUTA model variant
        progress_queue: Queue for progress updates

    Returns:
        List of (filename, embedding) tuples
    """
    try:
        # Initialize embedder on this GPU
        embedder = TUTAEmbedder(
            model_path=model_path,
            target=model_type,
            device_id=gpu_id
        )

        results = []
        failed = []

        for csv_file in csv_files:
            try:
                # Extract table-level embedding
                table_emb = embedder.csv_to_embeddings(
                    csv_path=str(csv_file),
                    output_format='numpy',
                    aggregate='cls'
                )

                # Store as (filename, embedding) tuple
                # embedding shape: (1, 768) -> squeeze to (768,)
                results.append((csv_file.name, table_emb.squeeze()))

                # Update progress
                if progress_queue:
                    progress_queue.put(1)

            except Exception as e:
                failed.append((csv_file.name, str(e)))
                if progress_queue:
                    progress_queue.put(1)
                continue

        return {
            'results': results,
            'failed': failed,
            'gpu_id': gpu_id
        }

    except Exception as e:
        return {
            'results': [],
            'failed': [(f'GPU {gpu_id} initialization', str(e))],
            'gpu_id': gpu_id
        }


def extract_batch_embeddings_multi_gpu(model_path, input_dir, output_path, model_type='tuta',
                                       gpu_ids=None, limit=None):
    """
    Extract TUTA embeddings using multiple GPUs in parallel.

    Args:
        model_path: Path to TUTA checkpoint (.bin file)
        input_dir: Directory containing CSV files
        output_path: Path to save embeddings pickle file
        model_type: TUTA model variant ('tuta', 'tuta_explicit', 'base')
        gpu_ids: List of GPU IDs to use (None=auto-detect all)
        limit: Maximum number of tables to process (None=all)
    """
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    # This must be done before any multiprocessing calls
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, that's fine
        pass

    # Detect available GPUs
    if gpu_ids is None:
        if torch.cuda.is_available():
            gpu_ids = list(range(torch.cuda.device_count()))
        else:
            print("Warning: No GPUs detected. Falling back to single CPU mode.")
            return extract_batch_embeddings(
                model_path, input_dir, output_path, model_type, device_id=-1, limit=limit
            )

    if len(gpu_ids) == 0:
        print("Error: No GPUs specified")
        return

    print("="*80)
    print("TUTA BATCH EMBEDDING EXTRACTION (MULTI-GPU)")
    print("="*80)
    print(f"Model:      {model_path}")
    print(f"Input dir:  {input_dir}")
    print(f"Output:     {output_path}")
    print(f"GPUs:       {gpu_ids} ({len(gpu_ids)} devices)")
    if limit:
        print(f"Limit:      {limit} tables")
    print("="*80)
    print()

    # Find all CSV files
    input_path = Path(input_dir)
    csv_files = sorted(
        entry.path
        for entry in os.scandir(input_path)
        if entry.is_file() and entry.name.endswith(".csv")
    )

    if limit:
        csv_files = csv_files[:limit]

    total_files = len(csv_files)
    print(f"Found {total_files} CSV files")
    print()

    # Split files across GPUs
    files_per_gpu = total_files // len(gpu_ids)
    remainder = total_files % len(gpu_ids)

    file_splits = []
    start_idx = 0

    for i, gpu_id in enumerate(gpu_ids):
        # Distribute remainder files to first few GPUs
        extra = 1 if i < remainder else 0
        end_idx = start_idx + files_per_gpu + extra
        file_splits.append((gpu_id, csv_files[start_idx:end_idx]))
        start_idx = end_idx

        print(f"GPU {gpu_id}: {len(file_splits[-1][1])} files")

    print()
    print(f"Starting parallel extraction on {len(gpu_ids)} GPUs...")
    print()

    # Create progress queue and bar
    manager = mp.Manager()
    progress_queue = manager.Queue()

    # Start progress bar updater in main process
    pbar = tqdm(total=total_files, desc="Extracting embeddings")

    # Create worker function with fixed arguments
    worker_fn = partial(
        process_files_on_gpu,
        model_path=model_path,
        model_type=model_type,
        progress_queue=progress_queue
    )

    # Launch parallel workers
    with mp.Pool(processes=len(gpu_ids)) as pool:
        # Start async workers
        async_results = []
        for gpu_id, files in file_splits:
            async_result = pool.apply_async(worker_fn, (gpu_id, files))
            async_results.append(async_result)

        # Update progress bar while workers run
        completed = 0
        while completed < total_files:
            try:
                progress_queue.get(timeout=0.1)
                completed += 1
                pbar.update(1)
            except:
                # Check if all workers are done
                if all(r.ready() for r in async_results):
                    break

        # Get results
        all_results = []
        for async_result in async_results:
            all_results.append(async_result.get())

    pbar.close()
    print()

    # Combine results from all GPUs
    embeddings_data = []
    failed_files = []

    for result_dict in all_results:
        embeddings_data.extend(result_dict['results'])
        failed_files.extend(result_dict['failed'])

    # Sort by filename for consistency
    embeddings_data.sort(key=lambda x: x[0])

    print(f"Successfully processed: {len(embeddings_data)} tables")
    if failed_files:
        print(f"Failed: {len(failed_files)} tables")
        print("\nFailed files:")
        for fname, error in failed_files[:10]:
            print(f"  - {fname}: {error}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")
    print()

    # Save embeddings
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(embeddings_data, f)

    print(f"Embeddings saved to: {output_path}")
    print(f"Format: List of (filename, embedding_array) tuples")
    print(f"Total size: {len(embeddings_data)} tables")
    print()
    print("="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)

    return embeddings_data


def extract_batch_embeddings(model_path, input_dir, output_path, model_type='tuta',
                             device_id=None, limit=None):
    """
    Extract TUTA table embeddings for all CSV files in a directory (single GPU/CPU).

    Args:
        model_path: Path to TUTA checkpoint (.bin file)
        input_dir: Directory containing CSV files
        output_path: Path to save embeddings pickle file
        model_type: TUTA model variant ('tuta', 'tuta_explicit', 'base')
        device_id: GPU device ID (None=auto, -1=CPU)
        limit: Maximum number of tables to process (None=all)
    """
    print("="*80)
    print("TUTA BATCH EMBEDDING EXTRACTION")
    print("="*80)
    print(f"Model:      {model_path}")
    print(f"Input dir:  {input_dir}")
    print(f"Output:     {output_path}")
    print(f"Device:     {'GPU' if device_id != -1 else 'CPU'}")
    if limit:
        print(f"Limit:      {limit} tables")
    print("="*80)
    print()

    # Initialize embedder
    print("Initializing TUTA model...")
    embedder = TUTAEmbedder(
        model_path=model_path,
        target=model_type,
        device_id=device_id
    )
    print()

    # Find all CSV files
    input_path = Path(input_dir)
    csv_files = sorted(
        entry.path
        for entry in os.scandir(input_path)
        if entry.is_file() and entry.name.endswith(".csv")
    )

    if limit:
        csv_files = csv_files[:limit]

    print(f"Found {len(csv_files)} CSV files")
    print()

    # Extract embeddings
    embeddings_data = []
    failed_files = []

    for csv_file in tqdm(csv_files, desc="Extracting embeddings"):
        try:
            # Extract table-level embedding
            table_emb = embedder.csv_to_embeddings(
                csv_path=str(csv_file),
                output_format='numpy',
                aggregate='cls'
            )

            # Store as (filename, embedding) tuple
            # embedding shape: (1, 768) -> squeeze to (768,)
            embeddings_data.append((csv_file.name, table_emb.squeeze()))

        except Exception as e:
            print(f"\nWarning: Failed to process {csv_file.name}: {e}")
            failed_files.append((csv_file.name, str(e)))
            continue

    print()
    print(f"Successfully processed: {len(embeddings_data)} tables")
    if failed_files:
        print(f"Failed: {len(failed_files)} tables")
        print("\nFailed files:")
        for fname, error in failed_files[:10]:
            print(f"  - {fname}: {error}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")
    print()

    # Save embeddings
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(embeddings_data, f)

    print(f"Embeddings saved to: {output_path}")
    print(f"Format: List of (filename, embedding_array) tuples")
    print(f"Total size: {len(embeddings_data)} tables")
    print()
    print("="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)

    return embeddings_data


def main():
    parser = argparse.ArgumentParser(
        description='Batch extract TUTA table embeddings from CSV files'
    )
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to TUTA checkpoint (.bin file)')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing CSV files')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Output path for embeddings pickle file')
    parser.add_argument('--model_type', type=str, default='tuta',
                       choices=['tuta', 'tuta_explicit', 'base'],
                       help='TUTA model variant')
    parser.add_argument('--device_id', type=int, default=None,
                       help='GPU device ID for single-GPU mode (None=auto-detect, -1=CPU only)')
    parser.add_argument('--use_multi_gpu', action='store_true',
                       help='Enable multi-GPU parallel processing (auto-detect all GPUs)')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=None,
                       help='Specific GPU IDs to use for multi-GPU mode (e.g., 0 1 2 3)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Maximum number of tables to process')

    args = parser.parse_args()

    # Determine mode
    if args.use_multi_gpu or args.gpu_ids is not None:
        # Multi-GPU mode
        gpu_ids = args.gpu_ids  # None means auto-detect
        extract_batch_embeddings_multi_gpu(
            model_path=args.model_path,
            input_dir=args.input_dir,
            output_path=args.output_path,
            model_type=args.model_type,
            gpu_ids=gpu_ids,
            limit=args.limit
        )
    else:
        # Single GPU/CPU mode
        extract_batch_embeddings(
            model_path=args.model_path,
            input_dir=args.input_dir,
            output_path=args.output_path,
            model_type=args.model_type,
            device_id=args.device_id,
            limit=args.limit
        )


if __name__ == '__main__':
    main()
