from tabsketchfm.data_processing.data_prep import prep_data
import os
import argparse
from multiprocessing import Pool

def load_processed_manifest(manifest_path):
    """Load set of already-processed files from manifest."""
    if not os.path.exists(manifest_path):
        return set()

    with open(manifest_path, 'r') as f:
        return set(line.strip() for line in f if line.strip())

def append_to_manifest(manifest_path, file_path):
    """Append a processed file to the manifest."""
    with open(manifest_path, 'a') as f:
        f.write(file_path + '\n')
        f.flush()

def process_single_file_worker_opendata(args):
    """
    Worker function for parallel processing (OpenData with metadata).
    Returns (csv_path, success, error_msg).
    """
    csv_path, metadata_file, output_path, manifest_path = args
    try:
        prep_data(csv_path, output_path, metadata_file, None)
        append_to_manifest(manifest_path, csv_path)
        return (csv_path, True, None)
    except Exception as e:
        return (csv_path, False, str(e)[:200])

def get_unprocessed_files_opendata(input_dir, metadata_dir, manifest_path, resume=True):
    """Generator that yields unprocessed CSV files with their metadata."""
    processed_set = load_processed_manifest(manifest_path) if resume else set()

    for folder, subs, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith(".csv"):
                csv_path = f"{folder}/{filename}"
                metadata_file = f"{metadata_dir}/{os.path.basename(folder)}/{filename}.meta"

                # Skip if already processed or metadata missing
                if (not resume or csv_path not in processed_set) and os.path.exists(metadata_file):
                    yield (csv_path, metadata_file)

def preprocess_pretrain_data_parallel(input_dir, metadata_dir, output_path, resume=True, workers=8, recycle_after=50):
    """
    Memory-safe parallel preprocessing for OpenData with metadata.

    Args:
        input_dir: Directory containing CSV files
        metadata_dir: Directory containing metadata files
        output_path: Directory to save preprocessed files
        resume: If True, skip already-processed files
        workers: Number of parallel workers
        recycle_after: Recycle worker process after N files (prevents memory leaks)
    """
    manifest_path = os.path.join(output_path, '.processed_files.txt')
    processed_set = load_processed_manifest(manifest_path) if resume else set()

    if resume and processed_set:
        print(f"Resume mode: Found {len(processed_set)} already-processed files")

    # Count total unprocessed files
    print("Counting unprocessed files...")
    unprocessed_files = list(get_unprocessed_files_opendata(input_dir, metadata_dir, manifest_path, resume))
    total_files = len(unprocessed_files)

    print(f"Found {total_files} files to process")
    print(f"Using {workers} workers, recycling every {recycle_after} files")
    print("=" * 60)

    if total_files == 0:
        print("No files to process. All done!")
        return

    processed_count = 0
    failed_count = 0

    # Create args for workers
    args_list = [(csv_path, metadata_file, output_path, manifest_path)
                 for csv_path, metadata_file in unprocessed_files]

    # Use Pool with maxtasksperchild to recycle workers periodically
    with Pool(processes=workers, maxtasksperchild=recycle_after) as pool:
        for i, (csv_path, success, error_msg) in enumerate(pool.imap_unordered(process_single_file_worker_opendata, args_list), 1):
            if success:
                print(f"[{i}/{total_files}] ✅ {os.path.basename(csv_path)}")
                processed_count += 1
            else:
                print(f"[{i}/{total_files}] ❌ {os.path.basename(csv_path)}: {error_msg}")
                failed_count += 1

            if i % 100 == 0:
                print(f"\n{'='*60}")
                print(f"Progress: {i}/{total_files} files ({100*i//total_files}%)")
                print(f"Processed: {processed_count}, Failed: {failed_count}")
                print(f"{'='*60}\n")

    print(f"\n{'='*60}")
    print(f"PARALLEL PREPROCESSING COMPLETE!")
    print(f"Total files: {total_files}")
    print(f"Successfully processed: {processed_count}")
    print(f"Failed: {failed_count}")
    print(f"Output directory: {output_path}")
    print(f"{'='*60}")

def preprocess_pretrain_data(input_dir, metadata_dir, output_path, resume=True):
    total_files = 0
    processed_files = 0
    skipped_files = 0
    already_processed = 0

    # Setup manifest for tracking processed files
    manifest_path = os.path.join(output_path, '.processed_files.txt')
    processed_set = load_processed_manifest(manifest_path) if resume else set()

    if resume and processed_set:
        print(f"Resume mode: Found {len(processed_set)} already-processed files")

    # Count total CSV files first
    for folder, subs, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith(".csv"):
                total_files += 1

    print(f"Found {total_files} CSV files to process")
    print("=" * 60)

    for folder, subs, files in os.walk(input_dir):
        for index, filename in enumerate(files):
            lower_filename = filename.lower()
            if not lower_filename.endswith(".csv"):
                continue

            csv_path = f"{folder}/{filename}"

            # Check if already processed
            if resume and csv_path in processed_set:
                already_processed += 1
                if already_processed % 100 == 0:
                    print(f"[{already_processed + processed_files + skipped_files}/{total_files}] Skipped (already processed): {csv_path}")
                continue

            print(f"\nFolder: {folder}, Parent: {os.path.basename(folder)}")
            metadata_file = f"{metadata_dir}/{os.path.basename(folder)}/{filename}.meta"

            if not os.path.exists(metadata_file):
                print(f"⚠️  Metadata not found: {metadata_file}")
                skipped_files += 1
                continue

            print(f"Processing [{already_processed + processed_files + skipped_files + 1}/{total_files}]: {filename}")
            try:
                prep_data(csv_path, output_path, metadata_file, None)
                processed_files += 1

                # Add to manifest after successful processing
                if resume:
                    append_to_manifest(manifest_path, csv_path)
                    processed_set.add(csv_path)
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
                skipped_files += 1

            # Print progress every 100 files
            if (already_processed + processed_files + skipped_files) % 100 == 0:
                print(f"\n{'='*60}")
                print(f"Progress: {already_processed + processed_files + skipped_files}/{total_files} files")
                print(f"Already processed (skipped): {already_processed}")
                print(f"Newly processed: {processed_files}, Failed: {skipped_files}")
                print(f"{'='*60}\n")

            # NOTE: Original script had "if index > 20: break" here
            # This has been REMOVED to process all files!

    print(f"\n{'='*60}")
    print(f"PREPROCESSING COMPLETE!")
    print(f"Total CSV files found: {total_files}")
    print(f"Already processed (skipped): {already_processed}")
    print(f"Newly processed: {processed_files}")
    print(f"Failed: {skipped_files}")
    print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='path to the directory with tables')
    parser.add_argument('--metadata_dir', required=True, help='path to the directory with metadata')
    parser.add_argument('--output_dir', required=True, help='path to processed data dir')
    parser.add_argument('--no-resume', action='store_true', help='Disable resume mode (reprocess all files)')
    parser.add_argument('--workers', type=int, default=0, help='Number of parallel workers (0=sequential, default: 0)')
    parser.add_argument('--recycle_after', type=int, default=50, help='Recycle worker after N files (default: 50, only for parallel mode)')
    args = parser.parse_args()

    # Choose mode based on workers argument
    if args.workers > 0:
        print(f"Running in PARALLEL mode ({args.workers} workers)")
        preprocess_pretrain_data_parallel(args.input_dir, args.metadata_dir, args.output_dir,
                                         resume=not args.no_resume, workers=args.workers,
                                         recycle_after=args.recycle_after)
    else:
        print("Running in SEQUENTIAL mode")
        preprocess_pretrain_data(args.input_dir, args.metadata_dir, args.output_dir, resume=not args.no_resume)
