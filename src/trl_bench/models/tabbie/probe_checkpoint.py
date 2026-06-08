"""
TABBIE Checkpoint Probe

Extracts weights.th from mix.tar.gz, prints state dict structure,
validates the expected 24 transformer blocks, and saves weights.pt
for downstream use.

Usage:
    python models/tabbie/probe_checkpoint.py --archive checkpoints/tabbie/mix.tar.gz
"""

import argparse
import os
import sys
import tarfile
import tempfile

import torch


def probe_checkpoint(archive_path, output_path=None, verbose=False):
    """Extract and probe the TABBIE checkpoint archive.

    Args:
        archive_path: Path to mix.tar.gz
        output_path: Where to save weights.pt (default: same dir as archive)
        verbose: If True, print every parameter name/shape/dtype

    Returns:
        True if validation passes, False otherwise.
    """
    if not os.path.exists(archive_path):
        print(f"ERROR: Archive not found: {archive_path}")
        return False

    # Extract weight file from archive (weights.th or best.th)
    print(f"Extracting weights from {archive_path}...")
    with tarfile.open(archive_path, "r:gz") as tar:
        weights_member = None
        for candidate in ("weights.th", "best.th"):
            for member in tar.getmembers():
                if member.name.endswith(candidate):
                    weights_member = member
                    break
            if weights_member is not None:
                break

        if weights_member is None:
            print("ERROR: No weight file (weights.th or best.th) found in archive")
            print("Archive contents:")
            for m in tar.getmembers():
                print(f"  {m.name}")
            return False

        print(f"Found: {weights_member.name} ({weights_member.size / 1e6:.1f} MB)")

        # Extract to temp file, then load
        with tempfile.TemporaryDirectory() as tmpdir:
            tar.extract(weights_member, tmpdir)
            weights_path = os.path.join(tmpdir, weights_member.name)
            state_dict = torch.load(weights_path, map_location="cpu")

    # Analyze parameter names, shapes, dtypes
    print(f"\nState dict: {len(state_dict)} parameters")

    row_transformers = set()
    col_transformers = set()
    has_row_pos = False
    has_col_pos = False
    has_module_prefix = False

    for name, param in sorted(state_dict.items()):
        if verbose:
            print(f"  {name:80s}  {str(param.shape):20s}  {param.dtype}")

        # Check for _module. prefix (AllenNLP lazy module wrapping)
        if "_module." in name:
            has_module_prefix = True

        # Identify transformer blocks
        for i in range(1, 13):
            if f"transformer_row{i}" in name:
                row_transformers.add(i)
            if f"transformer_col{i}" in name:
                col_transformers.add(i)

        # Check positional embeddings
        if "row_pos_embedding" in name:
            has_row_pos = True
        if "col_pos_embedding" in name:
            has_col_pos = True

    # Validation
    print(f"\n{'='*80}")
    print("Validation")
    print(f"{'='*80}")

    expected_rows = set(range(1, 13))
    expected_cols = set(range(1, 13))

    print(f"Row transformers found: {sorted(row_transformers)} (expected {sorted(expected_rows)})")
    print(f"Col transformers found: {sorted(col_transformers)} (expected {sorted(expected_cols)})")
    print(f"Row positional embedding: {has_row_pos}")
    print(f"Col positional embedding: {has_col_pos}")
    print(f"Has _module. prefix: {has_module_prefix}")

    ok = True
    if row_transformers != expected_rows:
        print(f"FAIL: Missing row transformers: {expected_rows - row_transformers}")
        ok = False
    if col_transformers != expected_cols:
        print(f"FAIL: Missing col transformers: {expected_cols - col_transformers}")
        ok = False
    if not has_row_pos:
        print("FAIL: row_pos_embedding not found")
        ok = False
    if not has_col_pos:
        print("FAIL: col_pos_embedding not found")
        ok = False

    if ok:
        print("\nPASS: All 24 transformer blocks and positional embeddings found")
    else:
        print("\nFAIL: Checkpoint structure does not match expectations")
        return False

    # Save weights.pt (raw state dict, no key remapping)
    if output_path is None:
        output_path = os.path.join(os.path.dirname(archive_path), "weights.pt")

    print(f"\nSaving state dict to {output_path}...")
    torch.save(state_dict, output_path)
    print(f"Saved: {os.path.getsize(output_path) / 1e6:.1f} MB")

    return True


def main():
    parser = argparse.ArgumentParser(description="Probe TABBIE checkpoint archive")
    parser.add_argument(
        "--archive",
        type=str,
        default="checkpoints/tabbie/mix.tar.gz",
        help="Path to mix.tar.gz archive",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for weights.pt (default: same dir as archive)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print every parameter name, shape, and dtype",
    )
    args = parser.parse_args()

    ok = probe_checkpoint(args.archive, args.output, verbose=args.verbose)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
