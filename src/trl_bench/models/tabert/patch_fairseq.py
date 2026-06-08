#!/usr/bin/env python
"""
Patch fairseq for NumPy 2.x compatibility.

fairseq 0.10.2 uses deprecated numpy types (np.float, np.int) that were
removed in NumPy 1.24+. This script patches the installed fairseq package.

Usage:
    python patch_fairseq.py

Run this after installing fairseq in your environment.
"""

import os
import sys


def get_fairseq_path():
    """Get the path to the installed fairseq package."""
    try:
        import fairseq
        return os.path.dirname(fairseq.__file__)
    except ImportError:
        return None
    except AttributeError:
        # fairseq import failed due to np.float issue - find it manually
        import site
        for site_dir in site.getsitepackages() + [site.getusersitepackages()]:
            fairseq_path = os.path.join(site_dir, 'fairseq')
            if os.path.isdir(fairseq_path):
                return fairseq_path
        return None


def patch_file(filepath, replacements):
    """Apply replacements to a file."""
    if not os.path.exists(filepath):
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    original = content
    for old, new in replacements:
        content = content.replace(old, new)

    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False


def main():
    fairseq_path = get_fairseq_path()

    if fairseq_path is None:
        print("Error: fairseq not found. Please install it first:")
        print("  pip install fairseq==0.10.2")
        sys.exit(1)

    print(f"Patching fairseq at: {fairseq_path}")

    patches = [
        # indexed_dataset.py: np.float -> np.float64
        (
            os.path.join(fairseq_path, 'data', 'indexed_dataset.py'),
            [
                ('6: np.float,', '6: np.float64,'),
                ('np.float: 4,', 'np.float64: 4,'),
            ]
        ),
        # dynamic_crf_layer.py: np.float("inf") -> float("inf")
        (
            os.path.join(fairseq_path, 'modules', 'dynamic_crf_layer.py'),
            [
                ('np.float("inf")', 'float("inf")'),
            ]
        ),
        # data_utils.py: np.int -> np.int64
        (
            os.path.join(fairseq_path, 'data', 'data_utils.py'),
            [
                ('np.int,', 'np.int64,'),
            ]
        ),
    ]

    patched_count = 0
    for filepath, replacements in patches:
        if patch_file(filepath, replacements):
            print(f"  Patched: {os.path.basename(filepath)}")
            patched_count += 1
        else:
            relpath = os.path.relpath(filepath, fairseq_path)
            if os.path.exists(filepath):
                print(f"  Skipped (already patched or no changes needed): {relpath}")
            else:
                print(f"  Warning: File not found: {relpath}")

    if patched_count > 0:
        print(f"\nSuccessfully patched {patched_count} file(s).")
    else:
        print("\nNo files needed patching (already patched).")

    # Verify the patch worked
    print("\nVerifying fairseq import...")
    try:
        # Force reimport
        if 'fairseq' in sys.modules:
            del sys.modules['fairseq']
        import fairseq
        print("Success: fairseq imports correctly!")
    except Exception as e:
        print(f"Warning: fairseq import still fails: {e}")
        print("You may need to restart Python after patching.")


if __name__ == '__main__':
    main()
