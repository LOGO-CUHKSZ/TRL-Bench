"""
Convert wiki-join-search JSONL labels to pickle format for embedding_search.py.

Usage:
    python scripts/utils/convert_wiki_join_labels.py \
        --input wiki-join-search/labels/join_search_jaccard_gt.jsonl \
        --output wiki_join_ground_truth.pkl \
        --min_score 0.0
"""

import json
import pickle
import argparse


def convert_wiki_join_labels(jsonl_file, min_score=0.0, keep_csv_extension=False):
    """
    Convert wiki-join-search JSONL to pickle format.

    Input format (JSONL):
        {
            "source": {"filename": "FILE1", "col": "0"},
            "joinable_list": [
                {"filename": "FILE2", "col": "1", "score": 0.85},
                {"filename": "FILE3", "col": "0", "score": 0.42},
                ...
            ]
        }

    Output format (pickle):
        {
            "FILE1:0": ["FILE2:1", "FILE3:0", ...],  # if keep_csv_extension=False
            "FILE1.csv:0": ["FILE2.csv:1", ...],      # if keep_csv_extension=True
            ...
        }

    Args:
        jsonl_file: Path to JSONL file
        min_score: Minimum joinability score to include (default: 0.0)
        keep_csv_extension: Whether to keep .csv in keys (default: False for union=False mode)

    Returns:
        Dict mapping query table:col to list of joinable table:col
    """
    ground_truth = {}

    with open(jsonl_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip() == '':
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Skipping line {line_num} due to JSON error: {e}")
                continue

            # Get source table and column
            source_file = data['source']['filename']
            source_col = data['source']['col']
            if keep_csv_extension:
                source_key = f"{source_file}.csv:{source_col}"
            else:
                source_key = f"{source_file}:{source_col}"

            # Get joinable candidates (filter by score)
            joinable = []
            for item in data['joinable_list']:
                if item['score'] >= min_score:
                    target_file = item['filename']
                    target_col = item['col']
                    if keep_csv_extension:
                        target_key = f"{target_file}.csv:{target_col}"
                    else:
                        target_key = f"{target_file}:{target_col}"
                    joinable.append(target_key)

            ground_truth[source_key] = joinable

    return ground_truth


def main():
    parser = argparse.ArgumentParser(description="Convert wiki-join-search JSONL to pickle")
    parser.add_argument('--input', type=str, required=True,
                        help='Input JSONL file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output pickle file')
    parser.add_argument('--min_score', type=float, default=0.0,
                        help='Minimum joinability score (default: 0.0)')
    parser.add_argument('--keep_csv_extension', action='store_true',
                        help='Keep .csv in table names (for union=True mode)')

    args = parser.parse_args()

    print(f"📂 Loading labels from: {args.input}")
    print(f"   Min score threshold: {args.min_score}")
    print(f"   Keep .csv extension: {args.keep_csv_extension}")

    ground_truth = convert_wiki_join_labels(args.input, args.min_score, args.keep_csv_extension)

    print(f"\n📊 Conversion statistics:")
    print(f"   Total queries: {len(ground_truth)}")

    # Stats about joinable candidates
    joinable_counts = [len(v) for v in ground_truth.values()]
    if joinable_counts:
        print(f"   Avg candidates per query: {sum(joinable_counts) / len(joinable_counts):.1f}")
        print(f"   Min candidates: {min(joinable_counts)}")
        print(f"   Max candidates: {max(joinable_counts)}")

    print(f"\n💾 Saving to: {args.output}")
    with open(args.output, 'wb') as f:
        pickle.dump(ground_truth, f)

    print("✅ Conversion complete!")

    # Show sample
    print("\n" + "="*60)
    print("SAMPLE GROUND TRUTH")
    print("="*60)
    sample_key = list(ground_truth.keys())[0]
    sample_vals = ground_truth[sample_key][:5]  # First 5
    print(f"Query: {sample_key}")
    print(f"Joinable tables (showing first 5 of {len(ground_truth[sample_key])}):")
    for val in sample_vals:
        print(f"  - {val}")
    print("="*60)


if __name__ == '__main__':
    main()
