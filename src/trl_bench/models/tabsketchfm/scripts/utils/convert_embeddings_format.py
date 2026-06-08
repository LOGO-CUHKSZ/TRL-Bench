"""
Convert embeddings from extract_embeddings_unified.py format to embedding_search.py format.

Usage:
    python scripts/utils/convert_embeddings_format.py \
        --input embeddings_unified.pkl \
        --output embeddings_legacy.pkl
"""

import pickle
import argparse


def convert_embeddings_format(unified_embeddings):
    """
    Convert from unified format to legacy format.

    Unified format (extract_embeddings_unified.py):
        [
            {
                'table_embedding': [...],
                'column_embeddings': {...},  # unified format (plural)
                'cls_embedding': [...],
                'table': 'file.csv'
            },
            ...
        ]

    Also supports legacy input format with 'column_embedding' (singular).

    Legacy format (extract_embeddings.py, expected by embedding_search.py):
        {
            0: [  # layer 0
                {
                    'table embedding': [...],  # NOTE: space not underscore
                    'column_embedding': {...},
                    'cls_embedding': [...],
                    'table': 'file.csv'
                },
                ...
            ]
        }
    """
    import os

    legacy_embeddings = {0: []}

    for item in unified_embeddings:
        # Extract just the filename (remove path if present)
        table_name = item['table']
        if '/' in table_name:
            table_name = os.path.basename(table_name)

        # Handle both 'column_embeddings' (unified) and 'column_embedding' (legacy)
        col_emb_data = item.get('column_embeddings') or item.get('column_embedding', {})

        legacy_item = {
            'table embedding': item['table_embedding'],  # Fix key name
            'column_embedding': col_emb_data,  # Keep singular for legacy format
            'cls_embedding': item['cls_embedding'],
            'table': table_name  # Use just filename
        }
        legacy_embeddings[0].append(legacy_item)

    return legacy_embeddings


def main():
    parser = argparse.ArgumentParser(description="Convert embeddings to legacy format")
    parser.add_argument('--input', type=str, required=True,
                        help='Input pickle file (unified format)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output pickle file (legacy format)')

    args = parser.parse_args()

    print(f"📂 Loading embeddings from: {args.input}")
    with open(args.input, 'rb') as f:
        unified_embeddings = pickle.load(f)

    print(f"   Loaded {len(unified_embeddings)} table embeddings")

    print("\n🔄 Converting to legacy format...")
    legacy_embeddings = convert_embeddings_format(unified_embeddings)

    print(f"   Converted {len(legacy_embeddings[0])} embeddings")

    print(f"\n💾 Saving to: {args.output}")
    with open(args.output, 'wb') as f:
        pickle.dump(legacy_embeddings, f)

    print("✅ Conversion complete!")

    # Print summary
    sample = legacy_embeddings[0][0]
    print("\n" + "="*60)
    print("CONVERSION SUMMARY")
    print("="*60)
    print(f"Total tables: {len(legacy_embeddings[0])}")
    print(f"Sample table: {sample['table']}")
    print(f"Table embedding dim: {len(sample['table embedding'])}")
    col_emb = sample.get('column_embedding', {})
    print(f"Num columns: {len(col_emb)}")
    print(f"CLS embedding dim: {len(sample['cls_embedding'])}")
    print("="*60)


if __name__ == '__main__':
    main()
