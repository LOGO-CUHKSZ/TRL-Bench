#!/usr/bin/env python3
"""
Generate embeddings for TabFact dataset using various table embedding models.

This script is model-agnostic and supports:
- TAPAS: Joint table+statement encoding (recommended for TabFact)
- TaBERT: Joint table+context encoding
- Doduo + BERT: Separate encoders for table and statement

Usage:
    # TAPAS embeddings
    python generate_embeddings.py \
        --model tapas \
        --data_dir datasets/tabfact \
        --output_file embeddings/tabfact/tapas_base.pkl \
        --split train \
        --device cuda

    # Doduo + BERT embeddings
    python generate_embeddings.py \
        --model doduo \
        --data_dir datasets/tabfact \
        --output_file embeddings/tabfact/doduo_bert.pkl \
        --split train \
        --device cuda
"""

import os
import sys
import pickle
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from downstream_tasks.table_fact_verification.utils.data_utils import (
    load_tabfact_examples,
    load_table,
)
from trl_bench.utils.unified_embedding_format import get_table_level_embedding


def generate_tapas_embeddings(
    data_dir: str,
    output_file: str,
    split: str = 'train',
    model_name: str = 'google/tapas-base',
    device: str = 'cuda',
    batch_size: int = 8,
    max_examples: Optional[int] = None,
) -> Dict:
    """
    Generate embeddings using TAPAS.

    TAPAS jointly encodes table+statement, producing a CLS embedding
    that captures the interaction between them.
    """
    from models.tapas import TAPASEmbedder

    print(f"Loading TAPAS model: {model_name}")
    embedder = TAPASEmbedder(model_name=model_name, device=device)

    examples = load_tabfact_examples(data_dir, split)
    if max_examples:
        examples = examples[:max_examples]

    print(f"Generating embeddings for {len(examples)} examples...")

    embeddings = {}
    errors = []

    for ex in tqdm(examples, desc=f"Generating {split} embeddings"):
        try:
            # Load table
            table_path = Path(data_dir) / "tables" / f"{ex['table_id']}.csv"

            # Generate embedding with statement as question
            result = embedder.encode_csv(
                str(table_path),
                question=ex['statement'],
            )

            # Extract embeddings using helper for v1.0/v2.0 compatibility
            cls_emb = get_table_level_embedding(result, variant='cls_embedding')
            table_emb = get_table_level_embedding(result, variant='column_mean')

            embeddings[ex['id']] = {
                'cls_embedding': cls_emb,
                'table_embedding': table_emb,
                'table_id': ex['table_id'],
                'statement': ex['statement'],
            }

        except Exception as e:
            errors.append({'id': ex['id'], 'error': str(e)})

    print(f"Generated {len(embeddings)} embeddings, {len(errors)} errors")

    # Save embeddings
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)

    print(f"Saved embeddings to {output_file}")

    # Save error log if any
    if errors:
        error_file = output_path.with_suffix('.errors.json')
        import json
        with open(error_file, 'w') as f:
            json.dump(errors, f, indent=2)
        print(f"Saved {len(errors)} errors to {error_file}")

    return {
        'num_embeddings': len(embeddings),
        'num_errors': len(errors),
        'embedding_dim': embeddings[next(iter(embeddings))]['cls_embedding'].shape[0] if embeddings else 0,
    }


def generate_tabert_embeddings(
    data_dir: str,
    output_file: str,
    split: str = 'train',
    device: str = 'cuda',
    max_examples: Optional[int] = None,
) -> Dict:
    """
    Generate embeddings using TaBERT.

    TaBERT jointly encodes table+context, using statement as context.
    """
    # TaBERT requires its own environment
    try:
        from models.tabert import TaBERTEmbedder
    except ImportError:
        raise ImportError(
            "TaBERT not available. Please activate TaBERT environment: "
            "source models/tabert/venv/bin/activate"
        )

    print("Loading TaBERT model...")
    embedder = TaBERTEmbedder(device=device)

    examples = load_tabfact_examples(data_dir, split)
    if max_examples:
        examples = examples[:max_examples]

    print(f"Generating embeddings for {len(examples)} examples...")

    embeddings = {}
    errors = []

    for ex in tqdm(examples, desc=f"Generating {split} embeddings"):
        try:
            table_path = Path(data_dir) / "tables" / f"{ex['table_id']}.csv"

            result = embedder.encode_csv(
                str(table_path),
                context=ex['statement'],
            )

            # Extract embeddings using helper for v1.0/v2.0 compatibility
            # For TaBERT, context_embedding is the statement embedding
            context_emb = result.get('context_embedding')
            table_emb = get_table_level_embedding(result, variant='column_mean')
            cls_emb = context_emb if context_emb is not None else table_emb

            embeddings[ex['id']] = {
                'cls_embedding': cls_emb,
                'table_embedding': table_emb,
                'column_embeddings': result.get('column_embeddings'),
                'table_id': ex['table_id'],
                'statement': ex['statement'],
            }

        except Exception as e:
            errors.append({'id': ex['id'], 'error': str(e)})

    print(f"Generated {len(embeddings)} embeddings, {len(errors)} errors")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)

    print(f"Saved embeddings to {output_file}")

    return {
        'num_embeddings': len(embeddings),
        'num_errors': len(errors),
    }


def generate_doduo_bert_embeddings(
    data_dir: str,
    output_file: str,
    split: str = 'train',
    device: str = 'cuda',
    max_examples: Optional[int] = None,
) -> Dict:
    """
    Generate embeddings using Doduo (for table) + BERT (for statement).

    Since Doduo doesn't have native question/statement support,
    we encode them separately and concatenate.
    """
    from transformers import AutoTokenizer, AutoModel

    # Load BERT for statement encoding
    print("Loading BERT for statement encoding...")
    bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    bert_model = AutoModel.from_pretrained('bert-base-uncased').to(device)
    bert_model.eval()

    # Load Doduo for table encoding
    print("Loading Doduo for table encoding...")
    from models.doduo import Doduo
    doduo = Doduo(device=device)

    examples = load_tabfact_examples(data_dir, split)
    if max_examples:
        examples = examples[:max_examples]

    print(f"Generating embeddings for {len(examples)} examples...")

    embeddings = {}
    errors = []

    for ex in tqdm(examples, desc=f"Generating {split} embeddings"):
        try:
            # Encode statement with BERT
            inputs = bert_tokenizer(
                ex['statement'],
                return_tensors='pt',
                truncation=True,
                max_length=128,
                padding=True,
            ).to(device)

            with torch.no_grad():
                bert_output = bert_model(**inputs)
                statement_emb = bert_output.last_hidden_state[:, 0, :].cpu().numpy()[0]

            # Encode table with Doduo
            table_path = Path(data_dir) / "tables" / f"{ex['table_id']}.csv"
            table_result = doduo.encode_csv(str(table_path))

            # Use helper to get table embedding (handles v1.0/v2.0 formats)
            table_emb = get_table_level_embedding(table_result, variant='column_mean')
            if table_emb is None:
                # Fallback: compute from column embeddings directly
                if table_result.get('column_embeddings'):
                    col_embs = list(table_result['column_embeddings'].values())
                    table_emb = np.mean(col_embs, axis=0).astype(np.float32)
                else:
                    table_emb = np.zeros(768, dtype=np.float32)

            # Combine: concatenate or use element-wise operations
            # Here we concatenate statement and table embeddings
            cls_embedding = np.concatenate([statement_emb, table_emb])

            embeddings[ex['id']] = {
                'cls_embedding': cls_embedding,
                'statement_embedding': statement_emb,
                'table_embedding': table_emb,
                'table_id': ex['table_id'],
                'statement': ex['statement'],
            }

        except Exception as e:
            errors.append({'id': ex['id'], 'error': str(e)})

    print(f"Generated {len(embeddings)} embeddings, {len(errors)} errors")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)

    print(f"Saved embeddings to {output_file}")

    return {
        'num_embeddings': len(embeddings),
        'num_errors': len(errors),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for TabFact using various models"
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['tapas', 'tabert', 'doduo'],
        help='Embedding model to use'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='datasets/tabfact',
        help='Directory containing TabFact dataset'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='Output file for embeddings (pickle)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'validation', 'test'],
        help='Dataset split to process'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default=None,
        help='Specific model name/checkpoint (for TAPAS: google/tapas-base)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu)'
    )
    parser.add_argument(
        '--max_examples',
        type=int,
        default=None,
        help='Maximum examples to process (for debugging)'
    )

    args = parser.parse_args()

    if args.model == 'tapas':
        model_name = args.model_name or 'google/tapas-base'
        stats = generate_tapas_embeddings(
            data_dir=args.data_dir,
            output_file=args.output_file,
            split=args.split,
            model_name=model_name,
            device=args.device,
            max_examples=args.max_examples,
        )
    elif args.model == 'tabert':
        stats = generate_tabert_embeddings(
            data_dir=args.data_dir,
            output_file=args.output_file,
            split=args.split,
            device=args.device,
            max_examples=args.max_examples,
        )
    elif args.model == 'doduo':
        stats = generate_doduo_bert_embeddings(
            data_dir=args.data_dir,
            output_file=args.output_file,
            split=args.split,
            device=args.device,
            max_examples=args.max_examples,
        )
    else:
        raise ValueError(f"Model {args.model} not yet implemented")

    print(f"\nDone! Stats: {stats}")


if __name__ == '__main__':
    main()
