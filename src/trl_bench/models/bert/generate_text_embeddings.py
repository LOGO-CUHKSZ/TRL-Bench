#!/usr/bin/env python
"""
Generate textual embeddings using pretrained BERT.

General-purpose script for encoding arbitrary text (questions, queries,
sentences) into BERT embeddings. Consolidates the BERT text encoding logic
previously duplicated across generate_tabert_embeddings.py and
scripts/tabsketchfm/utils/generate_wtq_embeddings.py.

Two modes:
  - cls  (default): CLS token embedding per text → shape (768,)
                     Used for table retrieval query embeddings.
  - token:          Full token-level embeddings (excluding CLS/SEP) → shape (seq_len, 768)
                     Used for semantic parsing question embeddings.

Input formats:
  - --input_json FILE --text_field FIELD  (reads a JSON list of dicts)
  - --input_text FILE                     (plain text, one sentence per line)

Output: pickle file — list of dicts, each:
    {"text_id": str, "text": str, "embedding": np.ndarray,
     "model_name": str, "mode": "cls"|"token"}

Usage:
    # CLS embeddings from JSON
    python generate_text_embeddings.py --mode cls \
        --input_json data.json --text_field question \
        --output queries.pkl

    # Token embeddings from text file
    python generate_text_embeddings.py --mode token \
        --input_text sentences.txt \
        --output token_embs.pkl

    # With batching and GPU
    python generate_text_embeddings.py --mode cls \
        --input_json data.json --text_field question \
        --batch_size 64 --device cuda --output queries.pkl
"""

import os
import sys
import json
import pickle
import argparse
from typing import List, Dict, Any

import torch
import numpy as np
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_texts(args) -> List[Dict[str, str]]:
    """
    Load texts from input source. Returns list of {"text_id": ..., "text": ...}.
    """
    texts = []
    if args.input_json:
        with open(args.input_json, 'r') as f:
            if args.input_json.endswith('.jsonl'):
                data = [json.loads(line) for line in f if line.strip()]
            else:
                data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of objects")
        for i, item in enumerate(data):
            if args.tokens_field:
                tokens = item.get(args.tokens_field)
                if tokens is None:
                    raise KeyError(f"Tokens field '{args.tokens_field}' not found in item {i}")
                if not isinstance(tokens, list):
                    raise TypeError(f"Tokens field '{args.tokens_field}' in item {i} is not a list")
                text = ' '.join(str(t) for t in tokens)
            else:
                text = item.get(args.text_field)
            if text is None:
                field = args.tokens_field or args.text_field
                raise KeyError(f"Field '{field}' not found in item {i}")
            if args.id_field:
                if args.id_field not in item:
                    raise KeyError(f"ID field '{args.id_field}' not found in item {i}")
                text_id = str(item[args.id_field])
            else:
                text_id = str(item.get('id', item.get('text_id', i)))
            texts.append({"text_id": text_id, "text": str(text)})
    elif args.input_text:
        with open(args.input_text, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    texts.append({"text_id": str(i), "text": line})
    else:
        raise ValueError("Provide --input_json or --input_text")

    if not texts:
        raise ValueError("No texts loaded from input")
    return texts


def encode_texts(
    texts: List[Dict[str, str]],
    model_name: str,
    mode: str,
    batch_size: int,
    max_length: int,
    device: str,
) -> List[Dict[str, Any]]:
    """Encode texts using BERT. Returns list of result dicts."""
    from transformers import BertModel, BertTokenizer

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Loading BERT model: {model_name}")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    print(f"Model loaded — device: {device}, mode: {mode}")

    results = []
    for batch_start in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[batch_start:batch_start + batch_size]
        batch_texts = [item["text"] for item in batch]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        hidden_states = outputs.last_hidden_state  # (batch, seq, dim)
        attention_mask = inputs['attention_mask']   # (batch, seq)

        for i, item in enumerate(batch):
            if mode == 'cls':
                embedding = hidden_states[i, 0].cpu().numpy().astype(np.float32)
            else:  # token
                # Exclude CLS (idx 0) and SEP tokens; use attention mask
                mask = attention_mask[i].cpu()
                seq_len = int(mask.sum()) - 2  # subtract CLS and SEP
                if seq_len <= 0:
                    embedding = np.empty((0, hidden_states.shape[-1]), dtype=np.float32)
                else:
                    # Tokens 1..seq_len (inclusive) are the content tokens
                    embedding = hidden_states[i, 1:1 + seq_len].cpu().numpy().astype(np.float32)

            results.append({
                "text_id": item["text_id"],
                "text": item["text"],
                "embedding": embedding,
                "model_name": model_name,
                "mode": mode,
            })

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Generate textual embeddings using BERT'
    )
    parser.add_argument('--mode', type=str, default='cls', choices=['cls', 'token'],
                        help='Embedding mode: cls (default) or token')
    parser.add_argument('--input_json', type=str, default=None,
                        help='JSON file with list of objects')
    parser.add_argument('--text_field', type=str, default='text',
                        help='Field name containing text in JSON objects (default: text)')
    parser.add_argument('--tokens_field', type=str, default=None,
                        help='JSON field containing pre-tokenized word list (joined with spaces as text). '
                             'Mutually exclusive with --text_field when specified.')
    parser.add_argument('--id_field', type=str, default=None,
                        help='JSON field to use as text_id (default: falls back to id/text_id/index)')
    parser.add_argument('--input_text', type=str, default=None,
                        help='Plain text file, one sentence per line')
    parser.add_argument('--model', type=str, default='bert-base-uncased',
                        help='HuggingFace model name (default: bert-base-uncased)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for encoding (default: 32)')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length (default: 512)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu, default: auto-detect)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output pickle file')

    args = parser.parse_args()

    texts = load_texts(args)
    print(f"Loaded {len(texts)} texts")

    results = encode_texts(
        texts=texts,
        model_name=args.model,
        mode=args.mode,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )

    with open(args.output, 'wb') as f:
        pickle.dump(results, f, protocol=4)

    # Summary
    sample = results[0]['embedding']
    print(f"\n{'=' * 60}")
    print("TEXT EMBEDDING EXTRACTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    print(f"Texts encoded: {len(results)}")
    print(f"Embedding shape: {sample.shape}")
    print(f"Output saved to: {args.output}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
