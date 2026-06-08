#!/usr/bin/env python
"""Main entry point for semantic parsing training.

This script routes to the appropriate task + decoder combination.

Usage:
    python -m downstream_tasks.semantic_parsing.run_training \
        --task wiki_table_questions \
        --decoder mapo \
        --column-pkl embeddings/column/tapas/semantic_parsing.pkl \
        --question-pkls embeddings/semantic_parsing/wiki_table_questions/bert/questions_train.pkl \
                        embeddings/semantic_parsing/wiki_table_questions/bert/questions_dev.pkl \
        --dataset-path datasets/semantic_parsing/wiki_table_questions \
        --output-dir assets/checkpoints/semantic_parsing/wiki_table_questions/mapo/tapas \
        --config downstream_tasks/semantic_parsing/config/mapo.json \
        --cuda
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Train a semantic parsing decoder')

    # Task and decoder selection
    parser.add_argument('--task', type=str, default='wiki_table_questions',
                        help='Task name (e.g., wiki_table_questions)')
    parser.add_argument('--decoder', type=str, default='mapo',
                        help='Decoder name (e.g., mapo)')

    # Paths
    parser.add_argument('--column-pkl', type=str, required=True,
                        help='Path to column embeddings pkl file')
    parser.add_argument('--question-pkls', type=str, nargs='+', required=True,
                        help='Paths to question embeddings pkl files')
    parser.add_argument('--dataset-path', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='Directory to save training logs')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')

    # Training options
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint in output-dir')

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f'Error: Config file not found: {config_path}', file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    # Get task and decoder
    from .tasks import get_task
    from .decoders import get_decoder

    try:
        task_cls = get_task(args.task)
    except ValueError as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)

    try:
        decoder_cls = get_decoder(args.decoder)
    except ValueError as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)

    # Initialize task and load dataset
    task = task_cls()
    print(f'[run_training] Loading dataset from {args.dataset_path}...', file=sys.stderr)
    dataset = task.load_dataset(Path(args.dataset_path))
    print(f'[run_training] Loaded {len(dataset["train"])} train examples, {len(dataset["dev"])} dev examples',
          file=sys.stderr)

    # Update config with dataset paths
    config['table_file'] = str(Path(args.dataset_path) / 'tables.jsonl')
    config['train_shard_dir'] = str(Path(args.dataset_path) / 'data_split_1')
    config['train_shard_prefix'] = 'train_split_shard_90-'
    config['dev_file'] = str(Path(args.dataset_path) / 'data_split_1' / 'dev_split.jsonl')
    config['saved_program_file'] = str(Path(args.dataset_path) / 'saved_programs.json')

    # Initialize decoder and train
    decoder = decoder_cls()
    print(f'[run_training] Starting training with {args.decoder} decoder...', file=sys.stderr)

    results = decoder.train(
        config=config,
        column_pkl_path=Path(args.column_pkl),
        question_pkl_paths=[Path(p) for p in args.question_pkls],
        dataset=dataset,
        output_dir=Path(args.output_dir),
        log_dir=Path(args.log_dir) if args.log_dir else None,
        cuda=args.cuda,
        seed=args.seed,
        resume=args.resume,
    )

    print(f'[run_training] Training complete. Results saved to {results["output_dir"]}', file=sys.stderr)


if __name__ == '__main__':
    main()
