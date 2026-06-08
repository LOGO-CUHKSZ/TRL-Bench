#!/usr/bin/env python
"""Main entry point for semantic parsing evaluation.

Usage:
    python -m downstream_tasks.semantic_parsing.run_test \
        --model assets/checkpoints/semantic_parsing/wiki_table_questions/mapo/model.best.bin \
        --column-pkl embeddings/column/tapas/semantic_parsing.pkl \
        --question-pkls embeddings/semantic_parsing/wiki_table_questions/bert/questions_dev.pkl \
        --test-file datasets/semantic_parsing/wiki_table_questions/data_split_1/test_split.jsonl \
        --table-file datasets/semantic_parsing/wiki_table_questions/tables.jsonl \
        --cuda
"""

import argparse
import json
import sys
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Evaluate a semantic parsing model')

    # Required paths
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (.bin file)')
    parser.add_argument('--column-pkl', type=str, required=True,
                        help='Path to column embeddings pkl file')
    parser.add_argument('--question-pkls', type=str, nargs='+', required=True,
                        help='Paths to question embeddings pkl files')
    parser.add_argument('--test-file', type=str, required=True,
                        help='Path to test file (.jsonl)')
    parser.add_argument('--table-file', type=str, required=True,
                        help='Path to tables file (.jsonl)')

    # Optional
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA')
    parser.add_argument('--beam-size', type=int, default=5,
                        help='Beam size for decoding')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--save-decode-to', type=str, default=None,
                        help='Save decoding results to file')

    args = parser.parse_args()

    # Validate paths
    for path_arg, path_val in [('model', args.model), ('column-pkl', args.column_pkl),
                                ('test-file', args.test_file), ('table-file', args.table_file)]:
        if not os.path.exists(path_val):
            print(f'Error: {path_arg} not found: {path_val}', file=sys.stderr)
            sys.exit(1)
    for pkl_path in args.question_pkls:
        if not os.path.exists(pkl_path):
            print(f'Error: question-pkl not found: {pkl_path}', file=sys.stderr)
            sys.exit(1)

    # Import local modules (no TaBERT/NSM dependency)
    from trl_bench.tasks.semantic_parsing.execution.env_factory import load_environments
    from trl_bench.tasks.semantic_parsing.decoders.mapo.agent import PGAgent
    from trl_bench.tasks.semantic_parsing.decoders.mapo.decoder import BertDecoder
    from trl_bench.tasks.semantic_parsing.decoders.mapo.evaluator_process import Evaluation
    from trl_bench.tasks.semantic_parsing.decoders.mapo.encoder import EmbeddingEncoder
    from transformers import BertTokenizer
    import torch
    import numpy as np

    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}', file=sys.stderr)

    # Load model checkpoint
    print(f'Loading model from {args.model}...', file=sys.stderr)
    checkpoint = torch.load(args.model, map_location=device)
    config = checkpoint['config']

    # EmbeddingEncoder uses pre-computed embeddings aligned to original tokens,
    # so BERT subtokenization must be skipped to avoid index mismatches.
    tokenizer = None

    # Load test environments
    print(f'Loading test file: {args.test_file}', file=sys.stderr)
    test_envs = load_environments(
        [args.test_file],
        table_file=args.table_file,
        table_representation_method=config.get('table_representation', 'canonical'),
        bert_tokenizer=tokenizer,
        max_n_mem=config.get('max_n_mem', 60),
    )
    print(f'Loaded {len(test_envs)} test examples', file=sys.stderr)

    # Set example IDs for embedding lookup
    for env in test_envs:
        env.example_id = env.name

    # Build encoder
    encoder = EmbeddingEncoder.build(
        config=config,
        column_pkl_path=args.column_pkl,
        question_pkl_paths=args.question_pkls,
        master=None
    )

    # Build agent using the EmbeddingEncoder directly
    # (PGAgent.build() would ignore our encoder and construct a BertEncoder)
    from trl_bench.tasks.semantic_parsing.execution.worlds.wikitablequestions import world_config as wikitablequestions_config
    config['builtin_func_num'] = wikitablequestions_config['interpreter_builtin_func_num']
    decoder = BertDecoder.build(config, encoder, master=None)
    agent = PGAgent(encoder, decoder, config=config)
    state_key = 'agent_state_dict' if 'agent_state_dict' in checkpoint else 'state_dict'
    agent.load_state_dict(checkpoint[state_key])
    agent = agent.to(device)
    agent.eval()

    # Decode
    print(f'Decoding with beam size {args.beam_size}...', file=sys.stderr)
    decode_results = agent.decode_examples(
        test_envs,
        beam_size=args.beam_size,
        batch_size=args.batch_size
    )

    # Evaluate
    assert len(test_envs) == len(decode_results)
    eval_results = Evaluation.evaluate_decode_results(test_envs, decode_results)

    # Print results
    print('\n' + '=' * 60)
    print('TEST RESULTS')
    print('=' * 60)
    print(f'Accuracy: {eval_results["accuracy"]:.4f} ({eval_results["accuracy"]*100:.2f}%)')
    print(f'Oracle Accuracy: {eval_results["oracle_accuracy"]:.4f} ({eval_results["oracle_accuracy"]*100:.2f}%)')
    print('=' * 60)

    # Save results
    results = {
        'accuracy': eval_results['accuracy'],
        'oracle_accuracy': eval_results['oracle_accuracy']
    }

    # Write to test.log in model directory
    model_dir = Path(args.model).parent
    log_path = model_dir / 'test.log'
    with open(log_path, 'w') as f:
        json.dump(results, f)
    print(f'Results saved to {log_path}', file=sys.stderr)

    # Optionally save decode results
    if args.save_decode_to:
        decode_dict = {}
        for env, hyp_list in zip(test_envs, decode_results):
            decode_dict[env.name] = [
                {'program': str(hyp.trajectory), 'score': float(hyp.prob)}
                for hyp in hyp_list
            ]
        with open(args.save_decode_to, 'w') as f:
            json.dump(decode_dict, f, indent=2)
        print(f'Decode results saved to {args.save_decode_to}', file=sys.stderr)


if __name__ == '__main__':
    main()
