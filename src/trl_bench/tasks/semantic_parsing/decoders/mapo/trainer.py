"""MAPO Decoder implementation.

This module provides the MAPODecoder class that wraps the MAPO training
and inference logic with the DecoderBase interface.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch

from ..base import DecoderBase
from .. import register_decoder


@register_decoder('mapo')
class MAPODecoder(DecoderBase):
    """MAPO (Memory Augmented Policy Optimization) decoder.

    Uses actor-learner architecture with beam search decoding.
    Designed for use with pre-computed embeddings.
    """

    def __init__(self):
        self.agent = None
        self.config = None
        self.column_pkl_path = None
        self.question_pkl_paths = None

    @property
    def name(self) -> str:
        return 'mapo'

    def train(
        self,
        config: Dict,
        column_pkl_path: Path,
        question_pkl_paths: List[Path],
        dataset: Dict,
        output_dir: Path,
        log_dir: Optional[Path] = None,
        cuda: bool = True,
        seed: int = 0,
        resume: bool = False,
    ) -> Dict[str, Any]:
        """Train the MAPO decoder using actor-learner architecture.

        This launches multiple processes:
        - Learner: Receives samples and updates the model
        - Actors: Explore trajectories and send samples to learner
        - Evaluator: Periodically evaluates on dev set
        """
        import torch.multiprocessing as mp

        from .learner import EmbeddingLearner, EmbeddingAgent
        from .actor import EmbeddingActor
        from .evaluator_process import EmbeddingEvaluator
        from .program_cache import SharedProgramCache

        # Ensure output directories exist
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / 'log').mkdir(exist_ok=True)

        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

        # Handle resume: find checkpoint info
        resume_info = None
        if resume:
            resume_info = self._find_resume_checkpoint(output_dir)
            if resume_info:
                print(f"[MAPODecoder] Resuming from iteration {resume_info['start_iter']}", file=sys.stderr)
                print(f"[MAPODecoder]   Model: {resume_info['model_path']}", file=sys.stderr)
                print(f"[MAPODecoder]   Program cache: {resume_info['program_cache_path']}", file=sys.stderr)
            else:
                print("[MAPODecoder] No checkpoint found, starting from scratch", file=sys.stderr)

        # Update config with paths
        config = dict(config)
        config['work_dir'] = str(output_dir)
        config['seed'] = seed
        config['column_pkl_path'] = str(column_pkl_path)
        config['question_pkl_paths'] = [str(p) for p in question_pkl_paths]

        # Save config (only if not resuming, to preserve original config)
        if not resume:
            with open(output_dir / 'config.json', 'w') as f:
                json.dump(config, f, indent=2)

        # IMPORTANT: Set start method BEFORE creating any multiprocessing objects
        # This ensures Manager() and Value() use the spawn context
        mp.set_start_method('spawn', force=True)

        # Setup devices
        if cuda and torch.cuda.is_available():
            device = torch.device('cuda:0')
            gpu_ids = list(range(torch.cuda.device_count()))
        else:
            device = torch.device('cpu')
            gpu_ids = [-1]

        # Create shared program cache (must be after set_start_method)
        shared_program_cache = SharedProgramCache(config.get('n_replay_samples', 1))

        # Load programs into cache
        cache_loaded = False
        if resume_info and resume_info.get('program_cache_path'):
            # Load from checkpoint program cache
            print(f"[MAPODecoder] Loading program cache from checkpoint...", file=sys.stderr)
            try:
                self._load_program_cache(shared_program_cache, resume_info['program_cache_path'])
                cache_loaded = True
            except (json.JSONDecodeError, OSError) as e:
                print(f"[MAPODecoder] WARNING: Corrupted program cache, falling back to saved programs: {e}", file=sys.stderr)
        if not cache_loaded and 'programs' in dataset and dataset['programs']:
            # Load from saved programs file
            print('[MAPODecoder] Loading saved programs into cache...', file=sys.stderr)
            for example_id, programs in dataset['programs'].items():
                for program in programs:
                    shared_program_cache.add(example_id, program)

        # Create learner
        learner = EmbeddingLearner(
            config=config,
            column_pkl_path=str(column_pkl_path),
            question_pkl_paths=[str(p) for p in question_pkl_paths],
            devices=device,
            shared_program_cache=shared_program_cache,
            resume_info=resume_info
        )

        # Get all training example IDs and distribute among actors
        all_example_ids = [ex['id'] for ex in dataset['train']]
        actor_num = config.get('actor_num', 16)

        # Distribute examples among actors
        example_ids_per_actor = [[] for _ in range(actor_num)]
        for i, ex_id in enumerate(all_example_ids):
            example_ids_per_actor[i % actor_num].append(ex_id)

        # Create actors
        actors = []
        for i in range(actor_num):
            actor_gpu_id = gpu_ids[i % len(gpu_ids)] if gpu_ids[0] >= 0 else -1
            actor = EmbeddingActor(
                actor_id=i,
                example_ids=example_ids_per_actor[i],
                shared_program_cache=shared_program_cache,
                config=config,
                column_pkl_path=str(column_pkl_path),
                question_pkl_paths=[str(p) for p in question_pkl_paths],
                device=torch.device(f'cuda:{actor_gpu_id}' if actor_gpu_id >= 0 else 'cpu')
            )
            learner.register_actor(actor)
            actors.append(actor)

        # Create evaluator
        eval_file = config.get('dev_file', '')

        evaluator = EmbeddingEvaluator(
            config=config,
            eval_file=eval_file,
            column_pkl_path=str(column_pkl_path),
            question_pkl_paths=[str(p) for p in question_pkl_paths],
            device=device
        )
        learner.register_evaluator(evaluator)

        # Start all processes (order: actors first, then evaluator, then learner)
        print('[MAPODecoder] Starting training...', file=sys.stderr)
        print(f'[MAPODecoder] Starting {len(actors)} actors...', file=sys.stderr)
        for actor in actors:
            actor.start()

        print('[MAPODecoder] Starting evaluator...', file=sys.stderr)
        evaluator.start()

        print('[MAPODecoder] Starting learner...', file=sys.stderr)
        learner.start()

        # Wait for learner to complete
        learner.join()

        # Terminate actors and evaluator
        for actor in actors:
            actor.terminate()
            actor.join()

        evaluator.terminate()
        evaluator.join()

        print('[MAPODecoder] Training complete.', file=sys.stderr)

        # Return training results
        return {
            'output_dir': str(output_dir),
            'config': config,
        }

    def decode(
        self,
        embedding_path: Path,
        examples: List[Dict],
        beam_size: int = 10,
        cuda: bool = True,
    ) -> List[Dict]:
        """Decode examples to programs using beam search."""
        if self.agent is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        from .learner import EmbeddingAgent

        predictions = []
        for example in examples:
            # Get beam predictions
            result = self.agent.beam_search(
                example,
                beam_size=beam_size,
            )
            predictions.append({
                'id': example.get('id', ''),
                'program': result.get('program', ''),
                'score': result.get('score', 0.0),
                'beam': result.get('beam', []),
                'correct': result.get('correct', False),
            })

        return predictions

    def load(self, model_path: Path):
        """Load a trained model."""
        from .learner import EmbeddingAgent

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Load config from same directory
        config_path = model_path.parent / 'config.json'
        if config_path.exists():
            with open(config_path) as f:
                self.config = json.load(f)
        else:
            # Try to load from model itself
            params = torch.load(model_path, map_location='cpu')
            self.config = params.get('config', {})

        # Get embedding paths from config
        self.column_pkl_path = self.config.get('column_pkl_path')
        self.question_pkl_paths = self.config.get('question_pkl_paths', [])

        # Load agent
        gpu_id = 0 if torch.cuda.is_available() else -1
        self.agent = EmbeddingAgent.load(
            str(model_path),
            self.column_pkl_path,
            self.question_pkl_paths,
            gpu_id=gpu_id
        )

        print(f'[MAPODecoder] Loaded model from {model_path}', file=sys.stderr)

    def _find_resume_checkpoint(self, output_dir: Path) -> Optional[Dict]:
        """Find the latest checkpoint in the output directory for resuming.

        Returns a dict with:
          - model_path: Path to the model state file
          - program_cache_path: Path to the program cache file (optional)
          - start_iter: Iteration to resume from
        """
        import re

        output_dir = Path(output_dir)

        # Find model checkpoints (agent_state.iter{N}.bin)
        model_files = list(output_dir.glob('agent_state.iter*.bin'))
        if not model_files:
            return None

        # Extract iteration numbers and find the latest
        iter_pattern = re.compile(r'agent_state\.iter(\d+)\.bin')
        model_iters = []
        for f in model_files:
            match = iter_pattern.match(f.name)
            if match:
                model_iters.append((int(match.group(1)), f))

        if not model_iters:
            return None

        model_iters.sort(key=lambda x: x[0], reverse=True)
        latest_iter, latest_model = model_iters[0]

        # Find matching program cache (or closest one)
        log_dir = output_dir / 'log'
        program_cache_path = None
        if log_dir.exists():
            cache_files = list(log_dir.glob('program_cache.iter*.json'))
            if cache_files:
                cache_pattern = re.compile(r'program_cache\.iter(\d+)\.json')
                cache_iters = []
                for f in cache_files:
                    match = cache_pattern.match(f.name)
                    if match:
                        cache_iters.append((int(match.group(1)), f))

                if cache_iters:
                    cache_iters.sort(key=lambda x: x[0], reverse=True)
                    # Find the cache closest to (but not greater than) the model iter
                    for cache_iter, cache_file in cache_iters:
                        if cache_iter <= latest_iter:
                            program_cache_path = cache_file
                            break
                    if program_cache_path is None:
                        # Use the latest available
                        program_cache_path = cache_iters[0][1]

        return {
            'model_path': str(latest_model),
            'program_cache_path': str(program_cache_path) if program_cache_path else None,
            'start_iter': latest_iter,
        }

    def _load_program_cache(self, shared_program_cache, cache_path: str):
        """Load program cache from a JSON file into SharedProgramCache."""
        with open(cache_path) as f:
            cache_data = json.load(f)

        count = 0
        for env_name, hypotheses in cache_data.items():
            for hyp in hypotheses:
                if isinstance(hyp, dict):
                    # Format: {"program": [...], "prob": float}
                    program = hyp.get('program', [])
                    prob = hyp.get('prob', 1.0)
                    if isinstance(program, str):
                        program = program.strip().split()
                    shared_program_cache.add_hypothesis(env_name, program, prob)
                elif isinstance(hyp, list):
                    # Format: just the program as list
                    shared_program_cache.add_hypothesis(env_name, hyp, 1.0)
                elif isinstance(hyp, str):
                    # Format: program as string
                    shared_program_cache.add(env_name, hyp)
                count += 1

        print(f"[MAPODecoder] Loaded {count} programs for {len(cache_data)} examples", file=sys.stderr)
