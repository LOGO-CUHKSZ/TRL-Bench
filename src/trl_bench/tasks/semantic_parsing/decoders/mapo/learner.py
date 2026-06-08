"""
Simplified learner for embedding-based training.

This learner is designed for training with pre-computed embeddings,
which removes the need for BERT optimizer and freeze logic.
"""

import ctypes
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
import torch.multiprocessing as torch_mp
import multiprocessing

from tensorboardX import SummaryWriter

from . import nn_util
from .agent import PGAgent
from .encoder import EmbeddingEncoder
from .decoder import BertDecoder
from .program_cache import SharedProgramCache

# Signal for stopping distributed processes
STOP_SIGNAL = 'STOP'


class EmbeddingAgent(PGAgent):
    """
    Agent using pre-computed embeddings instead of TaBERT.

    This is a lightweight version of PGAgent that uses EmbeddingEncoder
    instead of BertEncoder.
    """

    @classmethod
    def build(cls, config, column_pkl_path: str, question_pkl_paths: list, master=None):
        from ...execution.worlds.wikitablequestions import world_config as wikitablequestions_config
        config['builtin_func_num'] = wikitablequestions_config['interpreter_builtin_func_num']

        encoder = EmbeddingEncoder.build(config, column_pkl_path, question_pkl_paths, master=master)
        decoder = BertDecoder.build(config, encoder, master=master)

        return cls(
            encoder, decoder,
            config=config
        )

    def save(self, model_path, kwargs=None):
        """Save model without TaBERT weights."""
        params = {
            'config': self.config,
            'state_dict': self.state_dict(),
            'kwargs': kwargs
        }
        torch.save(params, model_path)

    @classmethod
    def load(cls, model_path, column_pkl_path: str, question_pkl_paths: list, gpu_id=-1, **kwargs):
        """Load model for embedding-based inference."""
        device = torch.device("cuda:%d" % gpu_id if gpu_id >= 0 else "cpu")
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        config = params['config']
        config.update(kwargs)

        model = cls.build(config, column_pkl_path, question_pkl_paths)
        incompatible_keys = model.load_state_dict(params['state_dict'], strict=False)
        if incompatible_keys.missing_keys:
            print('Loading agent, got missing keys {}'.format(incompatible_keys.missing_keys), file=sys.stderr)
        if incompatible_keys.unexpected_keys:
            print('Loading agent, got unexpected keys {}'.format(incompatible_keys.unexpected_keys), file=sys.stderr)

        model = model.to(device)
        model.eval()

        return model


class EmbeddingLearner(torch_mp.Process):
    """
    Simplified learner for embedding-based training.

    This removes BERT-specific logic:
    - No BERT optimizer
    - No freeze_bert logic
    - Single optimizer for all parameters
    """

    def __init__(
        self,
        config: Dict,
        column_pkl_path: str,
        question_pkl_paths: List[str],
        devices: Union[List[torch.device], torch.device],
        shared_program_cache: SharedProgramCache = None,
        resume_info: Dict = None
    ):
        super(EmbeddingLearner, self).__init__(daemon=True)

        self.train_queue = multiprocessing.Queue()
        self.checkpoint_queue = multiprocessing.Queue()
        self.config = config
        self.column_pkl_path = column_pkl_path
        self.question_pkl_paths = question_pkl_paths
        self.devices = devices
        self.actor_message_vars = []
        self.current_model_path = None
        self.shared_program_cache = shared_program_cache
        self.resume_info = resume_info

        self.actor_num = 0

    def run(self):
        # Initialize cuda context
        devices = self.devices if isinstance(self.devices, list) else [self.devices]
        self.devices = [torch.device(device) for device in devices]

        if 'cuda' in self.devices[0].type:
            torch.cuda.set_device(self.devices[0])

        # Seed random number generators
        for device in self.devices:
            nn_util.init_random_seed(self.config['seed'], device)

        # Build embedding-based agent
        self.agent = EmbeddingAgent.build(
            self.config,
            self.column_pkl_path,
            self.question_pkl_paths,
            master='learner'
        ).to(self.devices[0]).train()

        # Load checkpoint if resuming
        start_iter = 0
        if self.resume_info and self.resume_info.get('model_path'):
            model_path = self.resume_info['model_path']
            print(f'[EmbeddingLearner] Loading checkpoint from {model_path}', file=sys.stderr)
            state_dict = torch.load(model_path, map_location=self.devices[0])
            self.agent.load_state_dict(state_dict, strict=False)
            start_iter = self.resume_info.get('start_iter', 0)
            print(f'[EmbeddingLearner] Resuming from iteration {start_iter}', file=sys.stderr)

        self.train(start_iter=start_iter)

    def train(self, start_iter: int = 0):
        model = self.agent
        config = self.config
        work_dir = Path(config['work_dir'])
        train_iter = start_iter
        save_every_niter = config['save_every_niter']
        summary_writer = SummaryWriter(os.path.join(config['work_dir'], 'tb_log/train'))
        max_train_step = config['max_train_step']
        save_program_cache_niter = config.get('save_program_cache_niter', 0)
        max_program_cache_files = config.get('max_program_cache_files', 5)
        gradient_accumulation_niter = config.get('gradient_accumulation_niter', 1)

        # Single optimizer for ALL parameters (no BERT optimizer needed)
        all_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(all_params, lr=config.get('learning_rate', 0.001))

        # Restore optimizer state if resuming
        if self.resume_info and self.resume_info.get('model_path'):
            optim_path = self.resume_info['model_path'].replace('agent_state.', 'optimizer_state.')
            if os.path.exists(optim_path):
                print(f'[EmbeddingLearner] Loading optimizer state from {optim_path}', file=sys.stderr)
                optimizer.load_state_dict(torch.load(optim_path, map_location=self.devices[0]))
            else:
                print(f'[EmbeddingLearner] No optimizer state found, starting fresh', file=sys.stderr)

        self.optimizer = optimizer

        cum_loss = cum_examples = 0.
        t1 = time.time()

        optimizer.zero_grad()

        while train_iter < max_train_step:
            if 'cuda' in self.devices[0].type:
                torch.cuda.set_device(self.devices[0])

            train_iter += 1

            train_samples, samples_info = self.train_queue.get()
            try:
                queue_size = self.train_queue.qsize()
                summary_writer.add_scalar('train_queue_size', queue_size, train_iter)
            except NotImplementedError:
                pass

            train_trajectories = [sample.trajectory for sample in train_samples]

            # Process in chunks if needed (for memory efficiency)
            chunk_size = config.get('train_chunk_size', len(train_samples))
            chunk_num = int(math.ceil(len(train_samples) / chunk_size))

            if chunk_num > 1:
                for chunk_id in range(0, chunk_num):
                    train_samples_chunk = train_samples[chunk_size * chunk_id: chunk_size * chunk_id + chunk_size]
                    loss_val = self.train_step(train_samples_chunk, train_iter, summary_writer)
                    cum_loss += loss_val

                grad_multiply_factor = 1 / len(train_samples)
                for p in self.agent.parameters():
                    if p.grad is not None:
                        p.grad.data.mul_(grad_multiply_factor)
            else:
                loss_val = self.train_step(train_samples, train_iter, summary_writer, reduction='mean')
                cum_loss += loss_val * len(train_samples)

            # Clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(all_params, 5.)

            if (train_iter - start_iter) % gradient_accumulation_niter == 0:
                optimizer.step()
                optimizer.zero_grad()

            if 'clip_frac' in samples_info:
                summary_writer.add_scalar('sample_clip_frac', samples_info['clip_frac'], train_iter)

            cum_examples += len(train_samples)

            self.try_update_model_to_actors(train_iter)

            if train_iter % save_every_niter == 0:
                print(f'[EmbeddingLearner] train_iter={train_iter} avg. loss={cum_loss / cum_examples}, '
                      f'{cum_examples} examples ({cum_examples / (time.time() - t1)} examples/s)', file=sys.stderr)
                cum_loss = cum_examples = 0.
                t1 = time.time()

                # Log program cache stats
                program_cache_stat = self.shared_program_cache.stat()
                summary_writer.add_scalar(
                    'avg_num_programs_in_cache',
                    program_cache_stat['num_entries'] / program_cache_stat['num_envs'],
                    train_iter
                )
                summary_writer.add_scalar(
                    'num_programs_in_cache',
                    program_cache_stat['num_entries'],
                    train_iter
                )

            if save_program_cache_niter > 0 and train_iter % save_program_cache_niter == 0:
                program_cache_file = work_dir / 'log' / f'program_cache.iter{train_iter}.json'
                program_cache = self.shared_program_cache.all_programs()
                json.dump(
                    program_cache,
                    program_cache_file.open('w'),
                    indent=2
                )

                # Delete old program cache files, keeping only the latest N
                import re as _re
                cache_dir = work_dir / 'log'
                cache_files = sorted(
                    cache_dir.glob('program_cache.iter*.json'),
                    key=lambda f: int(_re.search(r'iter(\d+)', f.name).group(1)),
                    reverse=True
                )
                for old_file in cache_files[max_program_cache_files:]:
                    old_file.unlink()

        # Flush any remaining accumulated gradients after loop exit
        if (train_iter - start_iter) % gradient_accumulation_niter != 0:
            optimizer.step()
            # Save directly rather than via update_model_to_actors to avoid
            # deleting a same-iteration checkpoint from the last loop pass.
            final_path = os.path.join(config['work_dir'], 'agent_state.iter%d.bin' % train_iter)
            torch.save(self.agent.state_dict(), final_path)
            torch.save(self.optimizer.state_dict(),
                       os.path.join(config['work_dir'], 'optimizer_state.iter%d.bin' % train_iter))
            self.push_new_model(final_path)

    def train_step(self, train_samples, train_iter, summary_writer, reduction='sum'):
        train_trajectories = [sample.trajectory for sample in train_samples]

        # (batch_size)
        batch_log_prob, meta_info = self.agent(train_trajectories, return_info=True)

        train_sample_weights = batch_log_prob.new_tensor([s.weight for s in train_samples])
        batch_log_prob = batch_log_prob * train_sample_weights

        if reduction == 'sum':
            loss = -batch_log_prob.sum()
        elif reduction == 'mean':
            loss = -batch_log_prob.mean()
        else:
            raise ValueError(f'Unknown reduction {reduction}')

        # Capture unscaled loss for logging before gradient scaling
        unscaled_loss_val = loss.item()

        gradient_accumulation_niter = self.config.get('gradient_accumulation_niter', 1)
        if gradient_accumulation_niter > 1:
            loss /= gradient_accumulation_niter

        summary_writer.add_scalar('parser_loss', unscaled_loss_val, train_iter)

        loss.backward()

        return unscaled_loss_val

    def try_update_model_to_actors(self, train_iter):
        save_every_niter = self.config.get('save_every_niter')
        if train_iter % save_every_niter == 0:
            self.update_model_to_actors(train_iter)
        else:
            self.push_new_model(self.current_model_path)

    def update_model_to_actors(self, train_iter):
        t1 = time.time()
        model_state = self.agent.state_dict()
        model_save_path = os.path.join(self.config['work_dir'], 'agent_state.iter%d.bin' % train_iter)
        optim_save_path = os.path.join(self.config['work_dir'], 'optimizer_state.iter%d.bin' % train_iter)
        torch.save(model_state, model_save_path)
        torch.save(self.optimizer.state_dict(), optim_save_path)

        self.push_new_model(model_save_path)
        print(f'[EmbeddingLearner] pushed model [{model_save_path}] (took {time.time() - t1}s)', file=sys.stderr)

        if self.current_model_path:
            os.remove(self.current_model_path)
            # Also remove old optimizer state
            old_optim_path = self.current_model_path.replace('agent_state.', 'optimizer_state.')
            if os.path.exists(old_optim_path):
                os.remove(old_optim_path)

        self.current_model_path = model_save_path

    def push_new_model(self, model_path):
        self.checkpoint_queue.put(model_path)
        if model_path:
            self.eval_msg_val.value = model_path.encode()

    def register_actor(self, actor):
        actor.checkpoint_queue = self.checkpoint_queue
        actor.train_queue = self.train_queue
        self.actor_num += 1

    def register_evaluator(self, evaluator):
        msg_var = multiprocessing.Array(ctypes.c_char, 4096)
        self.eval_msg_val = msg_var
        evaluator.message_var = msg_var
