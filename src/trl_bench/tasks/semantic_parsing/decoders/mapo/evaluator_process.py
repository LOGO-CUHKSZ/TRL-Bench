import json
import os
import re
import sys
import time
from typing import List, Union
import numpy as np
from tensorboardX import SummaryWriter

from . import nn_util
from .agent import PGAgent
from ...execution.env_factory import QAProgrammingEnv, Sample

import torch.multiprocessing as torch_mp
from multiprocessing import Queue, Process

import torch

# Stubs for features not used in embedding-based training
class ConsistencyModel:
    pass

class QuestionSimilarityModel:
    pass

def get_parser_agent_by_name(name):
    return PGAgent

STOP_SIGNAL = 'STOP'


class Evaluation(object):
    @staticmethod
    def evaluate(model, dataset: List[QAProgrammingEnv], beam_size: int):
        was_training = model.training
        model.eval()

        decode_results = model.decode_examples(dataset, beam_size=beam_size)
        eval_results = Evaluation.evaluate_decode_results(dataset, decode_results)

        if was_training:
            model.train()

        return eval_results

    @staticmethod
    def evaluate_decode_results(dataset: List[QAProgrammingEnv], decoding_results=Union[List[List[Sample]], List[Sample]], verbose=False):
        if isinstance(decoding_results[0], Sample):
            decoding_results = [[hyp] for hyp in decoding_results]

        acc_list = []
        oracle_acc_list = []

        for env, hyp_list in zip(dataset, decoding_results):
            is_top_correct = len(hyp_list) > 0 and hyp_list[0].trajectory.reward == 1.
            has_correct_program = any(hyp.trajectory.reward == 1. for hyp in hyp_list)

            acc_list.append(is_top_correct)
            oracle_acc_list.append(has_correct_program)

        eval_result = dict(accuracy=np.average(acc_list),
                           oracle_accuracy=np.average(oracle_acc_list))

        return eval_result


class Evaluator(torch_mp.Process):
    def __init__(self, config, eval_file, device, column_pkl_path=None, question_pkl_paths=None):
        super(Evaluator, self).__init__(daemon=True)
        self.eval_queue = Queue()
        self.config = config
        self.eval_file = eval_file
        self.device = device
        self.column_pkl_path = column_pkl_path
        self.question_pkl_paths = question_pkl_paths or []

        self.model_path = 'INIT_MODEL'
        self.message_var = None

    def run(self):
        # initialize cuda context
        self.device = torch.device(self.device)
        if 'cuda' in self.device.type:
            torch.cuda.set_device(self.device)

        # seed the random number generators
        nn_util.init_random_seed(self.config['seed'], self.device)

        # Use embedding-based agent if column pkl is provided
        if self.column_pkl_path is not None:
            from .learner import EmbeddingAgent
            self.agent = EmbeddingAgent.build(
                self.config,
                self.column_pkl_path,
                self.question_pkl_paths,
                master='evaluator'
            ).to(self.device).eval()
        else:
            agent_name = self.config.get('parser', 'vanilla')
            self.agent = get_parser_agent_by_name(agent_name).build(self.config, master='evaluator').to(self.device).eval()

        self.load_environments()
        summary_writer = SummaryWriter(os.path.join(self.config['work_dir'], 'tb_log/dev'))

        dev_scores = []

        while True:
            if self.check_and_load_new_model():
                print(f'[Evaluator] evaluate model [{self.model_path}]', file=sys.stderr)
                t1 = time.time()

                decode_results = self.agent.decode_examples(self.environments, beam_size=self.config['beam_size'], batch_size=32)

                eval_results = Evaluation.evaluate_decode_results(self.environments, decode_results)

                t2 = time.time()
                print(f'[Evaluator] step={self.get_global_step()}, result={repr(eval_results)}, took {t2 - t1}s', file=sys.stderr)

                summary_writer.add_scalar('eval/accuracy', eval_results['accuracy'], self.get_global_step())
                summary_writer.add_scalar('eval/oracle_accuracy', eval_results['oracle_accuracy'], self.get_global_step())

                dev_score = eval_results['accuracy']
                if not dev_scores or max(dev_scores) < dev_score:
                    print(f'[Evaluator] save the current best model', file=sys.stderr)

                    with open(os.path.join(self.config['work_dir'], 'dev.log'), 'w') as f:
                        f.write(json.dumps(eval_results))

                    self.agent.save(os.path.join(self.config['work_dir'], 'model.best.bin'))

                dev_scores.append(dev_score)

                sys.stderr.flush()

            time.sleep(2)  # in seconds

    def load_environments(self):
        from ...execution.env_factory import load_environments
        from transformers import BertTokenizer

        # Get tokenizer: EmbeddingEncoder uses pre-computed embeddings aligned to
        # original tokens, so BERT subtokenization must be skipped (None).
        # Only use BERT tokenizer when a real BertEncoder provides one.
        if hasattr(self.agent.encoder, 'bert_model') and hasattr(self.agent.encoder.bert_model, 'tokenizer'):
            bert_tokenizer = self.agent.encoder.bert_model.tokenizer
        else:
            bert_tokenizer = None

        envs = load_environments(
            example_files=[self.eval_file],
            table_file=self.config['table_file'],
            table_representation_method=self.config['table_representation'],
            bert_tokenizer=bert_tokenizer,
            max_n_mem=self.config.get('max_n_mem', 60))
        for env in envs:
            env.use_cache = False
            env.punish_extra_work = False

        self.environments = envs

    def check_and_load_new_model(self):
        new_model_path = self.message_var.value.decode()
        # if new_model_path == STOP_SIGNAL:
        #     print('[Evaluator] Exited', file=sys.stderr)
        #     exit(0)
        # print(f'[Evaluator] current newest model path [{new_model_path}]', file=sys.stderr)
        if new_model_path and new_model_path != self.model_path:
            t1 = time.time()

            state_dict = torch.load(new_model_path, map_location=lambda storage, loc: storage)
            self.agent.load_state_dict(state_dict)
            self.model_path = new_model_path

            t2 = time.time()
            print('[Evaluator] loaded new model [%s] (took %.2f s)' % (new_model_path, t2 - t1), file=sys.stderr)

            return True
        else:
            return False

    def get_global_step(self):
        if not self.model_path:
            return 0

        model_name = self.model_path.split('/')[-1]
        train_iter = re.search('iter(\d+)?', model_name).group(1)

        return int(train_iter)


# Alias for embedding-based training
EmbeddingEvaluator = Evaluator
