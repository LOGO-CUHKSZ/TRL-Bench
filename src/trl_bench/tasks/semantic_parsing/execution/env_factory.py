"A collections of environments of sequence generations tasks."
import sys
import json
from collections import OrderedDict
from typing import List, Dict, Any, Union, Iterable, Optional
import collections
import pprint
import numpy as np

# import nlp_utils
import torch
from . import computer_factory
from .worlds.wikitablequestions import WikiTableExecutor
from .wtq import evaluator as wtq_evaluator

import bloom_filter


# Simple Column class for table representation
class Column:
    def __init__(self, name, raw_name=None, type='text', sample_value=None,
                 name_tokens=None, sample_value_tokens=None):
        self.name = name
        self.raw_name = raw_name or name
        self.type = type
        self.sample_value = sample_value
        self.name_tokens = name_tokens or []
        self.sample_value_tokens = sample_value_tokens or []


# Simple Table class for table representation
class Table:
    def __init__(self, id, header=None, data=None, column_info=None):
        self.id = id
        self.header = header or []
        self.data = data or []
        self.column_info = column_info or {}


def load_jsonl(fn):
    """Load JSONL file."""
    data = []
    with open(fn, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def wtq_score(prediction, answer):
    """Score function for WikiTableQuestions."""
    processed_answer = wtq_evaluator.target_values_map(*answer)
    correct = wtq_evaluator.check_prediction(
        [str(p) for p in prediction], processed_answer)
    if correct:
        return 1.0
    else:
        return 0.0


def get_sample_value(raw_column_name, table):
    """Get a sample value from the table for a column."""
    sample_value = None
    for row_id, row in table['kg'].items():
        if raw_column_name in row and isinstance(row[raw_column_name], list) and len(str(row[raw_column_name][0])) > 0:
            sample_value = row[raw_column_name][0]
            break
    return sample_value


def get_columns_canonical(example, table):
    """Get canonical columns from table."""
    canonical_columns = OrderedDict()
    canonical_column_ids = OrderedDict()
    columns = []
    raw_column_canonical_ids = []

    for col_id, raw_column_name in enumerate(table['props']):
        column_name = raw_column_name[len('r.'):]
        type_pos = column_name.rfind('-')
        column_name = untyped_column_name = column_name[:type_pos]
        column_name = column_name.replace('-', ' ').replace('_', ' ')

        raw_type_string = raw_column_name[raw_column_name.rfind('-') + 1:]

        if raw_type_string == 'string':
            type_string = 'text'
        elif raw_type_string.startswith('num') or raw_type_string.startswith('date'):
            type_string = 'real'
        else:
            type_string = 'text'

        sample_value = get_sample_value(raw_column_name, table)

        if untyped_column_name in canonical_columns:
            column_entry = canonical_columns[untyped_column_name]
            if sample_value is not None and column_entry.type == 'text' and type_string == 'real':
                column_entry.type = 'real'
                column_entry.sample_value = sample_value
            raw_column_canonical_ids.append(canonical_column_ids[untyped_column_name])
        else:
            column = Column(name=column_name,
                            raw_name=raw_column_name,
                            type=type_string,
                            sample_value=sample_value)
            canonical_columns[untyped_column_name] = column
            canonical_column_ids[untyped_column_name] = col_id
            raw_column_canonical_ids.append(col_id)

        columns.append(Column(name=raw_column_name, type=raw_type_string))

    canonical_columns = list(canonical_columns.values())

    column_info = {
        'raw_columns': columns,
        'raw_column_canonical_ids': raw_column_canonical_ids
    }

    return canonical_columns, column_info


def annotate_example_for_bert(
    example: Dict, table: Dict,
    bert_tokenizer,
    table_representation_method: Optional[str] = 'canonical'
):
    """Annotate example with BERT tokenization."""
    e_id = example['id']

    # sub-tokenize the question
    question_tokens = example['tokens']
    example['original_tokens'] = question_tokens
    token_position_map = OrderedDict()

    question_feature = example['features']

    cur_idx = 0
    new_question_feature = []
    question_subtokens = []
    for old_idx, token in enumerate(question_tokens):
        if token == '<DECODE>': token = '[MASK]'
        if token == '<START>': token = '[MASK]'

        sub_tokens = bert_tokenizer.tokenize(token)
        question_subtokens.extend(sub_tokens)

        token_new_idx_start = cur_idx
        token_new_idx_end = cur_idx + len(sub_tokens)
        token_position_map[old_idx] = (token_new_idx_start, token_new_idx_end)
        new_question_feature.extend([question_feature[old_idx]] * len(sub_tokens))

        cur_idx = token_new_idx_end

    token_position_map[len(question_tokens)] = (len(question_subtokens), len(question_subtokens))

    example['tokens'] = question_subtokens
    example['features'] = new_question_feature

    for entity in example['entities']:
        old_token_start = entity['token_start']
        old_token_end = entity['token_end']

        new_token_start = token_position_map[old_token_start][0]
        new_token_end = token_position_map[old_token_end][0]

        entity['token_start'] = new_token_start
        entity['token_end'] = new_token_end

    # Get columns
    if table_representation_method == 'canonical':
        columns, column_info = get_columns_canonical(example, table)
    else:
        columns, column_info = get_columns_canonical(example, table)

    # gather table data
    for column in columns:
        column.name_tokens = bert_tokenizer.tokenize(str(column.name))
        column.sample_value_tokens = bert_tokenizer.tokenize(str(column.sample_value) if column.sample_value else '')

    rows = [table['kg'][row_id] for row_id in sorted(table['kg'])]
    valid_rows = []
    untokenized_rows = []
    for row in rows:
        valid_row = {}
        untokenized_row = {}
        for col in columns:
            cell_val = row.get(col.raw_name, [])
            if cell_val:
                cell_val = str(cell_val[0])
                untokenized_row[col.name] = cell_val
                cell_tokens = bert_tokenizer.tokenize(cell_val)
            else:
                cell_tokens = []
                untokenized_row[col.name] = ''

            valid_row[col.name] = cell_tokens

        valid_rows.append(valid_row)
        untokenized_rows.append(untokenized_row)

    table_obj = Table(id=example['context'], header=columns, data=valid_rows, column_info=column_info)
    untokenized_table = Table(id=example['context'], header=columns, data=untokenized_rows)

    example['table'] = table_obj
    example['untokenized_table'] = untokenized_table

    return example


def create_environment(
    example_dict: Dict, table_kg: Dict,
    table_representation_method: str,
    executor_type: str = 'wtq',
    max_n_mem: int = 60, max_n_exp: int = 3,
    bert_tokenizer = None,
) -> 'QAProgrammingEnv':
    """Create a single QA environment."""
    if executor_type == 'wtq':
        score_fn = wtq_score
        process_answer_fn = lambda x: x
        executor_fn = WikiTableExecutor
    else:
        raise ValueError('Unknown executor {}'.format(executor_type))

    executor = executor_fn(table_kg)
    api = executor.get_api()
    type_hierarchy = api['type_hierarchy']
    func_dict = api['func_dict']
    constant_dict = api['constant_dict']

    interpreter = computer_factory.LispInterpreter(
        type_hierarchy=type_hierarchy,
        max_mem=max_n_mem,
        max_n_exp=max_n_exp,
        assisted=True
    )

    for v in func_dict.values():
        interpreter.add_function(**v)

    interpreter.add_constant(
        value=table_kg['row_ents'],
        type='entity_list',
        name='all_rows')

    if bert_tokenizer:
        example = annotate_example_for_bert(
            example_dict, table_kg, bert_tokenizer,
            table_representation_method=table_representation_method
        )
    else:
        # EmbeddingEncoder path: build Table object without BERT subtokenization.
        # Tokens and entity spans stay as original dataset values, matching
        # the pre-computed embeddings which are aligned to original tokens.
        example = example_dict
        columns, column_info = get_columns_canonical(example, table_kg)
        for column in columns:
            column.name_tokens = str(column.name).split()
            column.sample_value_tokens = str(column.sample_value).split() if column.sample_value else []
        rows = [table_kg['kg'][row_id] for row_id in sorted(table_kg['kg'])]
        valid_rows = []
        for row in rows:
            valid_row = {}
            for col in columns:
                cell_val = row.get(col.raw_name, [])
                valid_row[col.name] = [str(cell_val[0])] if cell_val else []
            valid_rows.append(valid_row)
        table_obj = Table(id=example['context'], header=columns, data=valid_rows, column_info=column_info)
        example['table'] = table_obj
        example['untokenized_table'] = table_obj

    env = QAProgrammingEnv(
        question_annotation=example,
        kg=table_kg,
        answer=process_answer_fn(example['answer']),
        constants=constant_dict.values(),
        interpreter=interpreter,
        score_fn=score_fn,
        name=example['id']
    )

    return env


def create_environments(
    table_dict, dataset,
    table_representation_method,
    executor_type,
    max_n_mem=60, max_n_exp=3,
    bert_tokenizer=None
) -> List['QAProgrammingEnv']:
    """Create environments for all examples in dataset."""
    all_envs = []

    for i, example in enumerate(dataset):
        if i % 100 == 0:
            print('creating environment #{}'.format(i))

        kg_info = table_dict[example['context']]

        env = create_environment(
            example, kg_info,
            table_representation_method,
            executor_type,
            max_n_mem, max_n_exp,
            bert_tokenizer
        )

        all_envs.append(env)

    return all_envs


def load_environments(
    example_files: List[str],
    table_file: str,
    table_representation_method: str = 'canonical',
    example_ids: Iterable = None,
    bert_tokenizer = None,
    max_n_mem: int = 60,
) -> List['QAProgrammingEnv']:
    """Load environments from example files and table file.

    Args:
        example_files: List of paths to JSONL files containing examples
        table_file: Path to JSONL file containing tables
        table_representation_method: How to represent tables ('canonical')
        example_ids: Optional set of example IDs to filter
        bert_tokenizer: BERT tokenizer for tokenization

    Returns:
        List of QAProgrammingEnv environments
    """
    dataset = []
    if example_ids is not None:
        example_ids = set(example_ids)

    for fn in example_files:
        data = load_jsonl(fn)
        for example in data:
            if example_ids:
                if example['id'] in example_ids:
                    dataset.append(example)
            else:
                dataset.append(example)

    print('{} examples in dataset.'.format(len(dataset)))

    tables = load_jsonl(table_file)
    table_dict = {table['name']: table for table in tables}
    print('{} tables.'.format(len(table_dict)))

    environments = create_environments(
        table_dict, dataset,
        table_representation_method=table_representation_method,
        executor_type='wtq',
        max_n_mem=max_n_mem,
        bert_tokenizer=bert_tokenizer,
    )
    print('{} environments in total'.format(len(environments)))

    return environments


class Observation(object):
    def __init__(self, read_ind, write_ind, valid_action_indices, output_features=None, valid_action_mask=None):
        self.read_ind = read_ind
        self.write_ind = write_ind
        self.valid_action_indices = valid_action_indices
        self.output_features = output_features
        self.valid_action_mask = valid_action_mask

    def to(self, device: torch.device):
        if self.read_ind.device == device:
            return self

        self.read_ind = self.read_ind.to(device)
        self.write_ind = self.write_ind.to(device)
        if self.valid_action_indices is not None:
            self.valid_action_indices = self.valid_action_indices.to(device)
        self.output_features = self.output_features.to(device)
        self.valid_action_mask = self.valid_action_mask.to(device)

        return self

    def slice(self, t: int):
        return Observation(self.read_ind[:, t],
                           self.write_ind[:, t],
                           None,
                           self.output_features[:, t],
                           self.valid_action_mask[:, t])

    def remove_action(self, action_id):
        action_rel_id = self.valid_action_indices.index(action_id)
        del self.valid_action_indices[action_rel_id]
        if self.output_features:
            del self.output_features[action_rel_id]

    @staticmethod
    def empty():
        """create an empty observation for padding"""
        return Observation(0, -1, [], [])

    @staticmethod
    def get_valid_action_masks(obs: List['Observation'], memory_size):
        batch_size = len(obs)

        # initialize valid action mask
        valid_action_mask = torch.zeros(batch_size, memory_size)
        for i, observation in enumerate(obs):
            valid_action_mask[i, observation.valid_action_indices] = 1.

        return valid_action_mask

    @staticmethod
    def to_batched_input(obs: List['Observation'], memory_size) -> 'Observation':
        batch_size = len(obs)

        read_ind = torch.tensor([ob.read_ind for ob in obs])
        write_ind = torch.tensor([ob.write_ind for ob in obs])

        # pad output features
        feat_num = len(obs[0].output_features[0])
        output_feats = np.zeros((batch_size, memory_size, feat_num), dtype=np.float32)
        valid_action_mask = torch.zeros(batch_size, memory_size)

        for i, observation in enumerate(obs):
            if observation.valid_action_indices:
                output_feats[i, observation.valid_action_indices] = observation.output_features
                valid_action_mask[i, observation.valid_action_indices] = 1.

        output_feats = torch.from_numpy(output_feats)

        # valid_action_mask = Observation.get_valid_action_masks(obs, memory_size=memory_size)

        return Observation(read_ind, write_ind, None, output_feats, valid_action_mask)

    @staticmethod
    def to_batched_sequence_input(obs_seq: List[List['Observation']], memory_size) -> 'Observation':
        batch_size = len(obs_seq)
        seq_len = max(len(ob_seq) for ob_seq in obs_seq)

        read_ind = torch.zeros(batch_size, seq_len, dtype=torch.long)
        write_ind = torch.zeros(batch_size, seq_len, dtype=torch.long).fill_(-1.)
        valid_action_mask = torch.zeros(batch_size, seq_len, memory_size)
        feat_num = len(obs_seq[0][0].output_features[0])
        output_feats = np.zeros((batch_size, seq_len, memory_size, feat_num), dtype=np.float32)

        for batch_id in range(batch_size):
            ob_seq_i = obs_seq[batch_id]
            for t in range(len(ob_seq_i)):
                ob = obs_seq[batch_id][t]
                read_ind[batch_id, t] = ob.read_ind
                write_ind[batch_id, t] = ob.write_ind

                valid_action_mask[batch_id, t, ob.valid_action_indices] = 1.
                output_feats[batch_id, t, ob.valid_action_indices] = ob.output_features

        output_feats = torch.from_numpy(output_feats)

        return Observation(read_ind, write_ind, None, output_feats, valid_action_mask)

    def __repr__(self):
        return f'Observation(read_id={repr(self.read_ind)}, write_id={repr(self.write_ind)}, ' \
            f'valid_actions={repr(self.valid_action_indices)})'

    __str__ = __repr__


class Trajectory(object):
    def __init__(self, environment_name: str,
                 observations: List[Observation],
                 context: Dict,
                 tgt_action_ids: List[int],
                 answer: Any,
                 reward: float,
                 program: List[str] = None,
                 human_readable_program: List[str] = None,
                 id: str = None):
        self.id = id
        self.environment_name = environment_name

        self.observations = observations
        self.context = context
        self.tgt_action_ids = tgt_action_ids
        self.answer = answer
        self.reward = reward
        self.program = program
        self.human_readable_program = human_readable_program

        self._hash = hash((self.environment_name, ' '.join(str(a) for a in self.tgt_action_ids)))

    def __hash__(self):
        return self._hash

    def __repr__(self):
        if self.human_readable_program:
            return ' '.join(self.human_readable_program)

        elif self.program:
            return ' '.join(self.program)

        return '[Undecoded Program] ' + ' '.join(map(str, self.tgt_action_ids))

    __str__ = __repr__

    @classmethod
    def from_environment(cls, env):
        return Trajectory(
            env.name,
            observations=env.obs,
            context=env.get_context(),
            tgt_action_ids=env.mapped_actions,
            answer=env.interpreter.result,
            reward=env.rewards[-1],
            program=env.program,
            human_readable_program=env.to_human_readable_program()
        )

    @classmethod
    def from_program(cls, env, program):
        env = env.clone()
        env.use_cache = False
        ob = env.start_ob

        for token in program:
            action_id = env.de_vocab.lookup(token)
            # try:
            rel_action_id = ob.valid_action_indices.index(action_id)
            # except ValueError:
            #    return None

            ob, _, _, _ = env.step(rel_action_id)

        trajectory = Trajectory.from_environment(env)

        return trajectory

    @classmethod
    def to_batched_sequence_tensors(cls, trajectories: List['Trajectory'], memory_size):
        batch_size = len(trajectories)

        obs_seq = [traj.observations for traj in trajectories]
        max_seq_len = max(len(ob_seq) for ob_seq in obs_seq)

        batched_obs_seq = Observation.to_batched_sequence_input(obs_seq, memory_size=memory_size)

        tgt_action_ids = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        tgt_action_mask = torch.zeros(batch_size, max_seq_len)
        for batch_id in range(batch_size):
            traj_i_action_ids = trajectories[batch_id].tgt_action_ids
            tgt_action_ids[batch_id, :len(traj_i_action_ids)] = traj_i_action_ids
            tgt_action_mask[batch_id, :len(traj_i_action_ids)] = 1.

        tgt_action_ids = torch.from_numpy(tgt_action_ids)

        return batched_obs_seq, dict(tgt_action_ids=tgt_action_ids, tgt_action_mask=tgt_action_mask)


class Environment(object):
    """Environment with OpenAI Gym like interface."""

    def step(self, action):
        """
        Args:
          action: an action to execute against the environment.

        Returns:
          observation:
          reward:
          done:
          info:
        """
        raise NotImplementedError


# Use last action and the new variable's memory location as input.
ProgrammingObservation = collections.namedtuple(
    'ProgramObservation', ['last_actions', 'output', 'valid_actions'])


class QAProgrammingEnv(Environment):
    """
    An RL environment wrapper around an interpreter to
    learn to write programs based on question.
    """

    def __init__(self,
                 question_annotation,
                 kg,
                 answer,
                 score_fn, interpreter,
                 de_vocab=None, constants=None,
                 punish_extra_work=True,
                 init_interp=True, trigger_words_dict=None,
                 max_cache_size=1e4,
                 context=None, id_feature_dict=None,
                 cache=None,
                 reset=True,
                 name='qa_programming'):

        self.name = name
        self.de_vocab = de_vocab or interpreter.get_vocab()
        self.end_action = self.de_vocab.end_id
        self.score_fn = score_fn
        self.interpreter = interpreter
        self.answer = answer
        self.question_annotation = question_annotation
        self.kg = kg
        self.constants = constants
        self.punish_extra_work = punish_extra_work
        self.error = False
        self.trigger_words_dict = trigger_words_dict
        tokens = question_annotation['tokens']

        self.n_builtin = len(self.de_vocab.vocab) - interpreter.max_mem
        self.n_mem = interpreter.max_mem
        self.n_exp = interpreter.max_n_exp
        max_n_constants = self.n_mem - self.n_exp

        if context:
            self.context = context
        else:
            # initialize constants to be used in the interpreter

            constant_spans = []
            constant_values = []
            if constants is None:
                constants = []
            for c in constants:
                constant_spans.append([-1, -1])
                constant_values.append(c['value'])
                if init_interp:
                    self.interpreter.add_constant(
                        value=c['value'], type=c['type'])

            for entity in question_annotation['entities']:
                constant_spans.append(
                    [entity['token_start'], entity['token_end'] - 1])
                constant_values.append(entity['value'])

                if init_interp:
                    self.interpreter.add_constant(
                        value=entity['value'], type=entity['type'])

            constant_spans = constant_spans[:max_n_constants]

            if len(constant_values) > (self.n_mem - self.n_exp):
                print('Not enough memory slots for example {}, which has {} constants.'.format(
                    self.name, len(constant_values)))

            self.context = dict(
                constant_spans=constant_spans,
                question_features=question_annotation['features'],
                question_tokens=tokens,
                table=question_annotation['table'] if 'table' in question_annotation else None
            )

        # Create output features.
        if id_feature_dict:
            self.id_feature_dict = id_feature_dict
        else:
            prop_features = question_annotation['prop_features']
            feat_num = len(list(prop_features.values())[0])
            self.id_feature_dict = {}
            for name, id in self.de_vocab.vocab.items():
                self.id_feature_dict[id] = [0] * feat_num
                if name in self.interpreter.namespace:
                    val = self.interpreter.namespace[name]['value']
                    if (isinstance(val, str)) and val in prop_features:
                        self.id_feature_dict[id] = prop_features[val]

        self.context['id_feature_dict'] = self.id_feature_dict

        if 'original_tokens' in self.context:
            self.context['original_tokens'] = question_annotation['original_tokens']

        if cache:
            self.cache = cache
        else:
            self.cache = SearchCache(name=name, max_elements=max_cache_size)

        self.use_cache = False

        if reset:
            self.reset()

    def get_context(self):
        return self.context

    def step(self, action, debug=False):
        self.actions.append(action)
        if debug:
            print('-' * 50)
            print(self.de_vocab.lookup(self.valid_actions, reverse=True))
            print('pick #{} valid action'.format(action))
            print('history:')
            print(self.de_vocab.lookup(self.mapped_actions, reverse=True))
            print('env: {}, cache size: {}'.format(self.name, len(self.cache._set)))
            print('obs')
            pprint.pprint(self.obs)

        if 0 <= action < len(self.valid_actions):
            mapped_action = self.valid_actions[action]
        else:
            print('-' * 50)
            # print('env: {}, cache size: {}'.format(self.name, len(self.cache._set)))
            print('action out of range.')
            print('action:')
            print(action)
            print('valid actions:')
            print(self.de_vocab.lookup(self.valid_actions, reverse=True))
            print('pick #{} valid action'.format(action))
            print('history:')
            print(self.de_vocab.lookup(self.mapped_actions, reverse=True))
            print('obs')
            pprint.pprint(self.obs)
            print('-' * 50)
            mapped_action = self.valid_actions[action]

        self.mapped_actions.append(mapped_action)
        mapped_action_token = self.de_vocab.lookup(mapped_action, reverse=True)
        self.program.append(mapped_action_token)

        result = self.interpreter.read_token(mapped_action_token)

        self.done = self.interpreter.done
        # Only when the program is finished and it doesn't have
        # extra work or we don't care, its result will be
        # scored, and the score will be used as reward.
        if self.done and not (self.punish_extra_work and self.interpreter.has_extra_work()):
            try:
                reward = self.score_fn(self.interpreter.result, self.answer)
            except:
                print(f'Error: Env {self.name}, program [{" ".join(self.interpreter.history)}], result=[{repr(self.interpreter.result)}]', file=sys.stderr)
                exit(-1)
        else:
            reward = 0.0

        if self.done and self.interpreter.result == [computer_factory.ERROR_TK]:
            self.error = True

        if result is None or self.done:
            new_var_id = -1
        else:
            new_var_id = self.de_vocab.lookup(self.interpreter.namespace.last_var)

        valid_tokens = self.interpreter.valid_tokens()
        valid_actions = self.de_vocab.lookup(valid_tokens)

        # For each action, check the cache for the program, if
        # already tried, then not valid anymore.
        if self.use_cache:
            new_valid_actions = []
            cached_actions = []
            partial_program = self.de_vocab.lookup(self.mapped_actions, reverse=True)
            for ma in valid_actions:
                new_program = partial_program + [self.de_vocab.lookup(ma, reverse=True)]
                if not self.cache.check(new_program):
                    new_valid_actions.append(ma)
                else:
                    cached_actions.append(ma)
            valid_actions = new_valid_actions

        self.valid_actions = valid_actions
        self.rewards.append(reward)
        ob = Observation(read_ind=mapped_action,
                         write_ind=new_var_id,
                         valid_action_indices=self.valid_actions,
                         output_features=[self.id_feature_dict[a] for a in valid_actions])

        # If no valid actions are available, then stop.
        if not self.valid_actions:
            self.done = True
            self.error = True

        # If the program is not finished yet, collect the
        # observation.
        if not self.done:
            # Add the actions that are filtered by cache into the
            # training example because at test time, they will be
            # there (no cache is available).

            # Note that this part is a bit tricky, `self.obs.valid_actions`
            # maintains all valid actions regardless of the cache, while the returned
            # observation `ob` only has valid continuating actions not covered by
            # the cache. `self.obs` shall only be used in training to compute
            # valid action masks for trajectories
            if self.use_cache:
                valid_actions = self.valid_actions + cached_actions

                true_ob = Observation(read_ind=mapped_action, write_ind=new_var_id, valid_action_indices=valid_actions,
                                      output_features=[self.id_feature_dict[a] for a in valid_actions])
                self.obs.append(true_ob)
            else:
                self.obs.append(ob)
        elif self.use_cache:
            # If already finished, save it in the cache.
            self.cache.save(self.de_vocab.lookup(self.mapped_actions, reverse=True))

        return ob, reward, self.done, {}
        # 'valid_actions': valid_actions, 'new_var_id': new_var_id}

    def reset(self):
        self.actions = []
        self.mapped_actions = []
        self.program = []
        self.rewards = []
        self.done = False
        valid_actions = self.de_vocab.lookup(self.interpreter.valid_tokens())
        if self.use_cache:
            new_valid_actions = []
            for ma in valid_actions:
                partial_program = self.de_vocab.lookup(
                    self.mapped_actions + [ma], reverse=True)
                if not self.cache.check(partial_program):
                    new_valid_actions.append(ma)
            valid_actions = new_valid_actions
        self.valid_actions = valid_actions
        self.start_ob = Observation(self.de_vocab.decode_id,
                                    -1,
                                    valid_actions,
                                    [self.id_feature_dict[a] for a in valid_actions])
        self.obs = [self.start_ob]

    def interactive(self):
        self.interpreter.interactive()
        print('reward is: %s' % self.score_fn(self.interpreter))

    def clone(self):
        new_interpreter = self.interpreter.clone()
        new = QAProgrammingEnv(
            question_annotation=self.question_annotation,
            kg=self.kg,
            answer=self.answer,
            score_fn=self.score_fn,
            interpreter=new_interpreter,
            de_vocab=self.de_vocab,
            constants=self.constants,
            init_interp=False,
            context=self.context,
            id_feature_dict=self.id_feature_dict,
            cache=self.cache,
            reset=False,
        )
        new.actions = self.actions[:]
        new.mapped_actions = self.mapped_actions[:]
        new.program = self.program[:]
        new.rewards = self.rewards[:]
        new.obs = self.obs[:]
        new.done = self.done
        new.name = self.name
        # Cache is shared among all copies of this environment.
        new.cache = self.cache
        new.use_cache = self.use_cache
        new.valid_actions = self.valid_actions
        new.start_ob = self.start_ob
        new.error = self.error
        new.id_feature_dict = self.id_feature_dict
        new.punish_extra_work = self.punish_extra_work
        new.trigger_words_dict = self.trigger_words_dict

        return new

    def show(self):
        program = ' '.join(
            self.de_vocab.lookup([o.read_ind for o in self.obs], reverse=True))
        valid_tokens = ' '.join(self.de_vocab.lookup(self.valid_actions, reverse=True))
        return 'program: {}\nvalid tokens: {}'.format(program, valid_tokens)

    def get_human_readable_action_token(self, program_token: str) -> str:
        if program_token.startswith('v'):
            mem_entry = self.interpreter.namespace[program_token]
            if mem_entry['is_constant']:
                if isinstance(mem_entry['value'], list):
                    value = ', '.join(map(str, mem_entry['value']))
                else:
                    value = str(mem_entry['value'])

                token = f"{program_token}:{value}"
            else:
                token = program_token
        else:
            token = program_token

        return token

    def to_human_readable_program(self):
        readable_program = []
        for token in self.program:
            readable_token = self.get_human_readable_action_token(token)
            readable_program.append(readable_token)

        return readable_program


class SearchCache(object):
    def __init__(self, name, size=None, max_elements=1e4, error_rate=1e-8):
        self.name = name
        self.max_elements = max_elements
        self.error_rate = error_rate
        self._set = bloom_filter.BloomFilter(
            max_elements=max_elements, error_rate=error_rate)

    def check(self, tokens):
        return ' '.join(tokens) in self._set

    def save(self, tokens):
        string = ' '.join(tokens)
        self._set.add(string)

    def is_full(self):
        return '(' in self._set

    def reset(self):
        self._set = bloom_filter.BloomFilter(
            max_elements=self.max_elements, error_rate=self.error_rate)


class Sample(object):
    def __init__(self, trajectory: Trajectory, prob: Union[float, torch.Tensor], **kwargs):
        self.trajectory = trajectory
        self.prob = prob

        for field, value in kwargs.items():
            setattr(self, field, value)

    def to(self, device: torch.device):
        for ob in self.trajectory.observations:
            ob.to(device)

        return self

    def __repr__(self):
        return 'Sample({}, prob={})'.format(self.trajectory, self.prob)

    __str__ = __repr__