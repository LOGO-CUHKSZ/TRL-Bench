"""WikiTableQuestions task implementation."""

import json
from pathlib import Path
from typing import Dict, List, Any

from ..base import TaskBase
from .. import register_task


@register_task('wiki_table_questions')
class WTQTask(TaskBase):
    """WikiTableQuestions semantic parsing task.

    Task: Given a natural language question and a Wikipedia table,
    generate a program that executes on the table to produce the answer.
    """

    @property
    def name(self) -> str:
        return 'wiki_table_questions'

    def load_dataset(self, dataset_path: Path) -> Dict[str, Any]:
        """Load WTQ dataset from the given path.

        Expected directory structure:
            dataset_path/
                tables.jsonl
                saved_programs.json
                data_split_1/
                    train_split.jsonl
                    dev_split.jsonl
                    train_split_shard_90-*.jsonl
        """
        dataset_path = Path(dataset_path)

        # Load tables
        tables = {}
        tables_file = dataset_path / 'tables.jsonl'
        if tables_file.exists():
            with open(tables_file) as f:
                for line in f:
                    table = json.loads(line)
                    tables[table['name']] = table

        # Load saved programs
        programs = {}
        programs_file = dataset_path / 'saved_programs.json'
        if programs_file.exists():
            with open(programs_file) as f:
                programs = json.load(f)

        # Load train and dev splits (from data_split_1)
        split_dir = dataset_path / 'data_split_1'

        train_examples = []
        train_file = split_dir / 'train_split.jsonl'
        if train_file.exists():
            with open(train_file) as f:
                for line in f:
                    train_examples.append(json.loads(line))

        dev_examples = []
        dev_file = split_dir / 'dev_split.jsonl'
        if dev_file.exists():
            with open(dev_file) as f:
                for line in f:
                    dev_examples.append(json.loads(line))

        return {
            'tables': tables,
            'train': train_examples,
            'dev': dev_examples,
            'programs': programs,
            'dataset_path': str(dataset_path),
        }

    def create_environments(
        self,
        examples: List[Dict],
        tables: Dict,
        config: Dict
    ) -> List[Any]:
        """Create QA environments for WTQ examples.

        This delegates to the execution layer.
        """
        # Import here to avoid circular imports
        from ...execution import create_wtq_environments
        return create_wtq_environments(examples, tables, config)

    def get_vocabulary(self) -> Dict[str, Any]:
        """Return WTQ vocabulary.

        WTQ uses a Lisp-like DSL with table operations.
        """
        return {
            'operators': [
                'select', 'filter', 'argmax', 'argmin',
                'count', 'sum', 'average', 'max', 'min',
                'diff', 'same', 'hop', 'first', 'last',
                'and', 'or', 'not', 'all_rows',
            ],
            'type_hierarchy': {
                'row': ['row'],
                'primitive': ['num', 'date', 'str'],
                'entity': ['cell', 'column'],
            }
        }


def load_train_shards(dataset_path: Path, shard_prefix: str = 'train_split_shard_90-',
                      shard_start: int = 0, shard_end: int = 90) -> List[str]:
    """Get list of training shard files.

    Args:
        dataset_path: Path to dataset directory
        shard_prefix: Prefix for shard files
        shard_start: Starting shard ID
        shard_end: Ending shard ID (exclusive)

    Returns:
        List of shard file paths
    """
    split_dir = Path(dataset_path) / 'data_split_1'
    shard_files = []
    for i in range(shard_start, shard_end):
        shard_file = split_dir / f'{shard_prefix}{i}.jsonl'
        if shard_file.exists():
            shard_files.append(str(shard_file))
    return shard_files
