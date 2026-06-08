"""Execution layer for semantic parsing.

This module provides:
- Program execution on tables
- Environment factory for creating QA environments
- Computer factory for Lisp interpretation
"""

from .env_factory import (
    QAProgrammingEnv,
    Trajectory,
    Sample,
    Environment,
    Observation,
)
from .worlds.wikitablequestions import WikiTableExecutor
from .computer_factory import LispInterpreter

# Stub for ProgramTypeError (not used in embedding-based training)
class ProgramTypeError(Exception):
    pass


def create_wtq_environments(examples, tables, config):
    """Create WikiTableQuestions environments.

    This is a convenience wrapper for env_factory.create_environments.
    """
    from .env_factory import create_environments
    return create_environments(
        examples=examples,
        tables=tables,
        config=config,
        world_name='wikitablequestions'
    )


__all__ = [
    'create_wtq_environments',
    'WikiTableExecutor',
    'QAProgrammingEnv',
    'LispInterpreter',
    'ProgramTypeError',
]
