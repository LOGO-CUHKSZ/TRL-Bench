"""Task registry for semantic parsing tasks."""

from .base import TaskBase

TASK_REGISTRY = {}


def register_task(name: str):
    """Decorator to register a task."""
    def decorator(cls):
        TASK_REGISTRY[name] = cls
        return cls
    return decorator


def get_task(name: str) -> TaskBase:
    """Get a task class by name."""
    if name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {name}. Available: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[name]


# Import tasks to register them
from .wiki_table_questions import WTQTask
