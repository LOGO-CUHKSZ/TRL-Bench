"""Base interface for semantic parsing tasks."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any


class TaskBase(ABC):
    """Base interface for all semantic parsing tasks.

    Each task defines:
    - How to load the dataset
    - How to create execution environments
    - Task-specific vocabulary (operators, functions)
    """

    @abstractmethod
    def load_dataset(self, dataset_path: Path) -> Dict[str, Any]:
        """Load task-specific dataset.

        Args:
            dataset_path: Path to the dataset directory

        Returns:
            Dictionary containing:
            - 'tables': Table definitions
            - 'train': Training examples
            - 'dev': Development examples
            - 'programs': Saved programs (optional)
        """
        pass

    @abstractmethod
    def create_environments(
        self,
        examples: List[Dict],
        tables: Dict,
        config: Dict
    ) -> List[Any]:
        """Create execution environments for examples.

        Args:
            examples: List of example dictionaries
            tables: Table definitions
            config: Configuration dictionary

        Returns:
            List of environment objects
        """
        pass

    @abstractmethod
    def get_vocabulary(self) -> Dict[str, Any]:
        """Return task-specific vocabulary.

        Returns:
            Dictionary containing:
            - 'operators': List of operators
            - 'functions': List of functions
            - 'type_hierarchy': Type hierarchy definition
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the task name."""
        pass
