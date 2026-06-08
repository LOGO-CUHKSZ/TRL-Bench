"""Base interface for semantic parsing decoders."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional


class DecoderBase(ABC):
    """Base interface for all semantic parsing decoders.

    A decoder takes encoded representations (from embeddings)
    and produces programs as output.
    """

    @abstractmethod
    def train(
        self,
        config: Dict,
        embedding_path: Path,
        dataset: Dict,
        output_dir: Path,
        log_dir: Optional[Path] = None,
        cuda: bool = True,
        seed: int = 0,
    ) -> Dict[str, Any]:
        """Train the decoder.

        Args:
            config: Training configuration
            embedding_path: Path to pre-computed embeddings
            dataset: Dataset dictionary from task.load_dataset()
            output_dir: Directory to save checkpoints
            log_dir: Directory to save training logs
            cuda: Whether to use CUDA
            seed: Random seed

        Returns:
            Training results/metrics
        """
        pass

    @abstractmethod
    def decode(
        self,
        embedding_path: Path,
        examples: List[Dict],
        beam_size: int = 10,
        cuda: bool = True,
    ) -> List[Dict]:
        """Decode examples to programs.

        Args:
            embedding_path: Path to pre-computed embeddings
            examples: List of example dictionaries
            beam_size: Beam search size
            cuda: Whether to use CUDA

        Returns:
            List of predictions, each containing:
            - 'program': Predicted program
            - 'score': Confidence score
            - 'beam': Full beam (optional)
        """
        pass

    @abstractmethod
    def load(self, model_path: Path):
        """Load a trained model.

        Args:
            model_path: Path to model checkpoint
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the decoder name."""
        pass
