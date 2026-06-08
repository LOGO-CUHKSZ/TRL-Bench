"""MAPO (Memory Augmented Policy Optimization) decoder.

This decoder uses:
- Actor-learner architecture for distributed training
- Beam search decoding
- Replay buffer for off-policy learning
- Pre-computed embeddings (model-agnostic)
"""

from .trainer import MAPODecoder

__all__ = ['MAPODecoder']
