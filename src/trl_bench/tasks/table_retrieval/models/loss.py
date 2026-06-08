"""
Loss functions for retrieval training.

Adapted from NQT-Retrieval (DPR): dpr/models/biencoder.py
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, List, Optional


def dot_product_scores(q_vectors: Tensor, ctx_vectors: Tensor) -> Tensor:
    """
    Calculate dot product scores between query and context vectors.

    Args:
        q_vectors: Query vectors [num_queries, dim]
        ctx_vectors: Context vectors [num_contexts, dim]

    Returns:
        Scores matrix [num_queries, num_contexts]
    """
    return torch.matmul(q_vectors, ctx_vectors.transpose(0, 1))


def cosine_scores(q_vectors: Tensor, ctx_vectors: Tensor) -> Tensor:
    """
    Calculate cosine similarity scores between query and context vectors.

    Args:
        q_vectors: Query vectors [num_queries, dim]
        ctx_vectors: Context vectors [num_contexts, dim]

    Returns:
        Scores matrix [num_queries, num_contexts]
    """
    q_norm = F.normalize(q_vectors, p=2, dim=1)
    ctx_norm = F.normalize(ctx_vectors, p=2, dim=1)
    return torch.matmul(q_norm, ctx_norm.transpose(0, 1))


class BiEncoderNllLoss:
    """
    Bi-encoder negative log-likelihood loss.

    Computes contrastive loss for retrieval training using in-batch negatives.
    Adapted from DPR/NQT-Retrieval.
    """

    def __init__(
        self,
        similarity_fn: str = 'dot_product',
        temperature: float = 1.0,
        hard_neg_weight: float = 1.0,
    ):
        """
        Initialize loss function.

        Args:
            similarity_fn: Similarity function ('dot_product' or 'cosine')
            temperature: Temperature for softmax scaling
            hard_neg_weight: Weight multiplier for hard negative scores (>1.0 makes them harder)
        """
        self.temperature = temperature
        self.hard_neg_weight = hard_neg_weight
        if similarity_fn == 'dot_product':
            self.sim_fn = dot_product_scores
        elif similarity_fn == 'cosine':
            self.sim_fn = cosine_scores
        else:
            raise ValueError(f"Unknown similarity function: {similarity_fn}")

    def calc(
        self,
        q_vectors: Tensor,
        ctx_vectors: Tensor,
        positive_idx_per_question: List[int],
        hard_negative_idx_per_question: Optional[List[List[int]]] = None,
        loss_scale: Optional[float] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Calculate NLL loss for bi-encoder training.

        This is the core loss function from DPR/NQT-Retrieval. It:
        1. Computes similarity scores between queries and all contexts
        2. Applies log softmax over contexts for each query
        3. Computes NLL loss using positive context indices as targets

        Args:
            q_vectors: Query embeddings [num_queries, dim]
            ctx_vectors: Context (table) embeddings [num_contexts, dim]
            positive_idx_per_question: Index of positive context for each query
            hard_negative_idx_per_question: Indices of hard negatives per query
            loss_scale: Optional scaling factor for loss

        Returns:
            Tuple of (loss, num_correct_predictions)
        """
        # Compute similarity scores
        scores = self.sim_fn(q_vectors, ctx_vectors)  # [Q, C]

        # Apply temperature scaling
        if self.temperature != 1.0:
            scores = scores / self.temperature

        # Ensure scores are 2D
        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)

        # Apply hard negative weighting if specified and hard negatives provided
        if self.hard_neg_weight != 1.0 and hard_negative_idx_per_question is not None:
            # Boost hard negative scores to make them harder to beat
            # This increases their contribution to the softmax denominator
            for q_idx, hard_neg_indices in enumerate(hard_negative_idx_per_question):
                if hard_neg_indices:
                    for neg_idx in hard_neg_indices:
                        scores[q_idx, neg_idx] = scores[q_idx, neg_idx] * self.hard_neg_weight

        # Log softmax over all contexts for each query
        softmax_scores = F.log_softmax(scores, dim=1)

        # Convert positive indices to tensor
        positive_idx_tensor = torch.tensor(
            positive_idx_per_question,
            dtype=torch.long,
            device=softmax_scores.device
        )

        # NLL loss - push positive contexts to the top
        loss = F.nll_loss(softmax_scores, positive_idx_tensor, reduction="mean")

        # Count correct predictions (for monitoring)
        max_score, max_idxs = torch.max(softmax_scores, dim=1)
        correct_predictions = (max_idxs == positive_idx_tensor).sum()

        # Apply loss scaling if specified
        if loss_scale is not None:
            loss = loss * loss_scale

        return loss, correct_predictions

    @staticmethod
    def get_scores(q_vectors: Tensor, ctx_vectors: Tensor) -> Tensor:
        """Get similarity scores using dot product."""
        return dot_product_scores(q_vectors, ctx_vectors)

    @staticmethod
    def get_similarity_function():
        """Return the default similarity function."""
        return dot_product_scores


class InBatchNegativeLoss(BiEncoderNllLoss):
    """
    In-batch negative loss for contrastive learning.

    Uses other samples in the batch as negatives. This is more efficient
    than explicit negative sampling.
    """

    def calc_in_batch(
        self,
        q_vectors: Tensor,
        ctx_vectors: Tensor,
        loss_scale: Optional[float] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Calculate loss with in-batch negatives.

        Assumes each query has exactly one positive context, and the positive
        for query i is context i (diagonal of the score matrix).

        Args:
            q_vectors: Query embeddings [batch_size, dim]
            ctx_vectors: Positive context embeddings [batch_size, dim]
            loss_scale: Optional scaling factor

        Returns:
            Tuple of (loss, num_correct_predictions)
        """
        batch_size = q_vectors.size(0)

        # Positive indices are diagonal (query i -> context i)
        positive_idx = list(range(batch_size))

        return self.calc(q_vectors, ctx_vectors, positive_idx, loss_scale=loss_scale)
