"""
Table Utility Functions for TABBIE

Patched from: table_embedder/models/embedder_util.py (the TableUtil class)

Replacements:
  - Removed allennlp imports
  - All methods use plain PyTorch tensors
  - Simplified for inference-only use

Key methods:
  - add_cls_tokens(): Prepends CLS row and CLS column to the cell embedding grid
  - to_bert_emb(): Extracts BERT [CLS] embeddings for each cell
  - get_row_embs(): Reshapes embeddings into row-wise sequences for row transformers
  - get_col_embs(): Reshapes embeddings into column-wise sequences for col transformers
"""

import torch
import numpy as np


class TableUtil:
    """Static utility methods for TABBIE's table grid operations."""

    @staticmethod
    def add_cls_tokens(table_emb, cls_row, cls_col, nrow, ncol):
        """Prepend CLS row and CLS column to the table embedding grid.

        The CLS row is prepended as row 0 (across all columns + corner).
        The CLS column is prepended as col 0 (across all rows).
        The corner cell (0,0) is the average of cls_row and cls_col.

        Args:
            table_emb: (batch, nrow*ncol, hidden_dim) — flat cell embeddings
            cls_row: (hidden_dim,) — learned CLS row vector
            cls_col: (hidden_dim,) — learned CLS column vector
            nrow: number of data rows
            ncol: number of data columns

        Returns:
            (batch, (nrow+1)*(ncol+1), hidden_dim) — grid with CLS row/col prepended
        """
        batch_size = table_emb.size(0)
        hidden_dim = table_emb.size(2)
        device = table_emb.device

        # Reshape to grid: (batch, nrow, ncol, hidden_dim)
        grid = table_emb.view(batch_size, nrow, ncol, hidden_dim)

        # Create CLS column: (batch, nrow, 1, hidden_dim)
        cls_col_expanded = cls_col.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        cls_col_expanded = cls_col_expanded.expand(batch_size, nrow, 1, hidden_dim)

        # Prepend CLS column to each row
        grid_with_col = torch.cat([cls_col_expanded, grid], dim=2)  # (batch, nrow, ncol+1, hidden_dim)

        # Create CLS row: (batch, 1, ncol+1, hidden_dim)
        cls_row_expanded = cls_row.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        cls_row_expanded = cls_row_expanded.expand(batch_size, 1, ncol + 1, hidden_dim)

        # Corner cell (0,0) = average of cls_row and cls_col
        corner = (cls_row + cls_col) / 2.0
        cls_row_with_corner = cls_row_expanded.clone()
        cls_row_with_corner[:, :, 0, :] = corner.unsqueeze(0)

        # Prepend CLS row
        full_grid = torch.cat([cls_row_with_corner, grid_with_col], dim=1)  # (batch, nrow+1, ncol+1, hidden_dim)

        # Flatten back: (batch, (nrow+1)*(ncol+1), hidden_dim)
        return full_grid.view(batch_size, (nrow + 1) * (ncol + 1), hidden_dim)

    @staticmethod
    def get_row_embs(table_emb, nrow, ncol):
        """Reshape flat embeddings into row-wise sequences.

        Args:
            table_emb: (batch, nrow*ncol, hidden_dim)
            nrow: number of rows (including CLS row)
            ncol: number of columns (including CLS column)

        Returns:
            (batch*nrow, ncol, hidden_dim) — each row is a sequence
        """
        batch_size = table_emb.size(0)
        hidden_dim = table_emb.size(2)

        # (batch, nrow, ncol, hidden_dim)
        grid = table_emb.view(batch_size, nrow, ncol, hidden_dim)

        # (batch*nrow, ncol, hidden_dim)
        return grid.view(batch_size * nrow, ncol, hidden_dim)

    @staticmethod
    def get_col_embs(table_emb, nrow, ncol):
        """Reshape flat embeddings into column-wise sequences.

        Args:
            table_emb: (batch, nrow*ncol, hidden_dim)
            nrow: number of rows (including CLS row)
            ncol: number of columns (including CLS column)

        Returns:
            (batch*ncol, nrow, hidden_dim) — each column is a sequence
        """
        batch_size = table_emb.size(0)
        hidden_dim = table_emb.size(2)

        # (batch, nrow, ncol, hidden_dim)
        grid = table_emb.view(batch_size, nrow, ncol, hidden_dim)

        # Transpose rows and columns: (batch, ncol, nrow, hidden_dim)
        grid_t = grid.permute(0, 2, 1, 3).contiguous()

        # (batch*ncol, nrow, hidden_dim)
        return grid_t.view(batch_size * ncol, nrow, hidden_dim)
