"""
TABBIE Table Embedder (inference-only)

Patched from: table_embedder/models/pretrain.py (the PretrainDev class)

Architecture:
  12 alternating row-wise and column-wise self-attention layers.
  Input: per-cell BERT [CLS] embeddings arranged in a grid.
  Output: row embeddings and column embeddings after 12 row+col transformer passes.
  Table embedding: average of row_embs[0,0,:] and col_embs[0,0,:] (CLS intersection).

Checkpoint key structure (verified by probe_checkpoint.py):
  - No _module. prefix
  - Positional: row_pos_embedding.weight (31, 768), col_pos_embedding.weight (25, 768)
  - Per block: transformer_{row,col}{1..12}.{self_attention_0, feedforward_0,
    layer_norm_0, feedforward_layer_norm_0} (new-style keys)
  - Attention: combined Q/K/V via _combined_projection (2304, 768)
  - LayerNorm: uses .gamma/.beta (old PyTorch names for .weight/.bias)
  - Prediction head: feedforward._linear_layers (skipped — inference only)
"""

import os

import numpy as np
import torch
import torch.nn as nn

from .stacked_self_attention import TransformerBlock
from .embedder_util import TableUtil


class TableEmbedder(nn.Module):
    """TABBIE's dual-transformer table encoder.

    Architecture:
        For each layer i (1..12):
          1. Reshape to row sequences -> apply transformer_row_i
          2. Reshape to col sequences -> apply transformer_col_i
        Positional embeddings (row_pos, col_pos) are added before the first layer.

    The CLS row and CLS column (loaded from .npy files) are prepended to the
    grid before the transformer layers, creating a (nrow+1) x (ncol+1) grid.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        feedforward_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        max_rows: int = 30,
        max_cols: int = 20,
        row_pos_size: int = 31,
        col_pos_size: int = 25,
        dropout: float = 0.1,
        clsrow_path: str = None,
        clscol_path: str = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_rows = max_rows
        self.max_cols = max_cols

        # Row transformers (1-indexed in original: transformer_row1..12)
        self.row_transformers = nn.ModuleList()
        for _ in range(num_layers):
            self.row_transformers.append(
                TransformerBlock(
                    input_dim=hidden_dim,
                    feedforward_hidden_dim=feedforward_dim,
                    num_attention_heads=num_heads,
                    dropout_prob=dropout,
                )
            )

        # Column transformers (1-indexed in original: transformer_col1..12)
        self.col_transformers = nn.ModuleList()
        for _ in range(num_layers):
            self.col_transformers.append(
                TransformerBlock(
                    input_dim=hidden_dim,
                    feedforward_hidden_dim=feedforward_dim,
                    num_attention_heads=num_heads,
                    dropout_prob=dropout,
                )
            )

        # Positional embeddings — sizes must match checkpoint exactly
        # Checkpoint: row_pos_embedding (31, 768), col_pos_embedding (25, 768)
        self.row_pos_embedding = nn.Embedding(row_pos_size, hidden_dim)
        self.col_pos_embedding = nn.Embedding(col_pos_size, hidden_dim)

        # CLS tokens (loaded from .npy files) — fail fast if missing
        if not clsrow_path or not os.path.exists(clsrow_path):
            raise FileNotFoundError(
                f"clsrow.npy not found at {clsrow_path}. "
                f"Vendor this file from the original TABBIE repo's data/ directory."
            )
        if not clscol_path or not os.path.exists(clscol_path):
            raise FileNotFoundError(
                f"clscol.npy not found at {clscol_path}. "
                f"Vendor this file from the original TABBIE repo's data/ directory."
            )
        cls_row_np = np.load(clsrow_path)
        cls_col_np = np.load(clscol_path)
        self._cls_row = nn.Parameter(torch.from_numpy(cls_row_np).float(), requires_grad=False)
        self._cls_col = nn.Parameter(torch.from_numpy(cls_col_np).float(), requires_grad=False)

    def get_tabemb(self, bert_embs, nrow, ncol):
        """Run the dual-transformer encoder on cell embeddings.

        Args:
            bert_embs: (batch, nrow*ncol, hidden_dim) -- per-cell BERT [CLS] embeddings
            nrow: number of data rows (before CLS prepend)
            ncol: number of data columns (before CLS prepend)

        Returns:
            row_embs: (batch, (nrow+1)*(ncol+1), hidden_dim) after final row transformer
            col_embs: (batch, (nrow+1)*(ncol+1), hidden_dim) after final col transformer
        """
        batch_size = bert_embs.size(0)
        device = bert_embs.device

        # Add CLS row and column
        table_emb = TableUtil.add_cls_tokens(
            bert_embs, self._cls_row.to(device), self._cls_col.to(device), nrow, ncol
        )

        # New dimensions with CLS
        nrow_cls = nrow + 1
        ncol_cls = ncol + 1

        # Build positional embedding indices
        row_indices = torch.arange(nrow_cls, device=device).unsqueeze(1).expand(nrow_cls, ncol_cls)
        row_indices = row_indices.reshape(1, nrow_cls * ncol_cls).expand(batch_size, -1)

        col_indices = torch.arange(ncol_cls, device=device).unsqueeze(0).expand(nrow_cls, ncol_cls)
        col_indices = col_indices.reshape(1, nrow_cls * ncol_cls).expand(batch_size, -1)

        # Add positional embeddings
        pos_emb = self.row_pos_embedding(row_indices) + self.col_pos_embedding(col_indices)
        table_emb = table_emb + pos_emb

        # Alternating row and column transformers
        row_embs = None
        col_embs = None

        for layer_idx in range(self.num_layers):
            # Row-wise: reshape to (batch*nrow_cls, ncol_cls, hidden_dim)
            row_input = TableUtil.get_row_embs(table_emb, nrow_cls, ncol_cls)
            row_output = self.row_transformers[layer_idx](row_input)
            # Reshape back to flat: (batch, nrow_cls*ncol_cls, hidden_dim)
            row_embs = row_output.view(batch_size, nrow_cls * ncol_cls, self.hidden_dim)

            # Column-wise: reshape to (batch*ncol_cls, nrow_cls, hidden_dim)
            col_input = TableUtil.get_col_embs(row_embs, nrow_cls, ncol_cls)
            col_output = self.col_transformers[layer_idx](col_input)
            # Reshape back: get_col_embs transposes, so we transpose back
            col_embs_grid = col_output.view(batch_size, ncol_cls, nrow_cls, self.hidden_dim)
            col_embs = col_embs_grid.permute(0, 2, 1, 3).contiguous().view(
                batch_size, nrow_cls * ncol_cls, self.hidden_dim
            )

            # Feed col output as input to next layer
            table_emb = col_embs

        return row_embs, col_embs

    def load_weights(self, weights_path):
        """Load pretrained weights from checkpoint.

        Key remapping:
          1. transformer_row{i} -> row_transformers.{i-1}
          2. transformer_col{i} -> col_transformers.{i-1}
          3. LayerNorm .gamma -> .weight, .beta -> .bias
          4. Skip old-style duplicate keys (_attention_layers, _feedfoward_layers, etc.)
          5. Skip prediction head (top-level feedforward)

        Args:
            weights_path: Path to weights.pt (pre-extracted by probe_checkpoint.py)
        """
        state_dict = torch.load(weights_path, map_location="cpu")

        # Remap checkpoint keys to our model structure
        remapped = {}
        for key, value in state_dict.items():
            new_key = key

            # Skip old-style AllenNLP duplicate keys (we use new-style)
            if "._attention_layers." in key or "._feedfoward_layers." in key:
                continue
            if "._layer_norm_layers." in key or "._feed_forward_layer_norm_layers." in key:
                continue

            # Skip prediction head (top-level feedforward, not inside a transformer)
            if key.startswith("feedforward."):
                continue

            # 1. Remap 1-indexed transformer names to ModuleList indices
            for i in range(1, 13):
                new_key = new_key.replace(f"transformer_row{i}.", f"row_transformers.{i-1}.")
                new_key = new_key.replace(f"transformer_col{i}.", f"col_transformers.{i-1}.")

            # 2. Remap LayerNorm gamma/beta -> weight/bias
            new_key = new_key.replace(".gamma", ".weight")
            new_key = new_key.replace(".beta", ".bias")

            remapped[new_key] = value

        # Match against model keys
        model_keys = set(self.state_dict().keys())
        cls_keys = {k for k in model_keys if k.startswith("_cls_")}
        matchable_model_keys = model_keys - cls_keys

        # Check for shape mismatches
        own_state = self.state_dict()
        shape_ok = {}
        shape_mismatches = []
        for k in matchable_model_keys:
            if k in remapped:
                if remapped[k].shape == own_state[k].shape:
                    shape_ok[k] = remapped[k]
                else:
                    shape_mismatches.append(
                        f"    {k}: checkpoint {remapped[k].shape} vs model {own_state[k].shape}"
                    )

        missing_keys = matchable_model_keys - set(remapped.keys())
        unexpected_keys = set(remapped.keys()) - model_keys

        # Report
        print(f"  Weight loading summary:")
        print(f"    Model parameters (excl CLS): {len(matchable_model_keys)}")
        print(f"    Checkpoint keys (remapped):   {len(remapped)}")
        print(f"    Matched & shape-OK:           {len(shape_ok)}")
        if shape_mismatches:
            print(f"    Shape mismatches:             {len(shape_mismatches)}")
            for m in shape_mismatches:
                print(m)
        if missing_keys:
            print(f"    Missing from checkpoint:      {len(missing_keys)}")
            for k in sorted(missing_keys):
                print(f"      {k}")
        if unexpected_keys:
            print(f"    Extra checkpoint keys:        {len(unexpected_keys)}")

        # Fail if no parameters matched
        if len(shape_ok) == 0:
            raise RuntimeError(
                "No checkpoint parameters matched the model. "
                "Key remapping is likely wrong -- run probe_checkpoint.py to inspect key names."
            )

        coverage = len(shape_ok) / len(matchable_model_keys) if matchable_model_keys else 0
        if coverage < 0.9:
            raise RuntimeError(
                f"Only {coverage:.1%} of model parameters loaded from checkpoint "
                f"({len(shape_ok)}/{len(matchable_model_keys)}). "
                f"This indicates a key remapping problem."
            )

        self.load_state_dict(shape_ok, strict=False)
        print(f"  Loaded {len(shape_ok)}/{len(matchable_model_keys)} parameters ({coverage:.1%} coverage)")
