"""
TABBIE CSV to Embeddings

Extracts table-level, column-level, and row-level embeddings from CSV files
using TABBIE's dual-transformer architecture. Each cell is first embedded via
BERT [CLS] tokens, then processed through 12 alternating row-wise and
column-wise self-attention layers.

Embedding modes:
  - cls (table): CLS intersection average = (row_embs[0,0] + col_embs[0,0]) / 2
  - column: Per-column vectors from CLS row of the final column transformer
  - row: Per-row vectors via batched header+row mini-table processing

Usage:
    from csv_to_embeddings import TABBIEEmbedder

    embedder = TABBIEEmbedder('checkpoints/tabbie/weights.pt')
    emb = embedder.csv_to_embeddings('table.csv')                    # (768,) np.float32
    col = embedder.csv_to_embeddings('table.csv', aggregate='column') # dict
    row = embedder.csv_to_row_embeddings('table.csv')                # (n_rows, 768) np.float32
"""

import os
import argparse

import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

from tabbie_model.table_embedder import TableEmbedder


# TABBIE's grid size limits (from original code)
MAX_ROWS = 30
MAX_COLS = 20
MAX_CELL_TOKENS = 128  # Max BERT tokens per cell


class TABBIEEmbedder:
    """Extracts table, column, and row embeddings using TABBIE.

    Architecture:
        1. Each cell is tokenized and embedded via BERT → [CLS] token extracted
        2. Cell embeddings arranged in a grid, CLS row/col prepended
        3. 12 alternating row & column transformer layers
        4. Table embedding = avg of CLS intersection from row & col outputs

    Args:
        model_path: Path to weights.pt (pre-extracted by probe_checkpoint.py)
        device_id: GPU device ID (None=auto, -1=CPU, int=specific GPU)
        clsrow_path: Path to clsrow.npy (default: models/tabbie/data/clsrow.npy)
        clscol_path: Path to clscol.npy (default: models/tabbie/data/clscol.npy)
        max_rows: Max data rows to read from CSV (default: 30, max: 30).
            The checkpoint's row_pos_embedding supports at most 30 data rows + 1 CLS row.
    """

    def __init__(self, model_path, device_id=None, clsrow_path=None, clscol_path=None,
                 bert_model_name="bert-base-uncased", max_rows=None):
        self.model_path = model_path

        # Clamp max_rows to architectural limit
        if max_rows is None:
            self.max_rows = MAX_ROWS
        elif max_rows > MAX_ROWS:
            raise ValueError(
                f"max_rows={max_rows} exceeds TABBIE's architectural limit of {MAX_ROWS}. "
                f"The checkpoint's row_pos_embedding only supports {MAX_ROWS} data rows + 1 CLS row."
            )
        else:
            self.max_rows = max_rows

        # Resolve device
        if device_id is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                print("GPU detected, using cuda:0")
            else:
                self.device = torch.device("cpu")
                print("No GPU detected, using CPU")
        elif device_id == -1:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{device_id}")

        # Resolve CLS vector paths
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        if clsrow_path is None:
            clsrow_path = os.path.join(data_dir, "clsrow.npy")
        if clscol_path is None:
            clscol_path = os.path.join(data_dir, "clscol.npy")

        # Fail fast if weights missing (no fallback extraction)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"weights.pt not found at {model_path}. "
                f"Run probe_checkpoint.py first to extract from mix.tar.gz"
            )

        # Load BERT tokenizer and model for cell embedding
        # bert_model_name can be a HuggingFace model name or a local directory path,
        # so SLURM nodes without internet can use a pre-cached copy.
        print(f"Loading BERT tokenizer and model from '{bert_model_name}'...")
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert.to(self.device)
        self.bert.eval()

        # Build TABBIE model
        # Positional embedding sizes match checkpoint exactly:
        #   row_pos_embedding: (31, 768), col_pos_embedding: (25, 768)
        print("Building TABBIE model...")
        self.model = TableEmbedder(
            hidden_dim=768,
            feedforward_dim=3072,
            num_heads=12,
            num_layers=12,
            max_rows=MAX_ROWS,
            max_cols=MAX_COLS,
            row_pos_size=31,  # Matches checkpoint row_pos_embedding.weight shape
            col_pos_size=25,  # Matches checkpoint col_pos_embedding.weight shape
            dropout=0.0,  # No dropout at inference
            clsrow_path=clsrow_path,
            clscol_path=clscol_path,
        )

        # Load pretrained weights
        print(f"Loading TABBIE weights from {model_path}...")
        self.model.load_weights(model_path)
        self.model.to(self.device)
        self.model.eval()

        print(f"TABBIEEmbedder initialized on {self.device}")

    def _read_csv(self, csv_path):
        """Read CSV preserving raw string fidelity.

        Uses header=None so pandas does NOT consume any row as a DataFrame header.
        Row 0 is treated as the table header, rows 1+ as data.
        Truncates to self.max_rows x MAX_COLS.

        Returns:
            cells: list of list of str (header + data rows, truncated)
            nrow: number of rows (including header)
            ncol: number of columns
        """
        try:
            df = pd.read_csv(csv_path, header=None, keep_default_na=False, dtype=str)
        except Exception:
            # Fall back to python engine for CSVs with embedded \r or field mismatches
            df = pd.read_csv(csv_path, header=None, keep_default_na=False, dtype=str,
                             engine='python', on_bad_lines='warn')

        nrow = min(len(df), self.max_rows)
        ncol = min(len(df.columns), MAX_COLS)

        if len(df) > self.max_rows or len(df.columns) > MAX_COLS:
            print(f"  Truncated from {len(df)}x{len(df.columns)} to {nrow}x{ncol}")

        cells = []
        for i in range(nrow):
            row = []
            for j in range(ncol):
                row.append(str(df.iloc[i, j]))
            cells.append(row)

        return cells, nrow, ncol

    def _get_bert_cls_embeddings(self, cells, batch_size=64):
        """Get BERT [CLS] embedding for each cell.

        This reproduces TABBIE's original to_bert_emb() → bert_embedder.forward()
        → BertModel(...)[0][:, :, 0, :] chain: for each cell, tokenize with BERT,
        run through BERT, and extract position 0 ([CLS] token) of the last hidden state.

        Args:
            cells: list of list of str
            batch_size: Number of cells to process in a single BERT forward pass

        Returns:
            (1, nrow*ncol, 768) tensor on self.device
        """
        nrow = len(cells)
        ncol = len(cells[0]) if cells else 0

        # Flatten cells to a list for batched processing
        flat_cells = []
        for row in cells:
            for cell in row:
                flat_cells.append(cell if cell else " ")  # Empty cells → single space

        # Process in batches
        all_cls = []
        for start in range(0, len(flat_cells), batch_size):
            batch_texts = flat_cells[start : start + batch_size]

            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_CELL_TOKENS,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            # BERT forward — extract [CLS] at position 0
            with torch.no_grad():
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                cls_emb = outputs.last_hidden_state[:, 0, :]  # (batch, 768)

            all_cls.append(cls_emb)

        # Concatenate and reshape: (nrow*ncol, 768) → (1, nrow*ncol, 768)
        all_cls = torch.cat(all_cls, dim=0)  # (nrow*ncol, 768)
        return all_cls.unsqueeze(0)  # (1, nrow*ncol, 768)

    def csv_to_embeddings(self, csv_path, aggregate="cls", output_format="numpy"):
        """Extract table or column embeddings from a CSV file.

        Args:
            csv_path: Path to CSV file
            aggregate: 'cls' (table-level CLS intersection) or
                       'column' (per-column embeddings + CLS)
            output_format: 'numpy' or 'tensor'

        Returns:
            If aggregate='cls': (768,) np.float32 array or tensor
            If aggregate='column': dict with 'column_embeddings' and 'cls_embedding'
        """
        # Read and truncate CSV
        cells, nrow, ncol = self._read_csv(csv_path)

        if nrow == 0 or ncol == 0:
            raise ValueError(f"Empty table: {csv_path}")

        # Get per-cell BERT [CLS] embeddings
        bert_embs = self._get_bert_cls_embeddings(cells)  # (1, nrow*ncol, 768)

        # Run through TABBIE's dual transformers
        with torch.no_grad():
            row_embs, col_embs = self.model.get_tabemb(bert_embs, nrow, ncol)

        if aggregate == "column":
            nrow_cls = nrow + 1
            ncol_cls = ncol + 1
            # col_embs is flat: (1, nrow_cls*ncol_cls, 768)
            # Reshape to grid: (1, nrow_cls, ncol_cls, 768)
            col_grid = col_embs.view(1, nrow_cls, ncol_cls, self.model.hidden_dim)
            # CLS row (row 0), data columns (j+1 for j in 0..ncol-1)
            col_vectors = col_grid[0, 0, 1:, :]  # (ncol, 768)
            # CLS table embedding: average of row and col CLS intersections
            cls_emb = (row_embs[:, 0, :] + col_embs[:, 0, :]) / 2.0  # (1, 768)

            if output_format == "numpy":
                return {
                    "column_embeddings": {
                        j: col_vectors[j].cpu().numpy().astype(np.float32)
                        for j in range(ncol)
                    },
                    "cls_embedding": cls_emb.squeeze(0).cpu().numpy().astype(np.float32),
                }
            else:
                return {
                    "column_embeddings": {j: col_vectors[j] for j in range(ncol)},
                    "cls_embedding": cls_emb.squeeze(0),
                }

        # Default: aggregate='cls'
        # Extract table embedding: CLS intersection = (row[0,0] + col[0,0]) / 2
        # After CLS prepend, the grid is (nrow+1) x (ncol+1).
        # Position (0,0) in the flat representation is index 0.
        table_emb = (row_embs[:, 0, :] + col_embs[:, 0, :]) / 2.0  # (1, 768)

        # Squeeze to (768,)
        table_emb = table_emb.squeeze(0)  # (768,)

        if output_format == "numpy":
            return table_emb.cpu().numpy().astype(np.float32)
        else:
            return table_emb


    def _read_csv_all_rows(self, csv_path):
        """Read CSV with all rows but truncated columns.

        Unlike _read_csv() which truncates rows to self.max_rows (for the
        full-grid case), this reads ALL rows — suitable for row-by-row
        mini-table processing where each mini-table is only 2 rows.

        Returns:
            cells: list of list of str (header + all data rows, columns truncated)
            nrow: number of rows (including header)
            ncol: number of columns (truncated to MAX_COLS)
        """
        try:
            df = pd.read_csv(csv_path, header=None, keep_default_na=False, dtype=str)
        except Exception:
            df = pd.read_csv(csv_path, header=None, keep_default_na=False, dtype=str,
                             engine='python', on_bad_lines='warn')

        nrow = len(df)
        ncol = min(len(df.columns), MAX_COLS)

        if len(df.columns) > MAX_COLS:
            print(f"  Columns truncated from {len(df.columns)} to {ncol}")

        cells = []
        for i in range(nrow):
            row = []
            for j in range(ncol):
                row.append(str(df.iloc[i, j]))
            cells.append(row)

        return cells, nrow, ncol

    def csv_to_row_embeddings(self, csv_path, row_batch_size=32, output_format="numpy"):
        """Extract per-row embeddings via batched header+row mini-table processing.

        For each data row, creates a 2-row mini-table (header + data row),
        runs through TABBIE, and extracts the CLS intersection as that row's
        embedding.

        Unlike csv_to_embeddings(), this processes ALL data rows (no row
        truncation) since each mini-table is only 2 rows and always fits
        within the positional embedding limits.

        Batching strategy (k = row_batch_size):
          - Header cell BERT [CLS] embeddings computed once, reused for every row.
          - Data row cell BERT embeddings batched (k rows' cells in one BERT pass).
          - k mini-tables batched in a single get_tabemb() call (batch dim = k).

        Args:
            csv_path: Path to CSV file
            row_batch_size: Number of rows to process in each batch
            output_format: 'numpy' or 'tensor'

        Returns:
            (n_data_rows, 768) np.float32 array (if numpy) or tensor
        """
        cells, nrow, ncol = self._read_csv_all_rows(csv_path)
        if nrow < 2:
            raise ValueError(f"Need header + at least 1 data row: {csv_path}")

        header_cells = cells[0]
        data_rows = cells[1:]

        # Pre-compute header BERT [CLS] embeddings — computed once
        header_bert = self._get_bert_cls_embeddings([header_cells])  # (1, ncol, 768)

        all_row_embs = []
        for batch_start in range(0, len(data_rows), row_batch_size):
            batch_rows = data_rows[batch_start : batch_start + row_batch_size]
            k = len(batch_rows)

            # Batched BERT encoding for k data rows' cells
            data_bert = self._get_bert_cls_embeddings(batch_rows)  # (1, k*ncol, 768)
            data_bert = data_bert.view(k, ncol, self.model.hidden_dim)  # (k, ncol, 768)

            # Expand header: (k, ncol, 768)
            header_expanded = header_bert.expand(k, -1, -1)

            # Each mini-table: header + 1 data row → (k, 2*ncol, 768)
            mini_tables = torch.cat([header_expanded, data_bert], dim=1)

            # Batched TABBIE forward: k mini-tables at once
            with torch.no_grad():
                row_embs, col_embs = self.model.get_tabemb(mini_tables, nrow=2, ncol=ncol)
            # CLS intersection for each mini-table: (k, 768)
            cls_embs = (row_embs[:, 0, :] + col_embs[:, 0, :]) / 2.0
            all_row_embs.append(cls_embs)

        # Stack: (n_data_rows, 768)
        result = torch.cat(all_row_embs, dim=0)
        if output_format == "numpy":
            return result.cpu().numpy().astype(np.float32)
        return result


def main():
    parser = argparse.ArgumentParser(description="Extract TABBIE table embeddings from CSV")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to weights.pt")
    parser.add_argument("--device_id", type=int, default=None, help="GPU device ID (None=auto, -1=CPU)")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save embedding (.npy)")
    parser.add_argument("--bert_model_name", type=str, default="bert-base-uncased",
                        help="BERT model name or local path (default: bert-base-uncased)")
    args = parser.parse_args()

    embedder = TABBIEEmbedder(args.model_path, device_id=args.device_id,
                              bert_model_name=args.bert_model_name)
    emb = embedder.csv_to_embeddings(args.csv_path)

    print(f"Embedding shape: {emb.shape}, dtype: {emb.dtype}")
    print(f"Norm: {np.linalg.norm(emb):.4f}")

    if args.output_path:
        np.save(args.output_path, emb)
        print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
