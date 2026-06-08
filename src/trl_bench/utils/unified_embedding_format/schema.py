"""
Schema definitions for unified embedding format.

This module defines the standardized data structures for table embeddings
across all models in the TRL benchmark.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import numpy as np

# Version constants
UNIFIED_TABLE_VERSION = '2.0'  # Updated for new table_embedding structure
UNIFIED_BATCH_VERSION = '2.0'
UNIFIED_ROW_VERSION = '1.0'


def _array_to_list(arr: Optional[np.ndarray]) -> Optional[List[float]]:
    """Convert numpy array to list for serialization."""
    if arr is None:
        return None
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    return arr


def _list_to_array(lst: Optional[List[float]]) -> Optional[np.ndarray]:
    """Convert list to numpy array for deserialization."""
    if lst is None:
        return None
    return np.array(lst, dtype=np.float32)


@dataclass
class TableLevelEmbedding:
    """
    Container for table-level embedding variants.

    This structure holds different representations of table-level embeddings,
    distinguishing between native model outputs and aggregated embeddings.

    Attributes:
        cls_embedding: CLS token embedding if the model's CLS token is designed
            to represent the entire table (e.g., TAPAS, TabSketchFM).
            None if not supported or CLS doesn't represent the table.
        table_embedding: Native table-level embedding from the model itself.
            None if the model doesn't natively output table embeddings.
        column_mean: Mean-pooled column embeddings. Computed via aggregation
            module from column_embeddings. None if not computable.
        token_mean: Mean of all non-padding token hidden states from the
            model's last layer. Available for BERT, GTE, TAPAS, TAPEX,
            TabSketchFM. None if not supported.
    """
    cls_embedding: Optional[np.ndarray] = None
    table_embedding: Optional[np.ndarray] = None
    column_mean: Optional[np.ndarray] = None
    token_mean: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'cls_embedding': _array_to_list(self.cls_embedding),
            'table_embedding': _array_to_list(self.table_embedding),
            'column_mean': _array_to_list(self.column_mean),
            'token_mean': _array_to_list(self.token_mean),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TableLevelEmbedding':
        """Create from dictionary (deserialization)."""
        if data is None:
            return cls()
        return cls(
            cls_embedding=_list_to_array(data.get('cls_embedding')),
            table_embedding=_list_to_array(data.get('table_embedding')),
            column_mean=_list_to_array(data.get('column_mean')),
            token_mean=_list_to_array(data.get('token_mean')),
        )

    def has_any(self) -> bool:
        """Check if any embedding is present."""
        return any([
            self.cls_embedding is not None,
            self.table_embedding is not None,
            self.column_mean is not None,
            self.token_mean is not None,
        ])


@dataclass
class TableEmbeddingResult:
    """
    Unified format for single table column/table-level embeddings.

    This is the standardized output format for column/table embedding models
    like TaBERT, TAPAS, Doduo, TURL, TabSketchFM, etc.

    Attributes:
        table_id: Unique identifier (typically filename without extension)
        model_name: Name of the embedding model (e.g., 'tapas-base', 'doduo')
        embedding_dim: Embedding dimension (e.g., 768, 1024)
        column_embeddings: Dict mapping column index to embedding array
                          {col_idx: np.ndarray of shape (embedding_dim,)}
        table_embedding: TableLevelEmbedding containing various table-level
            representations (CLS, native, mean-pooled, sum-pooled)
        context_embedding: Optional context/question embedding
        column_names: List of column header names
        source_path: Original file path (optional)
        version: Format version (default: '2.0')
        format: Format identifier (default: 'unified_table_embedding')
    """
    # Required fields
    table_id: str
    model_name: str
    embedding_dim: int
    column_embeddings: Dict[int, np.ndarray]

    # Table-level embeddings (new structure in v2.0)
    table_embedding: Optional[TableLevelEmbedding] = None

    # Optional embedding fields
    context_embedding: Optional[np.ndarray] = None

    # Metadata fields
    column_names: List[str] = field(default_factory=list)
    source_path: Optional[str] = None

    # Version info
    version: str = UNIFIED_TABLE_VERSION
    format: str = 'unified_table_embedding'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            'version': self.version,
            'format': self.format,
            'table_id': self.table_id,
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'column_embeddings': {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.column_embeddings.items()
            },
            'column_names': self.column_names,
            'source_path': self.source_path,
        }

        if self.table_embedding is not None:
            result['table_embedding'] = self.table_embedding.to_dict()

        if self.context_embedding is not None:
            result['context_embedding'] = _array_to_list(self.context_embedding)

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TableEmbeddingResult':
        """Create from dictionary (deserialization)."""
        # Handle both 'column_embeddings' (plural) and legacy 'column_embedding' (singular)
        col_emb_data = data.get('column_embeddings') or data.get('column_embedding') or {}

        # Convert lists back to numpy arrays
        column_embeddings = {
            int(k): np.array(v, dtype=np.float32)
            for k, v in col_emb_data.items()
        }

        # Handle table_embedding (v2.0 format with dict, or v1.0 with array)
        table_embedding_data = data.get('table_embedding')
        if table_embedding_data is None:
            table_embedding = None
        elif isinstance(table_embedding_data, dict):
            # v2.0 format
            table_embedding = TableLevelEmbedding.from_dict(table_embedding_data)
        else:
            # v1.0 backward compatibility: array was mean-pooled
            table_embedding = TableLevelEmbedding(
                column_mean=_list_to_array(table_embedding_data)
            )
            # Also check for old cls_embedding at top level
            if 'cls_embedding' in data and data['cls_embedding'] is not None:
                table_embedding.cls_embedding = _list_to_array(data['cls_embedding'])

        context_embedding = _list_to_array(data.get('context_embedding'))

        # Robust table_id/source_path resolution (legacy + unified outputs)
        table_id = data.get('table_id') or data.get('table_name') or data.get('table') or ''
        if isinstance(table_id, str):
            if '/' in table_id or '\\' in table_id:
                table_id = Path(table_id).stem
            # Handle double extensions like .csv.gz / .csv.bz2
            if table_id.endswith('.csv'):
                table_id = table_id[:-4]

        source_path = data.get('source_path') or data.get('table')

        return cls(
            table_id=table_id,
            model_name=data.get('model_name', ''),
            embedding_dim=data.get('embedding_dim', 0),
            column_embeddings=column_embeddings,
            table_embedding=table_embedding,
            context_embedding=context_embedding,
            column_names=data.get('column_names', []),
            source_path=source_path,
            version=data.get('version', UNIFIED_TABLE_VERSION),
            format=data.get('format', 'unified_table_embedding'),
        )

    def num_columns(self) -> int:
        """Return number of columns with embeddings."""
        return len(self.column_embeddings)

    def get_table_embedding(self, variant: str = 'column_mean') -> Optional[np.ndarray]:
        """
        Get a specific table embedding variant.

        Args:
            variant: One of 'cls_embedding', 'table_embedding', 'column_mean', 'token_mean'

        Returns:
            The requested embedding or None if not available.
        """
        if self.table_embedding is None:
            return None
        return getattr(self.table_embedding, variant, None)


@dataclass
class EmbeddingBatch:
    """
    Batch format for multiple table embeddings.

    Used when processing multiple tables and storing results together.

    Attributes:
        model_name: Name of the embedding model
        embedding_dim: Embedding dimension
        results: List of TableEmbeddingResult objects
        version: Format version
        format: Format identifier
    """
    model_name: str
    embedding_dim: int
    results: List[TableEmbeddingResult] = field(default_factory=list)
    version: str = UNIFIED_BATCH_VERSION
    format: str = 'unified_batch_embedding'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'version': self.version,
            'format': self.format,
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'results': [
                # Strip version/format from individual results
                {k: v for k, v in r.to_dict().items() if k not in ('version', 'format')}
                for r in self.results
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbeddingBatch':
        """Create from dictionary (deserialization)."""
        results = [
            TableEmbeddingResult.from_dict(r)
            for r in data.get('results', [])
        ]

        return cls(
            model_name=data.get('model_name', ''),
            embedding_dim=data.get('embedding_dim', 0),
            results=results,
            version=data.get('version', UNIFIED_BATCH_VERSION),
            format=data.get('format', 'unified_batch_embedding'),
        )

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def __getitem__(self, idx) -> TableEmbeddingResult:
        return self.results[idx]

    def append(self, result: TableEmbeddingResult):
        """Add a result to the batch."""
        self.results.append(result)


@dataclass
class RowEmbeddingMetadata:
    """
    Metadata format for row-level embeddings.

    Row-level models (SCARF, VIME, SubTab, etc.) output NumPy arrays
    to separate files. This metadata accompanies them.

    Attributes:
        model_name: Name of the embedding model
        embedding_dim: Embedding dimension
        embedding_level: Always 'row' for this format
        train_samples: Number of training samples
        test_samples: Number of test samples
        has_labels: Whether labels are included
        label_column: Name of the label column (if any)
        feature_columns: List of feature column names
        checkpoint_path: Path to model checkpoint used
        generation_config: CLI arguments and generation settings
        version: Format version
        format: Format identifier
    """
    model_name: str
    embedding_dim: int
    train_samples: int = 0
    test_samples: int = 0
    has_labels: bool = False
    label_column: Optional[str] = None
    feature_columns: List[str] = field(default_factory=list)
    checkpoint_path: Optional[str] = None
    generation_config: Dict[str, Any] = field(default_factory=dict)
    embedding_level: str = 'row'
    version: str = UNIFIED_ROW_VERSION
    format: str = 'unified_row_embedding'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'version': self.version,
            'format': self.format,
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'embedding_level': self.embedding_level,
            'train_samples': self.train_samples,
            'test_samples': self.test_samples,
            'has_labels': self.has_labels,
            'label_column': self.label_column,
            'feature_columns': self.feature_columns,
            'checkpoint_path': self.checkpoint_path,
            'generation_config': self.generation_config,
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RowEmbeddingMetadata':
        """Create from dictionary (deserialization)."""
        return cls(
            model_name=data.get('model_name', ''),
            embedding_dim=data.get('embedding_dim', 0),
            train_samples=data.get('train_samples', 0),
            test_samples=data.get('test_samples', 0),
            has_labels=data.get('has_labels', False),
            label_column=data.get('label_column'),
            feature_columns=data.get('feature_columns', []),
            checkpoint_path=data.get('checkpoint_path'),
            generation_config=data.get('generation_config', {}),
            embedding_level=data.get('embedding_level', 'row'),
            version=data.get('version', UNIFIED_ROW_VERSION),
            format=data.get('format', 'unified_row_embedding'),
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'RowEmbeddingMetadata':
        """Create from JSON string."""
        import json
        return cls.from_dict(json.loads(json_str))


@dataclass
class SplitInfo:
    """Metadata for a single split within a v2.0 row embedding directory.

    Attributes:
        num_samples: Number of rows in this split.
        embeddings_file: Filename for the embeddings .npy file.
        labels_file: Filename for the labels .npy file (optional).
        row_indices_file: Filename for the row indices .npy file (optional).
    """
    num_samples: int = 0
    embeddings_file: str = ""
    labels_file: Optional[str] = None
    row_indices_file: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            'num_samples': self.num_samples,
            'embeddings_file': self.embeddings_file,
        }
        if self.labels_file is not None:
            d['labels_file'] = self.labels_file
        if self.row_indices_file is not None:
            d['row_indices_file'] = self.row_indices_file
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SplitInfo':
        return cls(
            num_samples=data.get('num_samples', 0),
            embeddings_file=data.get('embeddings_file', ''),
            labels_file=data.get('labels_file'),
            row_indices_file=data.get('row_indices_file'),
        )


UNIFIED_ROW_VERSION_V2 = '2.0'


@dataclass
class RowEmbeddingMetadataV2:
    """V2 metadata for split-aware row-level embeddings.

    Instead of fixed train/test counts, uses a ``splits`` dict that maps
    split names to ``SplitInfo`` entries describing per-split files.

    Attributes:
        model_name: Name of the embedding model.
        embedding_dim: Embedding dimension.
        embedding_level: Always 'row'.
        label_columns: Label column name(s) used.
        feature_columns: Feature column names.
        splits: Mapping from split name to file info.
        generation_config: CLI arguments and generation settings.
        dataset: Dataset provenance (path, fingerprint, etc.).
        checkpoint_path: Path to model checkpoint used (if any).
        version: Format version ('2.0').
        format: Format identifier ('unified_row_embedding').
    """
    model_name: str
    embedding_dim: int = 0
    embedding_level: str = 'row'
    label_columns: List[str] = field(default_factory=list)
    label_task_types: Dict[str, str] = field(default_factory=dict)
    label_filename_map: Dict[str, str] = field(default_factory=dict)
    feature_columns: List[str] = field(default_factory=list)
    splits: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    generation_config: Dict[str, Any] = field(default_factory=dict)
    dataset: Dict[str, Any] = field(default_factory=dict)
    checkpoint_path: Optional[str] = None
    version: str = UNIFIED_ROW_VERSION_V2
    format: str = 'unified_row_embedding'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'version': self.version,
            'format': self.format,
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'embedding_level': self.embedding_level,
            'label_columns': self.label_columns,
            'label_task_types': self.label_task_types,
            'label_filename_map': self.label_filename_map,
            'feature_columns': self.feature_columns,
            'splits': self.splits,
            'generation_config': self.generation_config,
            'dataset': self.dataset,
            'checkpoint_path': self.checkpoint_path,
        }

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RowEmbeddingMetadataV2':
        return cls(
            model_name=data.get('model_name', ''),
            embedding_dim=data.get('embedding_dim', 0),
            embedding_level=data.get('embedding_level', 'row'),
            label_columns=data.get('label_columns', []),
            label_task_types=data.get('label_task_types', {}),
            label_filename_map=data.get('label_filename_map', {}),
            feature_columns=data.get('feature_columns', []),
            splits=data.get('splits', {}),
            generation_config=data.get('generation_config', {}),
            dataset=data.get('dataset', {}),
            checkpoint_path=data.get('checkpoint_path'),
            version=data.get('version', UNIFIED_ROW_VERSION_V2),
            format=data.get('format', 'unified_row_embedding'),
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'RowEmbeddingMetadataV2':
        import json
        return cls.from_dict(json.loads(json_str))
