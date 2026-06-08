"""TRL-EffBench specification: workload definitions, scaling levels, result schema,
and model-to-script mapping for the existing embedding generation pipelines.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Workload definitions
# ---------------------------------------------------------------------------

class Workload(enum.Enum):
    COLUMN = "column"
    TABLE = "table"
    ROW = "row"


class ModelFamily(enum.Enum):
    TEXT_ENCODER = "text_encoder"
    TABLE_AWARE_LM = "table_aware_lm"
    STRUCTURE_AWARE = "structure_aware"
    COLUMN_SPECIALIZED = "column_specialized"
    META_PRETRAINED = "meta_pretrained"
    SELF_SUPERVISED = "self_supervised"
    TRANSFER = "transfer"
    API = "api"


# ---------------------------------------------------------------------------
# Model registry: maps model_name to metadata + script info
#
# Each entry:
#   family          - ModelFamily enum
#   workloads       - list of Workload enums this model supports
#   dim             - embedding dimension
#   needs_training  - whether the model requires per-table training
#   scripts         - dict mapping workload -> {script, input_arg, output_arg, ...}
#     input_arg:  the CLI flag for input path (e.g. "--input", "--input_dir", "--data_dir")
#     output_arg: the CLI flag for output path (e.g. "--output", "--output_path", "--embedding_dir")
#     extra_args: list of additional CLI arguments needed
#     needs_checkpoint_dir: if True, adds --checkpoint_base_dir
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # --- Text Encoders (frozen, inference only) ---
    "bert": {
        "family": ModelFamily.TEXT_ENCODER,
        "workloads": [Workload.COLUMN, Workload.TABLE, Workload.ROW],
        "dim": 768, "needs_training": False,
        "scripts": {
            Workload.COLUMN: {"script": "models/bert/generate_column_embeddings.py",
                              "input_arg": "--input", "output_arg": "--output"},
            Workload.ROW:    {"script": "models/bert/generate_row_embeddings.py",
                              "input_arg": "--input_dir", "output_arg": "--output_path"},
            Workload.TABLE:  {"script": "models/bert/generate_column_embeddings.py",
                              "input_arg": "--input", "output_arg": "--output"},
        },
    },
    "gte": {
        "family": ModelFamily.TEXT_ENCODER,
        "workloads": [Workload.COLUMN, Workload.TABLE, Workload.ROW],
        "dim": 768, "needs_training": False,
        "scripts": {
            Workload.COLUMN: {"script": "models/gte/generate_column_embeddings.py",
                              "input_arg": "--input", "output_arg": "--output"},
            Workload.ROW:    {"script": "models/gte/generate_row_embeddings.py",
                              "input_arg": "--input_dir", "output_arg": "--output_path"},
            Workload.TABLE:  {"script": "models/gte/generate_column_embeddings.py",
                              "input_arg": "--input", "output_arg": "--output"},
        },
    },
    # --- Table-Aware LMs (frozen) ---
    "tapas": {
        "family": ModelFamily.TABLE_AWARE_LM,
        "workloads": [Workload.COLUMN, Workload.TABLE],
        "dim": 768, "needs_training": False,
        "scripts": {
            Workload.COLUMN: {"script": "models/tapas/generate_column_embeddings.py",
                              "input_arg": "--input", "output_arg": "--output"},
            Workload.TABLE:  {"script": "models/tapas/generate_column_embeddings.py",
                              "input_arg": "--input", "output_arg": "--output"},
        },
    },
    "tabert": {
        "family": ModelFamily.TABLE_AWARE_LM,
        "workloads": [Workload.COLUMN, Workload.TABLE],
        "dim": 768, "needs_training": False,
        "env_setup": "source models/tabert/load_env",
        "scripts": {
            Workload.COLUMN: {"script": "models/tabert/generate_column_embeddings.py",
                              "input_arg": "--input", "output_arg": "--output",
                              "extra_args": ["--checkpoint", "checkpoints/tabert/tabert_base_k3/model.bin"]},
            Workload.TABLE:  {"script": "models/tabert/generate_column_embeddings.py",
                              "input_arg": "--input", "output_arg": "--output",
                              "extra_args": ["--checkpoint", "checkpoints/tabert/tabert_base_k3/model.bin"]},
        },
    },
    "tapex": {
        "family": ModelFamily.TABLE_AWARE_LM,
        "workloads": [Workload.TABLE],
        "dim": 768, "needs_training": False,
        "scripts": {
            Workload.TABLE: {"script": "models/tapex/generate_table_embeddings.py",
                             "input_arg": "--input_dir", "output_arg": "--output_path"},
        },
    },
    # --- Structure-Aware (frozen) ---
    "tabbie": {
        "family": ModelFamily.STRUCTURE_AWARE,
        "workloads": [Workload.COLUMN, Workload.TABLE, Workload.ROW],
        "dim": 768, "needs_training": False,
        "scripts": {
            Workload.COLUMN: {"script": "models/tabbie/generate_column_embeddings.py",
                              "input_arg": "--input", "output_arg": "--output",
                              "extra_args": ["--model_path", "checkpoints/tabbie/weights.pt"]},
            Workload.ROW:    {"script": "models/tabbie/generate_row_embeddings.py",
                              "input_arg": "--input", "output_arg": "--output",
                              "extra_args": ["--model_path", "checkpoints/tabbie/weights.pt"]},
            Workload.TABLE:  {"script": "models/tabbie/generate_column_embeddings.py",
                              "input_arg": "--input", "output_arg": "--output",
                              "extra_args": ["--model_path", "checkpoints/tabbie/weights.pt"]},
        },
    },
    "turl": {
        "family": ModelFamily.STRUCTURE_AWARE,
        "workloads": [Workload.COLUMN],
        "dim": 312, "needs_training": False,
        "scripts": {
            Workload.COLUMN: {"script": "models/turl/generate_column_embeddings_dataset.py",
                              "input_arg": "--input_dir", "output_arg": "--output_file",
                              "extra_args": ["--checkpoint", "checkpoints/turl/pretrained",
                                             "--mode", "table_directory"]},
        },
    },
    "tuta": {
        "family": ModelFamily.STRUCTURE_AWARE,
        "workloads": [Workload.TABLE, Workload.ROW],
        "dim": 768, "needs_training": False,
        "scripts": {
            Workload.TABLE: {"script": "models/tuta/generate_embeddings_directory.py",
                             "input_arg": "--input_dir", "output_arg": "--output_path",
                             "extra_args": ["--model_path", "checkpoints/tuta/tuta.bin"]},
            Workload.ROW:   {"script": "models/tuta/generate_embeddings_directory.py",
                             "input_arg": "--input_dir", "output_arg": "--output_path",
                             "extra_args": ["--model_path", "checkpoints/tuta/tuta.bin"]},
        },
    },
    # --- Column-Specialized (frozen) ---
    # NOTE: Starmie requires per-dataset pretraining (no universal checkpoint).
    # For efficiency testing, it's excluded from scale_suite runs but included
    # on eff_real anchors where a pretrained checkpoint already exists.
    "starmie": {
        "family": ModelFamily.COLUMN_SPECIALIZED,
        "workloads": [Workload.COLUMN, Workload.TABLE],
        "dim": 768, "needs_training": True,  # Mark as training since it pretrains per-dataset
        "requires_pretrained_checkpoint": True,  # Skip when no checkpoint exists
        "scripts": {
            Workload.COLUMN: {"script": "models/starmie/generate_column_embeddings.py",
                              "input_arg": "--input_dir", "output_arg": "--output_path"},
            Workload.TABLE:  {"script": "models/starmie/generate_column_embeddings.py",
                              "input_arg": "--input_dir", "output_arg": "--output_path"},
        },
    },
    "tabsketchfm": {
        "family": ModelFamily.COLUMN_SPECIALIZED,
        "workloads": [Workload.COLUMN],
        "dim": 768, "needs_training": False,
        "env_setup": "source venv_tabsketchfm/bin/activate",
        "scripts": {
            Workload.COLUMN: {"script": "models/tabsketchfm/generate_column_embeddings.py",
                              "input_arg": "--input", "output_arg": "--output",
                              "extra_args": ["--checkpoint", "checkpoints/tabsketchfm/epoch=10-step=27786.ckpt"]},
        },
    },
    # --- Meta-Pretrained Row (fit + embed per dataset) ---
    # NOTE: requires_dataset_json=True means these only run on datasets with dataset.json + splits
    "tabpfn": {
        "family": ModelFamily.META_PRETRAINED,
        "workloads": [Workload.ROW],
        "dim": 192, "needs_training": False,
        "requires_dataset_json": True,
        "scripts": {
            Workload.ROW: {"script": "models/TabPFN/generate_embeddings_train_test.py",
                           "input_arg": "--data_dir", "output_arg": "--embedding_dir",
                           "output_kind": "dir"},
        },
    },
    "tabicl": {
        "family": ModelFamily.META_PRETRAINED,
        "workloads": [Workload.ROW],
        "dim": 512, "needs_training": False,
        "requires_dataset_json": True,
        "scripts": {
            Workload.ROW: {"script": "models/TabICL/generate_embeddings_train_test.py",
                           "input_arg": "--data_dir", "output_arg": "--embedding_dir",
                           "output_kind": "dir"},
        },
    },
    # --- Self-Supervised Row (train + embed per table, directory mode) ---
    "scarf": {
        "family": ModelFamily.SELF_SUPERVISED,
        "workloads": [Workload.ROW], "dim": 512, "needs_training": True,
        "scripts": {Workload.ROW: {"script": "models/scarf/generate_embeddings_directory.py",
                                   "input_arg": "--input_dir", "output_arg": "--output_path",
                                   "needs_checkpoint_dir": True}},
    },
    "dae": {
        "family": ModelFamily.SELF_SUPERVISED,
        "workloads": [Workload.ROW], "dim": 512, "needs_training": True,
        "scripts": {Workload.ROW: {"script": "models/dae/generate_embeddings_directory.py",
                                   "input_arg": "--input_dir", "output_arg": "--output_path",
                                   "needs_checkpoint_dir": True}},
    },
    "saint": {
        "family": ModelFamily.SELF_SUPERVISED,
        "workloads": [Workload.ROW], "dim": 512, "needs_training": True,
        "scripts": {Workload.ROW: {"script": "models/saint/generate_embeddings_directory.py",
                                   "input_arg": "--input_dir", "output_arg": "--output_path",
                                   "needs_checkpoint_dir": True}},
    },
    "subtab": {
        "family": ModelFamily.SELF_SUPERVISED,
        "workloads": [Workload.ROW], "dim": 512, "needs_training": True,
        "scripts": {Workload.ROW: {"script": "models/subtab/generate_embeddings_directory.py",
                                   "input_arg": "--input_dir", "output_arg": "--output_path",
                                   "needs_checkpoint_dir": True}},
    },
    "tabtransformer": {
        "family": ModelFamily.SELF_SUPERVISED,
        "workloads": [Workload.ROW], "dim": 512, "needs_training": True,
        "scripts": {Workload.ROW: {"script": "models/tabtransformer/generate_embeddings_directory.py",
                                   "input_arg": "--input_dir", "output_arg": "--output_path",
                                   "needs_checkpoint_dir": True}},
    },
    "tabular_binning": {
        "family": ModelFamily.SELF_SUPERVISED,
        "workloads": [Workload.ROW], "dim": 512, "needs_training": True,
        "scripts": {Workload.ROW: {"script": "models/tabular_binning/generate_embeddings_directory.py",
                                   "input_arg": "--input_dir", "output_arg": "--output_path",
                                   "needs_checkpoint_dir": True}},
    },
    "vime": {
        "family": ModelFamily.SELF_SUPERVISED,
        "workloads": [Workload.ROW], "dim": 512, "needs_training": True,
        "scripts": {Workload.ROW: {"script": "models/vime/generate_embeddings_directory.py",
                                   "input_arg": "--input_dir", "output_arg": "--output_path",
                                   "needs_checkpoint_dir": True}},
    },
    # --- Transfer (train + embed per table, directory mode) ---
    "transtab": {
        "family": ModelFamily.TRANSFER,
        "workloads": [Workload.ROW], "dim": 512, "needs_training": True,
        "scripts": {Workload.ROW: {"script": "models/transtab/generate_embeddings_directory.py",
                                   "input_arg": "--input_dir", "output_arg": "--output_path",
                                   "needs_checkpoint_dir": True}},
    },
}


# ---------------------------------------------------------------------------
# Controlled scaling suite: sweep levels
# ---------------------------------------------------------------------------

ROW_BASELINE = {
    "n_rows": 10_000,
    "n_features": 32,
    "cat_share": 0.5,
    "cat_cardinality": 32,
    "missingness": 0.1,
}

ROW_SWEEPS = {
    "n_rows":          [500, 1_000, 5_000, 10_000, 50_000, 100_000],
    "n_features":      [8, 16, 32, 64, 128, 256],
    "cat_share":       [0.0, 0.25, 0.5, 0.75, 1.0],
    "cat_cardinality": [4, 16, 32, 64, 128],
    "missingness":     [0.0, 0.05, 0.1, 0.2, 0.3],
}

COL_BASELINE = {
    "n_columns": 16,
    "n_context_rows": 16,
    "avg_cell_tokens": 4,
    "type_mix": "mixed",
}

COL_SWEEPS = {
    "n_columns":       [4, 8, 16, 32, 64, 128],
    "n_context_rows":  [1, 4, 16, 32, 64],
    "avg_cell_tokens": [1, 2, 4, 8, 16, 32],
    "type_mix":        ["numeric", "mixed", "text"],
}


# ---------------------------------------------------------------------------
# Result schema (simplified for wall-clock timing approach)
# ---------------------------------------------------------------------------

@dataclass
class EffBenchResult:
    """Result for one (model, workload, dataset) timed run."""
    model_name: str = ""
    workload: str = ""
    dataset_id: str = ""
    dataset_source: str = ""     # "eff_real" | "eff_scale" | "bridge"

    # What was measured
    needs_training: bool = False  # Does this run include training?
    script: str = ""              # The actual script that was run

    # Timing
    wall_clock_seconds: float = 0.0
    timeout_seconds: int = 0

    # Memory (peak during run, polled via nvidia-smi)
    peak_gpu_vram_mb: float = 0.0
    vram_monitor_ok: bool = False  # Did nvidia-smi polling succeed at least once?

    # Dataset stats
    n_rows: int = 0
    n_columns: int = 0

    # Output verification
    output_verified: bool = False
    expected_output: str = ""

    # Run metadata
    hardware: str = ""
    gpu_name: str = ""
    device: str = ""
    hostname: str = ""
    slurm_job_id: str = ""
    slurm_array_task_id: str = ""
    return_code: int = -1
    status: str = "success"      # "success" | "oom" | "timeout" | "error"
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()
