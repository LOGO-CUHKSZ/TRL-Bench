import argparse
import numpy as np
import random
import torch
import mlflow
import sys
import os

# Make this model's own directory importable so the vendored ``sdd`` package
# (Starmie/PVLDB-2023's upstream module name) resolves as a top-level module.
# generate_column_embeddings.py does the same; run_pretrain.py was missing it,
# so `python -m trl_bench.models.starmie.run_pretrain` failed with
# ModuleNotFoundError: No module named 'sdd'.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from sdd.dataset import PretrainTableDataset
from sdd.pretrain import train, configure_mlflow_logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--checkpoint_dir", type=str, default="assets/union_search/starmie",
                        help="Root directory for saving checkpoints (a dataset-specific subfolder will be created).")
    parser.add_argument("--checkpoint_subdir", type=str, default=None,
                        help="Subdirectory under --checkpoint_dir to save the checkpoint in "
                             "(default: basename of --data_path). Pin this so the saved .pt "
                             "path is independent of the input tables-dir name.")
    parser.add_argument("--mlflow_dir", type=str, default=None,
                        help="Directory for MLflow tracking data. Defaults to <checkpoint_dir>/<dataset>/mlflow_runs.")
    parser.add_argument("--mlflow_experiment", type=str, default=None,
                        help="Name of the MLflow experiment to use (defaults to dataset-specific name).")
    parser.add_argument("--logdir", dest="deprecated_logdir", type=str, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--size", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--lm", type=str, default='roberta')
    parser.add_argument("--projector", type=int, default=768)
    parser.add_argument("--augment_op", type=str, default='drop_col,sample_row')
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    # single-column mode without table context
    parser.add_argument("--single_column", dest="single_column", action="store_true")
    # row / column-ordered for preprocessing
    parser.add_argument("--table_order", type=str, default='column')
    # for sampling
    parser.add_argument("--sample_meth", type=str, default='head')
    # max rows per table (for performance on large tables, matches extractVectors.py)
    parser.add_argument("--max_rows", type=int, default=None,
                        help="Maximum rows to read per table. Use 1000 for large datasets like OpenData.")
    # mlflow tag
    parser.add_argument("--mlflow_tag", type=str, default=None)

    hp = parser.parse_args()

    if getattr(hp, "deprecated_logdir", None):
        if hp.checkpoint_dir == parser.get_default("checkpoint_dir"):
            hp.checkpoint_dir = hp.deprecated_logdir
        else:
            print("--logdir is deprecated and ignored because --checkpoint_dir is already set.",
                  file=sys.stderr)
        delattr(hp, "deprecated_logdir")

    configure_mlflow_logging(hp)

    dataset_name = os.path.basename(os.path.normpath(hp.data_path))
    default_experiment = f"starmie_pretrain_{dataset_name}" if dataset_name else "starmie_pretrain"
    experiment_name = hp.mlflow_experiment or default_experiment
    mlflow.set_experiment(experiment_name)

    # mlflow logging
    for variable in ["batch_size", "lr", "n_epochs", "augment_op", "sample_meth", "table_order", "max_rows"]:
        mlflow.log_param(variable, getattr(hp, variable))
    mlflow.log_param("data_path", hp.data_path)

    if hp.mlflow_tag:
        mlflow.set_tag("tag", hp.mlflow_tag)

    # set seed
    seed = hp.run_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    trainset = PretrainTableDataset.from_hp(hp.data_path, hp)

    train(trainset, hp)
