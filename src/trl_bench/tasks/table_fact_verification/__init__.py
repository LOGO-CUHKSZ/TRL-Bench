"""
Table Fact Verification (TabFact) Downstream Task

This module implements the TabFact task as a downstream task that uses
frozen embeddings from any table embedding model (TAPAS, TaBERT, Doduo, etc.)

Task: Binary classification - given a table and statement, classify as:
- ENTAILED (1): Statement is supported by the table
- REFUTED (0): Statement is contradicted by the table

Dataset: TabFact - 117,854 statements, 16,573 Wikipedia tables
Paper: Chen et al. "TabFact: A Large-scale Dataset for Table-based Fact Verification" (ICLR 2020)

Benchmarks:
- TAPAS (finetuned): 81.0%
- Human: 92.1%
- Random baseline: 50.0%

Usage:
    1. Download dataset:
       python download_tabfact.py --output_dir datasets/tabfact

    2. Generate embeddings:
       python generate_embeddings.py --model tapas --data_dir datasets/tabfact \
           --output_file embeddings/tabfact/tapas_base.pkl

    3. Train classifier:
       python train.py --train_embeddings embeddings/tabfact/tapas/train.pkl \
           --val_embeddings embeddings/tabfact/tapas/validation.pkl \
           --output_dir checkpoints/tabfact/tapas

    4. Evaluate:
       python evaluate.py --model_checkpoint checkpoints/tabfact/tapas/best_model.pt \
           --test_embeddings embeddings/tabfact/tapas/test.pkl
"""

from .download_tabfact import download_tabfact
from .train import TabFactClassifier, LinearClassifier
from .evaluate import evaluate_model
