"""TRL-EffBench: Efficiency Benchmark for Tabular Embedding Generation.

A companion suite to TRL-Bench that measures the computational cost of
materializing reusable embeddings across 21 tabular representation learning
models. Uses a thin wall-clock timer around the existing embedding generation
scripts — no custom wrappers, so measurements reflect the actual benchmark code.
"""

__version__ = "0.1.0"
