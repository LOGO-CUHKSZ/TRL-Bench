"""Local overrides for slurm orchestration.

Override these constants/functions if your environment differs from the
defaults. Public-release default is identity / passthrough: configs are
returned as-is, job names are truncated to 64 chars (Slurm limit), env
setup strings pass through unchanged.

Downstream sites in generate_scripts.py / generate_downstream_scripts.py
import this as ``local`` and call ``local.apply()``, ``local.job_name()``,
``local.env_setup()``, ``local.partition()``, and
``local.append_extra_slurm_directives()`` as cluster hooks. Replace any of
these in a site-local fork to inject cluster-specific tweaks (account
prefix, partition rename, extra `#SBATCH` lines) without touching the
generator source.
"""
from pathlib import Path

# Root for the cloned repo. Override if running from a different location.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where dataset files live on the local filesystem.
# Default: HuggingFace cache. Override only if you have a local mirror.
DATA_ROOT = None  # None => use HF datasets cache

# Where to write per-job results. Default: <PROJECT_ROOT>/results.
RESULTS_ROOT = PROJECT_ROOT / "results"

# Slurm partition / account / time limits. Override per-cluster.
SLURM_PARTITION = "gpu"
SLURM_ACCOUNT = ""
SLURM_TIME = "12:00:00"
SLURM_GPUS = 1


def apply(config):
    """Apply local overrides to a parsed YAML config. Default: identity.

    Override to mutate the config dict before it reaches the generator
    (e.g., rewrite a partition name cluster-wide).
    """
    return config


def job_name(name: str) -> str:
    """Sanitize / truncate a slurm job name. Slurm caps at ~64 chars."""
    return str(name)[:64]


def env_setup(setup: str) -> str:
    """Return the bash env-setup line to inject in templates. Default:
    passthrough."""
    return setup


def partition(name: str) -> str:
    """Map a logical partition name to the cluster-specific name.
    Default: identity."""
    return name


def append_extra_slurm_directives(directives, gpu_requested: bool = False):
    """Hook for adding extra ``#SBATCH`` lines (qos, constraint, etc).
    Mutates ``directives`` in-place. Default: no-op."""
    return directives
