"""
Semantic parsing task framework.

Supports multiple semantic parsing tasks (WikiTableQuestions, Spider, etc.)
with swappable decoders (MAPO, Transformer, etc.)
"""

from . import tasks
from . import decoders
from . import execution
from . import evaluation
from . import data
