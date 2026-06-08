"""
TabSketchFM: Sketch-based Tabular Representation Learning for Data Discovery

A foundation model for tabular data that uses MinHash sketches and BERT-based
architecture for pretraining and finetuning on data discovery tasks.

Paper: TabSketchFM: Sketch-based Tabular Representation Learning for Data Discovery over Data Lakes

NOTE
----
The vendored inner package (this directory's ``tabsketchfm/`` subpackage) uses
*absolute* imports of the form ``from tabsketchfm.<sub> import ...`` (inherited
from the upstream IBM repository). To resolve those without a separate
top-level install, we prepend our own directory to ``sys.path`` so the inner
``tabsketchfm/`` directory is importable as a top-level package. Matches the
tuta/tabbie pattern.

We deliberately do NOT eagerly re-export symbols from the inner subpackage at
this top-level ``__init__.py`` — running ``python -m
trl_bench.models.tabsketchfm.generate_column_embeddings`` first triggers
package import, and pulling unused subpackages here would force every probe
dispatch to import all of pytorch-lightning. Callers that need the public
symbols can import them via ``trl_bench.models.tabsketchfm.tabsketchfm.<sym>``
or, once ``_THIS_DIR`` is on ``sys.path``, via ``tabsketchfm.<sym>``.
"""

import os as _os
import sys as _sys

_THIS_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _THIS_DIR not in _sys.path:
    _sys.path.insert(0, _THIS_DIR)

__version__ = "1.0.0"
