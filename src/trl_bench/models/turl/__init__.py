"""TURL wrapper package.

Vendored TURL ships its own ``code/`` subpackage, which is shadowed by Python's
stdlib ``code.py`` module when the runners under this package are invoked via
``python -m trl_bench.models.turl.<runner>``. Without the path hack below,
``from code.model.configuration import TableConfig`` resolves to the stdlib
``code`` module and fails with ``'code' is not a package``.

Prepending this directory to ``sys.path`` ensures the local ``code/`` package
wins import resolution for any runner under ``trl_bench.models.turl.*``.
Surfaced by the Layer-2 smoke matrix's TURL cell.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
