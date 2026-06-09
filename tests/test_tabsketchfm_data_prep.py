"""Regression test for the TabSketchFM CSV reader segfault on giant-exponent
ID strings (e.g. ``1E1032981008100``).

Root cause: ``read_table_from_original`` reads CSVs with pandas' default type
inference (``engine='c'``). When an otherwise-string ID column contains a token
that *looks* like scientific notation with an astronomically large exponent
(``1E1032981008100`` == 1 x 10^1032981008100), pandas' C float-parsing path
overflows and **segfaults** (SIGSEGV, rc=-11) instead of falling back to an
object column. This is unrecoverable in-process (a segfault bypasses the
existing try/except in ``read_table_from_original``), so it crashed Stage-1
table-embedding extraction for ``tabsketchfm x ckan_subset`` on the real table
``sdmr-tfmf.csv.1.neg.csv``.

The faithful fix keeps default per-column inference for every well-behaved
column and only coerces the pathological column(s) to string -- which is exactly
what a non-buggy parser (e.g. pandas' ``engine='pyarrow'``) produces for that
column anyway.

Because the bug manifests as a process-level SIGSEGV, the read is exercised in a
child process so that on unfixed code the test observes the crash (negative
return code) rather than taking the whole pytest run down with it.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


# The minimal pathological table: a string-ID column ("IUM") whose values mix
# normal IDs with tokens that look like floats carrying a >308-digit-magnitude
# exponent. A second column is genuinely numeric and must stay numeric.
_PATHOLOGICAL_CSV = textwrap.dedent(
    """\
    ,IUM,QTY
    0,1E1032981008100,49000.0
    1,1A1000281000101,95000.0
    2,1C1029821000100,12000.0
    """
)


def _run_reader_in_subprocess(csv_path: Path) -> subprocess.CompletedProcess:
    """Invoke read_table_from_original on csv_path in a fresh interpreter.

    Isolating the read means a SIGSEGV in pandas shows up as a negative
    returncode here instead of killing the pytest process.
    """
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    code = textwrap.dedent(
        f"""
        import sys, faulthandler
        faulthandler.enable()
        sys.path.insert(0, {str(src)!r})
        from trl_bench.models.tabsketchfm.tabsketchfm.data_processing.data_prep import (
            read_table_from_original,
        )
        res = read_table_from_original({str(csv_path)!r})
        assert res is not None, "reader returned None"
        df = res[0][0]
        assert df is not None, "reader produced a None dataframe"
        # The pathological column must survive as a (string/object) column with
        # its original tokens preserved -- NOT silently dropped or coerced to inf.
        assert "IUM" in df.columns, f"IUM missing: {{list(df.columns)}}"
        ium = [str(v) for v in df["IUM"].tolist()]
        assert "1E1032981008100" in ium, f"giant-exponent token lost: {{ium}}"
        assert "1A1000281000101" in ium, f"mixed id token lost: {{ium}}"
        # The genuinely-numeric column must still be numeric.
        import pandas.api.types as pat
        assert pat.is_numeric_dtype(df["QTY"]), f"QTY not numeric: {{df['QTY'].dtype}}"
        print("READER_OK")
        """
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )


def test_read_table_handles_giant_exponent_id_strings(tmp_path: Path) -> None:
    csv_path = tmp_path / "giant_exp.csv"
    csv_path.write_text(_PATHOLOGICAL_CSV, encoding="utf-8")

    proc = _run_reader_in_subprocess(csv_path)

    # A SIGSEGV surfaces as returncode == -11 (negative => killed by signal).
    assert proc.returncode >= 0, (
        f"reader crashed with signal (rc={proc.returncode}); "
        f"giant-exponent ID string segfaulted pandas.\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert proc.returncode == 0 and "READER_OK" in proc.stdout, (
        f"reader did not produce a faithful dataframe (rc={proc.returncode}).\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


def test_read_table_normal_inference_unchanged(tmp_path: Path) -> None:
    """The guard must be a no-op for well-behaved tables: numeric columns stay
    numeric and string columns stay object, exactly as default inference."""
    csv_path = tmp_path / "normal.csv"
    csv_path.write_text(
        "id,name,score\n0,alpha,1.5\n1,beta,2.5\n2,gamma,3.0\n",
        encoding="utf-8",
    )
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    code = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(src)!r})
        import pandas.api.types as pat
        from trl_bench.models.tabsketchfm.tabsketchfm.data_processing.data_prep import (
            read_table_from_original,
        )
        df = read_table_from_original({str(csv_path)!r})[0][0]
        assert pat.is_integer_dtype(df["id"]), df["id"].dtype
        # pandas >=3.0 infers a dedicated ``str`` dtype for text columns (was
        # ``object`` in 2.x); accept either -- the guarantee is "not coerced to numeric".
        assert pat.is_object_dtype(df["name"]) or pat.is_string_dtype(df["name"]), df["name"].dtype
        assert pat.is_float_dtype(df["score"]), df["score"].dtype
        print("NORMAL_OK")
        """
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0 and "NORMAL_OK" in proc.stdout, (
        f"rc={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
