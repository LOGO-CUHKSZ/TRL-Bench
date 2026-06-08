"""Tests for the canonical join-search query-disjoint split generator.

``trl_bench.tasks.join_search.splits.generate_canonical_split`` reproduces the
paper's fixed 20/80 query-role-disjoint split for the learned-projection join
search probe (join_search_learned). The split is a deterministic function of
(ground_truth, curated query-list, seed=42, train_ratio=0.2): preprocess the GT
(basename, drop self-pairs, dedup, intersect with the query list), then
partition unique query keys with random.Random(42).
"""
from __future__ import annotations
from pathlib import Path

import pytest

from trl_bench.tasks.join_search.splits import (
    preprocess_gt,
    split_by_query,
    generate_canonical_split,
)


def _write_csv(path: Path, header: str, rows: list[str]) -> None:
    path.write_text(header + "\n" + "\n".join(rows) + "\n")


def test_generate_canonical_split_deterministic_and_disjoint(tmp_path):
    """Split is deterministic (seed) and query-role-disjoint: no query key
    appears in both train and test; self-pairs + dups are dropped first."""
    gt = tmp_path / "ground_truth.csv"
    # 10 clean queries (q0..q9) each with one candidate, + 1 self-pair + 1 dup.
    rows = [f"t{i}.csv,c{i}.csv,q{i},c{i}" for i in range(10)]
    rows.append("tS.csv,tS.csv,qS,cS")            # self-pair -> dropped
    rows.append("t0.csv,c0.csv,q0,c0")            # exact dup of first -> dropped
    _write_csv(gt, "query_table,candidate_table,query_column,candidate_column", rows)

    # Query list covers all 10 clean queries (+ the self-pair query, which is
    # removed by self-pair filtering regardless).
    ql = tmp_path / "queries.csv"
    _write_csv(ql, "query_table,query_column",
               [f"t{i}.csv,q{i}" for i in range(10)] + ["tS.csv,qS"])

    out = tmp_path / "splits" / "join_search"
    info = generate_canonical_split(
        ground_truth_path=gt, query_list_path=ql, out_dir=out,
        seed=42, train_ratio=0.2,
    )

    # 10 clean queries survive preprocessing; 20% -> 2 train, 8 test.
    assert info["train_queries"] == 2
    assert info["test_queries"] == 8
    assert info["seed"] == 42 and info["train_ratio"] == 0.2

    # Files written with the canonical headers.
    train_q = (out / "train_queries.csv").read_text().splitlines()
    test_q = (out / "test_queries.csv").read_text().splitlines()
    test_gt = (out / "test_gt.csv").read_text().splitlines()
    assert train_q[0] == "query_table,query_column"
    assert test_q[0] == "query_table,query_column"
    assert test_gt[0] == "query_table,candidate_table,query_column,candidate_column"
    assert len(train_q) == 1 + 2
    assert len(test_q) == 1 + 8

    # Disjoint train/test query keys; self-pair + dup excluded everywhere.
    train_keys = {tuple(l.split(",")) for l in train_q[1:]}
    test_keys = {tuple(l.split(",")) for l in test_q[1:]}
    assert train_keys.isdisjoint(test_keys)
    assert ("tS.csv", "qS") not in (train_keys | test_keys)
    # test_gt only references test queries.
    for line in test_gt[1:]:
        qt, _ct, qc, _cc = line.split(",")
        assert (qt, qc) in test_keys

    # Determinism: identical re-run yields identical files.
    out2 = tmp_path / "splits2"
    generate_canonical_split(ground_truth_path=gt, query_list_path=ql,
                             out_dir=out2, seed=42, train_ratio=0.2)
    assert (out2 / "train_queries.csv").read_text() == (out / "train_queries.csv").read_text()


def test_preprocess_gt_drops_self_pairs_and_dups(tmp_path):
    gt = tmp_path / "gt.csv"
    _write_csv(gt, "query_table,candidate_table,query_column,candidate_column", [
        "a.csv,b.csv,qa,cb",
        "a.csv,b.csv,qa,cb",     # dup
        "x.csv,x.csv,qx,cx",     # self-pair
        "p/a.csv,q/b.csv,qa,cb", # basenamed -> same as row 1 after basename+dedup
    ])
    ql = tmp_path / "ql.csv"
    _write_csv(ql, "query_table,query_column", ["a.csv,qa", "x.csv,qx"])
    df = preprocess_gt(str(gt), str(ql))
    # Only the single unique (a.csv,b.csv,qa,cb) row survives.
    assert len(df) == 1
    assert set(zip(df["query_table"], df["query_column"])) == {("a.csv", "qa")}
