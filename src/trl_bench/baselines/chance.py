"""Analytical chance baselines for all downstream tasks.

Computes the expected performance of a random model for each task/dataset
combination, producing results in the same dict format as the aggregator.
"""

from __future__ import annotations

import csv
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml


# ── Math helpers ──────────────────────────────────────────────────────────


def harmonic_number(n: int) -> float:
    """H(n) = sum(1/i for i in 1..n)."""
    return sum(1.0 / i for i in range(1, n + 1))


def chance_hit_at_k(k: int, n: int) -> float:
    """Chance Hit@K for single-relevant-doc retrieval: K/N."""
    return k / n


def chance_mrr_at_k(k: int, n: int) -> float:
    """Chance MRR@K for single-relevant-doc retrieval: H(min(K,N))/N."""
    return harmonic_number(min(k, n)) / n


def chance_precision_multi(k: int, r: int, n: int) -> float:
    """Chance precision when r relevant items exist among N: r/N."""
    return r / n


def chance_recall_multi(k: int, r: int, n: int) -> float:
    """Chance recall when drawing K from N with r relevant: min(K/N, 1)."""
    return min(k / n, 1.0)


def chance_recall_at_gt(gt_size: int, m: int, n: int) -> float:
    """Chance recall@GT for schema matching: gt_size/(m*n)."""
    return gt_size / (m * n)


# ── Result dict builder ──────────────────────────────────────────────────


def _base_result(task: str, dataset: str, **metrics) -> dict:
    """Build a result dict in the aggregator's format."""
    result = {
        "task": task,
        "model": "chance",
        "dataset": dataset,
        "seed": None,
        "variant": None,
        "head_type": None,
        "retrieval_mode": None,
        "status": "analytical",
    }
    result.update(metrics)
    return result


# ── Per-task functions ────────────────────────────────────────────────────


def _chance_table_retrieval(project_root: Path) -> list[dict]:
    """Chance baselines for table_retrieval (nq_tables)."""
    N = 169885
    K = 100

    metrics = {}
    for k in range(1, K + 1):
        metrics[f"Recall@{k}"] = chance_hit_at_k(k, N)
    metrics["MRR"] = chance_mrr_at_k(K, N)

    return [_base_result("table_retrieval", "nq_tables", **metrics)]


def _chance_join_search(project_root: Path) -> list[dict]:
    """Chance baselines for join_search (opendata variants)."""
    stats_path = Path(__file__).parent / "dataset_stats.yaml"
    with open(stats_path) as f:
        stats = yaml.safe_load(f)

    col_counts = stats["join_search"]["datalake_column_counts"]
    datasets = ["opendata_main", "opendata_can", "opendata_usa", "opendata_uk_sg"]
    K = 10
    results = []

    for ds in datasets:
        N = col_counts.get(ds)
        if N is None:
            print(f"[chance] Skipping join_search/{ds}: no column count in stats")
            continue

        gt_path = project_root / f"datasets/{ds}/gt/opendata_join_ground_truth.csv"
        if not gt_path.exists():
            print(f"[chance] Skipping join_search/{ds}: {gt_path} not found")
            continue

        # Count GT rows per (query_table, query_column) pair
        query_counts: dict[tuple[str, str], int] = defaultdict(int)
        with open(gt_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["query_table"], row["query_column"])
                query_counts[key] += 1

        r_values = list(query_counts.values())

        col_prec = np.mean([r / N for r in r_values])
        col_recall = np.mean([min(K / N, 1.0) for _ in r_values])
        if col_prec + col_recall > 0:
            col_f1 = 2 * col_prec * col_recall / (col_prec + col_recall)
        else:
            col_f1 = 0.0
        col_map = np.mean([r / N for r in r_values])

        results.append(_base_result(
            "join_search", ds,
            col_precision_at_10=float(col_prec),
            col_recall_at_10=float(col_recall),
            col_f1_at_10=float(col_f1),
            col_map=float(col_map),
            # Legacy keys
            precision_at_10=float(col_prec),
            recall_at_10=float(col_recall),
            f1_at_10=float(col_f1),
            map=float(col_map),
        ))

    return results


def _chance_union_search(project_root: Path) -> list[dict]:
    """Chance baselines for union_search (santos)."""
    gt_path = project_root / "datasets/santos/santosUnionBenchmark.pickle"
    if not gt_path.exists():
        print(f"[chance] Skipping union_search/santos: {gt_path} not found")
        return []

    with open(gt_path, "rb") as f:
        gt = pickle.load(f)

    N = 550
    K = 10

    # Starmie MAP: mean over queries of r_q / N
    r_values = [len(v) for v in gt.values()]
    map_at_k = float(np.mean([r / N for r in r_values]))
    prec_at_k = float(np.mean([r / N for r in r_values]))
    recall_at_k = float(np.mean([min(K / N, 1.0) for _ in r_values]))

    return [_base_result(
        "union_search", "santos",
        map_at_k=map_at_k,
        precision_at_k=prec_at_k,
        recall_at_k=recall_at_k,
    )]


def _chance_schema_matching(project_root: Path) -> list[dict]:
    """Chance baselines for schema_matching (valentine)."""
    pairs_path = project_root / "datasets/valentine/pairs.json"
    gt_path = project_root / "datasets/valentine/ground_truth.csv"
    tables_dir = project_root / "datasets/valentine/tables"

    if not pairs_path.exists():
        print(f"[chance] Skipping schema_matching/valentine: {pairs_path} not found")
        return []
    if not gt_path.exists():
        print(f"[chance] Skipping schema_matching/valentine: {gt_path} not found")
        return []

    with open(pairs_path) as f:
        pairs = json.load(f)

    # Load GT counts per pair_id
    gt_counts: dict[str, int] = Counter()
    with open(gt_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_counts[row["pair_id"]] += 1

    recall_values = []
    for pair in pairs:
        pair_id = pair["pair_id"]
        table_a_path = tables_dir / pair["table_a"]
        table_b_path = tables_dir / pair["table_b"]

        if not table_a_path.exists() or not table_b_path.exists():
            continue

        # Count columns in each table
        with open(table_a_path, newline="", errors="replace") as f:
            reader = csv.reader(f)
            header_a = next(reader, None)
            m = len(header_a) if header_a else 0

        with open(table_b_path, newline="", errors="replace") as f:
            reader = csv.reader(f)
            header_b = next(reader, None)
            n = len(header_b) if header_b else 0

        if m == 0 or n == 0:
            continue

        gt_size = gt_counts.get(pair_id, 0)
        if gt_size == 0:
            continue

        recall_values.append(chance_recall_at_gt(gt_size, m, n))

    if not recall_values:
        print("[chance] schema_matching/valentine: no valid pairs found")
        return []

    recall_at_gt = float(np.mean(recall_values))
    return [_base_result("schema_matching", "valentine", recall_at_gt=recall_at_gt)]


def _chance_column_clustering(project_root: Path) -> list[dict]:
    """Chance baselines for column_clustering (sato, SOTAB)."""
    datasets = ["sato", "sotab"]
    results = []

    for ds in datasets:
        labels_path = project_root / f"datasets/{ds}/all.csv"
        if not labels_path.exists():
            print(f"[chance] Skipping column_clustering/{ds}: {labels_path} not found")
            continue

        class_counts: dict[str, int] = Counter()
        with open(labels_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                class_counts[row["class"]] += 1

        total = sum(class_counts.values())
        largest = max(class_counts.values())
        purity = largest / total

        results.append(_base_result(
            "column_clustering", ds,
            purity=float(purity),
            nmi=0.0,
            ari=0.0,
        ))

    return results


def _chance_supervised_classification(
    project_root: Path, task: str, dataset: str, labels_path: Path
) -> list[dict]:
    """Chance baselines for binary/multi-class classification tasks.

    Predicts the majority class from train split on the test split.
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    if not labels_path.exists():
        print(f"[chance] Skipping {task}/{dataset}: {labels_path} not found")
        return []

    with open(labels_path) as f:
        data = json.load(f)

    train_labels = [item["label"] for item in data["train"]]
    test_labels = [item["label"] for item in data["test"]]

    # Majority class from training set
    label_counts = Counter(train_labels)
    majority_class = label_counts.most_common(1)[0][0]

    y_true = test_labels
    y_pred = [majority_class] * len(test_labels)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    precision = precision_score(y_true, y_pred, average="binary", zero_division=0)
    recall = recall_score(y_true, y_pred, average="binary", zero_division=0)

    return [_base_result(
        task, dataset,
        test_results_accuracy=float(accuracy),
        test_results_f1=float(f1),
        test_results_macro_f1=float(macro_f1),
        test_results_weighted_f1=float(weighted_f1),
        test_results_precision=float(precision),
        test_results_recall=float(recall),
    )]


def _chance_table_fact_verification(project_root: Path) -> list[dict]:
    """Chance baselines for table_fact_verification (tabfact)."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    train_path = project_root / "datasets/tabfact/train.jsonl"
    test_path = project_root / "datasets/tabfact/test.jsonl"

    if not train_path.exists():
        print(f"[chance] Skipping table_fact_verification: {train_path} not found")
        return []
    if not test_path.exists():
        print(f"[chance] Skipping table_fact_verification: {test_path} not found")
        return []

    # Read train labels
    train_labels = []
    with open(train_path) as f:
        for line in f:
            obj = json.loads(line)
            train_labels.append(obj["label"])

    # Read test labels
    test_labels = []
    with open(test_path) as f:
        for line in f:
            obj = json.loads(line)
            test_labels.append(obj["label"])

    # Majority class from training
    label_counts = Counter(train_labels)
    majority_class = label_counts.most_common(1)[0][0]

    y_true = test_labels
    y_pred = [majority_class] * len(test_labels)

    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)

    return [_base_result(
        "table_fact_verification", "tabfact",
        accuracy=float(accuracy),
        f1_macro=float(f1_macro),
        f1_weighted=float(f1_weighted),
        precision_macro=float(precision_macro),
        recall_macro=float(recall_macro),
        head_type="mlp",
    )]


def _chance_supervised_regression(
    project_root: Path, task: str, dataset: str, labels_path: Path
) -> list[dict]:
    """Chance baselines for regression tasks.

    Predicts the training mean on the test split, then computes R², MSE, MAE.
    """
    if not labels_path.exists():
        print(f"[chance] Skipping {task}/{dataset}: {labels_path} not found")
        return []

    with open(labels_path) as f:
        data = json.load(f)

    # Extract labels — field might be 'label' or 'containment_fraction'
    def _get_label(item: dict) -> float:
        if "label" in item:
            return float(item["label"])
        elif "containment_fraction" in item:
            return float(item["containment_fraction"])
        raise KeyError(f"No label field found in {list(item.keys())}")

    train_labels = [_get_label(item) for item in data["train"]]
    test_labels = [_get_label(item) for item in data["test"]]

    y_true = np.array(test_labels)
    train_mean = np.mean(train_labels)
    y_pred = np.full_like(y_true, train_mean)

    # R² = 1 - SS_res / SS_tot
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    return [_base_result(
        task, dataset,
        test_results_r2=float(r2),
        test_results_mse=mse,
        test_results_mae=mae,
    )]


# ── Public API ────────────────────────────────────────────────────────────


def compute_chance_baselines(
    project_root: Path, tasks: list | None = None
) -> list[dict]:
    """Compute analytical chance baselines for downstream tasks.

    Parameters
    ----------
    project_root : Path
        Root directory of the project (where datasets/ lives).
    tasks : list or None
        If given, only compute baselines for these tasks.
        If None, compute all available baselines.

    Returns
    -------
    list[dict]
        One dict per (task, dataset) with metric values.
    """
    # Load task_datasets config to discover classification/regression tasks
    config_path = project_root / "slurm/config/downstream/task_datasets.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    task_datasets = config["task_datasets"]

    results: list[dict] = []

    # Map of task -> handler
    task_handlers: dict[str, callable] = {}

    # 1. Table retrieval
    def _do_table_retrieval():
        return _chance_table_retrieval(project_root)
    task_handlers["table_retrieval"] = _do_table_retrieval

    # 2. Join search
    def _do_join_search():
        return _chance_join_search(project_root)
    task_handlers["join_search"] = _do_join_search

    # 3. Union search
    def _do_union_search():
        return _chance_union_search(project_root)
    task_handlers["union_search"] = _do_union_search

    # 4. Schema matching
    def _do_schema_matching():
        return _chance_schema_matching(project_root)
    task_handlers["schema_matching"] = _do_schema_matching

    # 5. Column clustering
    def _do_column_clustering():
        return _chance_column_clustering(project_root)
    task_handlers["column_clustering"] = _do_column_clustering

    # 6. Table fact verification
    def _do_table_fact_verification():
        return _chance_table_fact_verification(project_root)
    task_handlers["table_fact_verification"] = _do_table_fact_verification

    # 7. Classification tasks: table_subset, join_classification,
    #    union_classification, record_linkage
    classification_tasks = {
        "table_subset": {
            "ckan_subset": project_root / "datasets/ckan_subset/labels.json",
        },
        "join_classification": {
            "spider_join": project_root / "datasets/spider_join/spider-join/labels.json",
        },
        "union_classification": {
            "wiki_union": project_root / "datasets/wiki_union/labels.json",
        },
    }

    # Record linkage: all 16 datasets
    if "record_linkage" in task_datasets:
        rl_datasets = {}
        for ds_name in task_datasets["record_linkage"]["datasets"]:
            rl_datasets[ds_name] = (
                project_root / f"datasets/record_linkage/{ds_name}/labels.json"
            )
        classification_tasks["record_linkage"] = rl_datasets

    for task_name, ds_map in classification_tasks.items():
        def _make_handler(tn, dm):
            def _handler():
                sub_results = []
                for ds, lpath in dm.items():
                    sub_results.extend(
                        _chance_supervised_classification(project_root, tn, ds, lpath)
                    )
                return sub_results
            return _handler
        task_handlers[task_name] = _make_handler(task_name, ds_map)

    # 8. Regression tasks: join_containment, union_regression
    regression_tasks = {
        "join_containment": {
            "wiki_containment": project_root / "datasets/wiki_containment/labels.json",
        },
        "union_regression": {
            "ecb_union": project_root / "datasets/ecb_union/labels.json",
        },
    }

    for task_name, ds_map in regression_tasks.items():
        def _make_handler(tn, dm):
            def _handler():
                sub_results = []
                for ds, lpath in dm.items():
                    sub_results.extend(
                        _chance_supervised_regression(project_root, tn, ds, lpath)
                    )
                return sub_results
            return _handler
        task_handlers[task_name] = _make_handler(task_name, ds_map)

    # Run all requested tasks
    for task_name, handler in task_handlers.items():
        if tasks is not None and task_name not in tasks:
            continue
        try:
            task_results = handler()
            results.extend(task_results)
        except Exception as e:
            print(f"[chance] WARNING: {task_name} failed: {e}")

    return results
