"""Step 11 Aggregate: Combine per-combination metrics into global summary.

Discovers all combo directories under metrics/ via glob (handles both coupled
and cross-model combos) and aggregates their summary.csv files.

Usage:
    python downstream_tasks/dlte/scripts/step11_aggregate.py
    python downstream_tasks/dlte/scripts/step11_aggregate.py --output_root results/evaluation/dlte/cls_embedding
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Resolved at runtime by resolve_paths()
PROJECT_ROOT = METRICS_ROOT = None


def resolve_paths(args):
    """Resolve project root and output paths from CLI args."""
    global PROJECT_ROOT, METRICS_ROOT
    PROJECT_ROOT = Path(args.project_root) if args.project_root else Path(__file__).resolve().parents[3]
    output_root = Path(args.output_root) if args.output_root else PROJECT_ROOT / "assets" / "evaluation_results" / "dlte"
    METRICS_ROOT = output_root / "metrics"


def _fmt(v):
    """Format a metric value, handling NaN/None gracefully."""
    return f"{v:>8.4f}" if pd.notna(v) else "     N/A"


def _print_leaderboard(global_df, split, has_table_model):
    """Print a CellF1 leaderboard for the given split."""
    f1_col = f"cell_f1_{split}"
    rr_col = f"parent_row_recall_{split}"
    cr_col = f"parent_col_recall_{split}"
    if f1_col not in global_df.columns:
        return

    print(f"\n{'='*70}")
    print(f"CellF1 Leaderboard ({split} split)")
    print(f"{'='*70}")
    if has_table_model:
        print(f"{'Table Model':<14} {'Col Model':<14} {'Row Model':<10} {'CellF1':>8} {'RowR':>8} {'ColR':>8}")
        print("-" * 64)
    else:
        print(f"{'Col Model':<14} {'Row Model':<10} {'CellF1':>8} {'RowR':>8} {'ColR':>8}")
        print("-" * 50)

    for _, row in global_df.sort_values(f1_col, ascending=False, na_position='last').iterrows():
        if has_table_model:
            print(f"{row.get('table_model', ''):<14} {row['col_model']:<14} {row['row_model']:<10} "
                  f"{_fmt(row[f1_col])} {_fmt(row[rr_col])} {_fmt(row[cr_col])}")
        else:
            print(f"{row['col_model']:<14} {row['row_model']:<10} "
                  f"{_fmt(row[f1_col])} {_fmt(row[rr_col])} {_fmt(row[cr_col])}")


def main():
    parser = argparse.ArgumentParser(description="Step 11 Aggregate")
    parser.add_argument("--output_root", type=str, default=None,
                        help="Root directory for DLTE results")
    parser.add_argument("--project_root", type=str, default=None,
                        help="Project root directory")
    args = parser.parse_args()
    resolve_paths(args)

    print("Step 11 Aggregate: Combining per-combination metrics")
    print("=" * 60)

    all_summaries = []

    if METRICS_ROOT.exists():
        for combo_dir in sorted(METRICS_ROOT.iterdir()):
            if not combo_dir.is_dir():
                continue
            summary_path = combo_dir / "summary.csv"
            if summary_path.exists():
                df = pd.read_csv(summary_path)
                all_summaries.append(df.iloc[0].to_dict())

    if all_summaries:
        global_df = pd.DataFrame(all_summaries)
        global_path = METRICS_ROOT / "all_combinations_summary.csv"
        global_df.to_csv(global_path, index=False)
        print(f"\nGlobal summary: {global_path} ({len(all_summaries)} combinations)")

        has_table_model = "table_model" in global_df.columns and (global_df["table_model"] != global_df["col_model"]).any()
        _print_leaderboard(global_df, "dev", has_table_model)
        _print_leaderboard(global_df, "test", has_table_model)

    print(f"\n{'='*60}")
    print(f"Done: {len(all_summaries)} combinations found")
    print(f"{'='*60}")

    return 0 if all_summaries else 1


if __name__ == "__main__":
    sys.exit(main())
