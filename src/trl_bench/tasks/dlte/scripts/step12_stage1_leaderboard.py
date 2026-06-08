"""
Step 12: Stage 1 Leaderboard — Compare 7 column models for table retrieval.

Produces:
  - experiments/stage1_leaderboard_topk_{K}.csv  (7 rows, one per model)
  - experiments/stage1_leaderboard.pdf            (grouped bar chart, all K values)
  - experiments/stage1_tier_breakdown.pdf          (per-tier recall curves)

Usage:
    python downstream_tasks/dlte/scripts/step12_stage1_leaderboard.py
    python downstream_tasks/dlte/scripts/step12_stage1_leaderboard.py --splits dev test
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Paths (resolved at runtime by resolve_paths()) ───────────────

PROJECT_ROOT = STAGE1_ROOT = EXPERIMENTS_ROOT = None

COLUMN_MODELS = ["bert", "gte", "starmie", "tabbie", "tabert", "tabsketchfm", "tapas", "turl"]
NATIVE_TABLE_MODELS = ["tapex"]
ALL_TABLE_MODELS = COLUMN_MODELS + NATIVE_TABLE_MODELS
K_VALUES = [10, 50, 100]
METRICS = ["recall_any", "recall_union", "recall_join", "mrr_any"]

# Display names
MODEL_LABELS = {
    "bert": "BERT",
    "gte": "GTE",
    "starmie": "Starmie",
    "tabbie": "TABBIE",
    "tabert": "TaBERT",
    "tabsketchfm": "TabSketchFM",
    "tapas": "TAPAS",
    "turl": "TURL",
    "tapex": "TAPEX",
}

METRIC_LABELS = {
    "recall_any": "Recall@K (Any)",
    "recall_union": "Recall@K (Union)",
    "recall_join": "Recall@K (Join)",
    "mrr_any": "MRR (Any)",
}


def resolve_paths(args):
    """Resolve project root and output paths from CLI args."""
    global PROJECT_ROOT, STAGE1_ROOT, EXPERIMENTS_ROOT
    PROJECT_ROOT = Path(args.project_root) if args.project_root else Path(__file__).resolve().parents[3]
    output_root = Path(args.output_root) if args.output_root else PROJECT_ROOT / "assets" / "evaluation_results" / "dlte"
    STAGE1_ROOT = output_root / "stage1"
    EXPERIMENTS_ROOT = output_root / "experiments"


# ── Data Loading ───────────────────────────────────────────────────

def load_stage1_metrics(split, k):
    """Load Stage 1 metrics for all table models at given split and K."""
    rows = []
    for model in ALL_TABLE_MODELS:
        path = STAGE1_ROOT / model / f"metrics_{split}_topk_{k}.json"
        if not path.exists():
            print(f"  WARN: missing {path}")
            continue
        data = json.loads(path.read_text())
        row = {
            "model": model,
            "model_label": MODEL_LABELS[model],
            "k": k,
            "split": split,
            "n_queries": data["n_queries"],
            "recall_any": data["recall_any"],
            "recall_union": data["recall_union"],
            "recall_join": data["recall_join"],
            "mrr_any": data["mrr_any"],
        }
        # Per-tier
        for tier, tier_data in data.get("per_tier", {}).items():
            row[f"recall_any_t{tier}"] = tier_data["recall_any"]
            row[f"recall_union_t{tier}"] = tier_data["recall_union"]
            row[f"recall_join_t{tier}"] = tier_data["recall_join"]

        rows.append(row)
    return pd.DataFrame(rows)


# ── Figure: Grouped bar chart ─────────────────────────────────────

def plot_leaderboard(all_dfs, split, output_path):
    """Grouped bar chart: models × metrics, one group per K value."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle(f"Stage 1: Table Retrieval Leaderboard ({split} split)",
                 fontsize=14, fontweight="bold", y=1.02)

    palette = sns.color_palette("Set2", n_colors=len(ALL_TABLE_MODELS))
    bar_metrics = ["recall_any", "recall_union", "recall_join"]

    for ax_idx, k in enumerate(K_VALUES):
        ax = axes[ax_idx]
        df = all_dfs[k].sort_values("recall_any", ascending=True)

        x = np.arange(len(df))
        width = 0.25

        for i, metric in enumerate(bar_metrics):
            bars = ax.barh(x + i * width, df[metric], width,
                           label=METRIC_LABELS[metric] if ax_idx == 0 else "",
                           alpha=0.85)

        ax.set_yticks(x + width)
        ax.set_yticklabels(df["model_label"])
        ax.set_xlabel("Recall")
        ax.set_title(f"K = {k}")
        ax.set_xlim(0, 1.05)
        ax.grid(axis="x", alpha=0.3)

    # Single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.05), fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Figure: MRR comparison ───────────────────────────────────────

def plot_mrr(all_dfs, split, output_path):
    """Bar chart of MRR@K for each model, grouped by K."""
    fig, ax = plt.subplots(figsize=(10, 5))

    models = all_dfs[K_VALUES[0]].sort_values("mrr_any", ascending=False)["model_label"].tolist()
    x = np.arange(len(models))
    width = 0.25
    palette = sns.color_palette("Blues_d", n_colors=len(K_VALUES))

    for i, k in enumerate(K_VALUES):
        df = all_dfs[k].set_index("model_label").loc[models]
        ax.bar(x + i * width, df["mrr_any"], width,
               label=f"K={k}", color=palette[i], alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("MRR")
    ax.set_title(f"Stage 1: Mean Reciprocal Rank ({split} split)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(0.6, ax.get_ylim()[1] * 1.1))

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Figure: Tier breakdown ───────────────────────────────────────

def plot_tier_breakdown(all_dfs, split, output_path):
    """Line plot: recall_any by tier for each model at K=100."""
    df = all_dfs[100]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle(f"Stage 1: Recall by Noise Tier ({split} split, K=100)",
                 fontsize=14, fontweight="bold", y=1.02)

    palette = sns.color_palette("tab10", n_colors=len(ALL_TABLE_MODELS))
    tier_metrics = ["recall_any", "recall_union", "recall_join"]
    tiers = [0, 1, 2, 3]

    for ax_idx, metric_base in enumerate(tier_metrics):
        ax = axes[ax_idx]
        for model_idx, (_, row) in enumerate(df.iterrows()):
            vals = [row.get(f"{metric_base}_t{t}", 0) for t in tiers]
            ax.plot(tiers, vals, "o-", label=row["model_label"],
                    color=palette[model_idx], linewidth=2, markersize=6)

        ax.set_xlabel("Noise Tier")
        ax.set_ylabel("Recall" if ax_idx == 0 else "")
        ax.set_title(METRIC_LABELS[metric_base])
        ax.set_xticks(tiers)
        ax.set_xlim(-0.2, 3.2)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.08), fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Step 12: Stage 1 Leaderboard")
    parser.add_argument("--splits", nargs="+", default=["dev", "test"])
    parser.add_argument("--output_root", type=str, default=None,
                        help="Root directory for DLTE results")
    parser.add_argument("--project_root", type=str, default=None,
                        help="Project root directory")
    args = parser.parse_args()
    resolve_paths(args)

    print("Step 12: Stage 1 Leaderboard")
    print("=" * 60)

    EXPERIMENTS_ROOT.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        print(f"\n  Split: {split}")

        # Load metrics for all K values
        all_dfs = {}
        for k in K_VALUES:
            df = load_stage1_metrics(split, k)
            all_dfs[k] = df

            # Save CSV
            csv_path = EXPERIMENTS_ROOT / f"stage1_leaderboard_{split}_topk_{k}.csv"
            df_out = df[["model", "model_label", "n_queries",
                         "recall_any", "recall_union", "recall_join", "mrr_any"]].copy()
            df_out = df_out.sort_values("recall_any", ascending=False)
            df_out.to_csv(csv_path, index=False)
            print(f"    CSV: {csv_path.name} ({len(df_out)} models)")

            # Print leaderboard
            print(f"\n    K={k}:")
            print(f"    {'Model':<14} {'R@K(any)':>10} {'R@K(union)':>12} {'R@K(join)':>11} {'MRR':>8}")
            print(f"    {'-'*55}")
            for _, row in df_out.iterrows():
                print(f"    {row['model_label']:<14} {row['recall_any']:>10.4f} "
                      f"{row['recall_union']:>12.4f} {row['recall_join']:>11.4f} "
                      f"{row['mrr_any']:>8.4f}")

        # Generate figures
        plot_leaderboard(all_dfs, split,
                         EXPERIMENTS_ROOT / f"stage1_leaderboard_{split}.pdf")
        plot_mrr(all_dfs, split,
                 EXPERIMENTS_ROOT / f"stage1_mrr_{split}.pdf")
        plot_tier_breakdown(all_dfs, split,
                            EXPERIMENTS_ROOT / f"stage1_tier_breakdown_{split}.pdf")

    print(f"\n{'='*60}")
    print(f"Outputs: {EXPERIMENTS_ROOT}")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
