"""Aggregate results and generate figures/tables for TRL-EffBench.

Usage::

    python -m effbench.analyze --results-dir effbench/results --output-dir effbench/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from effbench.spec import MODEL_REGISTRY, ModelFamily


# ---------------------------------------------------------------------------
# Color and style maps
# ---------------------------------------------------------------------------

FAMILY_COLORS = {
    "text_encoder": "#1f77b4",
    "table_aware_lm": "#ff7f0e",
    "structure_aware": "#2ca02c",
    "column_specialized": "#d62728",
    "meta_pretrained": "#9467bd",
    "self_supervised": "#8c564b",
    "transfer": "#e377c2",
    "api": "#7f7f7f",
}

FAMILY_LABELS = {
    "text_encoder": "Text Encoder",
    "table_aware_lm": "Table-Aware LM",
    "structure_aware": "Structure-Aware",
    "column_specialized": "Column-Specialized",
    "meta_pretrained": "Meta-Pretrained",
    "self_supervised": "Self-Supervised",
    "transfer": "Transfer",
    "api": "API",
}


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

def load_results(results_dir: Path) -> pd.DataFrame:
    """Load all JSON result files into a DataFrame."""
    records = []
    for f in results_dir.glob("*.json"):
        with open(f) as fp:
            data = json.load(fp)

        record = {
            "model": data.get("model_name", ""),
            "workload": data.get("workload", ""),
            "dataset_id": data.get("dataset_id", ""),
            "dataset_source": data.get("dataset_source", ""),
            "needs_training": data.get("needs_training", False),
            "wall_clock": data.get("wall_clock_seconds", 0),
            "peak_gpu_mb": data.get("peak_gpu_vram_mb", 0),
            "n_rows": data.get("n_rows", 0),
            "n_columns": data.get("n_columns", 0),
            "status": data.get("status", ""),
            "hardware": data.get("hardware", ""),
        }

        # Look up family
        info = MODEL_REGISTRY.get(record["model"], {})
        record["family"] = info.get("family", ModelFamily.TEXT_ENCODER).value if info else ""

        records.append(record)

    return pd.DataFrame(records)


def parse_sweep_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Extract sweep factor and level from dataset_id for scale suite results."""
    df = df.copy()

    def _parse(dataset_id: str):
        # e.g., "row_sweep_n_rows_10000" -> ("n_rows", 10000)
        # e.g., "col_sweep_n_columns_32" -> ("n_columns", 32)
        if "_sweep_" not in dataset_id:
            return None, None
        # Split: prefix_sweep_factor_level
        parts = dataset_id.split("_sweep_", 1)
        if len(parts) != 2:
            return None, None
        remainder = parts[1]  # e.g., "n_rows_10000" or "type_mix_numeric"
        # Try to split off the last token as the level
        tokens = remainder.rsplit("_", 1)
        if len(tokens) == 2:
            factor, level_str = tokens
            try:
                level = float(level_str)
            except ValueError:
                level = level_str
            return factor, level
        return remainder, None

    parsed = df["dataset_id"].apply(_parse)
    df["sweep_factor"] = [p[0] for p in parsed]
    df["sweep_level"] = [p[1] for p in parsed]
    return df


# ---------------------------------------------------------------------------
# Figure 1: Time comparison across models (bar chart per workload)
# ---------------------------------------------------------------------------

def plot_time_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart of median wall-clock time per model, by workload."""
    for wl in df["workload"].unique():
        subset = df[(df["workload"] == wl) & (df["status"] == "success")]
        if subset.empty:
            continue

        avg = subset.groupby("model").agg(
            time_median=("wall_clock", "median"),
            time_iqr_lo=("wall_clock", lambda x: x.quantile(0.25)),
            time_iqr_hi=("wall_clock", lambda x: x.quantile(0.75)),
        ).sort_values("time_median")

        fig, ax = plt.subplots(figsize=(max(8, len(avg) * 0.6), 5))

        colors = []
        for model in avg.index:
            info = MODEL_REGISTRY.get(model, {})
            fam = info.get("family", ModelFamily.TEXT_ENCODER).value if info else ""
            colors.append(FAMILY_COLORS.get(fam, "#999999"))

        yerr_lo = avg["time_median"] - avg["time_iqr_lo"]
        yerr_hi = avg["time_iqr_hi"] - avg["time_median"]

        ax.barh(range(len(avg)), avg["time_median"], color=colors,
                xerr=[yerr_lo.values, yerr_hi.values],
                capsize=3, edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(avg)))
        ax.set_yticklabels(avg.index, fontsize=8)
        ax.set_xlabel("Wall-clock time (s)")
        ax.set_xscale("log")
        ax.set_title(f"Embedding Generation Time: {wl.capitalize()} Workload")
        ax.grid(True, alpha=0.3, axis="x")

        # Add training indicator
        for i, model in enumerate(avg.index):
            info = MODEL_REGISTRY.get(model, {})
            if info.get("needs_training"):
                ax.annotate(" (train+embed)", (avg.loc[model, "time_median"], i),
                           fontsize=6, va="center", alpha=0.6)

        plt.tight_layout()
        fig.savefig(output_dir / f"time_comparison_{wl}.pdf", bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  Saved time_comparison_{wl}.pdf")


# ---------------------------------------------------------------------------
# Figure 2: Scaling curves from controlled sweeps
# ---------------------------------------------------------------------------

def plot_scaling_curves(df: pd.DataFrame, output_dir: Path) -> None:
    """Log-log plots of runtime vs scaling factor."""
    sweeps = df[df["dataset_source"] == "eff_scale"].copy()
    if sweeps.empty:
        print("  No scaling data found, skipping.")
        return

    sweeps = parse_sweep_metadata(sweeps)
    sweeps = sweeps[sweeps["status"] == "success"]

    for factor in sweeps["sweep_factor"].dropna().unique():
        factor_data = sweeps[sweeps["sweep_factor"] == factor]
        # Only plot if levels are numeric
        numeric_data = factor_data[factor_data["sweep_level"].apply(lambda x: isinstance(x, (int, float)))]
        if numeric_data.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        for model, group in numeric_data.groupby("model"):
            group = group.sort_values("sweep_level")
            ax.plot(group["sweep_level"], group["wall_clock"],
                    marker="o", label=model, linewidth=1.5, markersize=4)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(factor.replace("_", " ").title())
        ax.set_ylabel("Wall-clock time (s)")
        ax.set_title(f"Scaling: {factor.replace('_', ' ').title()}")
        ax.legend(fontsize=7, ncol=2, loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(output_dir / f"scaling_{factor}.pdf", bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  Saved scaling_{factor}.pdf")


# ---------------------------------------------------------------------------
# Figure 3: GPU VRAM comparison
# ---------------------------------------------------------------------------

def plot_vram_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart of peak GPU VRAM per model."""
    for wl in df["workload"].unique():
        subset = df[(df["workload"] == wl) & (df["status"] == "success")]
        if subset.empty:
            continue

        avg = subset.groupby("model")["peak_gpu_mb"].median().sort_values()

        fig, ax = plt.subplots(figsize=(max(8, len(avg) * 0.6), 5))
        colors = []
        for model in avg.index:
            info = MODEL_REGISTRY.get(model, {})
            fam = info.get("family", ModelFamily.TEXT_ENCODER).value if info else ""
            colors.append(FAMILY_COLORS.get(fam, "#999999"))

        ax.barh(range(len(avg)), avg.values, color=colors,
                edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(avg)))
        ax.set_yticklabels(avg.index, fontsize=8)
        ax.set_xlabel("Peak GPU VRAM (MB)")
        ax.set_title(f"Peak GPU Memory: {wl.capitalize()} Workload")
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        fig.savefig(output_dir / f"vram_{wl}.pdf", bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  Saved vram_{wl}.pdf")


# ---------------------------------------------------------------------------
# Figure 4: Support heatmap (success/OOM/error at each scale)
# ---------------------------------------------------------------------------

def plot_support_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Heatmap of which models succeed at which dataset scales."""
    sweeps = df[df["dataset_source"] == "eff_scale"].copy()
    if sweeps.empty:
        return

    sweeps = parse_sweep_metadata(sweeps)

    for factor in ["n_rows", "n_features", "n_columns"]:
        factor_data = sweeps[sweeps["sweep_factor"] == factor]
        if factor_data.empty:
            continue

        numeric_data = factor_data[factor_data["sweep_level"].apply(lambda x: isinstance(x, (int, float)))]
        if numeric_data.empty:
            continue

        pivot = numeric_data.pivot_table(
            index="model", columns="sweep_level",
            values="status", aggfunc="first",
        )

        status_map = {"success": 1, "oom": 0, "error": -1, "timeout": -0.5}
        pivot_num = pivot.map(lambda x: status_map.get(x, -1) if isinstance(x, str) else -1)

        fig, ax = plt.subplots(figsize=(8, max(4, len(pivot_num) * 0.35)))
        im = ax.imshow(pivot_num.values, aspect="auto", cmap=plt.cm.RdYlGn, vmin=-1, vmax=1)

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{int(c):,}" for c in pivot.columns], fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_xlabel(factor.replace("_", " ").title())
        ax.set_title(f"Support Heatmap: {factor}")
        plt.colorbar(im, ax=ax, shrink=0.6)
        plt.tight_layout()
        fig.savefig(output_dir / f"support_{factor}.pdf", bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  Saved support_{factor}.pdf")


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def generate_latex_table(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate LaTeX summary tables of median timings per model."""
    for wl in df["workload"].unique():
        subset = df[(df["workload"] == wl) & (df["status"] == "success")]
        if subset.empty:
            continue

        summary = subset.groupby("model").agg(
            time_median=("wall_clock", "median"),
            time_iqr=("wall_clock", lambda x: x.quantile(0.75) - x.quantile(0.25)),
            vram_median=("peak_gpu_mb", "median"),
            n_datasets=("dataset_id", "count"),
        ).sort_values("time_median")

        latex_path = output_dir / f"table_{wl}.tex"
        with open(latex_path, "w") as f:
            f.write(r"\begin{table}[t]" + "\n")
            f.write(r"\centering" + "\n")
            f.write(r"\caption{Efficiency results: " + wl + r" embedding generation (median over datasets).}" + "\n")
            f.write(r"\label{tab:eff_" + wl + r"}" + "\n")
            f.write(r"\resizebox{\linewidth}{!}{" + "\n")
            f.write(r"\begin{tabular}{llrrr}" + "\n")
            f.write(r"\toprule" + "\n")
            f.write(r"Model & Type & Time (s) & VRAM (MB) & Datasets \\" + "\n")
            f.write(r"\midrule" + "\n")
            for model, row in summary.iterrows():
                info = MODEL_REGISTRY.get(model, {})
                mtype = "train+infer" if info.get("needs_training") else "inference"
                f.write(f"{model} & {mtype} & "
                        f"{row['time_median']:.2f} $\\pm$ {row['time_iqr']:.2f} & "
                        f"{row['vram_median']:.0f} & {int(row['n_datasets'])} \\\\\n")
            f.write(r"\bottomrule" + "\n")
            f.write(r"\end{tabular}}" + "\n")
            f.write(r"\end{table}" + "\n")

        print(f"  Saved {latex_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze TRL-EffBench results")
    parser.add_argument("--results-dir", type=str,
                        default=str(PROJECT_ROOT / "effbench" / "results"))
    parser.add_argument("--output-dir", type=str,
                        default=str(PROJECT_ROOT / "effbench" / "figures"))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    df = load_results(results_dir)
    if df.empty:
        print("No results found. Run experiments first.")
        return

    print(f"  {len(df)} measurements")
    print(f"  Models: {sorted(df['model'].unique())}")
    print(f"  Workloads: {sorted(df['workload'].unique())}")
    print(f"  Success: {(df['status'] == 'success').sum()}/{len(df)}")

    print("\nGenerating figures...")
    plot_time_comparison(df, output_dir)
    plot_scaling_curves(df, output_dir)
    plot_vram_comparison(df, output_dir)
    plot_support_heatmap(df, output_dir)

    print("\nGenerating LaTeX tables...")
    generate_latex_table(df, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
