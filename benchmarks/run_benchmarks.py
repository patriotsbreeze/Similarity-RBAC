"""
run_benchmarks.py — Master benchmark runner
============================================
Runs all three VLDB experiments in sequence and generates publication-quality
figures.

Usage
-----
  python benchmarks/run_benchmarks.py [--quick] [--n N] [--nq NQ]

  --quick   Use small N/NQ for a fast sanity check (N=50k, NQ=200)
  --n       Number of database vectors (default 200k; paper uses 1M)
  --nq      Number of query vectors (default 2k; paper uses 10k)
"""

from __future__ import annotations

import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

RESULTS_DIR = ROOT / "results"
PLOTS_DIR   = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour palette (consistent with IEEE/VLDB figures) ────────────────────────
PALETTE = {
    "RBAC-HNSW (filter)":       "#2196F3",   # blue
    "RBAC-HNSW (routing)":      "#4CAF50",   # green
    "Post-filter baseline":     "#F44336",   # red
    "Pre-filter (brute-force)": "#FF9800",   # orange
}
MARKERS = {
    "RBAC-HNSW (filter)":       "o",
    "RBAC-HNSW (routing)":      "s",
    "Post-filter baseline":     "^",
    "Pre-filter (brute-force)": "D",
}


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Recall@10 vs. Selectivity  (one subplot per ef value)
# ─────────────────────────────────────────────────────────────────────────────

def plot_recall_vs_selectivity(df: pd.DataFrame, ef: int = 200) -> None:
    sub = df[df["ef"] == ef].copy()
    sel_order = ["ultra", "strict", "restricted", "medium", "open"]
    sel_pct   = {r["selectivity_name"]: r["selectivity_frac"] * 100
                 for _, r in sub.iterrows()}

    fig, ax = plt.subplots(figsize=(7, 4))
    for method in sub["method"].unique():
        mdf = sub[sub["method"] == method].copy()
        mdf["sel_pct"] = mdf["selectivity_name"].map(sel_pct)
        mdf = mdf.sort_values("sel_pct")
        ax.plot(mdf["sel_pct"], mdf["recall_at_k"],
                marker=MARKERS.get(method, "o"),
                color=PALETTE.get(method, "grey"),
                label=method, linewidth=2, markersize=7)

    ax.set_xscale("log")
    ax.set_xlabel("Accessible fraction of database (%)", fontsize=12)
    ax.set_ylabel("Recall@10", fontsize=12)
    ax.set_title(f"Recall@10 vs. Selectivity  (ef={ef})", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.9, color="grey", linestyle="--", linewidth=1, alpha=0.6,
               label="0.90 recall threshold")
    ax.legend(fontsize=9, loc="lower right")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = PLOTS_DIR / f"fig1_recall_vs_selectivity_ef{ef}.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


def plot_recall_ef_curves(df: pd.DataFrame) -> None:
    """Recall vs. ef for each selectivity level (one subplot each)."""
    sel_names = df["selectivity_name"].unique()
    fig, axes = plt.subplots(1, len(sel_names), figsize=(4 * len(sel_names), 4),
                             sharey=True)
    if len(sel_names) == 1:
        axes = [axes]

    for ax, sel_name in zip(axes, sel_names):
        sub = df[df["selectivity_name"] == sel_name]
        sel_frac = sub["selectivity_frac"].iloc[0]
        for method in sub["method"].unique():
            mdf = sub[sub["method"] == method].sort_values("ef")
            ax.plot(mdf["ef"], mdf["recall_at_k"],
                    marker=MARKERS.get(method, "o"),
                    color=PALETTE.get(method, "grey"),
                    label=method, linewidth=2, markersize=6)
        ax.set_title(f"{sel_name}\n({sel_frac*100:.1f}%)", fontsize=10)
        ax.set_xlabel("ef", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Recall@10", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.02), fontsize=9)
    fig.suptitle("Recall@10 vs. ef across Selectivity Levels", fontsize=12,
                 fontweight="bold", y=1.05)
    fig.tight_layout()
    out = PLOTS_DIR / "fig2_recall_ef_curves.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: QPS vs. Selectivity
# ─────────────────────────────────────────────────────────────────────────────

def plot_qps_vs_selectivity(df_qps: pd.DataFrame, ef: int = 200) -> None:
    sub = df_qps[df_qps["ef"].isin([ef, -1])].copy()
    sel_pct = {r["selectivity_name"]: r["selectivity_frac"] * 100
               for _, r in sub.iterrows()}

    fig, ax = plt.subplots(figsize=(7, 4))
    for method in sub["method"].unique():
        mdf = sub[sub["method"] == method].copy()
        mdf["sel_pct"] = mdf["selectivity_name"].map(sel_pct)
        mdf = mdf.sort_values("sel_pct")
        ax.plot(mdf["sel_pct"], mdf["qps"],
                marker=MARKERS.get(method, "o"),
                color=PALETTE.get(method, "grey"),
                label=method, linewidth=2, markersize=7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Accessible fraction of database (%)", fontsize=12)
    ax.set_ylabel("Throughput (QPS)", fontsize=12)
    ax.set_title(f"Throughput vs. Selectivity  (ef={ef})", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = PLOTS_DIR / f"fig3_qps_vs_selectivity_ef{ef}.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Memory comparison vs. SIEVE
# ─────────────────────────────────────────────────────────────────────────────

def plot_memory_comparison(df_theo: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    archs = df_theo["architecture"].tolist()
    gb    = [b / 1e9 for b in df_theo["total_bytes"]]
    colors = ["#2196F3" if "RBAC-HNSW" in a else
              "#9E9E9E" if "baseline" in a.lower() or "HNSW" in a else
              "#F44336"
              for a in archs]
    bars = ax.barh(archs, gb, color=colors, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, gb):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f} GB", va="center", fontsize=9)
    ax.set_xlabel("Memory (GB)  [N=1M, d=768, M=16]", fontsize=11)
    ax.set_title("Memory Footprint Comparison (1M vectors)", fontsize=12,
                 fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    out = PLOTS_DIR / "fig4_memory_comparison.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick",  action="store_true")
    parser.add_argument("--n",      type=int, default=None)
    parser.add_argument("--nq",     type=int, default=None)
    parser.add_argument("--skip-exp1", action="store_true")
    parser.add_argument("--skip-exp2", action="store_true")
    parser.add_argument("--skip-exp3", action="store_true")
    parser.add_argument("--plots-only", action="store_true",
                        help="Only regenerate plots from existing CSVs")
    args = parser.parse_args()

    n_vecs    = args.n  or (50_000  if args.quick else 200_000)
    n_queries = args.nq or (200     if args.quick else 2_000)

    t_total = time.perf_counter()

    # ── Experiment 1: Recall ─────────────────────────────────────────────────
    exp1_csv = RESULTS_DIR / "exp1_selectivity_recall.csv"
    if not args.skip_exp1 and not args.plots_only:
        from experiment1_selectivity_recall import run_experiment as run_exp1
        df_recall = run_exp1(n_vectors=n_vecs, n_queries=min(500, n_queries))
    elif exp1_csv.exists():
        df_recall = pd.read_csv(exp1_csv)
        print(f"[exp1] Loaded cached results from {exp1_csv}")
    else:
        df_recall = None

    # ── Experiment 2: QPS ────────────────────────────────────────────────────
    exp2_csv = RESULTS_DIR / "exp2_selectivity_qps.csv"
    if not args.skip_exp2 and not args.plots_only:
        from experiment2_selectivity_qps import run_experiment as run_exp2
        df_qps = run_exp2(n_vectors=n_vecs, n_queries=n_queries)
    elif exp2_csv.exists():
        df_qps = pd.read_csv(exp2_csv)
        print(f"[exp2] Loaded cached results from {exp2_csv}")
    else:
        df_qps = None

    # ── Experiment 3: Memory ─────────────────────────────────────────────────
    exp3_csv = RESULTS_DIR / "exp3_memory_theoretical.csv"
    if not args.skip_exp3 and not args.plots_only:
        from experiment3_memory import run_experiment as run_exp3
        run_exp3()

    # ── Generate plots ────────────────────────────────────────────────────────
    print("\nGenerating publication figures …")
    if df_recall is not None:
        plot_recall_vs_selectivity(df_recall, ef=200)
        plot_recall_ef_curves(df_recall)

    if df_qps is not None:
        plot_qps_vs_selectivity(df_qps, ef=200)

    if exp3_csv.exists():
        df_theo = pd.read_csv(exp3_csv)
        plot_memory_comparison(df_theo)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\nTotal benchmark time: {(time.perf_counter()-t_total)/60:.1f} min")
    print(f"Plots written to:     {PLOTS_DIR}")

    if df_recall is not None:
        print("\n─── Recall@10 at ef=200 ───")
        best = df_recall[df_recall["ef"] == 200]
        tbl = best.pivot_table(index="selectivity_name", columns="method",
                               values="recall_at_k")
        print(tbl.to_string())

    if df_qps is not None:
        print("\n─── QPS at ef=200 ───")
        best_qps = df_qps[df_qps["ef"].isin([200, -1])]
        tbl = best_qps.pivot_table(index="selectivity_name", columns="method",
                                   values="qps", aggfunc="first")
        print(tbl.to_string())


if __name__ == "__main__":
    main()
