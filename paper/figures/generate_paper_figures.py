"""
generate_paper_figures.py
Generate publication-quality figures for the BioDMS @ VLDB 2026 paper.
All figures use IEEE/ACM style: serif fonts, tight layout, 3.33"-wide (single-column).
"""

import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from pathlib import Path

ROOT    = Path(__file__).parent.parent.parent
FIGDIR  = Path(__file__).parent
RESDIR  = ROOT / "results"

# ── ACM/IEEE publication style ────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          9,
    "axes.titlesize":     9,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    7.5,
    "legend.framealpha":  0.9,
    "lines.linewidth":    1.6,
    "lines.markersize":   5,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linewidth":     0.5,
})

# Single-column width for ACM proceedings (3.33 in)
W1 = 3.33
# Double-column width
W2 = 6.85

COLORS = {
    "rbac":  "#1565C0",   # deep blue
    "post":  "#C62828",   # deep red
    "pre":   "#E65100",   # deep orange
    "sieve": "#6A1B9A",   # purple
    "hnsw":  "#2E7D32",   # green
}
MARKERS = {
    "rbac": "o",
    "post": "^",
    "pre":  "s",
}


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: Recall@10 vs Selectivity  (2-panel: ef=100 and ef=400)
# ─────────────────────────────────────────────────────────────────────────────
def fig1_recall_vs_selectivity():
    csv = RESDIR / "exp1_selectivity_recall.csv"
    if not csv.exists():
        print(f"  [skip] {csv} not found"); return

    df = pd.read_csv(csv)
    # Map selectivity names to approximate % values
    sel_map = {"open": 39.87, "medium": 1.57, "restricted": 0.37,
               "strict": 0.02, "ultra": 0.001}
    df["sel_pct"] = df["selectivity_name"].map(sel_map)
    df = df[df["selectivity_name"] != "ultra"]   # 0 accessible in small dataset

    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.3), sharey=True)
    for ax, ef, label in zip(axes, [100, 400], ["(a) ef = 100", "(b) ef = 400"]):
        sub = df[df["ef"] == ef].sort_values("sel_pct")
        for method, color, marker, mname in [
            ("RBAC-HNSW (filter)", COLORS["rbac"], "o", "RBAC-HNSW (ours)"),
            ("Post-filter baseline", COLORS["post"], "^", "Post-filter"),
        ]:
            m = sub[sub["method"] == method]
            ax.plot(m["sel_pct"], m["recall_at_k"],
                    color=color, marker=marker, label=mname,
                    linewidth=1.8, markersize=5, zorder=3)

        ax.axhline(0.90, color="grey", linestyle="--", linewidth=0.9, alpha=0.7)
        ax.text(0.025, 0.91, "0.90 threshold", transform=ax.get_xaxis_transform(),
                fontsize=6.5, color="grey", va="bottom")
        ax.set_xscale("log")
        ax.set_xlim(0.01, 100)
        ax.set_ylim(0, 1.08)
        ax.set_xlabel("Accessible fraction of DB (%)")
        ax.xaxis.set_major_formatter(mticker.PercentFormatter())
        ax.set_title(label, pad=3)

    axes[0].set_ylabel("Recall@10")
    axes[0].legend(loc="lower right", frameon=True)
    fig.suptitle("Recall@10 vs. Query Selectivity", fontweight="bold",
                 fontsize=9, y=1.01)
    fig.tight_layout(w_pad=1.5)
    out = FIGDIR / "fig1_recall_vs_selectivity.pdf"
    fig.savefig(out); fig.savefig(str(out).replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Saved {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: ef sweep curves  (recall vs ef at 4 selectivity levels)
# ─────────────────────────────────────────────────────────────────────────────
def fig2_recall_ef_curves():
    csv = RESDIR / "exp1_selectivity_recall.csv"
    if not csv.exists():
        print(f"  [skip] {csv} not found"); return

    df = pd.read_csv(csv)
    levels = ["open", "medium", "restricted", "strict"]
    pcts   = {"open": "~40%", "medium": "~1.6%", "restricted": "~0.4%", "strict": "~0.02%"}

    fig, axes = plt.subplots(1, 4, figsize=(W2, 2.0), sharey=True)
    for ax, level in zip(axes, levels):
        sub = df[df["selectivity_name"] == level]
        for method, color, marker, mname in [
            ("RBAC-HNSW (filter)", COLORS["rbac"], "o", "RBAC-HNSW"),
            ("Post-filter baseline", COLORS["post"], "^", "Post-filter"),
        ]:
            m = sub[sub["method"] == method].sort_values("ef")
            ax.plot(m["ef"], m["recall_at_k"],
                    color=color, marker=marker, label=mname,
                    linewidth=1.6, markersize=4)
        ax.set_title(f"{level}\n({pcts[level]})", fontsize=8, pad=2)
        ax.set_xlabel("ef", fontsize=8)
        ax.set_ylim(0, 1.08)
        ax.set_xticks([50, 200, 800])
        ax.set_xticklabels(["50", "200", "800"], fontsize=7)

    axes[0].set_ylabel("Recall@10")
    axes[0].legend(loc="lower right", fontsize=7)
    fig.suptitle("Recall@10 vs. ef Beam Width", fontweight="bold", fontsize=9, y=1.03)
    fig.tight_layout(w_pad=1.2)
    out = FIGDIR / "fig2_recall_ef_curves.pdf"
    fig.savefig(out); fig.savefig(str(out).replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Saved {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: Memory footprint comparison vs. SIEVE
# ─────────────────────────────────────────────────────────────────────────────
def fig3_memory_comparison():
    csv = RESDIR / "exp3_memory_theoretical.csv"
    if not csv.exists():
        print(f"  [skip] {csv} not found"); return

    df = pd.read_csv(csv)
    df["gb"] = df["total_bytes"] / 1e9

    # Sort: RBAC-HNSW first, then SIEVE ascending
    order = ["RBAC-HNSW (ours)", "HNSW (no RBAC)",
             "SIEVE (8 roles)", "SIEVE (16 roles)",
             "SIEVE (32 roles)", "SIEVE (64 roles)"]
    df["arch_order"] = df["architecture"].apply(
        lambda x: order.index(x) if x in order else 99)
    df = df.sort_values("arch_order")

    colors = [COLORS["rbac"] if "RBAC-HNSW" in a
              else COLORS["hnsw"] if "HNSW" in a and "RBAC" not in a
              else COLORS["sieve"]
              for a in df["architecture"]]

    fig, ax = plt.subplots(figsize=(W1 + 0.3, 2.4))
    bars = ax.barh(df["architecture"], df["gb"],
                   color=colors, edgecolor="white", linewidth=0.4, height=0.6)
    for bar, val in zip(bars, df["gb"]):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f} GB", va="center", ha="left", fontsize=7.5)
    ax.set_xlabel("Memory (GB)  [N=1M vectors, d=768, M=16]", fontsize=8)
    ax.set_title("Memory Footprint vs. SIEVE Multi-Index", fontweight="bold",
                 fontsize=9, pad=4)
    ax.set_xlim(0, 240)
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=8)

    # Legend patches
    p1 = mpatches.Patch(color=COLORS["rbac"],  label="RBAC-HNSW (ours)")
    p2 = mpatches.Patch(color=COLORS["hnsw"],  label="Vanilla HNSW")
    p3 = mpatches.Patch(color=COLORS["sieve"], label="SIEVE multi-index")
    ax.legend(handles=[p1, p2, p3], fontsize=7, loc="lower right")
    fig.tight_layout()
    out = FIGDIR / "fig3_memory_comparison.pdf"
    fig.savefig(out); fig.savefig(str(out).replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Saved {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: Algorithm diagram — RBAC gate in HNSW traversal
# ─────────────────────────────────────────────────────────────────────────────
def fig4_algorithm_diagram():
    """Schematic of the two-gate search (conceptual figure for the paper)."""
    fig, ax = plt.subplots(figsize=(W1, 2.8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.axis("off")

    def node(x, y, label, color, r=0.55, fs=8):
        circ = plt.Circle((x, y), r, color=color, zorder=3)
        ax.add_patch(circ)
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fs, color="white", fontweight="bold", zorder=4)

    def arrow(x1, y1, x2, y2, color="grey", style="-"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color,
                                   lw=1.4, linestyle=style),
                    zorder=2)

    # Nodes: accessible (blue) and denied (red)
    node(2, 8,  "q",   "#1565C0",  r=0.5, fs=9)   # query
    node(5, 8,  "A",   "#1565C0")                   # accessible
    node(5, 5,  "B",   "#1565C0")                   # accessible
    node(8, 8,  "C",   "#C62828")                   # denied
    node(8, 5,  "D",   "#1565C0")                   # accessible
    node(8, 2,  "E",   "#C62828")                   # denied

    # Edges
    arrow(2.5, 8, 4.45, 8)           # q → A (enter)
    arrow(5.55, 8, 7.45, 8)          # A → C (denied: dashed)
    arrow(5.55, 7.6, 7.45, 5.4)      # A → D
    arrow(5,  7.45, 5, 5.55)         # A → B
    arrow(8,  7.45, 8, 5.55)         # C → D (route through denied)
    arrow(8,  4.45, 8, 2.55)         # D → E

    # Labels on arrows
    ax.text(3.2, 8.25, "enter", fontsize=6.5, color="grey")
    ax.text(6.3, 8.25, "denied\n(route)", fontsize=6, color="#C62828",
            ha="center", style="italic")
    ax.text(6.0, 6.8,  "dist(q,D)", fontsize=6, color="#1565C0", ha="center")
    ax.text(4.4, 6.5,  "dist(q,B)", fontsize=6, color="#1565C0", ha="center")

    # Legend
    p_acc  = mpatches.Patch(color="#1565C0", label="Accessible node (distance computed)")
    p_den  = mpatches.Patch(color="#C62828", label="Denied node (route only, no distance)")
    ax.legend(handles=[p_acc, p_den], fontsize=6.5,
              loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=1,
              frameon=True, framealpha=0.9)

    ax.set_title("RBAC-HNSW: Two-Gate Graph Traversal\n"
                 "Denied nodes route the beam; distance skipped.",
                 fontsize=8, fontweight="bold", pad=6)
    fig.tight_layout()
    out = FIGDIR / "fig4_algorithm_diagram.pdf"
    fig.savefig(out); fig.savefig(str(out).replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Saved {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: RBAC bitmask schema diagram
# ─────────────────────────────────────────────────────────────────────────────
def fig5_rbac_schema():
    """64-bit bitmask layout as a colour-coded figure."""
    fig, ax = plt.subplots(figsize=(W2, 1.1))
    ax.axis("off")

    groups = [
        (0,  8,  "#1565C0", "Dept\n(bits 0–7)"),
        (8,  16, "#C62828", "Role\n(bits 8–15)"),
        (16, 24, "#E65100", "Sensitivity\n(bits 16–23)"),
        (24, 32, "#2E7D32", "Consent\n(bits 24–31)"),
        (32, 40, "#6A1B9A", "Research\n(bits 32–39)"),
        (40, 48, "#00838F", "Temporal\n(bits 40–47)"),
        (48, 64, "#78909C", "Reserved\n(bits 48–63)"),
    ]
    total = 64
    y = 0.5
    h = 0.5
    for start, end, color, label in groups:
        width = (end - start) / total
        x = start / total
        rect = plt.Rectangle((x, y - h/2), width, h,
                              color=color, ec="white", lw=1.0, zorder=2)
        ax.add_patch(rect)
        cx = x + width / 2
        ax.text(cx, y, label, ha="center", va="center",
                fontsize=6.5, color="white", fontweight="bold", zorder=3,
                multialignment="center")
        ax.plot([x, x], [y - h/2 - 0.08, y - h/2 - 0.18],
                color="grey", lw=0.5, transform=ax.transData)
        ax.text(x, y - h/2 - 0.22, str(start), ha="center",
                fontsize=5.5, color="grey", transform=ax.transData)

    ax.text(1.0, y - h/2 - 0.22, "63", ha="center",
            fontsize=5.5, color="grey", transform=ax.transData)
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.3, 1.0)
    ax.set_title("64-bit RBAC Bitmask Layout (NIST RBAC hospital hierarchy)",
                 fontsize=8, fontweight="bold", pad=3)
    fig.tight_layout(pad=0.3)
    out = FIGDIR / "fig5_rbac_schema.pdf"
    fig.savefig(out); fig.savefig(str(out).replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Saved {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6: Recall@ef=200 heatmap (method × selectivity)
# ─────────────────────────────────────────────────────────────────────────────
def fig6_recall_heatmap():
    csv = RESDIR / "exp1_selectivity_recall.csv"
    if not csv.exists():
        print(f"  [skip] {csv} not found"); return

    df = pd.read_csv(csv)
    df = df[df["ef"] == 200]

    pivot = df.pivot_table(index="method", columns="selectivity_name",
                           values="recall_at_k")
    col_order = ["ultra", "strict", "restricted", "medium", "open"]
    col_order = [c for c in col_order if c in pivot.columns]
    pivot = pivot[col_order]
    row_order = ["RBAC-HNSW (filter)", "Post-filter baseline"]
    row_order = [r for r in row_order if r in pivot.index]
    pivot = pivot.loc[row_order]
    pivot.index = ["RBAC-HNSW\n(ours)", "Post-filter\nbaseline"]
    pivot.columns = [f"{c}\n({{'ultra':'~0.001%','strict':'~0.02%','restricted':'~0.4%','medium':'~1.6%','open':'~40%'}}[c])"
                     for c in col_order]
    # Simpler column labels
    col_labels = {"ultra": "ultra\n(~0%)", "strict": "strict\n(~0.02%)",
                  "restricted": "restr.\n(~0.4%)", "medium": "medium\n(~1.6%)",
                  "open": "open\n(~40%)"}
    pivot.columns = [col_labels.get(c, c) for c in col_order]

    fig, ax = plt.subplots(figsize=(W1 + 0.4, 1.4))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=7.5)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = "black" if val > 0.6 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")
    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cb.set_label("Recall@10", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    ax.set_title("Recall@10 Heatmap (ef = 200)", fontweight="bold",
                 fontsize=9, pad=4)
    fig.tight_layout()
    out = FIGDIR / "fig6_recall_heatmap.pdf"
    fig.savefig(out); fig.savefig(str(out).replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Saved {out.name}")


if __name__ == "__main__":
    print("Generating paper figures …\n")
    fig1_recall_vs_selectivity()
    fig2_recall_ef_curves()
    fig3_memory_comparison()
    fig4_algorithm_diagram()
    fig5_rbac_schema()
    fig6_recall_heatmap()
    print("\nAll figures saved to:", FIGDIR)
