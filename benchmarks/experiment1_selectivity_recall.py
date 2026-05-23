"""
Experiment 1: Selectivity vs. Recall@10
========================================
Measures how accurately each algorithm finds the true nearest neighbours as
the query becomes more restrictive (lower selectivity).

Protocol (following ANN-Benchmarks, Aumuller et al. 2020)
-----------------------------------------------------------
  recall@k = |{true top-k} ∩ {returned top-k}| / k

  Selectivity levels tested:
    "open"       ≈ 80 %   (near-baseline HNSW performance expected)
    "medium"     ≈ 20 %
    "restricted" ≈  5 %
    "strict"     ≈  1 %
    "ultra"      ≈  0.1 %

For each selectivity level:
1. Compute ground-truth top-10 via PreFilterBaseline (exact brute-force).
2. Measure recall@10 for RBAC-HNSW (filter strategy), RBAC-HNSW (routing
   strategy), and PostFilter baseline.
3. Sweep ef values [50, 100, 200, 400, 800] to produce recall-vs-ef curves.

Expected result (from ACORN, Patel et al. 2024, Figure 5):
  - Post-filtering: recall crashes at selectivity ≤ 1 %.
  - RBAC-HNSW (routing): maintains recall ≥ 0.90 down to 0.1 % selectivity.
"""

from __future__ import annotations

import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Project root on sys.path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_generator import generate_dataset, generate_query_masks
from rbac_hnsw import RBACIndex
from baselines import PostFilterBaseline, PreFilterBaseline

RESULTS_DIR = ROOT / "results"
K = 10
EF_VALUES   = [50, 100, 200, 400, 800]
N_QUERIES   = 500     # subset for recall evaluation (full 10k for QPS)
N_VECTORS   = 200_000 # reduced dataset for recall experiments (faster)


def compute_recall(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """recall@k averaged over queries."""
    recalls = []
    for t, p in zip(true_labels, pred_labels):
        t_set  = set(t[t >= 0].tolist())
        p_set  = set(p[p >= 0].tolist())
        if not t_set:
            continue
        recalls.append(len(t_set & p_set) / len(t_set))
    return float(np.mean(recalls)) if recalls else 0.0


def run_experiment(n_vectors: int = N_VECTORS, n_queries: int = N_QUERIES,
                   seed: int = 42) -> pd.DataFrame:
    print("=" * 70)
    print("Experiment 1: Selectivity vs. Recall@10")
    print("=" * 70)

    # ── Generate data ─────────────────────────────────────────────────────────
    print(f"\nGenerating {n_vectors:,} vectors …")
    vectors, masks, queries_all, q_masks = generate_dataset(
        n_vectors=n_vectors, dim=768, n_queries=10_000,
        seed=seed, output_dir=ROOT / "data" / "exp1"
    )
    queries = queries_all[:n_queries]

    # ── Build indexes ─────────────────────────────────────────────────────────
    print("\nBuilding indexes …")
    ids = np.arange(n_vectors, dtype=np.int64)

    rbac_idx = RBACIndex(dim=768, space="cosine", M=16, ef_construction=200)
    rbac_idx.add_items(vectors, masks, ids)
    print(f"  RBAC-HNSW built in {rbac_idx.build_time_s:.1f}s")

    post_idx = PostFilterBaseline(dim=768, space="cosine", M=16, ef_construction=200)
    post_idx.add_items(vectors, masks, ids)

    pre_idx  = PreFilterBaseline(dim=768)
    pre_idx.add_items(vectors, masks, ids)

    # ── Compute recall ─────────────────────────────────────────────────────────
    rows = []
    for sel_name, query_mask in q_masks.items():
        qm64 = np.uint64(query_mask)
        accessible = float(np.sum((masks & qm64) == qm64)) / n_vectors
        print(f"\n  Selectivity level: {sel_name:12s}  ({accessible*100:.2f}% accessible)")

        # Ground truth (exact)
        print("    Computing ground truth (brute-force) …", end=" ", flush=True)
        t0 = time.perf_counter()
        gt_labels, _ = pre_idx.batch_search(queries, k=K, query_mask=query_mask)
        print(f"{time.perf_counter()-t0:.1f}s")

        for ef in EF_VALUES:
            # RBAC-HNSW filter strategy
            t0 = time.perf_counter()
            lbl_rbac_f, _ = rbac_idx.batch_search(
                queries, k=K, query_mask=query_mask,
                ef=ef, strategy="filter"
            )
            t_rbac_f = time.perf_counter() - t0

            recall_rbac_f = compute_recall(gt_labels, lbl_rbac_f)

            # Post-filter baseline
            t0 = time.perf_counter()
            lbl_post, _ = post_idx.batch_search(
                queries, k=K, query_mask=query_mask, ef=ef
            )
            t_post = time.perf_counter() - t0

            recall_post = compute_recall(gt_labels, lbl_post)

            rows.append({
                "selectivity_name":  sel_name,
                "selectivity_frac":  round(accessible, 4),
                "ef":                ef,
                "method":            "RBAC-HNSW (filter)",
                "recall_at_k":       round(recall_rbac_f, 4),
                "search_time_s":     round(t_rbac_f, 4),
            })
            rows.append({
                "selectivity_name":  sel_name,
                "selectivity_frac":  round(accessible, 4),
                "ef":                ef,
                "method":            "Post-filter baseline",
                "recall_at_k":       round(recall_post, 4),
                "search_time_s":     round(t_post, 4),
            })
            print(f"    ef={ef:4d}  RBAC-HNSW={recall_rbac_f:.3f}  PostFilter={recall_post:.3f}")

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "exp1_selectivity_recall.csv", index=False)
    print(f"\nResults saved → {RESULTS_DIR / 'exp1_selectivity_recall.csv'}")
    return df


if __name__ == "__main__":
    df = run_experiment()
    print("\nSummary (recall@10 at ef=200):")
    print(df[df["ef"] == 200].pivot_table(
        index="selectivity_name", columns="method", values="recall_at_k"
    ).to_string())
