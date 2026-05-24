"""
experiment_scale.py — N=1M Scale Evaluation + QPS vs. Recall Curves
=====================================================================
Addresses the two critical reviewer concerns:

1. Scale (N=1,000,000): at 1M vectors the routing benefit becomes empirically
   visible because the accessible set at 0.1% selectivity (~1,000 nodes) is
   a genuine island in a million-node graph.

2. QPS vs. Recall trade-off: the gold-standard ANN benchmark metric
   (ANN-Benchmarks, Aumuller et al. 2020). Sweeping ef produces a
   Pareto frontier of recall vs. throughput for each algorithm.

Experiment design
-----------------
For each selectivity level:
  - Sweep ef ∈ {10, 20, 50, 100, 200, 400, 800, 1600}
  - For each ef: measure recall@10 AND QPS
  - Plot recall (x-axis) vs QPS (y-axis) — higher-right = better

Expected result (validates main paper claim):
  - At "open" (40%): both methods lie on the same Pareto curve.
  - At "medium" (1.6%): RBAC-HNSW begins to pull ahead in recall.
  - At "strict" (0.02%): post-filtering recall collapses at all ef;
    RBAC-HNSW maintains recall with a small QPS penalty.

Note on Python throughput:
  The Python filter closure still has per-candidate overhead.
  v2 uses frozenset membership (O(1)) instead of dict lookup + numpy,
  giving ~5-10x improvement over v1.  All QPS numbers include the full
  Python overhead and are labelled as "Python prototype" in the figure.
  C++ numbers (estimated from hnswlib native throughput * gate_ratio)
  are shown as dashed projections.
"""

from __future__ import annotations

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_generator import generate_dataset, generate_query_masks
from rbac_hnsw import RBACIndex
from baselines import PostFilterBaseline, PreFilterBaseline

RESULTS_DIR = ROOT / "results"
K           = 10
EF_SWEEP    = [10, 20, 50, 100, 200, 400, 800]
WARMUP_Q    = 20
N_QPS_Q     = 200       # queries for throughput measurement
N_RECALL_Q  = 500       # queries for recall measurement


def measure_qps_recall(search_fn, queries_qps, queries_recall,
                        gt_labels, k=10, warmup=WARMUP_Q):
    """Measure both QPS and recall in one round."""
    # Warm up
    search_fn(queries_recall[:warmup])
    # QPS timing
    t0 = time.perf_counter()
    search_fn(queries_qps)
    qps = len(queries_qps) / (time.perf_counter() - t0)
    # Recall
    pred_labels, _ = search_fn(queries_recall)
    recalls = []
    for gt, pred in zip(gt_labels, pred_labels):
        gt_s   = set(gt[gt >= 0].tolist())
        pred_s = set(pred[pred >= 0].tolist())
        recalls.append(len(gt_s & pred_s) / len(gt_s) if gt_s else 1.0)
    return qps, float(np.mean(recalls))


def run_scale_experiment(n_vectors: int = 1_000_000,
                         n_queries: int = N_RECALL_Q + N_QPS_Q,
                         seed: int = 42) -> pd.DataFrame:
    print("=" * 70)
    print(f"Scale Experiment: N={n_vectors:,}, d=768")
    print("=" * 70)

    # ── Generate data ─────────────────────────────────────────────────────────
    out_dir = ROOT / "data" / f"scale_{n_vectors//1000}k"
    vecs_path = out_dir / "vectors.npy"
    if vecs_path.exists():
        print(f"\n[cache] Loading cached dataset from {out_dir}")
        vectors = np.load(out_dir / "vectors.npy")
        masks   = np.load(out_dir / "masks.npy")
        queries = np.load(out_dir / "queries.npy")
        q_masks = generate_query_masks()
    else:
        vectors, masks, queries, q_masks = generate_dataset(
            n_vectors=n_vectors, dim=768, n_queries=n_queries,
            seed=seed, output_dir=out_dir)

    queries_recall = queries[:N_RECALL_Q]
    queries_qps    = queries[N_RECALL_Q:N_RECALL_Q + N_QPS_Q]
    ids = np.arange(n_vectors, dtype=np.int64)

    # ── Build indexes ─────────────────────────────────────────────────────────
    idx_path = out_dir / "rbac_index"
    rbac_idx = RBACIndex(dim=768, space="cosine", M=16, ef_construction=200)
    post_idx = PostFilterBaseline(dim=768, space="cosine", M=16, ef_construction=200)

    if (idx_path / "hnsw.bin").exists():
        print(f"\n[cache] Loading cached RBAC index from {idx_path}")
        rbac_idx.load(idx_path, max_elements=n_vectors)
        # Re-add masks to post_idx (not saved separately)
        print("[cache] Rebuilding PostFilter index …")
        post_idx.add_items(vectors, masks, ids)
    else:
        print(f"\nBuilding HNSW indexes for {n_vectors:,} vectors …")
        rbac_idx.add_items(vectors, masks, ids)
        post_idx.add_items(vectors, masks, ids)
        print(f"  Built in {rbac_idx.build_time_s:.0f}s")
        rbac_idx.save(idx_path)

    # Pre-filter (exact) — used only for ground-truth, not QPS
    pre_idx = PreFilterBaseline(dim=768)
    pre_idx.add_items(vectors, masks, ids)

    print(f"\nMemory: {rbac_idx.memory_bytes()['total_bytes']/1e9:.2f} GB")
    print(f"Mask overhead: {rbac_idx.memory_bytes()['mask_overhead_pct']:.3f}%\n")

    rows = []
    for sel_name, query_mask in q_masks.items():
        qm64 = np.uint64(query_mask)
        n_acc = int(np.sum((masks & qm64) == qm64))
        sel   = n_acc / n_vectors
        print(f"  Selectivity: {sel_name:12s}  "
              f"({sel*100:.3f}%,  {n_acc:,} accessible vectors)")

        # Ground truth
        print("    Computing ground truth …", end=" ", flush=True)
        t0 = time.perf_counter()
        gt_labels, _ = pre_idx.batch_search(queries_recall, k=K,
                                             query_mask=query_mask)
        print(f"{time.perf_counter()-t0:.1f}s")

        for ef in EF_SWEEP:
            # RBAC-HNSW
            rbac_qps, rbac_recall = measure_qps_recall(
                lambda q, ef=ef, qm=query_mask: rbac_idx.batch_search(
                    q, k=K, query_mask=qm, ef=ef, strategy="filter"),
                queries_qps, queries_recall, gt_labels)

            # Post-filter
            post_qps, post_recall = measure_qps_recall(
                lambda q, ef=ef, qm=query_mask: post_idx.batch_search(
                    q, k=K, query_mask=qm, ef=ef),
                queries_qps, queries_recall, gt_labels)

            print(f"    ef={ef:4d}  "
                  f"RBAC: recall={rbac_recall:.3f} qps={rbac_qps:7.1f}  "
                  f"Post: recall={post_recall:.3f} qps={post_qps:7.1f}")

            for method, recall, qps in [
                ("RBAC-HNSW", rbac_recall, rbac_qps),
                ("Post-filter", post_recall, post_qps),
            ]:
                rows.append({
                    "n_vectors":        n_vectors,
                    "selectivity_name": sel_name,
                    "selectivity_frac": round(sel, 6),
                    "n_accessible":     n_acc,
                    "ef":               ef,
                    "method":           method,
                    "recall_at_10":     round(recall, 4),
                    "qps":              round(qps, 2),
                })

        # Brute-force QPS (ef-independent)
        t0 = time.perf_counter()
        pre_idx.batch_search(queries_qps, k=K, query_mask=query_mask)
        pre_qps = len(queries_qps) / (time.perf_counter() - t0)
        print(f"    Brute-force exact QPS = {pre_qps:.1f}")
        rows.append({
            "n_vectors": n_vectors, "selectivity_name": sel_name,
            "selectivity_frac": round(sel, 6), "n_accessible": n_acc,
            "ef": -1, "method": "Brute-force",
            "recall_at_10": 1.0, "qps": round(pre_qps, 2)})

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / f"scale_{n_vectors//1000}k_qps_recall.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nResults → {out_csv}")
    return df


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n",    type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run_scale_experiment(n_vectors=args.n, seed=args.seed)
