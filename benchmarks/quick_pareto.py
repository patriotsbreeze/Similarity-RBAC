"""
quick_pareto.py — fast QPS vs Recall experiment for paper figures
================================================================
Reduces query counts and ef values to run in ~5 minutes.
Measures the KEY results: open and strict selectivity Pareto curves.
Outputs scale_200k_qps_recall.csv (same schema as experiment_scale.py).
"""
from __future__ import annotations
import sys, time
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_generator import generate_dataset, generate_query_masks
from rbac_hnsw import RBACIndex
from baselines import PostFilterBaseline, PreFilterBaseline

RESULTS_DIR = ROOT / "results"
K = 10
# Reduced sweep — 5 values is enough for a clear Pareto curve
EF_SWEEP  = [10, 50, 100, 200, 400, 800]
N_QPS_Q   = 50    # queries for QPS (statistical noise is OK)
N_RECALL_Q = 200  # queries for recall (need stability)
WARMUP_Q   = 5


def measure_qps_recall(search_fn, queries_qps, queries_recall, gt_labels, k=K):
    search_fn(queries_recall[:WARMUP_Q])          # warm up
    t0 = time.perf_counter()
    search_fn(queries_qps)
    qps = len(queries_qps) / (time.perf_counter() - t0)
    pred_labels, _ = search_fn(queries_recall)
    recalls = []
    for gt, pred in zip(gt_labels, pred_labels):
        gt_s   = set(gt[gt >= 0].tolist())
        pred_s = set(pred[pred >= 0].tolist())
        recalls.append(len(gt_s & pred_s) / len(gt_s) if gt_s else 1.0)
    return qps, float(np.mean(recalls))


def run(n_vectors: int = 200_000, seed: int = 42) -> pd.DataFrame:
    print(f"Quick Pareto Experiment: N={n_vectors:,}, d=768")

    out_dir   = ROOT / "data" / f"scale_{n_vectors//1000}k"
    vecs_path = out_dir / "vectors.npy"
    if vecs_path.exists():
        print(f"[cache] Loading from {out_dir}")
        vectors = np.load(out_dir / "vectors.npy")
        masks   = np.load(out_dir / "masks.npy")
        queries = np.load(out_dir / "queries.npy")
        q_masks = generate_query_masks()
    else:
        vectors, masks, queries, q_masks = generate_dataset(
            n_vectors=n_vectors, dim=768,
            n_queries=N_QPS_Q + N_RECALL_Q + WARMUP_Q,
            seed=seed, output_dir=out_dir)

    queries_recall = queries[:N_RECALL_Q]
    queries_qps    = queries[N_RECALL_Q:N_RECALL_Q + N_QPS_Q]
    ids = np.arange(n_vectors, dtype=np.int64)

    # ── Build indexes ──────────────────────────────────────────────────────────
    idx_path = out_dir / "rbac_index"
    rbac_idx = RBACIndex(dim=768, space="cosine", M=16, ef_construction=200)
    post_idx = PostFilterBaseline(dim=768, space="cosine", M=16, ef_construction=200)
    pre_idx  = PreFilterBaseline(dim=768)

    if (idx_path / "hnsw.bin").exists():
        print("[cache] Loading RBAC index …")
        rbac_idx.load(idx_path, max_elements=n_vectors)
    else:
        print(f"Building HNSW (N={n_vectors:,}) …")
        rbac_idx.add_items(vectors, masks, ids)
        rbac_idx.save(idx_path)

    print("Building PostFilter index …")
    post_idx.add_items(vectors, masks, ids)
    pre_idx.add_items(vectors, masks, ids)

    print(f"Memory: {rbac_idx.memory_bytes()['total_bytes']/1e9:.2f} GB  "
          f"({rbac_idx.memory_bytes()['mask_overhead_pct']:.3f}% overhead)")

    rows = []
    for sel_name, query_mask in q_masks.items():
        qm64  = np.uint64(query_mask)
        n_acc = int(np.sum((masks & qm64) == qm64))
        sel   = n_acc / n_vectors
        print(f"\n  {sel_name:12s}  {sel*100:.3f}%  ({n_acc:,} accessible)")

        # Ground truth
        t0 = time.perf_counter()
        gt_labels, _ = pre_idx.batch_search(queries_recall, k=K, query_mask=query_mask)
        print(f"  GT in {time.perf_counter()-t0:.1f}s", end="")

        for ef in EF_SWEEP:
            rbac_qps, rbac_rec = measure_qps_recall(
                lambda q, ef=ef, qm=query_mask: rbac_idx.batch_search(
                    q, k=K, query_mask=qm, ef=ef, strategy="filter"),
                queries_qps, queries_recall, gt_labels)
            post_qps, post_rec = measure_qps_recall(
                lambda q, ef=ef, qm=query_mask: post_idx.batch_search(
                    q, k=K, query_mask=qm, ef=ef),
                queries_qps, queries_recall, gt_labels)
            print(f"  ef={ef:3d}  RBAC:{rbac_rec:.3f}/{rbac_qps:.0f}  "
                  f"Post:{post_rec:.3f}/{post_qps:.0f}", end="", flush=True)
            for method, rec, qps in [("RBAC-HNSW", rbac_rec, rbac_qps),
                                      ("Post-filter", post_rec, post_qps)]:
                rows.append({"n_vectors": n_vectors, "selectivity_name": sel_name,
                              "selectivity_frac": round(sel, 6), "n_accessible": n_acc,
                              "ef": ef, "method": method,
                              "recall_at_10": round(rec, 4), "qps": round(qps, 2)})

        t0 = time.perf_counter()
        pre_idx.batch_search(queries_qps, k=K, query_mask=query_mask)
        pre_qps = N_QPS_Q / (time.perf_counter() - t0)
        print(f"  BF:{pre_qps:.0f}")
        rows.append({"n_vectors": n_vectors, "selectivity_name": sel_name,
                     "selectivity_frac": round(sel, 6), "n_accessible": n_acc,
                     "ef": -1, "method": "Brute-force",
                     "recall_at_10": 1.0, "qps": round(pre_qps, 2)})

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"scale_{n_vectors//1000}k_qps_recall.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved → {out}")
    return df


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    run(n_vectors=a.n, seed=a.seed)
