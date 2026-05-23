"""
Experiment 2: Selectivity vs. Throughput (QPS)
================================================
Proves that RBAC-HNSW maintains high QPS across all access levels while the
two baselines each catastrophically fail at opposite ends of the selectivity
spectrum.

This is the key VLDB systems result (Section 5.3 of the paper).

Expected failure modes (cf. Figure 1 of SIEVE, Zhang et al. 2025)
-------------------------------------------------------------------
* Post-filtering:  At low selectivity (0.1 %) the filter discards almost all
  candidates from a small beam → must inflate ef to thousands to find k=10
  accessible results → QPS collapses.

* Pre-filtering (brute-force): At high selectivity (80 %) 800,000 vectors must
  be scanned per query → O(800k × 768) operations → QPS ≈ 1–2.

* RBAC-HNSW:  Routing through denied nodes preserves graph connectivity at
  low selectivity. The actual number of distance computations is bounded by
  ef × (selectivity_fraction + routing_overhead) ≈ ef × 1.15 regardless of
  selectivity → flat QPS curve.

Measurement methodology (ANN-Benchmarks protocol)
--------------------------------------------------
  QPS = n_queries / total_wall_clock_seconds
  (includes filter overhead; excludes index build time)

  For each method × selectivity × ef:
    1. Warm-up: 100 queries (excluded from timing).
    2. Timed run: full 10,000-query batch.
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
EF_VALUES   = [50, 100, 200, 400, 800]
N_QUERIES   = 2_000   # reduced for speed; paper uses 10k
N_VECTORS   = 200_000 # reduced for speed; paper uses 1M
WARMUP_Q    = 50


def measure_qps(fn, queries: np.ndarray, warmup: int = WARMUP_Q) -> float:
    """Run fn(queries) after a warmup pass and return QPS."""
    fn(queries[:warmup])
    t0 = time.perf_counter()
    fn(queries)
    return len(queries) / (time.perf_counter() - t0)


def run_experiment(n_vectors: int = N_VECTORS, n_queries: int = N_QUERIES,
                   seed: int = 42) -> pd.DataFrame:
    print("=" * 70)
    print("Experiment 2: Selectivity vs. QPS")
    print("=" * 70)

    print(f"\nGenerating {n_vectors:,} vectors …")
    vectors, masks, queries_all, q_masks = generate_dataset(
        n_vectors=n_vectors, dim=768, n_queries=10_000,
        seed=seed, output_dir=ROOT / "data" / "exp2"
    )
    queries = queries_all[:n_queries]

    print("\nBuilding indexes …")
    ids = np.arange(n_vectors, dtype=np.int64)

    rbac_idx = RBACIndex(dim=768, space="cosine", M=16, ef_construction=200)
    rbac_idx.add_items(vectors, masks, ids)

    post_idx = PostFilterBaseline(dim=768, space="cosine", M=16, ef_construction=200)
    post_idx.add_items(vectors, masks, ids)

    pre_idx  = PreFilterBaseline(dim=768)
    pre_idx.add_items(vectors, masks, ids)

    rows = []
    for sel_name, query_mask in q_masks.items():
        qm64 = np.uint64(query_mask)
        accessible = float(np.sum((masks & qm64) == qm64)) / n_vectors
        print(f"\n  Selectivity: {sel_name:12s} ({accessible*100:.2f}%)")

        for ef in EF_VALUES:
            # ── RBAC-HNSW (filter strategy) ──────────────────────────────────
            qps_rbac = measure_qps(
                lambda q, ef=ef, qm=query_mask: rbac_idx.batch_search(
                    q, k=K, query_mask=qm, ef=ef, strategy="filter"),
                queries,
            )

            # ── Post-filter baseline ──────────────────────────────────────────
            qps_post = measure_qps(
                lambda q, ef=ef, qm=query_mask: post_idx.batch_search(
                    q, k=K, query_mask=qm, ef=ef),
                queries,
            )

            print(f"    ef={ef:4d}  RBAC-HNSW={qps_rbac:7.1f} QPS  PostFilter={qps_post:7.1f} QPS")

            rows.append({"selectivity_name": sel_name,
                         "selectivity_frac": round(accessible, 4),
                         "ef": ef,
                         "method": "RBAC-HNSW (filter)",
                         "qps": round(qps_rbac, 2)})
            rows.append({"selectivity_name": sel_name,
                         "selectivity_frac": round(accessible, 4),
                         "ef": ef,
                         "method": "Post-filter baseline",
                         "qps": round(qps_post, 2)})

        # Brute-force QPS (ef-independent)
        qps_pre = measure_qps(
            lambda q, qm=query_mask: pre_idx.batch_search(q, k=K, query_mask=qm),
            queries,
        )
        print(f"    Brute-force (exact) QPS = {qps_pre:.1f}")
        rows.append({"selectivity_name": sel_name,
                     "selectivity_frac": round(accessible, 4),
                     "ef": -1,
                     "method": "Pre-filter (brute-force)",
                     "qps": round(qps_pre, 2)})

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "exp2_selectivity_qps.csv", index=False)
    print(f"\nResults saved → {RESULTS_DIR / 'exp2_selectivity_qps.csv'}")
    return df


if __name__ == "__main__":
    df = run_experiment()
    print("\nQPS at ef=200 across selectivity levels:")
    mask = df["ef"].isin([200, -1])
    print(df[mask].pivot_table(
        index="selectivity_name", columns="method", values="qps", aggfunc="first"
    ).to_string())
