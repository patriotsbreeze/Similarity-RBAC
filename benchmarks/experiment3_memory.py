"""
Experiment 3: Memory Overhead Analysis
=======================================
Measures and compares the RAM footprint of RBAC-HNSW against the baseline
approaches and competitive systems.

Memory model (Section 5.2 of the paper)
-----------------------------------------
For N vectors of dimension d (float32), M HNSW edges:

  HNSW baseline:
    vectors : N × d × 4 bytes
    edges   : N × M × 2 × 4 × 2 bytes  (bidirectional, two layers approx)
    total   : N × (4d + 16M) bytes

  RBAC-HNSW (this work):
    = HNSW baseline + N × 8 bytes  (one uint64 per vector)
    overhead: 8 / (4d + 16M) × 100 %
    For d=768, M=16:  8 / (3072 + 256) ≈ 0.24 % overhead  ← negligible

  SIEVE multi-index (Zhang et al. 2025):
    Maintains O(|roles|) separate HNSW indexes.
    Worst case (no index sharing): |roles| × N × (4d + 16M) bytes.
    For 64 roles: 64× the single-index footprint.

  Pre-filter brute-force:
    = vectors only: N × d × 4 bytes (no graph overhead)
    Cannot scale to high QPS — included for completeness.

Literature
----------
* Drepper (2007): Cache-line layout — 64-byte lines; co-locating the 8-byte
  RBAC mask with the 768-dim vector pointer on the same cache line.
* SIEVE (Zhang et al. 2025): Multi-index baseline for memory comparison.
* FilteredDiskANN (Gollapudi et al. 2023): Separate entry-point structures —
  O(|labels| × N) overhead for entry-point arrays.
"""

from __future__ import annotations

import sys
import os
import gc
import psutil
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rbac_hnsw import RBACIndex
from baselines import PostFilterBaseline, PreFilterBaseline

RESULTS_DIR = ROOT / "results"


def _rss_mb() -> float:
    """Resident set size of current process in MiB."""
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / (1024 ** 2)


def theoretical_memory(n: int, d: int, M: int = 16) -> dict:
    """
    Compute theoretical memory estimates (bytes) for all approaches.

    For SIEVE we model n_roles ∈ {8, 16, 32, 64} separate HNSW indexes.
    """
    vec_bytes      = n * d * 4
    edge_bytes     = n * M * 2 * 4 * 2   # both HNSW layers, bidirectional
    hnsw_total     = vec_bytes + edge_bytes
    rbac_overhead  = n * 8               # one uint64 per vector
    rbac_total     = hnsw_total + rbac_overhead

    results = {
        "brute_force_bytes": vec_bytes,
        "hnsw_baseline_bytes": hnsw_total,
        "rbac_hnsw_bytes": rbac_total,
        "rbac_overhead_bytes": rbac_overhead,
        "rbac_overhead_pct": 100 * rbac_overhead / hnsw_total,
    }
    for n_roles in [8, 16, 32, 64]:
        results[f"sieve_{n_roles}roles_bytes"] = n_roles * hnsw_total
        results[f"sieve_{n_roles}roles_vs_rbac_multiplier"] = (
            n_roles * hnsw_total / rbac_total
        )
    return results


def measure_empirical_memory(n_vectors: int = 100_000, dim: int = 768,
                              seed: int = 42) -> pd.DataFrame:
    """
    Empirically measure RSS before and after building each index.

    Returns a DataFrame with actual memory deltas.
    """
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((n_vectors, dim)).astype(np.float32)
    norms   = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors /= norms
    masks   = rng.integers(0, np.iinfo(np.uint64).max, size=n_vectors,
                           dtype=np.uint64)
    ids     = np.arange(n_vectors, dtype=np.int64)

    rows = []

    def _measure(name: str, build_fn, search_fn=None):
        gc.collect()
        rss_before = _rss_mb()
        obj = build_fn()
        gc.collect()
        rss_after = _rss_mb()
        delta_mb = rss_after - rss_before
        print(f"  {name:35s} Δ RSS = {delta_mb:+8.1f} MiB  (total = {rss_after:.0f} MiB)")
        rows.append({"method": name, "n_vectors": n_vectors, "dim": dim,
                     "rss_delta_mib": round(delta_mb, 1),
                     "rss_total_mib": round(rss_after, 1)})
        return obj

    print(f"\nBuilding indexes for {n_vectors:,} × {dim}-dim …\n")

    _measure(
        "PreFilter (brute-force)",
        lambda: (lambda idx: (idx.add_items(vectors, masks, ids), idx)[1])(
            PreFilterBaseline(dim=dim)),
    )

    _measure(
        "PostFilter (HNSW, no masks)",
        lambda: (lambda idx: (idx.add_items(vectors, masks, ids), idx)[1])(
            PostFilterBaseline(dim=dim, M=16, ef_construction=200)),
    )

    _measure(
        "RBAC-HNSW (M=16, our method)",
        lambda: (lambda idx: (idx.add_items(vectors, masks, ids), idx)[1])(
            RBACIndex(dim=dim, M=16, ef_construction=200)),
    )

    _measure(
        "RBAC-HNSW (M=32, higher recall)",
        lambda: (lambda idx: (idx.add_items(vectors, masks, ids), idx)[1])(
            RBACIndex(dim=dim, M=32, ef_construction=200)),
    )

    return pd.DataFrame(rows)


def run_experiment() -> pd.DataFrame:
    print("=" * 70)
    print("Experiment 3: Memory Overhead")
    print("=" * 70)

    # ── Theoretical analysis ─────────────────────────────────────────────────
    N, D = 1_000_000, 768
    print(f"\nTheoretical memory estimates for N={N:,}, d={D}, M=16:\n")
    theo = theoretical_memory(N, D, M=16)
    for k, v in theo.items():
        if "bytes" in k:
            print(f"  {k:45s} = {v/1e9:7.2f} GB  ({v:,} bytes)")
        else:
            print(f"  {k:45s} = {v:.2f}×")

    rows_theo = []
    for n_roles in [8, 16, 32, 64]:
        rows_theo.append({
            "architecture": f"SIEVE ({n_roles} roles)",
            "n_vectors": N,
            "total_bytes": theo[f"sieve_{n_roles}roles_bytes"],
            "vs_rbac_hnsw_multiplier": theo[f"sieve_{n_roles}roles_vs_rbac_multiplier"],
        })
    rows_theo.append({"architecture": "RBAC-HNSW (ours)", "n_vectors": N,
                       "total_bytes": theo["rbac_hnsw_bytes"], "vs_rbac_hnsw_multiplier": 1.0})
    rows_theo.append({"architecture": "HNSW (no RBAC)", "n_vectors": N,
                       "total_bytes": theo["hnsw_baseline_bytes"], "vs_rbac_hnsw_multiplier":
                       theo["hnsw_baseline_bytes"] / theo["rbac_hnsw_bytes"]})

    df_theo = pd.DataFrame(rows_theo)

    # ── Empirical measurement (smaller dataset for speed) ───────────────────
    df_emp = measure_empirical_memory(n_vectors=100_000, dim=768)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_theo.to_csv(RESULTS_DIR / "exp3_memory_theoretical.csv", index=False)
    df_emp.to_csv(RESULTS_DIR / "exp3_memory_empirical.csv", index=False)
    print(f"\nResults saved → {RESULTS_DIR}")
    return df_emp


if __name__ == "__main__":
    run_experiment()
