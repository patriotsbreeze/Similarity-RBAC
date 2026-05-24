"""
baselines.py — Post-filtering and Pre-filtering Baselines  (v2)
================================================================
v2: numpy-vectorized accessible-set computation + frozenset filter closure,
matching the v2 RBACIndex approach for fair throughput comparison.
"""

from __future__ import annotations

import time
import numpy as np
import hnswlib
from functools import lru_cache
from typing import Tuple, Dict, Optional


class PostFilterBaseline:
    """
    TRUE post-filtering baseline (v2).

    Algorithm: run vanilla HNSW without any access filter to retrieve ef
    nearest-neighbor candidates, then apply the RBAC bitmask filter to
    the result set.  This is the natural "retrieve-then-filter" approach.

    This is strictly DIFFERENT from RBAC-HNSW which routes through the
    graph with the filter active (hnswlib filter= mechanism).  At low
    selectivity, the difference is critical:

      - Post-filter (this): explores ef nodes, of which ~ef*selectivity are
        accessible.  At strict (0.02%), ef=800 yields ~0.16 accessible
        results → recall ≈ 0.  (Recall collapse.)

      - RBAC-HNSW: hnswlib routes through denied nodes and keeps searching
        until k accessible are found → recall maintained, QPS lower.
    """

    def __init__(self, dim: int, space: str = "cosine",
                 M: int = 16, ef_construction: int = 200) -> None:
        self.dim = dim
        self._space = space
        self._M  = M
        self._ef_construction = ef_construction
        self._index = hnswlib.Index(space=space, dim=dim)
        self._index.init_index(max_elements=0,
                               ef_construction=ef_construction, M=M)
        self._mask_array:  Optional[np.ndarray] = None
        self._label_array: Optional[np.ndarray] = None
        self.build_time_s = 0.0

    def add_items(self, vectors: np.ndarray, masks: np.ndarray,
                  ids: Optional[np.ndarray] = None) -> None:
        n = len(vectors)
        if ids is None:
            ids = np.arange(n, dtype=np.int64)
        t0 = time.perf_counter()
        self._index.resize_index(self._index.get_current_count() + n)
        self._index.add_items(vectors, ids)
        new_masks  = masks.astype(np.uint64)
        new_labels = ids.astype(np.int64)
        if self._mask_array is None:
            self._mask_array  = new_masks
            self._label_array = new_labels
        else:
            self._mask_array  = np.concatenate([self._mask_array,  new_masks])
            self._label_array = np.concatenate([self._label_array, new_labels])
        self._get_accessible_set.cache_clear()
        self.build_time_s += time.perf_counter() - t0

    @lru_cache(maxsize=64)
    def _get_accessible_set(self, query_mask: int) -> frozenset:
        if self._mask_array is None:
            return frozenset()
        qm = np.uint64(query_mask)
        ok = (self._mask_array & qm) == qm
        return frozenset(self._label_array[ok].tolist())

    def batch_search(self, queries: np.ndarray, k: int, query_mask: int,
                     ef: int = 200,
                     num_threads: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        TRUE post-filter: retrieve ef candidates (no access filter during
        search), then filter the result set by the RBAC bitmask.

        At ef candidates, the expected accessible count is ef * selectivity.
        At strict selectivity (0.02%), ef=800 yields ~0.16 accessible → 0
        results → recall collapse.  This is the *correct* baseline to compare
        against RBAC-HNSW's in-search filter.
        """
        accessible = self._get_accessible_set(query_mask)
        nq = len(queries)
        if not accessible:
            return (np.full((nq, k), -1, dtype=np.int64),
                    np.full((nq, k), np.inf, dtype=np.float32))

        # Retrieve ef candidates WITHOUT access filter
        n_candidates = min(ef, self._index.get_current_count())
        if n_candidates < 1:
            return (np.full((nq, k), -1, dtype=np.int64),
                    np.full((nq, k), np.inf, dtype=np.float32))

        self._index.set_ef(n_candidates)     # ef must be >= k for hnswlib
        try:
            raw_labels, raw_dists = self._index.knn_query(
                queries, k=n_candidates, num_threads=num_threads)
        except RuntimeError:
            return (np.full((nq, k), -1, dtype=np.int64),
                    np.full((nq, k), np.inf, dtype=np.float32))

        # Post-filter: keep only accessible results
        result_labels = np.full((nq, k), -1,       dtype=np.int64)
        result_dists  = np.full((nq, k), np.inf,   dtype=np.float32)
        for i, (rl, rd) in enumerate(zip(raw_labels, raw_dists)):
            acc_mask  = np.array([l in accessible for l in rl], dtype=bool)
            acc_l     = rl[acc_mask][:k]
            acc_d     = rd[acc_mask][:k]
            n_found   = len(acc_l)
            if n_found > 0:
                result_labels[i, :n_found] = acc_l
                result_dists[i, :n_found]  = acc_d

        return result_labels, result_dists


class PreFilterBaseline:
    """
    Brute-force exact linear scan over the accessible subset.
    Ground truth for recall computation.  Vectorised via numpy BLAS.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._vectors: Optional[np.ndarray] = None
        self._masks:   Optional[np.ndarray] = None
        self._ids:     Optional[np.ndarray] = None

    def add_items(self, vectors: np.ndarray, masks: np.ndarray,
                  ids: Optional[np.ndarray] = None) -> None:
        n = len(vectors)
        new_ids = np.arange(n, dtype=np.int64) if ids is None else ids
        if self._vectors is None:
            self._vectors = vectors.astype(np.float32)
            self._masks   = masks.astype(np.uint64)
            self._ids     = new_ids.astype(np.int64)
        else:
            self._vectors = np.vstack([self._vectors, vectors.astype(np.float32)])
            self._masks   = np.concatenate([self._masks, masks.astype(np.uint64)])
            self._ids     = np.concatenate([self._ids, new_ids.astype(np.int64)])

    def search(self, query: np.ndarray, k: int,
               query_mask: int) -> Tuple[np.ndarray, np.ndarray]:
        qm = np.uint64(query_mask)
        ok  = (self._masks & qm) == qm
        idx = np.where(ok)[0]
        if len(idx) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        vecs      = self._vectors[idx]
        distances = (1.0 - vecs @ query).astype(np.float32)
        k_actual  = min(k, len(idx))
        top_idx   = np.argpartition(distances, k_actual - 1)[:k_actual]
        top_idx   = top_idx[np.argsort(distances[top_idx])]
        return self._ids[idx[top_idx]], distances[top_idx]

    def batch_search(self, queries: np.ndarray, k: int,
                     query_mask: int,
                     chunk_size: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorised batch exact search.
        Uses chunked matrix multiply to stay within memory limits at 1M scale.
        """
        qm = np.uint64(query_mask)
        ok  = (self._masks & qm) == qm
        idx = np.where(ok)[0]
        nq  = len(queries)
        if len(idx) == 0:
            return (np.full((nq, k), -1, dtype=np.int64),
                    np.full((nq, k), np.inf, dtype=np.float32))

        acc_vecs = self._vectors[idx]   # (n_acc, dim)
        k_actual = min(k, len(idx))
        all_labels    = np.full((nq, k), -1, dtype=np.int64)
        all_distances = np.full((nq, k), np.inf, dtype=np.float32)

        for qi_start in range(0, nq, chunk_size):
            qi_end = min(qi_start + chunk_size, nq)
            q_chunk = queries[qi_start:qi_end]           # (chunk, dim)
            dots    = q_chunk @ acc_vecs.T               # (chunk, n_acc)
            dists   = (1.0 - dots).astype(np.float32)

            for qi_local, qi in enumerate(range(qi_start, qi_end)):
                top_idx = np.argpartition(dists[qi_local], k_actual-1)[:k_actual]
                top_idx = top_idx[np.argsort(dists[qi_local, top_idx])]
                all_labels[qi,    :k_actual] = self._ids[idx[top_idx]]
                all_distances[qi, :k_actual] = dists[qi_local, top_idx]

        return all_labels, all_distances

    def memory_bytes(self) -> Dict[str, int]:
        n = len(self._vectors) if self._vectors is not None else 0
        return {"vectors_bytes": n*self.dim*4, "masks_bytes": n*8,
                "total_bytes": n*self.dim*4 + n*8}
