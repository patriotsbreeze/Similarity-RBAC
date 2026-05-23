"""
baselines.py — Post-filtering and Pre-filtering Baselines
==========================================================
Implements the two standard literature baselines for filtered ANN search.

Baseline 1: Post-filtering (Section 2.1)
-----------------------------------------
Standard HNSW search retrieves a large candidate set of size ef >> k,
then discards vectors the querying user lacks permission to see.

Failure mode (low selectivity): at 0.1 % selectivity only 1000 of the 1M
vectors are accessible. An ef=200 beam search explores only ~200 candidates —
far fewer than the ~100,000 needed to probabilistically hit 10 accessible
neighbours.  Result: recall collapses to near 0 % (see Experiment 2).

Literature: Malkov & Yashunin (2020), §6 "comparison with filtering".

Baseline 2: Pre-filtering / Brute-Force Linear Scan (Section 2.2)
------------------------------------------------------------------
For each query, first apply the RBAC gate to build an accessible subset,
then compute exact nearest neighbours over that subset.

Failure mode (high selectivity): at 80 % selectivity the accessible set
contains 800,000 vectors.  A brute-force linear scan requires 800,000
distance computations per query → QPS ≈ 1–2 on modern hardware.

Literature: FilteredDiskANN (Gollapudi et al. 2023) §4 "naive baselines".

Both baselines serve as ground-truth recall references:
  * Pre-filtering provides *exact* results for the accessible subset.
  * These exact results are the gold standard for recall@k evaluation.
"""

from __future__ import annotations

import time
import numpy as np
import hnswlib
from typing import Tuple, Dict, Optional


class PostFilterBaseline:
    """
    Standard HNSW with post-filtering.

    Retrieves ef_multiplier × k candidates from a vanilla HNSW index, then
    filters by RBAC mask.  ef_multiplier is tuned at search time.

    Per Malkov & Yashunin (2020): optimal ef for recall@10 ≈ 100–500.
    At low selectivity we inflate ef aggressively (√(1/sel)) following the
    analysis in ACORN (Patel et al. 2024).
    """

    def __init__(self, dim: int, space: str = "cosine", M: int = 16,
                 ef_construction: int = 200) -> None:
        self.dim = dim
        self._index = hnswlib.Index(space=space, dim=dim)
        self._index.init_index(max_elements=0, ef_construction=ef_construction, M=M)
        self._masks: Dict[int, int] = {}
        self.build_time_s = 0.0

    def add_items(self, vectors: np.ndarray, masks: np.ndarray,
                  ids: Optional[np.ndarray] = None) -> None:
        n = len(vectors)
        if ids is None:
            ids = np.arange(n, dtype=np.int64)
        t0 = time.perf_counter()
        self._index.resize_index(self._index.get_current_count() + n)
        self._index.add_items(vectors, ids)
        for i, label in enumerate(ids):
            self._masks[int(label)] = int(masks[i])
        self.build_time_s += time.perf_counter() - t0

    def search(self, query: np.ndarray, k: int, query_mask: int,
               ef: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve ef candidates, then filter.

        ef is set externally; callers should tune it per selectivity level
        to make the comparison fair (i.e. give post-filtering its best shot).
        """
        self._index.set_ef(ef)
        qm = np.uint64(query_mask)

        def _gate(label: int) -> bool:
            return (np.uint64(self._masks.get(label, 0)) & qm) == qm

        labels, distances = self._index.knn_query(query.reshape(1, -1), k=k,
                                                   filter=_gate)
        return labels[0], distances[0]

    def batch_search(self, queries: np.ndarray, k: int, query_mask: int,
                     ef: int = 200, num_threads: int = 1
                     ) -> Tuple[np.ndarray, np.ndarray]:
        self._index.set_ef(ef)
        qm = np.uint64(query_mask)

        def _gate(label: int) -> bool:
            return (np.uint64(self._masks.get(label, 0)) & qm) == qm

        accessible_count = sum(
            1 for m in self._masks.values()
            if (np.uint64(m) & qm) == qm
        )
        nq = len(queries)
        if accessible_count == 0:
            return (np.full((nq, k), -1, dtype=np.int64),
                    np.full((nq, k), np.inf, dtype=np.float32))

        k_eff = max(1, min(k, accessible_count))

        try:
            labels, distances = self._index.knn_query(
                queries, k=k_eff, filter=_gate, num_threads=num_threads)
        except RuntimeError:
            return (np.full((nq, k), -1, dtype=np.int64),
                    np.full((nq, k), np.inf, dtype=np.float32))

        # Pad columns to k
        if labels.shape[1] < k:
            pad = k - labels.shape[1]
            labels    = np.pad(labels,    ((0,0),(0,pad)), constant_values=-1)
            distances = np.pad(distances, ((0,0),(0,pad)), constant_values=np.inf)
        return labels, distances


class PreFilterBaseline:
    """
    Brute-force linear scan over the access-permitted subset.

    Returns *exact* nearest neighbours — serves as ground truth for recall
    computation.  QPS is O(|accessible| × dim) which makes it impractical at
    high selectivity but exact at all selectivity levels.

    Algorithm
    ---------
    1. Apply the RBAC gate:  accessible_ids = {i : (masks[i] & qmask) == qmask}
    2. Compute dot products between query and accessible[sub-batch] (vectorised).
    3. Return top-k by ascending cosine distance (1 − dot for unit vectors).

    For cosine similarity on unit-norm vectors:
        cosine_distance(a, b) = 1 − dot(a, b)
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._vectors: Optional[np.ndarray] = None
        self._masks:   Optional[np.ndarray] = None
        self._ids:     Optional[np.ndarray] = None

    def add_items(self, vectors: np.ndarray, masks: np.ndarray,
                  ids: Optional[np.ndarray] = None) -> None:
        """
        Store vectors and masks.  Appends to any previously added data.
        Vectors must already be L2-normalised.
        """
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
        """Exact nearest neighbours in the accessible subset."""
        qm = np.uint64(query_mask)
        accessible_mask = (self._masks & qm) == qm
        accessible_idx  = np.where(accessible_mask)[0]

        if len(accessible_idx) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        accessible_vecs = self._vectors[accessible_idx]
        # Cosine distance = 1 − dot (vectors are unit-normalised)
        dots      = accessible_vecs @ query
        distances = (1.0 - dots).astype(np.float32)

        k_actual  = min(k, len(accessible_idx))
        top_idx   = np.argpartition(distances, k_actual - 1)[:k_actual]
        top_idx   = top_idx[np.argsort(distances[top_idx])]

        labels    = self._ids[accessible_idx[top_idx]]
        return labels, distances[top_idx]

    def batch_search(self, queries: np.ndarray, k: int,
                     query_mask: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch exact search.  Vectorised over queries for efficiency.
        Memory: O(|accessible| × nq) floats — may require chunking for large nq.
        """
        qm = np.uint64(query_mask)
        accessible_mask = (self._masks & qm) == qm
        accessible_idx  = np.where(accessible_mask)[0]

        if len(accessible_idx) == 0:
            empty_l = np.full((len(queries), k), -1, dtype=np.int64)
            empty_d = np.full((len(queries), k), np.inf, dtype=np.float32)
            return empty_l, empty_d

        accessible_vecs = self._vectors[accessible_idx]   # (n_acc, dim)
        # (nq, n_acc) dot product matrix — vectorised BLAS
        dots      = queries @ accessible_vecs.T            # (nq, n_acc)
        distances = (1.0 - dots).astype(np.float32)       # (nq, n_acc)

        n_acc     = len(accessible_idx)
        k_actual  = min(k, n_acc)

        all_labels    = np.full((len(queries), k), -1, dtype=np.int64)
        all_distances = np.full((len(queries), k), np.inf, dtype=np.float32)

        for qi in range(len(queries)):
            top_idx = np.argpartition(distances[qi], k_actual - 1)[:k_actual]
            top_idx = top_idx[np.argsort(distances[qi, top_idx])]
            all_labels[qi,    :k_actual] = self._ids[accessible_idx[top_idx]]
            all_distances[qi, :k_actual] = distances[qi, top_idx]

        return all_labels, all_distances

    def memory_bytes(self) -> Dict[str, int]:
        n = len(self._vectors) if self._vectors is not None else 0
        return {
            "vectors_bytes": n * self.dim * 4,
            "masks_bytes":   n * 8,
            "total_bytes":   n * self.dim * 4 + n * 8,
        }
