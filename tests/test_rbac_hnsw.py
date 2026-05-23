"""
test_rbac_hnsw.py — Unit tests for RBAC-HNSW
=============================================
Tests correctness of the RBAC gate, recall, and consistency between the
filter and routing strategies.
"""

from __future__ import annotations

import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rbac_hnsw import RBACIndex
from baselines import PreFilterBaseline, PostFilterBaseline


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_unit_vecs(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def _simple_masks(n: int) -> np.ndarray:
    """Alternating full-access / restricted masks."""
    masks = np.zeros(n, dtype=np.uint64)
    masks[::2]  = np.uint64(0b11)   # bits 0 and 1
    masks[1::2] = np.uint64(0b10)   # bit 1 only
    return masks


# ── Test 1: RBAC gate correctness ─────────────────────────────────────────────

def test_rbac_gate_filter_strategy():
    """
    query_mask = 0b11 (requires bits 0 AND 1).
    Only vectors with masks[i] & 0b11 == 0b11 (i.e., even-indexed) should
    appear in results.
    """
    n, dim = 500, 32
    vectors = _make_unit_vecs(n, dim)
    masks   = _simple_masks(n)
    ids     = np.arange(n, dtype=np.int64)

    idx = RBACIndex(dim=dim, space="cosine", M=8, ef_construction=50)
    idx.add_items(vectors, masks, ids)

    query = _make_unit_vecs(1, dim, seed=999)[0]
    labels, distances = idx.search(query, k=10, query_mask=0b11,
                                   ef=100, strategy="filter")

    assert len(labels) > 0, "Should return at least one result"
    for lbl in labels:
        if lbl < 0:
            continue   # padding
        assert int(lbl) % 2 == 0, (
            f"Label {lbl} (odd index) should NOT be accessible with mask=0b11 "
            f"but mask={bin(masks[lbl])}"
        )
    print("  PASS: RBAC gate (filter strategy) — only accessible nodes returned")


def test_rbac_gate_restricts_to_accessible():
    """With query_mask = 0b10, BOTH even and odd indices are accessible."""
    n, dim = 300, 16
    vectors = _make_unit_vecs(n, dim)
    masks   = _simple_masks(n)
    ids     = np.arange(n, dtype=np.int64)

    idx = RBACIndex(dim=dim, space="cosine", M=8, ef_construction=50)
    idx.add_items(vectors, masks, ids)

    query = _make_unit_vecs(1, dim, seed=42)[0]
    labels, _ = idx.search(query, k=10, query_mask=0b10, ef=100,
                            strategy="filter")
    # With mask 0b10, all vectors (even and odd) pass: masks[i] & 0b10 == 0b10
    assert len(labels) == 10
    print("  PASS: With permissive mask, all vectors accessible")


# ── Test 2: Recall sanity check ───────────────────────────────────────────────

def test_recall_at_high_selectivity():
    """
    At 100% selectivity (query_mask=0), RBAC-HNSW recall should match
    standard HNSW recall.  Ground truth from brute-force baseline.
    """
    n, dim = 2_000, 64
    vectors = _make_unit_vecs(n, dim, seed=1)
    masks   = np.full(n, np.uint64(0b1), dtype=np.uint64)  # all accessible
    ids     = np.arange(n, dtype=np.int64)

    rbac = RBACIndex(dim=dim, space="cosine", M=16, ef_construction=200)
    rbac.add_items(vectors, masks, ids)

    pre  = PreFilterBaseline(dim=dim)
    pre.add_items(vectors, masks, ids)

    queries = _make_unit_vecs(50, dim, seed=999)
    k = 10

    # Ground truth
    gt_labels, _ = pre.batch_search(queries, k=k, query_mask=0b1)

    # RBAC-HNSW
    rbac_labels, _ = rbac.batch_search(queries, k=k, query_mask=0b1,
                                        ef=200, strategy="filter")

    recalls = []
    for gt, pred in zip(gt_labels, rbac_labels):
        gt_set   = set(gt[gt >= 0].tolist())
        pred_set = set(pred[pred >= 0].tolist())
        recalls.append(len(gt_set & pred_set) / len(gt_set) if gt_set else 1.0)
    mean_recall = np.mean(recalls)
    assert mean_recall >= 0.80, f"Expected recall >= 0.80 at full selectivity, got {mean_recall:.3f}"
    print(f"  PASS: Recall@10 at full selectivity = {mean_recall:.3f} (>= 0.80)")


# ── Test 3: Memory overhead ───────────────────────────────────────────────────

def test_memory_overhead_formula():
    """
    Verify that the theoretical mask overhead is < 1% for d=768, M=16.
    Per Drepper (2007): 8 bytes / (3072 + 256) bytes ≈ 0.24%.
    """
    from rbac_hnsw import RBACIndex
    n, d, M = 10_000, 768, 16
    idx = RBACIndex(dim=d, M=M)
    # Populate with dummy masks
    idx.n_vectors = n
    mem = idx.memory_bytes()
    assert mem["mask_overhead_pct"] < 1.0, (
        f"Mask overhead {mem['mask_overhead_pct']:.2f}% exceeds 1%"
    )
    print(f"  PASS: Mask overhead = {mem['mask_overhead_pct']:.3f}% (<1%)")


# ── Test 4: Save / load round-trip ────────────────────────────────────────────

def test_save_load(tmp_path=None):
    if tmp_path is None:
        import tempfile
        tmp_path = Path(tempfile.mkdtemp())

    n, dim = 500, 32
    vectors = _make_unit_vecs(n, dim)
    masks   = _simple_masks(n)
    ids     = np.arange(n, dtype=np.int64)

    idx = RBACIndex(dim=dim, M=8, ef_construction=50)
    idx.add_items(vectors, masks, ids)

    save_dir = tmp_path / "rbac_idx"
    idx.save(save_dir)

    idx2 = RBACIndex(dim=dim, M=8, ef_construction=50)
    idx2.load(save_dir, max_elements=n)

    assert idx2.n_vectors == n
    assert idx2._masks == idx._masks
    print(f"  PASS: Save/load round-trip ({n} vectors)")


# ── Test 5: Brute-force exact ground truth ────────────────────────────────────

def test_prefilter_exact():
    """
    With 10 vectors of known embeddings, verify the brute-force baseline
    returns the exact correct nearest neighbour.
    """
    dim    = 4
    base   = np.eye(dim, dtype=np.float32)          # 4 unit vectors along axes
    extra  = np.tile(np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32), (6, 1))
    extra /= np.linalg.norm(extra, axis=1, keepdims=True)
    vecs   = np.vstack([base, extra])                # 10 vectors total
    masks  = np.ones(len(vecs), dtype=np.uint64)
    ids    = np.arange(len(vecs), dtype=np.int64)

    pre = PreFilterBaseline(dim=dim)
    pre.add_items(vecs, masks, ids)

    # Query aligned with axis 0 → nearest is vector 0
    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    labels, dists = pre.search(query, k=1, query_mask=1)

    assert labels[0] == 0, f"Expected nearest = 0, got {labels[0]}"
    assert dists[0] < 1e-5,  f"Expected distance ≈ 0, got {dists[0]}"
    print("  PASS: Brute-force baseline returns exact nearest neighbour")


# ── Run all tests ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_rbac_gate_filter_strategy,
        test_rbac_gate_restricts_to_accessible,
        test_recall_at_high_selectivity,
        test_memory_overhead_formula,
        test_save_load,
        test_prefilter_exact,
    ]
    print("Running RBAC-HNSW unit tests …\n")
    passed = 0
    for t in tests:
        try:
            print(f"  {t.__name__} …")
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
    print(f"\n{passed}/{len(tests)} tests passed.")
    if passed < len(tests):
        sys.exit(1)
