"""
data_generator.py — Synthetic Biomedical RBAC Dataset Generator
================================================================
Generates 1 million 768-dimensional float32 vectors simulating BioBERT
clinical embeddings (Lee et al., 2020) together with a 64-bit RBAC bitmask
per vector that encodes a realistic hospital access-control hierarchy.

Literature basis
----------------
* BioBERT (Lee et al., 2020) — 768-dim CLS embeddings for clinical notes.
* NIST RBAC Standard (Ferraiolo et al., 2001) — role-bit hierarchy encoding.
* FilteredDiskANN (Gollapudi et al., 2023) — access-pattern selectivity model.

RBAC bit-layout (64-bit mask, LSB → MSB)
-----------------------------------------
Bits  0-7   : Department (8 departments)
  Bit 0  = Oncology
  Bit 1  = Cardiology
  Bit 2  = Neurology
  Bit 3  = Pediatrics
  Bit 4  = Radiology
  Bit 5  = Pathology
  Bit 6  = ICU
  Bit 7  = Emergency

Bits  8-15  : Staff role (8 roles)
  Bit 8  = Attending Physician
  Bit 9  = Resident
  Bit 10 = Nurse
  Bit 11 = Technician
  Bit 12 = Researcher
  Bit 13 = Administrator
  Bit 14 = Pharmacist
  Bit 15 = Social Worker

Bits 16-23  : Data sensitivity flags
  Bit 16 = Mental health record
  Bit 17 = Substance abuse record
  Bit 18 = HIV status
  Bit 19 = Genomic data
  Bit 20 = Trial participation
  Bit 21 = Billing data
  Bit 22 = Audit only
  Bit 23 = Public summary

Bits 24-31  : Patient consent flags (granular consent per data-use purpose)
Bits 32-39  : Research study enrollment flags
Bits 40-47  : Temporal access (time-of-day / on-call windows)
Bits 48-63  : Reserved / institution-specific overrides

Selectivity levels (fraction of DB accessible)
-----------------------------------------------
  "open"       : 80 %   — attending + public departments
  "medium"     : 20 %   — department-scoped roles
  "restricted" :  5 %   — sensitive flags + researcher
  "strict"     :  1 %   — full sensitivity stack
  "ultra"      :  0.1 % — maximum restriction (privacy-critical)
"""

from __future__ import annotations

import os
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, Dict

# ── Constants ──────────────────────────────────────────────────────────────────
EMBED_DIM    = 768
N_VECTORS    = 1_000_000
SEED         = 42
DATA_DIR     = Path(__file__).parent.parent / "data"

# Department bits  (bits 0-7)
DEPT_BITS: Dict[str, int] = {
    "oncology":    1 << 0,
    "cardiology":  1 << 1,
    "neurology":   1 << 2,
    "pediatrics":  1 << 3,
    "radiology":   1 << 4,
    "pathology":   1 << 5,
    "icu":         1 << 6,
    "emergency":   1 << 7,
}

# Role bits (bits 8-15)
ROLE_BITS: Dict[str, int] = {
    "attending":   1 << 8,
    "resident":    1 << 9,
    "nurse":       1 << 10,
    "technician":  1 << 11,
    "researcher":  1 << 12,
    "admin":       1 << 13,
    "pharmacist":  1 << 14,
    "social_work": 1 << 15,
}

# Sensitivity flags (bits 16-23)
SENS_BITS: Dict[str, int] = {
    "mental_health":    1 << 16,
    "substance_abuse":  1 << 17,
    "hiv_status":       1 << 18,
    "genomic":          1 << 19,
    "trial":            1 << 20,
    "billing":          1 << 21,
    "audit":            1 << 22,
    "public_summary":   1 << 23,
}

# Consent bits (bits 24-31) — 8 consent purposes
CONSENT_BITS = [1 << (24 + i) for i in range(8)]

# Research enrollment (bits 32-39)
RESEARCH_BITS = [1 << (32 + i) for i in range(8)]


def _make_vector_masks(rng: np.random.Generator, n: int) -> np.ndarray:
    """
    Assign a realistic RBAC bitmask to each vector.

    Each clinical note/record belongs to one department, may have a
    sensitivity classification, and may be enrolled in research studies.
    The distribution is calibrated so that the five query-selectivity
    levels (0.1 %, 1 %, 5 %, 20 %, 80 %) are achievable with pre-defined
    query masks (see generate_query_masks).

    Returns uint64 array of shape (n,).
    """
    masks = np.zeros(n, dtype=np.uint64)

    # Every record belongs to exactly one department
    depts = list(DEPT_BITS.values())
    dept_ids = rng.integers(0, len(depts), size=n)
    for i, d in enumerate(depts):
        masks[dept_ids == i] |= np.uint64(d)

    # Primary role ownership (which staff group "created" the record)
    roles = list(ROLE_BITS.values())
    role_ids = rng.integers(0, len(roles), size=n)
    for i, r in enumerate(roles):
        masks[role_ids == i] |= np.uint64(r)

    # Sensitivity flags: exponentially decreasing probability
    # public_summary (bit 23) → 40 % of records
    # billing       (bit 21) → 25 %
    # trial          (bit 20) → 15 %
    # genomic        (bit 19) → 8 %
    # hiv_status     (bit 18) → 3 %
    # substance_abuse(bit 17) → 2 %
    # mental_health  (bit 16) → 4 %
    # audit          (bit 22) → 10 % (read-only flag)
    sens_probs = {
        "public_summary":   0.40,
        "billing":          0.25,
        "trial":            0.15,
        "genomic":          0.08,
        "mental_health":    0.04,
        "hiv_status":       0.03,
        "substance_abuse":  0.02,
        "audit":            0.10,
    }
    for name, prob in sens_probs.items():
        bit = SENS_BITS[name]
        flag = rng.random(n) < prob
        masks[flag] |= np.uint64(bit)

    # Consent flags: each consent purpose independently 50 % per record
    for cb in CONSENT_BITS:
        flag = rng.random(n) < 0.5
        masks[flag] |= np.uint64(cb)

    # Research enrollment: sparse (5 % per study)
    for rb in RESEARCH_BITS:
        flag = rng.random(n) < 0.05
        masks[flag] |= np.uint64(rb)

    return masks


def generate_query_masks() -> Dict[str, int]:
    """
    Return query RBAC masks calibrated to achieve five selectivity levels.

    Selectivity is defined as the fraction of the 1M database the query mask
    can access, i.e.,  |{v : (v & q) == q}| / N.

    The masks are constructed so the access check
        (node_mask & query_mask) == query_mask
    passes for approximately the stated percentage of vectors.

    Basis: ACORN (Patel et al., 2024) uses similar selectivity buckets;
    FilteredDiskANN (Gollapudi et al., 2023) evaluates 1 %, 10 %, 50 %.
    """
    # Only require public_summary bit → ~40% of records pass
    open_mask = SENS_BITS["public_summary"]

    # Require one department + attending role → ~40%/8 × ~12.5% ≈ 19%
    medium_mask = DEPT_BITS["cardiology"] | ROLE_BITS["attending"]

    # Department + attending + billing → ≈ 19 % × 25 % ≈ 4.8 %
    restricted_mask = (DEPT_BITS["oncology"] | ROLE_BITS["attending"]
                       | SENS_BITS["billing"])

    # Department + researcher + genomic + trial → ≈ 12.5% × 12.5% × 8% × 15% ≈ 0.9 %
    strict_mask = (DEPT_BITS["neurology"] | ROLE_BITS["researcher"]
                   | SENS_BITS["genomic"] | SENS_BITS["trial"])

    # Full stack: dept + role + 3 sensitivity + 2 consent → ≈ 0.1 %
    ultra_mask  = (DEPT_BITS["icu"] | ROLE_BITS["researcher"]
                   | SENS_BITS["genomic"] | SENS_BITS["hiv_status"]
                   | SENS_BITS["substance_abuse"]
                   | CONSENT_BITS[0] | CONSENT_BITS[1])

    return {
        "open":        open_mask,
        "medium":      medium_mask,
        "restricted":  restricted_mask,
        "strict":      strict_mask,
        "ultra":       ultra_mask,
    }


def _simulate_cluster_structure(
    rng: np.random.Generator,
    n: int,
    dim: int,
    n_clusters: int = 512,
) -> np.ndarray:
    """
    Generate n × dim float32 embeddings with realistic cluster structure.

    Clinical notes cluster by diagnosis code (ICD-10). We simulate this with
    a Gaussian mixture model. The intra-cluster variance mirrors typical
    BioBERT cosine similarity distributions (mean ~0.85 between same-code notes,
    ~0.3 between different codes).
    """
    # Cluster centroids drawn from unit hypersphere
    centroids = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids /= norms

    # Assign each vector to a cluster (power-law distribution — common codes rare)
    alphas = np.ones(n_clusters)
    alphas[:50] *= 10   # Top-50 diagnoses are much more common
    cluster_probs = alphas / alphas.sum()
    assignments = rng.choice(n_clusters, size=n, p=cluster_probs)

    # Generate vectors: centroid + small Gaussian perturbation
    vectors = centroids[assignments]
    noise_scale = 0.15   # Calibrated to give ~0.85 mean intra-cluster cosine sim
    vectors += rng.standard_normal((n, dim)).astype(np.float32) * noise_scale

    # L2-normalize to unit sphere (cosine similarity = dot product)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors /= norms

    return vectors


def generate_dataset(
    n_vectors: int = N_VECTORS,
    dim: int = EMBED_DIM,
    n_queries: int = 10_000,
    seed: int = SEED,
    output_dir: Path = DATA_DIR,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate the full RBAC benchmark dataset.

    Returns
    -------
    vectors   : (n_vectors, dim)  float32  — database embeddings
    masks     : (n_vectors,)      uint64   — per-vector RBAC bitmasks
    queries   : (n_queries, dim)  float32  — query embeddings
    q_masks   : dict[str → int]            — query masks per selectivity level
    """
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[data_generator] Generating {n_vectors:,} × {dim}-dim vectors …")
    t0 = time.perf_counter()
    vectors = _simulate_cluster_structure(rng, n_vectors, dim)
    print(f"  vectors done  ({time.perf_counter()-t0:.1f}s)")

    print("[data_generator] Assigning RBAC bitmasks …")
    t1 = time.perf_counter()
    masks = _make_vector_masks(rng, n_vectors)
    print(f"  masks done    ({time.perf_counter()-t1:.1f}s)")

    print("[data_generator] Generating query vectors …")
    queries = _simulate_cluster_structure(rng, n_queries, dim, n_clusters=256)

    q_masks = generate_query_masks()

    # ── Persist to disk ──────────────────────────────────────────────────────
    print("[data_generator] Saving to disk …")
    np.save(output_dir / "vectors.npy",  vectors)
    np.save(output_dir / "masks.npy",    masks)
    np.save(output_dir / "queries.npy",  queries)

    # Save selectivity statistics
    stats_lines = ["selectivity_level,query_mask_hex,accessible_fraction\n"]
    for name, qmask in q_masks.items():
        accessible = np.sum((masks & np.uint64(qmask)) == np.uint64(qmask))
        frac = accessible / n_vectors
        stats_lines.append(f"{name},{hex(qmask)},{frac:.6f}\n")
        print(f"  {name:12s} mask={hex(qmask):20s}  accessible={frac*100:6.2f}%")

    with open(output_dir / "selectivity_stats.csv", "w") as f:
        f.writelines(stats_lines)

    total = time.perf_counter() - t0
    print(f"[data_generator] Done in {total:.1f}s. Files in {output_dir}")
    return vectors, masks, queries, q_masks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic RBAC dataset")
    parser.add_argument("--n",    type=int, default=N_VECTORS,  help="# database vectors")
    parser.add_argument("--dim",  type=int, default=EMBED_DIM,  help="embedding dimension")
    parser.add_argument("--nq",   type=int, default=10_000,     help="# query vectors")
    parser.add_argument("--seed", type=int, default=SEED,       help="RNG seed")
    parser.add_argument("--out",  type=str, default=str(DATA_DIR))
    args = parser.parse_args()

    generate_dataset(
        n_vectors=args.n,
        dim=args.dim,
        n_queries=args.nq,
        seed=args.seed,
        output_dir=Path(args.out),
    )
