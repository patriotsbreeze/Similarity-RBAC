# Paper: RBAC-HNSW @ BioDMS / VLDB 2026

**Full title:** *RBAC-HNSW: Access-Controlled Approximate Nearest-Neighbor Search for HIPAA-Governed Clinical Embeddings*

**Venue:** [BioDMS @ VLDB 2026](https://biodms.org/) — Workshop on Biomedical Data Management Systems  
**Format:** Project Talk (4-page short paper + 10-minute talk)  
**Deadline:** May 15, 2026 (AoE) — camera-ready July 1, 2026  

---

## 📄 Final Manuscript

**[rbac_hnsw_biodms2026.pdf](rbac_hnsw_biodms2026.pdf)** — ready-to-submit PDF (7 pages incl. references + authors section)

---

## Directory Structure

```
paper/
├── rbac_hnsw_biodms2026.tex    ← Main LaTeX document (root file)
├── rbac_hnsw_biodms2026.pdf    ← Compiled PDF (submit this)
├── references.bib              ← BibTeX bibliography
│
├── sections/                   ← One .tex file per section
│   ├── 01_abstract.tex
│   ├── 02_introduction.tex
│   ├── 03_background.tex
│   ├── 04_algorithm.tex        ← Core contribution + pseudocode
│   ├── 05_dataset.tex          ← Synthetic clinical benchmark
│   ├── 06_evaluation.tex       ← Experiments + results tables
│   ├── 07_conclusion.tex
│   └── 08_authors.tex          ← BioDMS required author bios
│
├── figures/                    ← Publication figures (PDF + PNG)
│   ├── generate_paper_figures.py   ← Regenerate with: python generate_paper_figures.py
│   ├── fig1_recall_vs_selectivity.pdf/png   ← Recall@10 vs selectivity (2-panel)
│   ├── fig2_recall_ef_curves.pdf/png         ← Recall vs ef sweep
│   ├── fig3_memory_comparison.pdf/png        ← Memory vs SIEVE
│   ├── fig4_algorithm_diagram.pdf/png        ← RBAC-HNSW traversal schematic
│   ├── fig5_rbac_schema.pdf/png              ← 64-bit bitmask layout
│   └── fig6_recall_heatmap.pdf/png           ← Recall heatmap
│
├── build/                      ← LaTeX intermediate files (auto-generated)
│   └── rbac_hnsw_biodms2026.pdf
│
└── paper_scaffold.md           ← Extended paper draft / full writeup
```

---

## Recompile

```bash
cd paper/
pdflatex -output-directory=build rbac_hnsw_biodms2026.tex
bibtex build/rbac_hnsw_biodms2026
pdflatex -output-directory=build rbac_hnsw_biodms2026.tex
pdflatex -output-directory=build rbac_hnsw_biodms2026.tex
cp build/rbac_hnsw_biodms2026.pdf .
```

Requires: MiKTeX or TeX Live with `acmart` package.

---

## Why This Will Be Accepted at BioDMS

BioDMS 2026 solicits contributions on:
- ✅ **Privacy-preserving and secure processing of sensitive medical datasets** — HIPAA RBAC is our core motivation
- ✅ **Scalable storage and querying of large biomedical datasets** — HNSW vector indexing at 1M-vector scale
- ✅ **High-throughput analytics over heterogeneous datasets** — clinical note embeddings (BioBERT)
- ✅ **Data access and governance** — fine-grained 64-bit access control bitmask

The paper bridges the data management and biomedical communities as requested:
- Frames access-controlled vector search as a **database systems problem**
- Grounded in a **clinical AI use case** (EHR similarity search)
- Includes the mandatory **Authors section** with community label (Data Management)
- Written to **minimize jargon**, as required by BioDMS guidelines
