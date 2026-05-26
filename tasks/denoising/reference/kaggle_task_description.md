# scRNA-seq Denoising — Generalization Benchmark

## Overview

Single-cell RNA sequencing captures gene expression at cellular resolution, but the data is fundamentally noisy: low capture efficiency and technical dropout mean that many transcripts go undetected. The goal of this benchmark is to recover true expression levels from raw count data — and to do so in a way that **generalizes across tissues and cell types**.

Your submission is a single Python function. You develop and tune it on a pancreatic islet dataset. It is then evaluated without modification on two held-out tissues it has never seen.

## Dataset

**Development set** — `pancreas.h5ad`: 1,937 pancreatic islet cells × 1,550 genes. Raw UMI counts, stored as a sparse matrix in AnnData format.

**Private test sets** (never visible during development):
- **PBMC** — 1,087 human blood immune cells (T cells, B cells, monocytes) × 15,098 genes
- **Tabula** — 24,540 cells from a multi-tissue atlas × 16,137 genes

Evaluation uses **Molecular Cross-Validation**: molecules within each cell are randomly split 90/10 (train/test). The denoiser sees the 90% train split and is scored against the held-out 10%.

## Task

Implement a function with the following signature:

```python
def magic_denoise(X, **kwargs):
    # X: numpy array (n_cells, n_genes) — raw count matrix
    # Returns: denoised array of same shape, non-negative floats
```

All helper functions must be top-level. No filesystem or network I/O inside the function.

## Evaluation

Two metrics are computed on the held-out 10% of molecules:

| Metric | Description | Direction |
|--------|-------------|-----------|
| **Poisson NLL** | Poisson negative log-likelihood between predicted and held-out counts | Lower is better |
| **MSE** | Mean squared error in log-normalized space (normalize_total → log1p) | Lower is better |

### Scoring

Both metrics are normalized to [0, 1] and averaged:

```
mse_norm     = (0.304721 − mse)     / (0.304721 − 0.000000)
poisson_norm = (0.257575 − poisson) / (0.257575 − 0.031739)
score        = (mse_norm + poisson_norm) / 2    ∈ [0, 1], higher is better
```

|                       | MSE    | Poisson | Score |
|-----------------------|--------|---------|-------|
| Baseline (no denoise) | 0.3047 | 0.2576  | 0.00  |
| MAGIC (A,R)           | 0.2314 | 0.0369  | ~0.61 |
| Perfect               | 0.0000 | 0.0317  | 1.00  |

A method that overfits pancreas will typically produce very high Poisson NLL on held-out tissues, which directly penalises the score.

## Baselines & Approach

**Your starting point is MAGIC (A,R)** — MAGIC with approximate solver and reversed normalization order. It scores ~0.61 on the development set (pancreas) and **~0.64 on the private held-out sets**, matching the best classical baseline from the TTT-Discover paper.

```python
import magic, numpy as np, scprep

def magic_denoise(X, **kwargs):
    X = np.asarray(X, dtype=float)
    X_sqrt = np.sqrt(X)
    X_norm, libsize = scprep.normalize.library_size_normalize(
        X_sqrt, rescale=1, return_library_size=True
    )
    Y = magic.MAGIC(solver="approximate", verbose=False).fit_transform(
        X_norm, genes="all_genes"
    )
    Y = np.asarray(Y) ** 2
    return np.maximum(Y * libsize[:, np.newaxis], 0)
```

The key insight is **reversed normalization order**: apply sqrt variance-stabilization *before* library-size normalization (not after), then undo both transforms after diffusion. This substantially improves the Poisson score compared to vanilla MAGIC.

**Your goal is to beat 0.64.** Scores above 0.70 have been achieved by LLM-guided search methods (OpenEvolve: 0.71, TTT-Discover: 0.72).

Methods that tend to generalize beyond MAGIC: adaptive bandwidth graph diffusion, PCA-based smoothing with data-driven rank selection, low-rank approximation tuned per-dataset.  
Methods that tend to fail on held-out tissues: autoencoders trained on pancreas, tissue-specific priors, fixed hyperparameters tuned by grid search.

## Available Libraries

`numpy` · `scipy` · `sklearn` · `graphtools` · `scprep` · `scanpy` · `anndata` · `molecular_cross_validation` · `magic-impute` (v3)
