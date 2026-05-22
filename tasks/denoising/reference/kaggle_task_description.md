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

**Poisson is a hard constraint.** Submissions are rejected if:
```
poisson_norm = (0.257575 − poisson) / (0.257575 − 0.031739) < 0.97
```

**Final score = MSE norm** (conditional on passing Poisson):
```
score = (0.304721 − mse) / (0.304721 − 0.000000)    ∈ [0, 1], higher is better
```

|                  | MSE    | Poisson |
|------------------|--------|---------|
| Baseline (MAGIC) | 0.3047 | 0.2576  |
| Perfect          | 0.0000 | 0.0317  |

The Poisson constraint is intentionally designed to detect overfitting: a method that memorizes pancreas structure typically collapses on PBMC and Tabula.

## Baselines & Approach

**MAGIC** (graph diffusion over a k-NN graph built on sqrt-transformed counts) is the canonical baseline. It is tissue-agnostic by design — it adapts to the local manifold structure of whatever dataset it receives.

Methods that tend to generalize: graph diffusion, PCA-based smoothing, low-rank approximation, manifold imputation.  
Methods that tend to fail on held-out tissues: autoencoders trained on pancreas, tissue-specific priors, fixed hyperparameters tuned by grid search on pancreas.

## Available Libraries

`numpy` · `scipy` · `sklearn` · `graphtools` · `scprep` · `scanpy` · `anndata` · `molecular_cross_validation`
