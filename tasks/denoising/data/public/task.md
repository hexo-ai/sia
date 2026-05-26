# Denoising Task

## Overview

Your task is to implement a `magic_denoise` function that denoises single-cell RNA-seq count data. You will write and iteratively improve a Python function, evaluating it against the pancreas dataset benchmark.

## Problem

Single-cell RNA-seq (scRNA-seq) data is noisy due to technical dropout and low capture efficiency. Given noisy count data, predict the true expression levels.

Your prediction is evaluated against held-out molecules using two metrics:
1. **MSE** - Mean Squared Error in log-normalized space (lower is better)
2. **Poisson Loss** - Poisson negative log-likelihood (lower is better)

## Data Format

- Input `X`: numpy array of shape (n_cells, n_genes) — **raw count data**
- Output: numpy array of same shape — your denoised counts

## Function Signature

```python
def magic_denoise(X, **kwargs):
    # kwargs may include: budget_s, random_state, knn, t, n_pca, solver, decay, knn_max, n_jobs
    # Return denoised counts (same shape as X, non-negative floats)
    return denoised_X
```

## Evaluation

The function is evaluated using these exact metrics:

```python
def evaluate_mse(test_data, denoised):
    """MSE metric in log-normalized space."""
    import anndata, scanpy as sc, sklearn.metrics
    test_adata = anndata.AnnData(X=test_data.copy())
    denoised_adata = anndata.AnnData(X=denoised.copy())
    sc.pp.normalize_total(test_adata, target_sum=10000)
    sc.pp.log1p(test_adata)
    sc.pp.normalize_total(denoised_adata, target_sum=10000)
    sc.pp.log1p(denoised_adata)
    return sklearn.metrics.mean_squared_error(test_adata.X, denoised_adata.X)

def evaluate_poisson(train_data, test_data, denoised):
    """Poisson negative log-likelihood."""
    from molecular_cross_validation.mcv_sweep import poisson_nll_loss
    import scprep
    test_X = scprep.utils.toarray(test_data)
    denoised_X = np.asarray(denoised).copy()
    denoised_scaled = denoised_X * test_X.sum() / max(train_data.sum(), 1e-12)
    return float(poisson_nll_loss(test_X, denoised_scaled).mean())
```

## Scoring

Both metrics are normalized to [0, 1] and averaged:

```
mse_norm     = (0.304721 - mse)     / (0.304721 - 0.000000)
poisson_norm = (0.257575 - poisson) / (0.257575 - 0.031739)
score        = (mse_norm + poisson_norm) / 2
```

Higher is better (range 0–1). MAGIC scores ~0.61 on this development benchmark.

Normalization reference values (computed on pancreas):
- `baseline_mse = 0.304721`, `baseline_poisson = 0.257575`  ← identity (no denoising)
- `perfect_mse = 0.000000`, `perfect_poisson = 0.031739`    ← oracle upper bound

## Available Libraries

numpy, scipy, sklearn, graphtools, scprep, scanpy, anndata, molecular_cross_validation, **magic-impute** (v3)

The `magic-impute` library is installed. The recommended starting point (MAGIC with reversed normalization and approximate solver) scores ~0.61 on pancreas and ~0.64 on held-out datasets:

```python
import magic
import numpy as np
import scprep

def magic_denoise(X, **kwargs):
    X = np.asarray(X, dtype=float)
    # Reversed norm order: sqrt first, then library-size normalize
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

**Start here and improve from this baseline.**

## Rules

1. Write `solution.py` containing your `magic_denoise` function (all helpers must be top-level)
2. Evaluate using `python {dataset_dir}/evaluate.py solution.py` where `{dataset_dir}` is the path given via `--dataset_dir`
3. After each evaluation, a `results.json` file is automatically written to your working directory — do not write it yourself
4. At the end of your run, your working directory **must** contain `solution.py` — this is what the final evaluator will score
5. Iterate and improve until time runs out
6. No filesystem or network IO inside `magic_denoise`
7. No closures or lambdas — all helper functions must be top-level

## Required Output

Your working directory must contain at the end:
- `solution.py` — your best `magic_denoise` implementation
- `results.json` — written automatically by `evaluate.py` after the last evaluation

`results.json` format (written automatically):
```json
{
  "accuracy": 0.87,
  "score": 0.87,
  "mse": 0.039,
  "poisson": 0.035,
  "mse_norm": 0.87,
  "poisson_norm": 0.99,
  "elapsed_seconds": 12.3,
  "error": null
}
```

## Dataset Directory Layout

Everything you need is already in place — no filesystem exploration required:

```
{dataset_dir}/
├── pancreas.h5ad   ← development dataset (load with anndata.read_h5ad)
├── evaluate.py     ← evaluation script
└── task.md         ← this file

{working_dir}/      ← your read/write workspace (initially empty)
```

Your shell's working directory is `{working_dir}` — relative paths (e.g. `ls`, `find .`) resolve there.
To access the dataset directory use its absolute path explicitly (e.g. `ls {dataset_dir}`).
Do not explore the filesystem beyond these two directories. All paths you need are above.

## Evaluation Script

Run `python {dataset_dir}/evaluate.py solution.py` from your working directory to score your solution.

## Generalization

**The development dataset is `pancreas.h5ad` (pancreatic islet cells).** You evaluate against it during your run. But your final score is computed privately on two held-out datasets from entirely different tissues and cell types — datasets you never see during development.

These tissues have completely different gene expression profiles. A solution tuned to pancreas will fail on them.

This means:
- Methods that fit parameters *to the pancreas data* (e.g. autoencoders, tissue-specific models) will likely fail on PBMC and Tabula.
- Methods that use only the *structure of each dataset at inference time* (graph diffusion, PCA-based smoothing, low-rank approximation) generalize naturally because they adapt to whatever data they receive.
Prefer parameter-free or self-adapting approaches. MAGIC (graph diffusion over a k-NN graph) is the canonical baseline for this task precisely because it is tissue-agnostic.

## Tips

- **NORMALIZATION ORDER MATTERS**: For MAGIC, use reversed normalization order (sqrt first, then library-size normalize) — this is what achieves ~0.61 on pancreas and ~0.64 on private datasets
- The MAGIC (approximate solver, reversed norm) approach: `sqrt(X)` → library_size_normalize → MAGIC(solver="approximate") → square → multiply by libsize
- Square root transform is variance-stabilizing for Poisson distributions
- Poisson loss is sensitive to small non-zero values
- Focus on MSE reduction while maintaining Poisson norm ≥ 0.97
