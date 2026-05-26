# Sample Task Descriptions — scRNA-seq Denoising

These are example descriptions of the denoising task as it should be presented to a target agent.
They emphasize that a good solution must generalize across tissues and cell types, not just perform
well on the development dataset.

---

## Example A

Implement `magic_denoise(X, **kwargs)` in `solution.py`. `X` is a raw count matrix
(cells × genes, non-negative integers) from a single-cell RNA-seq experiment. Return a denoised
matrix of the same shape.

Your solution is **developed on pancreatic islet cells** (`pancreas.h5ad`), but the final score
is computed privately on two held-out datasets: **PBMC** (human blood immune cells) and **Tabula**
(multi-tissue atlas). Overfitting to pancreas-specific biology (e.g., hard-coded gene lists,
tissue-specific priors) will hurt performance on PBMC and Tabula.

Evaluate with `python {dataset_dir}/evaluate.py solution.py`. Score = mean(mse_norm, poisson_norm), range 0–1, higher is better. The `magic-impute` library is available — `magic.MAGIC().fit_transform(X)` scores ~0.61 and is the recommended starting point.

---

## Example B

Your task is to denoise single-cell RNA-seq data by implementing `magic_denoise(X, **kwargs)`.
The input `X` contains raw transcript counts for a population of cells — noisy because of technical
dropout during sequencing.

A good denoising algorithm works on the structure of the data itself (graph diffusion, manifold
smoothing, low-rank approximation) rather than on tissue-specific assumptions. The pancreas
development set (`pancreas.h5ad`) is your proxy: the real test is whether your approach recovers
true expression levels on **PBMC** and **Tabula** — two held-out datasets from completely different
biological contexts that you never see during development.

Start with `magic.MAGIC().fit_transform(X)` from the `magic-impute` library (available, scores ~0.61), which is robust across tissues. Then iterate: tune diffusion parameters (knn, t, decay), try different normalization orders, or explore complementary approaches to improve the mean(mse_norm, poisson_norm) score.

---

## Example C

Implement a generalizable RNA-seq denoising function. The key challenge: scRNA-seq dropout noise
follows a Poisson-like process regardless of tissue type — a cell that captured 5% of its true
transcripts looks the same whether it's a T-cell or a pancreatic beta cell. Your denoiser should
exploit this shared statistical structure.

Develop on the pancreas benchmark (`python {dataset_dir}/evaluate.py solution.py`), but design
as if you will never see the tissue type at test time. The private evaluation runs on **PBMC**
(blood immune cells) and **Tabula** (multi-tissue atlas) — techniques that rely on cell-type
markers, tissue-specific gene programs, or hyperparameters tuned only on pancreas will
underperform on these held-out tissues. The `magic-impute` library is available: start from `magic.MAGIC().fit_transform(X)` (~0.61) and improve. Score = mean(mse_norm, poisson_norm).
