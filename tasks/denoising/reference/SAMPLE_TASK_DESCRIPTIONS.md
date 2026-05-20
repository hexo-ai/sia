# Sample Task Descriptions — scRNA-seq Denoising

These are example descriptions of the denoising task as it should be presented to a target agent.
They emphasize that a good solution must generalize across tissues and cell types, not just perform
well on the development dataset.

---

## Example A

Implement `magic_denoise(X, **kwargs)` in `solution.py`. `X` is a raw count matrix
(cells × genes, non-negative integers) from a single-cell RNA-seq experiment. Return a denoised
matrix of the same shape.

Your solution will be **developed and evaluated on pancreatic islet cells**, but a strong solution
must generalize: the same function, unchanged, should work well on blood cells (PBMC) and lung
tissue. Overfitting to pancreas-specific biology (e.g., hard-coded gene lists, tissue-specific
priors) will hurt performance on other tissues.

Evaluate with `python {dataset_dir}/evaluate.py solution.py`. Two metrics:
- **Poisson norm ≥ 0.97** — hard constraint, solution is rejected if not met
- **MSE norm** — your score (higher is better, range 0–1)

---

## Example B

Your task is to denoise single-cell RNA-seq data by implementing `magic_denoise(X, **kwargs)`.
The input `X` contains raw transcript counts for a population of cells — noisy because of technical
dropout during sequencing.

A good denoising algorithm works on the structure of the data itself (graph diffusion, manifold
smoothing, low-rank approximation) rather than on tissue-specific assumptions. The pancreas
development set is a proxy: the real test is whether your approach recovers true expression levels
in **any** tissue.

Start with MAGIC (graph diffusion on a sqrt-transformed count matrix), which is robust across
tissues. Then iterate: reduce MSE by tuning diffusion parameters or exploring alternatives, while
keeping Poisson norm ≥ 0.97.

---

## Example C

Implement a generalizable RNA-seq denoising function. The key challenge: scRNA-seq dropout noise
follows a Poisson-like process regardless of tissue type — a cell that captured 5% of its true
transcripts looks the same whether it's a T-cell or a pancreatic beta cell. Your denoiser should
exploit this shared statistical structure.

Develop on the pancreas benchmark (`python {dataset_dir}/evaluate.py solution.py`), but design
as if you will never see the tissue type at test time. Techniques that rely on cell-type markers,
tissue-specific gene programs, or fixed hyperparameters tuned only on pancreas will underperform
on held-out organs.
