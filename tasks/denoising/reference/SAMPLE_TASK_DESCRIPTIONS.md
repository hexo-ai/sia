# Sample Task Descriptions

## Task: scRNA-seq Denoising

Your task is to implement and iteratively improve a `magic_denoise(X, **kwargs)` function that denoises single-cell RNA-seq count data. The function takes a raw count matrix and returns denoised counts.

You will evaluate your implementation using `python evaluate.py solution.py` and iterate to improve the score.

---

## Task: scRNA-seq Denoising (Improvement)

You have a working `magic_denoise` function. Your goal is to improve it by:
1. Reducing MSE in log-normalized space
2. Maintaining the Poisson constraint (poisson_norm ≥ 0.97)

Run `python evaluate.py solution.py` to evaluate, then improve the implementation.
