# Denoising — Reference Scores

Scores from the TTT-Discover paper (Table 7), reproduced with our evaluation setup.

**Scoring formula** (identical to the paper):
```
score = (mse_norm + poisson_norm) / 2
```
where normalization reference points are `Identity` (baseline) and `Oracle` (perfect).

Our private evaluator reproduces the paper's MAGIC score exactly (0.641 vs 0.64).

---

## Known scores

| Method | Model | PBMC | Tabula | **Avg** |
|--------|-------|-----:|-------:|--------:|
| Identity (no denoising) | — | 0.00 | 0.00 | **0.00** |
| MAGIC | — | 0.42 | 0.40 | **0.41** |
| MAGIC (A) | — | 0.42 | 0.40 | **0.41** |
| ALRA (S, RN) | — | 0.50 | 0.47 | **0.49** |
| Best-of-25600 | gpt-oss-120b | 0.62 | 0.65 | **0.64** |
| MAGIC (R) | — | 0.64 | 0.64 | **0.64** |
| MAGIC (A, R) | — | 0.64 | 0.64 | **0.64** |
| OpenEvolve | gpt-oss-120b | 0.70 | 0.71 | **0.71** |
| TTT-Discover | gpt-oss-120b | 0.71 | 0.73 | **0.72** |

Source: TTT-Discover paper, Table 7. Scores are `mean(mse_norm, poisson_norm)` per dataset.

**MAGIC (A, R)** = MAGIC approximate solver + reversed normalization (best classical baseline).  
**OpenEvolve** = evolutionary LLM search (closest to SIA's approach).  
**TTT-Discover** = test-time training with PUCT reuse tree (current SOTA).

---

## Our own verified values

Computed by running `denoising_utils.magic_denoise` on our datasets with `_PRIVATE_SEED = 48_291_736`:

| Method | PBMC MSE | PBMC Poisson | PBMC Score | Tabula MSE | Tabula Poisson | Tabula Score | **Avg** |
|--------|----------|-------------|-----------|------------|---------------|-------------|---------|
| MAGIC (A,R) | 0.1885 | 0.0494 | 0.641 | 0.1839 | 0.0297 | 0.641 | **0.641** |
| Identity | 0.2709 | 0.3004 | 0.000 | 0.2618 | 0.2065 | 0.000 | **0.000** |
| Oracle | — | 0.0436 | — | — | 0.0270 | — | — |
