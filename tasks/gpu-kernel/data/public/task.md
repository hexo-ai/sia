# TriMul — Triangular Matrix Multiplication Kernel Optimization

## Overview

Your task is to write an optimized implementation of the `trimul` function — a core computational primitive from AlphaFold2's architecture used in the GPUMode TriMul competition. You will iteratively improve your kernel's runtime on a fixed H100 benchmark.

## Operation

Given a pair representation tensor `Z` of shape `(N, N, C)`, the TriMul operation computes a triangular multiplicative update:

```python
import torch
import torch.nn.functional as F

def trimul_reference(Z: torch.Tensor) -> torch.Tensor:
    """Unoptimized reference implementation — your starting baseline."""
    N, _, C = Z.shape

    # Input LayerNorm
    Z_norm = F.layer_norm(Z, [C])

    # Input gating (memory-bound: 3 separate kernel launches)
    a = Z_norm * torch.sigmoid(Z_norm)     # (N, N, C)
    b = Z_norm * torch.sigmoid(-Z_norm)    # (N, N, C)

    # Triangular matmul: Y[i,j,c] = sum_{k<=j} a[i,k,c] * b[j,k,c]
    # Equivalent to batched matmul per channel, then upper-triangular mask
    a_perm = a.permute(2, 0, 1)   # (C, N, N)
    b_perm = b.permute(2, 1, 0)   # (C, N, N)
    Y = torch.bmm(a_perm, b_perm).permute(1, 2, 0)  # (N, N, C)

    # Upper triangular mask (sets k > j entries to 0)
    mask = torch.triu(torch.ones(N, N, device=Z.device, dtype=Z.dtype))
    Y = Y * mask.unsqueeze(-1)

    # Output gating with LayerNorm (another 3 kernel launches)
    g = torch.sigmoid(Z_norm)
    Y_norm = F.layer_norm(Y, [C])

    return Y_norm * g
```

## Function Signature

```python
def trimul(Z: torch.Tensor) -> torch.Tensor:
    """
    Optimized triangular matrix multiplication.

    Args:
        Z: float32 CUDA tensor of shape (N, N, C)
    Returns:
        float32 CUDA tensor of shape (N, N, C)
    """
```

## Evaluation

The function is benchmarked on shape N=256, C=128. Runtime is measured as the median over 50 trials (after 10 warmup iterations) with `torch.cuda.synchronize()` for accurate GPU timing.

**Scoring**:
```
speedup = reference_median_time / solution_median_time
score   = speedup   (higher is better, baseline = 1.0)
```

A solution running at the same speed as the reference scores 1.0. A 3× speedup scores 3.0.

## Rules

1. Write `solution.py` containing your `trimul(Z)` function
2. Evaluate using `python {dataset_dir}/evaluate.py solution.py`
3. After each evaluation, `results.json` is written to your working directory — do not write it yourself
4. At the end of your run, your working directory **must** contain `solution.py`
5. No side effects inside `trimul`: no file I/O, no print statements, no global state mutation between calls
6. Output must match the reference numerically (atol=1e-2, rtol=1e-2 in float32)
7. The function must handle any (N, N, C) shape, not just the benchmark shape

## Available Libraries

`torch`, `triton`, `numpy`, `scipy`. A CUDA GPU is required.

## Optimization Strategies

The reference has several bottlenecks to attack:

1. **Excessive kernel launches**: LayerNorm, sigmoid, multiply, bmm, and mask each launch a separate kernel. Each launch costs ~5–10 µs of overhead on top of memory transfers.
2. **FP32 matmul**: The `torch.bmm` call does not exploit tensor cores. Converting to FP16 for the matmul alone can give 4–8× speedup on that step.
3. **Memory-bound elementwise ops**: The gating sequence (norm → sigmoid → multiply) reads and writes `Z_norm` three times when it could be done in one pass.
4. **Redundant computations**: `Z_norm` is computed once but `sigmoid(Z_norm)` and `sigmoid(-Z_norm)` are computed as separate ops.

**Quick wins to try first:**
- `torch.compile(trimul_reference, mode="max-autotune")` — zero code change, often 2–3× speedup
- Manual FP16 cast before `bmm`: `a.half() @ b.half()` through `bmm`
- Fuse the gating in Triton: one kernel that reads `Z_norm` once and writes `a` and `b` simultaneously

**Deeper optimizations:**
- Fuse LayerNorm + sigmoid gating into a single Triton kernel (eliminates 4–5 kernel launches)
- Use cuBLAS GEMM directly in FP16 for the triangular matmul (delegate to tensor cores)
- Fuse the output LayerNorm + gating into one kernel
- Integrate the upper-triangular mask into the matmul kernel (apply it per-tile at zero extra cost)

## Profiling

```python
# Quick profiling with torch.profiler
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    trimul(Z)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

## Dataset Directory Layout

```
{dataset_dir}/
├── evaluate.py     ← evaluation script
└── task.md         ← this file

{working_dir}/      ← your read/write workspace (initially empty)
```

Your shell's working directory is `{working_dir}`. Use absolute paths to access `{dataset_dir}`.

## Evaluation Script

```bash
python {dataset_dir}/evaluate.py solution.py
```

## Generalization

The development benchmark uses N=256, C=128. The private score averages over multiple shapes:
`N ∈ {128, 192, 256, 320, 384}` with `C=128`. Solutions that hardcode tile sizes or assume a specific N will generalize poorly.
