# GPUMode TriMul — Triangular Matrix Multiplication Competition

## Overview

GPUMode is an open community for GPU kernel development that hosts competitions for domain experts. The **TriMul** competition asks participants to write the fastest possible implementation of a triangular matrix multiplication primitive — a core building block in AlphaFold2's architecture for protein structure prediction.

Each GPU architecture (NVIDIA H100, A100, B200, AMD MI300X) has its own leaderboard, since performant implementations differ across hardware. Submissions must pass correctness checks before runtime is measured.

Your submission is a single Python file containing a `trimul(Z)` function. You develop and profile it against an H100 benchmark. The evaluation runs on a fixed set of input shapes and reports the median runtime.

## The Operation

The TriMul primitive computes a triangular multiplicative update on pair representations from AlphaFold2. Given a tensor `Z` of shape `(N, N, C)` representing pairwise features between sequence positions:

```
Z_norm = LayerNorm(Z)                        # normalize along channel dim C
a      = Z_norm ⊙ σ(Z_norm)                 # input gate A    — (N, N, C)
b      = Z_norm ⊙ σ(−Z_norm)               # input gate B    — (N, N, C)
Y[i,j,c] = Σ_{k≤j} a[i,k,c] · b[j,k,c]    # upper-triangular matmul
output = LayerNorm(Y) ⊙ σ(Z_norm)           # output gating   — (N, N, C)
```

The operation is dominated by two distinct bottlenecks:
- **Memory-bound elementwise ops**: the gating sequence (LayerNorm → sigmoid → multiply) reads and writes the full `(N, N, C)` tensor multiple times
- **Compute-bound triangular matmul**: O(N³·C) complexity, amenable to tensor core acceleration

## Benchmark

**Primary benchmark shape**: `N=256, C=128` (matching typical AlphaFold2 inference at medium sequence length)

**Hardware**: NVIDIA H100 SXM5 (80 GB)

**Measurement**: median over 50 trials with 10 warmup iterations, `torch.cuda.synchronize()` for accurate GPU timing

**Correctness tolerance**: atol=1e-2, rtol=1e-2 (float32 output)

## Known Results (H100)

From the TTT-Discover paper (reported runtimes in µs):

| Method | H100 (µs) |
|--------|-----------|
| 5th human | 4,233 |
| 4th human | 3,655 |
| 3rd human | 2,546 |
| 2nd human | 2,368 |
| **1st human** | **1,371** |
| Best-of-25600 (gpt-oss-120b) | 5,390 |
| **TTT-Discover (gpt-oss-120b)** | **1,161** |

TTT-Discover achieves **~1.18× speedup** over the best human submission and **~15%+ improvement** over all human submissions. The key insight found by the agent: the operation is **memory-bound** because of the surrounding elementwise ops, so the winning strategy is maximum **operation fusion** to reduce memory traffic and kernel launch overhead.

## Winning Strategy (Expert Review, GPUMode Organizers)

> *"The referenced solution correctly determined that the problem is memory bound because of the surrounding point-wise operations so the agent focuses as much as possible on operation fusions, lowering the memory traffic and kernel launch overhead."*
>
> *"Its strategy is to reduce memory bandwidth via fusions, lower precision and delegating the big matrix multiplications to cuBLAS, as those are non-trivial to beat. This is similar to the current best human solutions, but executed on better."*
> — Matej Sirovatka, Alex Zhang, Mark Saroufim (GPUMode)

Concretely, the winning kernel:
1. **Fuses** the input LayerNorm operations into a single kernel
2. **Fuses** sigmoid + elementwise multiplication (input gating)
3. **Fuses** the output LayerNorm + gating
4. **Converts inputs to FP16** and delegates the triangular matmul to **cuBLAS/cuBLASLt**, leveraging tensor cores

## Starting Point

The unoptimized PyTorch reference (equivalent to a naive first submission):

```python
import torch
import torch.nn.functional as F

def trimul(Z: torch.Tensor) -> torch.Tensor:
    N, _, C = Z.shape
    Z_norm = F.layer_norm(Z, [C])
    a = Z_norm * torch.sigmoid(Z_norm)
    b = Z_norm * torch.sigmoid(-Z_norm)
    a_perm = a.permute(2, 0, 1)
    b_perm = b.permute(2, 1, 0)
    Y = torch.bmm(a_perm, b_perm).permute(1, 2, 0)
    mask = torch.triu(torch.ones(N, N, device=Z.device, dtype=Z.dtype))
    Y = Y * mask.unsqueeze(-1)
    g = torch.sigmoid(Z_norm)
    Y_norm = F.layer_norm(Y, [C])
    return Y_norm * g
```

This baseline launches ~10 separate CUDA kernels and runs the matmul in FP32, leaving substantial room for improvement.

**Your goal is to beat the best human submission (1,371 µs on H100).**  
Scores above 2× speedup over the reference baseline are strong results. The TTT-Discover kernel achieves ~3.6× speedup over the unoptimized baseline.

## Available Libraries

`torch` · `triton` · `numpy`

Triton is pre-installed and can be used for custom fused kernels. For the triangular matmul, delegating to `torch.mm` / cuBLAS in FP16 is recommended over a hand-written Triton kernel, since cuBLAS already saturates tensor core utilization.

## Generalization

The development benchmark uses `N=256, C=128`. The private evaluation averages the geometric mean of runtimes over `N ∈ {128, 192, 256, 320, 384}` with `C=128` — consistent with the GPUMode leaderboard methodology of benchmarking across a fixed set of input shapes. A solution that hardcodes tile sizes or assumes a specific N will degrade on other shapes.
