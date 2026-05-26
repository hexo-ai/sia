# Sample Task Descriptions — GPU Kernel Optimization

These are example descriptions of the TriMul kernel optimization task.
They emphasize that a good solution must (a) be correct for any (N, N, C) input and (b) generalize
across matrix sizes, not just the development benchmark shape.

---

## Example A

Implement `trimul(Z)` in `solution.py`. `Z` is a float32 CUDA tensor of shape `(N, N, C)`.
The function must return a float32 CUDA tensor of the same shape that matches the reference
implementation's output (atol=1e-2, rtol=1e-2).

Your implementation is **benchmarked on N=256, C=128**, but the private score averages across
N ∈ {128, 192, 256, 320, 384}. Hard-coded tile sizes or shape-specific tricks that break
on other sizes will hurt the private score.

Evaluate with `python {dataset_dir}/evaluate.py solution.py`. Score = speedup over the
unoptimized PyTorch reference — 1.0 means no improvement, 3.0 means 3× faster.

The recommended starting point is `torch.compile(fn, mode="max-autotune")`, which typically
gives 2–3× speedup with zero code change.

---

## Example B

Your task is to speed up the TriMul operation (triangular multiplicative update from AlphaFold2)
by writing an optimized `trimul(Z)` function.

The reference implementation is memory-bound: it runs ~10 separate CUDA kernels for operations
that could be fused into 2–3 kernels. The matmul step (`torch.bmm`) runs in FP32 and does
not exploit tensor cores.

Key optimization opportunities:
1. Fuse the LayerNorm + sigmoid gating (input side) into a single Triton kernel
2. Use FP16 for the bmm to leverage tensor cores (cuBLAS picks this up automatically)
3. Fuse the output LayerNorm + gating
4. Integrate the upper-triangular mask into the matmul

The development benchmark runs on N=256, C=128. Write a general solution that handles any N and C.
Score = speedup over the reference; higher is better.

---

## Example C

Implement a high-performance GPU kernel for the triangular multiplicative update used in protein
structure prediction (AlphaFold2). The mathematical operation is:

```
Z_norm = LayerNorm(Z)
a = Z_norm * sigmoid(Z_norm)
b = Z_norm * sigmoid(-Z_norm)
Y[i,j,c] = sum_{k<=j} a[i,k,c] * b[j,k,c]
output = LayerNorm(Y) * sigmoid(Z_norm)
```

The challenge: a naive implementation launches 8+ separate CUDA kernels and runs the matmul
in FP32. Expert GPU kernels achieve 4–10× speedup by fusing operations and using mixed precision.

Write `trimul(Z)` in `solution.py`. Use any combination of PyTorch, `torch.compile`, Triton,
or raw CUDA. Measure improvement with `python {dataset_dir}/evaluate.py solution.py`.
Score = speedup over the unoptimized baseline (higher is better). A 4× speedup is a strong result.
