# GPU Kernel (TriMul) — Reference Scores

Scores from the TTT-Discover paper (Table 4), reproduced with our evaluation setup.

**Scoring formula**:
```
speedup = reference_median_time_ms / solution_median_time_ms
score   = speedup   (higher is better, baseline = 1.0)
```

The reference is the unoptimized PyTorch implementation (`torch.bmm` in FP32, unfused ops).
The private score is the geometric mean of speedups across N ∈ {128, 192, 256, 320, 384} with C=128.

---

## Known results — H100 (primary leaderboard)

Runtimes in µs from the TTT-Discover paper. Speedup computed relative to the 5th human submission
(used as a conservative unoptimized-ish baseline for comparison; our PyTorch reference is slower).

| Method | Model | H100 (µs) | vs. 5th human |
|--------|-------|----------:|--------------:|
| 5th human | — | 4,233 | 1.00× |
| 4th human | — | 3,655 | 1.16× |
| 3rd human | — | 2,546 | 1.66× |
| 2nd human | — | 2,368 | 1.79× |
| **1st human** | — | **1,371** | **3.09×** |
| Best-of-25600 | gpt-oss-120b | 5,390 | 0.79× |
| **TTT-Discover** | gpt-oss-120b | **1,161** | **3.65×** |

Source: TTT-Discover paper, Table 4.

**TTT-Discover** beats the best human by ~15% (1,161 µs vs 1,371 µs).  
**Best-of-25600** (random sampling without RL) scores worse than all human submissions — the optimization gain comes entirely from the training/search process.

---

## Known results — other GPUs

| Method | Model | A100 (µs) | B200 (µs) | MI300X (µs) |
|--------|-------|----------:|----------:|------------:|
| 1st human | — | 4,532 | 1,039 | 2,516 |
| Best-of-25600 | gpt-oss-120b | 9,220 | 3,255 | 4,902 |
| **TTT-Discover** | gpt-oss-120b | **2,198** | **914** | **1,556** |

The TTT-Discover kernel **generalizes to all GPU types** even though training used only H100 runtimes as reward. On A100 specifically, it is **50% faster than the top human**, despite never being optimized for A100.

Source: TTT-Discover paper, Table 4.

---

## Winning strategy (from expert review)

The TTT-Discover kernel is memory-bound aware:

1. **Fuses input LayerNorm** — eliminates a separate kernel launch + memory round-trip
2. **Fuses sigmoid + elementwise multiply** — input gating `a` and `b` computed in one pass
3. **Fuses output LayerNorm + gating** — eliminates another kernel launch
4. **Converts to FP16 → delegates matmul to cuBLAS** — exploits tensor cores, non-trivial to beat with a hand-written Triton kernel

The key insight: the operation is **memory-bound** (not compute-bound) because of the surrounding pointwise ops. Fusing them reduces memory traffic far more than tuning the matmul itself.

---

## Our own baseline (PyTorch reference)

Measured on H100 with our evaluator (median over 50 trials, N=256, C=128):

| Implementation | H100 median (ms) | Speedup |
|----------------|----------------:|--------:|
| PyTorch reference (unfused, FP32) | ~TBD | 1.00× |
| `torch.compile(mode="max-autotune")` | ~TBD | ~2–3× |
| FP16 bmm only | ~TBD | ~TBD |

*To be filled in after running on the target hardware.*
