#!/usr/bin/env python3
"""
Evaluate a trimul solution against the TriMul benchmark.

Usage:
    python evaluate.py solution.py

The solution.py must define a top-level `trimul(Z)` function.
Outputs results.json next to solution.py.
"""

import sys
import os
import json
import time
import importlib.util
import traceback
from pathlib import Path

# Development benchmark: single shape
BENCHMARK_SHAPE = (256, 256, 128)  # (N, N, C)
WARMUP_ITERS = 10
BENCH_ITERS = 50
CORRECTNESS_ATOL = 1e-2
CORRECTNESS_RTOL = 1e-2


def load_solution(solution_path: str):
    spec = importlib.util.spec_from_file_location("solution", solution_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "trimul"):
        raise AttributeError(f"solution.py must define trimul(Z), not found in {solution_path}")
    return module.trimul


def reference_trimul(Z):
    import torch
    import torch.nn.functional as F
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


def benchmark_fn(fn, Z, warmup: int = 10, iters: int = 50) -> float:
    """Returns median runtime in milliseconds."""
    import torch
    for _ in range(warmup):
        fn(Z)
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(Z)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2] * 1000.0  # median in ms


def run_evaluation(trimul_fn) -> dict:
    import torch
    if not torch.cuda.is_available():
        return {
            "error": "CUDA not available — a GPU is required for kernel benchmarking",
            "score": 0.0,
        }

    N, _, C = BENCHMARK_SHAPE
    torch.manual_seed(42)
    Z = torch.randn(*BENCHMARK_SHAPE, device="cuda", dtype=torch.float32)

    print(f"Shape: {tuple(Z.shape)}  device: {Z.device}", flush=True)

    # Reference run (establishes baseline time)
    print("Benchmarking reference implementation...", flush=True)
    ref_time_ms = benchmark_fn(reference_trimul, Z, WARMUP_ITERS, BENCH_ITERS)
    ref_out = reference_trimul(Z).float()
    print(f"Reference: {ref_time_ms:.3f} ms", flush=True)

    # Correctness check
    print("Running correctness check...", flush=True)
    try:
        sol_out = trimul_fn(Z.clone()).float()
    except Exception as e:
        return {"error": f"trimul() raised an exception: {e}\n{traceback.format_exc()}", "score": 0.0}

    if sol_out.shape != ref_out.shape:
        return {
            "error": f"Shape mismatch: expected {tuple(ref_out.shape)}, got {tuple(sol_out.shape)}",
            "score": 0.0,
        }

    if not torch.isfinite(sol_out).all():
        return {"error": "Non-finite values in output (NaN or Inf)", "score": 0.0}

    if not torch.allclose(ref_out, sol_out, atol=CORRECTNESS_ATOL, rtol=CORRECTNESS_RTOL):
        max_diff = (ref_out - sol_out).abs().max().item()
        mean_diff = (ref_out - sol_out).abs().mean().item()
        return {
            "error": f"Correctness check failed: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f} "
                     f"(atol={CORRECTNESS_ATOL}, rtol={CORRECTNESS_RTOL})",
            "score": 0.0,
        }
    print("Correctness check passed.", flush=True)

    # Solution benchmark
    print("Benchmarking solution...", flush=True)
    sol_time_ms = benchmark_fn(trimul_fn, Z.clone(), WARMUP_ITERS, BENCH_ITERS)
    print(f"Solution:  {sol_time_ms:.3f} ms", flush=True)

    speedup = ref_time_ms / sol_time_ms

    return {
        "ref_time_ms": ref_time_ms,
        "sol_time_ms": sol_time_ms,
        "speedup": speedup,
        "score": speedup,
        "lower_is_better": False,
        "error": None,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py solution.py", file=sys.stderr)
        sys.exit(1)

    solution_path = os.path.abspath(sys.argv[1])
    if not os.path.exists(solution_path):
        result = {"error": f"File not found: {solution_path}", "score": 0.0}
    else:
        print(f"Loading solution from: {solution_path}", flush=True)
        try:
            trimul_fn = load_solution(solution_path)
        except Exception as e:
            result = {"error": f"Failed to load solution: {e}", "score": 0.0}
        else:
            try:
                result = run_evaluation(trimul_fn)
            except Exception as e:
                result = {
                    "error": f"Evaluation failed: {e}\n{traceback.format_exc()}",
                    "score": 0.0,
                }

    result["accuracy"] = result.get("score", 0.0)
    result["lower_is_better"] = False

    try:
        with open(solution_path) as f:
            result["solution_code"] = f.read()
    except Exception:
        result["solution_code"] = None

    results_path = os.path.join(os.path.dirname(solution_path), "results.json")
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results written to: {results_path}", flush=True)

    # Print without solution_code (too large)
    display = {k: v for k, v in result.items() if k != "solution_code"}
    print("\n=== EVALUATION RESULT ===")
    print(json.dumps(display, indent=2))
    print("=========================")

    if result.get("score", 0.0) > 0:
        print(f"\nSPEEDUP: {result['speedup']:.2f}x")
        print(f"SCORE:   {result['score']:.4f}")
        print(f"Reference: {result.get('ref_time_ms', 0):.3f} ms")
        print(f"Solution:  {result.get('sol_time_ms', 0):.3f} ms")
    else:
        print(f"\nFAILED: {result.get('error', 'Unknown error')}")

    sys.exit(0)


if __name__ == "__main__":
    main()
