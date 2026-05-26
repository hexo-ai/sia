#!/usr/bin/env python3
"""
Private evaluator for the gpu-kernel task.

Evaluates solutions over multiple shapes to measure generalization across
matrix sizes. The development benchmark uses a single shape (N=256, C=128);
this private evaluator tests N ∈ {128, 192, 256, 320, 384} with C=128.

Usage:
    python evaluate.py --gen-dir runs/run_1/gen_3
    python evaluate.py --run-dir runs/run_1
    python evaluate.py path/to/solution.py
"""

import sys
import os
import json
import time
import glob
import argparse
import importlib.util
import traceback
from pathlib import Path

BENCHMARK_SHAPES = [
    (128, 128, 128),
    (192, 192, 128),
    (256, 256, 128),
    (320, 320, 128),
    (384, 384, 128),
]
WARMUP_ITERS = 10
BENCH_ITERS = 50
CORRECTNESS_ATOL = 1e-2
CORRECTNESS_RTOL = 1e-2


def load_solution(solution_path: str):
    spec = importlib.util.spec_from_file_location("solution", solution_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "trimul"):
        raise AttributeError("solution.py must define trimul(Z)")
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
    return times[len(times) // 2] * 1000.0


def evaluate_on_shape(trimul_fn, shape: tuple) -> dict:
    import torch
    N, _, C = shape
    torch.manual_seed(42)
    Z = torch.randn(*shape, device="cuda", dtype=torch.float32)
    label = f"N={N},C={C}"

    # Correctness
    ref_out = reference_trimul(Z).float()
    try:
        sol_out = trimul_fn(Z.clone()).float()
    except Exception as e:
        return {"error": f"[{label}] trimul() raised: {e}", "speedup": 0.0}

    if sol_out.shape != ref_out.shape:
        return {"error": f"[{label}] Shape mismatch: expected {tuple(ref_out.shape)}, got {tuple(sol_out.shape)}", "speedup": 0.0}
    if not torch.isfinite(sol_out).all():
        return {"error": f"[{label}] Non-finite values in output", "speedup": 0.0}
    if not torch.allclose(ref_out, sol_out, atol=CORRECTNESS_ATOL, rtol=CORRECTNESS_RTOL):
        max_diff = (ref_out - sol_out).abs().max().item()
        return {"error": f"[{label}] Correctness failed: max_diff={max_diff:.6f}", "speedup": 0.0}

    ref_ms = benchmark_fn(reference_trimul, Z, WARMUP_ITERS, BENCH_ITERS)
    sol_ms = benchmark_fn(trimul_fn, Z.clone(), WARMUP_ITERS, BENCH_ITERS)
    speedup = ref_ms / sol_ms

    print(f"  [{label}] ref={ref_ms:.3f}ms  sol={sol_ms:.3f}ms  speedup={speedup:.2f}x", flush=True)
    return {"ref_ms": ref_ms, "sol_ms": sol_ms, "speedup": speedup, "error": None}


def score_solution(solution_path: str) -> dict:
    import torch
    solution_path = os.path.abspath(solution_path)
    if not os.path.exists(solution_path):
        return {"error": f"File not found: {solution_path}", "score": 0.0}

    if not torch.cuda.is_available():
        return {"error": "CUDA not available", "score": 0.0}

    try:
        trimul_fn = load_solution(solution_path)
    except Exception as e:
        return {"error": f"Failed to load solution: {e}", "score": 0.0}

    per_shape = {}
    for shape in BENCHMARK_SHAPES:
        label = f"{shape[0]}x{shape[1]}x{shape[2]}"
        per_shape[label] = evaluate_on_shape(trimul_fn, shape)

    valid_speedups = [r["speedup"] for r in per_shape.values() if r.get("error") is None and r.get("speedup", 0) > 0]

    if not valid_speedups:
        avg_speedup = 0.0
        error = "All shapes failed"
    else:
        import math
        # Geometric mean of speedups (consistent with paper's reward = 1/geomean(runtimes))
        log_sum = sum(math.log(s) for s in valid_speedups)
        avg_speedup = math.exp(log_sum / len(valid_speedups))
        error = None if len(valid_speedups) == len(BENCHMARK_SHAPES) else f"Only {len(valid_speedups)}/{len(BENCHMARK_SHAPES)} shapes succeeded"

    try:
        with open(solution_path) as f:
            solution_code = f.read()
    except Exception:
        solution_code = None

    return {
        "score": avg_speedup,
        "accuracy": avg_speedup,
        "lower_is_better": False,
        "per_shape": per_shape,
        "num_shapes": len(valid_speedups),
        "error": error,
        "solution_code": solution_code,
    }


def main():
    parser = argparse.ArgumentParser(description="Private evaluator — gpu-kernel task")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir", help="Run directory (evaluates all gen_X/)")
    group.add_argument("--gen-dir", help="Single generation directory")
    group.add_argument("solution", nargs="?", help="Path to a single solution.py")
    args = parser.parse_args()

    if args.gen_dir:
        gen_dir = os.path.abspath(args.gen_dir)
        result = score_solution(os.path.join(gen_dir, "solution.py"))
        out_path = os.path.join(gen_dir, "private_result.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        display = {k: v for k, v in result.items() if k != "solution_code"}
        print(json.dumps(display, indent=2))
        print(f"\n[private] avg speedup: {result['score']:.4f}x  (written to {out_path})")

    elif args.run_dir:
        run_dir = os.path.abspath(args.run_dir)
        gen_dirs = sorted(glob.glob(os.path.join(run_dir, "gen_*")))
        if not gen_dirs:
            print(f"No gen_* directories found in {run_dir}", file=sys.stderr)
            sys.exit(1)

        all_scores = {}
        for gen_dir in gen_dirs:
            gen_name = os.path.basename(gen_dir)
            print(f"\n[{gen_name}] Evaluating...")
            result = score_solution(os.path.join(gen_dir, "solution.py"))
            all_scores[gen_name] = result
            out_path = os.path.join(gen_dir, "private_result.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)

        output_path = os.path.join(run_dir, "private_scores.json")
        with open(output_path, "w") as f:
            json.dump(all_scores, f, indent=2)

        print(f"\n=== PRIVATE SCORES SUMMARY ===")
        for gen, r in all_scores.items():
            if not r.get("error"):
                print(f"  {gen}: speedup={r['score']:.4f}x")
            else:
                print(f"  {gen}: FAILED — {r['error']}")
        print(f"\nResults saved to: {output_path}")

    else:
        result = score_solution(args.solution)
        display = {k: v for k, v in result.items() if k != "solution_code"}
        print(json.dumps(display, indent=2))
        print(f"\nGeometric mean speedup: {result['score']:.4f}x")


if __name__ == "__main__":
    main()
