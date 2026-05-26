#!/usr/bin/env python3
"""
Private evaluator for the denoising task.

Evaluates solution files against PBMC and Tabula Muris Senis Lung — the two
held-out datasets from the OpenProblems denoising benchmark. The development
dataset (pancreas) is used by the target agent via data/public/evaluate.py.

Usage:
    # Score all generations in a run:
    python evaluate.py --run-dir runs/run_1

    # Score a single generation directory:
    python evaluate.py --gen-dir runs/run_1/gen_3

    # Score a single solution file:
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

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_PRIVATE_SEED = 48_291_736

DATASETS = {
    "pbmc": {
        "baseline_mse": 0.270945,
        "baseline_poisson": 0.300447,
        "perfect_mse": 0.000000,
        "perfect_poisson": 0.043569,
    },
    "tabula": {
        "baseline_mse": 0.261763,
        "baseline_poisson": 0.206542,
        "perfect_mse": 0.000000,
        "perfect_poisson": 0.026961,
    },
}


def load_solution(solution_path: str):
    spec = importlib.util.spec_from_file_location("solution", solution_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "magic_denoise"):
        raise AttributeError(f"solution.py must define magic_denoise()")
    return module.magic_denoise


def evaluate_mse(test_data, denoised):
    import anndata, scanpy as sc, sklearn.metrics
    test_adata = anndata.AnnData(X=test_data.copy())
    denoised_adata = anndata.AnnData(X=denoised.copy())
    sc.pp.normalize_total(test_adata, target_sum=10000)
    sc.pp.log1p(test_adata)
    sc.pp.normalize_total(denoised_adata, target_sum=10000)
    sc.pp.log1p(denoised_adata)
    return float(sklearn.metrics.mean_squared_error(test_adata.X, denoised_adata.X))


def evaluate_poisson(train_data, test_data, denoised):
    import numpy as np
    from molecular_cross_validation.mcv_sweep import poisson_nll_loss
    import scprep
    test_X = scprep.utils.toarray(test_data)
    denoised_X = np.asarray(denoised).copy()
    denoised_scaled = denoised_X * test_X.sum() / max(train_data.sum(), 1e-12)
    return float(poisson_nll_loss(test_X, denoised_scaled).mean())


def _split_data(adata, seed: int = 0):
    import scipy.sparse, numpy as np
    import molecular_cross_validation.util
    random_state = np.random.RandomState(seed)
    X = np.array(adata.X.todense()) if scipy.sparse.issparse(adata.X) else adata.X
    if not np.allclose(X, X.astype(int)):
        raise TypeError("Molecular cross-validation requires integer count data.")
    X = X.astype(int)
    X_train, X_test = molecular_cross_validation.util.split_molecules(X, 0.9, 0.0, random_state)
    is_missing = X_train.sum(axis=0) == 0
    X_train, X_test = X_train[:, ~is_missing], X_test[:, ~is_missing]
    adata = adata[:, ~is_missing].copy()
    adata.obsm["train"] = scipy.sparse.csr_matrix(X_train).astype(float)
    adata.obsm["test"] = scipy.sparse.csr_matrix(X_test).astype(float)
    return adata


def _load_dataset(name: str):
    import anndata as ad
    import scprep

    data_path = Path(__file__).parent / f"{name}.h5ad"
    if not data_path.exists():
        raise FileNotFoundError(
            f"{name} dataset not found at {data_path}. "
            f"Run: bash tasks/denoising/download_data.sh {name}"
        )
    adata = ad.read_h5ad(data_path)
    adata = _split_data(adata, seed=_PRIVATE_SEED)
    X_train = scprep.utils.toarray(adata.obsm["train"])
    X_test = scprep.utils.toarray(adata.obsm["test"])
    return X_train, X_test


def run_evaluation_on_dataset(magic_denoise_fn, name: str) -> dict:
    import numpy as np

    baseline = DATASETS[name]
    print(f"  [{name}] Loading data...", flush=True)
    try:
        X_train, X_test = _load_dataset(name)
    except Exception as e:
        return {"error": f"Failed to load {name}: {e}", "score": 0.0}

    print(f"  [{name}] {X_train.shape[0]} cells x {X_train.shape[1]} genes", flush=True)

    t0 = time.time()
    try:
        Y_denoised = magic_denoise_fn(X_train, random_state=_PRIVATE_SEED)
    except Exception as e:
        return {"error": f"magic_denoise failed on {name}: {e}", "score": 0.0}
    elapsed = time.time() - t0

    Y_denoised = np.asarray(Y_denoised)
    if not np.isfinite(Y_denoised).all():
        return {"error": "Non-finite values in output", "score": 0.0}
    if np.any(Y_denoised < 0):
        return {"error": "Negative values in output", "score": 0.0}

    mse = evaluate_mse(X_test, Y_denoised)
    poisson = evaluate_poisson(X_train, X_test, Y_denoised)

    mse_range = baseline["baseline_mse"] - baseline["perfect_mse"]
    poisson_range = baseline["baseline_poisson"] - baseline["perfect_poisson"]
    mse_norm = max(0.0, min(1.0, (baseline["baseline_mse"] - mse) / mse_range))
    poisson_norm = max(0.0, min(1.0, (baseline["baseline_poisson"] - poisson) / poisson_range))

    score = (mse_norm + poisson_norm) / 2

    print(f"  [{name}] score={score:.4f}  mse={mse:.6f}  poisson={poisson:.6f}  ({elapsed:.1f}s)", flush=True)

    return {
        "score": score, "mse": mse, "poisson": poisson,
        "mse_norm": mse_norm, "poisson_norm": poisson_norm,
        "elapsed_seconds": elapsed, "error": None,
    }


def score_solution(solution_path: str) -> dict:
    solution_path = os.path.abspath(solution_path)
    if not os.path.exists(solution_path):
        return {"error": f"File not found: {solution_path}", "score": 0.0}

    try:
        magic_fn = load_solution(solution_path)
    except Exception as e:
        return {"error": f"Failed to load solution: {e}", "score": 0.0}

    results = {}
    for name in DATASETS:
        results[name] = run_evaluation_on_dataset(magic_fn, name)

    valid = [r["score"] for r in results.values() if r.get("error") is None]
    avg_score = sum(valid) / len(valid) if valid else 0.0

    try:
        with open(solution_path) as _sf:
            solution_code = _sf.read()
    except Exception:
        solution_code = None

    return {
        "score": avg_score,
        "accuracy": avg_score,
        "lower_is_better": False,
        "per_dataset": results,
        "error": None if valid else "All datasets failed",
        "solution_code": solution_code,
    }


def main():
    parser = argparse.ArgumentParser(description="Private evaluator — denoising task")
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
        print(json.dumps(result, indent=2))
        print(f"\n[private] avg SCORE: {result['score']:.4f}  (written to {out_path})")

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
            print(f"  {gen}: score={r['score']:.4f}" if not r.get("error") else f"  {gen}: FAILED")
        print(f"\nResults saved to: {output_path}")

    else:
        result = score_solution(args.solution)
        print(json.dumps(result, indent=2))
        print(f"\nAvg SCORE: {result['score']:.4f}")


if __name__ == "__main__":
    main()
