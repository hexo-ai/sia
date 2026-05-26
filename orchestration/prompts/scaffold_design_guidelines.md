## Scaffold Design Philosophy

The target_agent.py is an **orchestrator**, not a thin wrapper around a single LLM call.
You are free — and encouraged — to implement Python-level control flow (loops, branches,
conditionals) and to call `call_task_model()` at multiple points in the code. Do not
delegate all logic to the task model when Python can handle it more reliably and cheaply.

---

### Dataset exploration — do it in the scaffold, not in the task model

Before calling the task model, the scaffold can (and often should) read the dataset
directly using Python and inject what it finds into the prompt. This is **generic good
practice** that applies to any ML or data task — it is not task-specific optimization.

Examples of what the scaffold can extract and inject:
- File structure and available files in the dataset directory
- Column names, data types, shape, and sample rows of CSV/parquet files
- Basic statistics: mean, std, sparsity, class distribution, missing values
- Domain hints embedded in filenames or README/task.md files

Injecting this context into the task model's prompt means the model starts informed
instead of wasting turns on filesystem exploration. The scaffold reads once; the task
model acts immediately.

---

### Budget allocation when calling `call_task_model()` multiple times

`args.max_turns` is a **global hard budget** shared across every call to
`call_task_model()` in the scaffold. If you call it more than once, you must split the
budget explicitly and track the total consumed.

Rules:
- Pass a `sub_budget` (not `args.max_turns`) to each individual `call_task_model()` call.
- Track cumulative turns consumed; never exceed `args.max_turns` in total.
- Leave enough turns for later phases — plan the allocation before the first call.

Pattern:
    turns_used = 0
    turns_per_phase = args.max_turns // N_phases

    result_a = call_task_model(..., max_turns=turns_per_phase)
    turns_used += turns_per_phase

    remaining = args.max_turns - turns_used
    result_b = call_task_model(..., max_turns=remaining)

---

### Scaffold patterns you are encouraged to try

These are **not prescriptions** — use whichever fits the task. The key insight is that
the scaffold controls the outer strategy; the task model handles inner reasoning.

**Multi-candidate with selection** (robust against unlucky single runs):

    sub_budget = args.max_turns // N
    candidates = []
    for i in range(N):
        sol = call_task_model(prompt=f"Attempt {i+1}/{N}: ...", max_turns=sub_budget)
        score = evaluate(sol)
        candidates.append((score, sol))
    best_score, best_sol = max(candidates)

**Hill climbing** (each call starts from the previous best, not from scratch):

    best = initial_solution
    turns_remaining = args.max_turns
    while turns_remaining >= MIN_TURNS:
        sub_budget = min(TURNS_PER_ITER, turns_remaining)
        improved = call_task_model(
            prompt=f"Current best score: {score}. Improve this solution:\n{best}",
            max_turns=sub_budget
        )
        if evaluate(improved) > evaluate(best):
            best = improved
        turns_remaining -= sub_budget

**Strategy branching** (explore two directions, keep the winner):

    sub_budget = args.max_turns // 2
    sol_a = call_task_model(prompt="Try approach A: ...", max_turns=sub_budget)
    sol_b = call_task_model(prompt="Try approach B: ...", max_turns=sub_budget)
    best = sol_a if evaluate(sol_a) >= evaluate(sol_b) else sol_b

**Phased approach** (explore broadly first, then refine the best lead):

    explore_budget = args.max_turns // 3
    refine_budget  = args.max_turns - explore_budget
    rough = call_task_model(prompt="Explore multiple approaches quickly...", max_turns=explore_budget)
    final = call_task_model(prompt=f"Refine this solution:\n{rough}", max_turns=refine_budget)

---

### What belongs in the scaffold vs. the task model

| Scaffold (Python)                              | Task model (LLM call)                        |
|------------------------------------------------|----------------------------------------------|
| Read and summarize dataset structure           | Implement the core algorithm or solution     |
| Run evaluate.py and parse scores              | Reason about which approach to try next      |
| Select best candidate from multiple attempts  | Debug errors in generated code               |
| Allocate and track turn budgets               | Interpret domain hints and constraints       |
| Retry / fallback on crashes or invalid output | Write and iteratively refine code            |
| Implement outer loop and branching logic       |                                              |

The scaffold should never say "figure it out" when it can give the task model a rich,
pre-computed context. Informed prompts produce better solutions in fewer turns.
