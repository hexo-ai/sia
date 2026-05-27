# tau2-bench — Agentic Customer Service

## Overview

Your task is to implement a customer-service agent that handles airline support requests end-to-end. Unlike classical benchmarks, **the artifact here is the agent itself** — there is no static `solution.py`. What gets evaluated and improved across generations is your `target_agent.py`.

The benchmark is **tau2-bench** (Sierra Research): a dual-control agentic evaluation where both you (the service agent) and the simulated customer can modify a shared database. The evaluation checks whether the database ends up in the correct final state after each conversation.

## What your agent must do

For each episode:
1. Receive an opening user message (customer intent)
2. Make tool calls to query and update the airline database (flights, bookings, refunds, upgrades…)
3. Follow the airline's business policy (provided in `{dataset_dir}/episodes/policy.md`)
4. Resolve the customer's request and end the conversation

Your agent runs **all episodes** sequentially in a single execution, then writes `predictions.json` to `{working_dir}`.

## Function interface

Your target_agent.py must accept the standard SIA arguments and produce `predictions.json` + `results.json`:

```
python target_agent.py \
    --dataset_dir {dataset_dir} \
    --working_dir {working_dir} \
    --shared_dir  {shared_dir} \
    --model       {model}
```

After running all episodes, call evaluate.py to get the official score:
```bash
python {dataset_dir}/evaluate.py {working_dir}/predictions.json
```

## predictions.json format

Your agent must write this file to `{working_dir}/predictions.json`:

```json
{
  "episodes": [
    {
      "task_id": 0,
      "reward": 1.0,
      "num_turns": 7,
      "error": null
    },
    {
      "task_id": 1,
      "reward": 0.0,
      "num_turns": 12,
      "error": "agent_error: policy_violation"
    }
  ]
}
```

- `reward`: `1.0` if the task was completed correctly (tau2-bench environment verdict), `0.0` otherwise
- `num_turns`: number of conversation turns used
- `error`: null on success, error string if the episode crashed

## Scoring

```
pass_rate = mean([ep["reward"] for ep in episodes])
score     = pass_rate   (higher is better, range 0–1)
```

## Evaluation

```bash
python {dataset_dir}/evaluate.py {working_dir}/predictions.json
```

This reads `predictions.json`, computes the pass rate, and writes `results.json` to `{working_dir}`.

## Available tools (airline domain)

Provided by the tau2-bench environment — see `{dataset_dir}/episodes/tools.json`. Typical tools:

- `search_flights(origin, dest, date)` — query available flights
- `get_booking(booking_id)` — retrieve a booking
- `update_booking(booking_id, ...)` — change flight, seat, meal preference
- `cancel_booking(booking_id)` — cancel with refund
- `get_customer_info(customer_id)` — retrieve customer profile
- `send_message(message)` — send a message to the user (ends the agent turn)

## Policy

The airline's business policy is in `{dataset_dir}/episodes/policy.md`. Your agent must read and follow it. Example constraints: refund eligibility windows, upgrade rules, seat-change fees.

## Optimization strategies

The reference agent uses a simple single-turn prompt. Generations should improve:

1. **Policy awareness** — inject the full policy into the system prompt so the model respects business rules
2. **Tool call robustness** — handle API errors, retry with corrected arguments
3. **Conversation management** — detect when the task is resolved and end cleanly
4. **Context compression** — summarize long conversation histories to stay within context window
5. **Multi-step planning** — for complex requests (e.g., "cancel my trip and rebook for next week"), plan the sequence of tool calls before executing

## Dataset Directory Layout

```
{dataset_dir}/
├── episodes/
│   ├── tasks.json    ← episode definitions (task_id, instruction, initial_config)
│   ├── policy.md     ← airline business policy
│   └── tools.json    ← available tool definitions
├── evaluate.py       ← scoring script
└── task.md           ← this file

{working_dir}/        ← your read/write workspace
├── predictions.json  ← written by your agent after running all episodes
└── results.json      ← written by evaluate.py after scoring
```

## Generalization

The development set is **80 public episodes** from the airline domain. The private score is computed on **20 held-out airline episodes**. These are drawn from the same distribution, so a policy-aware generalizable agent should transfer well.

## Tips

- **Read the policy first** — most failures come from policy violations (wrong refund amounts, invalid upsells). Inject `policy.md` into the system prompt.
- **Verify tool calls** — tau2-bench checks the final DB state, not whether you called the right function. A booking that was "cancelled" but the DB still shows it active → reward = 0.
- **End the conversation explicitly** — call `send_message` with a closing message. An open conversation typically scores 0.
- **Handle partial failures** — if one step fails, recover gracefully rather than abandoning the task.
