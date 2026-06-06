# Experiment Knowledge Graph

SIA writes an experiment knowledge graph for each run. The graph gives the
feedback agent a compact, structured memory of what happened across generations:
artifacts produced, hypotheses tested, metrics observed, failures seen, and
whether an experiment improved or regressed against the previous generation.

## Artifacts

Each run directory contains:

```text
runs/run_{id}/
├── knowledge_graph.json
├── knowledge_graph.md
├── context.md
└── gen_{n}/
```

`knowledge_graph.json` is the machine-readable source of truth.
`knowledge_graph.md` is a human-readable digest for inspection and debugging.

## Model

The graph is append-only and temporal. Each observation has a subject, predicate,
object, timestamp, optional source path, and optional properties:

```json
{
  "subject": "experiment:gen_2",
  "predicate": "tested",
  "object": "hypothesis:validation_improves_answer_formatting",
  "observed_at": "2026-06-06T15:30:00Z",
  "source": "runs/run_1/gen_2/improvement.md",
  "properties": {
    "text": "Validation improves answer formatting."
  }
}
```

Repeated facts are preserved rather than deduplicated, so repeated sightings
across generations remain visible.

## Prompt Use

Before the feedback agent writes the next generation, SIA renders a compact graph
digest and injects it into the feedback prompt. The prompt asks the agent to use
the digest when choosing the next experiment, avoiding contradicted approaches
unless the current logs provide new evidence.

When the graph digest is present, the feedback prompt also asks `improvement.md`
to include these sections:

- Hypothesis
- Evidence
- Planned Change
- Expected Impact
- Risk

Those headings make later graph extraction more reliable while preserving the
existing two-file feedback-agent contract.

## Future Research Integration

The graph is designed to be seeded by an external research provider later. For
example, an Exa-backed research phase could add initial observations about the
task domain, evaluation contract, common failure modes, known methods, and
recommended experiment hypotheses before generation 1.
