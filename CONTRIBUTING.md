# Contributing to SIA

Thank you for your interest in contributing to SIA (Self-Improving AI). This document provides guidelines for contributing to the project.

## Before You Start

For small documentation fixes, typo fixes, and clearly scoped bug fixes, feel free to open a pull request directly.

For larger changes, new features, new tasks, or behavior changes, please open an issue first so maintainers can discuss the approach before you invest significant time.

Documentation contributions are welcome. Useful documentation changes include clarifying setup or usage instructions, improving examples, adding troubleshooting notes for common errors, correcting stale links or commands, and clarifying task authoring or evaluation behavior.


## How to Contribute

### Reporting Bugs

If you find a bug, please [open an issue](https://github.com/hexo-ai/sia/issues/new) with:

- A clear, descriptive title
- Steps to reproduce the problem
- Expected vs actual behavior
- Python version and OS
- Relevant logs or error messages

### Feature Requests

Have an idea for a new feature or improvement? [Open an issue](https://github.com/hexo-ai/sia/issues/new) and describe:

- What you'd like to see added or changed
- Why it would be useful
- Any implementation ideas you have

### Submitting Changes

1. Fork the repository
2. Create a branch from `main` (`git checkout -b my-change`)
3. Make your changes
4. Run the checks (see below)
5. Commit with a clear message describing the change
6. Push to your fork and open a pull request against `main`

## Development Setup

```bash
# Clone your fork
git clone https://github.com/<your-username>/sia.git
cd sia

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in development mode with all extras
pip install -e ".[dev]"
```

## Running Checks

All of these must pass before submitting a PR:

```bash
# Run tests
python -m pytest tests/ -v

# Lint
ruff check sia/ tests/

# Format check
ruff format --check sia/ tests/

# Type check
ty check sia/
```

To auto-fix lint and formatting issues:

```bash
ruff check --fix sia/ tests/
ruff format sia/ tests/
```

## Security-Sensitive Changes

SIA can execute agent-generated code. Changes to sandboxing, subprocess execution, environment-variable handling, task data access, or agent permissions should be treated as security-sensitive.

For these changes:

- Read [SECURITY.md](SECURITY.md) before starting.
- Prefer safer defaults for untrusted tasks and models.
- Do not log API keys, environment variables, private task data, or generated secrets.
- Explain the security impact in your pull request.

## Adding a New Task

To add a new task for SIA to work on, create the following structure:

```
tasks/<task-name>/
  data/
    public/
      task.md          # Task specification (required)
      evaluate.py      # Evaluation script (recommended)
    private/
      <ground-truth>   # Private evaluation data
  reference/
    reference_target_agent.py      # Agent template (required)
    SAMPLE_TASK_DESCRIPTIONS.md    # Similar task examples (required)
```

See existing tasks (`tasks/spaceship-titanic/`, `tasks/lawbench/`) for examples. The test suite validates that all tasks follow this structure.

## Code Style

- We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting (configured in `pyproject.toml`)
- Line length limit is 120 characters
- Use type hints where they aid clarity
- Follow existing patterns in the codebase

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
