# Contributing to SIA

Thank you for your interest in contributing to SIA (Self-Improving Auto-researcher). This document provides guidelines for contributing to the project.

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
2. Create a branch from `master` (`git checkout -b my-change`)
3. Make your changes
4. Run the checks (see below)
5. Commit with a clear message describing the change
6. Push to your fork and open a pull request against `master`

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
