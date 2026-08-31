# revise-langraph

Hands-on scenarios for learning [LangGraph](https://docs.langchain.com/oss/python/langgraph/). Each scenario lives in its own numbered folder, is self-contained, and builds on concepts from the previous ones.

## Scenarios

| Folder | Topic |
|---|---|
| [01-quickstart](01-quickstart/) | Building a basic tool-calling agent with the Graph API and the Functional API |

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/) in a single shared virtual environment for the whole repo (kept outside the project directory — see below).

```bash
V="$HOME/.venvs/revise-langraph"
UV_PROJECT_ENVIRONMENT="$V" uv sync
```

Each scenario expects an `ANTHROPIC_API_KEY` environment variable (get one from the [Anthropic Console](https://console.anthropic.com/)). Copy the scenario's `.env.example` to `.env` and fill it in, or export the variable in your shell.

Run a scenario's scripts with:

```bash
UV_PROJECT_ENVIRONMENT="$V" uv run python 01-quickstart/graph_api.py
```

> Note: virtual environments are created at `~/.venvs/revise-langraph` rather than inside the project because this sandbox mounts the project directory on a filesystem that doesn't support the symlinks `uv`/`venv` need.

## Conventions

- One folder per scenario, numbered in learning order.
- Each scenario folder has its own `README.md` explaining what it demonstrates and how to run it.
- Dependencies are shared at the repo root (`pyproject.toml`) unless a scenario needs something unusual enough to warrant calling it out.
