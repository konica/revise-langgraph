# 01 · Quickstart: a calculator agent

Source: [LangGraph quickstart](https://docs.langchain.com/oss/python/langgraph/quickstart)

A minimal tool-calling agent. Claude is given three arithmetic tools
(`add`, `multiply`, `divide`) and, on each turn, decides whether to call one
of them or answer directly. The same agent is built two ways to compare the
two styles LangGraph offers:

- **`graph_api.py`** — the **Graph API**. You define `State`, wire explicit
  nodes (`llm_call`, `tool_node`) and edges, and a conditional edge
  (`should_continue`) decides whether to loop back to the tool node or end.
  This is the more explicit, more visual style — good for graphs you want to
  inspect, modify, or render as a diagram.
- **`functional_api.py`** — the **Functional API**. The same control flow
  (call the LLM, run any requested tools, repeat until it stops asking) is
  written as a plain Python `while` loop inside a single `@entrypoint`. Steps
  you want tracked/checkpointed individually are marked with `@task`. This
  is the lighter-weight style — good when the flow is easier to express as
  code than as a graph.

Both produce the same behavior for the same input.

## Setup

From the repo root:

```bash
V="$HOME/.venvs/revise-langraph"
UV_PROJECT_ENVIRONMENT="$V" uv sync
```

Copy `.env.example` to `.env` in this folder and set your key:

```bash
cp 01-quickstart/.env.example 01-quickstart/.env
# then edit 01-quickstart/.env and set ANTHROPIC_API_KEY
```

## Run

```bash
V="$HOME/.venvs/revise-langraph"
UV_PROJECT_ENVIRONMENT="$V" uv run python 01-quickstart/graph_api.py
UV_PROJECT_ENVIRONMENT="$V" uv run python 01-quickstart/functional_api.py
```

Each script sends `"Add 3 and 4."` to the agent and pretty-prints the full
message trace, including the tool call and its result.

## What to notice

- The LLM never computes the arithmetic itself — it emits a *tool call*
  (`add(a=3, b=4)`), the tool node executes real Python and returns a
  `ToolMessage`, and that result is fed back to the LLM as context.
- The loop terminates via `should_continue` / the `while` condition checking
  `tool_calls` on the latest AI message — once the model stops requesting
  tools, it's ready to give a final answer.
- `MessagesState` accumulates messages with `operator.add` (Graph API) or
  `add_messages` (Functional API) rather than overwriting them — state is
  additive across the run.
