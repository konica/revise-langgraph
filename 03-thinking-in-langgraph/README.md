# 03 · Thinking in LangGraph: a customer support email agent

Source: [Thinking in LangGraph](https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph)

The previous two scenarios built one small tool-calling loop. This one
follows the doc's own worked example — a customer support email agent —
end to end, because it's dense enough to demonstrate every key aspect of
designing a LangGraph application in a single graph:

```
START -> read_email -> classify_intent -+-> search_documentation -+
                                         +-> bug_tracking ---------+-> draft_response -+-> send_reply -> END
                                         +-> human_review <--------------(if urgent)---+
                                         +-> draft_response (else)         |
                                                                            +-> send_reply
```

`classify_intent` and `draft_response` each decide their own next node —
the graph only declares the edges that are *always* the same
(`START -> read_email -> classify_intent`, `send_reply -> END`).

## What this demonstrates

- **State design (Step 3)** — `EmailAgentState` stores only raw data (the
  email, the classification dict, raw search hits, the draft). Prompts are
  built inside each node, on-demand, from that raw data — never stored
  pre-formatted.
- **Nodes that route themselves (Steps 1, 4, 5)** — instead of a
  conditional-edge function sitting between nodes, `classify_intent`,
  `draft_response`, and `human_review` each return a `Command(update=...,
  goto=...)`, updating state and picking the next node in one step. The
  graph's static edges are minimal because routing lives in the nodes.
- **All four error-handling strategies from the doc's table**, each on the
  node it fits naturally:

  | Node | Strategy | What you'll see |
  |---|---|---|
  | `search_documentation` | Transient -> `RetryPolicy` | The fake search API fails once; LangGraph retries the whole node automatically — no `try`/`except` needed for this case. |
  | `bug_tracking` | LLM-recoverable -> store error, keep going | The fake bug tracker rejects "crash" reports; the failure is stored in `search_results` so `draft_response`'s LLM call can see it and write around it. |
  | `human_review` | User-fixable -> `interrupt()` | Critical/billing emails and complex issues pause the graph entirely, persist via the checkpointer, and wait for a human decision. |
  | `send_reply` | Unexpected -> let it bubble | `run_unexpected_error_demo()` forces a simulated send failure that is **not** caught — it propagates out of `agent.invoke()` as a real exception. |

- **Human-in-the-loop with persistence** — the graph is compiled with an
  `InMemorySaver` checkpointer. When `human_review` calls `interrupt()`,
  `agent.invoke()` returns with an `"__interrupt__"` key instead of
  raising; `run_scenario()` reads the interrupt payload, simulates a human
  approving the draft, and resumes with `agent.invoke(Command(resume=...),
  config)` using the same `thread_id`.

## Setup

From the repo root:

```bash
V="$HOME/.venvs/revise-langraph"
UV_PROJECT_ENVIRONMENT="$V" uv sync
```

Copy `.env.example` to `.env` in this folder and set your key:

```bash
cp 03-thinking-in-langgraph/.env.example 03-thinking-in-langgraph/.env
# then edit 03-thinking-in-langgraph/.env and set ANTHROPIC_API_KEY
```

## Run

```bash
V="$HOME/.venvs/revise-langraph"
UV_PROJECT_ENVIRONMENT="$V" uv run python 03-thinking-in-langgraph/agent.py
```

This runs the five example scenarios from the doc (a simple question, a
bug report, an urgent billing issue, a feature request, and a complex
technical issue), followed by the unexpected-error demo. Only
`classify_intent` and `draft_response` call the real model; search, bug
tracking, and sending are simulated so the script runs offline apart from
those two calls.

## What to notice

- The **urgent billing** scenario is routed straight from
  `classify_intent` to `human_review`, *skipping* the AI draft — for
  billing/critical issues a human writes the reply themselves via
  `edited_response`, rather than editing an AI draft.
- The other scenarios instead go through `draft_response` first, which
  routes to `human_review` on its own whenever urgency is high/critical or
  the LLM judged the intent `"complex"` — the same node (`human_review`)
  is reused for two different triage points in the flow. Classification is
  a live LLM call, so exactly which emails land there can vary slightly
  run to run.
- Because every node only ever returns a *partial* state update, you can
  print `result` after `agent.invoke()` and see exactly what each step
  contributed — useful for debugging without adding any logging code.
- Node granularity here follows the doc's reasoning: `search_documentation`
  and `bug_tracking` are separate nodes from the LLM steps specifically so
  a `RetryPolicy` (or a caught, stored error) can be scoped to the
  external call without touching the LLM nodes at all.
