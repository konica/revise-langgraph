# 02 · Local Server: serving an agent with Agent Server

Sources: [LangGraph local server](https://docs.langchain.com/oss/python/langgraph/local-server) ·
[Agent Chat UI](https://docs.langchain.com/oss/python/langgraph/ui)

This scenario takes the exact same calculator agent from
[01-quickstart](../01-quickstart/) and serves it over HTTP with **Agent
Server** — no changes to the agent's graph logic at all. It exists to make
the relationship between "a LangGraph agent" and "Agent Server" concrete.
It also adds a browser-based chat UI ([**Agent Chat UI**](agent-chat-ui/))
on top, so a person — not just `client.py` — can talk to the agent.

## How an agent relates to Agent Server

- **The agent** (`agent.py`) is just a compiled `StateGraph` — the same
  object `01-quickstart/graph_api.py` builds and calls `.invoke()` on
  directly, in-process, from Python. On its own, a compiled graph is a
  library object: something *your own process* has to import and run.
- **Agent Server** (the `langgraph-cli[inmem]` dev server, run via
  `langgraph dev`) is a thin HTTP wrapper *around* that same object. It
  imports `agent.py`, finds the `agent` variable, and exposes it as a
  standard set of REST endpoints (`/runs`, `/threads`, `/assistants`, …) —
  turning "a Python object one process can call" into "a service any
  process, on any machine, can call over HTTP."
- **`langgraph.json`** is the map between the two: it tells Agent Server
  *which* Python variable is the graph (`"./agent.py:agent"`) and what
  name to expose it under (the key `"agent"`). Nothing in `agent.py`
  itself knows or cares that it's being served — you could delete
  `langgraph.json` and still `import agent; agent.agent.invoke(...)`
  exactly like in 01-quickstart.
- Because the graph is unmodified, everything LangGraph already gives you
  — checkpointing, streaming, conditional edges — comes along for free
  once it's served. Agent Server adds the *server* concerns on top:
  persistence across requests (threads), concurrent runs, a
  human-inspectable UI (LangGraph Studio), and an API surface other
  services or a frontend can call without embedding LangGraph themselves.

In short: **the agent is the logic; Agent Server is the delivery
mechanism.** You build and test the graph once (as in 01-quickstart), then
Agent Server is how you hand it to something other than a Python script —
a UI, a teammate, another service — without rewriting it.

## Files

- **`agent.py`** — the compiled graph. Deliberately has no
  `if __name__ == "__main__"`; Agent Server owns running it.
- **`langgraph.json`** — tells `langgraph dev` where the graph lives and
  what to call it:
  ```json
  {
      "dependencies": [".."],
      "graphs": { "agent": "./agent.py:agent" },
      "env": "./.env"
  }
  ```
  `"dependencies": [".."]` points at the repo root's `pyproject.toml` (this
  whole repo is one uv project), so the dev server runs with the same
  dependencies already installed for every other scenario — nothing extra
  to install for the graph itself.
- **`client.py`** — a minimal client using the `langgraph_sdk`, showing
  that talking to the served agent is now just an HTTP call: this script
  never imports LangGraph or `agent.py` at all.
- **`agent-chat-ui/`** — a vendored copy of
  [Agent Chat UI](https://github.com/langchain-ai/agent-chat-ui), a Next.js
  app that talks to Agent Server the same way `client.py` does (over HTTP,
  by graph name), but renders a chat window instead of printing to a
  terminal. See its own [README](agent-chat-ui/README.md) for the full
  feature set (tool call rendering, time travel, artifacts, …).

## Setup

From the repo root:

```bash
V="$HOME/.venvs/revise-langraph"
UV_PROJECT_ENVIRONMENT="$V" uv sync
```

Copy `.env.example` to `.env` in this folder and fill in your keys
(`ANTHROPIC_API_KEY` for the model, `LANGSMITH_API_KEY` for Studio/tracing
— see [Prerequisites](https://docs.langchain.com/oss/python/langgraph/local-server#prerequisites)):

```bash
cp 02-local-server/.env.example 02-local-server/.env
```

## Run

Start the dev server from **this folder** (it looks for `langgraph.json`
in the current directory):

```bash
V="$HOME/.venvs/revise-langraph"
cd 02-local-server
UV_PROJECT_ENVIRONMENT="$V" uv run langgraph dev --host 0.0.0.0
```

You should see:

```
- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs: http://127.0.0.1:2024/docs
```

Open the Studio UI link to interact with the agent visually (send a
message, watch it call `add`/`multiply`/`divide`, inspect state at each
step) — this is the same graph, just observed through Agent Server instead
of `pretty_print()`.

Or drive it from code, in a second terminal:

```bash
UV_PROJECT_ENVIRONMENT="$V" uv run python 02-local-server/client.py
```

This streams the same "Add 3 and 4." run as 01-quickstart, but over HTTP
via `langgraph_sdk` instead of an in-process `.invoke()` call.

### Web UI (Agent Chat UI)

With `langgraph dev` still running, start the chat UI in a third terminal:

```bash
cd 02-local-server/agent-chat-ui
npm install   # first time only
npm run dev
```

> **Filesystem quirks.** If this repo is checked out on a filesystem that
> doesn't support fast/atomic renames or symlinks (the same class of issue
> noted above for `uv`'s venv — e.g. a Windows-hosted volume bind-mounted
> into a Linux sandbox without symlink privileges), `npm install` here can
> hit a few things:
> - **Very slow installs.** Run `npm install` from a copy of
>   `02-local-server/agent-chat-ui/` on a local disk instead, then run
>   `npm run dev` from there.
> - **`Error: ENOENT: no such file or directory, uv_cwd`.** Node's cached
>   working-directory handle went stale, usually from a previous
>   `npm install`'s heavy renaming inside `node_modules`. `cd` out and
>   back into the directory (or open a fresh shell) to refresh the
>   handle, then retry.
> - **`EPERM: operation not permitted, symlink ...` under
>   `node_modules/.bin/`.** Already worked around: this app's
>   [`.npmrc`](agent-chat-ui/.npmrc) sets `bin-links=false` so npm never
>   attempts those symlinks, and its `package.json` scripts invoke each
>   tool with `node node_modules/<pkg>/<bin>` directly instead of relying
>   on them. You shouldn't see this one — if you do, some other tool
>   outside these scripts is still expecting a `.bin` symlink to exist.

Open <http://localhost:3000>. The app's `.env.example` already ships with
defaults that match this scenario's `langgraph.json`
(`NEXT_PUBLIC_API_URL=http://localhost:2024`,
`NEXT_PUBLIC_ASSISTANT_ID=agent`), so if you copy it to `.env` — or just
accept the setup form's defaults the first time the page loads — it
connects straight to the dev server with no other configuration. Type
"Add 3 and 4." and watch the same graph run as `client.py`, now with the
tool calls and results rendered as chat messages instead of printed
`chunk.event` / `chunk.data` pairs.

## What to notice

- `client.py` addresses the graph by the string `"agent"` — the key from
  `langgraph.json` — not by importing anything from `agent.py`. The
  server is the only thing that needs the Python object; clients only
  need the name and a URL.
- The Studio UI lets you inspect intermediate state (e.g. the
  `ToolMessage` the tool node produced) without adding any print
  statements or debugger to the agent's own code — that visibility comes
  from being served, not from anything in `agent.py`.
- `client.py`, Studio, and Agent Chat UI are three different clients of
  the exact same running server — none of them required a single change
  to `agent.py`. That's the payoff of "the agent is the logic; Agent
  Server is the delivery mechanism": any client that can speak the Agent
  Server HTTP API (or its `langgraph_sdk` wrapper) can drive the graph,
  including a full chat UI you didn't have to build yourself.
- `langgraph dev` is explicitly a **development-only** server (in-memory,
  no persistence across restarts). The doc notes that shipping this to
  production means using LangSmith Deployment instead — the local server
  is for the same kind of fast iteration `graph_api.py`'s `__main__` block
  gave you in 01-quickstart, just reachable over HTTP.
