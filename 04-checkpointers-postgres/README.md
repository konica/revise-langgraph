# 04 · Checkpointers: persisting the agent to PostgreSQL

Source: [LangGraph checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers)

The calculator agent from [01-quickstart](../01-quickstart/) works, but its
memory dies with the Python process: `InMemorySaver` (the default when you
don't pass a checkpointer) keeps every thread's state in a dict in RAM.
This scenario runs a real Postgres in Docker and shows two ways to make
that state durable, without relying on LangGraph's Agent Server (see
[02-local-server](../02-local-server/) for that story instead). Both ways
compile the exact same graph — **`calculator_agent.py`** defines it once
(tools, model, nodes, an *uncompiled* `agent_builder`, no checkpointer),
and every other file just imports `agent_builder` and decides how to
persist and run it:

- **The manual way** (`checkpointer_basics.py`, `production_pool.py`) —
  standalone scripts that wire up `PostgresSaver` / `AsyncPostgresSaver`
  themselves and call `.invoke()` in-process. Nothing is served over
  HTTP; these are still just scripts, like 01-quickstart but checkpointed.
- **The self-hosted way** (`fastapi_app.py`) — the same checkpointer
  wiring as `production_pool.py`, but now served over HTTP by a FastAPI
  app you write and own. This is what deploying the graph to your own
  infrastructure looks like — e.g. AWS ECS/Fargate or a Lambda — where
  you're responsible for the checkpointer *and* for generating thread IDs
  yourself, because there's no Agent Server Threads API doing either for
  you.

## The manual way

Both scripts import `agent_builder` from `calculator_agent.py` — neither
defines the graph itself.

- **`checkpointer_basics.py`** — sync `PostgresSaver`. A two-turn
  conversation on one `thread_id`, then `get_state` (the latest checkpoint)
  and `get_state_history` (every checkpoint the run produced), then **time
  travel**: forking the conversation from a past checkpoint with
  `update_state` to explore a different follow-up question without losing
  the original branch.
- **`production_pool.py`** — async `AsyncPostgresSaver` over a shared
  `psycopg_pool.AsyncConnectionPool`, serving two users' conversations
  concurrently with `asyncio.gather`. This is the shape you'd actually
  deploy: one pool, many threads, non-blocking calls.

### Setup

From the repo root:

```bash
V="$HOME/.venvs/revise-langraph"
UV_PROJECT_ENVIRONMENT="$V" uv sync
```

Start Postgres (runs on host port `5442`, not the Postgres default `5432`,
so it won't collide with one you already have running):

```bash
docker compose -f 04-checkpointers-postgres/docker-compose.yml up -d
```

Copy `.env.example` to `.env` in this folder and set your key — the
`DATABASE_URL` default already matches the `docker-compose.yml` above:

```bash
cp 04-checkpointers-postgres/.env.example 04-checkpointers-postgres/.env
# then edit 04-checkpointers-postgres/.env and set ANTHROPIC_API_KEY
```

`.env.example` also has `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` /
`LANGSMITH_PROJECT`, marked optional — set them (a free
[LangSmith](https://smith.langchain.com/settings) key works) to see each
run's full trace, including every LLM and tool call inside `llm_call` and
`tool_node`, in the `checkpointers-postgres` project. Leave them unset and
the scripts behave identically: tracing is opt-in and fails open, so a
missing or bad key just means no trace, not a crashed run.

### Run

```bash
V="$HOME/.venvs/revise-langraph"
UV_PROJECT_ENVIRONMENT="$V" uv run python 04-checkpointers-postgres/checkpointer_basics.py
UV_PROJECT_ENVIRONMENT="$V" uv run python 04-checkpointers-postgres/production_pool.py
```

`checkpointer_basics.py` reuses the same `thread_id` on every run, so
re-running it keeps extending (and re-forking) the same conversation. To
start over:

```bash
docker compose -f 04-checkpointers-postgres/docker-compose.yml down -v
docker compose -f 04-checkpointers-postgres/docker-compose.yml up -d
```

## The self-hosted way

- **`fastapi_app.py`** — imports `agent_builder` from `calculator_agent.py`
  and does only the hosting part: opens an `AsyncPostgresSaver` connection
  pool once at startup (same pattern as `production_pool.py`), compiles
  the graph with it, and exposes three endpoints:

  | Endpoint | What it does |
  |---|---|
  | `POST /threads` | Mints a new `thread_id` (`uuid.uuid4()`) — nothing is written to Postgres yet |
  | `POST /threads/{id}/runs` | Runs the agent on that thread: `agent.ainvoke({"messages": [...]}, config)` |
  | `GET /threads/{id}/state` | `agent.aget_state(config)` — 404s if the thread has never been run |

  This is everything Agent Server would give you automatically
  (`client.threads.create()`, `client.runs.wait(...)`, its own Threads
  API) — written by hand, because nothing here is Agent Server. No
  `langgraph-cli`, no license check, deployable anywhere that runs a
  Python web server.
- **`fastapi_client.py`** — a small demo client hitting those three
  endpoints: create a thread, run two turns, read the state back.

### Setup

Same Postgres and `.env` as [the manual way](#the-manual-way) above — no
additional setup. `fastapi` and `uvicorn` are already in the root
`pyproject.toml`.

### Run

```bash
V="$HOME/.venvs/revise-langraph"
UV_PROJECT_ENVIRONMENT="$V" uv run python 04-checkpointers-postgres/fastapi_app.py
```

Then, in a second terminal:

```bash
V="$HOME/.venvs/revise-langraph"
UV_PROJECT_ENVIRONMENT="$V" uv run python 04-checkpointers-postgres/fastapi_client.py
```

Or drive it with `curl` directly:

```bash
THREAD_ID=$(curl -s -X POST http://localhost:8000/threads | python3 -c 'import json,sys; print(json.load(sys.stdin)["thread_id"])')
curl -s -X POST http://localhost:8000/threads/$THREAD_ID/runs \
  -H 'Content-Type: application/json' -d '{"message": "What'\''s 6 times 7?"}'
curl -s http://localhost:8000/threads/$THREAD_ID/state
```

## What to notice

- **One graph, three callers.** `calculator_agent.py`'s nodes are plain
  sync functions (`model_with_tools.invoke(...)`, not `ainvoke`) even
  though `production_pool.py` and `fastapi_app.py` are async throughout.
  LangGraph runs sync nodes in a thread pool when you call `.ainvoke()`,
  so the same `agent_builder` works unmodified with `.invoke()`
  (`checkpointer_basics.py`) and `.ainvoke()` (the other two). The reverse
  doesn't hold: nodes defined as `async def` make sync `.invoke()` raise
  `TypeError: No synchronous function provided` — so sync nodes are the
  one shape that's compatible with every caller.
- **`thread_id` is the whole API surface for memory.** Every call that
  should share history passes the same `{"configurable": {"thread_id":
  ...}}` (manual way) or the same `thread_id` string (self-hosted way).
  Nothing else changes between a one-off run and a resumed conversation.
- **`setup()` creates the schema and is idempotent.** Both `checkpointer_basics.py`
  and `fastapi_app.py`'s startup call it every time for convenience; a real
  deployment runs it once as a migration step, not on every app boot.
- **A checkpoint exists per super-step, not per `invoke()` call.** One
  `invoke()` of this two-node agent produces several checkpoints (the
  input, `llm_call`, `tool_node`, `llm_call` again, ...) — see the `step=`
  and `next=` columns `checkpointer_basics.py` prints from
  `get_state_history`.
- **`update_state` never mutates a past checkpoint — it writes a new one.**
  Forking from `after_turn1` in `checkpointer_basics.py` leaves the
  original "add 8" branch fully intact in Postgres; the fork just becomes
  the new checkpoint the thread's `get_state()` returns by default. You
  could resume the original branch again by passing its `checkpoint_id`
  explicitly.
- **`as_node` matters for what "resume" means.** Injecting a brand-new
  human message with `update_state` needs `as_node=START` so the graph
  takes `START`'s edge to `llm_call` next — the default inference instead
  treats the update as coming from whichever node wrote the checkpoint you
  forked from (`llm_call`), which would route into `llm_call`'s
  *outgoing* branch against the new message instead of calling the model
  with it.
- **The pool needs the same connection kwargs `from_conn_string` sets for
  you.** `production_pool.py` and `fastapi_app.py` both pass
  `autocommit=True`, `prepare_threshold=0`, and `row_factory=dict_row`
  explicitly to `AsyncConnectionPool` — required because building the pool
  yourself skips the setup `PostgresSaver.from_conn_string` normally does
  on your behalf.
- **Concurrent threads through one pool stay isolated.** `production_pool.py`
  runs Alice's and Bob's first turns concurrently via `asyncio.gather`; each
  only ever sees its own `thread_id`'s history. `fastapi_app.py` gets this
  for free from the pool too — concurrent HTTP requests for different
  `thread_id`s don't interfere with each other.
- **A thread doesn't need to be "created" in Postgres before you run on
  it.** `fastapi_app.py`'s `POST /threads` just mints a UUID — no write
  happens until the first `ainvoke()` on that `thread_id` produces the
  first checkpoint. `GET /threads/{id}/state` treats an empty state
  snapshot as "unknown thread" (404) for exactly this reason: nothing
  distinguishes "never run" from "doesn't exist" at the checkpointer
  level, because the checkpointer never heard about the thread at all
  until a run happened.
- **`aget_state` mirrors `get_state`, just async.** Every sync method used
  in `checkpointer_basics.py` (`get_state`, `get_state_history`,
  `update_state`) has an `a`-prefixed async twin — `fastapi_app.py` uses
  `aget_state` for the same reason `production_pool.py` uses `ainvoke`:
  an async web server can't block on synchronous I/O without stalling
  every other in-flight request.
- **Tracing is a separate opt-in from the checkpointer, controlled purely
  by environment variables.** `PostgresSaver`/`AsyncPostgresSaver` persist
  *state* (what `get_state` returns); LangSmith tracing captures
  *execution* (every LLM call, tool call, and node run inside a single
  `invoke()`). Nothing in `calculator_agent.py` or any script here
  references LangSmith at all — `LANGSMITH_TRACING=true` +
  `LANGSMITH_API_KEY` in `.env` is the entire mechanism, picked up
  automatically by the `langchain`/`langgraph` packages themselves.
  Without them set, these scripts run identically; you just get no trace.
