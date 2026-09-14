# 04 · Checkpointers: persisting the agent to PostgreSQL

Source: [LangGraph checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers)

The calculator agent from [01-quickstart](../01-quickstart/) works, but its
memory dies with the Python process: `InMemorySaver` (the default when you
don't pass a checkpointer) keeps every thread's state in a dict in RAM.
Swap it for `PostgresSaver` and the same agent's conversations survive
restarts, because the checkpoints live in a database instead of the
process.

This scenario runs a real Postgres in Docker and uses it to show, in
order, the concepts from the docs:

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

## Setup

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

## Run

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

When you're done:

```bash
docker compose -f 04-checkpointers-postgres/docker-compose.yml down -v
```

## What to notice

- **`thread_id` is the whole API surface for memory.** Every call that
  should share history passes the same `{"configurable": {"thread_id":
  ...}}`. Nothing else changes between a one-off run and a resumed
  conversation — not the graph, not the invoke call.
- **`setup()` creates the schema and is idempotent.** Both scripts call it
  on every run for convenience; a real deployment runs it once as a
  migration step, not on every app boot.
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
  you.** `production_pool.py` passes `autocommit=True`, `prepare_threshold=0`,
  and `row_factory=dict_row` explicitly to `AsyncConnectionPool` — required
  because building the pool yourself skips the setup `PostgresSaver.from_conn_string`
  normally does on your behalf.
- **Concurrent threads through one pool stay isolated.** `production_pool.py`
  runs Alice's and Bob's first turns concurrently via `asyncio.gather`; each
  only ever sees its own `thread_id`'s history.
