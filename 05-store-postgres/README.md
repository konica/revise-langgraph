# 05 · Stores: cross-thread memory for the calculator agent

Source: [LangGraph stores](https://docs.langchain.com/oss/python/langgraph/stores)

[04-checkpointers-postgres](../04-checkpointers-postgres/) persisted state
*within* a thread: ask a follow-up on the same `thread_id`, and the agent
remembers earlier turns. But start a brand-new thread and all of that is
gone — a checkpointer only ever knows about one conversation.

A **store** is the other half of LangGraph persistence: memory that
survives *across* threads, scoped to a `user_id` instead of a `thread_id`.
This scenario extends the calculator agent so every calculation it
performs is remembered per-user, and recalled the next time that user
starts *any* conversation — even one that's never happened before.

## Files

- **`calculator_agent.py`** — the same tools, model, and `MessagesState`
  as 04's calculator agent, plus:
  - **`Context`**, a `user_id`-carrying dataclass passed to `compile()`
    as `context_schema` — the store's equivalent of what `thread_id`
    is for the checkpointer.
  - **`tool_node`** now takes a `runtime: Runtime[Context]` parameter and
    calls `runtime.store.put()` after every tool call, saving a summary
    like `"multiply(a=6, b=7) = 42"` under the namespace
    `(user_id, "calculations")`.
  - **`llm_call`** calls `runtime.store.search()` on that same namespace
    and folds the results into the system prompt *before* asking the
    model anything — this is what lets the agent answer "what was my
    last calculation?" on a thread that has no message history at all.
- **`store_basics.py`** — compiles the graph with **both**
  `PostgresSaver` (the checkpointer) and `PostgresStore` (the store) and
  runs two scenarios: Alice recalling a calculation on a brand-new thread,
  and Bob — on *his* first-ever thread — correctly recalling nothing,
  because the store is namespaced per user.

## Setup

From the repo root:

```bash
V="$HOME/.venvs/revise-langraph"
UV_PROJECT_ENVIRONMENT="$V" uv sync
```

Start Postgres for this scenario (own container, own port `5443` — it
doesn't share state with 04-checkpointers-postgres's Postgres):

```bash
docker compose -f 05-store-postgres/docker-compose.yml up -d
```

Copy `.env.example` to `.env` and set your key — `DATABASE_URL` already
matches the `docker-compose.yml` above. `LANGSMITH_TRACING` /
`LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` are optional (see
[04's README](../04-checkpointers-postgres/README.md#what-to-notice) for
what they do):

```bash
cp 05-store-postgres/.env.example 05-store-postgres/.env
# then edit 05-store-postgres/.env and set ANTHROPIC_API_KEY
```

## Run

```bash
V="$HOME/.venvs/revise-langraph"
UV_PROJECT_ENVIRONMENT="$V" uv run python 05-store-postgres/store_basics.py
```

`store_basics.py` uses fixed `thread_id`s, so re-running it keeps adding
to Alice's calculation history rather than starting fresh. To reset:

```bash
docker compose -f 05-store-postgres/docker-compose.yml down -v
docker compose -f 05-store-postgres/docker-compose.yml up -d
```

## What to notice

- **`user_id` and `thread_id` answer different questions.** `thread_id`
  (in `config`) scopes *this conversation*'s messages — the
  checkpointer's job. `user_id` (in `context`, via `Context`) scopes
  memories that should outlive any one conversation — the store's job.
  They're independent: Alice's two threads in this script have different
  `thread_id`s but the same `user_id`, which is exactly what makes her
  second thread able to see memories from her first.
- **The proof that recall comes from the store, not the thread:**
  `store_basics.py` prints `agent.get_state_history()` for Alice's second
  thread *before* running anything on it — `0` checkpoints, confirmed
  empty. The agent still correctly answers what her last calculation was,
  because `llm_call` never looked at thread history for that answer; it
  looked at `runtime.store.search((user_id, "calculations"))`.
- **Namespaces are just tuples, and they isolate by prefix.** `("alice",
  "calculations")` and `("bob", "calculations")` share no data. Bob's
  first-ever thread correctly finds nothing, the same way Alice's second
  thread would if a stranger's `user_id` were used instead of hers.
- **State ordering isn't guaranteed across store backends, and prompts
  need to say so explicitly.** The docs note `PostgresStore.search`
  returns results ordered by `updated_at` descending (most recent
  first) — but the LLM doesn't know that unless the prompt says it. The
  first version of `llm_call`'s system message just said "past
  calculations" with no ordering claim, and the model picked the wrong
  one as "most recent." Saying "most recent first" explicitly fixed it —
  worth remembering any time you inject a list from a store into a
  prompt.
- **`runtime.store.put` needs no extra LLM call.** Because `tool_node`
  already knows exactly which tool ran with which arguments and what it
  returned, it can write a precise memory deterministically. Compare this
  to the docs' generic "analyze conversation and create a new memory"
  example, which implies an LLM call to *extract* the memory — not needed
  here because the tool call itself already *is* the fact worth
  remembering.
- **Both `setup()` calls target the same database, no conflict.**
  `PostgresSaver` and `PostgresStore` create separate, unrelated tables.
  A real deployment commonly points both at the same Postgres instance,
  same as this script does.
